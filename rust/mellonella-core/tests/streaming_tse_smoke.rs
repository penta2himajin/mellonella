//! Smoke tests for the Stage C TSE wiring in
//! [`mellonella_core::streaming::StreamingPipeline`].
//!
//! Mirrors [`tests/pipeline_tse_smoke.rs`] (the offline-side smoke
//! coverage), adapted for the streaming engine's chunked
//! `push_samples` / `flush` shape:
//!
//! * **default behaviour preserved** — with `cfg.pipeline.tse == None`,
//!   the streaming engine emits `tse_applied == false` and the output
//!   matches the pre-Phase-5 streaming smoke contract (byte-identical
//!   between two independent runs with the same input).
//! * **TSE enabled produces non-trivial output** — with
//!   `cfg.pipeline.tse == Some(...)` and the smoke ONNX from
//!   `tests/tse_parity.rs`, the streaming pipeline emits
//!   `tse_applied == true`, the total emitted samples ≈ input length
//!   (modulo the natural TSE-buffer delay drained by `flush()`), the
//!   output is finite, and the output differs from the input.
//! * **rate-mismatch rejection** — with `cfg.pipeline.tse == Some(...)`
//!   and `audio_sample_rate` ≠ the configured TSE model's expected
//!   sample rate, `StreamingPipeline::new` returns
//!   `PipelineError::TseRateMismatch` before any audio is pushed.
//!   (The decision rate is independent and may legitimately differ.)
//! * **streaming-vs-offline parity** — running the same audio through
//!   `process_offline` (TSE-on) and through the streaming engine
//!   (TSE-on, chunked) produces emitted-sample sequences that agree
//!   within a tolerance accounting for the engine's known
//!   buffering / delay characteristics (the streaming TSE accumulator
//!   keeps up to `chunk_samples - 1` samples pending until flush).
//!
//! All tests are gated on the same env vars as
//! `pipeline_tse_smoke.rs` (`MELLONELLA_ECAPA_ONNX`,
//! `MELLONELLA_VAD_ONNX`, `ORT_DYLIB_PATH`) and additionally invoke
//! the Python export flow shared with `tse_parity.rs` to build
//! `build/tse_smoke.onnx`. If any of those isn't available, the
//! affected test prints a skip notice and passes — the cargo CI lane
//! stays green without the full training Python stack.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{
    process_offline, PipelineComponents, PipelineConfig, PipelineError,
};
use mellonella_core::streaming::{StreamingConfig, StreamingOutput, StreamingPipeline};
use mellonella_core::tse::{TseConfig, TSE_COND_DIM};
use mellonella_core::tse_stage::TseStageConfig;
use mellonella_core::vad::SileroVad;

const FILTERBANK: &[u8] = include_bytes!("fixtures/fbank_filterbank.bin");

// Same chunk length the smoke ONNX is exported with — pinned in both
// `tests/tse_parity.rs` and `tests/pipeline_tse_smoke.rs` so the
// `build/tse_smoke.onnx` fixture is shared across tests.
const TSE_CHUNK: usize = 160;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Same synthetic mixture as `pipeline_smoke.rs` and
/// `pipeline_tse_smoke.rs` — a 5-harmonic stack peaking at ±0.5.
/// Deterministic, length equals `duration_sec * sample_rate`.
fn synth_waveform(sample_rate: u32, duration_sec: f32, f0: f32) -> Vec<f32> {
    let sr = f64::from(sample_rate);
    let n = (sr * f64::from(duration_sec)) as usize;
    let mut wave = vec![0.0_f32; n];
    for (i, slot) in wave.iter_mut().enumerate() {
        let t = i as f32 / sr as f32;
        let mut v = 0.0_f32;
        for harmonic in 1..=5 {
            v += (1.0 / harmonic as f32)
                * (2.0 * std::f32::consts::PI * f0 * harmonic as f32 * t).sin();
        }
        *slot = v;
    }
    let peak = wave.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if peak > 0.0 {
        for v in &mut wave {
            *v = *v / peak * 0.5;
        }
    }
    wave
}

/// Locate the repo root by walking up from `CARGO_MANIFEST_DIR`. Same
/// pattern as `tests/tse_parity.rs` and `tests/pipeline_tse_smoke.rs`.
fn repo_root() -> PathBuf {
    let mut here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..4 {
        if here.join("scripts/export_tse_onnx.py").exists() {
            return here;
        }
        if !here.pop() {
            break;
        }
    }
    panic!(
        "cannot locate repo root from CARGO_MANIFEST_DIR={CARGO_MANIFEST_DIR}",
        CARGO_MANIFEST_DIR = env!("CARGO_MANIFEST_DIR")
    );
}

/// Build / reuse the smoke TSE ONNX. Returns `Some(path)` when the
/// fixture is ready, `None` when Python / torch isn't available and
/// the caller should skip the test. Identical to the helper in
/// `pipeline_tse_smoke.rs` so both test suites share the cached file.
fn ensure_tse_smoke_onnx() -> Option<PathBuf> {
    let root = repo_root();
    let build = root.join("build");
    let onnx_path = build.join("tse_smoke.onnx");
    let weights_path = build.join("tse_smoke.onnx.weights.pt");
    if onnx_path.exists() && weights_path.exists() {
        return Some(onnx_path);
    }
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = root.join("scripts").join("export_tse_onnx.py");
    let chunk_arg = TSE_CHUNK.to_string();
    let out_arg = onnx_path.to_string_lossy().into_owned();
    let output = Command::new(&python)
        .arg(&script)
        .args([
            "export", "--config", "poc_16k", "--chunk", &chunk_arg, "--output", &out_arg,
        ])
        .current_dir(&root)
        .output();
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[skip] failed to spawn '{python}': {e}");
            return None;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[skip] export_tse_onnx.py failed: {stderr}");
        return None;
    }
    Some(onnx_path)
}

/// Build a fresh `PipelineComponents` + single-anchor pool. The
/// anchor is normalised (`1/sqrt(192)` in every slot) so the centroid
/// the TSE stage gets is well-defined and non-zero. The smoke ONNX
/// has random weights, so the only thing the cond embedding controls
/// is *which* random output we get, not whether it's non-trivial.
fn build_smoke_components(ecapa_path: &str, vad_path: &str) -> (PipelineComponents, EmbeddingPool) {
    let fb_matrix = read_f32_buffer(FILTERBANK);
    let fbank = Fbank::new(&fb_matrix).expect("Fbank from fixture");
    let ecapa = EcapaTdnn::from_onnx_path(ecapa_path).expect("ECAPA load");
    let vad = SileroVad::from_onnx_path(vad_path, 16_000).expect("VAD load");
    let components = PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort: Vec::new(),
        tse: None,
    };
    let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
    let norm = (TSE_COND_DIM as f32).sqrt();
    pool.add_anchors([vec![1.0_f32 / norm; TSE_COND_DIM]]);
    (components, pool)
}

/// Skip-or-paths preamble shared by every test below.
fn ecapa_vad_paths() -> Option<(String, String)> {
    let Ok(ecapa_path) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
        return None;
    };
    let Ok(vad_path) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return None;
    };
    if !Path::new(&ecapa_path).exists() || !Path::new(&vad_path).exists() {
        eprintln!("[skip] ONNX file(s) missing");
        return None;
    }
    let Ok(dylib) = std::env::var("ORT_DYLIB_PATH") else {
        eprintln!("[skip] ORT_DYLIB_PATH not set");
        return None;
    };
    if !Path::new(&dylib).exists() {
        eprintln!("[skip] ORT_DYLIB_PATH={dylib} does not exist");
        return None;
    }
    Some((ecapa_path, vad_path))
}

/// Concatenate a sequence of `StreamingOutput`s into one final output.
/// `tse_applied` is true if **any** part observed TSE running — the
/// running state is sticky across a stream, so this matches what a
/// caller checking the flag at end-of-stream would see.
fn concat_outputs(parts: Vec<StreamingOutput>) -> StreamingOutput {
    let mut acc = StreamingOutput::default();
    for mut p in parts {
        acc.audio.append(&mut p.audio);
        acc.gate_decisions.append(&mut p.gate_decisions);
        acc.events.append(&mut p.events);
        acc.gate_per_frame.append(&mut p.gate_per_frame);
        acc.score_per_frame.append(&mut p.score_per_frame);
        acc.cos_sim_max_per_frame
            .append(&mut p.cos_sim_max_per_frame);
        acc.f0_match_per_frame.append(&mut p.f0_match_per_frame);
        acc.tse_applied |= p.tse_applied;
    }
    acc
}

/// Drive a streaming pipeline end-to-end in one shot (single push +
/// flush) and return the concatenated output.
fn run_streaming_once(
    cfg: StreamingConfig,
    audio: &[f32],
    components: PipelineComponents,
    pool: EmbeddingPool,
) -> StreamingOutput {
    let mut pipeline = StreamingPipeline::new(pool, cfg, components).expect("streaming new");
    let part1 = pipeline.push_samples(audio).expect("push");
    let part2 = pipeline.flush().expect("flush");
    concat_outputs(vec![part1, part2])
}

/// Default behaviour preserved: with `cfg.pipeline.tse == None`, the
/// streaming engine emits `tse_applied == false` and two independent
/// runs with identical inputs produce byte-identical outputs. This is
/// the streaming counterpart of
/// `process_offline_default_is_byte_identical_with_tse_off` in
/// `pipeline_tse_smoke.rs`.
#[test]
fn streaming_default_is_byte_identical_with_tse_off() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };

    let audio = synth_waveform(16_000, 2.0, 180.0);

    let pipeline_cfg = PipelineConfig::default();
    let gate_cfg = GateConfig::default();
    let cfg = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: gate_cfg,
        audio_sample_rate: 16_000,
        diagnostics: true,
    };

    // Run 1: fresh components + pool.
    let (components_a, pool_a) = build_smoke_components(&ecapa_path, &vad_path);
    let result_a = run_streaming_once(cfg.clone(), &audio, components_a, pool_a);

    // Run 2: independent fresh components + pool, same inputs.
    let (components_b, pool_b) = build_smoke_components(&ecapa_path, &vad_path);
    let result_b = run_streaming_once(cfg, &audio, components_b, pool_b);

    // The load-bearing default-off flag.
    assert!(!result_a.tse_applied);
    assert!(!result_b.tse_applied);

    // Two TSE-off streaming runs must be byte-identical.
    assert_eq!(
        result_a.audio, result_b.audio,
        "TSE-off streaming runs diverged"
    );
    assert_eq!(
        result_a.gate_per_frame, result_b.gate_per_frame,
        "TSE-off gate_per_frame diverged"
    );

    // Streaming output length is bounded — at identity rate (no
    // resampler) it equals the input length exactly.
    assert_eq!(result_a.audio.len(), audio.len());
}

/// TSE enabled produces non-trivial, finite output that differs from
/// the input. Mirrors
/// `process_offline_with_tse_extracts_non_trivial_audio`.
#[test]
fn streaming_with_tse_extracts_non_trivial_audio() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };
    let Some(tse_onnx) = ensure_tse_smoke_onnx() else {
        return;
    };

    let audio = synth_waveform(16_000, 2.0, 180.0);

    let (components, pool) = build_smoke_components(&ecapa_path, &vad_path);
    let pipeline_cfg = PipelineConfig {
        tse: Some(TseStageConfig {
            onnx_path: tse_onnx,
            chunk_samples: TSE_CHUNK,
            model: TseConfig::poc_16k(),
        }),
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();
    let cfg = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: gate_cfg,
        audio_sample_rate: 16_000,
        diagnostics: true,
    };
    let result = run_streaming_once(cfg, &audio, components, pool);

    // The wiring flagged the TSE branch on at least one call.
    assert!(result.tse_applied, "TSE-on stream did not flag tse_applied");

    // Output length is in the ballpark of the input length. At
    // identity rate with TSE active there's no resampler delay; the
    // only buffering is the TSE accumulator, which `flush()` drains
    // by zero-padding the final partial chunk and emitting one last
    // chunk. We allow a small slack to cover any rounding the
    // pending-gain logic does at the tail (the test input is 32 000
    // samples and the chunk size is 160, so 32 000 is a multiple of
    // 160 and the flush should produce exactly the input length —
    // but we keep the assertion generous for robustness).
    let expected = audio.len() as i64;
    let got = result.audio.len() as i64;
    let slack = (TSE_CHUNK as i64).max(expected / 50);
    assert!(
        (got - expected).abs() <= slack,
        "TSE-on streaming output length {got} far from input {expected} (slack {slack})"
    );

    // Everything is finite — no NaN / Inf leaked through TSE or the
    // envelope.
    assert!(result.audio.iter().all(|v| v.is_finite()));

    // TSE-extracted audio is not byte-identical to the input — proves
    // the model actually ran on the streaming path.
    let same = result.audio.len() == audio.len()
        && result
            .audio
            .iter()
            .zip(audio.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-9);
    assert!(
        !same,
        "TSE streaming output equals input — model did not run"
    );
}

/// Rate-mismatch rejection: with `cfg.pipeline.tse == Some(...)` and an
/// audio rate that doesn't match the configured TSE model's expected
/// sample rate, `StreamingPipeline::new` must error out with
/// `PipelineError::TseRateMismatch` before any audio is pushed. The
/// decision rate (`pipeline.sample_rate`) is independent and may
/// legitimately differ from the audio rate.
#[test]
fn streaming_with_tse_rejects_audio_sr_mismatch() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };

    let pipeline_cfg = PipelineConfig {
        tse: Some(TseStageConfig {
            // No real ONNX needed — the rate check fires before
            // `TseStage::from_config` would try to load anything.
            // Pointing at /dev/null catches a regression that would
            // swap the order.
            onnx_path: std::path::PathBuf::from("/dev/null"),
            chunk_samples: TSE_CHUNK,
            model: TseConfig::poc_16k(),
        }),
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();

    // `StreamingPipeline` doesn't derive `Debug` (the underlying
    // ONNX session handles aren't trivially printable), so the
    // tests below use `match` on `Result` directly rather than
    // `expect_err`.

    // Case 1: audio rate mismatch (48 kHz audio against the 16 kHz
    // poc_16k model).
    {
        let cfg = StreamingConfig {
            pipeline: pipeline_cfg.clone(),
            gate: gate_cfg,
            audio_sample_rate: 48_000,
            diagnostics: true,
        };
        let (components, pool) = build_smoke_components(&ecapa_path, &vad_path);
        match StreamingPipeline::new(pool, cfg, components) {
            Err(PipelineError::TseRateMismatch {
                audio_sr,
                expected_sr,
            }) => {
                assert_eq!(audio_sr, 48_000);
                assert_eq!(expected_sr, 16_000);
                let err = PipelineError::TseRateMismatch {
                    audio_sr,
                    expected_sr,
                };
                let msg = format!("{err}");
                assert!(
                    msg.contains("16000") || msg.contains("16 kHz"),
                    "expected expected_sr in error message, got {msg:?}"
                );
            }
            Err(other) => panic!("expected TseRateMismatch, got {other:?}"),
            Ok(_) => panic!("expected TseRateMismatch, got Ok"),
        }
    }

    // Case 2: async refresh + TSE is NOW SUPPORTED (Step 4 of the
    // Stage C 実適用 series). The early-return that used to fire here
    // was removed in PR #154; the path now attaches a TseStage to the
    // async-mode StreamingState just like it does for sync mode. The
    // `/dev/null` ONNX still can't load, so the call surfaces a
    // `TseStage` error rather than `TseAsyncUnsupported` — that's the
    // bit we assert here.
    {
        let mut p_cfg = pipeline_cfg.clone();
        p_cfg.async_refresh = true;
        let cfg = StreamingConfig {
            pipeline: p_cfg,
            gate: gate_cfg,
            audio_sample_rate: 16_000,
            diagnostics: true,
        };
        let (components, pool) = build_smoke_components(&ecapa_path, &vad_path);
        match StreamingPipeline::new(pool, cfg, components) {
            Err(PipelineError::TseAsyncUnsupported) => {
                panic!("Step 4 lifted the async+TSE rejection — this branch must not fire")
            }
            Err(PipelineError::TseStage(_)) => {
                // Expected — /dev/null isn't a valid ONNX, and we
                // reached the stage-building step.
            }
            Err(other) => panic!("expected TseStage(...) error from /dev/null, got {other:?}"),
            Ok(_) => panic!("expected TseStage error from /dev/null, got Ok"),
        }
    }
}

/// Streaming-vs-offline parity at identity rate with TSE on. Mirrors
/// the shape of `streaming_identity_rate_per_frame_matches_offline`
/// in `tests/streaming_smoke.rs`: both engines share the same
/// per-VAD-frame core for the *decision* path (TSE only changes the
/// audio path), so when audio is pushed in chunks the emitted samples
/// must agree with the offline reference within a tolerance that
/// covers the well-known streaming-vs-offline envelope-stride
/// micro-differences.
///
/// The audio comparison is sample-by-sample over the prefix that
/// both runs emitted; we don't pin a bit-exact equality because the
/// streaming engine advances the envelope one VAD frame at a time
/// while the offline path re-applies it as one run-length sweep.
#[test]
fn streaming_with_tse_matches_offline_within_tolerance() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };
    let Some(tse_onnx) = ensure_tse_smoke_onnx() else {
        return;
    };

    let audio = synth_waveform(16_000, 2.0, 180.0);

    let pipeline_cfg = PipelineConfig {
        tse: Some(TseStageConfig {
            onnx_path: tse_onnx,
            chunk_samples: TSE_CHUNK,
            model: TseConfig::poc_16k(),
        }),
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();

    // Offline reference run.
    let (mut components_a, mut pool_a) = build_smoke_components(&ecapa_path, &vad_path);
    let offline = process_offline(
        &audio,
        16_000,
        &mut pool_a,
        &pipeline_cfg,
        &gate_cfg,
        &mut components_a,
    )
    .expect("offline TSE run");
    assert!(offline.tse_applied);

    // Streaming run with TSE, chunked through several intermediate
    // sizes to exercise the cross-call accumulator state.
    let (components_b, pool_b) = build_smoke_components(&ecapa_path, &vad_path);
    let cfg = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: gate_cfg,
        audio_sample_rate: 16_000,
        diagnostics: true,
    };
    let mut pipeline = StreamingPipeline::new(pool_b, cfg, components_b).expect("streaming new");
    let mut parts: Vec<StreamingOutput> = Vec::new();
    // Use a chunk size that's neither a multiple of one VAD frame
    // (512) nor of one TSE chunk (160) — 1024 + 333 + remainder
    // exercises the buffered-residue path on both stages.
    let mut start = 0_usize;
    for &chunk in &[1024_usize, 333, 997, 512] {
        let end = (start + chunk).min(audio.len());
        if end <= start {
            break;
        }
        parts.push(pipeline.push_samples(&audio[start..end]).expect("push"));
        start = end;
        if start >= audio.len() {
            break;
        }
    }
    if start < audio.len() {
        parts.push(pipeline.push_samples(&audio[start..]).expect("push rest"));
    }
    parts.push(pipeline.flush().expect("flush"));
    let streaming = concat_outputs(parts);

    assert!(streaming.tse_applied);

    // Per-VAD-frame gate state must still match exactly — TSE only
    // touches the audio path, the gate decisions are byte-identical
    // to the offline reference.
    assert_eq!(
        streaming.gate_per_frame.len(),
        offline.gate_per_frame.len(),
        "frame counts diverge (streaming={} offline={})",
        streaming.gate_per_frame.len(),
        offline.gate_per_frame.len()
    );
    for (i, (&s, &o)) in streaming
        .gate_per_frame
        .iter()
        .zip(offline.gate_per_frame.iter())
        .enumerate()
    {
        assert_eq!(
            s, o,
            "TSE gate_per_frame[{i}] differs: streaming={s} offline={o}"
        );
    }

    // The streaming engine's audio output length should be close to
    // the offline reference. The offline path re-applies the
    // envelope over the full TSE-extracted buffer (exact input
    // length); the streaming engine emits the same number of samples
    // after `flush()` drains the TSE accumulator's tail.
    let expected = offline.audio.len() as i64;
    let got = streaming.audio.len() as i64;
    let slack = (TSE_CHUNK as i64).max(expected / 50);
    assert!(
        (got - expected).abs() <= slack,
        "streaming TSE output length {got} far from offline {expected} (slack {slack})"
    );

    // The output sequences should agree numerically over their
    // common prefix. The smoke ONNX is deterministic given the cond
    // embedding, so both runs see the same TSE output sample-for-
    // sample; the small differences are from the envelope being
    // advanced one VAD-frame at a time (streaming) vs. as one
    // run-length sweep (offline). 1e-3 covers that gap with margin.
    let n = streaming.audio.len().min(offline.audio.len());
    let mut max_abs = 0.0_f32;
    for i in 0..n {
        let d = (streaming.audio[i] - offline.audio[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs <= 1e-3,
        "streaming-vs-offline TSE max abs diff = {max_abs}, exceeds 1e-3"
    );
}
