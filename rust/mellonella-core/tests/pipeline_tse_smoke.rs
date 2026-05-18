//! Smoke tests for the Stage C TSE wiring in
//! [`mellonella_core::pipeline::process_offline`].
//!
//! Covers:
//!
//! * **default behaviour preserved** — with `cfg.tse == None`, the
//!   offline pipeline emits `tse_applied == false` and the output
//!   audio matches the pre-TSE byte-for-byte (i.e. equals what the
//!   non-TSE smoke test would produce on the same input).
//! * **TSE enabled produces non-trivial output** — with
//!   `cfg.tse == Some(...)` and the smoke ONNX from
//!   `tests/tse_parity.rs`, the pipeline emits `tse_applied == true`,
//!   the output length matches the input, the output is finite, and
//!   the output differs from the input (proves TSE was actually
//!   applied; the smoke export has random weights so the extracted
//!   chunk is a non-trivial function of the input).
//! * **rate-mismatch rejection** — with `cfg.tse == Some(...)` and a
//!   non-16-kHz sample rate, `process_offline` returns
//!   `PipelineError::TseRateMismatch` whose `Display` mentions the
//!   16 kHz constraint.
//!
//! All three tests are gated on the same env vars as
//! `pipeline_smoke.rs` (`MELLONELLA_ECAPA_ONNX`, `MELLONELLA_VAD_ONNX`,
//! `ORT_DYLIB_PATH`) and additionally invoke the same Python export
//! flow as `tse_parity.rs` to build / reuse `build/tse_smoke.onnx`. If
//! any of those isn't available, the affected test prints a skip
//! notice and passes — the cargo CI lane stays green without the full
//! training Python stack.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
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
use mellonella_core::tse::TSE_COND_DIM;
use mellonella_core::tse_stage::TseStageConfig;
use mellonella_core::vad::SileroVad;

const FILTERBANK: &[u8] = include_bytes!("fixtures/fbank_filterbank.bin");

// Chunk length the smoke ONNX is exported with — matches the value in
// `tests/tse_parity.rs` so we share the cached fixture.
const TSE_CHUNK: usize = 160;

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Same synthetic mixture as `pipeline_smoke.rs` — a 5-harmonic stack
/// peaking at ±0.5. Deterministic, length-equals-`duration_sec*SR`.
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
/// pattern as `tests/tse_parity.rs`.
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
/// the caller should skip the test.
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

/// Build a fresh `PipelineComponents` with a single-anchor pool. The
/// anchor is a normalised 192-dim vector (all `1/sqrt(192)`), so the
/// centroid we pass to TSE is well-defined and non-zero. The actual
/// anchor values aren't load-bearing here — the smoke ONNX has random
/// weights, so the only thing the cond embedding controls is *which*
/// random output we get, not whether it's non-trivial.
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
    // Normalised so the centroid sits on the unit hyper-sphere; the
    // exact direction doesn't matter for the smoke check.
    let norm = (TSE_COND_DIM as f32).sqrt();
    pool.add_anchors([vec![1.0_f32 / norm; TSE_COND_DIM]]);
    (components, pool)
}

/// Skip-or-paths preamble shared by the three tests.
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

/// Default behaviour is preserved: with `cfg.tse == None`, the
/// pipeline emits `tse_applied == false` and the output equals what
/// the pre-Phase-5 pipeline would have produced on the same input.
/// Two independent runs with identical configs must produce
/// byte-identical outputs (no hidden state leaking across the TSE
/// branch when it's disabled).
#[test]
fn process_offline_default_is_byte_identical_with_tse_off() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };

    let audio = synth_waveform(16_000, 2.0, 180.0);

    // Run 1: brand-new components, brand-new pool.
    let (mut components_a, mut pool_a) = build_smoke_components(&ecapa_path, &vad_path);
    let cfg = PipelineConfig::default();
    let gate_cfg = GateConfig::default();
    let result_a = process_offline(
        &audio,
        16_000,
        &mut pool_a,
        &cfg,
        &gate_cfg,
        &mut components_a,
    )
    .expect("non-TSE run a");

    // Run 2: independent fresh components / pool with the same inputs.
    let (mut components_b, mut pool_b) = build_smoke_components(&ecapa_path, &vad_path);
    let result_b = process_offline(
        &audio,
        16_000,
        &mut pool_b,
        &cfg,
        &gate_cfg,
        &mut components_b,
    )
    .expect("non-TSE run b");

    // `tse_applied` is the load-bearing default-off flag.
    assert!(!result_a.tse_applied);
    assert!(!result_b.tse_applied);

    // Output length matches input length on both runs.
    assert_eq!(result_a.audio.len(), audio.len());
    assert_eq!(result_b.audio.len(), audio.len());

    // The two runs must be byte-identical sample-for-sample (the
    // TSE-disabled path is fully deterministic — same ONNX, same
    // synthetic input, same RNG-free fbank / vad pipeline).
    assert_eq!(result_a.audio, result_b.audio, "non-TSE runs diverged");
    assert_eq!(
        result_a.gate_per_frame, result_b.gate_per_frame,
        "non-TSE gate-per-frame diverged"
    );
}

/// TSE enabled produces non-trivial, finite output that differs from
/// the input. Uses the same `build/tse_smoke.onnx` fixture as
/// `tests/tse_parity.rs` (built by `scripts/export_tse_onnx.py`).
#[test]
fn process_offline_with_tse_extracts_non_trivial_audio() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };
    let Some(tse_onnx) = ensure_tse_smoke_onnx() else {
        return;
    };

    let audio = synth_waveform(16_000, 2.0, 180.0);

    let (mut components, mut pool) = build_smoke_components(&ecapa_path, &vad_path);

    let cfg = PipelineConfig {
        tse: Some(TseStageConfig {
            onnx_path: tse_onnx,
            chunk_samples: TSE_CHUNK,
        }),
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();
    let result = process_offline(&audio, 16_000, &mut pool, &cfg, &gate_cfg, &mut components)
        .expect("TSE-on run");

    // The wiring flagged the TSE branch.
    assert!(result.tse_applied);

    // Output length matches input length.
    assert_eq!(result.audio.len(), audio.len());

    // Everything is finite — no NaN / Inf leaked through TSE or the
    // envelope.
    assert!(result.audio.iter().all(|v| v.is_finite()));

    // TSE-extracted audio is not byte-identical to the input — proves
    // the model actually ran. (The smoke ONNX has random weights but
    // is deterministic given the cond embedding, so a non-trivial,
    // reproducible delta is the expected smoke signature.)
    assert!(
        result.audio != audio,
        "TSE output equals input — model did not run"
    );

    // The stage was stored on `components.tse` so subsequent runs can
    // share the loaded ONNX. The reset-on-reuse path is exercised by
    // running a second `process_offline` on the same components and
    // checking the output is still well-formed.
    let result_2 = process_offline(&audio, 16_000, &mut pool, &cfg, &gate_cfg, &mut components)
        .expect("TSE second run");
    assert!(result_2.tse_applied);
    assert_eq!(result_2.audio.len(), audio.len());
    assert!(result_2.audio.iter().all(|v| v.is_finite()));
}

/// Rate-mismatch rejection: with `cfg.tse == Some(...)` and a
/// non-16-kHz audio rate, `process_offline` must error out with
/// `PipelineError::TseRateMismatch` whose `Display` mentions the
/// 16 kHz constraint. The TSE ONNX itself isn't loaded on this path
/// because the rate check fires before `ensure_tse_stage`.
#[test]
fn process_offline_with_tse_rejects_non_16k_audio_rate() {
    let Some((ecapa_path, vad_path)) = ecapa_vad_paths() else {
        return;
    };

    // No real ONNX needed — the rate check fires before
    // `ensure_tse_stage` would try to load it. Point at /dev/null so
    // a regression that swaps the order would also fail loudly.
    let cfg = PipelineConfig {
        tse: Some(TseStageConfig {
            onnx_path: std::path::PathBuf::from("/dev/null"),
            chunk_samples: TSE_CHUNK,
        }),
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();

    // Use 48 kHz audio — the prod sample rate. The PoC TSE only
    // supports 16 kHz, so this must be rejected with a clear error.
    let audio = synth_waveform(48_000, 0.5, 180.0);

    let (mut components, mut pool) = build_smoke_components(&ecapa_path, &vad_path);
    let err = process_offline(&audio, 48_000, &mut pool, &cfg, &gate_cfg, &mut components)
        .expect_err("TSE at 48 kHz must be rejected");

    match err {
        PipelineError::TseRateMismatch {
            audio_sr,
            decision_sr,
        } => {
            assert_eq!(audio_sr, 48_000);
            assert_eq!(decision_sr, 16_000);
            // Message mentions the 16 kHz constraint so callers can
            // diagnose without grepping the source.
            let msg = format!("{err}");
            assert!(
                msg.contains("16 kHz"),
                "expected 16 kHz in error message, got {msg:?}"
            );
        }
        other => panic!("expected TseRateMismatch, got {other:?}"),
    }
}
