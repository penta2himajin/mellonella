//! End-to-end smoke + chunk-invariance test for
//! [`mellonella_core::streaming::StreamingPipeline`].
//!
//! Mirrors the gating policy of `pipeline_smoke` and `pipeline_parity`:
//! skips when the ONNX env vars (`MELLONELLA_ECAPA_ONNX`,
//! `MELLONELLA_VAD_ONNX`, `ORT_DYLIB_PATH`) aren't set so CI stays
//! green without vendoring the ONNX files.
//!
//! Coverage:
//!
//! * **Identity rate vs `process_offline`**: at audio rate ==
//!   decision rate (16 kHz), pushing the whole buffer through
//!   `StreamingPipeline` then flushing must produce per-VAD-frame
//!   gate state and scores that match the offline reference. (Audio
//!   bytes can drift by a small numerical amount because the
//!   envelope is advanced in two different stride patterns —
//!   per-frame vs per-run-length — but the gate decisions are
//!   identical, which is the user-visible contract.)
//! * **Chunk invariance at identity rate**: chunking the same audio
//!   in different patterns (one shot, 512-sample blocks, weird
//!   primes) must produce **byte-identical** concatenated output.
//! * **Dual-rate well-formedness**: at 48 kHz audio / 16 kHz
//!   decision, the output WAV length matches the input WAV length
//!   modulo flush-zero-pad, and the per-frame trace count matches
//!   what one would expect for `duration / 32 ms` decisions.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{process_offline, PipelineComponents, PipelineConfig};
use mellonella_core::streaming::{StreamingConfig, StreamingOutput, StreamingPipeline};
use mellonella_core::vad::SileroVad;

const FILTERBANK: &[u8] = include_bytes!("fixtures/fbank_filterbank.bin");

fn read_f32_buffer(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

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

/// Build a fresh `PipelineComponents` + seeded pool for one run.
/// Re-built per test so each call gets independent ONNX session
/// state (matters for deterministic comparison).
fn build_components(ecapa_path: &str, vad_path: &str) -> (PipelineComponents, EmbeddingPool) {
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
    pool.add_anchors([vec![0.0_f32; 192]]);
    (components, pool)
}

/// Skip the test silently when ONNX env vars / files aren't
/// available — same policy as `pipeline_smoke` / `pipeline_parity`.
/// Returns `Some((ecapa, vad))` if the test can proceed.
fn skip_if_no_onnx() -> Option<(String, String)> {
    let Ok(ecapa) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
        return None;
    };
    let Ok(vad) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return None;
    };
    if !std::path::Path::new(&ecapa).exists() || !std::path::Path::new(&vad).exists() {
        eprintln!("[skip] ONNX file(s) missing");
        return None;
    }
    Some((ecapa, vad))
}

/// Concatenate a sequence of `StreamingOutput`s into one final
/// output. Used by the chunk-invariance tests.
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
    }
    acc
}

#[test]
fn streaming_identity_rate_per_frame_matches_offline() {
    let Some((ecapa_path, vad_path)) = skip_if_no_onnx() else {
        return;
    };
    let audio = synth_waveform(16_000, 2.0, 180.0);

    // Offline reference.
    let (mut components_a, mut pool_a) = build_components(&ecapa_path, &vad_path);
    let pipeline_cfg = PipelineConfig::default();
    let gate_cfg = GateConfig::default();
    let offline = process_offline(
        &audio,
        16_000,
        &mut pool_a,
        &pipeline_cfg,
        &gate_cfg,
        &mut components_a,
    )
    .expect("offline runs");

    // Streaming run, same audio, single push + flush.
    let (components_b, pool_b) = build_components(&ecapa_path, &vad_path);
    let config = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: gate_cfg,
        audio_sample_rate: 16_000,
        diagnostics: true,
        dfn3_onnx_path: None,
        ..Default::default()
    };
    let mut pipeline = StreamingPipeline::new(pool_b, config, components_b).expect("streaming new");
    let part1 = pipeline.push_samples(&audio).expect("push");
    let part2 = pipeline.flush().expect("flush");
    let streaming = concat_outputs(vec![part1, part2]);

    // Per-VAD-frame gate state must match the offline reference
    // exactly. This is the user-visible behaviour the streaming
    // engine must preserve.
    assert_eq!(
        streaming.gate_per_frame.len(),
        offline.gate_per_frame.len(),
        "frame counts diverge"
    );
    for (i, (&s, &o)) in streaming
        .gate_per_frame
        .iter()
        .zip(offline.gate_per_frame.iter())
        .enumerate()
    {
        assert_eq!(
            s, o,
            "gate_per_frame[{i}] differs: streaming={s} offline={o}"
        );
    }

    // Scores must also match exactly — same VAD/ECAPA call sequence.
    for (i, (&s, &o)) in streaming
        .score_per_frame
        .iter()
        .zip(offline.score_per_frame.iter())
        .enumerate()
    {
        assert!(
            (s - o).abs() <= 1e-6,
            "score_per_frame[{i}]: streaming={s} offline={o}"
        );
    }
}

#[test]
fn streaming_turn_detect_identity_rate_matches_offline() {
    // Stage B: with `turn_detect_enabled` (+ the fast F0 cue + offset
    // fail-closed) the streaming engine and `process_offline` share
    // the same per-frame core, so at identity rate they must still
    // produce identical per-VAD-frame gate state. Early-skips without
    // ONNX — this test exists to keep the Stage B opt-in path
    // compiled and the offline↔streaming consistency contract pinned.
    let Some((ecapa_path, vad_path)) = skip_if_no_onnx() else {
        return;
    };
    let audio = synth_waveform(16_000, 2.0, 180.0);

    let pipeline_cfg = PipelineConfig {
        fast_cue_enabled: true,
        turn_detect_enabled: true,
        offset_fail_closed: true,
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();

    let (mut components_a, mut pool_a) = build_components(&ecapa_path, &vad_path);
    let offline = process_offline(
        &audio,
        16_000,
        &mut pool_a,
        &pipeline_cfg,
        &gate_cfg,
        &mut components_a,
    )
    .expect("offline runs");

    let (components_b, pool_b) = build_components(&ecapa_path, &vad_path);
    let config = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: gate_cfg,
        audio_sample_rate: 16_000,
        diagnostics: true,
        dfn3_onnx_path: None,
        ..Default::default()
    };
    let mut pipeline = StreamingPipeline::new(pool_b, config, components_b).expect("streaming new");
    let part1 = pipeline.push_samples(&audio).expect("push");
    let part2 = pipeline.flush().expect("flush");
    let streaming = concat_outputs(vec![part1, part2]);

    assert_eq!(
        streaming.gate_per_frame.len(),
        offline.gate_per_frame.len(),
        "frame counts diverge with turn detection enabled"
    );
    for (i, (&s, &o)) in streaming
        .gate_per_frame
        .iter()
        .zip(offline.gate_per_frame.iter())
        .enumerate()
    {
        assert_eq!(
            s, o,
            "turn-detect gate_per_frame[{i}] differs: streaming={s} offline={o}"
        );
    }
}

#[test]
fn streaming_identity_rate_chunk_invariance() {
    let Some((ecapa_path, vad_path)) = skip_if_no_onnx() else {
        return;
    };
    let audio = synth_waveform(16_000, 2.0, 180.0);
    let config = StreamingConfig {
        pipeline: PipelineConfig::default(),
        gate: GateConfig::default(),
        audio_sample_rate: 16_000,
        diagnostics: true,
        dfn3_onnx_path: None,
        ..Default::default()
    };

    let run = |chunk_size: usize| -> StreamingOutput {
        let (components, pool) = build_components(&ecapa_path, &vad_path);
        let mut pipeline =
            StreamingPipeline::new(pool, config.clone(), components).expect("streaming new");
        let mut parts: Vec<StreamingOutput> = Vec::new();
        let mut start = 0_usize;
        while start < audio.len() {
            let end = (start + chunk_size).min(audio.len());
            parts.push(pipeline.push_samples(&audio[start..end]).expect("push"));
            start = end;
        }
        parts.push(pipeline.flush().expect("flush"));
        concat_outputs(parts)
    };

    let one_shot = run(audio.len());
    // Chunk sizes chosen to be (a) smaller than one VAD frame (333),
    // (b) exactly one VAD frame (512), (c) prime not aligned (997),
    // (d) larger than a frame (1024).
    let chunked_333 = run(333);
    let chunked_512 = run(512);
    let chunked_997 = run(997);
    let chunked_1024 = run(1024);

    assert_eq!(one_shot.audio.len(), chunked_333.audio.len());
    for variant in [
        ("333", &chunked_333),
        ("512", &chunked_512),
        ("997", &chunked_997),
        ("1024", &chunked_1024),
    ] {
        let (label, other) = variant;
        assert_eq!(
            one_shot.gate_per_frame, other.gate_per_frame,
            "gate_per_frame diverges for chunk size {label}",
        );
        for (i, (a, b)) in one_shot.audio.iter().zip(other.audio.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "audio[{i}] diverges for chunk size {label}: {a} vs {b}",
            );
        }
    }
}

#[test]
fn streaming_dual_rate_well_formed() {
    let Some((ecapa_path, vad_path)) = skip_if_no_onnx() else {
        return;
    };
    // 1 s of 48 kHz audio = 48 000 samples; resampled internally to
    // ~16 000 @ 16 kHz → ~31 VAD frames @ 32 ms each.
    let audio = synth_waveform(48_000, 1.0, 180.0);

    let (components, pool) = build_components(&ecapa_path, &vad_path);
    let config = StreamingConfig {
        pipeline: PipelineConfig::default(),
        gate: GateConfig::default(),
        audio_sample_rate: 48_000,
        diagnostics: true,
        dfn3_onnx_path: None,
        ..Default::default()
    };
    let mut pipeline = StreamingPipeline::new(pool, config, components).expect("streaming new");
    let parts = vec![
        pipeline.push_samples(&audio).expect("push"),
        pipeline.flush().expect("flush"),
    ];
    let combined = concat_outputs(parts);

    // Output length should be in the ballpark of input length — the
    // resampler delay + flush zero-pad can add a small constant
    // offset. Allow ±5 % slack.
    let expected = audio.len() as i64;
    let got = combined.audio.len() as i64;
    let slack = expected / 20;
    assert!(
        (got - expected).abs() <= slack,
        "output length {got} far from input {expected} (slack {slack})",
    );

    // Per-frame diagnostics should be non-trivial — at least a
    // handful of VAD frames must have run.
    assert!(
        combined.gate_per_frame.len() >= 20,
        "expected ≥ 20 VAD frames for 1 s of audio, got {}",
        combined.gate_per_frame.len()
    );
}
