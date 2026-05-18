//! End-to-end smoke test for [`mellonella_core::pipeline::process_offline`].
//!
//! Loads the ECAPA and VAD ONNX models from `MELLONELLA_ECAPA_ONNX` and
//! `MELLONELLA_VAD_ONNX` env vars (with `ORT_DYLIB_PATH` for the runtime),
//! runs the pipeline on a synthetic 2 s harmonic-stack waveform, and
//! checks that the result is well-formed: same length as input,
//! per-frame outputs aligned, decisions start at sample 0.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{process_offline, PipelineComponents, PipelineConfig};
use mellonella_core::vad::{SileroVad, CHUNK_SAMPLES_16K};
use mellonella_core::{embedding::EcapaTdnn, vad};

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

#[test]
fn process_offline_runs_end_to_end() {
    let Ok(ecapa_path) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
        return;
    };
    let Ok(vad_path) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return;
    };
    if !std::path::Path::new(&ecapa_path).exists() || !std::path::Path::new(&vad_path).exists() {
        eprintln!("[skip] ONNX file(s) missing");
        return;
    }
    // Sanity: `vad` module exports must be visible (used in the helper
    // for chunk size).
    assert_eq!(CHUNK_SAMPLES_16K, vad::CHUNK_SAMPLES_16K);

    let fb_matrix = read_f32_buffer(FILTERBANK);
    let fbank = Fbank::new(&fb_matrix).expect("Fbank from fixture");
    let ecapa = EcapaTdnn::from_onnx_path(&ecapa_path).expect("ECAPA load");
    let vad_model = SileroVad::from_onnx_path(&vad_path, 16_000).expect("VAD load");

    let mut components = PipelineComponents {
        vad: vad_model,
        fbank,
        ecapa,
        cohort: Vec::new(),
        tse: None,
    };

    // Pre-seed the pool with a synthetic anchor so cos_sim_max has
    // something to score against. The anchor doesn't have to be
    // mathematically related to the input — we're just smoke-testing
    // the orchestration, not the gating decisions themselves.
    let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
    pool.add_anchors([vec![0.0_f32; 192]]);

    let audio = synth_waveform(16_000, 2.0, 180.0);

    let pipeline_cfg = PipelineConfig::default();
    let gate_cfg = GateConfig::default();
    let result = process_offline(
        &audio,
        16_000,
        &mut pool,
        &pipeline_cfg,
        &gate_cfg,
        &mut components,
    )
    .expect("pipeline runs without error");

    // Output audio is the same length as input (no resample, no DFN3).
    assert_eq!(result.audio.len(), audio.len());

    // Per-frame arrays are aligned.
    assert_eq!(result.gate_per_frame.len(), result.score_per_frame.len());
    assert_eq!(
        result.gate_per_frame.len(),
        result.cos_sim_max_per_frame.len()
    );
    assert_eq!(result.gate_per_frame.len(), result.f0_match_per_frame.len());

    // Frame count: every full VAD frame, plus one zero-padded flush
    // frame for the trailing sub-frame remainder. `process_offline` is
    // a thin wrapper over the streaming engine, which flushes the tail
    // (see streaming.rs) — so the count is `ceil`, not `floor`, of
    // `audio.len() / vad_frame`.
    let n_frames_expected = audio.len().div_ceil(CHUNK_SAMPLES_16K);
    assert_eq!(result.gate_per_frame.len(), n_frames_expected);

    // Decisions always start at sample 0.
    assert!(!result.gate_decisions.is_empty());
    assert_eq!(result.gate_decisions[0].0, 0);

    // No NaNs leaked into the gated output.
    assert!(result.audio.iter().all(|v| v.is_finite()));
}

#[test]
fn process_offline_async_runs_end_to_end() {
    let Ok(ecapa_path) = std::env::var("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set");
        return;
    };
    let Ok(vad_path) = std::env::var("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set");
        return;
    };
    if !std::path::Path::new(&ecapa_path).exists() || !std::path::Path::new(&vad_path).exists() {
        eprintln!("[skip] ONNX file(s) missing");
        return;
    }

    let fb_matrix = read_f32_buffer(FILTERBANK);
    let fbank = Fbank::new(&fb_matrix).expect("Fbank from fixture");
    let ecapa = EcapaTdnn::from_onnx_path(&ecapa_path).expect("ECAPA load");
    let vad_model = SileroVad::from_onnx_path(&vad_path, 16_000).expect("VAD load");

    let mut components = PipelineComponents {
        vad: vad_model,
        fbank,
        ecapa,
        cohort: Vec::new(),
        tse: None,
    };

    let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
    pool.add_anchors([vec![0.0_f32; 192]]);

    let audio = synth_waveform(16_000, 2.0, 180.0);

    let pipeline_cfg = PipelineConfig {
        async_refresh: true,
        // Match the sync smoke-test's default behaviour otherwise so
        // any failure mode that's specific to async surfaces here.
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();
    let result = process_offline(
        &audio,
        16_000,
        &mut pool,
        &pipeline_cfg,
        &gate_cfg,
        &mut components,
    )
    .expect("async pipeline runs without error");

    assert_eq!(result.audio.len(), audio.len());
    assert_eq!(result.gate_per_frame.len(), result.score_per_frame.len());
    // `process_offline_async` still uses its own hand-rolled loop that
    // drops the trailing sub-frame remainder (`floor`), unlike the sync
    // path which wraps the streaming engine and flushes (`ceil`).
    // Rewiring the async path onto the streaming engine is a documented
    // follow-up (see the streaming.rs module docs); until then the two
    // offline paths legitimately differ by at most one tail frame.
    let n_frames_expected = audio.len() / CHUNK_SAMPLES_16K;
    assert_eq!(result.gate_per_frame.len(), n_frames_expected);
    assert!(!result.gate_decisions.is_empty());
    assert_eq!(result.gate_decisions[0].0, 0);
    assert!(result.audio.iter().all(|v| v.is_finite()));
    // Every per-frame score must be finite (no NaN propagation through
    // the worker channel).
    assert!(result.score_per_frame.iter().all(|v| v.is_finite()));
}
