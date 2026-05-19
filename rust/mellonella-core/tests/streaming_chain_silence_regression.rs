//! Regression test: the TSE → DFN3 chain must not emit NaN / Inf
//! when the input has a stretch of zero samples followed by real
//! audio.
//!
//! Background: TSE's STFT-feature normalisation divides by /
//! takes `log` of per-frame energy. An STFT window full of exact
//! zeros produces a NaN that gets latched into the model's GRU
//! hidden state and contaminates every subsequent frame. The
//! original PR #168 fix injected sub-LSB dither only when the
//! *entire* `audio_chunk` was below the silence threshold, which
//! missed the common case of an `audio_chunk` that opens with N
//! zero samples (WASAPI primer / leading file silence /
//! inter-phoneme pauses) and continues with real audio. The first
//! STFT window inside that chunk still sees pure zeros and NaN
//! still poisons the GRU. The follow-up fix dithers every chain
//! input sample unconditionally; this test pins that behaviour.
//!
//! Gated on `MELLONELLA_ECAPA_ONNX`, `MELLONELLA_VAD_ONNX`,
//! `MELLONELLA_TSE_PROD_48K_ONNX`, `MELLONELLA_DFN3_ONNX`,
//! `ORT_DYLIB_PATH`. Skips when any is missing.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::EmbeddingPoolConfig;
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{enroll_from_recording, PipelineComponents, PipelineConfig};
use mellonella_core::resample::resample_to;
use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
use mellonella_core::tse_stage::TseStageConfig;
use mellonella_core::vad::SileroVad;

fn skip_if_no_models() -> Option<(String, String, String, String)> {
    let ecapa = std::env::var("MELLONELLA_ECAPA_ONNX").ok()?;
    let vad = std::env::var("MELLONELLA_VAD_ONNX").ok()?;
    let tse = std::env::var("MELLONELLA_TSE_PROD_48K_ONNX").ok()?;
    let dfn3 = std::env::var("MELLONELLA_DFN3_ONNX").ok()?;
    std::env::var("ORT_DYLIB_PATH").ok()?;
    Some((ecapa, vad, tse, dfn3))
}

/// Build a 16 kHz buffer of synthetic vowel-ish audio so the
/// streaming pipeline sees something non-trivial after the leading
/// silence. A 200 Hz square + 400 Hz sine mix has enough harmonic
/// content for ECAPA's fbank stage, without needing an external WAV.
fn synth_speech_16k(secs: f32) -> Vec<f32> {
    let sr = 16_000.0_f32;
    let n = (secs * sr) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            // Vowel-ish: fundamental at 150 Hz plus two formants.
            let f0 = 150.0;
            let f1 = 700.0;
            let f2 = 1_200.0;
            let envelope = (2.0 * std::f32::consts::PI * 3.0 * t).sin().abs(); // 3 Hz amplitude wobble
            0.3 * envelope
                * ((2.0 * std::f32::consts::PI * f0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * f2 * t).sin())
        })
        .collect()
}

#[test]
fn leading_silence_then_speech_does_not_produce_nan() {
    let Some((ecapa, vad, tse, dfn3)) = skip_if_no_models() else {
        eprintln!("[skip] ONNX env vars not set");
        return;
    };

    // Build a 16 kHz signal: 100 ms of exact zeros, then 4 s of
    // synthetic vowel. This shape mirrors what cpal's WASAPI input
    // primer looks like — N samples of zero before real mic data
    // lands. Resample to the GUI's 48 kHz live rate before feeding
    // the chain.
    let mut signal_16k: Vec<f32> = vec![0.0; 1_600]; // 100 ms zeros
    signal_16k.extend_from_slice(&synth_speech_16k(4.0));
    let signal_48k = resample_to(&signal_16k, 16_000, 48_000).expect("resample 16k → 48k");

    // Enrol a pool on the synth speech (skip the leading zeros so
    // ECAPA sees real spectral content during enrolment).
    let speech_only = &signal_16k[1_600..];
    let mut comp = PipelineComponents {
        vad: SileroVad::from_onnx_path(&vad, 16_000).unwrap(),
        fbank: Fbank::with_speechbrain_filterbank().unwrap(),
        ecapa: EcapaTdnn::from_onnx_path(&ecapa).unwrap(),
        cohort: Vec::new(),
        tse: None,
    };
    let pool =
        enroll_from_recording(speech_only, &mut comp, EmbeddingPoolConfig::default()).unwrap();

    // GUI's exact live config: TSE Prod48k + DFN3 + default gate.
    // The only thing we care about is whether the chain emits
    // finite samples — the gate's binary decisions are not the
    // subject of this regression.
    let pipeline_cfg = PipelineConfig {
        tse: Some(TseStageConfig::new_prod_48k(PathBuf::from(&tse))),
        ..PipelineConfig::default()
    };
    let cfg = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: GateConfig::default(),
        audio_sample_rate: 48_000,
        diagnostics: false,
        dfn3_onnx_path: Some(PathBuf::from(&dfn3)),
    };
    let mut pipeline = StreamingPipeline::new(pool, cfg, comp).expect("streaming pipeline builds");

    // Push 10-ms chunks (480 samples @ 48 kHz) — same cadence as
    // the live worker.
    let mut all_output: Vec<f32> = Vec::with_capacity(signal_48k.len());
    for chunk in signal_48k.chunks(480) {
        let out = pipeline.push_samples(chunk).expect("push_samples");
        all_output.extend_from_slice(&out.audio);
    }
    let tail = pipeline.flush().expect("flush");
    all_output.extend_from_slice(&tail.audio);

    let nan_count = all_output.iter().filter(|s| s.is_nan()).count();
    let inf_count = all_output
        .iter()
        .filter(|s| !s.is_finite() && !s.is_nan())
        .count();
    assert_eq!(
        nan_count, 0,
        "chain emitted {nan_count} NaN samples after a leading-silence input — the always-on \
         chain dither is missing or insufficient"
    );
    assert_eq!(
        inf_count, 0,
        "chain emitted {inf_count} Inf samples — clipping or division-by-zero somewhere"
    );
    assert!(
        !all_output.is_empty(),
        "chain produced no output for {} input samples",
        signal_48k.len()
    );
}
