//! Realtime-factor benchmark for the adaptive chain. Not a pass/fail
//! test — it prints how fast the Solo (DFN3) and Overlap (TSE) chains
//! process audio relative to wall clock, which is what determines
//! whether the live worker keeps up (realtime_factor > 1.0) or
//! underruns (< 1.0). Used to validate the ONNX thread-count change.
//!
//! Run with the full ONNX env set + MELLONELLA_ORT_INTRA_THREADS to
//! compare thread counts:
//!   MELLONELLA_ORT_INTRA_THREADS=2 cargo test ... -- --nocapture
//!   MELLONELLA_ORT_INTRA_THREADS=4 cargo test ... -- --nocapture

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;
use std::time::Instant;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::EmbeddingPoolConfig;
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{enroll_from_recording, PipelineComponents, PipelineConfig};
use mellonella_core::resample::resample_to;
use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
use mellonella_core::tse_stage::TseStageConfig;
use mellonella_core::vad::SileroVad;

fn env_path(name: &str) -> Option<PathBuf> {
    let p = std::env::var_os(name).map(PathBuf::from)?;
    p.exists().then_some(p)
}

fn read_pcm16_mono_wav(path: &PathBuf) -> (Vec<f32>, u32) {
    let bytes = std::fs::read(path).expect("read WAV");
    let mut i = 12_usize;
    let mut fmt_off = None;
    let mut data_off = None;
    while i + 8 <= bytes.len() {
        let sz =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        match &bytes[i..i + 4] {
            b"fmt " => fmt_off = Some(i + 8),
            b"data" => data_off = Some((i + 8, sz)),
            _ => {}
        }
        i += 8 + sz;
    }
    let fmt = fmt_off.unwrap();
    let (data, dlen) = data_off.unwrap();
    let sr = u32::from_le_bytes([
        bytes[fmt + 4],
        bytes[fmt + 5],
        bytes[fmt + 6],
        bytes[fmt + 7],
    ]);
    let scale = 1.0_f32 / f32::from(i16::MAX);
    let samples = bytes[data..data + dlen]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) * scale)
        .collect();
    (samples, sr)
}

#[test]
fn realtime_factor_solo_and_overlap() {
    let (Some(ecapa), Some(vad), Some(dfn3), Some(tse), Some(seg)) = (
        env_path("MELLONELLA_ECAPA_ONNX"),
        env_path("MELLONELLA_VAD_ONNX"),
        env_path("MELLONELLA_DFN3_ONNX"),
        env_path("MELLONELLA_TSE_PROD_48K_ONNX"),
        env_path("MELLONELLA_OVERLAP_SEG_ONNX"),
    ) else {
        eprintln!("[skip] ONNX env vars missing");
        return;
    };
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        return;
    }
    let (Some(solo_wav), Some(mix_wav)) = (
        env_path("MELLONELLA_OVERLAP_TEST_SOLO_WAV"),
        env_path("MELLONELLA_OVERLAP_TEST_MIX_WAV"),
    ) else {
        eprintln!("[skip] test WAVs missing");
        return;
    };

    let threads =
        std::env::var("MELLONELLA_ORT_INTRA_THREADS").unwrap_or_else(|_| "(default cap)".into());
    eprintln!("[rtf] intra-op threads = {threads}");

    let to48 = |p: &PathBuf| {
        let (s, sr) = read_pcm16_mono_wav(p);
        let s16 = if sr == 16_000 {
            s
        } else {
            resample_to(&s, sr, 16_000).unwrap()
        };
        (resample_to(&s16, 16_000, 48_000).unwrap(), s16)
    };
    let (solo48, solo16) = to48(&solo_wav);
    let (mix48, _) = to48(&mix_wav);

    let mut comp = PipelineComponents {
        vad: SileroVad::from_onnx_path(&vad, 16_000).unwrap(),
        fbank: Fbank::with_speechbrain_filterbank().unwrap(),
        ecapa: EcapaTdnn::from_onnx_path(&ecapa).unwrap(),
        cohort: Vec::new(),
        tse: None,
    };
    let pool = enroll_from_recording(&solo16, &mut comp, EmbeddingPoolConfig::default()).unwrap();

    let pipeline_cfg = PipelineConfig {
        tse: Some(TseStageConfig::new_prod_48k(tse.clone())),
        ..PipelineConfig::default()
    };
    let cfg = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: GateConfig::default(),
        audio_sample_rate: 48_000,
        diagnostics: false,
        dfn3_onnx_path: Some(dfn3.clone()),
        overlap_onnx_path: Some(seg.clone()),
        ..StreamingConfig::default()
    };
    let mut pipeline = StreamingPipeline::new(pool, cfg, comp).unwrap();

    // Measure Solo (push solo audio, mode stays Solo) then Overlap
    // (push mix audio long enough for the mode to flip).
    let bench = |pipeline: &mut StreamingPipeline, audio: &[f32], label: &str| {
        let t0 = Instant::now();
        for c in audio.chunks(480) {
            let _ = pipeline.push_samples(c).unwrap();
        }
        let wall = t0.elapsed().as_secs_f64();
        let audio_secs = audio.len() as f64 / 48_000.0;
        let rtf = audio_secs / wall;
        eprintln!(
            "[rtf] {label:8}: {audio_secs:.2}s audio in {wall:.2}s wall → {rtf:.2}× realtime{}",
            if rtf < 1.0 { "  ⚠ UNDERRUNS" } else { "" }
        );
    };

    bench(&mut pipeline, &solo48, "Solo");
    // Push the mix repeatedly so the detector flips to Overlap and we
    // measure steady-state TSE throughput.
    let mix_long: Vec<f32> = mix48.iter().chain(mix48.iter()).copied().collect();
    bench(&mut pipeline, &mix_long, "Overlap");
    eprintln!("[rtf] chain mode at end: {:?}", pipeline.chain_mode());
}
