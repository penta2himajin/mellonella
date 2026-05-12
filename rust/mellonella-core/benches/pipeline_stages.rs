//! Per-stage micro-benchmarks for the offline pipeline.
//!
//! Measures the throughput of:
//!
//! * `Fbank::compute`        — STFT + mel projection + log_mel
//! * `SileroVad::score`      — VAD inference, one 32 ms chunk
//! * `EcapaTdnn::embed_features` — embedding-only ONNX, one window
//! * `Dfn3Pipeline::process` — full audio → audio chunk
//! * `process_offline`       — end-to-end gating pipeline
//!
//! All ONNX-dependent benches are gated on the same env vars the
//! integration tests use: `MELLONELLA_ECAPA_ONNX`,
//! `MELLONELLA_VAD_ONNX`, `MELLONELLA_DFN3_ONNX`, `ORT_DYLIB_PATH`.
//! Without them the bench function prints a skip notice and returns —
//! `cargo bench` still completes for the dep-free benches.
//!
//! Run:
//!     ORT_DYLIB_PATH=… MELLONELLA_ECAPA_ONNX=… MELLONELLA_VAD_ONNX=… \
//!     MELLONELLA_DFN3_ONNX=… cargo bench

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use mellonella_core::dfn3::{Dfn3Pipeline, SAMPLES_PER_CHUNK as DFN3_SAMPLES};
use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::f0::{estimate_f0_track, DEFAULT_F_MAX, DEFAULT_F_MIN};
use mellonella_core::features::{Fbank, N_MELS};
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{process_offline, PipelineComponents, PipelineConfig};
use mellonella_core::vad::{SileroVad, CHUNK_SAMPLES_16K};

fn synth_waveform(sample_rate: u32, duration_sec: f32, f0: f32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n = (sr * duration_sec) as usize;
    (0..n)
        .map(|i| {
            (2.0 * std::f32::consts::PI * f0 * (i as f32) / sr).sin() * 0.5
                + (2.0 * std::f32::consts::PI * f0 * 2.0 * (i as f32) / sr).sin() * 0.25
        })
        .collect()
}

fn env_path(var: &str) -> Option<std::path::PathBuf> {
    let p = std::env::var(var).ok()?;
    let path = std::path::PathBuf::from(p);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn bench_f0_track(c: &mut Criterion) {
    // 1 s @ 16 kHz at 200 Hz harmonic stack — what the pipeline calls
    // estimate_f0_track on after each SV-window refresh.
    let audio = synth_waveform(16_000, 1.0, 200.0);
    let mut group = c.benchmark_group("f0");
    group.throughput(Throughput::Elements(audio.len() as u64));
    group.bench_function("track_1s_16khz", |b| {
        b.iter(|| {
            let track = estimate_f0_track(
                black_box(&audio),
                16_000,
                2048,
                512,
                DEFAULT_F_MIN,
                DEFAULT_F_MAX,
            );
            black_box(track);
        });
    });
    group.finish();
}

fn bench_fbank(c: &mut Criterion) {
    let audio = synth_waveform(16_000, 1.0, 180.0);
    let mut fbank = Fbank::with_speechbrain_filterbank().expect("Fbank init");
    let mut group = c.benchmark_group("fbank");
    group.throughput(Throughput::Elements(audio.len() as u64));
    group.bench_function("1s_16khz", |b| {
        b.iter(|| {
            let out = fbank.compute(black_box(&audio));
            black_box(out);
        });
    });
    group.finish();
}

fn bench_vad(c: &mut Criterion) {
    let Some(vad_path) = env_path("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set — skipping VAD bench");
        return;
    };
    let mut vad = match SileroVad::from_onnx_path(&vad_path, 16_000) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[skip] VAD load failed: {e}");
            return;
        }
    };
    let chunk = vec![0.01_f32; CHUNK_SAMPLES_16K];
    let mut group = c.benchmark_group("vad");
    group.throughput(Throughput::Elements(CHUNK_SAMPLES_16K as u64));
    group.bench_function("32ms_chunk", |b| {
        b.iter(|| {
            let p = vad.score(black_box(&chunk)).expect("VAD score");
            black_box(p);
        });
    });
    group.finish();
}

fn bench_ecapa(c: &mut Criterion) {
    let Some(ecapa_path) = env_path("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set — skipping ECAPA bench");
        return;
    };
    let mut model = match EcapaTdnn::from_onnx_path(&ecapa_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[skip] ECAPA load failed: {e}");
            return;
        }
    };
    let n_frames = 100;
    let feats: Vec<f32> = (0..n_frames * N_MELS)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let mut group = c.benchmark_group("ecapa");
    group.sample_size(20); // ECAPA is heavy enough that 100 samples is overkill
    group.throughput(Throughput::Elements(1));
    group.bench_function("100frames_x_80mels", |b| {
        b.iter(|| {
            let emb = model
                .embed_features(black_box(&feats), n_frames, N_MELS)
                .expect("ECAPA embed");
            black_box(emb);
        });
    });
    group.finish();
}

fn bench_dfn3(c: &mut Criterion) {
    let Some(dfn3_path) = env_path("MELLONELLA_DFN3_ONNX") else {
        eprintln!("[skip] MELLONELLA_DFN3_ONNX not set — skipping DFN3 bench");
        return;
    };
    let mut pipeline = match Dfn3Pipeline::from_onnx_path(&dfn3_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[skip] DFN3 load failed: {e}");
            return;
        }
    };
    let audio = synth_waveform(48_000, 1.0, 180.0);
    let mut group = c.benchmark_group("dfn3");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Elements(DFN3_SAMPLES as u64));
    group.bench_function("1chunk_1.02s_48khz", |b| {
        b.iter(|| {
            let out = pipeline.process(black_box(&audio)).expect("DFN3 process");
            black_box(out);
        });
    });
    group.finish();
}

fn bench_pipeline(c: &mut Criterion) {
    let Some(ecapa_path) = env_path("MELLONELLA_ECAPA_ONNX") else {
        eprintln!("[skip] MELLONELLA_ECAPA_ONNX not set — skipping pipeline bench");
        return;
    };
    let Some(vad_path) = env_path("MELLONELLA_VAD_ONNX") else {
        eprintln!("[skip] MELLONELLA_VAD_ONNX not set — skipping pipeline bench");
        return;
    };
    let fbank = match Fbank::with_speechbrain_filterbank() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[skip] Fbank init failed: {e}");
            return;
        }
    };
    let ecapa = match EcapaTdnn::from_onnx_path(&ecapa_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[skip] ECAPA load failed: {e}");
            return;
        }
    };
    let vad = match SileroVad::from_onnx_path(&vad_path, 16_000) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[skip] VAD load failed: {e}");
            return;
        }
    };
    let mut components = PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort: Vec::new(),
    };
    let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
    pool.add_anchors([vec![0.1_f32; 192]]);
    let audio = synth_waveform(16_000, 2.0, 180.0);
    let pipeline_cfg = PipelineConfig {
        vad_threshold: -1.0,
        enable_auto_learn: false,
        ..PipelineConfig::default()
    };
    let gate_cfg = GateConfig::default();
    let mut group = c.benchmark_group("pipeline");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Elements(audio.len() as u64));
    group.bench_function("2s_16khz_offline", |b| {
        b.iter(|| {
            let result = process_offline(
                black_box(&audio),
                &mut pool,
                &pipeline_cfg,
                &gate_cfg,
                &mut components,
            )
            .expect("pipeline");
            black_box(result);
        });
    });
    let pipeline_cfg_async = PipelineConfig {
        async_refresh: true,
        ..pipeline_cfg
    };
    group.bench_function("2s_16khz_offline_async", |b| {
        b.iter(|| {
            let result = process_offline(
                black_box(&audio),
                &mut pool,
                &pipeline_cfg_async,
                &gate_cfg,
                &mut components,
            )
            .expect("pipeline");
            black_box(result);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_f0_track,
    bench_fbank,
    bench_vad,
    bench_ecapa,
    bench_dfn3,
    bench_pipeline
);
criterion_main!(benches);
