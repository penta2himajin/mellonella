//! Mellonella desktop CLI — Phase 3 step 14.
//!
//! Subcommands:
//!
//! * `enroll <input.wav> <enrollment.json>` — build an EmbeddingPool
//!   from a clean recording
//! * `process <input.wav> <enrollment.json> <output.wav>` — apply the
//!   offline gating pipeline
//! * `info` — print resolved config + ONNX model paths
//!
//! ONNX model paths are read from environment variables:
//!
//! * `MELLONELLA_ECAPA_ONNX` — path to the embedding-only ECAPA model
//!   produced by `scripts/export_ecapa_onnx.py --mode embedding-only`
//! * `MELLONELLA_VAD_ONNX`   — path to `silero_vad.onnx`
//! * `ORT_DYLIB_PATH`        — path to libonnxruntime (used by ort's
//!   `load-dynamic` feature)
//!
//! Audio I/O accepts 16-bit signed-integer mono WAVs at any common
//! sample rate; non-16 kHz inputs are resampled to 16 kHz via
//! `mellonella_core::resample` before being fed into the pipeline.
//! 24-bit/32-bit/float WAVs are still rejected upfront — the SV
//! pipeline operates on `f32` peak-normalised samples and re-coding
//! those extra depths isn't worth the complexity today.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_value
)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::pipeline::{
    enroll_from_recording, process_offline, PipelineComponents, PipelineConfig,
};
use mellonella_core::resample::resample_to;
use mellonella_core::vad::SileroVad;

const REQUIRED_SAMPLE_RATE: u32 = 16_000;

#[derive(Parser, Debug)]
#[command(
    name = "mellonella",
    about = "Single-target speaker hard-gating filter — desktop CLI",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Build an EmbeddingPool from a clean enrollment recording.
    Enroll(EnrollArgs),
    /// Apply the offline pipeline to a recording.
    Process(ProcessArgs),
    /// Print resolved configuration + ONNX model paths.
    Info,
}

#[derive(Args, Debug)]
struct EnrollArgs {
    /// Input WAV (16-bit signed, mono; any common sample rate — resampled
    /// to 16 kHz on the way in).
    input: PathBuf,
    /// Output enrollment JSON.
    output: PathBuf,
}

#[derive(Args, Debug)]
struct ProcessArgs {
    /// Input WAV (16-bit signed, mono; any common sample rate — resampled
    /// to 16 kHz on the way in).
    input: PathBuf,
    /// Enrollment JSON produced by `mellonella enroll`.
    enrollment: PathBuf,
    /// Output WAV (will be 16-bit, 16 kHz, mono).
    output: PathBuf,
}

#[derive(Debug)]
enum CliError {
    Io(std::io::Error),
    Wav(hound::Error),
    UnsupportedFormat(String),
    MissingEnv(&'static str),
    Pipeline(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Wav(e) => write!(f, "WAV error: {e}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported WAV format: {msg}"),
            Self::MissingEnv(var) => write!(f, "missing required env var: {var}"),
            Self::Pipeline(msg) => write!(f, "pipeline error: {msg}"),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<hound::Error> for CliError {
    fn from(e: hound::Error) -> Self {
        Self::Wav(e)
    }
}

fn read_wav_to_16k_mono(path: &Path) -> Result<Vec<f32>, CliError> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(CliError::UnsupportedFormat(format!(
            "expected mono, got {} channels",
            spec.channels
        )));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(CliError::UnsupportedFormat(format!(
            "expected 16-bit signed int, got {:?} {}-bit",
            spec.sample_format, spec.bits_per_sample
        )));
    }
    let scale = 1.0_f32 / f32::from(i16::MAX);
    let samples: Result<Vec<f32>, _> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) * scale))
        .collect();
    let samples = samples?;
    if spec.sample_rate == REQUIRED_SAMPLE_RATE {
        return Ok(samples);
    }
    let resampled = resample_to(&samples, spec.sample_rate, REQUIRED_SAMPLE_RATE)
        .map_err(|e| CliError::Pipeline(format!("resample {}→16000 Hz: {e}", spec.sample_rate)))?;
    eprintln!(
        "[info] resampled {} Hz → {} Hz ({} → {} samples)",
        spec.sample_rate,
        REQUIRED_SAMPLE_RATE,
        samples.len(),
        resampled.len()
    );
    Ok(resampled)
}

fn write_wav_16k_mono(path: &Path, audio: &[f32]) -> Result<(), CliError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: REQUIRED_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    let max = f32::from(i16::MAX);
    for &s in audio {
        let clipped = (s.clamp(-1.0, 1.0) * max) as i16;
        writer.write_sample(clipped)?;
    }
    writer.finalize()?;
    Ok(())
}

fn env_path(var: &'static str) -> Result<PathBuf, CliError> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .ok_or(CliError::MissingEnv(var))
}

fn build_components() -> Result<PipelineComponents, CliError> {
    let ecapa_path = env_path("MELLONELLA_ECAPA_ONNX")?;
    let vad_path = env_path("MELLONELLA_VAD_ONNX")?;
    let fbank = Fbank::with_speechbrain_filterbank()
        .map_err(|e| CliError::Pipeline(format!("Fbank init: {e}")))?;
    let ecapa = EcapaTdnn::from_onnx_path(&ecapa_path)
        .map_err(|e| CliError::Pipeline(format!("ECAPA load: {e}")))?;
    let vad = SileroVad::from_onnx_path(&vad_path, REQUIRED_SAMPLE_RATE)
        .map_err(|e| CliError::Pipeline(format!("VAD load: {e}")))?;
    Ok(PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort: Vec::new(),
    })
}

fn cmd_enroll(args: EnrollArgs) -> Result<(), CliError> {
    let audio = read_wav_to_16k_mono(&args.input)?;
    let mut components = build_components()?;
    let pool = enroll_from_recording(&audio, &mut components, EmbeddingPoolConfig::default())
        .map_err(|e| CliError::Pipeline(format!("enroll: {e}")))?;
    pool.save(&args.output)
        .map_err(|e| CliError::Pipeline(format!("save enrollment: {e}")))?;
    let m = pool.metadata();
    println!(
        "wrote {}: {} anchors, f0_mu={:.1} Hz, f0_sigma={:.1} Hz",
        args.output.display(),
        pool.anchors().len(),
        m.f0_mu,
        m.f0_sigma,
    );
    Ok(())
}

fn cmd_process(args: ProcessArgs) -> Result<(), CliError> {
    let audio = read_wav_to_16k_mono(&args.input)?;
    let mut pool = EmbeddingPool::load(&args.enrollment, EmbeddingPoolConfig::default())
        .map_err(|e| CliError::Pipeline(format!("load enrollment: {e}")))?;
    let mut components = build_components()?;
    let cfg = PipelineConfig::default();
    let gate = GateConfig::default();
    let result = process_offline(&audio, &mut pool, &cfg, &gate, &mut components)
        .map_err(|e| CliError::Pipeline(format!("process: {e}")))?;
    write_wav_16k_mono(&args.output, &result.audio)?;
    let on_frames = result.gate_per_frame.iter().filter(|&&v| v).count();
    let total = result.gate_per_frame.len().max(1);
    println!(
        "wrote {} ({:.2}s @ {} Hz, gate duty cycle {}%)",
        args.output.display(),
        result.audio.len() as f32 / REQUIRED_SAMPLE_RATE as f32,
        REQUIRED_SAMPLE_RATE,
        on_frames * 100 / total,
    );
    Ok(())
}

fn cmd_info() {
    let ecapa = std::env::var("MELLONELLA_ECAPA_ONNX").unwrap_or_else(|_| "<unset>".into());
    let vad = std::env::var("MELLONELLA_VAD_ONNX").unwrap_or_else(|_| "<unset>".into());
    let dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into());
    println!("mellonella-cli");
    println!("  required sample rate : {REQUIRED_SAMPLE_RATE} Hz, mono, 16-bit signed");
    println!("  MELLONELLA_ECAPA_ONNX = {ecapa}");
    println!("  MELLONELLA_VAD_ONNX   = {vad}");
    println!("  ORT_DYLIB_PATH        = {dylib}");
    let cfg = PipelineConfig::default();
    let gate = GateConfig::default();
    println!(
        "  pipeline: sv_window={}, sv_update={}, vad_threshold={}, theta_pass={}, hangover_ms={}, attack_ms={}, release_ms={}",
        cfg.sv_window_samples,
        cfg.sv_update_samples,
        cfg.vad_threshold,
        gate.theta_pass,
        gate.hangover_ms,
        gate.attack_ms,
        gate.release_ms,
    );
}

fn run() -> Result<(), CliError> {
    match Cli::parse().cmd {
        Cmd::Enroll(args) => cmd_enroll(args),
        Cmd::Process(args) => cmd_process(args),
        Cmd::Info => {
            cmd_info();
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
