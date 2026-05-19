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
//! sample rate.
//!
//! * **Enroll** resamples the input to 16 kHz (ECAPA-TDNN's native
//!   training rate) before feeding ECAPA.
//! * **Process** resamples the input to **48 kHz** (DFN3's native
//!   rate, full-band quality) for the audio path and downsamples
//!   internally to 16 kHz for the decision path (VAD / ECAPA / F0).
//!   Output is always written at 48 kHz mono — see `Sampling-rate
//!   policy` in `docs/architecture.md`.
//!
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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use mellonella_audio_io::{
    list_input_devices, list_output_devices, LiveSession, SessionConfig, SessionEvent,
};
use mellonella_core::dfn3::{Dfn3Pipeline, DFN3_SR, SAMPLES_PER_CHUNK};
use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::hf_fetch::{fetch_tse_prod_48k, FetchError, TSE_PROD_48K_REPO};
use mellonella_core::pipeline::{
    enroll_from_recording, process_offline, PipelineComponents, PipelineConfig,
};
use mellonella_core::resample::resample_to;
use mellonella_core::streaming::StreamingConfig;
use mellonella_core::tse_stage::TseStageConfig;
use mellonella_core::vad::SileroVad;

/// Internal decision rate (ECAPA-TDNN's training rate). The pipeline
/// downsamples to this rate for VAD / ECAPA / F0 only — the audio
/// path stays at [`OUTPUT_SAMPLE_RATE`].
const DECISION_SAMPLE_RATE: u32 = 16_000;

/// Output audio rate (DFN3's native rate; what `docs/architecture.md`
/// promises for the filtered output). Driven by the "decisions at
/// 16 kHz, audio at 48 kHz" split — see the Sampling-rate policy
/// section of that doc.
const OUTPUT_SAMPLE_RATE: u32 = 48_000;

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
    /// List input / output audio devices (cpal default host).
    Devices,
    /// Run the live filter end-to-end: mic → pipeline → speaker.
    Live(LiveArgs),
    /// Print resolved configuration + ONNX model paths.
    Info,
}

#[derive(Args, Debug)]
struct LiveArgs {
    /// Enrollment JSON produced by `mellonella enroll`.
    enrollment: PathBuf,
    /// Input device name (see `mellonella devices`). Defaults to the
    /// host default input.
    #[arg(long)]
    input_device: Option<String>,
    /// Output device name. Defaults to the host default output.
    #[arg(long)]
    output_device: Option<String>,
    /// Run DFN3 noise suppression on the live audio path. Requires
    /// `MELLONELLA_DFN3_ONNX` (the stateful per-frame export
    /// produced by `scripts/export_dfn3_onnx.py`). Adds ~30 ms of
    /// algorithmic latency (2-frame `conv_lookahead` + ~10 ms model
    /// time) on top of the no-NS path.
    #[arg(long)]
    enable_dfn3: bool,
    /// Multi-channel input downmix strategy. `average` (default)
    /// folds all channels together; `0` / `1` / `n` picks a
    /// specific channel (0-indexed) for setups like a podcast
    /// interface where one channel is the target signal.
    #[arg(long, default_value = "average")]
    input_channel: InputChannelArg,
    /// Stage B low-latency profile: opt into the model-free
    /// speaker-turn latency reduction (per-frame fast F0 cue + anchor
    /// fusion, adaptive window / turn detection, offset fail-closed).
    /// Off by default — with it off the pipeline is byte-identical to
    /// the pre-Stage-B behaviour.
    #[arg(long)]
    low_latency: bool,
}

#[derive(Clone, Debug)]
enum InputChannelArg {
    Average,
    Channel(u16),
}

impl std::str::FromStr for InputChannelArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        if s == "average" || s == "avg" || s == "mean" {
            return Ok(Self::Average);
        }
        s.parse::<u16>()
            .map(Self::Channel)
            .map_err(|_| format!("expected 'average' or a non-negative channel index, got {s:?}"))
    }
}

impl From<InputChannelArg> for mellonella_audio_io::ChannelStrategy {
    fn from(v: InputChannelArg) -> Self {
        match v {
            InputChannelArg::Average => Self::Average,
            InputChannelArg::Channel(n) => Self::Channel(n),
        }
    }
}

#[derive(Args, Debug)]
struct EnrollArgs {
    /// Input WAV (16-bit signed, mono; any common sample rate — resampled
    /// to 16 kHz on the way in).
    input: PathBuf,
    /// Output enrollment JSON.
    output: PathBuf,
    /// Run DFN3 noise suppression on the input before enrolling. Requires
    /// `MELLONELLA_DFN3_ONNX`. Input is processed at 48 kHz one chunk
    /// (≤ 1.02 s) at a time, then resampled to 16 kHz for ECAPA.
    #[arg(long)]
    enable_dfn3: bool,
}

#[derive(Args, Debug)]
struct ProcessArgs {
    /// Input WAV (16-bit signed, mono; any common sample rate —
    /// resampled to 48 kHz on the way in for the audio path).
    input: PathBuf,
    /// Enrollment JSON produced by `mellonella enroll`.
    enrollment: PathBuf,
    /// Output WAV (will be 16-bit, 48 kHz, mono).
    output: PathBuf,
    /// Run DFN3 noise suppression on the input before gating. See
    /// `EnrollArgs::enable_dfn3`.
    #[arg(long)]
    enable_dfn3: bool,
    /// Optional path to dump per-VAD-frame diagnostics as JSON
    /// (`gate_per_frame`, `score_per_frame`, `cos_sim_max_per_frame`,
    /// `f0_match_per_frame`, gate `decisions` run-length, plus the
    /// auto-learn event log). Used by the bench-side scenario runners
    /// to compare Rust output against the Python reference at the same
    /// granularity.
    #[arg(long)]
    gate_decisions: Option<PathBuf>,
    /// Stage B low-latency profile: opt into the model-free
    /// speaker-turn latency reduction (per-frame fast F0 cue + anchor
    /// fusion, adaptive window / turn detection, offset fail-closed).
    /// Off by default — with it off `process` is byte-identical to
    /// the pre-Stage-B behaviour, so `ci_baseline_rust.json` is
    /// unaffected. `bench/.../ci_accuracy.py --engine rust` passes
    /// `--low-latency` to exercise the new path.
    #[arg(long)]
    low_latency: bool,
    /// Path to a streaming TSE ONNX (Stage C). When set, the offline
    /// pipeline conditions on the enrolled anchor centroid and
    /// extracts only the target speaker's voice before applying the
    /// gate envelope. The companion `--tse-config` flag selects the
    /// model variant (default `prod_48k` — matches the canonical
    /// release at <https://huggingface.co/penta2himajin/tse-conv-tasnet-48k>).
    /// The audio path stays at 48 kHz end-to-end for `prod_48k`;
    /// `poc_16k` requires both the audio path and decision path to be
    /// at 16 kHz, so combining `poc_16k` with the default 48 kHz audio
    /// path will fail (resample input or pick `prod_48k`).
    #[arg(long, value_name = "PATH")]
    tse_onnx: Option<PathBuf>,
    /// TSE model variant. `prod_48k` (default) matches the production
    /// 48 kHz model on HuggingFace; `poc_16k` matches the original
    /// 16 kHz PoC.
    #[arg(long, value_enum, default_value_t = TseVariant::Prod48k)]
    tse_config: TseVariant,
    /// Download the canonical Stage C TSE model from HuggingFace
    /// (penta2himajin/tse-conv-tasnet-48k) into the local cache and
    /// use it. Mutually exclusive with `--tse-onnx`; implies
    /// `--tse-config prod_48k`. The cached file is reused on
    /// subsequent runs.
    #[arg(long, conflicts_with = "tse_onnx")]
    tse_from_hf: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum TseVariant {
    #[value(name = "poc_16k")]
    Poc16k,
    #[value(name = "prod_48k")]
    Prod48k,
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

/// Run DFN3 chunk-by-chunk over `audio_48k`, concatenating each chunk's
/// output. Audio shorter than one chunk gets a single chunk zero-pad;
/// longer audio is split into back-to-back chunks without overlap (the
/// streaming-overlap model lands in a later PR).
fn apply_dfn3(audio_48k: &[f32]) -> Result<Vec<f32>, CliError> {
    let onnx_path = env_path("MELLONELLA_DFN3_ONNX")?;
    let mut pipeline = Dfn3Pipeline::from_onnx_path(&onnx_path)
        .map_err(|e| CliError::Pipeline(format!("DFN3 load: {e}")))?;
    let mut enhanced = Vec::with_capacity(audio_48k.len());
    let mut start = 0_usize;
    while start < audio_48k.len() {
        let end = (start + SAMPLES_PER_CHUNK).min(audio_48k.len());
        let chunk = &audio_48k[start..end];
        let out = pipeline
            .process(chunk)
            .map_err(|e| CliError::Pipeline(format!("DFN3 process: {e}")))?;
        enhanced.extend_from_slice(&out[..end - start]);
        start = end;
    }
    if audio_48k.is_empty() {
        return Ok(Vec::new());
    }
    eprintln!(
        "[info] DFN3 noise suppression applied ({} samples @ 48 kHz)",
        enhanced.len()
    );
    Ok(enhanced)
}

/// Decode a 16-bit signed mono WAV into `f32` samples in `[-1, 1]`
/// and return `(samples, sample_rate)` at the file's native rate.
fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32), CliError> {
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
    Ok((samples?, spec.sample_rate))
}

/// Resample `samples` to `target_sr` if it isn't already there,
/// logging the transition for diagnostic noise.
fn resample_with_log(
    samples: &[f32],
    current_sr: u32,
    target_sr: u32,
    purpose: &str,
) -> Result<Vec<f32>, CliError> {
    if current_sr == target_sr {
        return Ok(samples.to_vec());
    }
    let out = resample_to(samples, current_sr, target_sr).map_err(|e| {
        CliError::Pipeline(format!(
            "resample {current_sr}→{target_sr} Hz ({purpose}): {e}"
        ))
    })?;
    eprintln!(
        "[info] resampled {} Hz → {} Hz for {purpose} ({} → {} samples)",
        current_sr,
        target_sr,
        samples.len(),
        out.len()
    );
    Ok(out)
}

/// Prepare audio for `enroll`: 16 kHz mono, optionally DFN3-cleaned
/// at 48 kHz first.
fn prepare_audio_for_enroll(path: &Path, enable_dfn3: bool) -> Result<Vec<f32>, CliError> {
    let (mut samples, mut current_sr) = read_wav_mono(path)?;
    if enable_dfn3 {
        samples = resample_with_log(&samples, current_sr, DFN3_SR as u32, "DFN3 input")?;
        current_sr = DFN3_SR as u32;
        samples = apply_dfn3(&samples)?;
    }
    resample_with_log(
        &samples,
        current_sr,
        DECISION_SAMPLE_RATE,
        "ECAPA enrollment",
    )
}

/// Prepare audio for `process`: 48 kHz mono (output rate),
/// optionally DFN3-cleaned. `process_offline` downsamples internally
/// for the decision path.
fn prepare_audio_for_process(path: &Path, enable_dfn3: bool) -> Result<Vec<f32>, CliError> {
    let (mut samples, mut current_sr) = read_wav_mono(path)?;
    samples = resample_with_log(&samples, current_sr, OUTPUT_SAMPLE_RATE, "audio path")?;
    current_sr = OUTPUT_SAMPLE_RATE;
    if enable_dfn3 {
        // DFN3's native rate is 48 kHz, which already matches the
        // audio path — no extra resample.
        debug_assert_eq!(current_sr, DFN3_SR as u32);
        samples = apply_dfn3(&samples)?;
    }
    Ok(samples)
}

fn write_wav_mono(path: &Path, audio: &[f32], sample_rate: u32) -> Result<(), CliError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
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
    let vad = SileroVad::from_onnx_path(&vad_path, DECISION_SAMPLE_RATE)
        .map_err(|e| CliError::Pipeline(format!("VAD load: {e}")))?;
    Ok(PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort: Vec::new(),
        tse: None,
    })
}

fn cmd_enroll(args: EnrollArgs) -> Result<(), CliError> {
    let audio = prepare_audio_for_enroll(&args.input, args.enable_dfn3)?;
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

/// Apply the Stage B low-latency profile to `cfg` when `enabled`.
/// Flips all three opt-in knobs (`fast_cue_enabled`,
/// `turn_detect_enabled`, `offset_fail_closed`) on at once; the
/// per-knob defaults (`fast_cue_weight`, `turn_drop_delta`, the turn
/// window / cadence) come from `PipelineConfig::default`. A no-op
/// when `enabled` is false, so the default pipeline is unchanged.
fn apply_low_latency_profile(cfg: &mut PipelineConfig, enabled: bool) {
    if !enabled {
        return;
    }
    cfg.fast_cue_enabled = true;
    cfg.turn_detect_enabled = true;
    cfg.offset_fail_closed = true;
}

/// Download the canonical Stage C prod_48k ONNX (and its `.data`
/// sidecar) from HuggingFace into the local cache and return the
/// `.onnx` path. Reuses a cached copy when present.
fn fetch_tse_from_hf() -> Result<PathBuf, CliError> {
    eprintln!("[info] fetching {TSE_PROD_48K_REPO} from HuggingFace (cached on subsequent runs)…");
    let onnx = fetch_tse_prod_48k(|file, so_far, total| {
        let mb = so_far as f32 / 1.0e6;
        if let Some(t) = total {
            let pct = (so_far as f32 / t as f32) * 100.0;
            eprint!(
                "\r  {file}: {mb:>6.2} / {:>6.2} MB ({pct:5.1}%)",
                t as f32 / 1.0e6
            );
        } else {
            eprint!("\r  {file}: {mb:>6.2} MB");
        }
        if total.is_some_and(|t| so_far >= t) {
            eprintln!();
        }
    })
    .map_err(|e| match e {
        FetchError::Cache(io) | FetchError::Io(io) => CliError::Io(io),
        other => CliError::Pipeline(format!("HuggingFace fetch: {other}")),
    })?;
    eprintln!("[info] TSE model ready at {}", onnx.display());
    Ok(onnx)
}

fn build_tse_stage_config(onnx: &Path, variant: TseVariant) -> Result<TseStageConfig, CliError> {
    if !onnx.exists() {
        return Err(CliError::Pipeline(format!(
            "--tse-onnx points to a non-existent path: {}",
            onnx.display()
        )));
    }
    Ok(match variant {
        TseVariant::Poc16k => TseStageConfig::new(onnx.to_path_buf()),
        TseVariant::Prod48k => TseStageConfig::new_prod_48k(onnx.to_path_buf()),
    })
}

fn cmd_process(args: ProcessArgs) -> Result<(), CliError> {
    let audio = prepare_audio_for_process(&args.input, args.enable_dfn3)?;
    let mut pool = EmbeddingPool::load(&args.enrollment, EmbeddingPoolConfig::default())
        .map_err(|e| CliError::Pipeline(format!("load enrollment: {e}")))?;
    let mut components = build_components()?;
    let mut cfg = PipelineConfig::default();
    apply_low_latency_profile(&mut cfg, args.low_latency);
    if let Some(tse_onnx) = args.tse_onnx.as_deref() {
        cfg.tse = Some(build_tse_stage_config(tse_onnx, args.tse_config)?);
        eprintln!(
            "[info] Stage C TSE enabled: variant={:?}, onnx={}",
            args.tse_config,
            tse_onnx.display()
        );
    } else if args.tse_from_hf {
        let onnx = fetch_tse_from_hf()?;
        cfg.tse = Some(TseStageConfig::new_prod_48k(onnx.clone()));
        eprintln!(
            "[info] Stage C TSE enabled: variant=Prod48k, onnx={} (from HuggingFace)",
            onnx.display()
        );
    }
    let gate = GateConfig::default();
    let result = process_offline(
        &audio,
        OUTPUT_SAMPLE_RATE,
        &mut pool,
        &cfg,
        &gate,
        &mut components,
    )
    .map_err(|e| CliError::Pipeline(format!("process: {e}")))?;
    write_wav_mono(&args.output, &result.audio, OUTPUT_SAMPLE_RATE)?;
    let on_frames = result.gate_per_frame.iter().filter(|&&v| v).count();
    let total = result.gate_per_frame.len().max(1);
    println!(
        "wrote {} ({:.2}s @ {} Hz, gate duty cycle {}%)",
        args.output.display(),
        result.audio.len() as f32 / OUTPUT_SAMPLE_RATE as f32,
        OUTPUT_SAMPLE_RATE,
        on_frames * 100 / total,
    );
    if let Some(diag_path) = args.gate_decisions {
        let payload = serde_json::json!({
            "version": 1,
            "sample_rate": DECISION_SAMPLE_RATE,
            "audio_sample_rate": OUTPUT_SAMPLE_RATE,
            "vad_frame_samples": 512,
            "gate_per_frame": result.gate_per_frame,
            "score_per_frame": result.score_per_frame,
            "cos_sim_max_per_frame": result.cos_sim_max_per_frame,
            "f0_match_per_frame": result.f0_match_per_frame,
            "gate_decisions": result
                .gate_decisions
                .iter()
                .map(|(s, o)| serde_json::json!([s, o]))
                .collect::<Vec<_>>(),
            "auto_learn_events": result
                .auto_learn_events
                .iter()
                .map(|ev| serde_json::json!({
                    "frame_idx": ev.frame_idx,
                    "kind": format!("{:?}", ev.kind),
                    "score": ev.score,
                    "f0_match": ev.f0_match,
                }))
                .collect::<Vec<_>>(),
        });
        std::fs::write(&diag_path, serde_json::to_string_pretty(&payload).unwrap())
            .map_err(CliError::Io)?;
        eprintln!("[info] diagnostics → {}", diag_path.display());
    }
    Ok(())
}

fn cmd_devices() -> Result<(), CliError> {
    let inputs =
        list_input_devices().map_err(|e| CliError::Pipeline(format!("list input devices: {e}")))?;
    let outputs = list_output_devices()
        .map_err(|e| CliError::Pipeline(format!("list output devices: {e}")))?;
    println!("Input devices (* = default):");
    if inputs.is_empty() {
        println!("  (none)");
    }
    for d in &inputs {
        println!("{d}");
    }
    println!();
    println!("Output devices (* = default):");
    if outputs.is_empty() {
        println!("  (none)");
    }
    for d in &outputs {
        println!("{d}");
    }
    Ok(())
}

fn cmd_live(args: LiveArgs) -> Result<(), CliError> {
    let pool = EmbeddingPool::load(&args.enrollment, EmbeddingPoolConfig::default())
        .map_err(|e| CliError::Pipeline(format!("load enrollment: {e}")))?;
    let components = build_components()?;

    let dfn3_onnx_path = if args.enable_dfn3 {
        Some(env_path("MELLONELLA_DFN3_ONNX")?)
    } else {
        None
    };

    // Live mode opts into the silence force-off + score EMA smoothing
    // that the library defaults leave dormant. Mirrors the GUI's
    // `default_live_pipeline_cfg`: 400 ms VAD-silence hangover (AND-
    // combined with the score-side gate so close time is 400 ms +
    // release_ms, not stacked on hangover_ms), 0.7 EMA blend to ride
    // out one-refresh dips at speech onset, and a 100 ms post-silence
    // early-refresh trigger so `last_score` catches up quickly on
    // resume. See project-mellonella-first-smoke for the diagnosis.
    let mut live_pipeline_cfg = PipelineConfig {
        silence_force_off_ms: 400.0,
        score_ema_alpha: 0.7,
        sv_min_new_samples_after_silence: 1_600,
        ..PipelineConfig::default()
    };
    apply_low_latency_profile(&mut live_pipeline_cfg, args.low_latency);
    let session_cfg = SessionConfig {
        input_device: args.input_device,
        output_device: args.output_device,
        streaming: StreamingConfig {
            pipeline: live_pipeline_cfg,
            gate: GateConfig::default(),
            audio_sample_rate: OUTPUT_SAMPLE_RATE,
            diagnostics: false,
            dfn3_onnx_path: None,
        },
        dfn3_onnx_path,
        input_channel: args.input_channel.into(),
    };

    let session = LiveSession::new(pool, components, session_cfg)
        .map_err(|e| CliError::Pipeline(format!("open live session: {e}")))?;
    eprintln!("[live] running. Ctrl+C to stop.");

    let stop = Arc::new(AtomicBool::new(false));
    let stop_handler = stop.clone();
    ctrlc::set_handler(move || {
        stop_handler.store(true, Ordering::Relaxed);
    })
    .map_err(|e| CliError::Pipeline(format!("install ctrl-c handler: {e}")))?;

    let mut last_status = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if let Some(SessionEvent::Error(msg)) = session.try_recv_event() {
            eprintln!("[live] pipeline error: {msg}");
            break;
        }
        if last_status.elapsed() >= Duration::from_secs(2) {
            let s = session.stats_snapshot();
            let secs = s.samples_processed as f32 / OUTPUT_SAMPLE_RATE as f32;
            eprintln!(
                "[live] processed {:.1} s, overruns={} underruns={}",
                secs, s.input_overruns, s.output_underruns,
            );
            last_status = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let final_stats = session
        .stop()
        .map_err(|e| CliError::Pipeline(format!("stop live session: {e}")))?;
    let secs = final_stats.samples_processed as f32 / OUTPUT_SAMPLE_RATE as f32;
    eprintln!(
        "[live] stopped. processed {:.1} s, overruns={} underruns={}",
        secs, final_stats.input_overruns, final_stats.output_underruns,
    );
    Ok(())
}

fn cmd_info() {
    let ecapa = std::env::var("MELLONELLA_ECAPA_ONNX").unwrap_or_else(|_| "<unset>".into());
    let vad = std::env::var("MELLONELLA_VAD_ONNX").unwrap_or_else(|_| "<unset>".into());
    let dfn3 = std::env::var("MELLONELLA_DFN3_ONNX").unwrap_or_else(|_| "<unset>".into());
    let dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into());
    println!("mellonella-cli");
    println!("  output sample rate    : {OUTPUT_SAMPLE_RATE} Hz, mono, 16-bit signed");
    println!("  decision sample rate  : {DECISION_SAMPLE_RATE} Hz (internal, VAD/ECAPA/F0)");
    println!("  MELLONELLA_ECAPA_ONNX = {ecapa}");
    println!("  MELLONELLA_VAD_ONNX   = {vad}");
    println!("  MELLONELLA_DFN3_ONNX  = {dfn3}");
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
        Cmd::Devices => cmd_devices(),
        Cmd::Live(args) => cmd_live(args),
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
