//! Windows-only APO implementation. Mirrors the LADSPA plugin's
//! lifecycle:
//!
//! | LADSPA                  | APO                       |
//! |-------------------------|---------------------------|
//! | `instantiate()`         | `ProcessingObject::new()` |
//! | `activate()`            | `lock_for_process()`      |
//! | `run()`                 | `process()`               |
//! | `deactivate()`          | `unlock_for_process()`    |
//!
//! Heavy initialisation (ONNX session creation, optional HF
//! downloads, enrollment JSON load) lives in `lock_for_process` so
//! `process` stays allocation-free in steady state — matching
//! `tympan-apo`'s realtime contract.

use std::collections::VecDeque;
use std::path::PathBuf;

use mellonella_core::embedding::EcapaTdnn;
use mellonella_core::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use mellonella_core::features::Fbank;
use mellonella_core::gating::GateConfig;
use mellonella_core::hf_fetch;
use mellonella_core::pipeline::{PipelineComponents, PipelineConfig};
use mellonella_core::streaming::{StreamingConfig, StreamingPipeline};
use mellonella_core::vad::SileroVad;
use tympan_apo::realtime::RealtimeContext;
use tympan_apo::{
    ApoCategory, BufferFlags, Clsid, Format, FormatNegotiation, HResult, ProcessInput,
    ProcessingObject,
};

const REQUIRED_SAMPLE_RATE: u32 = 48_000;
const REQUIRED_CHANNELS: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const DECISION_SAMPLE_RATE: u32 = 16_000;

pub struct MellonellaApo {
    pipeline: Option<StreamingPipeline>,
    input_scratch: Vec<f32>,
    output_ring: VecDeque<f32>,
    locked: bool,
}

impl ProcessingObject for MellonellaApo {
    // Random Clsid in the experimental range. Regenerate (and bump
    // the registry .inf) before any wide distribution so it doesn't
    // collide with another vendor's APO.
    const CLSID: Clsid = Clsid::from_u128(0xA1B2_C3D4_E5F6_4789_8A1B_2C3D_4E5F_6071);
    const NAME: &'static str = "Mellonella Target-Speaker Gate";
    const COPYRIGHT: &'static str = "Apache-2.0";
    // Sfx = stream effect, per-capture-stream — the right slot for a
    // microphone NS/gate. Mfx (mode effect) would be applied per
    // routing-mode and Efx (endpoint effect) per device, neither of
    // which matches "user-specific voice profile".
    const CATEGORY: ApoCategory = ApoCategory::Sfx;

    fn new() -> Self {
        Self {
            pipeline: None,
            input_scratch: Vec::with_capacity(8192),
            output_ring: VecDeque::with_capacity(8192),
            locked: false,
        }
    }

    fn is_input_format_supported(&self, format: &Format) -> FormatNegotiation {
        accept_or_suggest(format)
    }

    fn is_output_format_supported(&self, format: &Format) -> FormatNegotiation {
        accept_or_suggest(format)
    }

    fn lock_for_process(&mut self, input: &Format, output: &Format) -> Result<(), HResult> {
        if !is_supported(input) || !is_supported(output) {
            // Engine should never reach here once is_*_format_supported
            // is honoured, but reject defensively rather than driving
            // ONNX at a mismatched rate.
            return Err(HResult::E_INVALIDARG);
        }

        match build_pipeline(input.sample_rate()) {
            Ok((pipe, has_pool)) => {
                eprintln!(
                    "mellonella-apo: locked at {} Hz (enrollment loaded: {has_pool})",
                    input.sample_rate(),
                );
                if !has_pool {
                    eprintln!(
                        "mellonella-apo: no enrollment.json found — running in DFN3 \
                         noise-suppression-only mode. Enroll via mellonella-gui to enable \
                         target-speaker gating."
                    );
                }
                self.pipeline = Some(pipe);
                self.output_ring.clear();
                self.locked = true;
                Ok(())
            }
            Err(e) => {
                eprintln!("mellonella-apo: lock_for_process failed: {e}");
                Err(HResult::E_FAIL)
            }
        }
    }

    fn unlock_for_process(&mut self) {
        // Drop the pipeline so the ONNX session + async worker
        // thread are released. lock_for_process rebuilds them next
        // time the engine wants the stream up.
        self.pipeline = None;
        self.input_scratch.clear();
        self.output_ring.clear();
        self.locked = false;
    }

    fn process(
        &mut self,
        _rt: &RealtimeContext,
        input: ProcessInput<'_>,
        output: &mut [f32],
    ) -> BufferFlags {
        let samples = input.samples();
        let n = samples.len().min(output.len());

        let Some(pipe) = self.pipeline.as_mut() else {
            // Lock failed or we somehow got called outside the
            // locked window — pass dry signal through rather than
            // emitting silence; surface a soft warning via flags so
            // the engine can mark the chunk silent if it really is.
            output[..n].copy_from_slice(&samples[..n]);
            return input.flags();
        };

        if input.is_silent() {
            // Skip the pipeline on silent buffers — push_samples
            // would still spin VAD/DFN3 for no useful output.
            // Forward the silence flag through.
            for s in &mut output[..n] {
                *s = 0.0;
            }
            return input.flags();
        }

        self.input_scratch.clear();
        self.input_scratch.extend_from_slice(&samples[..n]);
        match pipe.push_samples(&self.input_scratch) {
            Ok(out) => {
                for &s in &out.audio {
                    self.output_ring.push_back(s);
                }
            }
            Err(e) => {
                eprintln!("mellonella-apo: pipeline process error: {e}");
            }
        }

        for i in 0..n {
            output[i] = self.output_ring.pop_front().unwrap_or(samples[i]);
        }
        input.flags()
    }
}

fn is_supported(format: &Format) -> bool {
    format.sample_rate() == REQUIRED_SAMPLE_RATE
        && format.channels() == REQUIRED_CHANNELS
        && format.format_tag() == WAVE_FORMAT_IEEE_FLOAT
}

fn accept_or_suggest(format: &Format) -> FormatNegotiation {
    if is_supported(format) {
        FormatNegotiation::Accept
    } else {
        FormatNegotiation::Suggest(Format::pcm_float32(REQUIRED_SAMPLE_RATE, REQUIRED_CHANNELS))
    }
}

fn build_pipeline(sample_rate: u32) -> Result<(StreamingPipeline, bool), String> {
    let ecapa_path =
        hf_fetch::ensure_ecapa_onnx(noop_progress).map_err(|e| format!("ECAPA fetch: {e}"))?;
    let vad_path =
        hf_fetch::ensure_vad_onnx(noop_progress).map_err(|e| format!("VAD fetch: {e}"))?;
    let dfn3_path =
        hf_fetch::ensure_dfn3_onnx(noop_progress).map_err(|e| format!("DFN3 fetch: {e}"))?;

    let fbank = Fbank::with_speechbrain_filterbank().map_err(|e| format!("Fbank init: {e}"))?;
    let ecapa = EcapaTdnn::from_onnx_path(&ecapa_path).map_err(|e| format!("ECAPA load: {e}"))?;
    let vad = SileroVad::from_onnx_path(&vad_path, DECISION_SAMPLE_RATE)
        .map_err(|e| format!("VAD load: {e}"))?;

    let components = PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort: Vec::new(),
        tse: None,
    };

    let pool_cfg = EmbeddingPoolConfig::default();
    let (pool, has_pool) = match resolve_enrollment_path() {
        Some(path) if path.exists() => match EmbeddingPool::load(&path, pool_cfg) {
            Ok(p) => {
                let usable = !p.is_empty();
                eprintln!(
                    "mellonella-apo: loaded enrollment from {} ({} anchors)",
                    path.display(),
                    p.anchors().len()
                );
                (p, usable)
            }
            Err(e) => {
                eprintln!(
                    "mellonella-apo: failed to parse {} ({e}); starting empty",
                    path.display()
                );
                (EmbeddingPool::new(pool_cfg), false)
            }
        },
        _ => (EmbeddingPool::new(pool_cfg), false),
    };

    let mut pipeline_cfg = PipelineConfig {
        async_refresh: true,
        silence_force_off_ms: 400.0,
        score_ema_alpha: 0.7,
        sv_min_new_samples_after_silence: 1_600,
        ..PipelineConfig::default()
    };
    let mut gate_cfg = GateConfig::default();

    if !has_pool {
        // NS-only fallback (matches mellonella-ladspa exactly): keep
        // the speaker gate fully open + skip auto-learn so DFN3 still
        // runs and audio passes through cleaned but ungated.
        gate_cfg.theta_pass = f32::MIN;
        gate_cfg.theta_pass_as_norm = f32::MIN;
        gate_cfg.adaptive_theta = false;
        gate_cfg.hangover_ms = 10_000.0;
        pipeline_cfg.silence_force_off_ms = 0.0;
        pipeline_cfg.vad_threshold = -1.0;
        pipeline_cfg.enable_auto_learn = false;
    }

    let config = StreamingConfig {
        pipeline: pipeline_cfg,
        gate: gate_cfg,
        audio_sample_rate: sample_rate,
        diagnostics: false,
        dfn3_onnx_path: Some(dfn3_path),
    };

    let pipe = StreamingPipeline::new(pool, config, components)
        .map_err(|e| format!("StreamingPipeline build: {e}"))?;
    Ok((pipe, has_pool))
}

fn noop_progress(_done: u64, _total: Option<u64>) {}

fn resolve_enrollment_path() -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("MELLONELLA_ENROLLMENT") {
        return Some(PathBuf::from(raw));
    }
    let dir = dirs::config_dir()?.join("mellonella");
    Some(dir.join("enrollment.json"))
}
