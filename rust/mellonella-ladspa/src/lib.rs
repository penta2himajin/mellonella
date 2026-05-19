//! Mellonella LADSPA plugin — target-speaker hard-gating + DFN3 noise
//! suppression exposed as a LADSPA effect, so the same engine that
//! drives `mellonella-gui` can be inserted as a virtual-mic filter
//! chain in PipeWire / JACK / Audacity etc.
//!
//! # Configuration
//!
//! All non-numeric configuration arrives out-of-band — LADSPA control
//! ports can only carry `f32` values.
//!
//! * **Enrollment**: loaded from `$MELLONELLA_ENROLLMENT` if set,
//!   otherwise from `<config_dir>/mellonella/enrollment.json` (the
//!   path `mellonella-gui` auto-saves to). When neither is present
//!   the plugin runs in DFN3-noise-suppression-only mode and logs a
//!   warning — the speaker gate stays fully open until a profile is
//!   provided.
//! * **ONNX models**: resolved through
//!   `mellonella_core::hf_fetch::ensure_*_onnx`, i.e. env-var → cache
//!   → first-run HuggingFace download. The download happens once in
//!   `activate()` so `run()` stays allocation-free.
//!
//! # Sample-rate policy
//!
//! 48 kHz only. DeepFilterNet 3 is 48 kHz-native, so we refuse any
//! other host rate at `instantiate()` time rather than resampling
//! (the audio server's existing graph-edge resampler does a better
//! job than anything we could add inline, with no additional
//! latency).

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
use tympan_ladspa::port::{PortDefault, PortDescriptor, Ports};
use tympan_ladspa::realtime::RealtimeContext;
use tympan_ladspa::{plugin_entry, InstantiateError, Plugin};

const REQUIRED_SAMPLE_RATE: u32 = 48_000;
const DECISION_SAMPLE_RATE: u32 = 16_000;

const PORT_IN: usize = 0;
const PORT_OUT: usize = 1;
const PORT_BYPASS: usize = 2;
const PORT_DRY_WET: usize = 3;
const PORT_VAD_THRESHOLD: usize = 4;
const PORT_GATE_OUT: usize = 5;
const PORT_SCORE_OUT: usize = 6;

pub struct MellonellaLadspa {
    sample_rate: u32,
    pipeline: Option<StreamingPipeline>,
    has_pool: bool,
    /// `true` when activate() loaded an empty pool — i.e. this
    /// session was running in DFN3-only + bootstrap-auto-learn mode.
    /// Tells deactivate() whether persisting the pool back to disk
    /// is meaningful: skip when an explicit enrollment was loaded so
    /// we never clobber a user-curated profile with runtime drift.
    started_empty: bool,
    /// Path used by activate() to load the pool (and by deactivate()
    /// to write it back when bootstrap admitted anything).
    enrollment_path: Option<PathBuf>,
    input_scratch: Vec<f32>,
    output_ring: VecDeque<f32>,
    last_gate: f32,
    last_score: f32,
    last_vad_threshold: f32,
}

impl Plugin for MellonellaLadspa {
    // Random ID in the LADSPA experimental range; replace with a
    // ladspa.org-assigned ID before any wide distribution.
    const UNIQUE_ID: u32 = 7_437_392;
    const LABEL: &'static str = "mellonella_gate";
    const NAME: &'static str = "Mellonella Target-Speaker Gate";
    const MAKER: &'static str = "mellonella contributors";
    const COPYRIGHT: &'static str = "Apache-2.0";

    fn ports() -> &'static [PortDescriptor] {
        static PORTS: &[PortDescriptor] = &[
            PortDescriptor::audio_input("In"),
            PortDescriptor::audio_output("Out"),
            PortDescriptor::control_input("Bypass")
                .with_default(PortDefault::Zero)
                .with_bounds(0.0, 1.0)
                .toggled(),
            PortDescriptor::control_input("Dry/Wet")
                .with_default(PortDefault::One)
                .with_bounds(0.0, 1.0),
            PortDescriptor::control_input("VAD Threshold")
                .with_default(PortDefault::Middle)
                .with_bounds(0.0, 1.0),
            PortDescriptor::control_output("Gate").with_bounds(0.0, 1.0),
            PortDescriptor::control_output("Score").with_bounds(0.0, 1.0),
        ];
        PORTS
    }

    fn instantiate(sample_rate: u32) -> Result<Self, InstantiateError> {
        if sample_rate != REQUIRED_SAMPLE_RATE {
            return Err(InstantiateError::SampleRateUnsupported(sample_rate));
        }
        Ok(Self {
            sample_rate,
            pipeline: None,
            has_pool: false,
            started_empty: false,
            enrollment_path: None,
            input_scratch: Vec::with_capacity(8192),
            output_ring: VecDeque::with_capacity(8192),
            last_gate: 0.0,
            last_score: 0.0,
            last_vad_threshold: f32::NAN,
        })
    }

    fn activate(&mut self) {
        // Heavy init lives here, not in run(): ONNX session creation,
        // optional HuggingFace downloads, enrollment JSON parse.
        let path = resolve_enrollment_path();
        let result = build_pipeline(self.sample_rate, path.as_deref());
        self.enrollment_path = path;
        match result {
            Ok((pipe, has_pool)) => {
                eprintln!(
                    "mellonella-ladspa: activated at {} Hz (enrollment loaded: {has_pool})",
                    self.sample_rate,
                );
                if !has_pool {
                    eprintln!(
                        "mellonella-ladspa: no enrollment.json found — running in DFN3 \
                         noise-suppression-only mode with bootstrap auto-learn enabled. \
                         The first anchor seeds itself from the first second of voiced \
                         speech; subsequent refreshes accumulate, and the pool persists \
                         on deactivate so the next session reuses the learned profile."
                    );
                }
                self.pipeline = Some(pipe);
                self.has_pool = has_pool;
                self.started_empty = !has_pool;
            }
            Err(e) => {
                eprintln!("mellonella-ladspa: activate failed ({e}); plugin will pass audio through unchanged");
                self.pipeline = None;
                self.has_pool = false;
                self.started_empty = false;
            }
        }
        self.output_ring.clear();
        self.last_gate = 0.0;
        self.last_score = 0.0;
        self.last_vad_threshold = f32::NAN;
    }

    fn deactivate(&mut self) {
        // Persist the bootstrapped pool back to disk before dropping
        // the pipeline. Only fires when this session started empty
        // (`started_empty`) so an explicit enrollment is never
        // overwritten by runtime drift, and only when the pool
        // actually accumulated anchors during the session.
        if self.started_empty {
            if let (Some(pipe), Some(path)) =
                (self.pipeline.as_ref(), self.enrollment_path.as_ref())
            {
                let pool = pipe.pool();
                if !pool.is_empty() {
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match pool.save(path) {
                        Ok(()) => eprintln!(
                            "mellonella-ladspa: persisted bootstrapped pool ({} anchors) to {}",
                            pool.anchors().len(),
                            path.display()
                        ),
                        Err(e) => eprintln!(
                            "mellonella-ladspa: failed to persist pool to {}: {e}",
                            path.display()
                        ),
                    }
                }
            }
        }

        self.pipeline = None;
        self.input_scratch.clear();
        self.output_ring.clear();
        self.started_empty = false;
        self.enrollment_path = None;
    }

    fn run(&mut self, _rt: &RealtimeContext, frames: usize, ports: &mut Ports<'_>) {
        let bypass = ports.control_input(PORT_BYPASS) >= 0.5;
        let dry_wet = ports.control_input(PORT_DRY_WET).clamp(0.0, 1.0);
        let vad_threshold = ports.control_input(PORT_VAD_THRESHOLD).clamp(0.0, 1.0);

        if (vad_threshold - self.last_vad_threshold).abs() > f32::EPSILON {
            self.last_vad_threshold = vad_threshold;
            // Live VAD threshold rewiring isn't supported by
            // StreamingPipeline yet — surface the request via stderr
            // so users can confirm the port is hooked up while we
            // add a setter upstream.
            //
            // TODO: expose `set_vad_threshold` on StreamingPipeline
            // so the control port becomes live.
        }

        let gate_out;
        let score_out;
        {
            let (input, output) = ports.audio_in_out(PORT_IN, PORT_OUT);
            let n = frames.min(input.len()).min(output.len());

            let pipe_opt = if bypass { None } else { self.pipeline.as_mut() };
            if let Some(pipe) = pipe_opt {
                self.input_scratch.clear();
                self.input_scratch.extend_from_slice(&input[..n]);
                match pipe.push_samples(&self.input_scratch) {
                    Ok(out) => {
                        for &s in &out.audio {
                            self.output_ring.push_back(s);
                        }
                        for &(_idx, is_on) in &out.gate_decisions {
                            self.last_gate = if is_on { 1.0 } else { 0.0 };
                        }
                    }
                    Err(e) => {
                        eprintln!("mellonella-ladspa: pipeline run error: {e}");
                    }
                }

                // Drain whatever the pipeline produced; if the
                // engine hasn't filled enough yet (warm-up frame
                // boundary), fall back to the dry signal so we
                // don't insert audible drop-outs.
                for i in 0..n {
                    let dry = input[i];
                    let wet = self.output_ring.pop_front().unwrap_or(dry);
                    output[i] = wet * dry_wet + dry * (1.0 - dry_wet);
                }
                gate_out = self.last_gate;
                score_out = self.last_score;
            } else {
                output[..n].copy_from_slice(&input[..n]);
                gate_out = if bypass { 1.0 } else { 0.0 };
                score_out = self.last_score;
            }
        }

        *ports.control_output(PORT_GATE_OUT) = gate_out;
        *ports.control_output(PORT_SCORE_OUT) = score_out;
    }
}

plugin_entry!(MellonellaLadspa);

fn build_pipeline(
    sample_rate: u32,
    enrollment_path: Option<&std::path::Path>,
) -> Result<(StreamingPipeline, bool), String> {
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

    let loaded = enrollment_path.and_then(|p| {
        if !p.exists() {
            return None;
        }
        // Use a default pool config for an existing enrollment — the
        // user's explicit anchors get the standard adapt residual /
        // drift threshold, not the bootstrap-relaxed variant.
        match EmbeddingPool::load(p, EmbeddingPoolConfig::default()) {
            Ok(pool) => {
                eprintln!(
                    "mellonella-ladspa: loaded enrollment from {} ({} anchors)",
                    p.display(),
                    pool.anchors().len()
                );
                Some(pool)
            }
            Err(e) => {
                eprintln!(
                    "mellonella-ladspa: failed to parse {} ({e}); starting empty",
                    p.display()
                );
                None
            }
        }
    });

    let (pool, has_pool) = match loaded {
        Some(p) if !p.is_empty() => (p, true),
        _ => {
            // Empty-pool mode: enable bootstrap so the first
            // VAD-positive voiced refresh seeds the initial anchor;
            // normal `adapt` then accumulates on top.
            let pool = EmbeddingPool::new(EmbeddingPoolConfig {
                allow_bootstrap_from_runtime: true,
                ..EmbeddingPoolConfig::default()
            });
            (pool, false)
        }
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
        // NS-only fallback while the pool is empty: keep the score-side
        // gate fully open so the user can be heard from the start.
        // `enable_auto_learn` stays on (its default) so the bootstrap
        // path in `apply_refresh_result` can seed the first anchor; once
        // an anchor exists the normal `adapt` admissions accumulate on
        // top. Gate stays open all session — next session loads the
        // bootstrapped anchors and runs the full gate.
        gate_cfg.theta_pass = f32::MIN;
        gate_cfg.theta_pass_as_norm = f32::MIN;
        gate_cfg.adaptive_theta = false;
        gate_cfg.hangover_ms = 10_000.0;
        pipeline_cfg.silence_force_off_ms = 0.0;
        pipeline_cfg.vad_threshold = -1.0;
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
        let p = PathBuf::from(raw);
        return Some(p);
    }
    let dir = dirs::config_dir()?.join("mellonella");
    Some(dir.join("enrollment.json"))
}
