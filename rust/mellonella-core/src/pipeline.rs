//! Offline orchestrator that wires VAD → Fbank → ECAPA → gating → envelope.
//!
//! Mirrors `mellonella_poc.pipeline.process_offline` minus the DFN3
//! noise-suppression stage (Phase 3 step TBD) and minus output-rate
//! resampling — the Rust pipeline operates entirely at the SV rate of
//! 16 kHz and assumes the caller has already cleaned and resampled the
//! input. The DFN3 stage will land alongside its own ONNX wrapper PR.
//!
//! Cadence (matches the Python PoC):
//!
//! * VAD frames are 512 samples (32 ms @ 16 kHz). One frame → one gate
//!   decision and one entry in `gate_per_frame`.
//! * The speech buffer accumulates frames whose `speech_prob > 0.5`,
//!   capped at `sv_window_samples` (1 s @ 16 kHz by default).
//! * Every `sv_update_samples` (4 000 = 250 ms by default), if the
//!   buffer is full, an embedding + F0 are recomputed and the gate
//!   score is updated.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use crate::embedding::{EcapaTdnn, EmbeddingError};
use crate::enrollment::EmbeddingPool;
use crate::f0::{estimate_f0_track, f0_statistics, DEFAULT_F_MAX, DEFAULT_F_MIN};
use crate::features::{Fbank, N_MELS};
use crate::gating::{
    apply_envelope, as_norm_score, cos_sim_max, f0_match, should_admit_auto_learn,
    ApplyEnvelopeError, GateConfig, GateState,
};
use crate::vad::{SileroVad, CHUNK_SAMPLES_16K};

/// SV-side cadence for the offline pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Sample rate of the input buffer. Currently fixed at 16 kHz so the
    /// component models match. Kept on the config struct so future
    /// non-SV-native rates can be flagged at construction time rather
    /// than failing inside Fbank.
    pub sample_rate: u32,
    /// Window length in samples used to compute speaker embeddings.
    /// Default 16 000 (1 s @ 16 kHz) per `docs/architecture.md`.
    pub sv_window_samples: usize,
    /// How many samples between embedding refreshes. Default 4 000
    /// (250 ms @ 16 kHz) per `docs/architecture.md`.
    pub sv_update_samples: usize,
    /// VAD speech-probability threshold above which a frame contributes
    /// to the speech buffer. Default 0.5.
    pub vad_threshold: f32,
    /// Master switch for auto-learn admission. When false, candidates
    /// are never admitted into the FIFO and `maybe_reset` is never
    /// invoked.
    pub enable_auto_learn: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            sv_window_samples: 16_000,
            sv_update_samples: 4_000,
            vad_threshold: 0.5,
            enable_auto_learn: true,
        }
    }
}

impl PipelineConfig {
    /// Frame duration (ms) at the configured sample rate; exposed so the
    /// caller can drive [`GateState`] / [`crate::f0::f0_statistics`] at
    /// the right cadence without hard-coding 32 ms.
    #[must_use]
    pub fn vad_frame_ms(&self) -> f32 {
        1000.0 * CHUNK_SAMPLES_16K as f32 / self.sample_rate as f32
    }
}

/// Container for the heavy stateful components consumed by
/// [`process_offline`].
///
/// The cohort is `Vec<Vec<f32>>` to share the AS-Norm primitive with
/// [`crate::gating::as_norm_score`] which takes any
/// `&[V: AsRef<[f32]>]`.
pub struct PipelineComponents {
    pub vad: SileroVad,
    pub fbank: Fbank,
    pub ecapa: EcapaTdnn,
    pub cohort: Vec<Vec<f32>>,
}

/// Auto-learn lifecycle event emitted during `process_offline`.
#[derive(Debug, Clone, Copy)]
pub struct AutoLearnEvent {
    pub frame_idx: usize,
    pub kind: AutoLearnKind,
    pub score: f32,
    pub f0_match: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoLearnKind {
    /// Candidate cleared every gate AND the pool's anchor-distance check
    /// — added to the auto-learn FIFO.
    Admit,
    /// Candidate cleared rule-based gates but the pool refused it via
    /// `can_auto_learn`.
    RejectAnchorDistance,
    /// `EmbeddingPool::maybe_reset` cleared the auto-learn FIFO due to
    /// drift.
    Reset,
}

/// Result of a single offline pipeline run.
#[derive(Debug, Default)]
pub struct ProcessResult {
    /// Envelope-gated audio at the input sample rate.
    pub audio: Vec<f32>,
    /// Run-length `(start_sample, is_on)` decisions consumed by
    /// [`apply_envelope`].
    pub gate_decisions: Vec<(usize, bool)>,
    /// Per-VAD-frame gate state.
    pub gate_per_frame: Vec<bool>,
    /// Per-VAD-frame integrated score (last update propagated forward
    /// between embedding refreshes).
    pub score_per_frame: Vec<f32>,
    /// Per-VAD-frame `cos_sim_max` (debug / calibration aid).
    pub cos_sim_max_per_frame: Vec<f32>,
    /// Per-VAD-frame F0 match (debug / calibration aid).
    pub f0_match_per_frame: Vec<f32>,
    /// Auto-learn admission / rejection / reset events in chronological
    /// order.
    pub auto_learn_events: Vec<AutoLearnEvent>,
}

/// Errors returned by [`process_offline`].
#[derive(Debug)]
pub enum PipelineError {
    Embedding(EmbeddingError),
    Envelope(ApplyEnvelopeError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedding(e) => write!(f, "embedding error: {e}"),
            Self::Envelope(e) => write!(f, "envelope error: {e}"),
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Embedding(e) => Some(e),
            Self::Envelope(_) => None,
        }
    }
}

impl From<EmbeddingError> for PipelineError {
    fn from(e: EmbeddingError) -> Self {
        Self::Embedding(e)
    }
}

impl From<ApplyEnvelopeError> for PipelineError {
    fn from(e: ApplyEnvelopeError) -> Self {
        Self::Envelope(e)
    }
}

/// Run the offline gating pipeline on a 16 kHz mono buffer.
///
/// `pool` is mutated in place: anchors stay untouched, but the
/// `auto_learn` FIFO and `metadata` may grow / drop entries during the
/// run depending on `pipeline_cfg.enable_auto_learn` and the pool's
/// `maybe_reset` outcome.
///
/// # Errors
/// * [`PipelineError::Embedding`] when ECAPA / VAD inference fails
/// * [`PipelineError::Envelope`] when the gate decision sequence
///   doesn't start at sample 0 (currently can't happen because the
///   pipeline always inserts a `(0, _)` head, but the error type is
///   surfaced to keep the contract explicit)
pub fn process_offline(
    audio: &[f32],
    pool: &mut EmbeddingPool,
    pipeline_cfg: &PipelineConfig,
    gate_cfg: &GateConfig,
    components: &mut PipelineComponents,
) -> Result<ProcessResult, PipelineError> {
    let vad_frame = CHUNK_SAMPLES_16K;
    let sv_sr = pipeline_cfg.sample_rate;
    let dt_ms = pipeline_cfg.vad_frame_ms();

    let mut speech_buffer: Vec<f32> = Vec::with_capacity(pipeline_cfg.sv_window_samples);
    let mut samples_since_update = 0_usize;
    let mut consecutive_speech_ms = 0.0_f32;
    let mut last_score = 0.0_f32;
    let mut last_cs = 0.0_f32;
    let mut last_fm = 1.0_f32;

    let mut gate_state = GateState::new(*gate_cfg);
    let mut decisions: Vec<(usize, bool)> = Vec::new();
    let mut current_decision: Option<bool> = None;
    let mut per_frame: Vec<bool> = Vec::new();
    let mut score_per_frame: Vec<f32> = Vec::new();
    let mut cs_per_frame: Vec<f32> = Vec::new();
    let mut fm_per_frame: Vec<f32> = Vec::new();
    let mut auto_learn_events: Vec<AutoLearnEvent> = Vec::new();

    let mut frame_start = 0_usize;
    let mut frame_idx = 0_usize;
    while frame_start + vad_frame <= audio.len() {
        let frame = &audio[frame_start..frame_start + vad_frame];

        let speech_prob = components.vad.score(frame)?;
        if speech_prob > pipeline_cfg.vad_threshold {
            speech_buffer.extend_from_slice(frame);
            if speech_buffer.len() > pipeline_cfg.sv_window_samples {
                let drop = speech_buffer.len() - pipeline_cfg.sv_window_samples;
                speech_buffer.drain(..drop);
            }
            consecutive_speech_ms += dt_ms;
        } else {
            consecutive_speech_ms = 0.0;
        }
        samples_since_update += vad_frame;

        if samples_since_update >= pipeline_cfg.sv_update_samples
            && speech_buffer.len() >= pipeline_cfg.sv_window_samples
        {
            samples_since_update = 0;
            let window = &speech_buffer[speech_buffer.len() - pipeline_cfg.sv_window_samples..];

            // Fbank → ECAPA → 192-dim embedding.
            let feats = components.fbank.compute(window);
            let n_frames = feats.len() / N_MELS;
            let embedding = components.ecapa.embed_features(&feats, n_frames, N_MELS)?;

            let f0_track =
                estimate_f0_track(window, sv_sr, 2048, 512, DEFAULT_F_MIN, DEFAULT_F_MAX);
            let (f0_mu, _) = f0_statistics(&f0_track);

            let cs = cos_sim_max(
                &embedding,
                &pool
                    .anchors()
                    .iter()
                    .chain(pool.auto_learn().iter())
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            let fm = f0_match(f0_mu, pool.metadata().f0_mu, pool.metadata().f0_sigma);
            last_cs = cs;
            last_fm = fm;
            last_score = if gate_cfg.use_as_norm && !components.cohort.is_empty() {
                // AS-Norm path: cohort-normalised similarity. F0 still
                // gates auto-learn admission via theta_f0 below.
                as_norm_score(&embedding, cs, &components.cohort, 20)
            } else {
                // Non-AS-Norm path. The Python PoC blends the score as
                // `α·cs + β·f0`, but `alpha`/`beta` live on Python's
                // monolithic GatingConfig and haven't been carved out
                // into the Rust GateConfig yet — for the offline path
                // raw `cos_sim_max` is a faithful enough proxy until
                // they land in a follow-up.
                cs
            };

            if pipeline_cfg.enable_auto_learn
                && should_admit_auto_learn(last_score, fm, consecutive_speech_ms, gate_cfg)
            {
                let admitted = pool.add_auto_learn(embedding);
                let kind = if admitted {
                    AutoLearnKind::Admit
                } else {
                    AutoLearnKind::RejectAnchorDistance
                };
                auto_learn_events.push(AutoLearnEvent {
                    frame_idx,
                    kind,
                    score: last_score,
                    f0_match: fm,
                });
                if admitted && pool.maybe_reset() {
                    auto_learn_events.push(AutoLearnEvent {
                        frame_idx,
                        kind: AutoLearnKind::Reset,
                        score: last_score,
                        f0_match: fm,
                    });
                }
            }
        }

        let is_on = gate_state.update(last_score, dt_ms);
        per_frame.push(is_on);
        score_per_frame.push(last_score);
        cs_per_frame.push(last_cs);
        fm_per_frame.push(last_fm);

        if current_decision != Some(is_on) {
            decisions.push((frame_start, is_on));
            current_decision = Some(is_on);
        }

        frame_start += vad_frame;
        frame_idx += 1;
    }

    if decisions.is_empty() {
        decisions.push((0, false));
    } else if decisions[0].0 != 0 {
        let first_on = decisions[0].1;
        decisions.insert(0, (0, first_on));
    }

    let audio_out = apply_envelope(audio, &decisions, sv_sr, *gate_cfg)?;
    Ok(ProcessResult {
        audio: audio_out,
        gate_decisions: decisions,
        gate_per_frame: per_frame,
        score_per_frame,
        cos_sim_max_per_frame: cs_per_frame,
        f0_match_per_frame: fm_per_frame,
        auto_learn_events,
    })
}
