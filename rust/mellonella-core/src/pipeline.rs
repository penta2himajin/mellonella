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
//! * Every `sv_update_samples` (8 000 = 500 ms by default — Phase 3.5
//!   step 3 bumped this from 250 ms), if the buffer is full, an
//!   embedding + F0 are recomputed and the gate score is updated.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use std::borrow::Cow;

use crate::embedding::{EcapaTdnn, EmbeddingError};
use crate::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use crate::f0::{estimate_f0_track, f0_statistics, DEFAULT_F_MAX, DEFAULT_F_MIN};
use crate::features::{Fbank, N_MELS};
use crate::gating::{
    apply_envelope, as_norm_score, f0_match, should_admit_auto_learn, ApplyEnvelopeError,
    GateConfig,
};
use crate::resample::{resample_to, ResampleError};
use crate::streaming::{
    AsyncRefresh, ScopedRefreshChannel, StreamingConfig, StreamingOutput, StreamingState,
};
use crate::tse::TSE_COND_DIM;
use crate::tse_stage::{TseStage, TseStageConfig, TseStageError};
use crate::vad::{SileroVad, CHUNK_SAMPLES_16K};

/// SV-side cadence for the offline pipeline.
#[derive(Debug, Clone)]
// A plain configuration struct — each bool is an independent,
// documented opt-in knob (async refresh, auto-learn, the three
// Stage B switches). Bundling them into sub-structs or an enum would
// only obscure the flat, greppable config surface callers rely on.
//
// `Copy` was dropped in Phase 5 — `PipelineConfig` now carries an
// optional [`TseStageConfig`] (with a `PathBuf` inside), so callers
// pass `&PipelineConfig` and clone when they need an owned copy.
#[allow(clippy::struct_excessive_bools)]
pub struct PipelineConfig {
    /// Sample rate of the input buffer. Currently fixed at 16 kHz so the
    /// component models match. Kept on the config struct so future
    /// non-SV-native rates can be flagged at construction time rather
    /// than failing inside Fbank.
    pub sample_rate: u32,
    /// Window length in samples used to compute speaker embeddings.
    /// Default 16 000 (1 s @ 16 kHz) per `docs/architecture.md`.
    pub sv_window_samples: usize,
    /// How many samples between embedding refreshes. Default 8 000
    /// (500 ms @ 16 kHz). Phase 3.5 step 3 bumped this from the
    /// original 4 000 (250 ms) to halve the ECAPA refresh cadence,
    /// which is the pipeline's dominant cost (~50 % of wall time).
    /// Tighter cadence (lower number) gives faster drift response;
    /// looser (higher number) saves CPU.
    pub sv_update_samples: usize,
    /// VAD speech-probability threshold above which a frame contributes
    /// to the speech buffer. Default 0.5.
    pub vad_threshold: f32,
    /// Master switch for auto-learn admission. When false, candidates
    /// are never folded into the pool's λ-residual adapted embedding.
    pub enable_auto_learn: bool,
    /// Minimum new-speech samples accumulated after a speech→silence
    /// transition before an *early* (cadence-skipping) ECAPA refresh
    /// is allowed. Default 4 000 (250 ms @ 16 kHz). Setting this to
    /// `usize::MAX` disables the early-refresh path entirely (every
    /// refresh waits for the full `sv_update_samples` cadence).
    ///
    /// Rationale: when [`Self::sv_update_samples`] grew to 500 ms
    /// (Phase 3.5 step 3) the worst-case ON→OFF latency on speaker
    /// turn went up by ~250 ms. Most real turns include a brief
    /// inter-speaker pause that the VAD already detects; firing the
    /// next ECAPA call as soon as we've heard enough of the new
    /// speaker recovers the latency without changing the
    /// steady-state refresh rate (no perf regression — the
    /// `pipeline_parity` test fixture uses `vad_threshold = -1.0`,
    /// so silence never fires and the trigger stays dormant).
    pub sv_min_new_samples_after_silence: usize,
    /// VAD pre-roll (lookback) window length in milliseconds. The
    /// streaming engine keeps the last `pre_roll_ms` of decision-rate
    /// audio in a ring; on every VAD OFF→ON transition the ring is
    /// **prepended** to the speech buffer so the next ECAPA refresh
    /// includes the pre-onset audio (issue #80). Default 100 ms — a
    /// compromise between covering weak fricative onsets (/s/, /f/,
    /// /h/, typical onset 30–100 ms) and limiting cross-speaker
    /// bleed on turn boundaries.
    ///
    /// In the **offline** path ([`crate::pipeline::process_offline`]),
    /// the same window is also used to **shift OFF→ON gate decisions
    /// back by `pre_roll_ms`** before applying the audio envelope, so
    /// the head of each utterance survives muting. The streaming
    /// engine does **not** apply that shift — doing so would require
    /// delaying live output by `pre_roll_ms`, which would break the
    /// sub-100 ms KPI from issue #66. Live callers that want the
    /// head-cut fix in audio output should buffer + reapply
    /// `apply_envelope` on the gate decisions returned by the engine.
    ///
    /// Set to 0 to disable both effects (identical to pre-#80
    /// behaviour).
    pub pre_roll_ms: u32,
    /// When `true`, ECAPA / Fbank / F0 for each refresh are dispatched
    /// to a worker thread so the VAD main loop can keep running while
    /// inference is in flight. Default `false` to preserve byte-equiv
    /// parity against the synchronous Python PoC.
    ///
    /// Trade-off: when enabled, the per-frame score `last_score`
    /// trails the refresh point by one ECAPA inference (~ECAPA wall
    /// time, e.g. ~44 ms on the dev VM). For streaming / real-time
    /// callers this is invisible — the gate's `hangover_ms` already
    /// smooths over multiples of that latency — but it changes
    /// `gate_per_frame` / `score_per_frame` outputs relative to the
    /// sync path, so the `pipeline_parity` fixture stays on the sync
    /// default. Phase 3.5 step 7 introduced this for sub-100 ms
    /// streaming RTF (see `docs/benchmarks.md`).
    pub async_refresh: bool,
    /// Independent VAD-silence hangover: when `silence_ms_since_speech`
    /// reaches this value, the gate is forced off **immediately**
    /// regardless of `last_score` — bypassing `hangover_ms` rather
    /// than stacking on top of it. The two hangovers combine via AND
    /// (`is_on = score_gate AND silence_ms_since_speech <
    /// silence_force_off_ms`), so the close time is exactly this
    /// value plus the envelope's `release_ms` fade, not
    /// `silence_force_off_ms + hangover_ms + release_ms`.
    ///
    /// Default `0.0` disables the VAD-silence path entirely, leaving
    /// the legacy purely-score-driven behaviour. Live callers (GUI,
    /// CLI `live`) opt into a positive value because in clean DFN3
    /// environments the noise that used to reset `last_score` via
    /// VAD-positive noise frames is suppressed, so the score-side
    /// gate stays open indefinitely on whatever the previous refresh
    /// scored.
    ///
    /// Tuning: pick a value larger than typical inter-word pauses
    /// (~250 ms) so normal speech doesn't trip the hangover; values
    /// below ~200 ms flicker on natural mid-sentence breath.
    pub silence_force_off_ms: f32,
    /// EMA smoothing factor applied to `last_score` on every refresh:
    /// `last_score = alpha * new_score + (1 - alpha) * last_score`.
    /// `1.0` disables smoothing — the new score replaces the old one.
    /// Lower values smooth out brief dips (e.g. an embedding shift at
    /// speech-onset transients, or one noisy refresh window) that would
    /// otherwise drag the gate below `theta_pass` for one refresh
    /// cycle. The first refresh always overwrites (no smoothing against
    /// the initial zero).
    ///
    /// Default `0.8`, calibrated against the `ci_accuracy` mini-scenario
    /// (history: #117 shipped `0.6`; #121 reverted to `1.0` after the
    /// TPR-only view showed a regression; once #125 added the
    /// `vary_snr` case and the `gate_transitions` chatter metric, an
    /// alpha sweep showed `0.8` strictly dominates `1.0` — it recovers
    /// the chatter reduction and the snr_15 / sim4_9db TPR gains while
    /// leaving snr_10 untouched, which is exactly the cell that
    /// regressed at `0.6` / `0.7`). `1.0` restores the no-smoothing
    /// behaviour and is pinned by the `pipeline_parity` fixture so the
    /// Rust↔Python byte-equal contract is unaffected.
    pub score_ema_alpha: f32,
    /// **Stage B, Part 1** — opt-in per-frame fast F0 cue. When `true`,
    /// the streaming core runs a single `yin_frame_with_cache` per VAD
    /// frame over a 2048-sample decision-rate ring, maps the
    /// instantaneous F0 through `f0_match` against the enrolled
    /// `f0_mu` / `f0_sigma`, and **fuses** the result into the score
    /// that drives the gate:
    /// `fused = last_score + fast_cue_weight * (fm_fast - fast_cue_f0_neutral)`.
    /// The slow ECAPA-refreshed `last_score` is the anchor; the F0 cue
    /// is a cheap per-frame nudge so a speaker turn moves the gate
    /// before the next embedding refresh lands.
    ///
    /// Default `false` — with the default the fused term is never
    /// computed and `last_score` feeds the gate exactly as before, so
    /// the `pipeline_parity` byte-equal contract is untouched.
    pub fast_cue_enabled: bool,
    /// Weight of the fused F0 nudge (see [`Self::fast_cue_enabled`]).
    /// Small by design — the cue is a low-confidence per-frame hint,
    /// not a decision on its own. Inert when `fast_cue_enabled` is
    /// `false`.
    pub fast_cue_weight: f32,
    /// Neutral point the per-frame `fm_fast` is measured against
    /// before scaling by [`Self::fast_cue_weight`]. `fm_fast` above
    /// this nudges the score up, below nudges down. `0.5` is the
    /// midpoint of `f0_match`'s `[0, 1]` range. Also used as the
    /// "no evidence" value when the F0 ring isn't full yet or YIN
    /// returns unvoiced. Inert when `fast_cue_enabled` is `false`.
    pub fast_cue_f0_neutral: f32,
    /// **Stage B, Part 2** — opt-in adaptive-window turn detection.
    /// When `true`, the streaming core tracks a fast/slow EMA of the
    /// per-frame `fm_fast` cue and, on a suspected speaker turn,
    /// temporarily shrinks the ECAPA window to
    /// [`Self::sv_turn_window_samples`] and tightens the refresh
    /// cadence to [`Self::sv_turn_update_samples`] so the embedding
    /// re-converges on the new speaker faster. Once the cue
    /// re-stabilises the window grows back to
    /// [`Self::sv_window_samples`].
    ///
    /// Default `false` — with the default `effective_window` stays
    /// pinned at `sv_window_samples` and every turn-detection branch
    /// is inert, so behaviour (and `pipeline_parity`) is unchanged.
    /// Requires the per-frame F0 cue to be meaningful; enabling this
    /// without [`Self::fast_cue_enabled`] still detects turns (the
    /// cue is computed whenever turn detection needs it) but does not
    /// fuse the cue into the gate score.
    pub turn_detect_enabled: bool,
    /// Drop in the fast `fm_fast` EMA, relative to the slow baseline,
    /// that marks an offset-suspect (target→other) turn. Inert when
    /// `turn_detect_enabled` is `false`.
    pub turn_drop_delta: f32,
    /// Consecutive frames the `fm_fast` EMA must sit close to its
    /// (new) baseline before the window is allowed to grow back from
    /// `Shrunk` → `Steady`. Inert when `turn_detect_enabled` is
    /// `false`.
    pub turn_stable_frames: u32,
    /// Shrunk ECAPA window length (samples) used while a turn is
    /// suspected. Default 8 000 (0.5 s @ 16 kHz) — half the steady
    /// `sv_window_samples`. Inert when `turn_detect_enabled` is
    /// `false`.
    pub sv_turn_window_samples: usize,
    /// Shrunk refresh cadence (samples) used while a turn is
    /// suspected. Default 2 000 (125 ms @ 16 kHz). Inert when
    /// `turn_detect_enabled` is `false`.
    pub sv_turn_update_samples: usize,
    /// **Stage B, Part 2** — opt-in offset fail-closed. When `true`
    /// *and* an offset-suspect turn fires, the gate is forced **off**
    /// immediately (an extra AND term in `is_on`, parallel to the
    /// `silence_force_off_ms` rule and likewise bypassing the gate
    /// hangover) until a refresh on the shrunk window confirms the
    /// new speaker. There is no symmetric onset fail-open — onset
    /// still has to earn the gate through normal scoring.
    ///
    /// Default `false`. Requires `turn_detect_enabled` to have any
    /// effect (the offset-suspect signal comes from turn detection).
    pub offset_fail_closed: bool,
    /// **Stage C, Phase 5** — opt-in target-speaker-extraction. When
    /// `Some`, the offline pipeline loads the streaming TSE ONNX at
    /// the supplied path, snapshots the enrolled target speaker's
    /// embedding from the pool, runs the audio through the TSE model
    /// **before** the envelope is applied, and feeds the extracted
    /// audio (instead of the raw input) to the envelope stage.
    /// Decisions from VAD / SV / gating are unchanged — they're still
    /// computed on the raw decision-rate audio — only the audio the
    /// gate's envelope is applied to changes.
    ///
    /// **Sample-rate contract:** the configured `TseStageConfig` carries
    /// a `TseConfig` (`poc_16k` → 16 kHz, `prod_48k` → 48 kHz) and the
    /// caller's `audio_sample_rate` must match it. The decision rate
    /// (`PipelineConfig::sample_rate` — VAD/SV) is independent;
    /// `apply_envelope_dual_rate` handles the dual-rate gate envelope.
    /// Mismatched audio rate is rejected with
    /// [`PipelineError::TseRateMismatch`].
    ///
    /// Default `None` — with the default the TSE stage is never
    /// constructed and the offline pipeline behaves byte-identically
    /// to pre-Phase-5 builds (the `pipeline_parity` byte-equal
    /// contract is unaffected).
    pub tse: Option<TseStageConfig>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            sv_window_samples: 16_000,
            sv_update_samples: 8_000,
            vad_threshold: 0.5,
            enable_auto_learn: true,
            sv_min_new_samples_after_silence: 4_000,
            pre_roll_ms: 100,
            async_refresh: false,
            silence_force_off_ms: 0.0,
            score_ema_alpha: 0.8,
            // Stage B — all opt-in, default OFF so behaviour (and the
            // `pipeline_parity` byte-equal contract) is unchanged.
            fast_cue_enabled: false,
            fast_cue_weight: 0.15,
            fast_cue_f0_neutral: 0.5,
            turn_detect_enabled: false,
            turn_drop_delta: 0.25,
            turn_stable_frames: 10,
            sv_turn_window_samples: 8_000,
            sv_turn_update_samples: 2_000,
            offset_fail_closed: false,
            // Stage C TSE — opt-in, default OFF so the offline pipeline
            // remains byte-identical to pre-Phase-5 builds.
            tse: None,
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

    /// Pre-roll ring capacity in decision-rate samples — `pre_roll_ms`
    /// converted to samples at [`Self::sample_rate`]. Zero when
    /// pre-roll is disabled.
    #[must_use]
    pub fn pre_roll_samples_decision(&self) -> usize {
        (u64::from(self.pre_roll_ms) * u64::from(self.sample_rate) / 1000) as usize
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
    /// Optional Stage C TSE stage. Lazily constructed by
    /// [`process_offline`] from the [`PipelineConfig::tse`] config on
    /// first use (and reset on each subsequent run); callers may also
    /// pre-build the stage with [`crate::tse_stage::TseStage::from_config`]
    /// and pass it in here to share the loaded ONNX across runs.
    ///
    /// Default `None`. The streaming engine ([`crate::streaming`])
    /// ignores this field — TSE in the streaming path is a Phase 5
    /// follow-up.
    pub tse: Option<TseStage>,
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
    /// — folded into the λ-residual adapted embedding (#118).
    Admit,
    /// Candidate cleared rule-based gates but the pool refused it via
    /// `can_auto_learn`.
    RejectAnchorDistance,
    /// Pool was empty + `allow_bootstrap_from_runtime` was on +
    /// VAD-positive speech ran long enough — the first anchor was
    /// seeded from this candidate via
    /// [`crate::enrollment::EmbeddingPool::bootstrap_seed`]. Subsequent
    /// refreshes flow back through the normal `Admit` /
    /// `RejectAnchorDistance` path.
    BootstrapSeed,
}

/// Bundle of the three per-frame scoring scalars carried between
/// VAD frames: the gate-feeding integrated score plus the two
/// diagnostic components (`cos_sim_max` and F0 match). Extracted as
/// a struct so the streaming per-frame core and both async refresh
/// paths can pass them around as one `&mut` reference rather than
/// three loose `&mut f32`s — the borrow checker is happier and the
/// refresh-strategy trait signature stays compact.
#[derive(Debug, Clone, Copy)]
// The `last_*` prefix is load-bearing — these mirror the long-lived
// `last_score` / `last_cs` / `last_fm` names used throughout the
// pipeline + streaming docs, so the shared vocabulary is worth the
// `struct_field_names` lint.
#[allow(clippy::struct_field_names)]
pub(crate) struct ScoreState {
    /// Integrated, EMA-smoothed score fed into the gate.
    pub last_score: f32,
    /// Last refresh's raw `cos_sim_max` (diagnostics only).
    pub last_cs: f32,
    /// Last refresh's F0 match (diagnostics only).
    pub last_fm: f32,
}

impl ScoreState {
    /// Initial state: zero score / cos-sim, neutral (1.0) F0 match —
    /// matches the historical inline initialisation in both the
    /// streaming engine and `process_offline_async`.
    pub(crate) fn new() -> Self {
        Self {
            last_score: 0.0,
            last_cs: 0.0,
            last_fm: 1.0,
        }
    }
}

/// EMA smoothing for `last_score`: `alpha * new + (1 - alpha) * old`,
/// with two shortcuts:
///
/// * `alpha >= 1.0` (or any value outside `[0, 1)`) → replace.
/// * `last_score == 0.0` → also replace, so the first refresh seeds
///   the smoother instead of blending against the initial zero.
#[inline]
#[must_use]
pub(crate) fn smooth_score(last_score: f32, new_score: f32, alpha: f32) -> f32 {
    if !(0.0..1.0).contains(&alpha) || last_score == 0.0 {
        return new_score;
    }
    alpha * new_score + (1.0 - alpha) * last_score
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
    /// `true` when the Stage C TSE stage was actually invoked on this
    /// run (i.e. `cfg.tse.is_some()` and the rate-check passed). Used
    /// by tests to assert the default-OFF / opt-in behaviour. Always
    /// `false` for the async path (TSE wiring there is a follow-up).
    pub tse_applied: bool,
}

/// Errors returned by [`process_offline`].
#[derive(Debug)]
pub enum PipelineError {
    Embedding(EmbeddingError),
    Envelope(ApplyEnvelopeError),
    Resample(ResampleError),
    /// Forwarded from the Stage C TSE stage (ONNX load failures,
    /// invalid chunk lengths, etc.).
    TseStage(TseStageError),
    /// The Stage C TSE config requires `audio_sample_rate ==
    /// pipeline_cfg.sample_rate == 16_000` (PoC model is 16 kHz only).
    /// The 48 kHz prod model will lift this restriction in Phase 4.
    TseRateMismatch {
        /// Sample rate the caller's audio buffer is at.
        audio_sr: u32,
        /// Sample rate the configured TSE model was exported at.
        expected_sr: u32,
    },
    /// The Stage C TSE config was set but the embedding pool has no
    /// anchors yet (no enrolled target speaker) — TSE needs a 192-dim
    /// cond embedding to condition on.
    TseMissingEnrollment,
    /// The Stage C TSE config was combined with `async_refresh = true`.
    /// TSE wiring through the async-refresh worker is a Phase 5
    /// follow-up; sync `process_offline` is the only supported path
    /// for now.
    TseAsyncUnsupported,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedding(e) => write!(f, "embedding error: {e}"),
            Self::Envelope(e) => write!(f, "envelope error: {e}"),
            Self::Resample(e) => write!(f, "resample error: {e}"),
            Self::TseStage(e) => write!(f, "TSE stage error: {e}"),
            Self::TseRateMismatch {
                audio_sr,
                expected_sr,
            } => write!(
                f,
                "Stage C TSE audio_sr={audio_sr} Hz != model's expected SR \
                 {expected_sr} Hz. Use the right `TseStageConfig` variant \
                 (e.g. `TseStageConfig::new_prod_48k` for the 48 kHz model) \
                 or resample audio to match."
            ),
            Self::TseMissingEnrollment => write!(
                f,
                "Stage C TSE was enabled but the embedding pool has no \
                 anchors — enroll a target speaker first"
            ),
            Self::TseAsyncUnsupported => write!(
                f,
                "Stage C TSE with async_refresh = true is not yet \
                 wired up — use the sync offline path"
            ),
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Embedding(e) => Some(e),
            Self::Envelope(_)
            | Self::TseRateMismatch { .. }
            | Self::TseMissingEnrollment
            | Self::TseAsyncUnsupported => None,
            Self::Resample(e) => Some(e),
            Self::TseStage(e) => Some(e),
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

impl From<ResampleError> for PipelineError {
    fn from(e: ResampleError) -> Self {
        Self::Resample(e)
    }
}

impl From<TseStageError> for PipelineError {
    fn from(e: TseStageError) -> Self {
        Self::TseStage(e)
    }
}

/// Snapshot the cond-conditioning embedding the Stage C TSE model is
/// conditioned on from the embedding pool.
///
/// Picks the **anchor centroid** (`E_enroll`, the element-wise mean of
/// the enrollment anchors — what
/// [`crate::enrollment::EmbeddingPool::anchor_centroid`] returns) as
/// Compute the 192-dim TSE conditioning vector from `pool`.
///
/// Returns the λ-residual **adapted** embedding when available,
/// falling back to the **anchor centroid** before the first
/// auto-learn admit (when `adapted` is still `None`):
///
/// * The adapted embedding is `λ·centroid + (1-λ)·evidence`; it
///   tracks runtime acoustic conditions (microphone, environment,
///   speaker state) once auto-learn has admitted at least one
///   high-confidence runtime embedding. Stage C Step 5 refreshes the
///   TSE stage's cond on every successful admit, so a long session
///   sees the model condition on the most current view of the target
///   speaker rather than the frozen enrollment.
/// * Before the first admit, `adapted` is `None`; in that window the
///   anchor centroid is the only signal we have and TSE conditions on
///   it. For a single-anchor pool the centroid is that anchor
///   verbatim (each `f32 * 1.0` is exact), so the accessor reduces to
///   "the enrolled embedding" on the simplest path.
///
/// # Errors
/// Returns [`PipelineError::TseMissingEnrollment`] when the pool has
/// no anchors yet (both `adapted` and `anchor_centroid` are `None`).
pub(crate) fn tse_cond_embedding(pool: &EmbeddingPool) -> Result<Vec<f32>, PipelineError> {
    let vec = pool
        .adapted()
        .or_else(|| pool.anchor_centroid())
        .ok_or(PipelineError::TseMissingEnrollment)?;
    if vec.len() != TSE_COND_DIM {
        return Err(PipelineError::TseStage(TseStageError::InvalidCondLength {
            got: vec.len(),
            expected: TSE_COND_DIM,
        }));
    }
    Ok(vec.to_vec())
}

/// Ensure `components.tse` holds a TSE stage compatible with the
/// supplied config + cond embedding.
///
/// Cases:
/// * stage already present → reset its accumulator + state, update
///   its cond embedding from the current snapshot.
/// * stage absent → load the ONNX from `config.tse.unwrap().onnx_path`
///   and stash it on `components`.
fn ensure_tse_stage(
    components: &mut PipelineComponents,
    pipeline_cfg: &PipelineConfig,
    cond: &[f32],
) -> Result<(), PipelineError> {
    let Some(stage_cfg) = pipeline_cfg.tse.as_ref() else {
        // Caller only invokes this when `tse.is_some()`; defensive
        // early-out keeps the body single-level.
        return Ok(());
    };
    if let Some(stage) = components.tse.as_mut() {
        stage.reset();
        stage.set_cond_embedding(cond)?;
    } else {
        let stage = TseStage::from_config(stage_cfg, cond)?;
        components.tse = Some(stage);
    }
    Ok(())
}

/// Drive a [`TseStage`] over a whole offline audio buffer at the
/// decision rate and return exactly `audio.len()` extracted samples.
/// Output is zero-padded if the underlying flush ran short or
/// truncated if it overshot — matching [`crate::dfn3::Dfn3Pipeline::process`]'s
/// fixed-length output contract so the envelope step can apply a
/// pre-computed decision sequence sample-for-sample.
fn run_tse_offline(stage: &mut TseStage, audio: &[f32]) -> Result<Vec<f32>, PipelineError> {
    stage.reset();
    let mut out = stage.process(audio)?;
    let tail = stage.flush()?;
    out.extend(tail);
    out.truncate(audio.len());
    if out.len() < audio.len() {
        out.resize(audio.len(), 0.0);
    }
    Ok(out)
}

/// Resample `audio` from `audio_sr` to `decision_sr` for the
/// decision-path consumers (VAD / ECAPA / F0). When the two rates
/// match we borrow the input slice and avoid the
/// `audio.to_vec()` round-trip baked into [`resample_to`]'s
/// identity branch.
fn audio_for_decisions(
    audio: &[f32],
    audio_sr: u32,
    decision_sr: u32,
) -> Result<Cow<'_, [f32]>, PipelineError> {
    if audio_sr == decision_sr {
        Ok(Cow::Borrowed(audio))
    } else {
        Ok(Cow::Owned(resample_to(audio, audio_sr, decision_sr)?))
    }
}

/// Map a decision-rate sample index onto the audio-rate axis using a
/// rounding integer formula (`u64` to avoid intermediate overflow on
/// long offline clips). Identity-equivalent when the two rates match.
fn scale_to_audio_rate(s_decision: usize, decision_sr: u32, audio_sr: u32) -> usize {
    if decision_sr == audio_sr {
        return s_decision;
    }
    let num = s_decision as u64 * u64::from(audio_sr);
    let denom = u64::from(decision_sr);
    // Round half-to-positive — symmetric and dependency-free.
    let rounded = (num + denom / 2) / denom;
    rounded as usize
}

/// Shift every OFF→ON boundary in `decisions` backwards by `shift`
/// samples (floored at the preceding boundary). Used in the offline
/// paths only — the streaming engine never rewinds emitted audio
/// (see [`PipelineConfig::pre_roll_ms`] for the trade-off).
///
/// Issue #80: VAD declares speech a few frames into the actual onset,
/// so the envelope's fade-in starts late and the head of each
/// utterance is muted. Shifting the boundary back lets the audio
/// preceding the trigger frame survive the envelope.
///
/// Zero-length regions that collapse after the shift (an `OFF` at
/// sample X immediately followed by an `ON` shifted onto the same X)
/// are dropped so [`apply_envelope`]'s monotonic-boundary contract
/// holds.
fn shift_off_on_decisions_back(decisions: &mut Vec<(usize, bool)>, shift: usize) {
    if shift == 0 || decisions.len() < 2 {
        return;
    }
    // The first decision is anchored at sample 0 (see `process_offline`
    // prefix normalisation); leave it alone.
    for i in 1..decisions.len() {
        let (idx, is_on) = decisions[i];
        if !is_on {
            continue;
        }
        let floor = decisions[i - 1].0;
        let new_idx = idx.saturating_sub(shift).max(floor);
        decisions[i].0 = new_idx;
    }
    // Drop OFF entries whose region collapsed to zero length after a
    // following ON was shifted back onto them. Walk in reverse so
    // indices stay valid.
    let mut i = decisions.len();
    while i >= 2 {
        i -= 1;
        if decisions[i].0 == decisions[i - 1].0 {
            // Keep the later entry (which carries the new is_on state).
            decisions.remove(i - 1);
        }
    }
}

/// Apply [`apply_envelope`] at the audio rate using decisions
/// emitted in the decision-rate sample space. The first decision is
/// always anchored at 0 (the offline orchestrators enforce that),
/// and subsequent boundaries are scaled into the audio-rate axis
/// before invocation. Output length matches `audio_out.len()` so
/// downstream WAV writers can encode at the audio rate unchanged.
fn apply_envelope_dual_rate(
    audio_out: &[f32],
    decisions_decision_rate: &[(usize, bool)],
    audio_sr: u32,
    decision_sr: u32,
    gate_cfg: GateConfig,
) -> Result<Vec<f32>, ApplyEnvelopeError> {
    if audio_sr == decision_sr {
        return apply_envelope(audio_out, decisions_decision_rate, audio_sr, gate_cfg);
    }
    let n_out = audio_out.len();
    let mut scaled: Vec<(usize, bool)> = decisions_decision_rate
        .iter()
        .map(|&(s, on)| (scale_to_audio_rate(s, decision_sr, audio_sr).min(n_out), on))
        .collect();
    // De-duplicate boundaries that collapse after rounding so
    // `apply_envelope`'s monotonic-boundary precondition holds.
    scaled.dedup_by(|next, prev| next.0 == prev.0);
    apply_envelope(audio_out, &scaled, audio_sr, gate_cfg)
}

/// Run the offline gating pipeline on a 16 kHz mono buffer.
///
/// `pool` is mutated in place: anchors stay untouched, but the
/// λ-residual adapted embedding may be updated during the run when
/// `pipeline_cfg.enable_auto_learn` is set and a refresh clears the
/// auto-learn gates.
///
/// # Errors
/// * [`PipelineError::Embedding`] when ECAPA / VAD inference fails
/// * [`PipelineError::Envelope`] when the gate decision sequence
///   doesn't start at sample 0 (currently can't happen because the
///   pipeline always inserts a `(0, _)` head, but the error type is
///   surfaced to keep the contract explicit)
pub fn process_offline(
    audio: &[f32],
    audio_sample_rate: u32,
    pool: &mut EmbeddingPool,
    pipeline_cfg: &PipelineConfig,
    gate_cfg: &GateConfig,
    components: &mut PipelineComponents,
) -> Result<ProcessResult, PipelineError> {
    // Stage C TSE setup runs on **both** sync and async offline paths.
    // The pool may evolve via auto-learn during the run; snapshotting
    // the cond embedding here ("frozen for the run") is the simplest
    // correct semantics for this PR. The model's expected sample rate
    // (16 kHz for poc_16k, 48 kHz for prod_48k) is carried by the
    // `TseStageConfig` and must match `audio_sample_rate`. The
    // decision rate (VAD/SV) is independent — `apply_envelope_dual_rate`
    // already handles the SR mismatch when applying the gate envelope.
    let tse_enabled = pipeline_cfg.tse.is_some();
    if let Some(stage_cfg) = pipeline_cfg.tse.as_ref() {
        let expected_sr = stage_cfg.model.sample_rate();
        if audio_sample_rate != expected_sr {
            return Err(PipelineError::TseRateMismatch {
                audio_sr: audio_sample_rate,
                expected_sr,
            });
        }
        let cond = tse_cond_embedding(pool)?;
        ensure_tse_stage(components, pipeline_cfg, &cond)?;
    }

    if pipeline_cfg.async_refresh {
        // Async path (Step 4 of Stage C 実適用): same wiring as sync
        // — the streaming-engine pass runs WITHOUT the TSE
        // post-processing inline (mirroring the sync path's
        // `tse: None` strip below), then `run_tse_offline` +
        // `apply_envelope_dual_rate` re-apply TSE + the gate envelope
        // on top of the async-mode result.
        let mut result = process_offline_async(
            audio,
            audio_sample_rate,
            pool,
            &PipelineConfig {
                tse: None,
                ..pipeline_cfg.clone()
            },
            gate_cfg,
            components,
        )?;
        if let Some(stage) = components.tse.as_mut().filter(|_| tse_enabled) {
            let extracted = run_tse_offline(stage, audio)?;
            result.audio = apply_envelope_dual_rate(
                &extracted,
                &result.gate_decisions,
                audio_sample_rate,
                pipeline_cfg.sample_rate,
                *gate_cfg,
            )?;
            result.tse_applied = true;
        }
        return Ok(result);
    }
    // Sync offline = streaming engine called once with the whole
    // buffer, then flushed. Per-frame outputs match the previous
    // monolithic implementation byte-for-byte at identity rate
    // (`streaming_identity_rate_per_frame_matches_offline`).
    let cfg = StreamingConfig {
        pipeline: PipelineConfig {
            // The streaming engine doesn't consume `tse` (and its
            // `StreamingConfig::Default` derives off `PipelineConfig`,
            // which now carries the new field). Strip it on the
            // engine's own config copy to be unambiguous — TSE in the
            // streaming engine itself is a Phase 5 follow-up.
            tse: None,
            ..pipeline_cfg.clone()
        },
        gate: *gate_cfg,
        audio_sample_rate,
        diagnostics: true,
        // Offline path doesn't drive DFN3 through the streaming engine —
        // CLI / GUI run DFN3 as a one-shot pre-process step today.
        dfn3_onnx_path: None,
        ..StreamingConfig::default()
    };
    let mut state = StreamingState::new(&cfg)?;
    let mut head = state.push_block(audio, pool, components, &cfg)?;
    let mut tail = state.flush(pool, components, &cfg)?;

    head.audio.append(&mut tail.audio);
    head.gate_decisions.append(&mut tail.gate_decisions);
    head.events.append(&mut tail.events);
    head.gate_per_frame.append(&mut tail.gate_per_frame);
    head.score_per_frame.append(&mut tail.score_per_frame);
    head.cos_sim_max_per_frame
        .append(&mut tail.cos_sim_max_per_frame);
    head.f0_match_per_frame.append(&mut tail.f0_match_per_frame);

    // Mirror the historical `gate_decisions[0].0 == 0` precondition
    // `apply_envelope` was designed to expect. The streaming engine
    // emits decisions on transitions starting at audio_samples_emitted=0,
    // so the first decision is already at 0 in non-empty cases. Empty
    // happens when audio is too short for one VAD frame.
    let mut decisions = head.gate_decisions;
    if decisions.is_empty() {
        decisions.push((0, false));
    } else if decisions[0].0 != 0 {
        let first_on = decisions[0].1;
        decisions.insert(0, (0, first_on));
    }

    // Pre-roll envelope shift (issue #80). The streaming engine
    // already applied the unshifted envelope into `head.audio`; when
    // `pre_roll_ms > 0`, we re-run `apply_envelope_dual_rate` over
    // the original input audio with the shifted decisions so the
    // head of each utterance survives muting. Streaming decisions are
    // at audio rate, so the shift is computed at audio rate.
    if pipeline_cfg.pre_roll_ms > 0 {
        let shift_audio =
            (u64::from(pipeline_cfg.pre_roll_ms) * u64::from(audio_sample_rate) / 1000) as usize;
        shift_off_on_decisions_back(&mut decisions, shift_audio);
        head.audio = apply_envelope_dual_rate(
            audio,
            &decisions,
            audio_sample_rate,
            audio_sample_rate,
            *gate_cfg,
        )?;
    }

    // Stage C TSE: run target-speaker extraction on the raw audio
    // **before** the envelope is applied, then re-apply the envelope
    // over the extracted audio with the (possibly pre-roll-shifted)
    // decision sequence. The decisions themselves came from the
    // unmodified streaming engine — i.e. VAD / SV / gating saw the
    // original mixture — so a future iteration may want to score on
    // TSE-cleaned audio; for now we keep gating untouched.
    //
    // Decisions were computed at the decision rate
    // (`pipeline_cfg.sample_rate` — 16 kHz for VAD). The audio is at
    // the model's expected SR (16 kHz for poc_16k, 48 kHz for prod_48k).
    // `apply_envelope_dual_rate` upsamples the per-frame envelope as
    // needed, so the two rates may legitimately differ.
    let mut tse_applied = false;
    if let Some(stage) = components.tse.as_mut().filter(|_| tse_enabled) {
        let extracted = run_tse_offline(stage, audio)?;
        head.audio = apply_envelope_dual_rate(
            &extracted,
            &decisions,
            audio_sample_rate,
            pipeline_cfg.sample_rate,
            *gate_cfg,
        )?;
        tse_applied = true;
    }

    Ok(ProcessResult {
        audio: head.audio,
        gate_decisions: decisions,
        gate_per_frame: head.gate_per_frame,
        score_per_frame: head.score_per_frame,
        cos_sim_max_per_frame: head.cos_sim_max_per_frame,
        f0_match_per_frame: head.f0_match_per_frame,
        auto_learn_events: head.events,
        tse_applied,
    })
}

/// Async variant of [`process_offline`]: ECAPA / Fbank / F0 for each
/// refresh run on a single worker thread so the VAD-driven main loop
/// keeps progressing while inference is in flight. Selected
/// automatically when `pipeline_cfg.async_refresh = true`.
///
/// Scoring (cos-sim, AS-Norm, auto-learn admission) stays on the main
/// thread — the worker returns `(embedding, f0_mu)` and the main loop
/// folds them into the pool / gate state. As a result the per-frame
/// `last_score` trails the refresh sample boundary by roughly one
/// ECAPA inference; see [`PipelineConfig::async_refresh`] for the
/// trade-off.
fn process_offline_async(
    audio: &[f32],
    audio_sample_rate: u32,
    pool: &mut EmbeddingPool,
    pipeline_cfg: &PipelineConfig,
    gate_cfg: &GateConfig,
    components: &mut PipelineComponents,
) -> Result<ProcessResult, PipelineError> {
    let vad_frame = CHUNK_SAMPLES_16K;
    let sv_sr = pipeline_cfg.sample_rate;

    let decision_audio = audio_for_decisions(audio, audio_sample_rate, sv_sr)?;
    let audio_dec: &[f32] = decision_audio.as_ref();

    // The async offline path now shares the streaming per-frame core.
    // `process_offline_async` still owns its own `std::thread::scope`
    // worker (it borrows `PipelineComponents`, so it can't move
    // `fbank` / `ecapa` into a persistent `AsyncWorker`); the loop
    // body becomes "build an `AsyncRefresh` over the scoped channel,
    // call `step_one_frame_core`". The audio is pre-resampled to the
    // decision rate, so the `StreamingState` runs at identity rate
    // (no internal resampler) and the per-frame `audio_chunk` is the
    // decision-rate frame itself — its envelope-gated output is
    // discarded here because the offline path re-applies the envelope
    // at the audio rate at the end from the run-length `decisions`.
    //
    // `diagnostics: true` mirrors the sync `process_offline` so the
    // core fills `score_per_frame` / `cos_sim_max_per_frame` /
    // `f0_match_per_frame` unconditionally — the historical
    // `process_offline_async` always emitted those regardless of the
    // (then non-existent) diagnostics flag.
    let cfg = StreamingConfig {
        pipeline: PipelineConfig {
            // `StreamingState::new` is the sync constructor and rejects
            // `async_refresh = true`; the async behaviour here comes
            // from the `AsyncRefresh` strategy + scoped worker below,
            // not from the `StreamingState` worker, so this flag must
            // be cleared on the state's own config copy.
            async_refresh: false,
            ..pipeline_cfg.clone()
        },
        gate: *gate_cfg,
        // Identity rate: `audio_dec` is already at the decision rate.
        audio_sample_rate: sv_sr,
        diagnostics: true,
        // Offline async path doesn't drive DFN3.
        dfn3_onnx_path: None,
        ..StreamingConfig::default()
    };
    let mut state = StreamingState::new(&cfg)?;

    let mut out = StreamingOutput::default();

    // Disjoint borrows of PipelineComponents so the worker thread can
    // own ecapa + fbank while the main loop keeps vad. The `cohort` is
    // read-only and shared by-reference.
    let PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort,
        // TSE wiring through the async-refresh worker is a Phase 5
        // follow-up; the sync `process_offline` rejects
        // `tse.is_some() && async_refresh` up front.
        tse: _,
    } = components;
    let cohort: &Vec<Vec<f32>> = cohort;

    let (work_tx, work_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let (res_tx, res_rx) = std::sync::mpsc::channel::<Result<(Vec<f32>, f32), EmbeddingError>>();

    std::thread::scope(|scope| -> Result<(), PipelineError> {
        scope.spawn(move || {
            while let Ok(window) = work_rx.recv() {
                let msg = match fbank_ecapa_one(&window, fbank, ecapa) {
                    Ok(embedding) => {
                        let f0_track = estimate_f0_track(
                            &window,
                            sv_sr,
                            2048,
                            512,
                            DEFAULT_F_MIN,
                            DEFAULT_F_MAX,
                        );
                        let (f0_mu, _) = f0_statistics(&f0_track);
                        Ok((embedding, f0_mu))
                    }
                    Err(e) => Err(e),
                };
                if res_tx.send(msg).is_err() {
                    break;
                }
            }
        });

        // The `ScopedRefreshChannel` carries the outstanding / pending
        // / FIFO-frame-index bookkeeping the old inline loop kept, so
        // the "at most one inference in flight + one queued window"
        // cadence is byte-identical.
        let mut channel = ScopedRefreshChannel::new(work_tx, res_rx);

        let mut frame_start = 0_usize;
        while frame_start + vad_frame <= audio_dec.len() {
            let frame = &audio_dec[frame_start..frame_start + vad_frame];
            let mut strategy = AsyncRefresh {
                channel: &mut channel,
                cohort,
                pool,
                enable_auto_learn: pipeline_cfg.enable_auto_learn,
                score_ema_alpha: pipeline_cfg.score_ema_alpha,
                gate_cfg: *gate_cfg,
            };
            state.step_one_frame_core(frame, frame, vad, &mut strategy, &cfg, &mut out)?;
            frame_start += vad_frame;
        }

        // Drain whatever ECAPA work is still in flight so the result
        // log is complete before we tear down the worker. Per-frame
        // outputs were already emitted for the main loop above, so
        // these only affect pool / auto_learn_events. Shares the
        // streaming engine's `drain_trailing_refreshes` tail.
        state.drain_trailing_refreshes(&mut channel, pool, cohort, &cfg, &mut out)?;

        Ok(())
    })?;

    let mut decisions = out.gate_decisions;
    let per_frame = out.gate_per_frame;
    let score_per_frame = out.score_per_frame;
    let cs_per_frame = out.cos_sim_max_per_frame;
    let fm_per_frame = out.f0_match_per_frame;
    let auto_learn_events = out.events;

    if decisions.is_empty() {
        decisions.push((0, false));
    } else if decisions[0].0 != 0 {
        let first_on = decisions[0].1;
        decisions.insert(0, (0, first_on));
    }

    // Pre-roll envelope shift (issue #80). Async-path decisions are
    // emitted in decision-rate sample-index space (`frame_start`),
    // so the shift is computed at the decision rate before the
    // dual-rate envelope scales boundaries onto the audio axis.
    if pipeline_cfg.pre_roll_ms > 0 {
        let shift_decision = pipeline_cfg.pre_roll_samples_decision();
        shift_off_on_decisions_back(&mut decisions, shift_decision);
    }

    let audio_out =
        apply_envelope_dual_rate(audio, &decisions, audio_sample_rate, sv_sr, *gate_cfg)?;
    Ok(ProcessResult {
        audio: audio_out,
        gate_decisions: decisions,
        gate_per_frame: per_frame,
        score_per_frame,
        cos_sim_max_per_frame: cs_per_frame,
        f0_match_per_frame: fm_per_frame,
        auto_learn_events,
        // TSE wiring through the async-refresh worker is a Phase 5
        // follow-up — the sync `process_offline` rejects
        // `tse.is_some() && async_refresh` up front.
        tse_applied: false,
    })
}

pub(crate) fn fbank_ecapa_one(
    window: &[f32],
    fbank: &mut Fbank,
    ecapa: &mut EcapaTdnn,
) -> Result<Vec<f32>, EmbeddingError> {
    let feats = fbank.compute(window);
    let n_frames = feats.len() / N_MELS;
    ecapa.embed_features(&feats, n_frames, N_MELS)
}

/// f0 sigma fallback used when bootstrap installs the first anchor
/// from a single refresh window. Empirical spread of intra-speaker f0
/// over short utterances; the value isn't critical because subsequent
/// admissions through `adapt` don't touch the F0 metadata — the
/// streaming pipeline's `f0_match` gate just stops being a hard
/// rejector. ~40 Hz is wide enough to keep the bootstrapped speaker
/// admissible.
pub(crate) const BOOTSTRAP_F0_SIGMA_FALLBACK: f32 = 40.0;

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_refresh_result(
    embedding: Vec<f32>,
    f0_mu: f32,
    trigger_frame: usize,
    consecutive_speech_ms: f32,
    pool: &mut EmbeddingPool,
    cohort: &[Vec<f32>],
    gate_cfg: &GateConfig,
    enable_auto_learn: bool,
    score_ema_alpha: f32,
    score: &mut ScoreState,
    auto_learn_events: &mut Vec<AutoLearnEvent>,
) {
    let cs = pool.match_score(&embedding);
    let fm = f0_match(f0_mu, pool.metadata().f0_mu, pool.metadata().f0_sigma);
    score.last_cs = cs;
    score.last_fm = fm;
    let new_score = if gate_cfg.use_as_norm && !cohort.is_empty() {
        as_norm_score(&embedding, cs, cohort, 20)
    } else {
        cs
    };
    score.last_score = smooth_score(score.last_score, new_score, score_ema_alpha);

    if !enable_auto_learn {
        return;
    }

    // Bootstrap path — when the pool is still empty and the
    // bootstrap flag is on, the score / f0 admission guards don't
    // make sense (there's no anchor to score against, and the F0
    // metadata is zeroed). Fall back to "we've heard at least
    // `min_continuous_speech_sec` of voiced speech" as the only
    // condition and seed the first anchor from this refresh.
    if pool.is_empty() && pool.config().allow_bootstrap_from_runtime {
        let min_run_ms = gate_cfg.min_continuous_speech_sec * 1000.0;
        if consecutive_speech_ms >= min_run_ms && f0_mu.is_finite() && f0_mu > 0.0 {
            let seeded = pool.bootstrap_seed(embedding, f0_mu, BOOTSTRAP_F0_SIGMA_FALLBACK);
            if seeded {
                auto_learn_events.push(AutoLearnEvent {
                    frame_idx: trigger_frame,
                    kind: AutoLearnKind::BootstrapSeed,
                    score: score.last_score,
                    f0_match: fm,
                });
            }
        }
        return;
    }

    if should_admit_auto_learn(score.last_score, fm, consecutive_speech_ms, gate_cfg) {
        let admitted = pool.adapt(embedding);
        let kind = if admitted {
            AutoLearnKind::Admit
        } else {
            AutoLearnKind::RejectAnchorDistance
        };
        auto_learn_events.push(AutoLearnEvent {
            frame_idx: trigger_frame,
            kind,
            score: score.last_score,
            f0_match: fm,
        });
    }
}

/// Default ECAPA enrollment chunk length (3 s @ 16 kHz).
pub const ENROLL_CHUNK_SAMPLES: usize = 48_000;
/// Default ECAPA enrollment chunk shift (1.5 s @ 16 kHz).
pub const ENROLL_SHIFT_SAMPLES: usize = 24_000;

/// Build an [`EmbeddingPool`] from a clean enrollment recording.
///
/// Per `docs/gating.md`:
///
/// * 5–10 anchor embeddings via sliding `ENROLL_CHUNK_SAMPLES` /
///   `ENROLL_SHIFT_SAMPLES` chunks
/// * F0 statistics from the voiced portion of the whole recording
///
/// Mirrors `mellonella_poc.pipeline.enroll_from_recording`. Currently
/// assumes the caller hands in 16 kHz audio; a Rust resampler will
/// land in a follow-up.
///
/// # Errors
/// Returns [`EmbeddingError`] when the recording is too short to feed
/// ECAPA at all, or when an inference call fails.
pub fn enroll_from_recording(
    audio: &[f32],
    components: &mut PipelineComponents,
    pool_cfg: EmbeddingPoolConfig,
) -> Result<EmbeddingPool, EmbeddingError> {
    let sv_sr = 16_000_u32;
    let mut anchors: Vec<Vec<f32>> = Vec::new();
    let mut start = 0_usize;
    while start + ENROLL_CHUNK_SAMPLES <= audio.len() {
        let window = &audio[start..start + ENROLL_CHUNK_SAMPLES];
        let feats = components.fbank.compute(window);
        let n_frames = feats.len() / N_MELS;
        anchors.push(components.ecapa.embed_features(&feats, n_frames, N_MELS)?);
        start += ENROLL_SHIFT_SAMPLES;
    }
    if anchors.is_empty() {
        if audio.len() < sv_sr as usize {
            return Err(EmbeddingError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }
        let window = &audio[..ENROLL_CHUNK_SAMPLES.min(audio.len())];
        let feats = components.fbank.compute(window);
        let n_frames = feats.len() / N_MELS;
        anchors.push(components.ecapa.embed_features(&feats, n_frames, N_MELS)?);
    }

    let track = estimate_f0_track(audio, sv_sr, 2048, 512, DEFAULT_F_MIN, DEFAULT_F_MAX);
    let (f0_mu, f0_sigma) = f0_statistics(&track);

    let mut pool = EmbeddingPool::new(pool_cfg);
    pool.add_anchors(anchors);
    pool.set_f0_stats(f0_mu, f0_sigma);
    Ok(pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_to_audio_rate_identity() {
        assert_eq!(scale_to_audio_rate(0, 16_000, 16_000), 0);
        assert_eq!(scale_to_audio_rate(512, 16_000, 16_000), 512);
        assert_eq!(scale_to_audio_rate(1_000_000, 48_000, 48_000), 1_000_000);
    }

    #[test]
    fn scale_to_audio_rate_16k_to_48k_is_exact_triple() {
        // 16 kHz → 48 kHz is the audio-path ratio when the CLI runs
        // the pipeline. Multiplication is exact integer, no rounding.
        assert_eq!(scale_to_audio_rate(0, 16_000, 48_000), 0);
        assert_eq!(scale_to_audio_rate(512, 16_000, 48_000), 1536);
        assert_eq!(scale_to_audio_rate(8000, 16_000, 48_000), 24_000);
    }

    #[test]
    fn scale_to_audio_rate_rounds_half_up() {
        // 16 kHz → 44.1 kHz is non-integer; verify boundary at 0.5
        // rounds up (avoids the always-floor drift bias).
        // 1 * 44100 / 16000 = 2.75625 → 3
        assert_eq!(scale_to_audio_rate(1, 16_000, 44_100), 3);
        // 100 * 44100 / 16000 = 275.625 → 276
        assert_eq!(scale_to_audio_rate(100, 16_000, 44_100), 276);
    }

    #[test]
    fn apply_envelope_dual_rate_identity_matches_apply_envelope() {
        // When audio_sr == decision_sr the helper must be a pure
        // pass-through to `apply_envelope` so existing parity fixtures
        // stay byte-identical.
        let audio = vec![0.5_f32; 1024];
        let decisions = vec![(0_usize, true)];
        let gate_cfg = GateConfig::default();
        let envelope_direct = apply_envelope(&audio, &decisions, 16_000, gate_cfg).expect("direct");
        let envelope_dual =
            apply_envelope_dual_rate(&audio, &decisions, 16_000, 16_000, gate_cfg).expect("dual");
        assert_eq!(envelope_direct, envelope_dual);
    }

    #[test]
    fn apply_envelope_dual_rate_output_length_matches_audio_rate() {
        // 1 s of 48 kHz audio (= 48 000 samples) with a single decision
        // at 16 kHz boundary 0 must yield exactly 48 000 output samples.
        let audio_48k = vec![0.5_f32; 48_000];
        let decisions_16k = vec![(0_usize, true)];
        let gate_cfg = GateConfig::default();
        let out = apply_envelope_dual_rate(&audio_48k, &decisions_16k, 48_000, 16_000, gate_cfg)
            .expect("dual-rate envelope");
        assert_eq!(out.len(), 48_000);
    }

    #[test]
    fn apply_envelope_dual_rate_collapses_rounding_dupes() {
        // Two consecutive decisions whose decision-rate indices map to
        // the same audio-rate sample after rounding (rare but possible
        // for non-integer ratios with very dense decisions) must
        // collapse so the monotonic-boundary precondition of
        // `apply_envelope` holds. Construct an artificial case at a
        // weird ratio.
        let audio = vec![0.5_f32; 64];
        // Decisions at 0 and 1 sample @ 16 kHz, output rate 1 kHz: both
        // map to sample 0. After dedup we keep the first only and
        // `apply_envelope` accepts the input.
        let decisions = vec![(0_usize, false), (1_usize, true)];
        let gate_cfg = GateConfig::default();
        let out = apply_envelope_dual_rate(&audio, &decisions, 1_000, 16_000, gate_cfg)
            .expect("dual-rate envelope tolerates collapsed boundaries");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn shift_off_on_noop_when_shift_is_zero() {
        let mut d = vec![(0_usize, false), (1000, true), (2000, false)];
        let snapshot = d.clone();
        shift_off_on_decisions_back(&mut d, 0);
        assert_eq!(d, snapshot);
    }

    #[test]
    fn shift_off_on_moves_only_on_boundaries() {
        // OFF at 0, ON at 1000, OFF at 2000 → shift 300 → ON moves to
        // 700, the OFFs stay put.
        let mut d = vec![(0_usize, false), (1000, true), (2000, false)];
        shift_off_on_decisions_back(&mut d, 300);
        assert_eq!(d, vec![(0, false), (700, true), (2000, false)]);
    }

    #[test]
    fn shift_off_on_floors_at_prev_boundary() {
        // ON at 500 with shift 1000 would underflow into the prior OFF
        // region (0..500). The shift floors at 0 (the previous
        // boundary), and the collapsed OFF entry at 0 is removed.
        let mut d = vec![(0_usize, false), (500, true)];
        shift_off_on_decisions_back(&mut d, 1000);
        assert_eq!(d, vec![(0, true)]);
    }

    #[test]
    fn shift_off_on_collapses_zero_length_runs() {
        // OFF at 0, ON at 500, OFF at 600, ON at 800 → shift 200 →
        // second ON lands on 600 (== prior OFF). The OFF disappears.
        let mut d = vec![(0_usize, false), (500, true), (600, false), (800, true)];
        shift_off_on_decisions_back(&mut d, 200);
        assert_eq!(d, vec![(0, false), (300, true), (600, true)]);
    }

    #[test]
    fn pre_roll_samples_decision_matches_default() {
        // pre_roll_ms = 100 at 16 kHz = 1600 samples.
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.pre_roll_ms, 100);
        assert_eq!(cfg.pre_roll_samples_decision(), 1_600);
    }

    #[test]
    fn audio_for_decisions_borrows_when_rates_match() {
        let audio = vec![0.1_f32, 0.2, 0.3];
        let cow = audio_for_decisions(&audio, 16_000, 16_000).expect("identity");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), audio.as_slice());
    }

    #[test]
    fn audio_for_decisions_resamples_when_rates_differ() {
        // 1 s @ 48 kHz of silence → roughly 16 kHz worth of samples after
        // the windowed-sinc downsampler. The exact length is bounded by
        // `resample_to`'s tests; we just verify the borrow/owned split.
        let audio = vec![0.0_f32; 48_000];
        let cow = audio_for_decisions(&audio, 48_000, 16_000).expect("downsample");
        assert!(matches!(cow, Cow::Owned(_)));
        let dec = cow.as_ref();
        let expected = 16_000_i32;
        let slack = expected / 100;
        let got = dec.len() as i32;
        assert!(
            (got - expected).abs() <= slack,
            "expected ≈ {expected} samples (±{slack}), got {got}"
        );
    }

    fn unit_vec(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn apply_refresh_result_bootstrap_seeds_first_anchor() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: true,
            ..EmbeddingPoolConfig::default()
        });
        let gate_cfg = GateConfig::default();
        let mut score = ScoreState::new();
        let mut events: Vec<AutoLearnEvent> = Vec::new();
        let consecutive_speech_ms = gate_cfg.min_continuous_speech_sec * 1000.0;

        apply_refresh_result(
            unit_vec(&[1.0, 0.0, 0.0]),
            150.0, // f0_mu — non-zero so the voiced guard passes
            42,
            consecutive_speech_ms,
            &mut pool,
            &[],
            &gate_cfg,
            true,
            1.0,
            &mut score,
            &mut events,
        );

        assert_eq!(pool.anchors().len(), 1);
        assert!((pool.metadata().f0_mu - 150.0).abs() < 1e-5);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, AutoLearnKind::BootstrapSeed));
    }

    #[test]
    fn apply_refresh_result_bootstrap_disabled_by_default() {
        // Without the opt-in flag, an empty pool emits no events and
        // adds no anchors regardless of how good the input looks.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let gate_cfg = GateConfig::default();
        let mut score = ScoreState::new();
        let mut events: Vec<AutoLearnEvent> = Vec::new();

        apply_refresh_result(
            unit_vec(&[1.0, 0.0, 0.0]),
            150.0,
            0,
            10_000.0,
            &mut pool,
            &[],
            &gate_cfg,
            true,
            1.0,
            &mut score,
            &mut events,
        );

        assert!(pool.is_empty());
        assert!(events.is_empty());
    }

    #[test]
    fn apply_refresh_result_bootstrap_requires_voiced_frame() {
        // Bootstrap mustn't seed off an unvoiced refresh (f0_mu == 0
        // or NaN), otherwise breath / background noise during a long
        // VAD-positive stretch would set the wrong anchor.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: true,
            ..EmbeddingPoolConfig::default()
        });
        let gate_cfg = GateConfig::default();
        let mut score = ScoreState::new();
        let mut events: Vec<AutoLearnEvent> = Vec::new();

        apply_refresh_result(
            unit_vec(&[1.0, 0.0, 0.0]),
            0.0,
            0,
            10_000.0,
            &mut pool,
            &[],
            &gate_cfg,
            true,
            1.0,
            &mut score,
            &mut events,
        );

        assert!(pool.is_empty());
        assert!(events.is_empty());
    }

    #[test]
    fn apply_refresh_result_bootstrap_requires_min_speech_run() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            allow_bootstrap_from_runtime: true,
            ..EmbeddingPoolConfig::default()
        });
        let gate_cfg = GateConfig::default();
        let mut score = ScoreState::new();
        let mut events: Vec<AutoLearnEvent> = Vec::new();
        let too_short = gate_cfg.min_continuous_speech_sec * 1000.0 - 1.0;

        apply_refresh_result(
            unit_vec(&[1.0, 0.0, 0.0]),
            150.0,
            0,
            too_short,
            &mut pool,
            &[],
            &gate_cfg,
            true,
            1.0,
            &mut score,
            &mut events,
        );

        assert!(pool.is_empty());
        assert!(events.is_empty());
    }
}
