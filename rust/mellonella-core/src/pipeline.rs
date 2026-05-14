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

use std::collections::VecDeque;

use std::borrow::Cow;

use crate::embedding::{EcapaTdnn, EmbeddingError};
use crate::enrollment::{EmbeddingPool, EmbeddingPoolConfig};
use crate::f0::{estimate_f0_track, f0_statistics, DEFAULT_F_MAX, DEFAULT_F_MIN};
use crate::features::{Fbank, N_MELS};
use crate::gating::{
    apply_envelope, as_norm_score, f0_match, should_admit_auto_learn, ApplyEnvelopeError,
    GateConfig, GateState,
};
use crate::resample::{resample_to, ResampleError};
use crate::streaming::{StreamingConfig, StreamingState};
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
}

/// Errors returned by [`process_offline`].
#[derive(Debug)]
pub enum PipelineError {
    Embedding(EmbeddingError),
    Envelope(ApplyEnvelopeError),
    Resample(ResampleError),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedding(e) => write!(f, "embedding error: {e}"),
            Self::Envelope(e) => write!(f, "envelope error: {e}"),
            Self::Resample(e) => write!(f, "resample error: {e}"),
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Embedding(e) => Some(e),
            Self::Envelope(_) => None,
            Self::Resample(e) => Some(e),
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
    if pipeline_cfg.async_refresh {
        return process_offline_async(
            audio,
            audio_sample_rate,
            pool,
            pipeline_cfg,
            gate_cfg,
            components,
        );
    }
    // Sync offline = streaming engine called once with the whole
    // buffer, then flushed. Per-frame outputs match the previous
    // monolithic implementation byte-for-byte at identity rate
    // (`streaming_identity_rate_per_frame_matches_offline`).
    let cfg = StreamingConfig {
        pipeline: *pipeline_cfg,
        gate: *gate_cfg,
        audio_sample_rate,
        diagnostics: true,
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

    Ok(ProcessResult {
        audio: head.audio,
        gate_decisions: decisions,
        gate_per_frame: head.gate_per_frame,
        score_per_frame: head.score_per_frame,
        cos_sim_max_per_frame: head.cos_sim_max_per_frame,
        f0_match_per_frame: head.f0_match_per_frame,
        auto_learn_events: head.events,
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
    let dt_ms = pipeline_cfg.vad_frame_ms();

    let decision_audio = audio_for_decisions(audio, audio_sample_rate, sv_sr)?;
    let audio_dec: &[f32] = decision_audio.as_ref();

    let mut speech_buffer: VecDeque<f32> = VecDeque::with_capacity(pipeline_cfg.sv_window_samples);
    let pre_roll_capacity = pipeline_cfg.pre_roll_samples_decision();
    let mut pre_roll_ring: VecDeque<f32> = VecDeque::with_capacity(pre_roll_capacity);
    let mut samples_since_update = 0_usize;
    let mut silence_seen_since_refresh = false;
    let mut new_speech_samples_after_silence = 0_usize;
    let mut prev_speech = false;
    let mut consecutive_speech_ms = 0.0_f32;
    let mut silence_ms_since_speech = 0.0_f32;
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

    // Disjoint borrows of PipelineComponents so the worker thread can
    // own ecapa + fbank while the main loop keeps vad. The `cohort` is
    // read-only and shared by-reference.
    let PipelineComponents {
        vad,
        fbank,
        ecapa,
        cohort,
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

        // outstanding = number of work_tx sends not yet matched by a
        // res_rx recv. Bench cadence + 2 s audio → worst case
        // outstanding stays at 1; pending holds one queued window so a
        // burst of two refreshes within one ECAPA wall time doesn't
        // drop work.
        let mut outstanding: u32 = 0;
        let mut pending: Option<Vec<f32>> = None;
        // Frame index of each refresh, in FIFO order, so AutoLearnEvent
        // gets the trigger-time index rather than the result-arrival
        // time.
        let mut refresh_frame_indices: VecDeque<usize> = VecDeque::new();

        let mut frame_start = 0_usize;
        let mut frame_idx = 0_usize;

        while frame_start + vad_frame <= audio_dec.len() {
            let frame = &audio_dec[frame_start..frame_start + vad_frame];

            let speech_prob = vad.score(frame)?;
            let now_speech = speech_prob > pipeline_cfg.vad_threshold;
            if now_speech {
                if !prev_speech && !pre_roll_ring.is_empty() {
                    // VAD OFF→ON: fold the pre-roll ring into the
                    // speech buffer (issue #80). Same semantics as the
                    // sync streaming engine.
                    for &sample in &pre_roll_ring {
                        if speech_buffer.len() == pipeline_cfg.sv_window_samples {
                            speech_buffer.pop_front();
                        }
                        speech_buffer.push_back(sample);
                    }
                }
                for &sample in frame {
                    if speech_buffer.len() == pipeline_cfg.sv_window_samples {
                        speech_buffer.pop_front();
                    }
                    speech_buffer.push_back(sample);
                }
                consecutive_speech_ms += dt_ms;
                if silence_seen_since_refresh {
                    new_speech_samples_after_silence += vad_frame;
                }
            } else {
                consecutive_speech_ms = 0.0;
            }
            if prev_speech && !now_speech {
                silence_seen_since_refresh = true;
                new_speech_samples_after_silence = 0;
            }
            if now_speech {
                silence_ms_since_speech = 0.0;
            } else {
                silence_ms_since_speech += dt_ms;
            }
            prev_speech = now_speech;
            samples_since_update += vad_frame;

            if pre_roll_capacity > 0 {
                for &sample in frame {
                    if pre_roll_ring.len() == pre_roll_capacity {
                        pre_roll_ring.pop_front();
                    }
                    pre_roll_ring.push_back(sample);
                }
            }

            let due_normal = samples_since_update >= pipeline_cfg.sv_update_samples;
            let due_early = silence_seen_since_refresh
                && now_speech
                && new_speech_samples_after_silence
                    >= pipeline_cfg.sv_min_new_samples_after_silence;
            if (due_normal || due_early) && speech_buffer.len() >= pipeline_cfg.sv_window_samples {
                samples_since_update = 0;
                silence_seen_since_refresh = false;
                new_speech_samples_after_silence = 0;
                let window: Vec<f32> = speech_buffer.iter().copied().collect();
                if outstanding == 0 {
                    work_tx.send(window).expect("worker alive");
                    outstanding = 1;
                    refresh_frame_indices.push_back(frame_idx);
                } else {
                    pending = Some(window);
                    refresh_frame_indices.push_back(frame_idx);
                }
            }

            // Drain at most one ready result per frame so the gate
            // score updates as soon as the worker is done.
            if outstanding > 0 {
                match res_rx.try_recv() {
                    Ok(res) => {
                        let (embedding, f0_mu) = res?;
                        let trigger_frame = refresh_frame_indices.pop_front().unwrap_or(frame_idx);
                        apply_refresh_result(
                            embedding,
                            f0_mu,
                            trigger_frame,
                            consecutive_speech_ms,
                            pool,
                            cohort,
                            gate_cfg,
                            pipeline_cfg.enable_auto_learn,
                            pipeline_cfg.score_ema_alpha,
                            &mut last_score,
                            &mut last_cs,
                            &mut last_fm,
                            &mut auto_learn_events,
                        );
                        outstanding -= 1;
                        if let Some(next) = pending.take() {
                            work_tx.send(next).expect("worker alive");
                            outstanding = 1;
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return Err(PipelineError::Embedding(EmbeddingError::Ort(
                            "ECAPA worker disconnected".into(),
                        )));
                    }
                }
            }

            let is_on_score = gate_state.update(last_score, dt_ms);
            let is_on = is_on_score
                && !(pipeline_cfg.silence_force_off_ms > 0.0
                    && silence_ms_since_speech >= pipeline_cfg.silence_force_off_ms);
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

        // Drain whatever ECAPA work is still in flight so the result
        // log is complete before we tear down the worker. Per-frame
        // outputs were already emitted for the main loop above, so
        // these only affect pool / auto_learn_events.
        while outstanding > 0 {
            let res = res_rx.recv().map_err(|_| {
                PipelineError::Embedding(EmbeddingError::Ort("ECAPA worker disconnected".into()))
            })?;
            let (embedding, f0_mu) = res?;
            let trigger_frame = refresh_frame_indices.pop_front().unwrap_or(frame_idx);
            apply_refresh_result(
                embedding,
                f0_mu,
                trigger_frame,
                consecutive_speech_ms,
                pool,
                cohort,
                gate_cfg,
                pipeline_cfg.enable_auto_learn,
                pipeline_cfg.score_ema_alpha,
                &mut last_score,
                &mut last_cs,
                &mut last_fm,
                &mut auto_learn_events,
            );
            outstanding -= 1;
            if let Some(next) = pending.take() {
                work_tx.send(next).expect("worker alive");
                outstanding = 1;
            }
        }

        // Closing the work channel lets the worker exit so the scope
        // can join it.
        drop(work_tx);
        Ok(())
    })?;

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
    last_score: &mut f32,
    last_cs: &mut f32,
    last_fm: &mut f32,
    auto_learn_events: &mut Vec<AutoLearnEvent>,
) {
    let cs = pool.match_score(&embedding);
    let fm = f0_match(f0_mu, pool.metadata().f0_mu, pool.metadata().f0_sigma);
    *last_cs = cs;
    *last_fm = fm;
    let new_score = if gate_cfg.use_as_norm && !cohort.is_empty() {
        as_norm_score(&embedding, cs, cohort, 20)
    } else {
        cs
    };
    *last_score = smooth_score(*last_score, new_score, score_ema_alpha);

    if enable_auto_learn
        && should_admit_auto_learn(*last_score, fm, consecutive_speech_ms, gate_cfg)
    {
        let admitted = pool.adapt(embedding);
        let kind = if admitted {
            AutoLearnKind::Admit
        } else {
            AutoLearnKind::RejectAnchorDistance
        };
        auto_learn_events.push(AutoLearnEvent {
            frame_idx: trigger_frame,
            kind,
            score: *last_score,
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
}
