//! Streaming / online pipeline API — stateful counterpart of
//! [`crate::pipeline::process_offline`].
//!
//! Intended for live microphone use, GUI integrations, and any
//! caller that produces samples incrementally rather than as one
//! contiguous buffer. Runs alongside the offline
//! [`crate::pipeline::process_offline`] (step 8) — a follow-up
//! step (10) will unify them so the offline path becomes a thin
//! wrapper over this engine.
//!
//! Inherits the **dual-rate split** from `process_offline` (Phase
//! 3.5 step 8): callers push audio at
//! `StreamingConfig::audio_sample_rate` (default 48 kHz, full-band
//! quality), the pipeline resamples internally to 16 kHz for the
//! VAD / ECAPA / F0 decision path, and emits envelope-gated audio
//! at the input rate.
//!
//! # Buffering model
//!
//! * **Input granularity**: any length, any cadence, at
//!   `StreamingConfig::audio_sample_rate`. Callers may push
//!   1-sample or 100 000-sample chunks and the pipeline does the
//!   right thing.
//! * **Internal alignment**: VAD frames are 512 samples (32 ms @
//!   16 kHz, [`crate::vad::CHUNK_SAMPLES_16K`]). The audio is
//!   resampled to 16 kHz on the decision path via a stateful
//!   `rubato::SincFixedOut`; sub-VAD-frame audio-rate residue is
//!   held in an internal ring buffer until enough samples
//!   accumulate to drive one more VAD frame.
//! * **Output granularity**: a multiple of one VAD frame's
//!   audio-rate equivalent per `push_samples` call (e.g. 1536
//!   samples @ 48 kHz, the integer scaling of 512 @ 16 kHz). When
//!   rates don't divide evenly the resampler determines the exact
//!   audio-rate input count per VAD frame via `input_frames_next()`.
//! * **Flush**: [`StreamingPipeline::flush`] zero-pads any residual
//!   audio-rate samples to the resampler's expected input size so
//!   the trailing audio gets one last decision pass.
//!
//! # Algorithm parity with `process_offline`
//!
//! With `StreamingConfig::pipeline.async_refresh == false` and
//! `audio_sample_rate == pipeline.sample_rate` (identity rate, no
//! internal resample), `StreamingPipeline::new → push_samples(audio)
//! → flush` produces per-VAD-frame `gate_per_frame` and
//! `score_per_frame` identical to `process_offline(audio, …)`.
//! Verified by the `streaming_smoke` integration test
//! (`streaming_identity_rate_per_frame_matches_offline`), gated on
//! the same ONNX env vars as the existing parity fixtures.
//!
//! At dual-rate (`audio_sample_rate != pipeline.sample_rate`), the
//! streaming engine uses a stateful `SincFixedOut` resampler and
//! `process_offline` uses a one-shot `resample_to`; outputs will
//! differ slightly because the two resamplers have different
//! startup-delay characteristics. A follow-up step will unify the
//! offline path to use the streaming engine.
//!
//! # Async refresh
//!
//! `StreamingPipeline` supports both sync (`async_refresh = false`)
//! and async (`async_refresh = true`) modes:
//!
//! * **Sync**: Fbank / ECAPA / F0 run inline on the caller's thread
//!   inside `push_samples`. Simple but blocks for ~30–50 ms per
//!   refresh.
//! * **Async**: at construction time, `fbank` + `ecapa` are moved
//!   into a persistent worker thread; the main thread sends speech
//!   windows over a channel and reads back `(embedding, f0_mu)`
//!   results, applying them via `apply_refresh_result` on the
//!   next frame after they arrive. Mirrors the cadence model of
//!   `process_offline_async` (at most one inference outstanding,
//!   one queued window so a burst doesn't drop work).
//!
//! `into_parts` joins the worker (waking it via channel close)
//! and reconstructs the original [`PipelineComponents`] from the
//! moved `fbank` + `ecapa` plus the main-thread `vad` + `cohort`.
//! `process_offline` still uses its own dedicated
//! `process_offline_async` for one-shot async runs; rewiring that
//! to `StreamingPipeline::new + push_samples + flush` is a
//! follow-up step.
//!
//! # Ownership
//!
//! `StreamingPipeline::new` takes ownership of the
//! [`crate::enrollment::EmbeddingPool`] and
//! [`crate::pipeline::PipelineComponents`]. They can be recovered
//! on shutdown via [`StreamingPipeline::into_parts`] — useful for
//! persisting the post-run pool (auto-learn updates) and for
//! tearing the ONNX sessions down deterministically.

#![allow(clippy::too_many_arguments, clippy::too_many_lines)]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;

use rubato::{
    Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::dfn3::Dfn3Streamer;
use crate::embedding::{EcapaTdnn, EmbeddingError};
use crate::enrollment::EmbeddingPool;
use crate::f0::{
    estimate_f0_track, f0_statistics, yin_frame_with_cache, FftCache, DEFAULT_F_MAX, DEFAULT_F_MIN,
    DEFAULT_THRESHOLD,
};
use crate::features::{Fbank, N_MELS};
use crate::gating::{
    as_norm_score, f0_match, should_admit_auto_learn, EnvelopeState, GateConfig, GateState,
};
use crate::pipeline::{
    apply_refresh_result, fbank_ecapa_one, smooth_score, tse_cond_embedding, AutoLearnEvent,
    AutoLearnKind, PipelineComponents, PipelineConfig, PipelineError, ScoreState,
};
use crate::tse_stage::TseStage;
use crate::vad::{SileroVad, CHUNK_SAMPLES_16K};

/// Pair returned by [`StreamingState::drain_one_frame`]: the
/// audio-rate samples consumed and the matching decision-rate VAD
/// frame. Aliased to satisfy `clippy::type_complexity`.
type DrainedFrame = (Vec<f32>, Vec<f32>);

/// Configuration for [`StreamingPipeline`].
///
/// `pipeline` and `gate` are the same structs the offline pipeline
/// consumes; `audio_sample_rate` and `diagnostics` are streaming-
/// specific.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Pipeline-side cadence (window / refresh / VAD threshold /
    /// auto-learn switch / async refresh). `pipeline.sample_rate`
    /// is the **decision** rate (16 kHz) — see the module
    /// "Buffering model" doc for the dual-rate split.
    pub pipeline: PipelineConfig,
    /// Gate-side parameters (hangover, attack/release, score
    /// threshold, F0 weight).
    pub gate: GateConfig,
    /// Sample rate of the audio path — the rate the caller pushes
    /// into [`StreamingPipeline::push_samples`] and the rate the
    /// returned envelope-gated audio is at. Default 48 000 Hz
    /// (DFN3's native, full-band rate, per the
    /// `docs/architecture.md` Sampling-rate policy). The pipeline
    /// resamples internally to `pipeline.sample_rate` (16 kHz) for
    /// the decision path.
    pub audio_sample_rate: u32,
    /// When `true`, [`StreamingOutput`] populates
    /// `gate_per_frame` / `score_per_frame` /
    /// `cos_sim_max_per_frame` / `f0_match_per_frame`. Default
    /// `false` to avoid the per-call allocation in the live path —
    /// the GUI can opt in only while a "live status" panel is open.
    pub diagnostics: bool,
    /// **Stage C, Phase 5** — opt-in DFN3 noise suppression on the
    /// audio-rate output path. When `Some`, the streaming engine
    /// loads the stateful DFN3 ONNX at the supplied path and routes
    /// the audio chain as `mic → TSE? → DFN3 → envelope → out`.
    /// Runs **after** TSE so DFN3 cleans up residual noise in the
    /// extracted target voice (the empirical TSE→DFN3 ordering beats
    /// DFN3→TSE by ~1-3 dB SI-SDR; see Phase 5 ordering experiment).
    ///
    /// Decision-rate features (VAD, ECAPA, F0 score, auto-learn) still
    /// observe the raw input audio so the SV path remains unaffected.
    ///
    /// `audio_sample_rate` must equal the DFN3 model rate (48 kHz).
    /// Default `None` — no NS, byte-identical to pre-Phase-5 behaviour.
    pub dfn3_onnx_path: Option<PathBuf>,
    /// Optional pyannote-3.0 segmentation ONNX path. When set, the
    /// streaming engine runs a per-second overlap detector alongside
    /// the audio chain and **toggles TSE vs. DFN3 mutually exclusively**:
    ///
    /// * solo speaker detected (mean overlap probability below
    ///   [`Self::overlap_threshold`] for [`Self::overlap_hold_off_ms`] ms):
    ///   chain runs **DFN3 only**, TSE is bypassed. Empirically the
    ///   solo-speaker DFN3-only path attenuates output by ≈ 8 dB,
    ///   versus ≈ 28 dB for the TSE+DFN3 cascade.
    /// * overlap detected (mean overlap probability above
    ///   `overlap_threshold` for [`Self::overlap_hold_on_ms`] ms):
    ///   chain runs **TSE only**, DFN3 is bypassed. Target-speaker
    ///   extraction acts as its own (mask-based) noise / interference
    ///   suppressor in this regime.
    ///
    /// Both `tse` (under `pipeline.tse`) and `dfn3_onnx_path` must
    /// also be configured for the toggle to have anything to switch
    /// between; with the overlap ONNX absent (or `tse` / `dfn3_onnx_path`
    /// unset) the engine keeps its legacy cascade behaviour.
    pub overlap_onnx_path: Option<PathBuf>,
    /// Mean per-frame overlap probability above which the engine
    /// considers two or more speakers active. Default `0.10` — sits
    /// comfortably between empirical solo (≲ 0.03) and overlap
    /// (≳ 0.30) cohorts on the pyannote-3.0 community ONNX.
    pub overlap_threshold: f32,
    /// Hysteresis ON: how long the mean overlap probability has to
    /// stay above [`Self::overlap_threshold`] before the chain
    /// transitions Solo → Overlap. Default `500 ms` — long enough to
    /// ignore single-decision blips, short enough to react to a real
    /// second voice within ~ a syllable.
    pub overlap_hold_on_ms: f32,
    /// Hysteresis OFF: how long the mean overlap probability has to
    /// stay below [`Self::overlap_threshold`] before transitioning
    /// Overlap → Solo. Default `2000 ms` — biased long so a brief
    /// interlocutor pause doesn't flap the chain back through a
    /// reset cycle.
    pub overlap_hold_off_ms: f32,
    /// Post-chain makeup gain (dB) applied while
    /// [`crate::streaming::ChainMode::Solo`] is active. The DFN3-only
    /// solo path attenuates real speech by ~4-6 dB RMS as a side
    /// effect of denoising; recovering the level on the output side
    /// is the standard "NS → AGC" pattern every consumer voice stack
    /// (Krisp / Teams / Zoom) uses. Static gain — biased low so a
    /// residual-noise burst during quiet sections doesn't get
    /// audibly amplified the way an AGC would.
    ///
    /// Default `5.0 dB`. Set `0.0` to disable. Applied **before** the
    /// envelope multiply, so gate-off frames stay silent regardless.
    pub makeup_gain_db_solo: f32,
    /// Post-chain makeup gain (dB) applied while
    /// [`crate::streaming::ChainMode::Overlap`] is active. Conv-TasNet
    /// style separators output a scale-arbitrary signal that's
    /// typically ~10 dB below the input mixture's RMS; a future
    /// phase will replace this static knob with input-RMS matching
    /// (Asteroid's canonical fix). Until then, the static value
    /// covers the typical case.
    ///
    /// Default `0.0 dB` — Overlap path stays unaltered in this
    /// phase. Phase-2 work will wire RMS-matching here.
    pub makeup_gain_db_overlap: f32,
    /// Post-makeup soft-saturation curve enable. When `true`, output
    /// samples are passed through `tanh` before the envelope
    /// multiply, so a peak that exceeds full-scale (because the
    /// makeup gain pushed it there) saturates smoothly instead of
    /// being clipped at the DAC. tanh is monotonic and adds
    /// 3rd-harmonic distortion only when the magnitude is already
    /// past ~0.5, so quiet audio is effectively unchanged.
    ///
    /// Default `true`. The soft-clip is cheap (~5 ns / sample on
    /// modern CPUs) and the audible artefact of a hard clip on a
    /// real plosive is much worse than the soft-saturation it
    /// replaces.
    pub chain_soft_clip: bool,
    /// **Phase 2.** When `true` and the chain is in
    /// [`crate::streaming::ChainMode::Overlap`], the TSE-only output
    /// chunk is rescaled so its RMS matches the corresponding raw-
    /// input chunk's RMS (Asteroid's canonical Conv-TasNet rescale).
    /// This is empirically a much better fix for the ~10 dB
    /// Conv-TasNet loss than a static gain — separator outputs are
    /// scale-arbitrary, so a fixed offset over-amplifies quiet
    /// stretches and under-amplifies loud ones.
    ///
    /// Solo mode ignores this flag and uses
    /// [`Self::makeup_gain_db_solo`] (DFN3's loss is fairly
    /// uniform, so a static offset is fine there).
    ///
    /// Default `true`. Set `false` to fall back to
    /// [`Self::makeup_gain_db_overlap`].
    pub overlap_rms_match: bool,
    /// Hard cap on the RMS-match scale factor expressed in dB. The
    /// raw ratio `in_rms / out_rms` can blow up during stretches
    /// where TSE attenuates aggressively (e.g. a target-speaker
    /// pause in the middle of an overlap) — clamping keeps a
    /// silent TSE chunk from getting boosted into pure noise.
    ///
    /// Default `+20 dB` (10× linear). The soft-clip after the
    /// match catches the rare case where this isn't enough.
    pub overlap_rms_match_max_gain_db: f32,
    /// **Phase 2.** Wet / dry mix applied while in Overlap mode:
    /// `out = α · clean + (1 − α) · raw`. Adds a small amount of
    /// the original mixture back into the TSE output to mask
    /// Conv-TasNet's frame-to-frame mask artefacts (the warble
    /// the user reported). Industry survey of commercial
    /// separators converges on values in the 0.85-0.9 range.
    ///
    /// Default `0.9` — overwhelmingly clean, just enough raw to
    /// fill in the spectral cracks. Set `1.0` to disable.
    pub overlap_wet_dry_alpha: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            gate: GateConfig::default(),
            audio_sample_rate: 48_000,
            diagnostics: false,
            dfn3_onnx_path: None,
            overlap_onnx_path: None,
            overlap_threshold: 0.10,
            overlap_hold_on_ms: 500.0,
            overlap_hold_off_ms: 2_000.0,
            makeup_gain_db_solo: 5.0,
            makeup_gain_db_overlap: 0.0,
            chain_soft_clip: true,
            overlap_rms_match: true,
            overlap_rms_match_max_gain_db: 20.0,
            overlap_wet_dry_alpha: 0.9,
        }
    }
}

/// Output of a single [`StreamingPipeline::push_samples`] or
/// [`StreamingPipeline::flush`] call.
///
/// All vectors describe **only what was produced by this call** —
/// they are *not* cumulative across calls. Callers writing to a sink
/// (audio device, WAV file, GUI ring) should consume `audio` every
/// call and forget it; the pipeline does not retain a copy.
///
/// `gate_per_frame` / `score_per_frame` / `cos_sim_max_per_frame` /
/// `f0_match_per_frame` are populated only when
/// [`StreamingConfig::diagnostics`] is `true`.
#[derive(Debug, Default, Clone)]
pub struct StreamingOutput {
    /// Envelope-gated audio at the pipeline's configured
    /// [`StreamingConfig::audio_sample_rate`] (default 48 kHz),
    /// mono, for the just-pushed chunk. Length is a multiple of one
    /// VAD frame's audio-rate equivalent (e.g. 1536 samples @
    /// 48 kHz, the integer scaling of
    /// [`crate::vad::CHUNK_SAMPLES_16K`] = 512 @ 16 kHz).
    pub audio: Vec<f32>,
    /// Audio-rate `(start_sample, is_on)` decisions consumed by the
    /// envelope. Indices are *cumulative since
    /// [`StreamingPipeline::new`] / `reset`*, so callers stitching
    /// successive outputs share a coherent timeline.
    pub gate_decisions: Vec<(usize, bool)>,
    /// Auto-learn admission / rejection / reset events that
    /// occurred during this call, in chronological order. `frame_idx`
    /// is cumulative since `new` / `reset`.
    pub events: Vec<AutoLearnEvent>,
    /// Per-VAD-frame gate state. Empty unless
    /// [`StreamingConfig::diagnostics`] is true.
    pub gate_per_frame: Vec<bool>,
    /// Per-VAD-frame integrated score. Empty unless diagnostics on.
    pub score_per_frame: Vec<f32>,
    /// Per-VAD-frame `cos_sim_max`. Empty unless diagnostics on.
    pub cos_sim_max_per_frame: Vec<f32>,
    /// Per-VAD-frame F0 match. Empty unless diagnostics on.
    pub f0_match_per_frame: Vec<f32>,
    /// **Stage C, Phase 5 part 3** — `true` when the streaming engine
    /// actually routed audio through a [`TseStage`] before applying the
    /// envelope on this call. Mirrors the offline
    /// [`crate::pipeline::ProcessResult::tse_applied`] flag so tests can
    /// assert the default-OFF / opt-in behaviour. Always `false` when
    /// `StreamingConfig::pipeline.tse` is `None`.
    pub tse_applied: bool,
}

/// Decision-rate sample window length the per-frame fast-F0 cue runs
/// YIN over. 2048 @ 16 kHz ≈ 128 ms — long enough for a stable YIN
/// estimate down to [`DEFAULT_F_MIN`] (50 Hz → τ = 320), short enough
/// that the cue reacts within ~4 VAD frames of a speaker turn.
pub(crate) const FAST_F0_RING_SAMPLES: usize = 2048;

/// **Stage B, Part 1** — per-VAD-frame instantaneous F0 estimator.
///
/// Maintains a ring of the last [`FAST_F0_RING_SAMPLES`]
/// decision-rate samples; once the ring is full, every VAD frame runs
/// a single `yin_frame_with_cache` over it. This is the *fast* cue —
/// it reacts within a VAD frame, unlike the batched F0 inside the
/// ECAPA refresh which only updates every `sv_update_samples`.
///
/// Returns `None` ("no evidence") while the ring is still filling or
/// when YIN returns unvoiced / non-finite; callers map a `Some`
/// estimate through `f0_match` and treat `None` as the configured
/// neutral value (and skip the turn-detection EMAs).
pub(crate) struct FastF0Cue {
    /// Decision-rate sample ring, capped at [`FAST_F0_RING_SAMPLES`].
    ring: VecDeque<f32>,
    /// Contiguous staging slice handed to YIN (reused per frame).
    scratch: Vec<f32>,
    /// FFT planner + scratch reuse across per-frame YIN calls.
    cache: FftCache,
}

impl FastF0Cue {
    pub(crate) fn new() -> Self {
        Self {
            ring: VecDeque::with_capacity(FAST_F0_RING_SAMPLES),
            scratch: Vec::with_capacity(FAST_F0_RING_SAMPLES),
            cache: FftCache::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.ring.clear();
        self.scratch.clear();
    }

    /// Push one decision-rate VAD frame into the ring and, once the
    /// ring is full, return the instantaneous F0 estimate over it.
    /// `None` until the ring fills, and `None` for unvoiced frames.
    pub(crate) fn push_frame(&mut self, decision_frame: &[f32], sample_rate: u32) -> Option<f32> {
        for &s in decision_frame {
            if self.ring.len() == FAST_F0_RING_SAMPLES {
                self.ring.pop_front();
            }
            self.ring.push_back(s);
        }
        if self.ring.len() < FAST_F0_RING_SAMPLES {
            return None;
        }
        self.scratch.clear();
        self.scratch.extend(self.ring.iter().copied());
        yin_frame_with_cache(
            &self.scratch,
            sample_rate,
            DEFAULT_F_MIN,
            DEFAULT_F_MAX,
            DEFAULT_THRESHOLD,
            None,
            &mut self.cache,
        )
    }
}

/// **Stage B, Part 1** — fuse the per-frame fast F0 cue into the
/// gate-feeding score.
///
/// `fused = last_score + weight * (fm_fast - neutral)`. Pure /
/// `#[cfg(test)]`-friendly so the fusion arithmetic can be checked
/// without ONNX. With `fm_fast == neutral` (the "no evidence" case)
/// this is the identity — `fused == last_score` — which is also why
/// the disabled path is byte-identical to today.
#[must_use]
pub(crate) fn fuse_fast_cue(last_score: f32, fm_fast: f32, weight: f32, neutral: f32) -> f32 {
    last_score + weight * (fm_fast - neutral)
}

/// Sub-LSB dither amplitude added unconditionally to the chain
/// input. ±1e-7 ≈ -140 dBFS, below the 24-bit noise floor (~-138
/// dBFS) and inaudible. Applied to every sample so no STFT window
/// inside the chain can be exactly zero-variance — a single
/// all-zero window is enough to latch TSE's GRU hidden state into
/// a NaN it never recovers from. Real audio is altered by an
/// inaudible amount.
const CHAIN_DITHER: f32 = 1e-7;

/// One-shot NaN diagnostic — logs the first time `samples` contains a
/// NaN under the given label, then suppresses further reports. The
/// suppression is per `(label, kind)` pair (kind ∈ {NaN, Inf}) so each
/// diagnostic site reports independently.
fn log_first_nan(label: &'static str, samples: &[f32]) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, &'static str)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let (nan_idx, inf_idx) =
        samples
            .iter()
            .enumerate()
            .fold((None::<usize>, None::<usize>), |(n, i), (idx, x)| {
                (
                    n.or_else(|| x.is_nan().then_some(idx)),
                    i.or_else(|| (!x.is_finite() && !x.is_nan()).then_some(idx)),
                )
            });
    if let Some(idx) = nan_idx {
        if let Ok(mut guard) = seen.lock() {
            if guard.insert((label, "nan")) {
                eprintln!(
                    "[diag] first NaN in `{label}` at index {idx} / {len} samples",
                    len = samples.len(),
                );
            }
        }
    }
    if let Some(idx) = inf_idx {
        if let Ok(mut guard) = seen.lock() {
            if guard.insert((label, "inf")) {
                eprintln!(
                    "[diag] first Inf in `{label}` at index {idx} / {len} samples (value {val})",
                    len = samples.len(),
                    val = samples[idx],
                );
            }
        }
    }
}

/// **Stage B, Part 2** — adaptive-window lifecycle state.
///
/// * `Steady`   — long window, normal cadence (the only reachable
///   state when `turn_detect_enabled` is `false`).
/// * `Shrunk`   — a turn is suspected; window shrunk to
///   `sv_turn_window_samples`, cadence tightened to
///   `sv_turn_update_samples`.
/// * `Regrowing`— the cue has re-stabilised and at least one refresh
///   on the shrunk window has landed; the window is restored and we
///   are waiting out the trailing book-keeping before returning to
///   `Steady`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnState {
    Steady,
    Shrunk,
    Regrowing,
}

/// Fast EMA rate for the per-frame `fm_fast` cue (reacts within a few
/// frames so a turn is visible quickly).
const FM_FAST_EMA_RATE: f32 = 0.3;
/// Slow EMA rate for the confirmed-target `fm_fast` baseline (only
/// advanced while the gate is ON, so it tracks the *enrolled*
/// speaker's typical cue level).
const FM_FAST_BASELINE_RATE: f32 = 0.02;
/// How close (absolute `fm_fast` EMA distance) to the baseline counts
/// as "re-stabilised" for the `Shrunk` → `Regrowing` transition.
const TURN_STABLE_BAND: f32 = 0.1;

/// **Stage B, Part 2** — pure turn-detection / adaptive-window state
/// machine. Extracted from [`StreamingState`] so the transition logic
/// is unit-testable without ONNX: it consumes a per-frame `fm_fast`
/// observation plus the current gate state and emits the
/// `effective_window` / cadence / fail-closed signals the streaming
/// core applies.
///
/// Entirely inert when `turn_detect_enabled` is `false` — `observe`
/// early-returns and `effective_window` stays at `sv_window_samples`.
#[derive(Debug, Clone)]
// Internal state machine — each bool is an independent latch
// (fail-closed active, shrunk-refresh-done, evidence-this-frame,
// saw-low-while-off). Folding them into an enum would not model the
// orthogonal latches faithfully.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct TurnDetector {
    /// Current adaptive-window phase.
    pub(crate) state: TurnState,
    /// Fast EMA of the per-frame `fm_fast` cue.
    pub(crate) fm_fast_ema: f32,
    /// Slow EMA of `fm_fast`, advanced only on confirmed-target
    /// (gate-ON) speech frames — the "this is the enrolled speaker"
    /// reference level.
    pub(crate) fm_fast_baseline: f32,
    /// Consecutive stable frames counted toward the `Shrunk` →
    /// `Regrowing` transition.
    pub(crate) turn_stable_counter: u32,
    /// Whether the offset fail-closed override is currently active.
    pub(crate) offset_failclosed_active: bool,
    /// Effective ECAPA window length (samples) — `sv_window_samples`
    /// in `Steady`, `sv_turn_window_samples` while `Shrunk`.
    pub(crate) effective_window: usize,
    /// `true` once at least one refresh has completed since the
    /// window shrank — gates both the fail-closed clear and the
    /// `Shrunk` → `Regrowing` transition.
    pub(crate) shrunk_refresh_done: bool,
    /// Whether the cue had any evidence (`fm_fast` was `Some`) on the
    /// most recent observed frame — exposed for diagnostics / tests.
    pub(crate) had_evidence: bool,
    /// `true` once the cue EMA has been seen well below baseline
    /// while the gate is OFF. Onset-suspect requires this dip to
    /// have happened first — otherwise *any* gate-OFF frame whose
    /// cue sits near baseline reads as an onset turn, and ordinary
    /// silence between the target's own utterances would spuriously
    /// trip the detector. Cleared when onset-suspect fires.
    pub(crate) saw_low_while_off: bool,
}

/// What [`TurnDetector::observe`] tells the streaming core to do for
/// the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TurnDecision {
    /// `true` on the frame a suspected turn is first detected — the
    /// core shrinks the speech buffer down to `effective_window` on
    /// this edge.
    pub(crate) entered_shrunk: bool,
    /// `true` while `Shrunk` — the core ORs a `due_turn` trigger into
    /// the refresh-due decision and uses the tight cadence.
    pub(crate) shrunk: bool,
}

impl TurnDetector {
    /// Construct with the steady window length — every field starts
    /// in the inert `Steady` configuration.
    pub(crate) fn new(sv_window_samples: usize) -> Self {
        Self {
            state: TurnState::Steady,
            fm_fast_ema: 0.0,
            fm_fast_baseline: 0.0,
            turn_stable_counter: 0,
            offset_failclosed_active: false,
            effective_window: sv_window_samples,
            shrunk_refresh_done: false,
            had_evidence: false,
            saw_low_while_off: false,
        }
    }

    /// Reset to the steady configuration (used by
    /// [`StreamingState::reset`]).
    pub(crate) fn reset(&mut self, sv_window_samples: usize) {
        *self = Self::new(sv_window_samples);
    }

    /// Record that a refresh on the shrunk window has completed.
    /// Clears the fail-closed override (the gate then follows normal
    /// scoring again) and arms the `Shrunk` → `Regrowing` transition.
    pub(crate) fn note_refresh_completed(&mut self) {
        if self.state == TurnState::Shrunk {
            self.shrunk_refresh_done = true;
            self.offset_failclosed_active = false;
        }
    }

    /// Per-`now_speech`-frame update. `fm_fast` is the per-frame cue
    /// (`None` = no evidence), `gate_on` is the gate state coming
    /// into this frame. Pure: mutates only `self`, returns the
    /// `TurnDecision` the core acts on. No-op (returns the inert
    /// decision) when `cfg.turn_detect_enabled` is `false`.
    pub(crate) fn observe(
        &mut self,
        fm_fast: Option<f32>,
        gate_on: bool,
        cfg: &PipelineConfig,
    ) -> TurnDecision {
        if !cfg.turn_detect_enabled {
            return TurnDecision {
                entered_shrunk: false,
                shrunk: false,
            };
        }
        self.had_evidence = fm_fast.is_some();
        if let Some(fm) = fm_fast {
            // Seed the EMAs on first evidence so they don't crawl up
            // from 0 and produce a spurious early "turn".
            if self.fm_fast_ema == 0.0 && self.fm_fast_baseline == 0.0 {
                self.fm_fast_ema = fm;
                self.fm_fast_baseline = fm;
            } else {
                self.fm_fast_ema =
                    FM_FAST_EMA_RATE * fm + (1.0 - FM_FAST_EMA_RATE) * self.fm_fast_ema;
                if gate_on {
                    self.fm_fast_baseline = FM_FAST_BASELINE_RATE * fm
                        + (1.0 - FM_FAST_BASELINE_RATE) * self.fm_fast_baseline;
                }
            }
        }

        // Track a confirmed dip while the gate is OFF — the precursor
        // an onset-suspect requires (a low cue followed by a rise back
        // toward baseline, not just "near baseline").
        if !gate_on
            && self.had_evidence
            && self.fm_fast_baseline > 0.0
            && self.fm_fast_ema < self.fm_fast_baseline - cfg.turn_drop_delta
        {
            self.saw_low_while_off = true;
        }

        let mut entered_shrunk = false;
        match self.state {
            TurnState::Steady => {
                // offset-suspect: gate ON, cue dropped well below the
                // confirmed-target baseline → suspected target→other.
                let offset_suspect = gate_on
                    && self.had_evidence
                    && self.fm_fast_ema < self.fm_fast_baseline - cfg.turn_drop_delta;
                // onset-suspect: gate OFF, cue dipped low and has now
                // risen sharply back toward the baseline → suspected
                // other→target. The `saw_low_while_off` precursor is
                // what makes this a *rise* rather than just "the cue
                // happens to sit near baseline".
                let onset_suspect = !gate_on
                    && self.had_evidence
                    && self.saw_low_while_off
                    && self.fm_fast_ema >= self.fm_fast_baseline - TURN_STABLE_BAND
                    && self.fm_fast_baseline > 0.0;
                if offset_suspect || onset_suspect {
                    self.state = TurnState::Shrunk;
                    self.effective_window = cfg.sv_turn_window_samples;
                    self.turn_stable_counter = 0;
                    self.shrunk_refresh_done = false;
                    self.saw_low_while_off = false;
                    entered_shrunk = true;
                    if offset_suspect && cfg.offset_fail_closed {
                        self.offset_failclosed_active = true;
                    }
                }
            }
            TurnState::Shrunk => {
                // Count consecutive frames the cue sits close to the
                // (evolving) baseline; once stable AND a refresh on
                // the shrunk window has landed, begin regrowing.
                let close = self.had_evidence
                    && (self.fm_fast_ema - self.fm_fast_baseline).abs() <= TURN_STABLE_BAND;
                if close {
                    self.turn_stable_counter += 1;
                } else {
                    self.turn_stable_counter = 0;
                }
                if self.shrunk_refresh_done && self.turn_stable_counter >= cfg.turn_stable_frames {
                    self.state = TurnState::Regrowing;
                    self.effective_window = cfg.sv_window_samples;
                }
            }
            TurnState::Regrowing => {
                // The window is already restored; one frame in
                // `Regrowing` is enough to settle back to `Steady`
                // (kept as a distinct state so a future
                // gradual-regrow policy has a hook).
                self.state = TurnState::Steady;
                self.turn_stable_counter = 0;
            }
        }

        TurnDecision {
            entered_shrunk,
            shrunk: self.state == TurnState::Shrunk,
        }
    }
}

/// Private engine state — the carry-over data the streaming loop
/// needs between `push_samples` / `flush` calls. Doesn't hold
/// `EmbeddingPool` or `PipelineComponents`; both
/// [`StreamingPipeline`] (owns) and the offline wrapper (borrows)
/// pass those in by mutable reference.
pub(crate) struct StreamingState {
    /// Audio-rate ring; sub-decision-frame residue waiting for
    /// enough samples to drive the next resampler call.
    audio_ring: VecDeque<f32>,
    /// Stateful sinc resampler from `audio_sample_rate` →
    /// `decision_sample_rate`. `None` when the two rates match
    /// (identity path — no resample).
    resampler: Option<SincFixedOut<f32>>,
    /// Scratch input/output buffers reused across resampler calls
    /// to avoid per-call allocation. `None` mirrors `resampler`.
    resampler_in: Option<Vec<f32>>,
    resampler_out: Option<Vec<Vec<f32>>>,
    /// SV-rate (16 kHz) speech accumulator — frames with
    /// `speech_prob > vad_threshold` flow in here.
    speech_buffer: VecDeque<f32>,
    /// Decision-rate (16 kHz) lookback ring holding the last
    /// `PipelineConfig::pre_roll_ms` of audio. Every decision frame
    /// is pushed in (excess popped from the front); on every VAD
    /// OFF→ON transition the ring's contents are **prepended** to
    /// `speech_buffer` so the first ECAPA refresh after onset
    /// includes the pre-trigger audio. Empty when
    /// `pre_roll_ms == 0`.
    pre_roll_ring: VecDeque<f32>,
    /// Cached capacity of `pre_roll_ring` so the hot loop avoids
    /// recomputing `pre_roll_ms * sample_rate / 1000` per frame.
    pre_roll_capacity: usize,
    /// Reusable contiguous staging slice for Fbank / F0 input.
    sv_window_scratch: Vec<f32>,
    /// SV refresh cadence + VAD-edge early-refresh state — mirrors
    /// `process_offline`'s locals.
    samples_since_update: usize,
    silence_seen_since_refresh: bool,
    new_speech_samples_after_silence: usize,
    prev_speech: bool,
    consecutive_speech_ms: f32,
    /// Per-frame scoring scalars (`last_score` / `last_cs` /
    /// `last_fm`) carried between frames. Bundled into [`ScoreState`]
    /// so the per-frame core and the refresh strategies can share a
    /// single `&mut` handle.
    score: ScoreState,
    /// Continuous VAD-silence duration in ms, reset to 0 the moment a
    /// VAD-positive frame arrives. Used by the optional silence
    /// force-off rule (`PipelineConfig::silence_force_off_ms`) to
    /// override `last_score` when nothing's been heard for a while.
    /// Necessary because `speech_buffer` only accumulates speech
    /// frames, so `last_score` never decays on its own during clean
    /// (DFN3-denoised) silence and the gate would otherwise stay open
    /// indefinitely on whatever the last refresh scored.
    silence_ms_since_speech: f32,
    gate_state: GateState,
    envelope_state: EnvelopeState,
    /// Last emitted `(start_sample, is_on)` decision (at audio
    /// rate). Used to detect runs and emit the new boundary only on
    /// transitions.
    current_decision: Option<bool>,
    /// Monotonic VAD-frame counter since construction / reset — used
    /// as `AutoLearnEvent.frame_idx`.
    frame_idx: usize,
    /// Monotonic audio-rate sample counter — used as
    /// `gate_decisions[i].0` so successive call outputs share a
    /// timeline.
    audio_samples_emitted: usize,
    /// Audio-rate samples per decision-rate VAD frame at identity
    /// rate (= `CHUNK_SAMPLES_16K`). Cached only for the identity
    /// branch; dual-rate uses `resampler.input_frames_next()`.
    identity_input_per_frame: usize,
    /// Persistent ECAPA / Fbank / F0 worker thread, present only
    /// when `config.pipeline.async_refresh = true`. The worker owns
    /// the `Fbank` + `EcapaTdnn` (moved at construction) and
    /// returns them via `shutdown` so [`StreamingPipeline::into_parts`]
    /// can reconstruct the full [`PipelineComponents`].
    pub(crate) async_worker: Option<AsyncWorker>,
    /// **Stage B, Part 1** — per-frame fast F0 cue ring + YIN cache.
    /// Always present; only *exercised* when `fast_cue_enabled` or
    /// `turn_detect_enabled` is set (the per-frame YIN call is the
    /// cost, so the disabled path skips `push_frame` entirely).
    fast_f0_cue: FastF0Cue,
    /// **Stage B, Part 2** — adaptive-window / turn-detection state
    /// machine. Inert (stays `Steady`, `effective_window ==
    /// sv_window_samples`) unless `turn_detect_enabled` is set.
    turn: TurnDetector,
    /// **Stage C, Phase 5 part 3** — optional target-speaker-extraction
    /// stage. Present only when `StreamingConfig::pipeline.tse` is
    /// `Some` (and `audio_sample_rate == decision_sr == 16_000`,
    /// enforced at construction time). When present, every emitted
    /// audio chunk is routed through the stage **before** the envelope
    /// is applied; the gate's `is_on` decisions still come from the
    /// raw decision-rate audio.
    ///
    /// Owned by the state (rather than left on `PipelineComponents`)
    /// so the buffered accumulator inside the stage persists naturally
    /// across `push_samples` calls — the same ownership pattern the
    /// async-refresh worker uses for `fbank` + `ecapa`.
    pub(crate) tse_stage: Option<TseStage>,
    /// **Stage C, Phase 5** — optional DFN3 noise-suppression stream.
    /// Sits **after** [`Self::tse_stage`] in the audio-rate chain so
    /// DFN3 cleans residual noise on the extracted target voice (the
    /// empirical TSE→DFN3 ordering beats DFN3→TSE; see Phase 5).
    /// Present when [`StreamingConfig::dfn3_onnx_path`] is `Some`.
    pub(crate) dfn3_stream: Option<Dfn3Streamer>,
    /// **Stage C, Phase 5** — gain values queued for emission behind
    /// the TSE+DFN3 chain's combined buffering. Pushed at the input-
    /// audio rate (one per audio-rate sample handed to the chain),
    /// consumed at the same rate as the **last** active stage emits.
    /// While streaming, up to `tse_chunk - 1 + dfn3_lookahead_samples`
    /// gains may be pending; the [`Self::flush`] tail drains the rest.
    ///
    /// Empty (and unused) when both `tse_stage` and `dfn3_stream` are
    /// `None`.
    chain_pending_gain: VecDeque<f32>,
    /// **Phase 2.** Raw (pre-chain, pre-dither) audio samples queued
    /// in lockstep with `chain_pending_gain`. Drained sample-by-
    /// sample alongside the chain output so the Overlap wet / dry
    /// mix can pull a time-aligned raw sample for each cleaned
    /// sample. Unused (always empty) in Solo mode.
    chain_pending_raw: VecDeque<f32>,
    /// Optional pyannote-3.0 segmentation detector. Present when
    /// [`StreamingConfig::overlap_onnx_path`] resolves and both TSE
    /// and DFN3 are also configured (the toggle has no effect
    /// otherwise). Drives the [`Self::chain_mode`] state machine.
    pub(crate) overlap_detector: Option<crate::overlap::OverlapDetector>,
    /// Active audio-chain mode for adaptive routing. Defaults to
    /// `Solo` (DFN3 only) — the empirically-best mode for the common
    /// single-speaker live-monitor case. Flips to `Overlap` (TSE
    /// only) once the detector reports sustained overlap.
    chain_mode: ChainMode,
    /// Hysteresis accumulator: ms spent above
    /// `overlap_threshold` since the last `Solo → Overlap` flip.
    overlap_above_ms: f32,
    /// Hysteresis accumulator: ms spent below `overlap_threshold`
    /// since the last `Overlap → Solo` flip.
    overlap_below_ms: f32,
}

/// Adaptive chain routing mode, driven by the overlap detector when
/// configured. See [`StreamingConfig::overlap_onnx_path`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainMode {
    /// Single-speaker situation: route audio through DFN3 only.
    /// Selected at session start and whenever the detector reports
    /// sustained non-overlap.
    Solo,
    /// Multiple speakers present: route audio through TSE only
    /// (target-speaker extraction). DFN3 stays bypassed because
    /// cascading TSE → DFN3 attenuates output by ~28 dB; TSE's
    /// mask-based suppression covers the noise case sufficiently
    /// during overlap.
    Overlap,
}

/// Persistent worker thread for `async_refresh = true` streaming.
///
/// The worker owns the `Fbank` + `EcapaTdnn` for the lifetime of
/// the streaming pipeline. Refresh windows arrive over `work_tx`;
/// each window is run through Fbank → ECAPA + F0 and the resulting
/// `(embedding, f0_mu)` flows back over `result_rx`.
///
/// Bookkeeping (`outstanding`, `pending`, `refresh_frame_indices`)
/// mirrors `process_offline_async`: at most one outstanding
/// inference at a time, one queued window so a burst of two
/// refreshes within one ECAPA wall time doesn't drop work, and
/// frame indices in FIFO order so [`AutoLearnEvent.frame_idx`]
/// reflects the *trigger* frame rather than the result-arrival
/// frame.
pub(crate) struct AsyncWorker {
    work_tx: Sender<Vec<f32>>,
    result_rx: Receiver<Result<(Vec<f32>, f32), EmbeddingError>>,
    join: Option<JoinHandle<(Fbank, EcapaTdnn)>>,
    outstanding: u32,
    pending: Option<Vec<f32>>,
    refresh_frame_indices: VecDeque<usize>,
}

impl AsyncWorker {
    fn spawn(
        mut fbank: Fbank,
        mut ecapa: EcapaTdnn,
        decision_sr: u32,
    ) -> Result<Self, PipelineError> {
        let (work_tx, work_rx) = channel::<Vec<f32>>();
        let (result_tx, result_rx) = channel::<Result<(Vec<f32>, f32), EmbeddingError>>();
        let join = std::thread::Builder::new()
            .name("mellonella-streaming-async-worker".into())
            .spawn(move || {
                while let Ok(window) = work_rx.recv() {
                    let msg = match fbank_ecapa_one(&window, &mut fbank, &mut ecapa) {
                        Ok(embedding) => {
                            let f0_track = estimate_f0_track(
                                &window,
                                decision_sr,
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
                    if result_tx.send(msg).is_err() {
                        break;
                    }
                }
                (fbank, ecapa)
            })
            .map_err(|e| {
                PipelineError::Embedding(EmbeddingError::Ort(format!("spawn async worker: {e}")))
            })?;
        Ok(Self {
            work_tx,
            result_rx,
            join: Some(join),
            outstanding: 0,
            pending: None,
            refresh_frame_indices: VecDeque::new(),
        })
    }

    /// Submit a refresh window. If the worker is idle (`outstanding
    /// == 0`) the window is sent immediately; otherwise it queues
    /// as the single pending window (overwriting any previous
    /// pending — the cadence guarantees that won't happen in
    /// normal use, but the fallback keeps the state machine well-
    /// defined under burst load).
    fn submit(&mut self, window: Vec<f32>, frame_idx: usize) {
        self.refresh_frame_indices.push_back(frame_idx);
        if self.outstanding == 0 {
            if self.work_tx.send(window).is_ok() {
                self.outstanding = 1;
            }
        } else {
            self.pending = Some(window);
        }
    }

    /// Non-blocking poll for the next completed inference.
    fn try_recv_result(&mut self) -> Result<Option<(usize, Vec<f32>, f32)>, EmbeddingError> {
        if self.outstanding == 0 {
            return Ok(None);
        }
        match self.result_rx.try_recv() {
            Ok(Ok((emb, f0_mu))) => {
                let frame_idx = self.refresh_frame_indices.pop_front().unwrap_or(0);
                self.outstanding -= 1;
                if let Some(next) = self.pending.take() {
                    if self.work_tx.send(next).is_ok() {
                        self.outstanding = 1;
                    }
                }
                Ok(Some((frame_idx, emb, f0_mu)))
            }
            Ok(Err(e)) => Err(e),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(EmbeddingError::Ort("async worker disconnected".into()))
            }
        }
    }

    /// Blocking drain of outstanding work — used by `flush_async`.
    fn drain_blocking(&mut self) -> Result<Vec<(usize, Vec<f32>, f32)>, EmbeddingError> {
        let mut results = Vec::new();
        while self.outstanding > 0 {
            let msg = self
                .result_rx
                .recv()
                .map_err(|_| EmbeddingError::Ort("async worker disconnected".into()))?;
            let (emb, f0_mu) = msg?;
            let frame_idx = self.refresh_frame_indices.pop_front().unwrap_or(0);
            self.outstanding -= 1;
            results.push((frame_idx, emb, f0_mu));
            if let Some(next) = self.pending.take() {
                if self.work_tx.send(next).is_ok() {
                    self.outstanding = 1;
                }
            }
        }
        Ok(results)
    }

    /// Tear the worker down and recover the owned components.
    /// Called from [`StreamingPipeline::into_parts`].
    fn shutdown(mut self) -> Result<(Fbank, EcapaTdnn), PipelineError> {
        drop(self.work_tx); // Triggers the worker's `recv()` to return Err and exit.
        let join = self
            .join
            .take()
            .expect("worker join handle present at shutdown");
        join.join().map_err(|_| {
            PipelineError::Embedding(EmbeddingError::Ort("async worker panicked".into()))
        })
    }
}

/// Per-frame refresh policy — the one piece of the streaming loop
/// that differs between the sync and async paths.
///
/// [`StreamingState::step_one_frame_core`] owns everything else
/// (VAD, buffering, gate, envelope, diagnostics); it calls
/// `on_refresh_due` when the cadence fires and `poll` on every
/// frame. Sync runs Fbank / ECAPA / F0 inline and applies the result
/// immediately; async submits the window to a worker and applies
/// whatever has come back by the next `poll`.
pub(crate) trait RefreshStrategy {
    /// A refresh is due. `window` is the contiguous speech buffer.
    /// Sync: run Fbank / ECAPA / F0 inline, update `score` and push
    /// any auto-learn events now. Async: submit `window` to the
    /// worker keyed on `frame_idx`.
    fn on_refresh_due(
        &mut self,
        window: &[f32],
        frame_idx: usize,
        consecutive_speech_ms: f32,
        score: &mut ScoreState,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError>;

    /// Called every frame. Sync: no-op. Async: poll the worker for at
    /// most one ready result and apply it via [`apply_refresh_result`].
    fn poll(
        &mut self,
        consecutive_speech_ms: f32,
        frame_idx: usize,
        score: &mut ScoreState,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError>;

    /// Enrolled `(f0_mu, f0_sigma)` from the pool — needed by the
    /// Stage B per-frame fast F0 cue, which lives in the core but
    /// only the strategy holds the `EmbeddingPool`.
    fn enrolled_f0(&self) -> (f32, f32);

    /// Borrow the strategy's underlying [`EmbeddingPool`] read-only.
    ///
    /// Stage C Step 5 (TSE cond refresh) uses this from
    /// [`StreamingState::step_one_frame_core`] to recompute the TSE
    /// conditioning vector after a successful auto-learn admit — the
    /// strategy owns the `&mut EmbeddingPool`, so the core's only
    /// path to the post-admit pool state is through the strategy.
    fn pool_ref(&self) -> &EmbeddingPool;
}

/// Submit / poll abstraction over an async ECAPA worker, so the same
/// [`AsyncRefresh`] strategy can drive both the persistent
/// [`AsyncWorker`] (live streaming) and the scoped `mpsc` worker
/// inside `process_offline_async` (borrowed components — can't move
/// `fbank` / `ecapa` into a persistent worker).
pub(crate) trait RefreshChannel {
    /// Hand a refresh window to the worker, keyed on the trigger
    /// frame index (FIFO order preserved so auto-learn events get the
    /// trigger-time frame, not the result-arrival frame).
    fn submit(&mut self, window: Vec<f32>, frame_idx: usize);
    /// Non-blocking poll for the next completed inference:
    /// `(trigger_frame, embedding, f0_mu)`.
    fn try_recv_result(&mut self) -> Result<Option<(usize, Vec<f32>, f32)>, EmbeddingError>;
    /// Blocking drain of every outstanding inference. Used at
    /// end-of-stream so the trailing scores / auto-learn events are
    /// captured before the worker is torn down.
    fn drain_blocking(&mut self) -> Result<Vec<(usize, Vec<f32>, f32)>, EmbeddingError>;
}

impl RefreshChannel for AsyncWorker {
    fn submit(&mut self, window: Vec<f32>, frame_idx: usize) {
        AsyncWorker::submit(self, window, frame_idx);
    }
    fn try_recv_result(&mut self) -> Result<Option<(usize, Vec<f32>, f32)>, EmbeddingError> {
        AsyncWorker::try_recv_result(self)
    }
    fn drain_blocking(&mut self) -> Result<Vec<(usize, Vec<f32>, f32)>, EmbeddingError> {
        AsyncWorker::drain_blocking(self)
    }
}

/// Sync refresh strategy: Fbank / ECAPA / F0 run inline on the
/// caller's thread, the score + auto-learn events are updated
/// immediately. `poll` is a no-op.
struct SyncRefresh<'a> {
    fbank: &'a mut Fbank,
    ecapa: &'a mut EcapaTdnn,
    cohort: &'a [Vec<f32>],
    pool: &'a mut EmbeddingPool,
    decision_sr: u32,
    enable_auto_learn: bool,
    score_ema_alpha: f32,
    use_as_norm: bool,
    gate_cfg: GateConfig,
}

impl RefreshStrategy for SyncRefresh<'_> {
    fn on_refresh_due(
        &mut self,
        window: &[f32],
        frame_idx: usize,
        consecutive_speech_ms: f32,
        score: &mut ScoreState,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        let feats = self.fbank.compute(window);
        let n_frames = feats.len() / N_MELS;
        let embedding = self.ecapa.embed_features(&feats, n_frames, N_MELS)?;

        let f0_track = estimate_f0_track(
            window,
            self.decision_sr,
            2048,
            512,
            DEFAULT_F_MIN,
            DEFAULT_F_MAX,
        );
        let (f0_mu, _) = f0_statistics(&f0_track);

        let cs = self.pool.match_score(&embedding);
        let fm = f0_match(
            f0_mu,
            self.pool.metadata().f0_mu,
            self.pool.metadata().f0_sigma,
        );
        score.last_cs = cs;
        score.last_fm = fm;
        let new_score = if self.use_as_norm && !self.cohort.is_empty() {
            as_norm_score(&embedding, cs, self.cohort, 20)
        } else {
            cs
        };
        score.last_score = smooth_score(score.last_score, new_score, self.score_ema_alpha);

        if self.enable_auto_learn
            && should_admit_auto_learn(score.last_score, fm, consecutive_speech_ms, &self.gate_cfg)
        {
            let admitted = self.pool.adapt(embedding);
            let kind = if admitted {
                AutoLearnKind::Admit
            } else {
                AutoLearnKind::RejectAnchorDistance
            };
            out.events.push(AutoLearnEvent {
                frame_idx,
                kind,
                score: score.last_score,
                f0_match: fm,
            });
        }
        Ok(())
    }

    fn poll(
        &mut self,
        _consecutive_speech_ms: f32,
        _frame_idx: usize,
        _score: &mut ScoreState,
        _out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        Ok(())
    }

    fn enrolled_f0(&self) -> (f32, f32) {
        let m = self.pool.metadata();
        (m.f0_mu, m.f0_sigma)
    }

    fn pool_ref(&self) -> &EmbeddingPool {
        self.pool
    }
}

/// A [`RefreshChannel`] over a scoped `mpsc` pair — used by
/// `process_offline_async`, which borrows `PipelineComponents` and so
/// can't move `fbank` / `ecapa` into a persistent [`AsyncWorker`].
/// The actual worker thread lives in `process_offline_async`'s
/// `std::thread::scope`; this struct just carries the channel ends
/// plus the same outstanding / pending / FIFO-frame-index bookkeeping
/// the inline loop used to keep, so the cadence is identical.
pub(crate) struct ScopedRefreshChannel {
    work_tx: std::sync::mpsc::Sender<Vec<f32>>,
    result_rx: std::sync::mpsc::Receiver<Result<(Vec<f32>, f32), EmbeddingError>>,
    outstanding: u32,
    pending: Option<Vec<f32>>,
    refresh_frame_indices: VecDeque<usize>,
}

impl ScopedRefreshChannel {
    pub(crate) fn new(
        work_tx: std::sync::mpsc::Sender<Vec<f32>>,
        result_rx: std::sync::mpsc::Receiver<Result<(Vec<f32>, f32), EmbeddingError>>,
    ) -> Self {
        Self {
            work_tx,
            result_rx,
            outstanding: 0,
            pending: None,
            refresh_frame_indices: VecDeque::new(),
        }
    }
}

impl RefreshChannel for ScopedRefreshChannel {
    fn submit(&mut self, window: Vec<f32>, frame_idx: usize) {
        self.refresh_frame_indices.push_back(frame_idx);
        if self.outstanding == 0 {
            if self.work_tx.send(window).is_ok() {
                self.outstanding = 1;
            }
        } else {
            self.pending = Some(window);
        }
    }

    fn try_recv_result(&mut self) -> Result<Option<(usize, Vec<f32>, f32)>, EmbeddingError> {
        if self.outstanding == 0 {
            return Ok(None);
        }
        match self.result_rx.try_recv() {
            Ok(Ok((emb, f0_mu))) => {
                let frame_idx = self.refresh_frame_indices.pop_front().unwrap_or(0);
                self.outstanding -= 1;
                if let Some(next) = self.pending.take() {
                    if self.work_tx.send(next).is_ok() {
                        self.outstanding = 1;
                    }
                }
                Ok(Some((frame_idx, emb, f0_mu)))
            }
            Ok(Err(e)) => Err(e),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(EmbeddingError::Ort("ECAPA worker disconnected".into()))
            }
        }
    }

    fn drain_blocking(&mut self) -> Result<Vec<(usize, Vec<f32>, f32)>, EmbeddingError> {
        let mut results = Vec::new();
        while self.outstanding > 0 {
            let msg = self
                .result_rx
                .recv()
                .map_err(|_| EmbeddingError::Ort("ECAPA worker disconnected".into()))?;
            let (emb, f0_mu) = msg?;
            let frame_idx = self.refresh_frame_indices.pop_front().unwrap_or(0);
            self.outstanding -= 1;
            results.push((frame_idx, emb, f0_mu));
            if let Some(next) = self.pending.take() {
                if self.work_tx.send(next).is_ok() {
                    self.outstanding = 1;
                }
            }
        }
        Ok(results)
    }
}

/// Async refresh strategy: a due refresh **submits** the window to a
/// [`RefreshChannel`]; every frame `poll` drains at most one ready
/// result and folds it into the pool / gate state via
/// [`apply_refresh_result`].
pub(crate) struct AsyncRefresh<'a> {
    pub(crate) channel: &'a mut dyn RefreshChannel,
    pub(crate) cohort: &'a [Vec<f32>],
    pub(crate) pool: &'a mut EmbeddingPool,
    pub(crate) enable_auto_learn: bool,
    pub(crate) score_ema_alpha: f32,
    pub(crate) gate_cfg: GateConfig,
}

impl RefreshStrategy for AsyncRefresh<'_> {
    fn on_refresh_due(
        &mut self,
        window: &[f32],
        frame_idx: usize,
        _consecutive_speech_ms: f32,
        _score: &mut ScoreState,
        _out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        self.channel.submit(window.to_vec(), frame_idx);
        Ok(())
    }

    fn poll(
        &mut self,
        consecutive_speech_ms: f32,
        _frame_idx: usize,
        score: &mut ScoreState,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        if let Some((trigger_frame, embedding, f0_mu)) = self.channel.try_recv_result()? {
            apply_refresh_result(
                embedding,
                f0_mu,
                trigger_frame,
                consecutive_speech_ms,
                self.pool,
                self.cohort,
                &self.gate_cfg,
                self.enable_auto_learn,
                self.score_ema_alpha,
                score,
                &mut out.events,
            );
        }
        Ok(())
    }

    fn enrolled_f0(&self) -> (f32, f32) {
        let m = self.pool.metadata();
        (m.f0_mu, m.f0_sigma)
    }

    fn pool_ref(&self) -> &EmbeddingPool {
        self.pool
    }
}

/// Owning storage for `PipelineComponents` inside a
/// [`StreamingPipeline`]. Sync stores the full struct; async splits
/// out `fbank` + `ecapa` into the persistent worker thread, keeping
/// only `vad` + `cohort` on the main thread.
///
/// The Sync variant is heap-sized (PipelineComponents holds ONNX
/// session handles); boxing keeps the enum compact even though we
/// only ever hold one variant per pipeline lifetime.
pub(crate) enum ComponentsStorage {
    Sync(Box<PipelineComponents>),
    Async {
        vad: SileroVad,
        cohort: Vec<Vec<f32>>,
    },
}

impl StreamingState {
    /// Sync constructor — `async_refresh` must be `false`. For
    /// async streaming, call [`Self::new_async`] (and route through
    /// [`StreamingPipeline`] which owns the worker lifecycle).
    pub(crate) fn new(config: &StreamingConfig) -> Result<Self, PipelineError> {
        if config.pipeline.async_refresh {
            return Err(PipelineError::Embedding(EmbeddingError::Ort(
                "StreamingState::new requires async_refresh = false; \
                 use StreamingState::new_async or StreamingPipeline::new for async"
                    .into(),
            )));
        }
        let decision_sr = config.pipeline.sample_rate;
        let audio_sr = config.audio_sample_rate;
        let (resampler, resampler_in, resampler_out) = if audio_sr == decision_sr {
            (None, None, None)
        } else {
            let ratio = f64::from(decision_sr) / f64::from(audio_sr);
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let r = SincFixedOut::<f32>::new(ratio, 1.1, params, CHUNK_SAMPLES_16K, 1).map_err(
                |e| {
                    PipelineError::Embedding(EmbeddingError::Ort(format!(
                        "streaming resampler init failed: {e}"
                    )))
                },
            )?;
            let in_cap = r.input_frames_max();
            (
                Some(r),
                Some(Vec::with_capacity(in_cap)),
                Some(vec![vec![0.0_f32; CHUNK_SAMPLES_16K]]),
            )
        };
        let pre_roll_capacity = config.pipeline.pre_roll_samples_decision();
        Ok(Self {
            audio_ring: VecDeque::with_capacity(audio_sr as usize), // ~1 s
            resampler,
            resampler_in,
            resampler_out,
            speech_buffer: VecDeque::with_capacity(config.pipeline.sv_window_samples),
            pre_roll_ring: VecDeque::with_capacity(pre_roll_capacity),
            pre_roll_capacity,
            sv_window_scratch: Vec::with_capacity(config.pipeline.sv_window_samples),
            samples_since_update: 0,
            silence_seen_since_refresh: false,
            new_speech_samples_after_silence: 0,
            prev_speech: false,
            consecutive_speech_ms: 0.0,
            score: ScoreState::new(),
            silence_ms_since_speech: 0.0,
            gate_state: GateState::new(config.gate),
            envelope_state: EnvelopeState::new(config.gate, audio_sr),
            current_decision: None,
            frame_idx: 0,
            audio_samples_emitted: 0,
            identity_input_per_frame: CHUNK_SAMPLES_16K,
            async_worker: None,
            fast_f0_cue: FastF0Cue::new(),
            turn: TurnDetector::new(config.pipeline.sv_window_samples),
            // The TSE stage is plumbed in by `StreamingPipeline::new`
            // after rate-checking and snapshotting the cond embedding.
            // Plain `StreamingState::new` callers (offline async path,
            // ad-hoc tests) keep the TSE-off byte-equal contract.
            tse_stage: None,
            dfn3_stream: None,
            chain_pending_gain: VecDeque::new(),
            chain_pending_raw: VecDeque::new(),
            overlap_detector: None,
            chain_mode: ChainMode::Solo,
            overlap_above_ms: 0.0,
            overlap_below_ms: 0.0,
        })
    }

    /// Async constructor — spawns the persistent ECAPA / Fbank / F0
    /// worker thread, moves `fbank` + `ecapa` into it. The
    /// resulting state's `step_one_frame_async` dispatches refresh
    /// windows to the worker via channels instead of calling
    /// fbank/ecapa inline.
    pub(crate) fn new_async(
        config: &StreamingConfig,
        fbank: Fbank,
        ecapa: EcapaTdnn,
    ) -> Result<Self, PipelineError> {
        if !config.pipeline.async_refresh {
            return Err(PipelineError::Embedding(EmbeddingError::Ort(
                "StreamingState::new_async requires async_refresh = true".into(),
            )));
        }
        // Build a sync-shaped state first by pretending async_refresh
        // is off, then attach the worker. This sidesteps duplicating
        // ~60 lines of resampler / buffer / gate / envelope init.
        let sync_cfg = StreamingConfig {
            pipeline: PipelineConfig {
                async_refresh: false,
                ..config.pipeline.clone()
            },
            ..config.clone()
        };
        let mut state = Self::new(&sync_cfg)?;
        let worker = AsyncWorker::spawn(fbank, ecapa, config.pipeline.sample_rate)?;
        state.async_worker = Some(worker);
        Ok(state)
    }

    /// Reset the carry-over state without touching pool or
    /// components. Used by [`StreamingPipeline::reset`].
    pub(crate) fn reset(&mut self, config: &StreamingConfig) {
        self.audio_ring.clear();
        if let Some(r) = self.resampler.as_mut() {
            r.reset();
        }
        self.speech_buffer.clear();
        self.pre_roll_ring.clear();
        self.sv_window_scratch.clear();
        self.samples_since_update = 0;
        self.silence_seen_since_refresh = false;
        self.new_speech_samples_after_silence = 0;
        self.prev_speech = false;
        self.consecutive_speech_ms = 0.0;
        self.score = ScoreState::new();
        self.silence_ms_since_speech = 0.0;
        self.gate_state = GateState::new(config.gate);
        self.envelope_state = EnvelopeState::new(config.gate, config.audio_sample_rate);
        self.current_decision = None;
        self.frame_idx = 0;
        self.audio_samples_emitted = 0;
        self.fast_f0_cue.reset();
        self.turn.reset(config.pipeline.sv_window_samples);
        // Stage C, Phase 5: drain the TSE/DFN3 chain accumulators and
        // the pending-gain queue so a `reset()` is a clean restart
        // even when neural processing is active.
        if let Some(stage) = self.tse_stage.as_mut() {
            stage.reset();
        }
        if let Some(stream) = self.dfn3_stream.as_mut() {
            stream.reset();
        }
        self.chain_pending_gain.clear();
        self.chain_pending_raw.clear();
    }

    /// Push a decision-rate frame into the pre-roll lookback ring,
    /// dropping the oldest samples once `pre_roll_capacity` is
    /// reached. No-op when pre-roll is disabled
    /// (`pre_roll_capacity == 0`).
    fn push_pre_roll(&mut self, decision_frame: &[f32]) {
        if self.pre_roll_capacity == 0 {
            return;
        }
        for &sample in decision_frame {
            if self.pre_roll_ring.len() == self.pre_roll_capacity {
                self.pre_roll_ring.pop_front();
            }
            self.pre_roll_ring.push_back(sample);
        }
    }

    /// Drain the pre-roll ring into the back of `speech_buffer` in
    /// chronological order, capped at `sv_window_samples` (excess
    /// pops from the front of the speech buffer). Called once on
    /// every VAD OFF→ON transition so the next ECAPA refresh sees
    /// the pre-trigger audio (issue #80). After this call the ring
    /// is **left in place** — overlapping pre-roll with the first
    /// frames of speech doesn't hurt and clearing would force the
    /// ring to refill from scratch on rapid re-onsets.
    fn prepend_pre_roll_to_speech_buffer(&mut self, sv_window_samples: usize) {
        if self.pre_roll_ring.is_empty() {
            return;
        }
        for &sample in &self.pre_roll_ring {
            if self.speech_buffer.len() == sv_window_samples {
                self.speech_buffer.pop_front();
            }
            self.speech_buffer.push_back(sample);
        }
    }

    /// Audio-rate samples needed to drive one VAD frame at the
    /// decision rate. Identity case: 512. Dual-rate: variable, asks
    /// the resampler.
    fn input_per_frame(&self) -> usize {
        match self.resampler.as_ref() {
            Some(r) => r.input_frames_next(),
            None => self.identity_input_per_frame,
        }
    }

    /// Drive one VAD frame's worth of audio-rate samples through
    /// the decision path. Pops samples from `audio_ring`, produces
    /// one decision-rate VAD frame, and returns the audio-rate
    /// slice that was consumed (for envelope application by the
    /// caller). `None` means not enough samples in the ring yet.
    fn drain_one_frame(&mut self) -> Result<Option<DrainedFrame>, PipelineError> {
        let n_input = self.input_per_frame();
        if self.audio_ring.len() < n_input {
            return Ok(None);
        }
        let mut audio_chunk = Vec::with_capacity(n_input);
        for _ in 0..n_input {
            audio_chunk.push(self.audio_ring.pop_front().unwrap_or(0.0));
        }
        let decision_chunk = match self.resampler.as_mut() {
            None => audio_chunk.clone(),
            Some(r) => {
                let buf_in = self
                    .resampler_in
                    .as_mut()
                    .expect("resampler_in present alongside resampler");
                buf_in.clear();
                buf_in.extend_from_slice(&audio_chunk);
                let buf_out = self
                    .resampler_out
                    .as_mut()
                    .expect("resampler_out present alongside resampler");
                let in_slices = [buf_in.as_slice()];
                r.process_into_buffer(&in_slices, buf_out, None)
                    .map_err(|e| {
                        PipelineError::Embedding(EmbeddingError::Ort(format!(
                            "streaming resampler step failed: {e}"
                        )))
                    })?;
                buf_out[0].clone()
            }
        };
        debug_assert_eq!(decision_chunk.len(), CHUNK_SAMPLES_16K);
        Ok(Some((audio_chunk, decision_chunk)))
    }

    /// Shared per-VAD-frame core for all three streaming call-sites
    /// (sync streaming, async streaming, `process_offline_async`).
    ///
    /// Everything common lives here: VAD scoring, speech-buffer
    /// accumulation, the pre-roll ring, silence bookkeeping, the
    /// refresh-trigger decision (`due_normal` / `due_early`), the
    /// gate update + `silence_force_off` AND-term, diagnostics
    /// pushes, the decision-boundary emit, the envelope advance, and
    /// Stage C Step 5 helper: after a strategy call that *may have*
    /// admitted a new embedding into the pool, scan `out.events` for
    /// `AutoLearnKind::Admit` entries past `events_before` and — if any
    /// — recompute the TSE cond embedding from the strategy's pool
    /// view and push it into [`Self::tse_stage`].
    ///
    /// No-ops when TSE isn't enabled or auto-learn didn't admit.
    fn refresh_tse_cond_if_admitted(
        &mut self,
        strategy: &mut dyn RefreshStrategy,
        out: &StreamingOutput,
        events_before: usize,
    ) -> Result<(), PipelineError> {
        let Some(stage) = self.tse_stage.as_mut() else {
            return Ok(());
        };
        let admitted = out
            .events
            .get(events_before..)
            .is_some_and(|tail| tail.iter().any(|e| matches!(e.kind, AutoLearnKind::Admit)));
        if !admitted {
            return Ok(());
        }
        let cond = tse_cond_embedding(strategy.pool_ref())?;
        stage.set_cond_embedding(&cond).map_err(PipelineError::from)
    }

    /// the counter bumps.
    ///
    /// The *only* thing that varies between sync and async is **how**
    /// a due refresh is executed and how its result is applied — that
    /// is parameterised via the [`RefreshStrategy`] trait. `vad` is
    /// passed in by `&mut` (rather than reached through
    /// `PipelineComponents`) because in async mode the rest of the
    /// components live in a worker thread.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn step_one_frame_core(
        &mut self,
        audio_chunk: &[f32],
        decision_frame: &[f32],
        vad: &mut SileroVad,
        strategy: &mut dyn RefreshStrategy,
        config: &StreamingConfig,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        let vad_frame = CHUNK_SAMPLES_16K;
        let pipeline_cfg = &config.pipeline;
        let dt_ms = pipeline_cfg.vad_frame_ms();
        // Stage B: when no opt-in flag is set this is always
        // `sv_window_samples`, so every site below is byte-identical
        // to the pre-Stage-B behaviour.
        let effective_window = self.turn.effective_window;
        let stage_b_active = pipeline_cfg.fast_cue_enabled || pipeline_cfg.turn_detect_enabled;

        let speech_prob = vad.score(decision_frame)?;
        let now_speech = speech_prob > pipeline_cfg.vad_threshold;
        if now_speech {
            if !self.prev_speech {
                // VAD OFF→ON: fold the pre-roll ring into the speech
                // buffer so the next ECAPA refresh sees the pre-trigger
                // audio. Issue #80.
                self.prepend_pre_roll_to_speech_buffer(effective_window);
            }
            for &sample in decision_frame {
                if self.speech_buffer.len() == effective_window {
                    self.speech_buffer.pop_front();
                }
                self.speech_buffer.push_back(sample);
            }
            self.consecutive_speech_ms += dt_ms;
            if self.silence_seen_since_refresh {
                self.new_speech_samples_after_silence += vad_frame;
            }
        } else {
            self.consecutive_speech_ms = 0.0;
        }
        if self.prev_speech && !now_speech {
            self.silence_seen_since_refresh = true;
            self.new_speech_samples_after_silence = 0;
        }
        if now_speech {
            self.silence_ms_since_speech = 0.0;
        } else {
            self.silence_ms_since_speech += dt_ms;
        }
        self.prev_speech = now_speech;
        self.samples_since_update += vad_frame;

        // Update the pre-roll ring with the just-processed frame *after*
        // the OFF→ON prepend check above so the ring never contains the
        // trigger frame itself.
        self.push_pre_roll(decision_frame);

        // --- Stage B, Part 1: per-frame fast F0 cue ---------------
        // Compute the instantaneous F0 cue whenever any Stage B
        // feature needs it. `fm_fast` is `Some` only with real
        // evidence (ring full + voiced frame); `None` = neutral / no
        // EMA update. The fused score is what feeds the gate when
        // `fast_cue_enabled`; otherwise the gate sees `last_score`
        // exactly as before.
        let mut fm_fast: Option<f32> = None;
        if stage_b_active {
            if let Some(f0_hz) = self
                .fast_f0_cue
                .push_frame(decision_frame, pipeline_cfg.sample_rate)
            {
                let (mu, sigma) = strategy.enrolled_f0();
                fm_fast = Some(f0_match(f0_hz, mu, sigma));
            }
        }

        // --- Stage B, Part 2: turn detection ----------------------
        // Observe the cue against the fast/slow EMA baseline. The
        // detector is fully inert unless `turn_detect_enabled`.
        // `gate_on` is the gate state *coming into* this frame
        // (`current_decision`), so the offset/onset-suspect tests see
        // the pre-update state. Only fed on `now_speech` frames.
        let gate_on_before = self.current_decision.unwrap_or(false);
        let turn_decision = if now_speech {
            self.turn.observe(fm_fast, gate_on_before, pipeline_cfg)
        } else {
            TurnDecision {
                entered_shrunk: false,
                shrunk: self.turn.state == TurnState::Shrunk,
            }
        };
        if turn_decision.entered_shrunk {
            // On the shrink edge, drop the speech buffer down to the
            // most recent `effective_window` samples so the next
            // refresh sees only post-turn audio.
            let new_window = self.turn.effective_window;
            while self.speech_buffer.len() > new_window {
                self.speech_buffer.pop_front();
            }
        }
        // While shrunk, the refresh cadence tightens to the turn
        // cadence and a refresh fires immediately on the shrink edge.
        let update_cadence = if turn_decision.shrunk {
            pipeline_cfg.sv_turn_update_samples
        } else {
            pipeline_cfg.sv_update_samples
        };
        let refresh_window = self.turn.effective_window;

        let due_normal = self.samples_since_update >= update_cadence;
        let due_early = self.silence_seen_since_refresh
            && now_speech
            && self.new_speech_samples_after_silence
                >= pipeline_cfg.sv_min_new_samples_after_silence;
        let due_turn = turn_decision.entered_shrunk;
        if (due_normal || due_early || due_turn) && self.speech_buffer.len() >= refresh_window {
            self.samples_since_update = 0;
            self.silence_seen_since_refresh = false;
            self.new_speech_samples_after_silence = 0;
            // Stage the contiguous window into the reusable scratch so
            // both strategies see the same `&[f32]` view. While
            // shrunk this is only the last `effective_window`
            // samples.
            self.sv_window_scratch.clear();
            self.sv_window_scratch
                .extend(self.speech_buffer.iter().copied());
            let events_before = out.events.len();
            strategy.on_refresh_due(
                &self.sv_window_scratch,
                self.frame_idx,
                self.consecutive_speech_ms,
                &mut self.score,
                out,
            )?;
            // Stage C Step 5: when auto-learn admitted a new embedding
            // on this refresh, recompute the TSE cond from the
            // updated pool and push it into the live stage. The pool
            // is only reachable through the strategy (it owns the
            // `&mut EmbeddingPool`); the strategy exposes a read-only
            // borrow via `pool_ref` after the admit.
            self.refresh_tse_cond_if_admitted(strategy, out, events_before)?;
            // A refresh on the shrunk window has now been issued. For
            // the sync strategy this completes inline; for the async
            // strategy the result trails by one inference (the async
            // path already trails `last_score` by design), so we
            // treat "submitted while shrunk" as "completed" — it arms
            // both the fail-closed clear and the regrow transition.
            if turn_decision.shrunk {
                self.turn.note_refresh_completed();
            }
        }

        // Every frame: sync is a no-op, async polls the worker and
        // applies any ready result before the gate update.
        let events_before_poll = out.events.len();
        strategy.poll(
            self.consecutive_speech_ms,
            self.frame_idx,
            &mut self.score,
            out,
        )?;
        // Async refresh path can deliver an Admit on `poll` too — same
        // recompute as the on_refresh_due branch above.
        self.refresh_tse_cond_if_admitted(strategy, out, events_before_poll)?;

        // Stage B, Part 1: the gate sees the *fused* score when the
        // fast cue is enabled, the bare `last_score` otherwise.
        let gate_score = if pipeline_cfg.fast_cue_enabled {
            fuse_fast_cue(
                self.score.last_score,
                fm_fast.unwrap_or(pipeline_cfg.fast_cue_f0_neutral),
                pipeline_cfg.fast_cue_weight,
                pipeline_cfg.fast_cue_f0_neutral,
            )
        } else {
            self.score.last_score
        };

        let is_on_score = self.gate_state.update(gate_score, dt_ms, now_speech);
        let is_on = is_on_score
            && !(pipeline_cfg.silence_force_off_ms > 0.0
                && self.silence_ms_since_speech >= pipeline_cfg.silence_force_off_ms)
            // Stage B, Part 2: offset fail-closed — an extra AND term
            // parallel to the silence rule, OUTSIDE `gate_state.update`
            // so it bypasses the gate hangover. Cleared once a refresh
            // on the shrunk window lands.
            && !self.turn.offset_failclosed_active;
        if config.diagnostics {
            out.gate_per_frame.push(is_on);
            // Stage B, Part 1: with the fast cue on, the diagnostics
            // score is the same fused value the gate consumed.
            out.score_per_frame.push(gate_score);
            out.cos_sim_max_per_frame.push(self.score.last_cs);
            out.f0_match_per_frame.push(self.score.last_fm);
        }

        // Decision boundary: record (audio_rate_index, is_on) on
        // transition. The index is cumulative across calls.
        let block_start_audio = self.audio_samples_emitted;
        if self.current_decision != Some(is_on) {
            out.gate_decisions.push((block_start_audio, is_on));
            self.current_decision = Some(is_on);
        }

        // Apply envelope at audio rate for this frame's audio-rate
        // span. `EnvelopeState::advance` returns one gain per
        // sample.
        let gain = self.envelope_state.advance(is_on, audio_chunk.len());
        // Run the overlap detector (when configured) and update the
        // chain-routing state machine. Hysteresis lives here so a
        // transient blip in the detector's output doesn't flap the
        // chain.
        let adaptive = self.overlap_detector.is_some()
            && self.tse_stage.is_some()
            && self.dfn3_stream.is_some();
        if adaptive {
            self.update_chain_mode(decision_frame, dt_ms, config)?;
        }

        if self.tse_stage.is_some() || self.dfn3_stream.is_some() {
            // Stage C, Phase 5: route this frame's audio through the
            // optional audio chain (TSE / DFN3, or both when
            // adaptive=false). Each stage buffers samples internally
            // (TSE up to `chunk_samples - 1`; DFN3 holds a 2-frame
            // conv lookahead), so we queue per-sample gains and pop
            // them as the chain's **last active stage** emits.
            self.chain_pending_gain.extend(gain.iter().copied());
            // Phase 2: queue the pre-dither, pre-chain audio so the
            // Overlap-mode wet / dry mix can read a time-aligned raw
            // sample for every cleaned sample the chain emits. Solo
            // mode never consumes from this queue, but pushing
            // unconditionally keeps the bookkeeping simple — the
            // queue is cleared on every Solo ↔ Overlap transition.
            self.chain_pending_raw.extend(audio_chunk.iter().copied());
            log_first_nan("chain input audio_chunk", audio_chunk);
            log_first_nan("chain input gain", &gain);
            if let Some(stage) = self.tse_stage.as_ref() {
                log_first_nan("chain input cond_embedding", stage.cond_embedding());
            }
            // Sub-LSB dither (±1e-7 ≈ -140 dBFS, below the 24-bit
            // noise floor) applied **unconditionally** so no STFT
            // window inside the chain can be exact-zero variance.
            // The conditional "all samples below 1e-9" gate this
            // replaced missed the leading-silence-then-speech case
            // (the first STFT window saw pure zeros, latched NaN
            // into the GRU state, contaminated the whole run).
            let chain_audio_owned: Vec<f32> = audio_chunk
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    s + if i % 2 == 0 {
                        CHAIN_DITHER
                    } else {
                        -CHAIN_DITHER
                    }
                })
                .collect();
            let chain_audio: &[f32] = &chain_audio_owned;

            // Adaptive XOR routing. When the overlap detector is wired
            // up, exactly one of TSE / DFN3 runs per chunk based on the
            // current `chain_mode`. Otherwise (`adaptive == false`) the
            // legacy cascade (TSE → DFN3, with either / both possibly
            // absent) runs.
            let after_tse: Vec<f32> = if let Some(stage) = self.tse_stage.as_mut() {
                if adaptive && self.chain_mode == ChainMode::Solo {
                    chain_audio.to_vec()
                } else {
                    stage.process(chain_audio).map_err(PipelineError::from)?
                }
            } else {
                chain_audio.to_vec()
            };
            log_first_nan("after TSE", &after_tse);
            let after_dfn3: Vec<f32> = if let Some(stream) = self.dfn3_stream.as_mut() {
                if adaptive && self.chain_mode == ChainMode::Overlap {
                    after_tse
                } else {
                    stream.push_samples(&after_tse)?
                }
            } else {
                after_tse
            };
            log_first_nan("after DFN3", &after_dfn3);
            // Mode-aware level recovery. The universal NS → makeup
            // pattern (Krisp / Teams / Zoom) — applied **before**
            // the envelope multiply so gate-off frames stay silent.
            //
            // Solo (DFN3-only): static dB gain. DFN3's attenuation
            // is fairly uniform across speech levels so a fixed
            // offset is appropriate.
            //
            // Overlap (TSE-only): RMS-match to the raw input over
            // this emit chunk (Asteroid's canonical Conv-TasNet
            // rescale). Conv-TasNet outputs are scale-arbitrary so
            // a static gain over-amplifies quiet stretches and
            // under-amplifies loud ones. The match ratio is
            // clamped to `overlap_rms_match_max_gain_db` so a
            // silent TSE chunk doesn't get boosted into pure
            // noise. Followed by a wet / dry mix that re-injects a
            // small amount of raw audio to mask Conv-TasNet's
            // frame-to-frame mask artefacts.
            let soft_clip_enabled = config.chain_soft_clip;
            let makeup_solo_lin = if config.makeup_gain_db_solo == 0.0 {
                1.0
            } else {
                10.0_f32.powf(config.makeup_gain_db_solo / 20.0)
            };
            let makeup_overlap_static_lin = if config.makeup_gain_db_overlap == 0.0 {
                1.0
            } else {
                10.0_f32.powf(config.makeup_gain_db_overlap / 20.0)
            };
            let rms_match_lin = if self.chain_mode == ChainMode::Overlap
                && config.overlap_rms_match
                && !after_dfn3.is_empty()
            {
                let n = after_dfn3.len();
                let in_sum_sq: f32 = self.chain_pending_raw.iter().take(n).map(|s| s * s).sum();
                let in_rms = (in_sum_sq / n.max(1) as f32).sqrt();
                let out_sum_sq: f32 = after_dfn3.iter().map(|s| s * s).sum();
                let out_rms = (out_sum_sq / n.max(1) as f32).sqrt();
                let raw_ratio = in_rms.max(1e-9) / out_rms.max(1e-9);
                let max_lin = 10.0_f32.powf(config.overlap_rms_match_max_gain_db / 20.0);
                raw_ratio.min(max_lin)
            } else {
                1.0
            };
            let alpha = config.overlap_wet_dry_alpha.clamp(0.0, 1.0);

            for &x in &after_dfn3 {
                let g = self
                    .chain_pending_gain
                    .pop_front()
                    .expect("chain_pending_gain length matches chain input cumulative length");
                let raw = self
                    .chain_pending_raw
                    .pop_front()
                    .expect("chain_pending_raw length matches chain input cumulative length");
                let chain_processed = match self.chain_mode {
                    ChainMode::Solo => x * makeup_solo_lin,
                    ChainMode::Overlap => x * rms_match_lin * makeup_overlap_static_lin,
                };
                // Wet / dry mix: identity in Solo (alpha is bypassed),
                // configured value in Overlap.
                let mixed = if self.chain_mode == ChainMode::Overlap && alpha < 1.0 {
                    alpha * chain_processed + (1.0 - alpha) * raw
                } else {
                    chain_processed
                };
                let limited = if soft_clip_enabled {
                    mixed.tanh()
                } else {
                    mixed
                };
                let sample = limited * g;
                // Belt-and-braces: replace any non-finite value
                // with 0 so a poisoned model state can't push NaN /
                // Inf to the DAC even if the dither earlier didn't
                // help.
                out.audio
                    .push(if sample.is_finite() { sample } else { 0.0 });
            }
            out.tse_applied = self.tse_stage.is_some();
        } else {
            for (k, &g) in gain.iter().enumerate() {
                out.audio.push(audio_chunk[k] * g);
            }
        }
        self.audio_samples_emitted += audio_chunk.len();
        self.frame_idx += 1;
        Ok(())
    }

    /// Feed one VAD-frame-worth of decision-rate audio to the overlap
    /// detector (when present), then advance the hysteresis state
    /// machine that decides which chain — TSE or DFN3 — handles the
    /// next audio_chunk.
    ///
    /// On a `Solo ↔ Overlap` transition the stage being newly
    /// activated has its state reset and `chain_pending_gain` is
    /// cleared. Resetting the freshly-active stage avoids carrying
    /// stale GRU / accumulator state from when it last ran; clearing
    /// the gain queue drops the in-flight samples that the old stage
    /// had buffered (a sub-chunk gap, typically < 30 ms).
    fn update_chain_mode(
        &mut self,
        decision_frame: &[f32],
        _dt_ms: f32,
        config: &StreamingConfig,
    ) -> Result<(), PipelineError> {
        let Some(detector) = self.overlap_detector.as_mut() else {
            return Ok(());
        };
        if let Some(decision) = detector
            .push(decision_frame)
            .map_err(PipelineError::Embedding)?
        {
            // Each detector firing represents `cadence_ms` of audio
            // since the previous one — use that as the hysteresis
            // increment rather than the per-VAD-frame dt_ms, which
            // would undercount by ~8x at the default 250 ms cadence
            // and leave the engine stuck in whichever mode it
            // entered first.
            let cadence_ms = detector.cadence_ms();
            if decision.mean_overlap_prob >= config.overlap_threshold {
                self.overlap_above_ms += cadence_ms;
                self.overlap_below_ms = 0.0;
            } else {
                self.overlap_below_ms += cadence_ms;
                self.overlap_above_ms = 0.0;
            }
            let new_mode = match self.chain_mode {
                ChainMode::Solo if self.overlap_above_ms >= config.overlap_hold_on_ms => {
                    ChainMode::Overlap
                }
                ChainMode::Overlap if self.overlap_below_ms >= config.overlap_hold_off_ms => {
                    ChainMode::Solo
                }
                other => other,
            };
            if new_mode != self.chain_mode {
                eprintln!(
                    "[streaming] chain mode {:?} → {:?} (overlap_prob mean {:.3})",
                    self.chain_mode, new_mode, decision.mean_overlap_prob
                );
                self.chain_mode = new_mode;
                // Reset the chain whose state was idle so the newly-
                // active path starts fresh; drop pending gains held
                // behind the now-bypassed stage.
                self.chain_pending_gain.clear();
                self.chain_pending_raw.clear();
                if let Some(t) = self.tse_stage.as_mut() {
                    t.reset();
                }
                if let Some(d) = self.dfn3_stream.as_mut() {
                    d.reset();
                }
                self.overlap_above_ms = 0.0;
                self.overlap_below_ms = 0.0;
            }
        }
        Ok(())
    }

    /// Single VAD-frame iteration, sync mode: builds a
    /// [`SyncRefresh`] over the borrowed components and runs the
    /// shared [`Self::step_one_frame_core`].
    fn step_one_frame(
        &mut self,
        audio_chunk: &[f32],
        decision_frame: &[f32],
        pool: &mut EmbeddingPool,
        components: &mut PipelineComponents,
        config: &StreamingConfig,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        // Disjoint borrows: `vad` drives the core, `fbank` / `ecapa` /
        // `cohort` go into the refresh strategy. The `tse` field is
        // ignored here — Phase 5 part 3 moves the live `TseStage` out
        // of `components.tse` and into `StreamingState::tse_stage` at
        // construction time so the buffered accumulator persists
        // across `push_samples` calls; `components.tse` is left as
        // `None` for the duration of the run (handed back through
        // `StreamingPipeline::into_parts`).
        let PipelineComponents {
            vad,
            fbank,
            ecapa,
            cohort,
            tse: _,
        } = components;
        let mut strategy = SyncRefresh {
            fbank,
            ecapa,
            cohort,
            pool,
            decision_sr: config.pipeline.sample_rate,
            enable_auto_learn: config.pipeline.enable_auto_learn,
            score_ema_alpha: config.pipeline.score_ema_alpha,
            use_as_norm: config.gate.use_as_norm,
            gate_cfg: config.gate,
        };
        self.step_one_frame_core(audio_chunk, decision_frame, vad, &mut strategy, config, out)
    }

    /// Single VAD-frame iteration, async mode: builds an
    /// [`AsyncRefresh`] over the persistent worker and runs the
    /// shared [`Self::step_one_frame_core`]. The worker is taken out
    /// of `self` for the duration of the call so the strategy can
    /// hold a `&mut` to it disjointly from `&mut self`.
    fn step_one_frame_async(
        &mut self,
        audio_chunk: &[f32],
        decision_frame: &[f32],
        pool: &mut EmbeddingPool,
        vad: &mut SileroVad,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        let mut worker = self
            .async_worker
            .take()
            .expect("step_one_frame_async requires an async worker");
        let mut strategy = AsyncRefresh {
            channel: &mut worker,
            cohort,
            pool,
            enable_auto_learn: config.pipeline.enable_auto_learn,
            score_ema_alpha: config.pipeline.score_ema_alpha,
            gate_cfg: config.gate,
        };
        let res =
            self.step_one_frame_core(audio_chunk, decision_frame, vad, &mut strategy, config, out);
        self.async_worker = Some(worker);
        res
    }

    /// Blocking-drain every outstanding inference on `channel` and
    /// fold the results into `pool` / `self.score` / `out.events`.
    ///
    /// Shared end-of-stream tail for both async paths
    /// (`StreamingState::flush_async` and `process_offline_async`).
    /// Per-frame audio / gate decisions were already emitted with
    /// whatever `last_score` held at the time; these trailing
    /// results only affect `pool` and the auto-learn event log, and
    /// the score carried forward for any subsequent run.
    pub(crate) fn drain_trailing_refreshes(
        &mut self,
        channel: &mut dyn RefreshChannel,
        pool: &mut EmbeddingPool,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
        out: &mut StreamingOutput,
    ) -> Result<(), PipelineError> {
        for (trigger_frame, embedding, f0_mu) in channel.drain_blocking()? {
            apply_refresh_result(
                embedding,
                f0_mu,
                trigger_frame,
                self.consecutive_speech_ms,
                pool,
                cohort,
                &config.gate,
                config.pipeline.enable_auto_learn,
                config.pipeline.score_ema_alpha,
                &mut self.score,
                &mut out.events,
            );
        }
        Ok(())
    }

    /// Drain as many full VAD frames as the audio_ring currently
    /// supports, returning the accumulated [`StreamingOutput`].
    pub(crate) fn push_block(
        &mut self,
        samples: &[f32],
        pool: &mut EmbeddingPool,
        components: &mut PipelineComponents,
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        self.audio_ring.extend(samples.iter().copied());
        let mut out = StreamingOutput::default();
        while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
            self.step_one_frame(
                &audio_chunk,
                &decision_chunk,
                pool,
                components,
                config,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Async counterpart of [`Self::push_block`]. Takes the
    /// async-friendly subset of components (vad + cohort) since
    /// fbank + ecapa live in the worker thread.
    pub(crate) fn push_block_async(
        &mut self,
        samples: &[f32],
        pool: &mut EmbeddingPool,
        vad: &mut SileroVad,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        self.audio_ring.extend(samples.iter().copied());
        let mut out = StreamingOutput::default();
        while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
            self.step_one_frame_async(
                &audio_chunk,
                &decision_chunk,
                pool,
                vad,
                cohort,
                config,
                &mut out,
            )?;
        }
        Ok(out)
    }

    /// Zero-pad any residual audio-rate samples to the resampler's
    /// next-expected input size so the trailing audio gets one last
    /// decision pass. Idempotent on a fully-drained state.
    pub(crate) fn flush(
        &mut self,
        pool: &mut EmbeddingPool,
        components: &mut PipelineComponents,
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        let mut out = StreamingOutput::default();
        if !self.audio_ring.is_empty() {
            let n_input = self.input_per_frame();
            if self.audio_ring.len() < n_input {
                // Pad with silence so one more frame can flow through.
                let pad = n_input - self.audio_ring.len();
                self.audio_ring.extend(std::iter::repeat(0.0_f32).take(pad));
            }
            // Drain remaining whole frames (may be > 1 if the caller
            // pushed a chunk that's just shy of multiple frames).
            // `drain_one_frame` itself returns `None` when the ring no
            // longer has a full frame's worth of samples.
            while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
                self.step_one_frame(
                    &audio_chunk,
                    &decision_chunk,
                    pool,
                    components,
                    config,
                    &mut out,
                )?;
            }
        }
        // Stage C, Phase 5: drain the TSE→DFN3 chain tail so the final
        // samples emerge. TSE.flush zero-pads its partial chunk and
        // runs one last inference; DFN3.push consumes that tail and
        // DFN3.flush drains its conv-lookahead frames. Pending gains
        // are applied to as many emitted samples as we have queued;
        // the rest correspond to zero-padded fillers and are dropped.
        self.drain_chain_tail(&mut out)?;
        Ok(out)
    }

    /// Drain any partial chunk left in the TSE accumulator and the
    /// DFN3 lookahead buffer into `out.audio`, applying queued gains
    /// 1-to-1 with the chain's final emit. No-op when neither stage is
    /// configured or the pending-gain queue is empty.
    fn drain_chain_tail(&mut self, out: &mut StreamingOutput) -> Result<(), PipelineError> {
        if self.tse_stage.is_none() && self.dfn3_stream.is_none() {
            return Ok(());
        }
        if self.chain_pending_gain.is_empty() {
            return Ok(());
        }
        // Flush TSE first; if no TSE, the tail is empty.
        let tse_tail = if let Some(stage) = self.tse_stage.as_mut() {
            stage.flush().map_err(PipelineError::from)?
        } else {
            Vec::new()
        };
        // Pass the TSE tail through DFN3 (if active) and then flush
        // DFN3's lookahead so the very last hop emerges.
        let chain_tail = if let Some(stream) = self.dfn3_stream.as_mut() {
            let mut tail = if tse_tail.is_empty() {
                Vec::new()
            } else {
                stream.push_samples(&tse_tail)?
            };
            tail.extend(stream.flush()?);
            tail
        } else {
            tse_tail
        };
        // Emit as many samples as we have pending gains for; any
        // surplus came from TSE's zero-pad and is discarded.
        let n_take = chain_tail.len().min(self.chain_pending_gain.len());
        for &x in &chain_tail[..n_take] {
            let g = self
                .chain_pending_gain
                .pop_front()
                .expect("n_take <= chain_pending_gain.len()");
            out.audio.push(x * g);
        }
        self.chain_pending_gain.clear();
        self.chain_pending_raw.clear();
        out.tse_applied = self.tse_stage.is_some();
        Ok(())
    }

    /// Async counterpart of [`Self::flush`]. Drains any residue +
    /// also blocks on outstanding worker inferences so the trailing
    /// auto-learn events / score updates are captured before the
    /// caller tears the pipeline down.
    pub(crate) fn flush_async(
        &mut self,
        pool: &mut EmbeddingPool,
        vad: &mut SileroVad,
        cohort: &[Vec<f32>],
        config: &StreamingConfig,
    ) -> Result<StreamingOutput, PipelineError> {
        let mut out = StreamingOutput::default();
        // Same audio-frame zero-pad path as the sync flush.
        if !self.audio_ring.is_empty() {
            let n_input = self.input_per_frame();
            if self.audio_ring.len() < n_input {
                let pad = n_input - self.audio_ring.len();
                self.audio_ring.extend(std::iter::repeat(0.0_f32).take(pad));
            }
            while let Some((audio_chunk, decision_chunk)) = self.drain_one_frame()? {
                self.step_one_frame_async(
                    &audio_chunk,
                    &decision_chunk,
                    pool,
                    vad,
                    cohort,
                    config,
                    &mut out,
                )?;
            }
        }
        // Drain any in-flight ECAPA work so its scores + auto-learn
        // events make it into the output.
        if let Some(mut worker) = self.async_worker.take() {
            self.drain_trailing_refreshes(&mut worker, pool, cohort, config, &mut out)?;
            self.async_worker = Some(worker);
        }
        // Stage C, Phase 5: drain the TSE→DFN3 chain tail. Mirrors the
        // sync flush so DFN3's lookahead buffer flushes deterministically
        // even in async-refresh mode. (TSE is rejected in async mode at
        // construction time, so only the DFN3-only branch can fire here.)
        self.drain_chain_tail(&mut out)?;
        Ok(out)
    }

    /// Take ownership of the worker (only meaningful in async
    /// mode). Returns the recovered `Fbank` + `EcapaTdnn` after
    /// joining the thread. Used by
    /// [`StreamingPipeline::into_parts`].
    pub(crate) fn shutdown_worker(&mut self) -> Result<Option<(Fbank, EcapaTdnn)>, PipelineError> {
        let Some(worker) = self.async_worker.take() else {
            return Ok(None);
        };
        worker.shutdown().map(Some)
    }
}

/// Stateful, single-target speaker gating pipeline driven by
/// incremental sample pushes.
///
/// Supports both sync mode (`async_refresh = false`, inline
/// Fbank / ECAPA on the main thread) and async mode
/// (`async_refresh = true`, persistent worker thread running
/// Fbank / ECAPA / F0 in parallel with VAD + gating on the
/// caller's thread). See the module-level docs for the buffering
/// model, parity contract, and ownership rules.
pub struct StreamingPipeline {
    state: StreamingState,
    config: StreamingConfig,
    pool: EmbeddingPool,
    components: ComponentsStorage,
}

impl StreamingPipeline {
    /// Build a streaming pipeline. The pool and components are
    /// moved in; recover them via [`Self::into_parts`].
    ///
    /// When `config.pipeline.async_refresh = true`, the components'
    /// `fbank` + `ecapa` move into a persistent worker thread for
    /// the lifetime of the pipeline; `into_parts` re-joins the
    /// worker to reconstruct the original [`PipelineComponents`]
    /// struct.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if the resampler can't be built
    /// (only when `audio_sample_rate != pipeline.sample_rate`), if
    /// spawning the async worker fails, or — when
    /// `config.pipeline.tse.is_some()` — if either the rate check
    /// (`PipelineError::TseRateMismatch`), the async-mode rejection
    /// (`PipelineError::TseAsyncUnsupported`), or the cond-embedding
    /// snapshot (`PipelineError::TseMissingEnrollment`) fails, or the
    /// TSE ONNX session can't be loaded.
    ///
    /// # Panics
    ///
    /// Panics only on internal invariant violation — the
    /// `tse_enabled implies Some` `expect` is unreachable because the
    /// surrounding `if tse_enabled` reads the same flag.
    pub fn new(
        pool: EmbeddingPool,
        config: StreamingConfig,
        mut components: PipelineComponents,
    ) -> Result<Self, PipelineError> {
        // Stage C, Phase 5 part 3 + Step 4: TSE wiring for streaming.
        // Both sync (`async_refresh=false`) and async
        // (`async_refresh=true`) paths now support TSE. The cond
        // embedding is snapshotted **once** here (mirroring offline's
        // "frozen for the run" semantics); the embedding pool may
        // evolve via auto-learn during streaming but the TSE stage
        // keeps the construction-time anchor centroid as its
        // conditioning until [`Self::refresh_tse_cond_from_pool`] is
        // called.
        //
        // TODO(phase-5-step5): refresh the cond embedding on each
        // auto-learn anchor update.
        let tse_enabled = config.pipeline.tse.is_some();
        if tse_enabled {
            let stage_cfg = config
                .pipeline
                .tse
                .as_ref()
                .expect("tse_enabled implies Some");
            let expected_sr = stage_cfg.model.sample_rate();
            if config.audio_sample_rate != expected_sr {
                return Err(PipelineError::TseRateMismatch {
                    audio_sr: config.audio_sample_rate,
                    expected_sr,
                });
            }
        }

        // Pre-build the TseStage (shared between sync and async
        // branches) before destructuring `components` for the async
        // path. Mirrors `ensure_tse_stage` in `pipeline.rs`.
        let live_stage = if tse_enabled {
            let stage_cfg = config
                .pipeline
                .tse
                .as_ref()
                .expect("tse_enabled implies Some");
            let cond = tse_cond_embedding(&pool)?;
            let stage = if let Some(mut existing) = components.tse.take() {
                existing.reset();
                existing
                    .set_cond_embedding(&cond)
                    .map_err(PipelineError::from)?;
                existing
            } else {
                TseStage::from_config(stage_cfg, &cond).map_err(PipelineError::from)?
            };
            Some(stage)
        } else {
            None
        };

        // Stage C, Phase 5: optional DFN3 noise-suppression stream.
        // Sits after TSE in the audio chain. The DFN3 model is 48 kHz
        // only; reject any audio_sample_rate mismatch up-front so the
        // engine doesn't quietly produce garbage at the wrong rate.
        let dfn3_stream = if let Some(path) = config.dfn3_onnx_path.as_deref() {
            let expected_sr = u32::try_from(crate::dfn3::DFN3_SR).expect("DFN3_SR fits in u32");
            if config.audio_sample_rate != expected_sr {
                return Err(PipelineError::TseRateMismatch {
                    audio_sr: config.audio_sample_rate,
                    expected_sr,
                });
            }
            Some(Dfn3Streamer::from_onnx_path(path)?)
        } else {
            None
        };

        // Adaptive-chain overlap detector. Only attached when **all
        // three** of TSE, DFN3, and the overlap ONNX are configured —
        // there's nothing to switch between otherwise.
        let overlap_detector = if config.overlap_onnx_path.is_some()
            && live_stage.is_some()
            && dfn3_stream.is_some()
        {
            let path = config.overlap_onnx_path.as_deref().expect("Some");
            Some(
                crate::overlap::OverlapDetector::from_onnx_path(path)
                    .map_err(PipelineError::Embedding)?,
            )
        } else {
            None
        };

        if config.pipeline.async_refresh {
            let PipelineComponents {
                vad,
                fbank,
                ecapa,
                cohort,
                // `tse` was already consumed by the shared `live_stage`
                // builder above; this destructure exhausts the struct.
                tse: _,
            } = components;
            let mut state = StreamingState::new_async(&config, fbank, ecapa)?;
            state.tse_stage = live_stage;
            state.dfn3_stream = dfn3_stream;
            state.overlap_detector = overlap_detector;
            Ok(Self {
                state,
                config,
                pool,
                components: ComponentsStorage::Async { vad, cohort },
            })
        } else {
            let mut state = StreamingState::new(&config)?;
            state.tse_stage = live_stage;
            state.dfn3_stream = dfn3_stream;
            state.overlap_detector = overlap_detector;
            Ok(Self {
                state,
                config,
                pool,
                components: ComponentsStorage::Sync(Box::new(components)),
            })
        }
    }

    /// Push an arbitrary-length chunk of `audio_sample_rate` Hz f32
    /// mono samples (range `[-1.0, 1.0]`).
    ///
    /// Returns the gated output, at the same sample rate,
    /// corresponding to all VAD frames that could be completed with
    /// the new samples. Sub-frame residue at the audio rate is
    /// buffered and produces no output until the next call (or a
    /// [`Self::flush`]).
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if an underlying ONNX inference
    /// fails, the resampler step fails, or (in async mode) the
    /// worker thread disconnects.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<StreamingOutput, PipelineError> {
        match &mut self.components {
            ComponentsStorage::Sync(c) => {
                self.state
                    .push_block(samples, &mut self.pool, c.as_mut(), &self.config)
            }
            ComponentsStorage::Async { vad, cohort } => {
                self.state
                    .push_block_async(samples, &mut self.pool, vad, cohort, &self.config)
            }
        }
    }

    /// Flush any residual sub-VAD-frame samples by zero-padding to
    /// the resampler's next-expected input size so the trailing
    /// audio gets one last decision pass. In async mode, also
    /// blocks on any in-flight ECAPA inferences so trailing
    /// scores / auto-learn events make it into the output.
    ///
    /// Call this once at end-of-stream (e.g. when the audio device
    /// closes) to avoid losing the tail.
    ///
    /// # Errors
    ///
    /// Same as [`Self::push_samples`].
    pub fn flush(&mut self) -> Result<StreamingOutput, PipelineError> {
        match &mut self.components {
            ComponentsStorage::Sync(c) => {
                self.state.flush(&mut self.pool, c.as_mut(), &self.config)
            }
            ComponentsStorage::Async { vad, cohort } => {
                self.state
                    .flush_async(&mut self.pool, vad, cohort, &self.config)
            }
        }
    }

    /// Read-only access to the owned pool.
    #[must_use]
    pub fn pool(&self) -> &EmbeddingPool {
        &self.pool
    }

    /// Current adaptive chain mode. `ChainMode::Solo` when DFN3 alone
    /// is processing the audio; `ChainMode::Overlap` when TSE alone
    /// is. Only meaningful when [`StreamingConfig::overlap_onnx_path`]
    /// is `Some` (and TSE + DFN3 are also configured); otherwise the
    /// value is the construction default (`Solo`) and not consulted
    /// by the chain.
    #[must_use]
    pub fn chain_mode(&self) -> ChainMode {
        self.state.chain_mode
    }

    /// Mutable access to the owned pool. Safe between
    /// `push_samples` / `flush` calls.
    pub fn pool_mut(&mut self) -> &mut EmbeddingPool {
        &mut self.pool
    }

    /// Reset stateful pieces (rings, gate, envelope, frame index)
    /// **without** rebuilding ONNX sessions or tearing down the
    /// async worker thread (if any). Pool is preserved.
    pub fn reset(&mut self) {
        self.state.reset(&self.config);
    }

    /// Tear the pipeline down, returning the owned pool and
    /// components. In async mode, this joins the worker thread
    /// (waking it via a channel close) and recombines `fbank` +
    /// `ecapa` with the main-thread `vad` + `cohort` into the
    /// original [`PipelineComponents`].
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if the async worker panicked.
    ///
    /// # Panics
    ///
    /// Panics if the storage is in `Async` mode but no worker is
    /// present — this is an unreachable invariant violation
    /// (`Async` is only ever constructed with a freshly-spawned
    /// worker).
    pub fn into_parts(mut self) -> Result<(EmbeddingPool, PipelineComponents), PipelineError> {
        let components = match self.components {
            ComponentsStorage::Sync(c) => {
                let mut components = *c;
                // Stage C, Phase 5 part 3: hand the live TSE stage
                // back to the caller via `components.tse` so it can be
                // persisted / inspected. The stage's accumulator is
                // already drained (callers should `flush()` before
                // `into_parts`), but the loaded ONNX session itself is
                // worth preserving — building it costs ~tens of ms.
                if let Some(stage) = self.state.tse_stage.take() {
                    components.tse = Some(stage);
                }
                components
            }
            ComponentsStorage::Async { vad, cohort } => {
                let (fbank, ecapa) = self
                    .state
                    .shutdown_worker()?
                    .expect("async storage always holds a worker");
                PipelineComponents {
                    vad,
                    fbank,
                    ecapa,
                    cohort,
                    // Stage C TSE in the async-refresh streaming worker
                    // is rejected up-front by `StreamingPipeline::new`
                    // (it returns `PipelineError::TseAsyncUnsupported`
                    // before reaching this branch), so the recovered
                    // `PipelineComponents` is always TSE-less here.
                    // The live TSE state in a sync pipeline is handed
                    // back via the `Sync` branch above.
                    tse: None,
                }
            }
        };
        Ok((self.pool, components))
    }
}

#[cfg(test)]
// The Stage B tests deliberately use exact `f32` equality (the
// identity / disabled paths must be bit-exact) and cast loop indices
// to `f32` for synthetic-tone generation — both inert in test code.
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn streaming_config_default_is_dual_rate() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.audio_sample_rate, 48_000);
        assert_eq!(cfg.pipeline.sample_rate, 16_000);
        assert!(!cfg.diagnostics);
    }

    #[test]
    fn state_new_identity_rate_has_no_resampler() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate;
        let state = StreamingState::new(&cfg).expect("identity-rate state");
        assert!(state.resampler.is_none());
        assert_eq!(state.identity_input_per_frame, CHUNK_SAMPLES_16K);
    }

    #[test]
    fn state_new_dual_rate_builds_resampler() {
        let cfg = StreamingConfig::default();
        let state = StreamingState::new(&cfg).expect("dual-rate state");
        assert!(state.resampler.is_some());
    }

    #[test]
    fn state_new_initialises_pre_roll_capacity_from_config() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate; // identity, simpler
        cfg.pipeline.pre_roll_ms = 200;
        let state = StreamingState::new(&cfg).expect("state");
        // 200 ms * 16 000 Hz / 1000 = 3 200 samples
        assert_eq!(state.pre_roll_capacity, 3_200);
        assert!(state.pre_roll_ring.is_empty());
    }

    #[test]
    fn push_pre_roll_caps_at_capacity() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate;
        cfg.pipeline.pre_roll_ms = 32; // 32 ms = 512 samples = 1 frame
        let mut state = StreamingState::new(&cfg).expect("state");
        let frame = vec![1.0_f32; CHUNK_SAMPLES_16K];
        state.push_pre_roll(&frame);
        assert_eq!(state.pre_roll_ring.len(), 512);
        // Pushing a second frame must keep the cap.
        let frame2 = vec![2.0_f32; CHUNK_SAMPLES_16K];
        state.push_pre_roll(&frame2);
        assert_eq!(state.pre_roll_ring.len(), 512);
        // After two pushes the ring should contain only the second
        // frame's samples (oldest popped).
        assert!(state.pre_roll_ring.iter().all(|&s| (s - 2.0).abs() < 1e-9));
    }

    #[test]
    fn push_pre_roll_is_noop_when_disabled() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate;
        cfg.pipeline.pre_roll_ms = 0;
        let mut state = StreamingState::new(&cfg).expect("state");
        let frame = vec![1.0_f32; CHUNK_SAMPLES_16K];
        state.push_pre_roll(&frame);
        assert!(state.pre_roll_ring.is_empty());
        assert_eq!(state.pre_roll_capacity, 0);
    }

    #[test]
    fn prepend_pre_roll_pushes_ring_into_speech_buffer_back() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate;
        cfg.pipeline.pre_roll_ms = 32;
        let mut state = StreamingState::new(&cfg).expect("state");
        // Populate the ring with a marker value.
        let frame = vec![0.5_f32; CHUNK_SAMPLES_16K];
        state.push_pre_roll(&frame);
        assert_eq!(state.pre_roll_ring.len(), 512);
        // Speech buffer empty before the OFF→ON transition.
        assert!(state.speech_buffer.is_empty());
        state.prepend_pre_roll_to_speech_buffer(cfg.pipeline.sv_window_samples);
        // Ring contents are now at the back of the speech buffer in
        // chronological order. Subsequent frames append after.
        assert_eq!(state.speech_buffer.len(), 512);
        assert!(state.speech_buffer.iter().all(|&s| (s - 0.5).abs() < 1e-9));
    }

    #[test]
    fn prepend_pre_roll_respects_sv_window_cap() {
        let mut cfg = StreamingConfig::default();
        cfg.audio_sample_rate = cfg.pipeline.sample_rate;
        cfg.pipeline.pre_roll_ms = 32;
        let mut state = StreamingState::new(&cfg).expect("state");
        // Pre-fill the speech buffer to one short of the cap.
        let cap = cfg.pipeline.sv_window_samples;
        for _ in 0..(cap - 100) {
            state.speech_buffer.push_back(0.1);
        }
        // Fill the ring with 512 samples of a different marker.
        let frame = vec![0.9_f32; CHUNK_SAMPLES_16K];
        state.push_pre_roll(&frame);
        state.prepend_pre_roll_to_speech_buffer(cap);
        // Buffer length stays at the cap; the oldest entries got
        // dropped to make room.
        assert_eq!(state.speech_buffer.len(), cap);
        let last_512: Vec<f32> = state
            .speech_buffer
            .iter()
            .rev()
            .take(512)
            .copied()
            .collect();
        assert!(last_512.iter().all(|&s| (s - 0.9).abs() < 1e-9));
    }

    #[test]
    fn state_new_rejects_async_refresh_on_sync_path() {
        // Sync `new` is for `async_refresh = false` only. Async
        // mode requires `new_async` (which routes through the
        // worker-spawning path; not exercised here because spawning
        // needs real Fbank + EcapaTdnn).
        let mut cfg = StreamingConfig::default();
        cfg.pipeline.async_refresh = true;
        match StreamingState::new(&cfg) {
            Err(PipelineError::Embedding(_)) => {}
            Err(other) => panic!("expected Embedding error, got: {other}"),
            Ok(_) => panic!("sync `new` must reject async_refresh = true"),
        }
    }

    // ---- Stage B, Part 1: fast F0 cue fusion --------------------

    #[test]
    fn fuse_fast_cue_is_identity_at_neutral() {
        // "No evidence" feeds `fm_fast == neutral`, so the fused
        // score must equal `last_score` exactly — this is the
        // arithmetic reason the disabled / no-evidence path is
        // byte-identical to today.
        let neutral = 0.5_f32;
        for &ls in &[0.0_f32, 0.31, 0.7, -0.2] {
            assert_eq!(fuse_fast_cue(ls, neutral, 0.15, neutral), ls);
        }
    }

    #[test]
    fn fuse_fast_cue_nudges_up_and_down() {
        // Above-neutral cue nudges up, below-neutral nudges down,
        // scaled by the weight.
        let up = fuse_fast_cue(0.3, 1.0, 0.2, 0.5);
        let down = fuse_fast_cue(0.3, 0.0, 0.2, 0.5);
        assert!((up - (0.3 + 0.2 * 0.5)).abs() < 1e-6);
        assert!((down - (0.3 - 0.2 * 0.5)).abs() < 1e-6);
        assert!(up > 0.3 && down < 0.3);
    }

    #[test]
    fn fast_f0_cue_no_evidence_until_ring_full() {
        // The ring needs FAST_F0_RING_SAMPLES before YIN runs;
        // before that every frame is "no evidence".
        let mut cue = FastF0Cue::new();
        let frame = vec![0.0_f32; CHUNK_SAMPLES_16K];
        let mut pushed = 0;
        while pushed + CHUNK_SAMPLES_16K <= FAST_F0_RING_SAMPLES {
            assert!(
                cue.push_frame(&frame, 16_000).is_none(),
                "ring not full yet at {pushed} samples"
            );
            pushed += CHUNK_SAMPLES_16K;
        }
        // One more frame fills the ring; silence is unvoiced so YIN
        // still returns None — but the point is the ring-fill gate
        // no longer suppresses it.
        let _ = cue.push_frame(&frame, 16_000);
    }

    #[test]
    fn fast_f0_cue_recovers_pure_tone() {
        // A sustained 150 Hz tone through a full ring should give a
        // voiced estimate near 150 Hz.
        let mut cue = FastF0Cue::new();
        let sr = 16_000_u32;
        let mut last: Option<f32> = None;
        for frame_idx in 0..16 {
            let mut frame = vec![0.0_f32; CHUNK_SAMPLES_16K];
            for (i, s) in frame.iter_mut().enumerate() {
                let n = frame_idx * CHUNK_SAMPLES_16K + i;
                *s = (std::f32::consts::TAU * 150.0 * n as f32 / sr as f32).sin();
            }
            last = cue.push_frame(&frame, sr);
        }
        let est = last.expect("voiced estimate once ring is full");
        assert!((est - 150.0).abs() / 150.0 < 0.05, "est={est}");
    }

    // ---- Stage B, Part 2: turn detection ------------------------

    /// Turn-detection cfg helper: enabled, with the rest at defaults.
    fn turn_cfg() -> PipelineConfig {
        PipelineConfig {
            turn_detect_enabled: true,
            ..PipelineConfig::default()
        }
    }

    #[test]
    fn turn_detector_inert_when_disabled() {
        // With `turn_detect_enabled = false` the detector never
        // leaves `Steady` and `effective_window` stays at
        // `sv_window_samples`, whatever the cue does.
        let cfg = PipelineConfig::default();
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        for fm in [0.9_f32, 0.1, 0.9, 0.05, 0.95] {
            let d = det.observe(Some(fm), true, &cfg);
            assert!(!d.entered_shrunk && !d.shrunk);
            assert_eq!(det.state, TurnState::Steady);
            assert_eq!(det.effective_window, cfg.sv_window_samples);
        }
    }

    #[test]
    fn turn_detector_shrinks_on_sharp_drop() {
        // Gate ON, baseline established high, then the cue collapses
        // → offset-suspect fires, window shrinks, `due_turn` would
        // fire (entered_shrunk == true).
        let cfg = turn_cfg();
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        // Seed + hold a high baseline while the gate is ON.
        for _ in 0..40 {
            det.observe(Some(0.95), true, &cfg);
        }
        assert_eq!(det.state, TurnState::Steady);
        assert!(det.fm_fast_baseline > 0.8);
        // Sharp drop in the cue — push it well below
        // baseline - turn_drop_delta.
        let mut entered = false;
        for _ in 0..10 {
            let d = det.observe(Some(0.1), true, &cfg);
            entered |= d.entered_shrunk;
            if det.state == TurnState::Shrunk {
                break;
            }
        }
        assert!(entered, "expected an entered_shrunk edge");
        assert_eq!(det.state, TurnState::Shrunk);
        assert_eq!(det.effective_window, cfg.sv_turn_window_samples);
    }

    #[test]
    fn turn_detector_offset_fail_closed_lifecycle() {
        // With `offset_fail_closed`, an offset-suspect sets
        // `offset_failclosed_active`; a refresh-completed note
        // clears it.
        let cfg = PipelineConfig {
            offset_fail_closed: true,
            ..turn_cfg()
        };
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        for _ in 0..40 {
            det.observe(Some(0.95), true, &cfg);
        }
        for _ in 0..10 {
            det.observe(Some(0.1), true, &cfg);
            if det.state == TurnState::Shrunk {
                break;
            }
        }
        assert_eq!(det.state, TurnState::Shrunk);
        assert!(
            det.offset_failclosed_active,
            "offset-suspect must arm fail-closed"
        );
        det.note_refresh_completed();
        assert!(
            !det.offset_failclosed_active,
            "a shrunk-window refresh must clear fail-closed"
        );
        assert!(det.shrunk_refresh_done);
    }

    #[test]
    fn turn_detector_fail_closed_off_when_flag_unset() {
        // Turn detection on, but `offset_fail_closed` off — the
        // window still shrinks but the gate-override flag never arms.
        let cfg = turn_cfg();
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        for _ in 0..40 {
            det.observe(Some(0.95), true, &cfg);
        }
        for _ in 0..10 {
            det.observe(Some(0.1), true, &cfg);
            if det.state == TurnState::Shrunk {
                break;
            }
        }
        assert_eq!(det.state, TurnState::Shrunk);
        assert!(!det.offset_failclosed_active);
    }

    #[test]
    fn turn_detector_regrows_after_stable_and_refresh() {
        // Shrunk → Regrowing requires BOTH a completed shrunk-window
        // refresh AND `turn_stable_frames` of cue close to baseline;
        // Regrowing → Steady on the next frame, window restored.
        let cfg = turn_cfg();
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        for _ in 0..40 {
            det.observe(Some(0.95), true, &cfg);
        }
        for _ in 0..10 {
            det.observe(Some(0.1), true, &cfg);
            if det.state == TurnState::Shrunk {
                break;
            }
        }
        assert_eq!(det.state, TurnState::Shrunk);
        // The gate has recovered onto the (new) target: feed
        // confirmed-target frames (gate ON) with the cue pinned at
        // baseline. The EMA climbs back up; WITHOUT a refresh note
        // the detector must NOT regrow no matter how stable it gets.
        let baseline = det.fm_fast_baseline;
        for _ in 0..60 {
            det.observe(Some(baseline), true, &cfg);
        }
        assert_eq!(
            det.state,
            TurnState::Shrunk,
            "no regrow without a completed shrunk-window refresh"
        );
        // Now note a refresh; the cue is already pinned at baseline,
        // so the next `turn_stable_frames` frames must trip the
        // regrow.
        det.note_refresh_completed();
        for _ in 0..=cfg.turn_stable_frames {
            det.observe(Some(baseline), true, &cfg);
        }
        assert!(matches!(
            det.state,
            TurnState::Regrowing | TurnState::Steady
        ));
        // One more frame settles to Steady with the window restored.
        det.observe(Some(baseline), true, &cfg);
        assert_eq!(det.state, TurnState::Steady);
        assert_eq!(det.effective_window, cfg.sv_window_samples);
    }

    #[test]
    fn turn_detector_effective_window_shrink_then_grow() {
        // End-to-end window-length trace: starts at sv_window_samples,
        // shrinks to sv_turn_window_samples, returns to
        // sv_window_samples.
        let cfg = turn_cfg();
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        assert_eq!(det.effective_window, cfg.sv_window_samples);
        for _ in 0..40 {
            det.observe(Some(0.95), true, &cfg);
        }
        for _ in 0..10 {
            det.observe(Some(0.1), true, &cfg);
        }
        assert_eq!(det.effective_window, cfg.sv_turn_window_samples);
        det.note_refresh_completed();
        // Confirmed-target frames (gate ON) with the cue pinned at
        // baseline: the EMA climbs back, `turn_stable_frames` close
        // frames accumulate, the detector regrows and settles to
        // Steady with the window restored.
        let baseline = det.fm_fast_baseline;
        for _ in 0..80 {
            det.observe(Some(baseline), true, &cfg);
        }
        assert_eq!(det.effective_window, cfg.sv_window_samples);
    }

    #[test]
    fn turn_detector_due_turn_fires_once_on_shrink_edge() {
        // `TurnDecision.entered_shrunk` is the `due_turn` refresh
        // trigger the streaming core ORs in — it must fire exactly
        // on the shrink edge, then stay false while `Shrunk`.
        let cfg = turn_cfg();
        let mut det = TurnDetector::new(cfg.sv_window_samples);
        for _ in 0..40 {
            det.observe(Some(0.95), true, &cfg);
        }
        let mut edges = 0_u32;
        for _ in 0..20 {
            let d = det.observe(Some(0.05), true, &cfg);
            if d.entered_shrunk {
                edges += 1;
                assert!(d.shrunk, "entered_shrunk implies shrunk");
            }
        }
        assert_eq!(edges, 1, "due_turn must fire exactly once per turn");
        assert_eq!(det.state, TurnState::Shrunk);
    }

    #[test]
    fn stage_b_disabled_leaves_gate_score_untouched() {
        // With every Stage B flag false the per-frame path is
        // unchanged: `effective_window` stays at `sv_window_samples`,
        // the turn detector is inert, and the score that feeds the
        // gate is `last_score` verbatim (the streaming core only
        // calls `fuse_fast_cue` when `fast_cue_enabled`). This mirror
        // of that branch documents the byte-equal contract.
        let cfg = PipelineConfig::default();
        assert!(!cfg.fast_cue_enabled);
        assert!(!cfg.turn_detect_enabled);
        assert!(!cfg.offset_fail_closed);
        let det = TurnDetector::new(cfg.sv_window_samples);
        assert_eq!(det.effective_window, cfg.sv_window_samples);
        // The gate-score selection the core makes when the cue is off.
        let last_score = 0.37_f32;
        let gate_score = if cfg.fast_cue_enabled {
            fuse_fast_cue(
                last_score,
                cfg.fast_cue_f0_neutral,
                cfg.fast_cue_weight,
                cfg.fast_cue_f0_neutral,
            )
        } else {
            last_score
        };
        assert_eq!(gate_score, last_score);
    }
}
