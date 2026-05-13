//! Streaming / online pipeline API — **design PR scaffold**.
//!
//! This module is the stateful counterpart of
//! [`crate::pipeline::process_offline`]. It is intended for live
//! microphone use, GUI integrations, and any caller that produces
//! samples incrementally rather than as one contiguous buffer.
//!
//! It inherits the **dual-rate split** that
//! [`crate::pipeline::process_offline`] lands in step 8: callers
//! push audio at `StreamingConfig::audio_sample_rate` (default
//! 48 kHz, full-band quality), the pipeline downsamples internally
//! to 16 kHz for the VAD / ECAPA / F0 decision path, and emits
//! envelope-gated audio at the input rate.
//!
//! # Status
//!
//! The types in this module are **API signatures only** — the bodies
//! are `unimplemented!()` pending design review. The implementation
//! lands in a follow-up PR (Phase 3.5 step 9). The intent of this PR
//! is to lock down the public surface (struct shape, method
//! signatures, ownership / borrowing model, error contract) so that
//! the `audio-io` and `gui` crates can be designed against a stable
//! target.
//!
//! # Buffering model
//!
//! * **Input granularity**: any length, any cadence, at
//!   `StreamingConfig::audio_sample_rate`. Callers may push
//!   1-sample or 100 000-sample chunks and the pipeline does the
//!   right thing.
//! * **Internal alignment**: VAD frames are 512 samples (32 ms @
//!   16 kHz, [`crate::vad::CHUNK_SAMPLES_16K`]). The audio is
//!   resampled to 16 kHz on the decision path; sub-frame residue is
//!   held in an internal ring at the audio rate until enough
//!   samples accumulate.
//! * **Output granularity**: a multiple of one VAD frame's
//!   audio-rate equivalent per `push_samples` call (e.g. 1536
//!   samples @ 48 kHz, the integer scaling of 512 @ 16 kHz), plus
//!   whatever the envelope's attack/release tail produces.
//!   Sub-frame residue produces zero output until the next call.
//! * **Flush**: [`StreamingPipeline::flush`] zero-pads any residual
//!   sub-frame samples to a full VAD frame so the trailing audio
//!   gets one last decision pass.
//!
//! # Algorithm parity with `process_offline`
//!
//! With `StreamingConfig::pipeline.async_refresh == false`, chunking
//! the same audio differently must produce the **same concatenated
//! output** as one call to `process_offline` at the same
//! `audio_sample_rate`. This is enforced by a
//! `streaming_chunk_invariance` test introduced alongside the
//! implementation PR.
//!
//! With `async_refresh == true`, output is byte-identical only
//! within a chunking strategy — across strategies, per-frame scores
//! may differ by ≤ 1 ECAPA refresh worth of delay (see the
//! `PipelineConfig::async_refresh` doc comment). The implementation
//! PR will document the exact invariant.
//!
//! # Ownership
//!
//! `StreamingPipeline::new` takes ownership of the
//! [`crate::enrollment::EmbeddingPool`] and
//! [`crate::pipeline::PipelineComponents`]. They can be recovered on
//! shutdown via [`StreamingPipeline::into_parts`] — useful for
//! persisting the post-run pool (auto-learn updates) and for tearing
//! the ONNX sessions down deterministically.

#![allow(
    dead_code,
    unused_variables,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value,
    clippy::unused_self
)]

use crate::enrollment::EmbeddingPool;
use crate::gating::GateConfig;
use crate::pipeline::{AutoLearnEvent, PipelineComponents, PipelineConfig, PipelineError};

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
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            gate: GateConfig::default(),
            audio_sample_rate: 48_000,
            diagnostics: false,
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
    /// Auto-learn admission / rejection / reset events that occurred
    /// during this call, in chronological order.
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
}

/// Stateful, single-target speaker gating pipeline driven by
/// incremental sample pushes.
///
/// Internally owns:
///
/// * the [`EmbeddingPool`] (anchors + auto-learn FIFO + F0 stats);
/// * the [`PipelineComponents`] (VAD, Fbank, ECAPA, cohort);
/// * a `GateState` / `EnvelopeState` pair carrying hangover / attack
///   / release state across calls;
/// * an input ring buffer (sub-VAD-frame residue);
/// * a rolling speaker-verification window (`sv_window_samples`
///   wide, refreshed every `sv_update_samples`);
/// * a monotonic frame index used for `AutoLearnEvent.frame_idx`;
/// * an optional async ECAPA worker (Phase 3.5 step 7, enabled by
///   `PipelineConfig::async_refresh`).
///
/// The struct is intentionally `!Send`-leaning until the
/// implementation PR sorts out the ONNX session thread-safety story
/// — pre-flag it as a TODO during review.
pub struct StreamingPipeline {
    pool: EmbeddingPool,
    config: StreamingConfig,
    components: PipelineComponents,
}

impl StreamingPipeline {
    /// Build a streaming pipeline. The pool and components are moved
    /// in; recover them via [`Self::into_parts`].
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if the async worker (when
    /// `config.pipeline.async_refresh` is true) cannot be started.
    pub fn new(
        pool: EmbeddingPool,
        config: StreamingConfig,
        components: PipelineComponents,
    ) -> Result<Self, PipelineError> {
        unimplemented!("Phase 3.5 step 9 — implementation lands in a follow-up PR")
    }

    /// Push an arbitrary-length chunk of `audio_sample_rate` Hz f32
    /// mono samples (range `[-1.0, 1.0]`).
    ///
    /// Returns the gated output, at the same sample rate, corresponding
    /// to all VAD frames that could be completed with the new samples.
    /// Internally the pipeline resamples to 16 kHz for the decision
    /// path; residue smaller than one VAD frame's audio-rate
    /// equivalent (e.g. 1536 samples @ 48 kHz) is buffered and produces
    /// no output until the next call (or a [`Self::flush`]).
    ///
    /// # Errors
    ///
    /// Returns `PipelineError` if an underlying ONNX inference fails
    /// or the envelope step fails.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<StreamingOutput, PipelineError> {
        unimplemented!("Phase 3.5 step 9 — implementation lands in a follow-up PR")
    }

    /// Flush any residual sub-VAD-frame samples by zero-padding to a
    /// full VAD frame so the trailing audio gets one last decision
    /// pass. Resets the input ring after.
    ///
    /// Call this once at end-of-stream (e.g. when the audio device
    /// closes) to avoid losing the tail.
    ///
    /// # Errors
    ///
    /// Same as [`Self::push_samples`].
    pub fn flush(&mut self) -> Result<StreamingOutput, PipelineError> {
        unimplemented!("Phase 3.5 step 9 — implementation lands in a follow-up PR")
    }

    /// Read-only access to the owned pool (e.g. for live status
    /// queries — anchor count, auto-learn depth, drift state).
    #[must_use]
    pub fn pool(&self) -> &EmbeddingPool {
        &self.pool
    }

    /// Mutable access to the owned pool. Use sparingly — direct
    /// mutation while a `push_samples` is in flight (across an
    /// `await` boundary if a future async wrapper is added) would
    /// race with the async ECAPA worker.
    pub fn pool_mut(&mut self) -> &mut EmbeddingPool {
        &mut self.pool
    }

    /// Reset stateful pieces (gate, envelope, ring buffers, async
    /// worker) **without** rebuilding ONNX sessions. Use after the
    /// GUI changes `GateConfig` or after the input device changes.
    /// The pool is preserved.
    pub fn reset(&mut self) {
        unimplemented!("Phase 3.5 step 9 — implementation lands in a follow-up PR")
    }

    /// Tear the pipeline down, returning the owned pool and
    /// components. Any async worker is joined before returning.
    #[must_use]
    pub fn into_parts(self) -> (EmbeddingPool, PipelineComponents) {
        (self.pool, self.components)
    }
}
