//! Streaming / online pipeline API — **design PR scaffold**.
//!
//! This module is the stateful counterpart of
//! [`crate::pipeline::process_offline`]. It is intended for live
//! microphone use, GUI integrations, and any caller that produces
//! samples incrementally rather than as one contiguous buffer.
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
//! * **Input granularity**: any length, any cadence. Callers may push
//!   1-sample or 100 000-sample chunks and the pipeline does the
//!   right thing.
//! * **Internal alignment**: VAD frames are 512 samples (32 ms @
//!   16 kHz, [`crate::vad::CHUNK_SAMPLES_16K`]). Sub-frame residue is
//!   held in an internal ring buffer until enough samples accumulate.
//! * **Output granularity**: a multiple of 512 samples per
//!   `push_samples` call, plus whatever the envelope's
//!   attack/release tail produces. Sub-frame residue produces zero
//!   output until the next call.
//! * **Flush**: [`StreamingPipeline::flush`] zero-pads any residual
//!   sub-frame samples to a full VAD frame so the trailing audio
//!   gets one last decision pass.
//!
//! # Algorithm parity with `process_offline`
//!
//! With `StreamingConfig::pipeline.async_refresh == false`, chunking
//! the same audio differently must produce the **same concatenated
//! output** as one call to `process_offline`. This is enforced by a
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
/// consumes; `diagnostics` is new — it gates per-VAD-frame trace
/// arrays on the output struct so the GUI hot path doesn't allocate
/// when those traces aren't needed.
#[derive(Debug, Clone, Default)]
pub struct StreamingConfig {
    /// Pipeline-side cadence (window / refresh / VAD threshold /
    /// auto-learn switch / async refresh).
    pub pipeline: PipelineConfig,
    /// Gate-side parameters (hangover, attack/release, score
    /// threshold, F0 weight).
    pub gate: GateConfig,
    /// When `true`, [`StreamingOutput`] populates
    /// `gate_per_frame` / `score_per_frame` /
    /// `cos_sim_max_per_frame` / `f0_match_per_frame`. Default
    /// `false` to avoid the per-call allocation in the live path —
    /// the GUI can opt in only while a "live status" panel is open.
    pub diagnostics: bool,
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
    /// Envelope-gated audio at 16 kHz mono, for the just-pushed
    /// chunk. Length is a multiple of [`crate::vad::CHUNK_SAMPLES_16K`].
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

    /// Push an arbitrary-length chunk of 16 kHz f32 mono samples
    /// (range `[-1.0, 1.0]`).
    ///
    /// Returns the gated output corresponding to all VAD frames that
    /// could be completed with the new samples. Residue smaller than
    /// one VAD frame (512 samples) is buffered internally; it
    /// produces no output until the next call (or a [`Self::flush`]).
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
