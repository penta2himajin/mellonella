//! Overlapping-speaker detection on a 1-second sliding window of
//! 16 kHz mono audio.
//!
//! Wraps the pyannote 3.0 segmentation ONNX (community export, ~5.7 MB,
//! [`crate::hf_fetch::OVERLAP_SEG_REPO`]). The model takes raw audio
//! and emits per-frame 7-class powerset probabilities:
//!
//! | class | meaning              |
//! |------:|---------------------|
//! |     0 | silence             |
//! |     1 | speaker 1 only      |
//! |     2 | speaker 2 only      |
//! |     3 | speaker 3 only      |
//! |     4 | speaker 1 + 2       |
//! |     5 | speaker 1 + 3       |
//! |     6 | speaker 2 + 3       |
//!
//! Classes 4-6 are the "overlap" cases. [`OverlapDetector::push`] feeds
//! a rolling 1-second buffer, runs the ONNX, and returns
//! [`OverlapDecision`] with the mean per-frame overlap probability.
//!
//! Empirical thresholds (see `tests/overlap_detector.rs`):
//!
//! * solo voice (any noise level): mean overlap prob ≲ 0.03
//! * two-speaker mix: mean overlap prob ≳ 0.55
//!
//! A threshold of ~0.10 cleanly separates the two on every audio
//! tested. The pipeline applies hysteresis on top so transient blips
//! don't flip the chain.
//!
//! Inference cost: ~3 ms per 1-second window on CPU — well within
//! the live audio path's budget.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use std::path::Path;

use ndarray::Array3;
use ort::session::Session;
use ort::value::TensorRef;

use crate::embedding::EmbeddingError;

/// Sample rate the segmentation model is trained on.
pub const OVERLAP_SR: u32 = 16_000;
/// Rolling window length in samples (1 s @ 16 kHz). The model accepts
/// variable-length input, but the community ONNX export was trained
/// with 1-second windows during the offline-comparison sweep that
/// settled on this size.
pub const OVERLAP_WINDOW_SAMPLES: usize = 16_000;
/// Output class count (powerset over 3 speakers).
pub const N_CLASSES: usize = 7;
/// Indices of overlap (two-speaker) classes within the powerset.
pub const OVERLAP_CLASSES: [usize; 3] = [4, 5, 6];

/// One inference's view of the rolling window.
#[derive(Debug, Clone, Copy)]
pub struct OverlapDecision {
    /// Mean P(overlap) across the window's frames. Compare against a
    /// threshold to decide chain mode.
    pub mean_overlap_prob: f32,
    /// Max per-frame P(overlap) — useful for transient flagging.
    pub max_overlap_prob: f32,
    /// Number of frames in the window whose argmax was an overlap class.
    pub overlap_frame_count: usize,
    /// Total number of frames the model emitted for this window.
    pub total_frames: usize,
}

impl OverlapDecision {
    /// Convenience: fraction of frames classified as overlap.
    #[must_use]
    pub fn overlap_frame_fraction(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.overlap_frame_count as f32 / self.total_frames as f32
    }
}

/// Stateful overlap detector. Accumulates audio into a 1-second
/// rolling buffer; `push` returns `Some(OverlapDecision)` whenever
/// the buffer is full and `samples_since_last_run` reached the cadence.
pub struct OverlapDetector {
    session: Session,
    /// Rolling buffer of the last 1 s of audio. Always
    /// [`OVERLAP_WINDOW_SAMPLES`] in length after first fill.
    buffer: Vec<f32>,
    /// Samples queued since the previous inference. Inference fires
    /// once this reaches `cadence_samples`.
    samples_since_last_run: usize,
    /// How often (in samples) to re-run inference. Default 4 000
    /// (250 ms @ 16 kHz) so the chain can switch on / off four times
    /// per second.
    cadence_samples: usize,
}

impl OverlapDetector {
    /// Load the segmentation ONNX from `path` and initialise the
    /// rolling buffer.
    ///
    /// # Errors
    /// Forwards ORT session-construction errors via
    /// [`EmbeddingError`] (same as the other ONNX wrappers in this
    /// crate).
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let session = Session::builder()?
            .with_intra_threads(crate::ort_threads::intra_op_threads())?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(Self {
            session,
            buffer: Vec::with_capacity(OVERLAP_WINDOW_SAMPLES),
            samples_since_last_run: 0,
            cadence_samples: 4_000,
        })
    }

    /// Override the inference cadence (default 4 000 samples =
    /// 250 ms). Smaller = faster reaction, more CPU; larger = less
    /// CPU, slower mode switching.
    pub fn set_cadence_samples(&mut self, cadence: usize) {
        self.cadence_samples = cadence.max(1);
    }

    /// Inference cadence in milliseconds. The streaming engine uses
    /// this to drive its hysteresis counters at the same granularity
    /// as the detector's decisions, rather than the audio-rate VAD
    /// frame size (which would undercount by ~8x at default settings).
    #[must_use]
    pub fn cadence_ms(&self) -> f32 {
        (self.cadence_samples as f32 / OVERLAP_SR as f32) * 1_000.0
    }

    /// Drop all buffered audio. Use between independent recordings.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.samples_since_last_run = 0;
    }

    /// Push 16 kHz mono samples into the rolling buffer. Returns
    /// `Some(OverlapDecision)` whenever an inference fires (i.e. the
    /// buffer is full **and** `samples_since_last_run >=
    /// cadence_samples`), else `None`.
    ///
    /// Callers in the streaming pipeline typically feed VAD-frame
    /// chunks (512 samples @ 16 kHz); pushing 100 such frames before
    /// the first inference fires is normal — the first decision can
    /// only be emitted once the model has seen a full second of
    /// audio.
    ///
    /// # Errors
    /// Forwards ORT runtime + shape errors via [`EmbeddingError`].
    pub fn push(&mut self, samples_16k: &[f32]) -> Result<Option<OverlapDecision>, EmbeddingError> {
        for &s in samples_16k {
            if self.buffer.len() == OVERLAP_WINDOW_SAMPLES {
                self.buffer.remove(0);
            }
            self.buffer.push(s);
        }
        self.samples_since_last_run = self
            .samples_since_last_run
            .saturating_add(samples_16k.len());

        if self.buffer.len() < OVERLAP_WINDOW_SAMPLES
            || self.samples_since_last_run < self.cadence_samples
        {
            return Ok(None);
        }
        self.samples_since_last_run = 0;
        Ok(Some(self.run_inference_on_current_buffer()?))
    }

    /// Force an inference on the current rolling buffer regardless of
    /// the cadence counter. Useful for the warm-up path where we want
    /// a decision as soon as the buffer fills, even if cadence
    /// suggests waiting.
    ///
    /// Returns `None` while the buffer is still filling.
    ///
    /// # Errors
    /// As for [`Self::push`].
    pub fn force_run(&mut self) -> Result<Option<OverlapDecision>, EmbeddingError> {
        if self.buffer.len() < OVERLAP_WINDOW_SAMPLES {
            return Ok(None);
        }
        Ok(Some(self.run_inference_on_current_buffer()?))
    }

    fn run_inference_on_current_buffer(&mut self) -> Result<OverlapDecision, EmbeddingError> {
        // Shape: (1, 1, OVERLAP_WINDOW_SAMPLES) — batch, channel, time.
        let arr =
            Array3::<f32>::from_shape_vec((1, 1, OVERLAP_WINDOW_SAMPLES), self.buffer.clone())?;
        let outputs = self.session.run(ort::inputs![
            "input_values" => TensorRef::from_array_view(&arr)?,
        ])?;
        // Logits: [1, num_frames, 7]. Convert to per-frame sigmoid and
        // summarise.
        let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] as usize != N_CLASSES {
            return Err(EmbeddingError::UnexpectedOutputShape {
                got: shape.to_vec(),
                expected_dim: N_CLASSES,
            });
        }
        let num_frames = shape[1] as usize;
        let mut sum_overlap = 0.0_f32;
        let mut max_overlap = 0.0_f32;
        let mut overlap_frames = 0_usize;
        for f in 0..num_frames {
            let base = f * N_CLASSES;
            // Sigmoid + sum overlap-class probabilities for this frame.
            let mut p_overlap = 0.0_f32;
            let mut max_class = 0_usize;
            let mut max_logit = f32::NEG_INFINITY;
            for c in 0..N_CLASSES {
                let logit = data[base + c];
                let prob = 1.0 / (1.0 + (-logit).exp());
                if OVERLAP_CLASSES.contains(&c) {
                    p_overlap += prob;
                }
                if logit > max_logit {
                    max_logit = logit;
                    max_class = c;
                }
            }
            sum_overlap += p_overlap;
            if p_overlap > max_overlap {
                max_overlap = p_overlap;
            }
            if OVERLAP_CLASSES.contains(&max_class) {
                overlap_frames += 1;
            }
        }
        Ok(OverlapDecision {
            mean_overlap_prob: sum_overlap / num_frames.max(1) as f32,
            max_overlap_prob: max_overlap,
            overlap_frame_count: overlap_frames,
            total_frames: num_frames,
        })
    }
}
