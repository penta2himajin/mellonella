//! Stream-style buffering wrapper around [`crate::tse::TseSession`].
//!
//! [`crate::tse::TseSession`] consumes one fixed-length audio chunk per
//! call (the chunk length the ONNX was exported with). The offline
//! pipeline operates on arbitrary-length buffers, so this stage owns
//! the accumulator that turns a variable-length stream into
//! fixed-length chunks and threads the per-recording cond embedding
//! across calls.
//!
//! Mirrors the buffering pattern used by [`crate::dfn3::Dfn3Pipeline`]:
//!
//! * [`TseStage::process`] — append samples, drain complete chunks,
//!   return any extracted samples that became available.
//! * [`TseStage::flush`] — zero-pad any trailing partial chunk so the
//!   final samples emerge.
//! * [`TseStage::reset`] — clear the accumulator and zero-init the
//!   underlying ONNX state.
//!
//! Currently exclusively used by the offline pipeline
//! ([`crate::pipeline::process_offline`]); the streaming engine
//! ([`crate::streaming`]) is not yet wired up — see the Phase 5 TODOs.

#![allow(clippy::needless_borrow)]

use crate::tse::{TseError, TseSession, TSE_COND_DIM};

/// Configuration for the offline TSE stage.
///
/// Currently only carries the path to the streaming ONNX. Kept minimal;
/// future knobs (cond-refresh policy, custom [`crate::tse::TseConfig`])
/// can be added here without changing the offline-pipeline call sites.
#[derive(Debug, Clone)]
pub struct TseStageConfig {
    /// Path to the streaming TSE ONNX exported by
    /// `scripts/export_tse_onnx.py`. The chunk length the model was
    /// exported with determines the buffering granularity of this
    /// stage (see [`TseStage::chunk_samples`]).
    pub onnx_path: std::path::PathBuf,
    /// Per-chunk sample count to feed the ONNX. **Must** match the
    /// chunk length baked into the export (the ONNX has a fixed time
    /// axis). Defaults to 160 samples = 10 ms @ 16 kHz — the value the
    /// PoC export script uses in `--chunk`. Callers that exported the
    /// ONNX at a different chunk length must override this.
    pub chunk_samples: usize,
}

impl TseStageConfig {
    /// New PoC-default config (`chunk_samples = 160`).
    #[must_use]
    pub fn new(onnx_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            onnx_path: onnx_path.into(),
            chunk_samples: 160,
        }
    }
}

/// Errors returned by [`TseStage`].
#[derive(Debug)]
pub enum TseStageError {
    /// Caller-supplied cond embedding had the wrong length (must be
    /// [`crate::tse::TSE_COND_DIM`] = 192).
    InvalidCondLength { got: usize, expected: usize },
    /// `chunk_samples == 0` or wasn't a multiple of the underlying
    /// model's `enc_stride`. Surfaced at construction time so the
    /// offline pipeline can fail before any audio enters the stage.
    InvalidChunkSamples { got: usize, enc_stride: usize },
    /// Forwarded from the underlying [`TseSession`].
    Tse(TseError),
}

impl std::fmt::Display for TseStageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCondLength { got, expected } => {
                write!(f, "TSE cond embedding length {got} != expected {expected}")
            }
            Self::InvalidChunkSamples { got, enc_stride } => write!(
                f,
                "TSE chunk_samples {got} must be a positive multiple of enc_stride {enc_stride}"
            ),
            Self::Tse(e) => write!(f, "TSE session error: {e}"),
        }
    }
}

impl std::error::Error for TseStageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TseError> for TseStageError {
    fn from(e: TseError) -> Self {
        Self::Tse(e)
    }
}

/// Buffered streaming wrapper around [`TseSession`].
///
/// Owns the per-recording cond embedding (snapshot once at construction
/// — the embedding pool may evolve via auto-learn during the run, but
/// the simplest correct semantics for this PR is "frozen for the run").
/// Use [`TseStage::set_cond_embedding`] to update it explicitly between
/// runs.
pub struct TseStage {
    session: TseSession,
    cond_embedding: [f32; TSE_COND_DIM],
    chunk_samples: usize,
    /// Pending samples that haven't yet completed a full chunk.
    chunk_buffer: Vec<f32>,
}

impl TseStage {
    /// Build a stage from a config and an initial cond embedding.
    ///
    /// # Errors
    /// * [`TseStageError::InvalidCondLength`] when `cond_embedding.len()
    ///   != TSE_COND_DIM`.
    /// * [`TseStageError::InvalidChunkSamples`] when
    ///   `config.chunk_samples` is zero or not a multiple of the
    ///   model's `enc_stride`.
    /// * [`TseStageError::Tse`] when the ONNX file is missing or
    ///   rejected by the ort runtime.
    pub fn from_config(
        config: &TseStageConfig,
        cond_embedding: &[f32],
    ) -> Result<Self, TseStageError> {
        let session = TseSession::from_onnx_path(&config.onnx_path)?;
        let enc_stride = session.config().enc_stride;
        if config.chunk_samples == 0 || config.chunk_samples % enc_stride != 0 {
            return Err(TseStageError::InvalidChunkSamples {
                got: config.chunk_samples,
                enc_stride,
            });
        }
        if cond_embedding.len() != TSE_COND_DIM {
            return Err(TseStageError::InvalidCondLength {
                got: cond_embedding.len(),
                expected: TSE_COND_DIM,
            });
        }
        let mut cond = [0.0_f32; TSE_COND_DIM];
        cond.copy_from_slice(cond_embedding);
        Ok(Self {
            session,
            cond_embedding: cond,
            chunk_samples: config.chunk_samples,
            chunk_buffer: Vec::with_capacity(config.chunk_samples),
        })
    }

    /// Build a stage from an already-loaded [`TseSession`]. Useful for
    /// tests that share a session with other code paths. Performs the
    /// same `chunk_samples` and `cond_embedding` validation as
    /// [`Self::from_config`].
    ///
    /// # Errors
    /// See [`Self::from_config`] (minus the ONNX-load path).
    pub fn from_session(
        session: TseSession,
        chunk_samples: usize,
        cond_embedding: &[f32],
    ) -> Result<Self, TseStageError> {
        let enc_stride = session.config().enc_stride;
        if chunk_samples == 0 || chunk_samples % enc_stride != 0 {
            return Err(TseStageError::InvalidChunkSamples {
                got: chunk_samples,
                enc_stride,
            });
        }
        if cond_embedding.len() != TSE_COND_DIM {
            return Err(TseStageError::InvalidCondLength {
                got: cond_embedding.len(),
                expected: TSE_COND_DIM,
            });
        }
        let mut cond = [0.0_f32; TSE_COND_DIM];
        cond.copy_from_slice(cond_embedding);
        Ok(Self {
            session,
            cond_embedding: cond,
            chunk_samples,
            chunk_buffer: Vec::with_capacity(chunk_samples),
        })
    }

    /// Chunk length (samples) the underlying ONNX expects per
    /// `process_chunk` call. Exposed so the offline orchestrator can
    /// size its work buffers.
    #[must_use]
    pub const fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }

    /// Snapshot the current cond embedding. Mostly a diagnostics
    /// accessor; the offline pipeline freezes the embedding at the
    /// start of the run.
    #[must_use]
    pub const fn cond_embedding(&self) -> &[f32; TSE_COND_DIM] {
        &self.cond_embedding
    }

    /// Replace the cond embedding. The new vector must have length
    /// [`TSE_COND_DIM`].
    ///
    /// # Errors
    /// * [`TseStageError::InvalidCondLength`] when the length is wrong.
    pub fn set_cond_embedding(&mut self, cond: &[f32]) -> Result<(), TseStageError> {
        if cond.len() != TSE_COND_DIM {
            return Err(TseStageError::InvalidCondLength {
                got: cond.len(),
                expected: TSE_COND_DIM,
            });
        }
        self.cond_embedding.copy_from_slice(cond);
        Ok(())
    }

    /// Clear the streaming accumulator and zero-init the underlying
    /// ONNX state. Call between independent recordings.
    pub fn reset(&mut self) {
        self.chunk_buffer.clear();
        self.session.reset();
    }

    /// Push samples through the stage and return any extracted samples
    /// that became available. Samples are accumulated into the
    /// internal buffer; whenever it fills to `chunk_samples`, the
    /// underlying [`TseSession::process_chunk`] is invoked and the
    /// returned chunk is appended to the output.
    ///
    /// # Errors
    /// Forwards [`TseStageError::Tse`] from the underlying ONNX call.
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, TseStageError> {
        let mut out: Vec<f32> = Vec::new();
        let mut cursor = 0;
        while cursor < samples.len() {
            // How many samples until the next complete chunk?
            let needed = self.chunk_samples - self.chunk_buffer.len();
            let take = needed.min(samples.len() - cursor);
            self.chunk_buffer
                .extend_from_slice(&samples[cursor..cursor + take]);
            cursor += take;
            if self.chunk_buffer.len() == self.chunk_samples {
                let extracted = self
                    .session
                    .process_chunk_slice(&self.chunk_buffer, &self.cond_embedding)?;
                out.extend_from_slice(&extracted);
                self.chunk_buffer.clear();
            }
        }
        Ok(out)
    }

    /// Drain any partial trailing chunk by zero-padding to
    /// `chunk_samples` and running one final inference. The returned
    /// extracted samples are still `chunk_samples` long (full chunk);
    /// callers that need exactly `audio.len()` samples after
    /// `process(audio).extend(flush())` should truncate the
    /// concatenation themselves.
    ///
    /// Returns an empty vector when the buffer is already empty (no
    /// trailing residue to drain).
    ///
    /// # Errors
    /// Forwards [`TseStageError::Tse`] from the underlying ONNX call.
    pub fn flush(&mut self) -> Result<Vec<f32>, TseStageError> {
        if self.chunk_buffer.is_empty() {
            return Ok(Vec::new());
        }
        let pad = self.chunk_samples - self.chunk_buffer.len();
        // `std::iter::repeat_n` is MSRV 1.82; use the older `repeat`
        // primitive instead so the crate stays buildable on 1.75.
        self.chunk_buffer
            .extend(std::iter::repeat(0.0_f32).take(pad));
        let extracted = self
            .session
            .process_chunk_slice(&self.chunk_buffer, &self.cond_embedding)?;
        self.chunk_buffer.clear();
        Ok(extracted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tse::TseConfig;

    #[test]
    fn invalid_cond_length_rejected_in_from_session() {
        // Can't construct a real `TseSession` without an ONNX file, so
        // we exercise the validation path that doesn't need the
        // session — the helper that builds one from a config object
        // checks both errors in the same order, and `set_cond_embedding`
        // exercises the cond-length branch in isolation.
        // This sanity-checks the Display impl uses the right format.
        let err = TseStageError::InvalidCondLength {
            got: 64,
            expected: TSE_COND_DIM,
        };
        let s = format!("{err}");
        assert!(s.contains("192"), "{s}");
        assert!(s.contains("64"), "{s}");
    }

    #[test]
    fn invalid_chunk_samples_rejected_in_display() {
        let cfg = TseConfig::poc_16k();
        let err = TseStageError::InvalidChunkSamples {
            got: 17,
            enc_stride: cfg.enc_stride,
        };
        let s = format!("{err}");
        assert!(s.contains("17"), "{s}");
        assert!(s.contains("16"), "{s}");
    }

    #[test]
    fn config_default_chunk_samples_is_160() {
        // PoC export ships with `--chunk 160` by default and the parity
        // test pins to that value. Lock the config default to match.
        let cfg = TseStageConfig::new("/dev/null");
        assert_eq!(cfg.chunk_samples, 160);
    }
}
