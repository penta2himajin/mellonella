//! Silero-VAD ONNX inference wrapper.
//!
//! Loads `silero_vad.onnx` (the standard release shipped by the
//! `silero-vad` PyPI package) and exposes a stateful per-chunk speech
//! probability function. The model expects 16 kHz mono audio in
//! 512-sample chunks (32 ms) — the same cadence the Python PoC drives
//! its gating loop at via [`crate::pipeline`].
//!
//! Runtime: `ort 2.0.0-rc.12` with `load-dynamic`. The shared library
//! is found at runtime via `ORT_DYLIB_PATH`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_borrow,
    clippy::match_same_arms
)]

use std::path::Path;

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::TensorRef;

use crate::embedding::EmbeddingError;

/// Required input chunk size in samples at 16 kHz (silero-vad >= 5.x
/// hard-codes this; older versions accepted variable chunks).
pub const CHUNK_SAMPLES_16K: usize = 512;
/// Required input chunk size in samples at 8 kHz.
pub const CHUNK_SAMPLES_8K: usize = 256;
/// LSTM hidden-state dimensionality (`(2, batch, 128)`).
pub const STATE_HIDDEN: usize = 128;
/// Number of context samples prepended at 16 kHz (matches the Python
/// wrapper: 64 for 16 kHz, 32 for 8 kHz).
pub const CONTEXT_SAMPLES_16K: usize = 64;
/// Context samples at 8 kHz.
pub const CONTEXT_SAMPLES_8K: usize = 32;

/// Stateful Silero-VAD inference wrapper.
pub struct SileroVad {
    session: Session,
    sample_rate: i64,
    /// LSTM state, shape `(2, 1, 128)`.
    state: Array3<f32>,
    /// 64-sample (16 kHz) or 32-sample (8 kHz) tail of the previous
    /// ONNX input, prepended to the next user chunk before inference.
    /// Mirrors Python `silero_vad.utils_vad.OnnxWrapper._context`.
    context: Vec<f32>,
}

impl SileroVad {
    /// Load the silero-vad ONNX from `path` and pin it to a sample
    /// rate (`16_000` or `8_000`).
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when the file is missing or not
    /// a valid ONNX model. Reuses the embedding crate's error enum so
    /// the call site can chain ECAPA and VAD failures uniformly.
    pub fn from_onnx_path(
        path: impl AsRef<Path>,
        sample_rate: u32,
    ) -> Result<Self, EmbeddingError> {
        let session = Session::builder()?.commit_from_file(path)?;
        let context_len = match sample_rate {
            8_000 => CONTEXT_SAMPLES_8K,
            _ => CONTEXT_SAMPLES_16K,
        };
        Ok(Self {
            session,
            sample_rate: i64::from(sample_rate),
            state: Array3::<f32>::zeros((2, 1, STATE_HIDDEN)),
            context: vec![0.0_f32; context_len],
        })
    }

    /// Reset the LSTM state and zero the context buffer — call between
    /// independent recordings.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    /// Run the VAD on one audio chunk. Returns the speech probability
    /// in `[0, 1]` and updates the internal LSTM state.
    ///
    /// # Errors
    /// * [`EmbeddingError::Shape`] when `chunk.len()` doesn't match the
    ///   expected size for the configured sample rate
    /// * [`EmbeddingError::Ort`] when the runtime rejects the input
    /// * [`EmbeddingError::UnexpectedOutputShape`] when the output is
    ///   not `(1, 1)`
    pub fn score(&mut self, chunk: &[f32]) -> Result<f32, EmbeddingError> {
        let expected = match self.sample_rate {
            16_000 => CHUNK_SAMPLES_16K,
            8_000 => CHUNK_SAMPLES_8K,
            _ => CHUNK_SAMPLES_16K,
        };
        if chunk.len() != expected {
            return Err(EmbeddingError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }

        // Prepend the carried context, then run ONNX on the (1, ctx+chunk)
        // buffer. After inference, the last `context_len` samples of this
        // buffer become the new context for the next call.
        let mut prepended = Vec::with_capacity(self.context.len() + chunk.len());
        prepended.extend_from_slice(&self.context);
        prepended.extend_from_slice(chunk);
        let prepended_len = prepended.len();
        let input_arr = Array2::from_shape_vec((1, prepended_len), prepended.clone())
            .map_err(EmbeddingError::Shape)?;
        let sr_arr = Array1::from_vec(vec![self.sample_rate]);
        // Borrow current state for input; new state replaces it after
        // the run.
        let state_view = self.state.view();

        let outputs = self.session.run(ort::inputs![
            "input" => TensorRef::from_array_view(&input_arr)?,
            "state" => TensorRef::from_array_view(state_view)?,
            "sr" => TensorRef::from_array_view(&sr_arr)?,
        ])?;

        // output: (1, 1) speech probability
        let (out_shape, out_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let out_dims: &[i64] = &out_shape;
        let prob = match out_dims {
            [b, d] if *b == 1 && *d == 1 => out_data[0],
            _ => {
                return Err(EmbeddingError::UnexpectedOutputShape {
                    got: out_dims.to_vec(),
                    expected_dim: 1,
                });
            }
        };

        // stateN: copy back into self.state
        let (state_shape, state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
        let state_dims: &[i64] = &state_shape;
        if state_dims.len() != 3 || state_dims[0] != 2 || state_dims[2] != STATE_HIDDEN as i64 {
            return Err(EmbeddingError::UnexpectedOutputShape {
                got: state_dims.to_vec(),
                expected_dim: STATE_HIDDEN,
            });
        }
        let new_state = Array3::from_shape_vec(
            (
                state_dims[0] as usize,
                state_dims[1] as usize,
                state_dims[2] as usize,
            ),
            state_data.to_vec(),
        )
        .map_err(EmbeddingError::Shape)?;
        self.state = new_state;

        // Carry the last context_len samples of the prepended buffer
        // forward — Python copies `x[..., -context_size:]` after each
        // call so subsequent chunks see the trailing waveform of the
        // previous fed buffer.
        let ctx_len = self.context.len();
        self.context
            .copy_from_slice(&prepended[prepended_len - ctx_len..]);

        Ok(prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENV_VAR: &str = "MELLONELLA_VAD_ONNX";

    fn skip_unless_onnx_available() -> Option<std::path::PathBuf> {
        let Ok(path) = std::env::var(ENV_VAR) else {
            eprintln!("[skip] {ENV_VAR} not set");
            return None;
        };
        let p = std::path::PathBuf::from(path);
        if !p.exists() {
            eprintln!("[skip] {ENV_VAR}={} does not exist", p.display());
            return None;
        }
        Some(p)
    }

    #[test]
    fn loads_and_returns_probability_in_unit_interval() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut vad = SileroVad::from_onnx_path(&path, 16_000).expect("load ONNX");
        let chunk = vec![0.0_f32; CHUNK_SAMPLES_16K];
        let prob = vad.score(&chunk).expect("inference");
        assert!((0.0..=1.0).contains(&prob), "prob={prob}");
    }

    #[test]
    fn reset_zeroes_state() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut vad = SileroVad::from_onnx_path(&path, 16_000).expect("load ONNX");
        // Drive the state non-zero with a simulated speech chunk.
        let mut chunk = vec![0.0_f32; CHUNK_SAMPLES_16K];
        for (i, s) in chunk.iter_mut().enumerate() {
            *s = (i as f32 * 0.05).sin() * 0.3;
        }
        let _ = vad.score(&chunk).expect("inference");
        let nonzero_before = vad.state.iter().any(|&v| v.abs() > 1e-6);
        assert!(nonzero_before, "state should have moved off zero");
        vad.reset();
        assert!(vad.state.iter().all(|&v| v.abs() <= 0.0));
    }

    #[test]
    fn rejects_wrong_chunk_size() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut vad = SileroVad::from_onnx_path(&path, 16_000).expect("load ONNX");
        let chunk = vec![0.0_f32; 100];
        assert!(vad.score(&chunk).is_err());
    }
}
