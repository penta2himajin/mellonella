//! DeepFilterNet 3 ONNX inference wrapper.
//!
//! Runs the patched DfNet exported by `scripts/export_dfn3_onnx.py`:
//! takes the pre-computed `(spec, feat_erb, feat_spec)` tensors and
//! returns `(enhanced_spec, mask, lsnr, df_alpha)`. The Fbank-style
//! pre / post pipeline (STFT, ERB features, iSTFT) is owned by
//! [`crate::dfn3::Pipeline`] (TBD); this module is **only** the ONNX
//! call.
//!
//! Why split it this way: the ONNX itself is a 9 MB blob that requires
//! a `MELLONELLA_DFN3_ONNX` env var and the ort runtime, while the
//! feature pipeline is pure-Rust and depends on the `deep_filter`
//! crate's `DFState`. Keeping the two layers separate lets each get
//! its own unit-test cadence.
//!
//! Shape contract (matches `scripts/export_dfn3_onnx.py`):
//!
//! | input       | shape                       | notes                            |
//! |-------------|-----------------------------|----------------------------------|
//! | spec        | `[batch, 1, 102, 481, 2]`   | real-valued (last dim re/im pair) |
//! | feat_erb    | `[batch, 1, 102, 32]`       | ERB-normalised dB                 |
//! | feat_spec   | `[batch, 1, 102, 96, 2]`    | unit-normed complex (real)        |
//!
//! Frame count is **fixed at 102** because the export's df_op uses
//! tensor.unfold which can't lower with symbolic time dims. Callers
//! chunk audio into 1.024-s windows (102 × 480 samples @ 48 kHz)
//! before invoking the wrapper.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_borrow,
    clippy::match_same_arms
)]

use std::path::Path;

use ndarray::{Array4, Array5};
use ort::session::Session;
use ort::value::TensorRef;

use crate::embedding::EmbeddingError;

/// Required STFT frame count per inference call. See module docs.
pub const FRAMES_PER_CHUNK: usize = 102;
/// Frequency bins in the spectrogram (`fft_size/2 + 1` for fft_size=960).
pub const N_FREQ: usize = 481;
/// Number of low frequency bins fed to the DF feature path.
pub const NB_DF: usize = 96;
/// Number of ERB filterbank bands.
pub const N_ERB: usize = 32;

/// DFN3 ONNX inference wrapper.
pub struct Dfn3 {
    session: Session,
}

impl Dfn3 {
    /// Load the patched DFN3 ONNX from `path`.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when the file is missing or
    /// rejected by the ort runtime. Reuses the embedding crate's
    /// error enum so call sites can chain ECAPA / VAD / DFN3 failures
    /// uniformly.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self { session })
    }

    /// Run inference and return the enhanced spectrogram.
    ///
    /// `spec` is the row-major flattened buffer for a
    /// `(1, 1, FRAMES_PER_CHUNK, N_FREQ, 2)` real-valued tensor.
    /// `feat_erb` is `(1, 1, FRAMES_PER_CHUNK, N_ERB)` and
    /// `feat_spec` is `(1, 1, FRAMES_PER_CHUNK, NB_DF, 2)`.
    /// The output is a fresh buffer matching `spec`'s shape.
    ///
    /// # Errors
    /// * [`EmbeddingError::Shape`] when an input length doesn't match
    ///   the contract
    /// * [`EmbeddingError::Ort`] when the runtime rejects the input
    /// * [`EmbeddingError::UnexpectedOutputShape`] when the output
    ///   isn't the expected 5-D shape
    pub fn enhance_spec(
        &mut self,
        spec: &[f32],
        feat_erb: &[f32],
        feat_spec: &[f32],
    ) -> Result<Vec<f32>, EmbeddingError> {
        let spec_expected = FRAMES_PER_CHUNK * N_FREQ * 2;
        let erb_expected = FRAMES_PER_CHUNK * N_ERB;
        let dfspec_expected = FRAMES_PER_CHUNK * NB_DF * 2;
        if spec.len() != spec_expected
            || feat_erb.len() != erb_expected
            || feat_spec.len() != dfspec_expected
        {
            return Err(EmbeddingError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }

        let spec_arr = Array5::from_shape_vec((1, 1, FRAMES_PER_CHUNK, N_FREQ, 2), spec.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let erb_arr = Array4::from_shape_vec((1, 1, FRAMES_PER_CHUNK, N_ERB), feat_erb.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let dfspec_arr =
            Array5::from_shape_vec((1, 1, FRAMES_PER_CHUNK, NB_DF, 2), feat_spec.to_vec())
                .map_err(EmbeddingError::Shape)?;

        let outputs = self.session.run(ort::inputs![
            "spec" => TensorRef::from_array_view(&spec_arr)?,
            "feat_erb" => TensorRef::from_array_view(&erb_arr)?,
            "feat_spec" => TensorRef::from_array_view(&dfspec_arr)?,
        ])?;

        let (shape, data) = outputs["enhanced_spec"].try_extract_tensor::<f32>()?;
        let dims: &[i64] = &shape;
        let total: i64 = spec_expected as i64;
        if dims.iter().product::<i64>() != total {
            return Err(EmbeddingError::UnexpectedOutputShape {
                got: dims.to_vec(),
                expected_dim: spec_expected,
            });
        }
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENV_VAR: &str = "MELLONELLA_DFN3_ONNX";

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
    fn loads_and_runs_zero_input() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = Dfn3::from_onnx_path(&path).expect("load DFN3 ONNX");
        let spec = vec![0.0_f32; FRAMES_PER_CHUNK * N_FREQ * 2];
        let feat_erb = vec![0.0_f32; FRAMES_PER_CHUNK * N_ERB];
        let feat_spec = vec![0.0_f32; FRAMES_PER_CHUNK * NB_DF * 2];
        let out = model
            .enhance_spec(&spec, &feat_erb, &feat_spec)
            .expect("inference");
        assert_eq!(out.len(), spec.len());
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite values in output"
        );
    }

    #[test]
    fn rejects_mis_shaped_inputs() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = Dfn3::from_onnx_path(&path).expect("load DFN3 ONNX");
        let bad = vec![0.0_f32; 10];
        let feat_erb = vec![0.0_f32; FRAMES_PER_CHUNK * N_ERB];
        let feat_spec = vec![0.0_f32; FRAMES_PER_CHUNK * NB_DF * 2];
        let res = model.enhance_spec(&bad, &feat_erb, &feat_spec);
        assert!(res.is_err());
    }
}
