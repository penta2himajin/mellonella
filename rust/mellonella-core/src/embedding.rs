//! ECAPA-TDNN ONNX inference wrapper.
//!
//! Loads the embedding-only ONNX produced by
//! ``scripts/export_ecapa_onnx.py --mode embedding-only`` and runs it
//! against a pre-computed Fbank `(T_frames, n_mels)` to get the
//! 192-dim speaker embedding.
//!
//! The Fbank itself is reproduced in [`crate::features`] (TBD) so the
//! Rust consumer takes raw 16 kHz audio at the API boundary; this
//! module only owns the ONNX session and the inference contract.
//!
//! Empirical parity vs. the PyTorch reference (synth clips, see
//! `scripts/export_ecapa_onnx.py verify`):
//!
//! | metric                       | value      | tolerance |
//! |------------------------------|------------|-----------|
//! | raw embedding `max\|Δ\|`      | 6.4 × 10⁻⁴ | 1 × 10⁻⁴  |
//! | cosine-similarity `max\|Δ\|`  | 2.1 × 10⁻⁷ | 1 × 10⁻⁴ ✅ |

// `i64 → usize` conversions on the ONNX shape vector and `usize → f32`
// in the test fixture are bounded: shapes come from a known 192-dim
// graph, fixture lengths are ≤ 8 000.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_borrow
)]

use std::path::Path;

use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, TensorRef};

/// Output dimensionality of the SpeechBrain ECAPA-TDNN backbone.
pub const EMBEDDING_DIM: usize = 192;

/// Errors returned by [`EcapaTdnn`].
#[derive(Debug)]
pub enum EmbeddingError {
    /// ONNX Runtime returned a failure (model load, kernel error, …).
    /// The string carries the upstream message; the typed recovery
    /// token from `ort::Error<R>` is dropped because we don't use it.
    Ort(String),
    Shape(ndarray::ShapeError),
    /// Output tensor did not match the expected `(1, 192)` shape — the
    /// ONNX file is incompatible with this build of `mellonella-core`.
    UnexpectedOutputShape {
        got: Vec<i64>,
        expected_dim: usize,
    },
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ort(e) => write!(f, "ONNX Runtime error: {e}"),
            Self::Shape(e) => write!(f, "ndarray shape error: {e}"),
            Self::UnexpectedOutputShape { got, expected_dim } => write!(
                f,
                "ECAPA ONNX output shape {got:?} not compatible with expected dim {expected_dim}"
            ),
        }
    }
}

impl std::error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ort(_) | Self::UnexpectedOutputShape { .. } => None,
            Self::Shape(e) => Some(e),
        }
    }
}

impl<R> From<ort::Error<R>> for EmbeddingError {
    fn from(e: ort::Error<R>) -> Self {
        Self::Ort(e.to_string())
    }
}

impl From<ndarray::ShapeError> for EmbeddingError {
    fn from(e: ndarray::ShapeError) -> Self {
        Self::Shape(e)
    }
}

/// ONNX inference wrapper around the ECAPA-TDNN embedding-only graph.
pub struct EcapaTdnn {
    session: Session,
}

impl EcapaTdnn {
    /// Load an embedding-only ECAPA-TDNN ONNX from `path`.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when the file is missing or not
    /// a valid ONNX model that the bundled ONNX Runtime can lower.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        Ok(Self { session })
    }

    /// Run inference on a pre-computed Fbank.
    ///
    /// `features` is the row-major flattened `(T_frames × n_mels)`
    /// buffer; the function reshapes it to `(1, T_frames, n_mels)`
    /// before feeding the ONNX. Output is a 192-element vector.
    ///
    /// # Errors
    /// * [`EmbeddingError::Shape`] when `features.len() != n_frames * n_mels`
    /// * [`EmbeddingError::Ort`] when the ONNX runtime rejects the input
    /// * [`EmbeddingError::UnexpectedOutputShape`] when the output is
    ///   not `(1, EMBEDDING_DIM)`
    pub fn embed_features(
        &mut self,
        features: &[f32],
        n_frames: usize,
        n_mels: usize,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let array: Array3<f32> = Array3::from_shape_vec((1, n_frames, n_mels), features.to_vec())?;
        let input = TensorRef::from_array_view(&array)?;
        let outputs = self.session.run(ort::inputs!["features" => input])?;
        let view = outputs["embedding"].try_extract_tensor::<f32>()?;
        let (shape, data) = view;
        let dims: &[i64] = &shape;
        let dim = match dims {
            [b, d] if *b == 1 && *d as usize == EMBEDDING_DIM => EMBEDDING_DIM,
            [d] if *d as usize == EMBEDDING_DIM => EMBEDDING_DIM,
            _ => {
                return Err(EmbeddingError::UnexpectedOutputShape {
                    got: dims.to_vec(),
                    expected_dim: EMBEDDING_DIM,
                });
            }
        };
        Ok(data[..dim].to_vec())
    }
}

// Tensor type referenced for symmetry with the API doc above; the actual
// inference path uses TensorRef so we don't move the input array.
#[allow(dead_code)]
fn _ensure_tensor_export() {
    let _ = std::marker::PhantomData::<Tensor<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to a local ECAPA ONNX file. Tests that need to run actual
    /// inference are gated on this env var; if unset the test exits
    /// early with a printed skip notice so the suite still reports
    /// success.
    const ENV_VAR: &str = "MELLONELLA_ECAPA_ONNX";

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
    fn loads_and_runs_against_local_onnx() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = EcapaTdnn::from_onnx_path(&path).expect("load ONNX");

        // Synthetic deterministic Fbank: 100 frames × 80 mels, sinusoidal.
        let n_frames = 100;
        let n_mels = 80;
        let mut feats = vec![0.0_f32; n_frames * n_mels];
        for (i, slot) in feats.iter_mut().enumerate() {
            let f = i as f32;
            *slot = (f * 0.01).sin();
        }

        let emb = model
            .embed_features(&feats, n_frames, n_mels)
            .expect("run inference");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        assert!(
            emb.iter().all(|v| v.is_finite()),
            "non-finite values in embedding"
        );
        // The embedding is non-trivial — at least one element above noise.
        assert!(emb.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn rejects_mis_shaped_features() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = EcapaTdnn::from_onnx_path(&path).expect("load ONNX");
        // 100 * 80 = 8000, but we hand it 1000 — should error in
        // Array3::from_shape_vec, not panic.
        let feats = vec![0.0_f32; 1000];
        let res = model.embed_features(&feats, 100, 80);
        assert!(res.is_err());
    }
}
