//! ECAPA-TDNN ONNX inference wrapper.
//!
//! Two backends, both producing the same 192-dim speaker embedding
//! from a pre-computed Fbank `(T_frames, n_mels)`:
//!
//! * **Monolithic** ([`EcapaTdnn::from_onnx_path`]) — the
//!   `embedding-only` ONNX produced by
//!   `scripts/export_ecapa_onnx.py export --mode embedding-only`. One
//!   session, `features (1, T, 80) → embedding (1, 192)`.
//! * **Split** ([`EcapaTdnn::from_split_onnx_paths`], #119) — the
//!   `encoder` + `pooler` ONNX pair. The encoder runs the heavy,
//!   time-invariant conv front-end (`features → frame_features
//!   (1, C_mfa, T)`); the pooler runs the cheap attentive-pooling
//!   head (`frame_features → embedding (1, 192)`). Exposed as
//!   [`EcapaTdnn::encode_features`] / [`EcapaTdnn::pool_frame_features`]
//!   so a streaming caller can run the two halves at different
//!   cadences. `pooler(encoder(x))` reproduces the monolithic graph
//!   (verify-split: cosine max|Δ| = 2.1e-7).
//!
//! The Fbank itself is reproduced in [`crate::features`] so the Rust
//! consumer takes raw 16 kHz audio at the API boundary; this module
//! only owns the ONNX session(s) and the inference contract.
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
    /// A split-only method ([`EcapaTdnn::encode_features`] /
    /// [`EcapaTdnn::pool_frame_features`]) was called on a monolithic
    /// `EcapaTdnn` built via [`EcapaTdnn::from_onnx_path`]. Build the
    /// model with [`EcapaTdnn::from_split_onnx_paths`] to use the
    /// encoder / pooler halves independently.
    NotSplit,
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
            Self::NotSplit => write!(
                f,
                "encode_features / pool_frame_features require a split (encoder + pooler) \
                 EcapaTdnn — build it with EcapaTdnn::from_split_onnx_paths"
            ),
        }
    }
}

impl std::error::Error for EmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ort(_) | Self::UnexpectedOutputShape { .. } | Self::NotSplit => None,
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

/// Per-frame feature map produced by the ECAPA encoder half (#119) —
/// the output of [`EcapaTdnn::encode_features`].
///
/// Stored **frame-contiguous** as `(n_frames, c_mfa)` row-major (frame
/// `t` occupies `data[t * c_mfa .. (t + 1) * c_mfa]`) so a streaming
/// ring buffer can append and slice whole frames cheaply. The ONNX
/// encoder graph emits `(1, C_mfa, T)`; [`EcapaTdnn::encode_features`]
/// transposes once on the way out so callers never see the
/// channel-major layout.
///
/// `c_mfa` (the multi-layer-feature-aggregation channel count) is
/// checkpoint-dependent — 3072 for the stock
/// `speechbrain/spkrec-ecapa-voxceleb` model — and is learned from the
/// ONNX output shape rather than hard-coded.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// `(n_frames * c_mfa)` row-major, frame-contiguous.
    pub data: Vec<f32>,
    /// Number of time frames.
    pub n_frames: usize,
    /// MFA channel count (feature dimension per frame).
    pub c_mfa: usize,
}

/// ONNX backend: a monolithic embedding-only graph, or the #119
/// encoder + pooler split.
enum Backend {
    /// `features (1, T, 80) → embedding (1, 192)`.
    Mono(Session),
    /// `features (1, T, 80) → frame_features (1, C_mfa, T)` (encoder)
    /// then `frame_features (1, C_mfa, T) → embedding (1, 192)` (pooler).
    Split { encoder: Session, pooler: Session },
}

/// ONNX inference wrapper around the ECAPA-TDNN embedding graph.
pub struct EcapaTdnn {
    backend: Backend,
}

/// Build an ORT session with the project's standard thread / opt
/// settings. Shared by the monolithic and split constructors so every
/// ECAPA session is configured identically.
fn build_session(path: impl AsRef<Path>) -> Result<Session, EmbeddingError> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // Pin intra-op threads to the physical-core count and disable
        // inter-op parallelism. The default (intra=num_cores,
        // inter=num_cores) thrashes on small 2-vCPU hosts because both
        // pools fight for the same cores. Single-batch inference
        // doesn't benefit from inter-op parallelism either — every op
        // runs serially on the graph anyway.
        .with_intra_threads(crate::ort_threads::intra_op_threads())?
        .with_inter_threads(1)?
        .commit_from_file(path)?;
    Ok(session)
}

/// Validate an `embedding` output view against the expected `(1, 192)`
/// (or bare `(192,)`) shape and copy out the 192-element vector.
fn embedding_from_view(dims: &[i64], data: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
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

/// Run the encoder graph: `features (1, T, n_mels) → frame_features
/// (1, C_mfa, T)`, transposed to frame-contiguous `(T, C_mfa)` on the
/// way out.
fn run_encoder(
    session: &mut Session,
    features: &[f32],
    n_frames: usize,
    n_mels: usize,
) -> Result<FrameFeatures, EmbeddingError> {
    let array: Array3<f32> = Array3::from_shape_vec((1, n_frames, n_mels), features.to_vec())?;
    let input = TensorRef::from_array_view(&array)?;
    let outputs = session.run(ort::inputs!["features" => input])?;
    let view = outputs["frame_features"].try_extract_tensor::<f32>()?;
    let (shape, data) = view;
    let dims: &[i64] = &shape;
    let (c_mfa, t) = match dims {
        [b, c, t] if *b == 1 => (*c as usize, *t as usize),
        _ => {
            return Err(EmbeddingError::UnexpectedOutputShape {
                got: dims.to_vec(),
                expected_dim: 0,
            });
        }
    };
    // ONNX emits channel-major `(C_mfa, T)` (batch dropped). Transpose
    // to frame-major `(T, C_mfa)` so the streaming ring buffer can
    // append / slice whole frames contiguously.
    let mut framewise = vec![0.0_f32; c_mfa * t];
    for ch in 0..c_mfa {
        let src_row = &data[ch * t..(ch + 1) * t];
        for (frame, &v) in src_row.iter().enumerate() {
            framewise[frame * c_mfa + ch] = v;
        }
    }
    Ok(FrameFeatures {
        data: framewise,
        n_frames: t,
        c_mfa,
    })
}

/// Run the pooler graph: frame-contiguous `(n_frames, c_mfa)` →
/// `embedding (192)`. Transposes back to the channel-major
/// `(1, C_mfa, T)` the ONNX graph expects.
fn run_pooler(
    session: &mut Session,
    frame_data: &[f32],
    n_frames: usize,
    c_mfa: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    if frame_data.len() != n_frames * c_mfa {
        return Err(EmbeddingError::Shape(ndarray::ShapeError::from_kind(
            ndarray::ErrorKind::IncompatibleShape,
        )));
    }
    let mut chanwise = vec![0.0_f32; c_mfa * n_frames];
    for frame in 0..n_frames {
        let src_row = &frame_data[frame * c_mfa..(frame + 1) * c_mfa];
        for (ch, &v) in src_row.iter().enumerate() {
            chanwise[ch * n_frames + frame] = v;
        }
    }
    let array: Array3<f32> = Array3::from_shape_vec((1, c_mfa, n_frames), chanwise)?;
    let input = TensorRef::from_array_view(&array)?;
    let outputs = session.run(ort::inputs!["frame_features" => input])?;
    let view = outputs["embedding"].try_extract_tensor::<f32>()?;
    let (shape, data) = view;
    embedding_from_view(&shape, data)
}

impl EcapaTdnn {
    /// Load a monolithic embedding-only ECAPA-TDNN ONNX from `path`.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when the file is missing or not
    /// a valid ONNX model that the bundled ONNX Runtime can lower.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        Ok(Self {
            backend: Backend::Mono(build_session(path)?),
        })
    }

    /// Load the #119 split: the `encoder` ONNX
    /// (`features → frame_features`) and the `pooler` ONNX
    /// (`frame_features → embedding`), both produced by
    /// `scripts/export_ecapa_onnx.py export --mode encoder|pooler`.
    ///
    /// A split model serves [`Self::embed_features`] (encoder then
    /// pooler, equivalent to the monolithic graph) **and** the
    /// independent [`Self::encode_features`] / [`Self::pool_frame_features`]
    /// halves used by the streaming engine to run the conv front-end
    /// and the pooling head at different cadences.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when either file is missing or
    /// not a valid ONNX model.
    pub fn from_split_onnx_paths(
        encoder: impl AsRef<Path>,
        pooler: impl AsRef<Path>,
    ) -> Result<Self, EmbeddingError> {
        Ok(Self {
            backend: Backend::Split {
                encoder: build_session(encoder)?,
                pooler: build_session(pooler)?,
            },
        })
    }

    /// `true` when this model was built from a split encoder + pooler
    /// pair and therefore supports [`Self::encode_features`] /
    /// [`Self::pool_frame_features`].
    #[must_use]
    pub fn supports_split(&self) -> bool {
        matches!(self.backend, Backend::Split { .. })
    }

    /// Run full inference on a pre-computed Fbank.
    ///
    /// `features` is the row-major flattened `(T_frames × n_mels)`
    /// buffer. Works for both backends: the monolithic graph runs in
    /// one session; the split graph chains encoder → pooler. Output is
    /// a 192-element vector.
    ///
    /// # Errors
    /// * [`EmbeddingError::Shape`] when `features.len() != n_frames * n_mels`
    /// * [`EmbeddingError::Ort`] when the ONNX runtime rejects the input
    /// * [`EmbeddingError::UnexpectedOutputShape`] when an output is
    ///   not the expected shape
    pub fn embed_features(
        &mut self,
        features: &[f32],
        n_frames: usize,
        n_mels: usize,
    ) -> Result<Vec<f32>, EmbeddingError> {
        match &mut self.backend {
            Backend::Mono(session) => {
                let array: Array3<f32> =
                    Array3::from_shape_vec((1, n_frames, n_mels), features.to_vec())?;
                let input = TensorRef::from_array_view(&array)?;
                let outputs = session.run(ort::inputs!["features" => input])?;
                let view = outputs["embedding"].try_extract_tensor::<f32>()?;
                let (shape, data) = view;
                embedding_from_view(&shape, data)
            }
            Backend::Split { encoder, pooler } => {
                let frames = run_encoder(encoder, features, n_frames, n_mels)?;
                run_pooler(pooler, &frames.data, frames.n_frames, frames.c_mfa)
            }
        }
    }

    /// Run the encoder half only: `features (T_frames × n_mels)` →
    /// per-frame [`FrameFeatures`]. Split-backend only.
    ///
    /// The encoder is the heavy, time-invariant conv front-end. A
    /// streaming caller runs it on overlapping windows and keeps the
    /// receptive-field-valid interior frames in a ring buffer (see
    /// `crate::streaming`), then pools that ring at a finer cadence
    /// via [`Self::pool_frame_features`].
    ///
    /// # Errors
    /// * [`EmbeddingError::NotSplit`] on a monolithic model
    /// * [`EmbeddingError::Shape`] when `features.len() != n_frames * n_mels`
    /// * [`EmbeddingError::Ort`] / [`EmbeddingError::UnexpectedOutputShape`]
    ///   on an ONNX failure or unexpected output rank
    pub fn encode_features(
        &mut self,
        features: &[f32],
        n_frames: usize,
        n_mels: usize,
    ) -> Result<FrameFeatures, EmbeddingError> {
        match &mut self.backend {
            Backend::Split { encoder, .. } => run_encoder(encoder, features, n_frames, n_mels),
            Backend::Mono(_) => Err(EmbeddingError::NotSplit),
        }
    }

    /// Run the pooler half only: a frame-contiguous `(n_frames, c_mfa)`
    /// feature map → 192-dim embedding. Split-backend only.
    ///
    /// `frame_data` is the row-major `(n_frames × c_mfa)` buffer — a
    /// slice of [`FrameFeatures::data`], or a window of the streaming
    /// engine's frame-feature ring.
    ///
    /// # Errors
    /// * [`EmbeddingError::NotSplit`] on a monolithic model
    /// * [`EmbeddingError::Shape`] when `frame_data.len() != n_frames * c_mfa`
    /// * [`EmbeddingError::Ort`] / [`EmbeddingError::UnexpectedOutputShape`]
    ///   on an ONNX failure or unexpected output shape
    pub fn pool_frame_features(
        &mut self,
        frame_data: &[f32],
        n_frames: usize,
        c_mfa: usize,
    ) -> Result<Vec<f32>, EmbeddingError> {
        match &mut self.backend {
            Backend::Split { pooler, .. } => run_pooler(pooler, frame_data, n_frames, c_mfa),
            Backend::Mono(_) => Err(EmbeddingError::NotSplit),
        }
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

    /// Path to a local monolithic ECAPA ONNX file. Tests that need to
    /// run actual inference are gated on this env var; if unset the
    /// test exits early with a printed skip notice so the suite still
    /// reports success.
    const ENV_VAR: &str = "MELLONELLA_ECAPA_ONNX";
    /// Paths to the #119 split encoder / pooler ONNX files.
    const ENV_ENCODER: &str = "MELLONELLA_ECAPA_ENCODER_ONNX";
    const ENV_POOLER: &str = "MELLONELLA_ECAPA_POOLER_ONNX";

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

    fn skip_unless_split_available() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let (Ok(enc), Ok(pool)) = (std::env::var(ENV_ENCODER), std::env::var(ENV_POOLER)) else {
            eprintln!("[skip] {ENV_ENCODER} / {ENV_POOLER} not set");
            return None;
        };
        let (enc, pool) = (std::path::PathBuf::from(enc), std::path::PathBuf::from(pool));
        if !enc.exists() || !pool.exists() {
            eprintln!("[skip] split ONNX file(s) missing");
            return None;
        }
        Some((enc, pool))
    }

    fn synthetic_fbank(n_frames: usize, n_mels: usize) -> Vec<f32> {
        let mut feats = vec![0.0_f32; n_frames * n_mels];
        for (i, slot) in feats.iter_mut().enumerate() {
            *slot = (i as f32 * 0.01).sin();
        }
        feats
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-12)
    }

    #[test]
    fn loads_and_runs_against_local_onnx() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = EcapaTdnn::from_onnx_path(&path).expect("load ONNX");
        assert!(!model.supports_split());

        let (n_frames, n_mels) = (100, 80);
        let feats = synthetic_fbank(n_frames, n_mels);

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

    #[test]
    fn split_methods_error_on_monolithic_model() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = EcapaTdnn::from_onnx_path(&path).expect("load ONNX");
        let feats = synthetic_fbank(100, 80);
        assert!(matches!(
            model.encode_features(&feats, 100, 80),
            Err(EmbeddingError::NotSplit)
        ));
        assert!(matches!(
            model.pool_frame_features(&[0.0; 16], 1, 16),
            Err(EmbeddingError::NotSplit)
        ));
    }

    #[test]
    fn split_encoder_pooler_matches_monolithic() {
        // verify-split parity, Rust side: pooler(encoder(x)) must
        // reproduce the monolithic embedding-only graph. Gated on all
        // three ONNX env vars being set.
        let Some(mono_path) = skip_unless_onnx_available() else {
            return;
        };
        let Some((enc_path, pool_path)) = skip_unless_split_available() else {
            return;
        };
        let mut mono = EcapaTdnn::from_onnx_path(&mono_path).expect("load mono ONNX");
        let mut split =
            EcapaTdnn::from_split_onnx_paths(&enc_path, &pool_path).expect("load split ONNX");
        assert!(split.supports_split());

        let (n_frames, n_mels) = (140, 80);
        let feats = synthetic_fbank(n_frames, n_mels);

        let emb_mono = mono
            .embed_features(&feats, n_frames, n_mels)
            .expect("mono inference");
        // Via the combined split path.
        let emb_split = split
            .embed_features(&feats, n_frames, n_mels)
            .expect("split inference");
        // Via the independent encoder + pooler halves.
        let frames = split
            .encode_features(&feats, n_frames, n_mels)
            .expect("encode");
        assert_eq!(frames.n_frames, n_frames);
        assert!(frames.c_mfa > 0);
        let emb_halves = split
            .pool_frame_features(&frames.data, frames.n_frames, frames.c_mfa)
            .expect("pool");

        // The split graph is mathematically the monolithic graph cut in
        // half; cosine parity should be essentially exact (the
        // export-side verify-split bar is 1e-4).
        assert!(
            cosine(&emb_mono, &emb_split) > 1.0 - 1e-4,
            "split embed_features diverges from monolithic: cos={}",
            cosine(&emb_mono, &emb_split)
        );
        assert!(
            cosine(&emb_mono, &emb_halves) > 1.0 - 1e-4,
            "encode+pool halves diverge from monolithic: cos={}",
            cosine(&emb_mono, &emb_halves)
        );
    }
}
