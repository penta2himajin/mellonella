//! Causal Conv-TasNet target speaker extraction (Stage C) — **stateful
//! per-chunk** ONNX wrapper.
//!
//! Loads the streaming ONNX exported by `scripts/export_tse_onnx.py`.
//! The exported graph consumes one fixed-size audio chunk plus the
//! model's causal-conv state tensors and returns the extracted chunk
//! plus the updated state. Live callers thread the state across calls.
//!
//! Mirrors the design of [`crate::dfn3::Dfn3`]: a single
//! [`ort::session::Session`] wrapped in [`TseSession`], with explicit
//! state buffers owned by the wrapper.
//!
//! # ONNX I/O contract
//!
//! Names (verbatim from `scripts/export_tse_onnx.py`):
//!
//! | name              | shape                          | notes                       |
//! |-------------------|--------------------------------|------------------------------|
//! | `audio_chunk`     | `[1, chunk_len]`                | mixture chunk (f32)         |
//! | `cond_embedding`  | `[1, 192]`                      | frozen ECAPA embedding      |
//! | `state_in_0`      | `[1, 1, enc_overlap]`           | encoder overlap             |
//! | `state_in_1..3`   | `[1, 1, 1]` × 3                 | input-norm sum/sqsum/count  |
//! | `state_in_4..`    | per TCN block × N (see below)   | depthwise ring + 2 cln triples |
//! | `state_in_{K-1}`  | `[1, 1, enc_overlap]`           | decoder overlap             |
//!
//! For each TCN block (in `(repeat, block)` order with dilations
//! `1, 2, 4, …, 2^(n_blocks-1)`), 7 tensors are emitted:
//!
//! | offset | shape                              | role               |
//! |--------|-------------------------------------|--------------------|
//! | 0      | `[1, hidden, (kernel-1) * dilation]` | depthwise ringbuf  |
//! | 1..3   | `[1, 1, 1]`                          | cln1 sum/sqsum/cnt |
//! | 4..6   | `[1, 1, 1]`                          | cln2 sum/sqsum/cnt |
//!
//! Output names: `extracted_chunk` plus `state_out_0..state_out_{K-1}`
//! with the same shape layout.
//!
//! # Streaming protocol
//!
//! 1. [`TseSession::from_onnx_path`] — load + zero-initialise state.
//! 2. [`TseSession::process_chunk`] for each chunk: input chunk length
//!    must be a positive multiple of `enc_stride` (and must match the
//!    fixed chunk length the ONNX was exported with — the graph has no
//!    dynamic axes).
//! 3. [`TseSession::reset`] between independent recordings.
//!
//! No wiring into the main pipeline yet; that's a follow-up PR.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_borrow
)]

use std::path::Path;

use ndarray::{Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::TensorRef;

use crate::embedding::EmbeddingError;

/// Dimensionality of the frozen ECAPA enrollment embedding the model
/// is conditioned on (SpeakerBeam-style FiLM).
pub const TSE_COND_DIM: usize = 192;

/// Frozen hyper-parameters of the streaming TSE model — **must** match
/// the values baked into the ONNX export.
///
/// The streaming ONNX has a fixed chunk length and fixed state shapes,
/// so the loader needs to know the architecture up front. These defaults
/// match `TSEConfig.poc_16k()` in `training/tse/config.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TseConfig {
    /// Encoder analysis-window stride. The chunk length must be a
    /// positive multiple of this.
    pub enc_stride: usize,
    /// Encoder analysis-window overlap (`enc_kernel - enc_stride`).
    /// Sets the size of the encoder + decoder overlap state tensors.
    pub enc_overlap: usize,
    /// TCN bottleneck hidden width (`H`). Sets the per-block ringbuf
    /// channel count.
    pub hidden: usize,
    /// Depthwise-conv kernel size.
    pub tcn_kernel: usize,
    /// Number of conv blocks per TCN repeat. Dilations follow the
    /// standard `1, 2, 4, …, 2^(n_blocks-1)` schedule.
    pub n_blocks: usize,
    /// Number of TCN repeats.
    pub n_repeats: usize,
}

impl TseConfig {
    /// Default PoC config — 16 kHz, encoder kernel 32 / stride 16,
    /// 2×6 TCN, hidden 256.
    #[must_use]
    pub const fn poc_16k() -> Self {
        Self {
            enc_stride: 16,
            enc_overlap: 16, // enc_kernel(32) - enc_stride(16)
            hidden: 256,
            tcn_kernel: 3,
            n_blocks: 6,
            n_repeats: 2,
        }
    }

    /// Number of TCN blocks in the separator (`n_repeats * n_blocks`).
    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.n_repeats * self.n_blocks
    }

    /// Number of state tensors threaded across `process_chunk` calls.
    ///
    /// Layout: `1 (enc overlap) + 3 (input-norm) + 7 * total_blocks
    /// + 1 (dec overlap)`.
    #[must_use]
    pub const fn n_state_tensors(&self) -> usize {
        1 + 3 + 7 * self.total_blocks() + 1
    }

    /// Dilation of the `block`-th block in any repeat — the dilation
    /// schedule resets at each repeat (`1, 2, 4, …, 2^(n_blocks-1)`).
    #[must_use]
    const fn block_dilation(block: usize) -> usize {
        1usize << block
    }

    /// Per-block depthwise ringbuffer left-context size in samples
    /// (`(kernel - 1) * dilation`).
    #[must_use]
    const fn block_pad(&self, block: usize) -> usize {
        (self.tcn_kernel - 1) * Self::block_dilation(block)
    }
}

/// Errors returned by [`TseSession`].
#[derive(Debug)]
pub enum TseError {
    /// ONNX Runtime returned a failure (model load, kernel error, …).
    Ort(String),
    /// ndarray shape construction failed — typically a wrong chunk
    /// length or state-tensor size.
    Shape(ndarray::ShapeError),
    /// Caller fed a chunk whose length is not a positive multiple of
    /// the encoder stride (the ONNX graph has a fixed time axis).
    InvalidChunkLength { got: usize, enc_stride: usize },
    /// `cond_embedding.len() != TSE_COND_DIM`.
    InvalidCondLength { got: usize, expected: usize },
    /// An ONNX output tensor had an unexpected shape — the ONNX file is
    /// incompatible with this build of `mellonella-core`.
    UnexpectedOutputShape {
        name: String,
        got: Vec<i64>,
        expected: Vec<usize>,
    },
}

impl std::fmt::Display for TseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ort(e) => write!(f, "ONNX Runtime error: {e}"),
            Self::Shape(e) => write!(f, "ndarray shape error: {e}"),
            Self::InvalidChunkLength { got, enc_stride } => write!(
                f,
                "TSE chunk length {got} must be a positive multiple of enc_stride {enc_stride}"
            ),
            Self::InvalidCondLength { got, expected } => {
                write!(f, "TSE cond_embedding length {got} != expected {expected}")
            }
            Self::UnexpectedOutputShape {
                name,
                got,
                expected,
            } => write!(
                f,
                "TSE ONNX output {name:?} has shape {got:?}, expected {expected:?}"
            ),
        }
    }
}

impl std::error::Error for TseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Shape(e) => Some(e),
            _ => None,
        }
    }
}

impl<R> From<ort::Error<R>> for TseError {
    fn from(e: ort::Error<R>) -> Self {
        Self::Ort(e.to_string())
    }
}

impl From<ndarray::ShapeError> for TseError {
    fn from(e: ndarray::ShapeError) -> Self {
        Self::Shape(e)
    }
}

// The TSE module currently reuses the embedding-side error string when
// callers bridge through `EmbeddingError`; the conversion is one-way
// (this is a TSE-specific error, but call sites that use `EmbeddingError`
// as a catch-all can still propagate it).
impl From<TseError> for EmbeddingError {
    fn from(e: TseError) -> Self {
        match e {
            TseError::Ort(s) => Self::Ort(s),
            TseError::Shape(s) => Self::Shape(s),
            TseError::InvalidChunkLength { .. } | TseError::InvalidCondLength { .. } => {
                Self::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                ))
            }
            TseError::UnexpectedOutputShape { got, expected, .. } => Self::UnexpectedOutputShape {
                got,
                expected_dim: expected.iter().product::<usize>(),
            },
        }
    }
}

/// Build a zero-initialised state list whose tensor shapes follow the
/// layout documented at the top of the file.
fn make_initial_state(config: TseConfig) -> Vec<ArrayD<f32>> {
    let mut state = Vec::with_capacity(config.n_state_tensors());
    // 1) Encoder overlap: (1, 1, enc_overlap).
    state.push(ArrayD::<f32>::zeros(IxDyn(&[1, 1, config.enc_overlap])));
    // 2) Input-norm running stats: (1, 1, 1) × 3.
    for _ in 0..3 {
        state.push(ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1])));
    }
    // 3) Per TCN block: ringbuf + cln1 + cln2.
    for _r in 0..config.n_repeats {
        for b in 0..config.n_blocks {
            let pad = config.block_pad(b);
            state.push(ArrayD::<f32>::zeros(IxDyn(&[1, config.hidden, pad])));
            for _ in 0..6 {
                state.push(ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1])));
            }
        }
    }
    // 4) Decoder overlap: (1, 1, enc_overlap).
    state.push(ArrayD::<f32>::zeros(IxDyn(&[1, 1, config.enc_overlap])));
    state
}

/// Stateful TSE ONNX inference wrapper.
///
/// Threads the explicit causal-conv state list across
/// [`TseSession::process_chunk`] calls; the state is owned internally
/// and cleared by [`TseSession::reset`].
pub struct TseSession {
    session: Session,
    config: TseConfig,
    state: Vec<ArrayD<f32>>,
}

impl TseSession {
    /// Load the streaming TSE ONNX from `path` with the PoC config.
    ///
    /// # Errors
    /// Returns [`TseError::Ort`] when the file is missing or rejected
    /// by the ort runtime.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, TseError> {
        Self::from_onnx_path_with_config(path, TseConfig::poc_16k())
    }

    /// Load the streaming TSE ONNX from `path` for a specific config.
    /// Exposed for future configs (e.g. `prod_48k`); for now only the
    /// PoC is exported by `scripts/export_tse_onnx.py`.
    ///
    /// # Errors
    /// Returns [`TseError::Ort`] when the file is missing or rejected
    /// by the ort runtime.
    pub fn from_onnx_path_with_config(
        path: impl AsRef<Path>,
        config: TseConfig,
    ) -> Result<Self, TseError> {
        let session = Session::builder()?
            .with_intra_threads(crate::ort_threads::intra_op_threads())?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(Self {
            session,
            config,
            state: make_initial_state(config),
        })
    }

    /// Frozen architectural config of this session.
    #[must_use]
    pub const fn config(&self) -> TseConfig {
        self.config
    }

    /// Number of state tensors threaded across `process_chunk` calls.
    #[must_use]
    pub const fn n_state_tensors(&self) -> usize {
        self.config.n_state_tensors()
    }

    /// Zero-initialise the streaming state — call between independent
    /// recordings.
    pub fn reset(&mut self) {
        self.state = make_initial_state(self.config);
    }

    /// Run one streaming forward pass on a single audio chunk.
    ///
    /// `chunk.len()` must be a positive multiple of `enc_stride` (and
    /// must match the chunk length the ONNX was exported with — the
    /// graph has a fixed time axis). `cond_embedding` must have length
    /// [`TSE_COND_DIM`].
    ///
    /// On success the internal state is replaced with the model's
    /// returned new state and the extracted chunk is returned.
    ///
    /// # Errors
    /// * [`TseError::InvalidChunkLength`] when `chunk.len()` is zero or
    ///   not a multiple of `enc_stride`.
    /// * [`TseError::InvalidCondLength`] when `cond_embedding.len() !=
    ///   TSE_COND_DIM`.
    /// * [`TseError::Ort`] / [`TseError::Shape`] when the runtime
    ///   rejects the input.
    /// * [`TseError::UnexpectedOutputShape`] when an output tensor
    ///   doesn't match the expected layout.
    pub fn process_chunk(
        &mut self,
        chunk: &[f32],
        cond_embedding: &[f32; TSE_COND_DIM],
    ) -> Result<Vec<f32>, TseError> {
        self.process_chunk_slice(chunk, cond_embedding)
    }

    /// Same as [`Self::process_chunk`] but accepts `cond_embedding` as
    /// a slice for convenience (must still have length `TSE_COND_DIM`).
    ///
    /// # Errors
    /// See [`Self::process_chunk`].
    pub fn process_chunk_slice(
        &mut self,
        chunk: &[f32],
        cond_embedding: &[f32],
    ) -> Result<Vec<f32>, TseError> {
        let chunk_len = chunk.len();
        if chunk_len == 0 || chunk_len % self.config.enc_stride != 0 {
            return Err(TseError::InvalidChunkLength {
                got: chunk_len,
                enc_stride: self.config.enc_stride,
            });
        }
        if cond_embedding.len() != TSE_COND_DIM {
            return Err(TseError::InvalidCondLength {
                got: cond_embedding.len(),
                expected: TSE_COND_DIM,
            });
        }

        let audio_arr = Array2::<f32>::from_shape_vec((1, chunk_len), chunk.to_vec())?;
        let cond_arr = Array2::<f32>::from_shape_vec((1, TSE_COND_DIM), cond_embedding.to_vec())?;

        // Build the named input feed. Names mirror the export script
        // (`audio_chunk`, `cond_embedding`, `state_in_{i}`). We use the
        // `Vec<(Cow<str>, SessionInputValue)>` form of `SessionInputs`,
        // which is what ort rc.12 documents for variable-length input
        // maps (the `ort::inputs!` macro fixes the arg count).
        let n_state = self.n_state_tensors();
        let mut inputs: Vec<(
            std::borrow::Cow<'static, str>,
            ort::session::SessionInputValue<'_>,
        )> = Vec::with_capacity(2 + n_state);
        inputs.push((
            std::borrow::Cow::Borrowed("audio_chunk"),
            TensorRef::from_array_view(&audio_arr)?.into(),
        ));
        inputs.push((
            std::borrow::Cow::Borrowed("cond_embedding"),
            TensorRef::from_array_view(&cond_arr)?.into(),
        ));
        for (i, state_tensor) in self.state.iter().enumerate() {
            inputs.push((
                std::borrow::Cow::Owned(format!("state_in_{i}")),
                TensorRef::from_array_view(state_tensor)?.into(),
            ));
        }
        let outputs = self.session.run(inputs)?;

        // Pull out the extracted chunk first.
        let (extr_shape, extr_data) = outputs["extracted_chunk"].try_extract_tensor::<f32>()?;
        let extr_dims: &[i64] = &extr_shape;
        if extr_dims.len() != 2 || extr_dims[0] != 1 || extr_dims[1] as usize != chunk_len {
            return Err(TseError::UnexpectedOutputShape {
                name: "extracted_chunk".to_string(),
                got: extr_dims.to_vec(),
                expected: vec![1, chunk_len],
            });
        }
        let extracted = extr_data.to_vec();

        // Replace the streaming state with the model's new state.
        let mut new_state: Vec<ArrayD<f32>> = Vec::with_capacity(n_state);
        for i in 0..n_state {
            let name = format!("state_out_{i}");
            let (shape, data) = outputs[name.as_str()].try_extract_tensor::<f32>()?;
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let arr = ArrayD::<f32>::from_shape_vec(IxDyn(&dims), data.to_vec())?;
            new_state.push(arr);
        }
        self.state = new_state;

        Ok(extracted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poc_config_invariants() {
        let cfg = TseConfig::poc_16k();
        assert_eq!(cfg.enc_stride, 16);
        assert_eq!(cfg.enc_overlap, 16);
        assert_eq!(cfg.hidden, 256);
        assert_eq!(cfg.tcn_kernel, 3);
        assert_eq!(cfg.n_blocks, 6);
        assert_eq!(cfg.n_repeats, 2);
        assert_eq!(cfg.total_blocks(), 12);
        // 1 enc + 3 input-norm + 7 * 12 blocks + 1 dec = 89.
        assert_eq!(cfg.n_state_tensors(), 89);
    }

    #[test]
    fn dilation_schedule_matches_python() {
        let cfg = TseConfig::poc_16k();
        // Python: tuple(2**i for i in range(n_blocks))
        let expected: [usize; 6] = [1, 2, 4, 8, 16, 32];
        for (b, &want) in expected.iter().enumerate() {
            assert_eq!(TseConfig::block_dilation(b), want);
            assert_eq!(cfg.block_pad(b), (cfg.tcn_kernel - 1) * want);
        }
    }

    #[test]
    fn initial_state_shapes_match_python() {
        let cfg = TseConfig::poc_16k();
        let state = make_initial_state(cfg);
        assert_eq!(state.len(), cfg.n_state_tensors());

        // enc overlap
        assert_eq!(state[0].shape(), &[1, 1, cfg.enc_overlap]);
        // input-norm sum/sqsum/count
        for tensor in state.iter().take(4).skip(1) {
            assert_eq!(tensor.shape(), &[1, 1, 1]);
        }
        // per block
        let mut idx = 4;
        for _r in 0..cfg.n_repeats {
            for b in 0..cfg.n_blocks {
                let pad = cfg.block_pad(b);
                assert_eq!(state[idx].shape(), &[1, cfg.hidden, pad]);
                idx += 1;
                for _ in 0..6 {
                    assert_eq!(state[idx].shape(), &[1, 1, 1]);
                    idx += 1;
                }
            }
        }
        // dec overlap
        assert_eq!(state[idx].shape(), &[1, 1, cfg.enc_overlap]);
        assert_eq!(idx + 1, cfg.n_state_tensors());
    }
}
