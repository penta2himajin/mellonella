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

use std::collections::VecDeque;
use std::path::Path;

// The `deep_filter` crate is published as the `df` lib (see its
// Cargo.toml `[lib] name = "df"`).
use df::{
    band_mean_norm_erb, band_unit_norm, compute_band_corr, Complex32, DFState, MEAN_NORM_INIT,
    UNIT_NORM_INIT,
};
use ndarray::{Array4, Array5};
use ort::session::Session;
use ort::value::TensorRef;

use crate::embedding::EmbeddingError;

/// Sample rate the DFN3 model is trained on.
pub const DFN3_SR: usize = 48_000;
/// FFT size used by the analysis / synthesis stages.
pub const DFN3_FFT: usize = 960;
/// Hop size (50 % overlap).
pub const DFN3_HOP: usize = 480;
/// Minimum frequency bins per ERB band (matches `df_state` defaults).
pub const MIN_NB_FREQS: usize = 2;
/// EMA factor for the ERB / unit-norm states (matches Python
/// `get_norm_alpha(False)` after model load).
pub const NORM_ALPHA: f32 = 0.99;

/// Required STFT frame count per inference call. See module docs.
pub const FRAMES_PER_CHUNK: usize = 102;
/// Frequency bins in the spectrogram (`fft_size/2 + 1` for fft_size=960).
pub const N_FREQ: usize = 481;
/// Number of low frequency bins fed to the DF feature path.
pub const NB_DF: usize = 96;
/// Number of ERB filterbank bands.
pub const N_ERB: usize = 32;
/// Audio samples consumed / produced by one chunk.
pub const SAMPLES_PER_CHUNK: usize = FRAMES_PER_CHUNK * DFN3_HOP;

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
        let session = Session::builder()?
            .with_intra_threads(crate::ort_threads::intra_op_threads())?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
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

/// Linear interpolation of a two-value initialiser across `n` bins.
/// `MEAN_NORM_INIT = [-60, -90]` and `UNIT_NORM_INIT = [0.001, 0.0001]`
/// are stored as endpoints by `deep_filter`; the live state is the
/// per-band interpolation between them. Mirrors libdf's binding.
fn init_state_lerp(init: [f32; 2], n: usize) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    if n <= 1 {
        return vec![init[0]; n];
    }
    (0..n)
        .map(|i| init[0] + (init[1] - init[0]) * (i as f32) / ((n as f32) - 1.0))
        .collect()
}

/// End-to-end DFN3 pipeline: 48 kHz audio in → 48 kHz audio out.
///
/// Wraps:
///
/// * `deep_filter::DFState` for STFT / iSTFT and ERB filterbank widths
/// * the [`Dfn3`] ONNX wrapper for the neural step
/// * EMA state for `band_mean_norm_erb` (32 bins) and
///   `band_unit_norm` (`NB_DF` bins)
///
/// Single-chunk only for now: callers hand in a buffer at most
/// [`SAMPLES_PER_CHUNK`] long (zero-padded internally) and get a fresh
/// buffer back of the same length. Streaming multi-chunk support
/// follows in a later PR once the state carry-over semantics across
/// chunk boundaries are tested.
pub struct Dfn3Pipeline {
    dfstate: DFState,
    net: Dfn3,
    erb_widths: Vec<usize>,
    erb_norm_state: Vec<f32>,
    df_norm_state: Vec<f32>,
    norm_alpha: f32,
}

impl Dfn3Pipeline {
    /// Build a pipeline that loads the patched DFN3 ONNX from `path`.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when the ONNX is missing or
    /// rejected by the ort runtime.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let dfstate = DFState::new(DFN3_SR, DFN3_FFT, DFN3_HOP, N_ERB, MIN_NB_FREQS);
        let erb_widths = dfstate.erb.clone();
        let net = Dfn3::from_onnx_path(path)?;
        Ok(Self {
            dfstate,
            net,
            erb_widths,
            erb_norm_state: init_state_lerp(MEAN_NORM_INIT, N_ERB),
            df_norm_state: init_state_lerp(UNIT_NORM_INIT, NB_DF),
            norm_alpha: NORM_ALPHA,
        })
    }

    /// Reset the STFT memory + EMA states. Call between independent
    /// recordings; not needed between consecutive frames of the same
    /// recording.
    pub fn reset(&mut self) {
        self.dfstate.reset();
        self.erb_norm_state = init_state_lerp(MEAN_NORM_INIT, N_ERB);
        self.df_norm_state = init_state_lerp(UNIT_NORM_INIT, NB_DF);
    }

    /// Enhance `audio` (≤ [`SAMPLES_PER_CHUNK`] samples @ 48 kHz mono).
    /// Returns a buffer of exactly [`SAMPLES_PER_CHUNK`] samples.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
        let mut padded = vec![0.0_f32; SAMPLES_PER_CHUNK];
        let copy_len = audio.len().min(SAMPLES_PER_CHUNK);
        padded[..copy_len].copy_from_slice(&audio[..copy_len]);

        let mut spec_buf = vec![0.0_f32; FRAMES_PER_CHUNK * N_FREQ * 2];
        let mut erb_buf = vec![0.0_f32; FRAMES_PER_CHUNK * N_ERB];
        let mut df_spec_buf = vec![0.0_f32; FRAMES_PER_CHUNK * NB_DF * 2];

        let mut spec_frame = vec![Complex32::new(0.0, 0.0); N_FREQ];
        let mut band_pow = vec![0.0_f32; N_ERB];

        for i in 0..FRAMES_PER_CHUNK {
            let start = i * DFN3_HOP;
            let audio_frame = &padded[start..start + DFN3_HOP];
            self.dfstate.analysis(audio_frame, &mut spec_frame);

            for (k, &c) in spec_frame.iter().enumerate() {
                spec_buf[(i * N_FREQ + k) * 2] = c.re;
                spec_buf[(i * N_FREQ + k) * 2 + 1] = c.im;
            }

            compute_band_corr(&mut band_pow, &spec_frame, &spec_frame, &self.erb_widths);
            for b in &mut band_pow {
                *b = 10.0 * (1e-10_f32 + *b).log10();
            }
            band_mean_norm_erb(&mut band_pow, &mut self.erb_norm_state, self.norm_alpha);
            for (k, &p) in band_pow.iter().enumerate() {
                erb_buf[i * N_ERB + k] = p;
            }

            let mut df_spec_frame: Vec<Complex32> = spec_frame[..NB_DF].to_vec();
            band_unit_norm(&mut df_spec_frame, &mut self.df_norm_state, self.norm_alpha);
            for (k, &c) in df_spec_frame.iter().enumerate() {
                df_spec_buf[(i * NB_DF + k) * 2] = c.re;
                df_spec_buf[(i * NB_DF + k) * 2 + 1] = c.im;
            }
        }

        let enhanced_spec = self.net.enhance_spec(&spec_buf, &erb_buf, &df_spec_buf)?;

        let mut out_audio = vec![0.0_f32; SAMPLES_PER_CHUNK];
        let mut out_frame = vec![0.0_f32; DFN3_HOP];
        let mut enh_frame = vec![Complex32::new(0.0, 0.0); N_FREQ];
        for i in 0..FRAMES_PER_CHUNK {
            for k in 0..N_FREQ {
                enh_frame[k] = Complex32::new(
                    enhanced_spec[(i * N_FREQ + k) * 2],
                    enhanced_spec[(i * N_FREQ + k) * 2 + 1],
                );
            }
            self.dfstate.synthesis(&mut enh_frame, &mut out_frame);
            out_audio[i * DFN3_HOP..(i + 1) * DFN3_HOP].copy_from_slice(&out_frame);
        }

        Ok(out_audio)
    }
}

/// Streaming wrapper around [`Dfn3Pipeline`] for live use.
///
/// `Dfn3Pipeline::process` requires a fixed [`SAMPLES_PER_CHUNK`]
/// (48 960 samples ≈ 1.02 s @ 48 kHz) buffer because the patched
/// DFN3 ONNX export is shape-locked to 102 STFT frames. Live audio
/// callbacks deliver much smaller chunks (5–50 ms typical), so the
/// streaming wrapper:
///
/// 1. accumulates incoming samples in an internal queue,
/// 2. fires `Dfn3Pipeline::process` once a full chunk's worth is
///    queued,
/// 3. emits the enhanced audio in the same call,
/// 4. on `flush`, zero-pads any residue to the next chunk boundary
///    so the tail of a recording is still suppressed.
///
/// # Added latency
///
/// Because the ONNX export forces a 1.02-s window, the live path
/// holds back **up to [`SAMPLES_PER_CHUNK`] samples worth of
/// latency** on top of the existing 50–65 ms gating envelope
/// budget. Live callers should surface this trade-off to users —
/// the CLI's `--enable-dfn3` flag and the GUI's "Enable noise
/// suppression" toggle both document it. A future export with a
/// symbolic time dimension will let the wrapper drop the latency
/// to the DFN3 model's intrinsic ~30 ms.
pub struct Dfn3Streamer {
    pipeline: Dfn3Pipeline,
    /// Holds at most `SAMPLES_PER_CHUNK - 1` un-consumed samples
    /// between `push_samples` calls. Grows briefly during a push
    /// before whole-chunk units are drained.
    input_buffer: VecDeque<f32>,
}

impl Dfn3Streamer {
    /// Build a streamer that loads the DFN3 ONNX from `path`.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from
    /// [`Dfn3Pipeline::from_onnx_path`].
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        Ok(Self {
            pipeline: Dfn3Pipeline::from_onnx_path(path)?,
            input_buffer: VecDeque::with_capacity(SAMPLES_PER_CHUNK * 2),
        })
    }

    /// Append `samples` to the internal queue and emit any complete
    /// enhanced chunks. The returned vector is a multiple of
    /// [`SAMPLES_PER_CHUNK`] samples long (possibly empty when the
    /// queue hasn't yet filled).
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
        self.input_buffer.extend(samples.iter().copied());
        let mut out: Vec<f32> = Vec::new();
        while self.input_buffer.len() >= SAMPLES_PER_CHUNK {
            let chunk: Vec<f32> = self.input_buffer.drain(..SAMPLES_PER_CHUNK).collect();
            let enhanced = self.pipeline.process(&chunk)?;
            out.extend_from_slice(&enhanced);
        }
        Ok(out)
    }

    /// Drain any sub-chunk residue: zero-pad to one full chunk,
    /// process it, and return the enhanced output. Returns an empty
    /// `Vec` when the queue was already drained.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn flush(&mut self) -> Result<Vec<f32>, EmbeddingError> {
        if self.input_buffer.is_empty() {
            return Ok(Vec::new());
        }
        let mut chunk = vec![0.0_f32; SAMPLES_PER_CHUNK];
        for (i, s) in self.input_buffer.drain(..).enumerate() {
            chunk[i] = s;
        }
        self.pipeline.process(&chunk)
    }

    /// Reset STFT memory + EMA states and clear the input queue.
    /// Use between independent recordings; not needed between
    /// consecutive frames of the same recording.
    pub fn reset(&mut self) {
        self.pipeline.reset();
        self.input_buffer.clear();
    }

    /// Sub-chunk residue currently buffered (in samples). Useful
    /// for the GUI's latency display once it lands.
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.input_buffer.len()
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

    #[test]
    fn streamer_invariants_dont_need_onnx() {
        // Compile-time assertions of the chunk-size contract. Catches
        // a future export that accidentally changes
        // `SAMPLES_PER_CHUNK` without updating call sites that rely
        // on the 1.02 s figure.
        assert_eq!(SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK * DFN3_HOP);
        assert_eq!(SAMPLES_PER_CHUNK, 48_960);
        // 1.02 s @ 48 kHz, to two decimals.
        let secs = SAMPLES_PER_CHUNK as f32 / DFN3_SR as f32;
        assert!((secs - 1.02).abs() < 0.01, "expected ~1.02 s, got {secs}");
    }

    #[test]
    fn streamer_push_and_flush_round_trip() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut streamer = Dfn3Streamer::from_onnx_path(&path).expect("load DFN3 ONNX");
        // Push less than one chunk — should emit nothing.
        let head = vec![0.0_f32; 1024];
        let out = streamer.push_samples(&head).expect("push");
        assert!(out.is_empty(), "first sub-chunk push must hold output");
        assert_eq!(streamer.buffered_samples(), 1024);

        // Push enough to complete one chunk (1024 + remainder = SAMPLES_PER_CHUNK).
        let body = vec![0.0_f32; SAMPLES_PER_CHUNK - 1024];
        let out = streamer.push_samples(&body).expect("push");
        assert_eq!(out.len(), SAMPLES_PER_CHUNK, "one full chunk should emit");
        assert_eq!(streamer.buffered_samples(), 0);

        // Trailing residue + flush.
        let tail = vec![0.0_f32; 480];
        let out = streamer.push_samples(&tail).expect("push");
        assert!(
            out.is_empty(),
            "residue smaller than one chunk shouldn't emit"
        );
        let flushed = streamer.flush().expect("flush");
        assert_eq!(
            flushed.len(),
            SAMPLES_PER_CHUNK,
            "flush zero-pads to a full chunk"
        );
        // After flush the queue is empty; a second flush yields nothing.
        let again = streamer.flush().expect("flush idempotent");
        assert!(again.is_empty());
    }
}
