//! DeepFilterNet 3 ONNX inference wrapper — **stateful per-frame**.
//!
//! Loads the stateful ONNX exported by `scripts/export_dfn3_onnx.py`:
//! the model takes one STFT frame at a time plus the three GRU
//! hidden states, and returns the enhanced frame plus the updated
//! states. The caller maintains a rolling `conv_lookahead`-frame
//! future-feature buffer so the encoder can "see" the future
//! features required by DFN3's lookahead mechanism while the
//! algorithm's intrinsic delay stays at ~30 ms.
//!
//! # ONNX I/O contract
//!
//! Inputs (all `f32`):
//!
//! | name      | shape                            | notes                       |
//! |-----------|----------------------------------|------------------------------|
//! | spec      | `[1, 1, 1, N_FREQ, 2]`           | current STFT frame, re/im    |
//! | feat_erb  | `[1, 1, 1, N_ERB]`               | feat for frame `t+conv_la`   |
//! | feat_spec | `[1, 1, 1, NB_DF, 2]`            | feat for frame `t+conv_la`   |
//! | enc_h     | `[ENC_GRU_LAYERS, 1, GRU_HIDDEN]` | encoder GRU state           |
//! | erb_h     | `[ERB_GRU_LAYERS, 1, GRU_HIDDEN]` | erb decoder GRU state       |
//! | df_h      | `[DF_GRU_LAYERS, 1, GRU_HIDDEN]`  | df decoder GRU state        |
//!
//! Outputs: `enhanced_spec` (same shape as `spec`) and the three new
//! GRU states (same shapes as the inputs).
//!
//! # Why "feat from `t+conv_la`"?
//!
//! Upstream DFN3 shifts feature inputs forward by `conv_lookahead`
//! frames inside `DfNet.forward` (a `ConstantPad2d` that drops the
//! first `conv_la` feature frames and pads zeros at the end). For
//! streaming, that built-in shift breaks at chunk boundaries, so
//! the export script replaces `pad_feat` / `pad_spec` with
//! `Identity` and expects the caller to feed pre-shifted features.
//! `Dfn3Pipeline` handles this by keeping a small queue of recent
//! features and pairing the oldest queued spec with the latest
//! queued features on each model call.
//!
//! # Streaming protocol
//!
//! `Dfn3Pipeline` exposes:
//!
//! * `push_samples(audio)` — append samples, return any enhanced
//!   samples that became available (frame-aligned to the 480-sample
//!   hop, delayed by `conv_lookahead` frames = 20 ms @ 48 kHz).
//! * `flush()` — zero-pad to drain the lookahead pipeline, emit the
//!   trailing `conv_lookahead` frames.
//! * `process(audio)` — convenience wrapper: `reset() +
//!   push_samples(...) + flush()`, returning exactly `audio.len()`
//!   samples (zero-padded internally so the offline parity test sees
//!   a deterministic shape).
//! * `reset()` — STFT memory + EMA + GRU state + queues all back to
//!   defaults.

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
use ndarray::{Array3, Array4, Array5};
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

/// Frequency bins in the spectrogram (`fft_size/2 + 1` for fft_size=960).
pub const N_FREQ: usize = 481;
/// Number of low frequency bins fed to the DF feature path.
pub const NB_DF: usize = 96;
/// Number of ERB filterbank bands.
pub const N_ERB: usize = 32;
/// Encoder GRU layer count.
pub const ENC_GRU_LAYERS: usize = 1;
/// ERB decoder GRU layer count.
pub const ERB_GRU_LAYERS: usize = 2;
/// DF decoder GRU layer count.
pub const DF_GRU_LAYERS: usize = 2;
/// Hidden size of every GRU layer in the model.
pub const GRU_HIDDEN: usize = 256;
/// Feature-side lookahead the model needs. The caller pairs the
/// current spec with feat from this many frames in the future.
pub const CONV_LOOKAHEAD: usize = 2;

/// Compatibility constant — frame count the *old* (T=102) export
/// expected. Kept for downstream call sites that still use it as a
/// "1 s @ 48 kHz" buffer-size hint; the new ONNX is per-frame.
pub const FRAMES_PER_CHUNK: usize = 102;
/// Audio samples a `FRAMES_PER_CHUNK`-frame window covers. Kept for
/// the same compatibility reason as `FRAMES_PER_CHUNK`.
pub const SAMPLES_PER_CHUNK: usize = FRAMES_PER_CHUNK * DFN3_HOP;

/// Result of one stateful inference call.
#[derive(Debug, Clone)]
pub struct EnhancedFrame {
    /// Enhanced spectrum, layout `(N_FREQ, 2)` flattened (real,imag pairs).
    pub spec: Vec<f32>,
    pub new_enc_h: Vec<f32>,
    pub new_erb_h: Vec<f32>,
    pub new_df_h: Vec<f32>,
}

/// DFN3 ONNX inference wrapper. One inference call enhances a
/// single STFT frame, threading the three GRU hidden states.
pub struct Dfn3 {
    session: Session,
}

impl Dfn3 {
    /// Load the stateful DFN3 ONNX from `path`.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Ort`] when the file is missing or
    /// rejected by the ort runtime.
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let session = Session::builder()?
            .with_intra_threads(crate::ort_threads::intra_op_threads())?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(Self { session })
    }

    /// Run inference on one STFT frame.
    ///
    /// # Errors
    /// * [`EmbeddingError::Shape`] when an input length doesn't match
    ///   the contract
    /// * [`EmbeddingError::Ort`] when the runtime rejects the input
    /// * [`EmbeddingError::UnexpectedOutputShape`] when an output
    ///   isn't the expected shape
    pub fn enhance_frame(
        &mut self,
        spec: &[f32],
        feat_erb: &[f32],
        feat_spec: &[f32],
        enc_h: &[f32],
        erb_h: &[f32],
        df_h: &[f32],
    ) -> Result<EnhancedFrame, EmbeddingError> {
        let spec_expected = N_FREQ * 2;
        let erb_expected = N_ERB;
        let dfspec_expected = NB_DF * 2;
        let enc_expected = ENC_GRU_LAYERS * GRU_HIDDEN;
        let erb_h_expected = ERB_GRU_LAYERS * GRU_HIDDEN;
        let df_h_expected = DF_GRU_LAYERS * GRU_HIDDEN;
        if spec.len() != spec_expected
            || feat_erb.len() != erb_expected
            || feat_spec.len() != dfspec_expected
            || enc_h.len() != enc_expected
            || erb_h.len() != erb_h_expected
            || df_h.len() != df_h_expected
        {
            return Err(EmbeddingError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            )));
        }

        let spec_arr = Array5::from_shape_vec((1, 1, 1, N_FREQ, 2), spec.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let erb_arr = Array4::from_shape_vec((1, 1, 1, N_ERB), feat_erb.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let dfspec_arr = Array5::from_shape_vec((1, 1, 1, NB_DF, 2), feat_spec.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let enc_arr = Array3::from_shape_vec((ENC_GRU_LAYERS, 1, GRU_HIDDEN), enc_h.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let erb_h_arr = Array3::from_shape_vec((ERB_GRU_LAYERS, 1, GRU_HIDDEN), erb_h.to_vec())
            .map_err(EmbeddingError::Shape)?;
        let df_h_arr = Array3::from_shape_vec((DF_GRU_LAYERS, 1, GRU_HIDDEN), df_h.to_vec())
            .map_err(EmbeddingError::Shape)?;

        let outputs = self.session.run(ort::inputs![
            "spec" => TensorRef::from_array_view(&spec_arr)?,
            "feat_erb" => TensorRef::from_array_view(&erb_arr)?,
            "feat_spec" => TensorRef::from_array_view(&dfspec_arr)?,
            "enc_h" => TensorRef::from_array_view(&enc_arr)?,
            "erb_h" => TensorRef::from_array_view(&erb_h_arr)?,
            "df_h" => TensorRef::from_array_view(&df_h_arr)?,
        ])?;

        let (shape, data) = outputs["enhanced_spec"].try_extract_tensor::<f32>()?;
        if shape.iter().product::<i64>() != spec_expected as i64 {
            return Err(EmbeddingError::UnexpectedOutputShape {
                got: shape.to_vec(),
                expected_dim: spec_expected,
            });
        }
        let enhanced = data.to_vec();
        let (_, new_enc) = outputs["new_enc_h"].try_extract_tensor::<f32>()?;
        let (_, new_erb) = outputs["new_erb_h"].try_extract_tensor::<f32>()?;
        let (_, new_df) = outputs["new_df_h"].try_extract_tensor::<f32>()?;

        Ok(EnhancedFrame {
            spec: enhanced,
            new_enc_h: new_enc.to_vec(),
            new_erb_h: new_erb.to_vec(),
            new_df_h: new_df.to_vec(),
        })
    }
}

/// Linear interpolation of a two-value initialiser across `n` bins.
fn init_state_lerp(init: [f32; 2], n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![init[0]; n];
    }
    (0..n)
        .map(|i| init[0] + (init[1] - init[0]) * (i as f32) / ((n as f32) - 1.0))
        .collect()
}

fn zero_state(layers: usize) -> Vec<f32> {
    vec![0.0_f32; layers * GRU_HIDDEN]
}

/// End-to-end DFN3 pipeline: 48 kHz audio in → 48 kHz audio out.
///
/// Wraps:
///
/// * `deep_filter::DFState` for STFT / iSTFT and ERB filterbank widths
/// * the [`Dfn3`] ONNX wrapper for the per-frame neural step
/// * EMA state for `band_mean_norm_erb` (32 bins) and
///   `band_unit_norm` (`NB_DF` bins)
/// * the three GRU hidden states (carried across `push_samples` calls)
/// * a `CONV_LOOKAHEAD + 1`-deep feature queue: each frame waits in
///   the front of the spec queue until enough future feat frames
///   have arrived to pair with it.
///
/// See module docs for the streaming protocol and ONNX I/O contract.
pub struct Dfn3Pipeline {
    dfstate: DFState,
    net: Dfn3,
    erb_widths: Vec<usize>,
    erb_norm_state: Vec<f32>,
    df_norm_state: Vec<f32>,
    norm_alpha: f32,
    enc_h: Vec<f32>,
    erb_h: Vec<f32>,
    df_h: Vec<f32>,
    /// Audio samples accumulated since the last completed hop.
    audio_buffer: VecDeque<f32>,
    /// Per-frame spec / feat queues. All three have equal length;
    /// `push_one_frame` extends them, `drain_emit` pops the front
    /// once a model call consumes the oldest spec.
    spec_queue: VecDeque<Vec<f32>>,
    feat_erb_queue: VecDeque<Vec<f32>>,
    feat_spec_queue: VecDeque<Vec<f32>>,
}

impl Dfn3Pipeline {
    /// Build a pipeline that loads the stateful DFN3 ONNX from
    /// `path`.
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
            enc_h: zero_state(ENC_GRU_LAYERS),
            erb_h: zero_state(ERB_GRU_LAYERS),
            df_h: zero_state(DF_GRU_LAYERS),
            audio_buffer: VecDeque::with_capacity(DFN3_HOP * 4),
            spec_queue: VecDeque::with_capacity(CONV_LOOKAHEAD + 2),
            feat_erb_queue: VecDeque::with_capacity(CONV_LOOKAHEAD + 2),
            feat_spec_queue: VecDeque::with_capacity(CONV_LOOKAHEAD + 2),
        })
    }

    /// Reset every piece of state: STFT memory, EMA, GRU hidden
    /// states, audio + feature queues. Call between independent
    /// recordings.
    pub fn reset(&mut self) {
        self.dfstate.reset();
        self.erb_norm_state = init_state_lerp(MEAN_NORM_INIT, N_ERB);
        self.df_norm_state = init_state_lerp(UNIT_NORM_INIT, NB_DF);
        self.enc_h = zero_state(ENC_GRU_LAYERS);
        self.erb_h = zero_state(ERB_GRU_LAYERS);
        self.df_h = zero_state(DF_GRU_LAYERS);
        self.audio_buffer.clear();
        self.spec_queue.clear();
        self.feat_erb_queue.clear();
        self.feat_spec_queue.clear();
    }

    /// Sub-hop audio residue currently buffered. Useful as a latency
    /// breakdown probe; not part of any algorithmic contract.
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.audio_buffer.len() + self.spec_queue.len() * DFN3_HOP
    }

    /// Compute STFT features for one hop of audio + push them onto
    /// the per-frame queues. Doesn't trigger a model call by itself
    /// — `drain_emit` does that when the queues are full enough.
    fn push_one_frame(&mut self, audio_hop: &[f32]) {
        debug_assert_eq!(audio_hop.len(), DFN3_HOP);
        let mut spec_frame = vec![Complex32::new(0.0, 0.0); N_FREQ];
        let mut band_pow = vec![0.0_f32; N_ERB];

        self.dfstate.analysis(audio_hop, &mut spec_frame);

        // Pack the complex spec into a flat (N_FREQ, 2) buffer.
        let mut spec_flat = vec![0.0_f32; N_FREQ * 2];
        for (k, c) in spec_frame.iter().enumerate() {
            spec_flat[k * 2] = c.re;
            spec_flat[k * 2 + 1] = c.im;
        }

        // ERB feature: log-power per band, EMA-normalised.
        compute_band_corr(&mut band_pow, &spec_frame, &spec_frame, &self.erb_widths);
        for b in &mut band_pow {
            *b = 10.0 * (1e-10_f32 + *b).log10();
        }
        band_mean_norm_erb(&mut band_pow, &mut self.erb_norm_state, self.norm_alpha);

        // DF spec feature: low-freq band, unit-norm normalised.
        let mut df_spec_complex: Vec<Complex32> = spec_frame[..NB_DF].to_vec();
        band_unit_norm(
            &mut df_spec_complex,
            &mut self.df_norm_state,
            self.norm_alpha,
        );
        let mut df_spec_flat = vec![0.0_f32; NB_DF * 2];
        for (k, c) in df_spec_complex.iter().enumerate() {
            df_spec_flat[k * 2] = c.re;
            df_spec_flat[k * 2 + 1] = c.im;
        }

        self.spec_queue.push_back(spec_flat);
        self.feat_erb_queue.push_back(band_pow);
        self.feat_spec_queue.push_back(df_spec_flat);
    }

    /// Drain as many enhanced hops as the queues currently support.
    /// Each call consumes the oldest spec paired with feat at index
    /// `CONV_LOOKAHEAD` (the "future" feat the model needs).
    fn drain_emit(&mut self) -> Result<Vec<f32>, EmbeddingError> {
        let mut out = Vec::new();
        let mut out_frame = vec![0.0_f32; DFN3_HOP];
        let mut enh_frame_complex = vec![Complex32::new(0.0, 0.0); N_FREQ];
        while self.spec_queue.len() > CONV_LOOKAHEAD {
            let spec = self.spec_queue.pop_front().expect("non-empty");
            // feat at index CONV_LOOKAHEAD is the future feature
            // paired with the oldest spec.
            let feat_erb = self.feat_erb_queue[CONV_LOOKAHEAD].clone();
            let feat_spec = self.feat_spec_queue[CONV_LOOKAHEAD].clone();
            // pop the front feats too — they're never the "future
            // feat" for any spec we'll process later (their pairing
            // spec just left the queue).
            self.feat_erb_queue.pop_front();
            self.feat_spec_queue.pop_front();

            let enhanced = self.net.enhance_frame(
                &spec,
                &feat_erb,
                &feat_spec,
                &self.enc_h,
                &self.erb_h,
                &self.df_h,
            )?;
            self.enc_h = enhanced.new_enc_h;
            self.erb_h = enhanced.new_erb_h;
            self.df_h = enhanced.new_df_h;

            for (k, slot) in enh_frame_complex.iter_mut().enumerate() {
                *slot = Complex32::new(enhanced.spec[k * 2], enhanced.spec[k * 2 + 1]);
            }
            self.dfstate
                .synthesis(&mut enh_frame_complex, &mut out_frame);
            out.extend_from_slice(&out_frame);
        }
        Ok(out)
    }

    /// Push `audio` samples through the pipeline and return any
    /// enhanced samples that became available. The output is
    /// hop-aligned (480-sample multiples) and is delayed by
    /// `CONV_LOOKAHEAD` frames = 20 ms @ 48 kHz hop = 480.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn push_samples(&mut self, audio: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
        self.audio_buffer.extend(audio.iter().copied());
        let mut out: Vec<f32> = Vec::new();
        while self.audio_buffer.len() >= DFN3_HOP {
            let mut hop = Vec::with_capacity(DFN3_HOP);
            for _ in 0..DFN3_HOP {
                hop.push(self.audio_buffer.pop_front().unwrap_or(0.0));
            }
            self.push_one_frame(&hop);
            out.extend(self.drain_emit()?);
        }
        Ok(out)
    }

    /// Drain the lookahead pipeline: zero-pad any sub-hop residue
    /// to one full hop, push it as the last "real" frame, then
    /// inject `CONV_LOOKAHEAD` zero feat frames to let the queue
    /// emit the remaining specs.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn flush(&mut self) -> Result<Vec<f32>, EmbeddingError> {
        let mut out: Vec<f32> = Vec::new();
        // Pad any sub-hop residue to a full hop so it gets enhanced.
        if !self.audio_buffer.is_empty() {
            let mut hop = vec![0.0_f32; DFN3_HOP];
            for (i, s) in self.audio_buffer.drain(..).enumerate() {
                hop[i] = s;
            }
            self.push_one_frame(&hop);
            out.extend(self.drain_emit()?);
        }
        // Append CONV_LOOKAHEAD zero feature frames so the trailing
        // specs in the queue can be paired with "future feat = 0".
        let zero_spec = vec![0.0_f32; N_FREQ * 2];
        let zero_erb = vec![0.0_f32; N_ERB];
        let zero_df = vec![0.0_f32; NB_DF * 2];
        for _ in 0..CONV_LOOKAHEAD {
            self.spec_queue.push_back(zero_spec.clone());
            self.feat_erb_queue.push_back(zero_erb.clone());
            self.feat_spec_queue.push_back(zero_df.clone());
            out.extend(self.drain_emit()?);
        }
        Ok(out)
    }

    /// One-shot offline processing: reset state, push all samples,
    /// flush, return a buffer of exactly `audio.len()` samples
    /// (truncated or zero-padded as needed).
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
        self.reset();
        let mut out = self.push_samples(audio)?;
        out.extend(self.flush()?);
        // The streaming flow may emit slightly more than `audio.len()`
        // (last partial hop zero-padded up to DFN3_HOP). Truncate so
        // callers see a same-length output.
        out.truncate(audio.len());
        if out.len() < audio.len() {
            out.resize(audio.len(), 0.0);
        }
        Ok(out)
    }
}

/// Streaming wrapper around [`Dfn3Pipeline`] for live use.
///
/// Thin alias over the pipeline's own `push_samples` / `flush` /
/// `reset` methods. Kept as a separate type so `mellonella-audio-io`
/// and downstream callers don't have to change imports across the
/// step-13 → step-13.5 cutover.
pub struct Dfn3Streamer {
    pipeline: Dfn3Pipeline,
}

impl Dfn3Streamer {
    /// Build a streamer that loads the stateful DFN3 ONNX from
    /// `path`.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from
    /// [`Dfn3Pipeline::from_onnx_path`].
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        Ok(Self {
            pipeline: Dfn3Pipeline::from_onnx_path(path)?,
        })
    }

    /// Push samples and get any enhanced samples that became
    /// available. Output is hop-aligned and delayed by
    /// `CONV_LOOKAHEAD` frames (~20 ms @ 48 kHz hop).
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn push_samples(&mut self, audio: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
        self.pipeline.push_samples(audio)
    }

    /// Drain the lookahead pipeline at end-of-stream.
    ///
    /// # Errors
    /// Forwards [`EmbeddingError`] from the underlying ONNX call.
    pub fn flush(&mut self) -> Result<Vec<f32>, EmbeddingError> {
        self.pipeline.flush()
    }

    /// Reset state between independent recordings.
    pub fn reset(&mut self) {
        self.pipeline.reset();
    }

    /// Sub-hop + in-flight queue depth (samples). Useful for future
    /// GUI latency display.
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.pipeline.buffered_samples()
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
    fn streamer_invariants_dont_need_onnx() {
        // Lock the per-frame I/O contract so a future export change
        // can't silently break call sites.
        assert_eq!(ENC_GRU_LAYERS * GRU_HIDDEN, 256);
        assert_eq!(ERB_GRU_LAYERS * GRU_HIDDEN, 512);
        assert_eq!(DF_GRU_LAYERS * GRU_HIDDEN, 512);
        assert_eq!(N_FREQ, DFN3_FFT / 2 + 1);
        assert_eq!(DFN3_HOP, 480);
        assert_eq!(SAMPLES_PER_CHUNK, FRAMES_PER_CHUNK * DFN3_HOP);
        // The wrapper's algorithmic delay is CONV_LOOKAHEAD frames.
        let delay_ms = (CONV_LOOKAHEAD * DFN3_HOP) as f32 / (DFN3_SR as f32) * 1000.0;
        assert!(
            (delay_ms - 20.0).abs() < 0.5,
            "expected ~20 ms lookahead delay, got {delay_ms}"
        );
    }

    #[test]
    fn rejects_mis_shaped_inputs() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut model = Dfn3::from_onnx_path(&path).expect("load DFN3 ONNX");
        let res = model.enhance_frame(
            &[0.0_f32; 10],
            &[0.0_f32; N_ERB],
            &[0.0_f32; NB_DF * 2],
            &[0.0_f32; ENC_GRU_LAYERS * GRU_HIDDEN],
            &[0.0_f32; ERB_GRU_LAYERS * GRU_HIDDEN],
            &[0.0_f32; DF_GRU_LAYERS * GRU_HIDDEN],
        );
        assert!(res.is_err());
    }

    #[test]
    fn pipeline_push_then_flush_emits_expected_length() {
        let Some(path) = skip_unless_onnx_available() else {
            return;
        };
        let mut pipeline = Dfn3Pipeline::from_onnx_path(&path).expect("load DFN3 ONNX");
        // 3 hops of silence; output should round-trip to the same
        // length once flushed.
        let audio = vec![0.0_f32; DFN3_HOP * 3];
        let out = pipeline.push_samples(&audio).expect("push");
        // After 3 frames pushed and CONV_LOOKAHEAD=2 lookahead, at most
        // 1 frame has emerged.
        assert!(out.len() <= DFN3_HOP, "push should emit ≤ 1 hop pre-flush");
        let tail = pipeline.flush().expect("flush");
        // Total output spans the pushed audio plus the flush-padded
        // lookahead frames.
        let total = out.len() + tail.len();
        assert!(total >= audio.len(), "total output must cover the input");
    }
}
