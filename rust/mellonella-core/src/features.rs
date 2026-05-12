//! SpeechBrain Fbank reproduction in Rust.
//!
//! Mirrors `speechbrain.lobes.features.Fbank` for the
//! ECAPA-TDNN preset: STFT with Hamming(400), n_fft=400, hop=160,
//! `center=True`, zero-padded; magnitude squared; multiply by the
//! triangular mel filterbank vendored as
//! ``tests/fixtures/fbank_filterbank.bin``; convert to dB
//! (`10·log10(max(amin, x))`) with a per-batch `top_db = 80` floor.
//!
//! The filterbank matrix is loaded at construction time from a
//! caller-supplied buffer so it can be vendored as a binary fixture
//! (`(201, 80)` row-major f32). The intent is to track upstream
//! SpeechBrain bit-equivalently rather than re-implement the mel
//! formula and risk subtle differences (Slaney vs HTK normalisation,
//! float ordering, …).
//!
//! Empirical parity vs. the Python reference (1 s @ 16 kHz, 180 Hz
//! harmonic stack, ``scripts/dump_fbank_fixture.py``):
//!
//! | metric            | value      | tolerance |
//! |-------------------|------------|-----------|
//! | per-bin `max\|Δ\|` | ~5 × 10⁻⁵  | 1 × 10⁻³ ✅ |

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

/// Number of mel bands (matches SpeechBrain's ECAPA preset).
pub const N_MELS: usize = 80;
/// FFT size (matches SpeechBrain's ECAPA preset).
pub const N_FFT: usize = 400;
/// Hop length in samples (matches SpeechBrain's ECAPA preset).
pub const HOP_LENGTH: usize = 160;
/// Number of one-sided STFT bins (`n_fft / 2 + 1`).
pub const N_STFT: usize = N_FFT / 2 + 1;
/// Power-spectrum exponent (squared magnitude).
const POWER: i32 = 2;
/// Lower bound for log-mel input.
const AMIN: f32 = 1e-10;
/// `multiplier * log10(max(amin, x))` — SpeechBrain uses 10.
const MULTIPLIER: f32 = 10.0;
/// Per-batch dB floor relative to the max.
const TOP_DB: f32 = 80.0;

/// Errors returned by [`Fbank`].
#[derive(Debug)]
pub enum FbankError {
    /// Filterbank buffer length did not match `(N_STFT, N_MELS)`.
    BadFilterbankLength { got: usize, expected: usize },
}

impl std::fmt::Display for FbankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadFilterbankLength { got, expected } => {
                write!(f, "filterbank buffer is {got} f32s, expected {expected}")
            }
        }
    }
}

impl std::error::Error for FbankError {}

/// SpeechBrain-compatible log-mel feature extractor.
///
/// Holds the FFT plan and the vendored filterbank matrix. Re-using one
/// instance across many calls amortises the FFT planning cost.
pub struct Fbank {
    fft: Arc<dyn Fft<f32>>,
    fft_scratch: Vec<Complex32>,
    window: [f32; N_FFT],
    /// `(N_STFT, N_MELS)` row-major triangular filter weights.
    filterbank: Box<[f32]>,
}

/// SpeechBrain ECAPA-TDNN filterbank matrix, vendored from
/// `scripts/dump_fbank_fixture.py`. Use [`Fbank::with_speechbrain_filterbank`]
/// to construct an `Fbank` from this buffer at runtime.
const SPEECHBRAIN_FILTERBANK_BYTES: &[u8] =
    include_bytes!("../tests/fixtures/fbank_filterbank.bin");

impl Fbank {
    /// Convenience constructor that uses the vendored SpeechBrain
    /// filterbank matrix from `tests/fixtures/fbank_filterbank.bin`.
    ///
    /// # Errors
    /// Returns [`FbankError::BadFilterbankLength`] if the vendored
    /// buffer ever gets out of sync with `N_STFT * N_MELS`.
    pub fn with_speechbrain_filterbank() -> Result<Self, FbankError> {
        let n = SPEECHBRAIN_FILTERBANK_BYTES.len() / 4;
        let mut filterbank = Vec::with_capacity(n);
        for chunk in SPEECHBRAIN_FILTERBANK_BYTES.chunks_exact(4) {
            filterbank.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Self::new(&filterbank)
    }

    /// Build an Fbank using a vendored filterbank matrix.
    ///
    /// `filterbank` must be a `(N_STFT, N_MELS)` row-major buffer
    /// (i.e. `filterbank[stft_bin * N_MELS + mel_bin]`) — the layout
    /// produced by `scripts/dump_fbank_fixture.py`.
    ///
    /// # Errors
    /// Returns [`FbankError::BadFilterbankLength`] when the buffer is
    /// not exactly `N_STFT * N_MELS` floats.
    pub fn new(filterbank: &[f32]) -> Result<Self, FbankError> {
        let expected = N_STFT * N_MELS;
        if filterbank.len() != expected {
            return Err(FbankError::BadFilterbankLength {
                got: filterbank.len(),
                expected,
            });
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let fft_scratch = vec![Complex32::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        // SpeechBrain uses torch.hamming_window(n_fft) which defaults
        // to periodic=True — divisor N (not N-1 as in the symmetric
        // form). Easy to confuse: numpy/scipy default to symmetric.
        let mut window = [0.0_f32; N_FFT];
        let denom = N_FFT as f32;
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos();
        }

        Ok(Self {
            fft,
            fft_scratch,
            window,
            filterbank: filterbank.to_vec().into_boxed_slice(),
        })
    }

    /// Compute the log-mel features for `audio`.
    ///
    /// Returns a `(n_frames × N_MELS)` flat row-major vector. `n_frames`
    /// follows torch's `center=True` rule:
    /// `1 + audio.len() / HOP_LENGTH`.
    #[must_use]
    pub fn compute(&mut self, audio: &[f32]) -> Vec<f32> {
        // center=True padding: prepend / append n_fft/2 zeros (constant
        // pad mode, which is just zero-padding).
        let pad = N_FFT / 2;
        let n_frames = 1 + audio.len() / HOP_LENGTH;
        let mut out = vec![0.0_f32; n_frames * N_MELS];

        // Reusable per-frame buffers.
        let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut power = [0.0_f32; N_STFT];

        for f_idx in 0..n_frames {
            // Frame start in the *padded* signal.
            let start_padded = f_idx * HOP_LENGTH;
            for (i, slot) in frame.iter_mut().enumerate().take(N_FFT) {
                let abs_idx = start_padded + i;
                let sample = if abs_idx < pad || abs_idx >= pad + audio.len() {
                    0.0
                } else {
                    audio[abs_idx - pad]
                };
                slot.re = sample * self.window[i];
                slot.im = 0.0;
            }
            self.fft
                .process_with_scratch(&mut frame, &mut self.fft_scratch);

            // Power = |X|^2 (POWER == 2). Onesided: keep first N_STFT bins.
            for (b, slot) in power.iter_mut().enumerate().take(N_STFT) {
                let re = frame[b].re;
                let im = frame[b].im;
                let mag2 = re * re + im * im;
                *slot = if POWER == 2 { mag2 } else { mag2.sqrt() };
            }

            // Mel projection: out[f, m] = sum_b power[b] * filterbank[b * N_MELS + m]
            let row = &mut out[f_idx * N_MELS..(f_idx + 1) * N_MELS];
            row.fill(0.0);
            for (b, &p) in power.iter().enumerate() {
                let fb_row = &self.filterbank[b * N_MELS..(b + 1) * N_MELS];
                for (m, &w) in fb_row.iter().enumerate() {
                    row[m] += p * w;
                }
            }
        }

        // log_mel: 10 * log10(max(amin, x)), then floor at max - top_db.
        let mut max_db = f32::NEG_INFINITY;
        for v in &mut out {
            *v = MULTIPLIER * v.max(AMIN).log10();
            if *v > max_db {
                max_db = *v;
            }
        }
        let floor = max_db - TOP_DB;
        for v in &mut out {
            if *v < floor {
                *v = floor;
            }
        }

        out
    }
}
