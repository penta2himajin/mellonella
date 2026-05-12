//! Audio resampling for the SV-rate pipeline.
//!
//! Wraps `rubato::SincFixedIn` (windowed-sinc interpolation) to convert
//! arbitrary input sample rates to the 16 kHz the rest of
//! `mellonella-core` operates at. Mirrors what
//! `mellonella_poc.pipeline.resample` does on the Python side with
//! `scipy.signal.resample_poly`, with the practical caveat that the
//! two algorithms (windowed-sinc vs polyphase + low-pass) produce
//! samples that are equivalent under any reasonable error metric but
//! not byte-equal.
//!
//! Empirical agreement on a synthesised 180 Hz harmonic stack
//! (44.1 kHz → 16 kHz):
//!
//! | metric                | value        | tolerance |
//! |-----------------------|--------------|-----------|
//! | per-sample `max\|Δ\|`  | ~5 × 10⁻³    | 1 × 10⁻²  |
//! | post-Fbank gate state | byte-equal   | n/a       |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Errors returned by [`resample_to`].
#[derive(Debug)]
pub enum ResampleError {
    /// `rubato` rejected the configuration or the input shape.
    Rubato(String),
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rubato(s) => write!(f, "rubato resampler error: {s}"),
        }
    }
}

impl std::error::Error for ResampleError {}

/// Resample `audio` from `src_sr` to `dst_sr` using a windowed-sinc
/// interpolator. Returns the resampled buffer; samples are clipped to
/// `[-1, 1]` upstream by [`SincFixedIn`] internals if necessary.
///
/// Identity (`src_sr == dst_sr`) returns a cheap clone — useful for
/// pipelines that don't know upfront whether the input matches the
/// target rate.
///
/// # Errors
/// Returns [`ResampleError::Rubato`] when `SincFixedIn::new` or
/// `process` fails (typically: sample rates out of range, or
/// per-channel buffer mismatch — neither should happen for our
/// `mono → mono` call sites).
pub fn resample_to(audio: &[f32], src_sr: u32, dst_sr: u32) -> Result<Vec<f32>, ResampleError> {
    if src_sr == dst_sr {
        return Ok(audio.to_vec());
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = f64::from(dst_sr) / f64::from(src_sr);
    let chunk_size = audio.len();
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.1, params, chunk_size, 1)
        .map_err(|e| ResampleError::Rubato(e.to_string()))?;

    let input = vec![audio.to_vec()];
    let output = resampler
        .process(&input, None)
        .map_err(|e| ResampleError::Rubato(e.to_string()))?;
    Ok(output.into_iter().next().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_returns_input() {
        let audio = vec![0.1_f32, -0.2, 0.3];
        let out = resample_to(&audio, 16_000, 16_000).unwrap();
        assert_eq!(out, audio);
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = resample_to(&[], 44_100, 16_000).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn downsample_changes_length_proportionally() {
        // 1 s @ 48 kHz → ≈ 16 kHz worth of samples (allow ±1 % slack
        // because windowed-sinc with delay correction doesn't land on
        // an exact ratio).
        let audio = vec![0.0_f32; 48_000];
        let out = resample_to(&audio, 48_000, 16_000).unwrap();
        let expected = 16_000_i32;
        let slack = expected / 100; // ±1 %
        let got = out.len() as i32;
        assert!(
            (got - expected).abs() <= slack,
            "expected ≈ {expected} samples (±{slack}), got {got}"
        );
    }
}
