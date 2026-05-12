//! YIN pitch estimator ported from `poc/mellonella_poc/f0.py`.
//!
//! Reference: De Cheveigné & Kawahara, *YIN, a fundamental frequency
//! estimator for speech and music*, JASA 2002.
//!
//! Used as the lightweight F0 path. CREPE is reserved for the optional
//! high-precision branch (see `docs/architecture.md` Stage 4).

// YIN is a numeric kernel: τ indices and audio sample counts cross between
// `usize`, `f32`, and `f64` constantly. The clippy::pedantic warnings about
// that are real in general but inert here because every cast is bounded:
// frame sizes ≤ 2^16, sample rates ≤ 2^20, τ ≤ frame_size / 2.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// Default voiced-pitch lower bound (Hz).
pub const DEFAULT_F_MIN: f32 = 50.0;
/// Default voiced-pitch upper bound (Hz).
pub const DEFAULT_F_MAX: f32 = 500.0;
/// Default YIN aperiodicity threshold.
pub const DEFAULT_THRESHOLD: f64 = 0.1;

/// Squared-difference function `d(τ)` over `frame`, computed up to (but
/// not including) `max_tau`. Returns a `f64` buffer to keep the
/// cumulative-mean stage numerically stable on long frames.
fn difference(frame: &[f32], max_tau: usize) -> Vec<f64> {
    let mut diff = vec![0.0_f64; max_tau];
    let n = frame.len();
    for tau in 1..max_tau {
        let mut s = 0.0_f64;
        let upper = n - tau;
        for i in 0..upper {
            let d = f64::from(frame[i]) - f64::from(frame[i + tau]);
            s += d * d;
        }
        diff[tau] = s;
    }
    diff
}

/// Cumulative mean normalised difference `d'(τ)`. `diff[0]` is replaced
/// by 1.0 to match the Python reference and keep the threshold check at
/// τ=0 consistent.
fn cumulative_mean_normalized(diff: &[f64]) -> Vec<f64> {
    let mut cmnd = vec![0.0_f64; diff.len()];
    if diff.is_empty() {
        return cmnd;
    }
    cmnd[0] = 1.0;
    let mut running = 0.0_f64;
    for tau in 1..diff.len() {
        running += diff[tau];
        cmnd[tau] = if running > 0.0 {
            diff[tau] * tau as f64 / running
        } else {
            1.0
        };
    }
    cmnd
}

/// Estimate F0 for a single frame. Returns `None` for unvoiced or
/// unstable frames. Mirrors `mellonella_poc.f0.yin_frame`.
#[must_use]
pub fn yin_frame(
    frame: &[f32],
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
    threshold: f64,
) -> Option<f32> {
    yin_frame_with_hint(frame, sample_rate, f_min, f_max, threshold, None)
}

/// Variant of [`yin_frame`] that narrows the τ search window around a
/// prior estimate. When `prev_hz` is `Some` and lands in `[f_min, f_max]`,
/// the inner squared-difference loop visits only `[τ_hint × 0.7, τ_hint × 1.4]`
/// — empirically ~5–8× faster on continuous-pitch tracks. The unhinted
/// path falls back to the standard `[τ_min, τ_max]` range and is what
/// [`yin_frame`] keeps using.
///
/// If the hint search fails (no dip below `threshold` in the narrow
/// window), the function widens to the full range so we don't get
/// stuck on a stale prior.
#[must_use]
pub fn yin_frame_with_hint(
    frame: &[f32],
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
    threshold: f64,
    prev_hz: Option<f32>,
) -> Option<f32> {
    if frame.len() < 64 {
        return None;
    }
    let sr_f = f64::from(sample_rate);
    let tau_min_full = ((sr_f / f64::from(f_max)) as usize).max(2);
    let tau_max_candidate = (sr_f / f64::from(f_min)) as usize;
    let tau_max_full = (frame.len() / 2).min(tau_max_candidate);
    if tau_max_full <= tau_min_full {
        return None;
    }

    let hint_range = prev_hz.and_then(|hz| {
        if !hz.is_finite() || hz <= 0.0 {
            return None;
        }
        let tau_hint = (sr_f / f64::from(hz)) as usize;
        if tau_hint < tau_min_full || tau_hint > tau_max_full {
            return None;
        }
        let lo = tau_min_full.max((tau_hint as f64 * 0.7) as usize);
        let hi = tau_max_full.min((tau_hint as f64 * 1.4) as usize);
        if hi > lo + 1 {
            Some((lo, hi))
        } else {
            None
        }
    });

    // Hot path: when the hint window is narrow we only compute
    // `difference` up to `hi`, which is what dominates the per-frame
    // cost (O(frame_len × τ_max) inner loop). The cumulative mean
    // term needs all τ from 1..=hi, so we have to keep the prefix —
    // just not run all the way to τ_max_full.
    let (diff, cmnd, scan_max) = if let Some((_lo, hi)) = hint_range {
        let d = difference(frame, hi + 1);
        let c = cumulative_mean_normalized(&d);
        (d, c, hi)
    } else {
        let d = difference(frame, tau_max_full + 1);
        let c = cumulative_mean_normalized(&d);
        (d, c, tau_max_full)
    };

    if let Some((lo, hi)) = hint_range {
        if let Some(found_tau) = find_tau_dip(&cmnd, lo, hi, threshold) {
            let _ = diff;
            return parabolic_refine(&cmnd, found_tau, scan_max, sr_f);
        }
        // Hint window missed — widen to the full τ range. Requires
        // recomputing diff over the wider span.
        let diff_full = difference(frame, tau_max_full + 1);
        let cmnd_full = cumulative_mean_normalized(&diff_full);
        let found_tau = find_tau_dip(&cmnd_full, tau_min_full, tau_max_full, threshold)?;
        return parabolic_refine(&cmnd_full, found_tau, tau_max_full, sr_f);
    }
    let _ = diff;
    let found_tau = find_tau_dip(&cmnd, tau_min_full, tau_max_full, threshold)?;
    parabolic_refine(&cmnd, found_tau, scan_max, sr_f)
}

/// Scan `cmnd[tau_min..tau_max]` for the first dip below `threshold`,
/// then descend to the local minimum. Returns `None` if no τ in the
/// range crosses the threshold.
fn find_tau_dip(cmnd: &[f64], tau_min: usize, tau_max: usize, threshold: f64) -> Option<usize> {
    let mut tau = tau_min;
    while tau < tau_max {
        if cmnd[tau] < threshold {
            while tau + 1 < tau_max && cmnd[tau + 1] < cmnd[tau] {
                tau += 1;
            }
            return Some(tau);
        }
        tau += 1;
    }
    None
}

/// Apply parabolic interpolation around `found_tau` against the
/// surrounding `cmnd` samples and convert to Hz.
fn parabolic_refine(cmnd: &[f64], found_tau: usize, tau_max: usize, sr_f: f64) -> Option<f32> {
    let tau_f: f64 = if found_tau >= 1 && found_tau + 1 < tau_max {
        let s0 = cmnd[found_tau - 1];
        let s1 = cmnd[found_tau];
        let s2 = cmnd[found_tau + 1];
        let denom = s0 + s2 - 2.0 * s1;
        if denom == 0.0 {
            found_tau as f64
        } else {
            found_tau as f64 + 0.5 * (s0 - s2) / denom
        }
    } else {
        found_tau as f64
    };
    if tau_f <= 0.0 {
        return None;
    }
    Some((sr_f / tau_f) as f32)
}

/// Compute a F0 track over `audio`. Unvoiced frames are stored as
/// `f32::NAN`, matching the Python implementation; use [`f0_statistics`]
/// to summarise.
#[must_use]
pub fn estimate_f0_track(
    audio: &[f32],
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    if audio.len() < frame_size || hop_size == 0 {
        return Vec::new();
    }
    let n_frames = 1 + (audio.len() - frame_size) / hop_size;
    let mut track = vec![f32::NAN; n_frames];
    let mut last_hz: Option<f32> = None;
    for (i, slot) in track.iter_mut().enumerate() {
        let start = i * hop_size;
        let frame = &audio[start..start + frame_size];
        let est = yin_frame_with_hint(frame, sample_rate, f_min, f_max, DEFAULT_THRESHOLD, last_hz);
        if let Some(hz) = est {
            *slot = hz;
            last_hz = Some(hz);
        }
        // Unvoiced frames don't update `last_hz` — preserves the prior
        // for the next voiced frame.
    }
    track
}

/// Mean and population standard deviation of voiced frames in `track`.
/// NaN-safe — `(0.0, 0.0)` when no voiced frames are present.
#[must_use]
pub fn f0_statistics(track: &[f32]) -> (f32, f32) {
    let voiced: Vec<f32> = track.iter().copied().filter(|v| v.is_finite()).collect();
    if voiced.is_empty() {
        return (0.0, 0.0);
    }
    let n = voiced.len() as f32;
    let sum: f32 = voiced.iter().copied().sum();
    let mean = sum / n;
    let var: f32 = voiced.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
        let sr = sample_rate as f32;
        let n = (sr * duration_sec) as usize;
        (0..n).map(|i| (TAU * freq * i as f32 / sr).sin()).collect()
    }

    #[test]
    fn yin_estimates_pure_tone_120() {
        let sr = 16_000;
        let frame = &sine(120.0, sr, 0.2)[..2048];
        let est = yin_frame(frame, sr, DEFAULT_F_MIN, DEFAULT_F_MAX, DEFAULT_THRESHOLD).unwrap();
        assert!((est - 120.0).abs() / 120.0 < 0.05, "est={est}");
    }

    #[test]
    fn yin_estimates_pure_tone_220() {
        let sr = 16_000;
        let frame = &sine(220.0, sr, 0.2)[..2048];
        let est = yin_frame(frame, sr, DEFAULT_F_MIN, DEFAULT_F_MAX, DEFAULT_THRESHOLD).unwrap();
        assert!((est - 220.0).abs() / 220.0 < 0.05, "est={est}");
    }

    #[test]
    fn yin_estimates_pure_tone_330() {
        let sr = 16_000;
        let frame = &sine(330.0, sr, 0.2)[..2048];
        let est = yin_frame(frame, sr, DEFAULT_F_MIN, DEFAULT_F_MAX, DEFAULT_THRESHOLD).unwrap();
        assert!((est - 330.0).abs() / 330.0 < 0.05, "est={est}");
    }

    #[test]
    fn yin_returns_none_or_low_for_white_noise() {
        // Deterministic LCG to avoid pulling rand as a dep.
        let sr = 16_000;
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let frame: Vec<f32> = (0..2048)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as i32 as f32) / i32::MAX as f32
            })
            .collect();
        let est = yin_frame(&frame, sr, DEFAULT_F_MIN, DEFAULT_F_MAX, 0.05);
        // Must either return None or a value below 1 kHz (mirrors the Python test).
        assert!(est.is_none() || est.unwrap() < 1000.0);
    }

    #[test]
    fn track_recovers_constant_pitch() {
        let sr = 16_000;
        let audio = sine(150.0, sr, 0.5);
        let track = estimate_f0_track(&audio, sr, 2048, 512, DEFAULT_F_MIN, DEFAULT_F_MAX);
        let mut voiced: Vec<f32> = track.into_iter().filter(|v| v.is_finite()).collect();
        assert!(!voiced.is_empty());
        voiced.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = voiced[voiced.len() / 2];
        assert!((median - 150.0).abs() / 150.0 < 0.05, "median={median}");
    }

    #[test]
    fn statistics_handles_empty_track() {
        let track = vec![f32::NAN; 10];
        let (mu, sigma) = f0_statistics(&track);
        assert!(mu.abs() <= 0.0);
        assert!(sigma.abs() <= 0.0);
    }
}
