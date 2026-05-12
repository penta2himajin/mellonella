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

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// Default voiced-pitch lower bound (Hz).
pub const DEFAULT_F_MIN: f32 = 50.0;
/// Default voiced-pitch upper bound (Hz).
pub const DEFAULT_F_MAX: f32 = 500.0;
/// Default YIN aperiodicity threshold.
pub const DEFAULT_THRESHOLD: f64 = 0.1;

/// Squared-difference function `d(τ)` over `frame`, computed up to (but
/// not including) `max_tau`. Returns a `f64` buffer to keep the
/// cumulative-mean stage numerically stable on long frames.
///
/// Uses the Wiener-Khinchin identity:
///
/// ```text
/// d(τ) = sum (x[i] - x[i+τ])²
///      = sum x[i]² + sum x[i+τ]² − 2 · autocorr(τ)
///      = energy(0, N−τ) + energy(τ, N) − 2 · IFFT(|FFT(x)|²)[τ]
/// ```
///
/// The FFT is real-input, zero-padded to the next power of two of
/// 2·N to make the autocorrelation linear (not circular). For
/// `N = 2048`, that's a single 4096-point real-FFT plus the matching
/// inverse — ~50 k flops total versus the naïve O(N·τ_max) ≈ 655 k
/// flops, an ~13× theoretical win at the full τ range.
///
/// `cache` reuses one `RealFftPlanner` allocation across calls; the
/// time-domain scratch and prefix-sum buffers are also pooled.
fn difference(frame: &[f32], max_tau: usize, cache: &mut FftCache) -> Vec<f64> {
    let n = frame.len();
    let mut diff = vec![0.0_f64; max_tau];
    if n == 0 || max_tau == 0 {
        return diff;
    }

    // FFT length: next power of two ≥ 2·N for linear autocorr.
    let fft_len = (2 * n).next_power_of_two();
    let (forward, inverse) = cache.plans(fft_len);

    let scratch = &mut cache.real_scratch;
    scratch.clear();
    scratch.resize(fft_len, 0.0);
    scratch[..n].copy_from_slice(frame);

    let spectrum = &mut cache.complex_scratch;
    spectrum.clear();
    spectrum.resize(fft_len / 2 + 1, Complex32::new(0.0, 0.0));

    let fwd_scratch = &mut cache.fwd_scratch;
    fwd_scratch.resize(forward.get_scratch_len(), Complex32::new(0.0, 0.0));
    let _ = forward.process_with_scratch(scratch, spectrum, fwd_scratch);

    // |X|² in-place (imag set to 0 so IFFT round-trips into real).
    for c in spectrum.iter_mut() {
        let p = c.re * c.re + c.im * c.im;
        c.re = p;
        c.im = 0.0;
    }

    let autocorr = &mut cache.autocorr_scratch;
    autocorr.clear();
    autocorr.resize(fft_len, 0.0);

    let inv_scratch = &mut cache.inv_scratch;
    inv_scratch.resize(inverse.get_scratch_len(), Complex32::new(0.0, 0.0));
    let _ = inverse.process_with_scratch(spectrum, autocorr, inv_scratch);

    // realfft inverse leaves the result unnormalised — divide by fft_len.
    let inv_norm = 1.0_f64 / f64::from(u32::try_from(fft_len).unwrap_or(u32::MAX));

    // Prefix sum of squared frame samples for energy(s, e) lookups.
    let energy = &mut cache.energy_scratch;
    energy.clear();
    energy.resize(n + 1, 0.0);
    for i in 0..n {
        energy[i + 1] = energy[i] + f64::from(frame[i]) * f64::from(frame[i]);
    }
    let total_energy = energy[n];

    let upper = max_tau.min(n);
    for tau in 1..upper {
        let left = energy[n - tau];
        let right = total_energy - energy[tau];
        let auto = f64::from(autocorr[tau]) * inv_norm;
        diff[tau] = (left + right - 2.0 * auto).max(0.0);
    }
    diff
}

/// FFT plan + scratch reuse across `yin_frame` calls in a track.
///
/// Building a `RealFftPlanner` is non-trivial (~ms) so we want one
/// per F0 call site; the time-domain / complex / energy scratch
/// vectors come from the same struct to skip allocation in the hot
/// path.
#[derive(Default)]
pub struct FftCache {
    planner: Option<RealFftPlanner<f32>>,
    forward: Option<(usize, Arc<dyn RealToComplex<f32>>)>,
    inverse: Option<(usize, Arc<dyn ComplexToReal<f32>>)>,
    real_scratch: Vec<f32>,
    complex_scratch: Vec<Complex32>,
    autocorr_scratch: Vec<f32>,
    fwd_scratch: Vec<Complex32>,
    inv_scratch: Vec<Complex32>,
    energy_scratch: Vec<f64>,
}

impl FftCache {
    /// Construct a fresh cache. The first `difference` call populates
    /// the planner + plans lazily for whatever FFT length it needs.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn plans(
        &mut self,
        fft_len: usize,
    ) -> (Arc<dyn RealToComplex<f32>>, Arc<dyn ComplexToReal<f32>>) {
        let planner = self.planner.get_or_insert_with(RealFftPlanner::<f32>::new);
        let forward = match &self.forward {
            Some((len, plan)) if *len == fft_len => Arc::clone(plan),
            _ => {
                let plan = planner.plan_fft_forward(fft_len);
                self.forward = Some((fft_len, Arc::clone(&plan)));
                plan
            }
        };
        let inverse = match &self.inverse {
            Some((len, plan)) if *len == fft_len => Arc::clone(plan),
            _ => {
                let plan = planner.plan_fft_inverse(fft_len);
                self.inverse = Some((fft_len, Arc::clone(&plan)));
                plan
            }
        };
        (forward, inverse)
    }
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
///
/// One-shot convenience entry. Builds a fresh [`FftCache`] each call,
/// so the per-call FFT-planner setup overhead is paid every time;
/// the per-track [`estimate_f0_track`] path amortises this across
/// hops.
#[must_use]
pub fn yin_frame(
    frame: &[f32],
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
    threshold: f64,
) -> Option<f32> {
    let mut cache = FftCache::new();
    yin_frame_with_cache(
        frame,
        sample_rate,
        f_min,
        f_max,
        threshold,
        None,
        &mut cache,
    )
}

/// Variant of [`yin_frame`] that narrows the τ search window around a
/// prior estimate. Kept as a convenience for one-off callers that
/// don't already hold an [`FftCache`].
#[must_use]
pub fn yin_frame_with_hint(
    frame: &[f32],
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
    threshold: f64,
    prev_hz: Option<f32>,
) -> Option<f32> {
    let mut cache = FftCache::new();
    yin_frame_with_cache(
        frame,
        sample_rate,
        f_min,
        f_max,
        threshold,
        prev_hz,
        &mut cache,
    )
}

/// Core YIN entry — same semantics as [`yin_frame_with_hint`] but
/// takes an external [`FftCache`] so the planner + scratch buffers
/// can be reused across many calls (used by [`estimate_f0_track`]).
///
/// When `prev_hz` is `Some` and lands in `[f_min, f_max]`, the inner
/// difference computation covers only `[τ_hint × 0.7, τ_hint × 1.4]`
/// — empirically ~5–8× faster on continuous-pitch tracks. If the
/// hint window misses (no dip below `threshold`), the function
/// widens to the full range so we don't get stuck on a stale prior.
#[must_use]
pub fn yin_frame_with_cache(
    frame: &[f32],
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
    threshold: f64,
    prev_hz: Option<f32>,
    cache: &mut FftCache,
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

    // With the FFT-based `difference` the cost is dominated by one
    // 2·N FFT pair (~50 k flops at N=2048) regardless of `max_tau`.
    // The hint window still narrows the *scan* range, but no longer
    // bounds the FFT cost; just running the full FFT once per frame
    // wins out over re-running for the fallback case.
    let diff = difference(frame, tau_max_full + 1, cache);
    let cmnd = cumulative_mean_normalized(&diff);

    if let Some((lo, hi)) = hint_range {
        if let Some(found_tau) = find_tau_dip(&cmnd, lo, hi, threshold) {
            return parabolic_refine(&cmnd, found_tau, tau_max_full, sr_f);
        }
    }
    let found_tau = find_tau_dip(&cmnd, tau_min_full, tau_max_full, threshold)?;
    parabolic_refine(&cmnd, found_tau, tau_max_full, sr_f)
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
    // One cache across every hop so the FFT planner + scratch buffers
    // are allocated exactly once.
    let mut cache = FftCache::new();
    for (i, slot) in track.iter_mut().enumerate() {
        let start = i * hop_size;
        let frame = &audio[start..start + frame_size];
        let est = yin_frame_with_cache(
            frame,
            sample_rate,
            f_min,
            f_max,
            DEFAULT_THRESHOLD,
            last_hz,
            &mut cache,
        );
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
