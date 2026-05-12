//! Gating logic ported from `poc/mellonella_poc/gating.py`.
//!
//! Pure-math primitives operating on already-computed embeddings:
//!
//! * [`cos_similarity`] / [`cos_sim_max`] — cosine similarity helpers
//! * [`f0_match`]                          — Gaussian F0 match
//! * [`as_norm_score`]                     — Adaptive S-Norm z-score
//! * [`target_score_as_norm`]              — pool-vs-cohort integrated score
//!
//! Stateful streaming pieces:
//!
//! * [`GateConfig`]      — thresholds & timing constants
//! * [`GateState`]       — binary gate with hangover
//! * [`EnvelopeState`]   — attack/release follower
//! * [`apply_envelope`]  — gain ramp applied to a buffer

// `usize → f32` for top-K mean / variance and sample-rate / time math:
// cohort top-K is bounded by literature 10–50 and sample rates fit
// comfortably in f32's 2^23 mantissa, so the loss clippy warns about
// cannot occur in practice.
#![allow(clippy::cast_precision_loss)]

const EPS: f32 = 1e-12;

/// Cosine similarity between two equal-length vectors. Returns `0.0` if
/// either input has zero norm. Mismatched lengths are a programmer error
/// and panic in debug builds; in release the shorter length wins.
#[must_use]
pub fn cos_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "cos_similarity needs equal-length vectors"
    );
    let mut dot = 0.0f32;
    let mut na2 = 0.0f32;
    let mut nb2 = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na2 += x * x;
        nb2 += y * y;
    }
    let na = na2.sqrt();
    let nb = nb2.sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Maximum cosine similarity between `emb` and any vector in `pool`.
/// Returns `0.0` when the pool is empty (matches the Python PoC).
#[must_use]
pub fn cos_sim_max<V: AsRef<[f32]>>(emb: &[f32], pool: &[V]) -> f32 {
    let mut best = 0.0f32;
    let mut found = false;
    for ref_vec in pool {
        found = true;
        let s = cos_similarity(emb, ref_vec.as_ref());
        if s > best {
            best = s;
        }
    }
    if found {
        best
    } else {
        0.0
    }
}

/// Iterator variant of [`cos_sim_max`]. Lets the caller chain
/// anchors + auto_learn without allocating a temporary `Vec` per call
/// (the offline pipeline refreshes embeddings 2-4× per second so the
/// alloc would otherwise show up in profiles).
#[must_use]
pub fn cos_sim_max_iter<'a, I>(emb: &[f32], pool: I) -> f32
where
    I: IntoIterator<Item = &'a [f32]>,
{
    let mut best = 0.0f32;
    let mut found = false;
    for ref_vec in pool {
        found = true;
        let s = cos_similarity(emb, ref_vec);
        if s > best {
            best = s;
        }
    }
    if found {
        best
    } else {
        0.0
    }
}

/// Gaussian fit of `f0_mean` against the enrollment F0 distribution.
/// Returns `1.0` (neutral) when `sigma <= 0` or `f0_mean` is non-finite,
/// matching the Python implementation.
#[must_use]
pub fn f0_match(f0_mean: f32, mu: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 || !f0_mean.is_finite() {
        return 1.0;
    }
    let z = (f0_mean - mu) / sigma;
    (-0.5 * z * z).exp()
}

/// Apply Adaptive S-Norm to `raw_score`.
///
/// `cohort` is expected to hold L2-normalised rows (one impostor per row).
/// Empty cohort, zero-norm `embedding`, or `top_k == 0` all fall back to
/// returning `raw_score` unchanged. When the top-K spread `σ ≈ 0`, the
/// divide is skipped and `raw_score - μ` is returned (matching Python).
#[must_use]
pub fn as_norm_score<V: AsRef<[f32]>>(
    embedding: &[f32],
    raw_score: f32,
    cohort: &[V],
    top_k: usize,
) -> f32 {
    if cohort.is_empty() {
        return raw_score;
    }
    let mut norm2 = 0.0f32;
    for &x in embedding {
        norm2 += x * x;
    }
    let norm = norm2.sqrt();
    if norm < EPS {
        return raw_score;
    }
    let inv = 1.0 / norm;

    let mut impostor_scores: Vec<f32> = cohort
        .iter()
        .map(|row| {
            let row = row.as_ref();
            debug_assert_eq!(row.len(), embedding.len());
            let mut dot = 0.0f32;
            for (&x, &y) in row.iter().zip(embedding.iter()) {
                dot += x * y;
            }
            dot * inv
        })
        .collect();

    let n = impostor_scores.len();
    let k = top_k.min(n);
    if k == 0 {
        return raw_score;
    }

    let top: &[f32] = if k == n {
        &impostor_scores
    } else {
        // Partition so the largest `k` end up at the tail.
        impostor_scores.select_nth_unstable_by(n - k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        &impostor_scores[n - k..]
    };

    let k_f = k as f32;
    let mu: f32 = top.iter().copied().sum::<f32>() / k_f;
    let var: f32 = top.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / k_f;
    let sigma = var.sqrt();
    if sigma < EPS {
        return raw_score - mu;
    }
    (raw_score - mu) / sigma
}

/// AS-Norm-only integrated score: take the max cosine similarity against
/// the enrollment `pool`, then z-normalise it against the impostor
/// `cohort`. Mirrors `gating.target_score_as_norm` in the Python PoC.
#[must_use]
pub fn target_score_as_norm<P: AsRef<[f32]>, C: AsRef<[f32]>>(
    embedding: &[f32],
    pool: &[P],
    cohort: &[C],
    as_norm_top_k: usize,
) -> f32 {
    let cs = cos_sim_max(embedding, pool);
    as_norm_score(embedding, cs, cohort, as_norm_top_k)
}

// ---------------------------------------------------------------------
// Stateful streaming primitives
// ---------------------------------------------------------------------

/// Thresholds and timing constants for the binary gate + envelope.
/// Mirrors the relevant fields of `mellonella_poc.config.GatingConfig`;
/// other PoC fields (auto-learn admission, anchor distances, alpha/beta
/// mix) live on dedicated configs in their own modules.
#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Pass threshold for the legacy `α·cs + β·f0` mixed score.
    pub theta_pass: f32,
    /// Pass threshold under Adaptive S-Norm (z-score scale).
    pub theta_pass_as_norm: f32,
    /// When `true`, [`GateState::update`] compares against
    /// `theta_pass_as_norm`; otherwise against `theta_pass`.
    pub use_as_norm: bool,
    /// How long the gate stays ON after the score drops below threshold.
    pub hangover_ms: f32,
    /// Envelope attack time in milliseconds.
    pub attack_ms: f32,
    /// Envelope release time in milliseconds.
    pub release_ms: f32,
    /// Auto-learn admission threshold for the legacy mixed score.
    /// Strictly greater than [`Self::theta_pass`].
    pub theta_learn: f32,
    /// Auto-learn admission threshold under AS-Norm. Strictly greater
    /// than [`Self::theta_pass_as_norm`].
    pub theta_learn_as_norm: f32,
    /// Minimum F0 match required before an embedding can join
    /// auto-learn.
    pub theta_f0: f32,
    /// Minimum continuous speech run length (seconds) before auto-learn
    /// admission. Mirrors `min_continuous_speech_sec` in Python.
    pub min_continuous_speech_sec: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        // Mirrors `mellonella_poc.config.GatingConfig` defaults for the
        // gate / envelope / auto-learn fields. Source:
        // poc/mellonella_poc/config.py.
        Self {
            theta_pass: 0.30,
            theta_pass_as_norm: 2.25,
            use_as_norm: false,
            hangover_ms: 300.0,
            attack_ms: 15.0,
            release_ms: 100.0,
            theta_learn: 0.80,
            theta_learn_as_norm: 3.25,
            theta_f0: 0.7,
            min_continuous_speech_sec: 1.0,
        }
    }
}

/// Mutable hangover state for the binary gate. `update` returns the
/// binary decision (`true` == pass).
#[derive(Debug, Clone, Copy)]
pub struct GateState {
    config: GateConfig,
    is_on: bool,
    elapsed_off_ms: f32,
}

impl GateState {
    #[must_use]
    pub fn new(config: GateConfig) -> Self {
        Self {
            config,
            is_on: false,
            elapsed_off_ms: 0.0,
        }
    }

    #[must_use]
    pub fn is_on(&self) -> bool {
        self.is_on
    }

    fn theta_pass(&self) -> f32 {
        if self.config.use_as_norm {
            self.config.theta_pass_as_norm
        } else {
            self.config.theta_pass
        }
    }

    /// Step the gate forward by `dt_ms` with the latest `score`. Returns
    /// the binary decision the caller should feed into the envelope.
    pub fn update(&mut self, score: f32, dt_ms: f32) -> bool {
        if score >= self.theta_pass() {
            self.is_on = true;
            self.elapsed_off_ms = 0.0;
            return true;
        }
        if self.is_on {
            self.elapsed_off_ms += dt_ms;
            if self.elapsed_off_ms < self.config.hangover_ms {
                return true;
            }
            self.is_on = false;
            return false;
        }
        false
    }
}

/// Attack/release envelope follower. Each [`advance`](Self::advance)
/// emits an `n_samples` gain ramp in `[0, 1]` to multiply against audio.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeState {
    config: GateConfig,
    sample_rate: u32,
    value: f32,
}

impl EnvelopeState {
    #[must_use]
    pub fn new(config: GateConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
            value: 0.0,
        }
    }

    /// Current envelope value (the last emitted gain). Useful for tests
    /// that need to seed the follower with a non-zero starting point.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn set_value(&mut self, v: f32) {
        self.value = v;
    }

    fn coef(&self, ms: f32) -> f32 {
        if ms <= 0.0 {
            return 1.0;
        }
        let tau_samples = ms * self.sample_rate as f32 / 1000.0;
        1.0 - (-1.0 / tau_samples).exp()
    }

    /// Step `n_samples` toward `target_on` ∈ {`true`, `false`}.
    pub fn advance(&mut self, target_on: bool, n_samples: usize) -> Vec<f32> {
        let coef = self.coef(if target_on {
            self.config.attack_ms
        } else {
            self.config.release_ms
        });
        let target = if target_on { 1.0 } else { 0.0 };
        let mut out = Vec::with_capacity(n_samples);
        let mut v = self.value;
        for _ in 0..n_samples {
            v += coef * (target - v);
            out.push(v);
        }
        self.value = v;
        out
    }
}

/// Error returned by [`apply_envelope`].
#[derive(Debug, PartialEq, Eq)]
pub enum ApplyEnvelopeError {
    /// `decisions` was empty or did not start at sample 0.
    DecisionsMustStartAtZero,
}

impl std::fmt::Display for ApplyEnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecisionsMustStartAtZero => f.write_str("decisions must start with sample 0"),
        }
    }
}

impl std::error::Error for ApplyEnvelopeError {}

/// Auto-learn admission rule (mirrors `gating.should_admit_auto_learn`
/// in the Python PoC). Returns `true` iff every guard is satisfied:
///
/// * `score >= theta_learn` (or `theta_learn_as_norm` under AS-Norm)
/// * `f0_match_value >= config.theta_f0`
/// * `continuous_speech_ms >= config.min_continuous_speech_sec * 1000`
///
/// Caller chains this with [`crate::enrollment::EmbeddingPool::can_auto_learn`]
/// to also enforce the anchor-distance guard before adding the candidate
/// to the auto-learn FIFO.
#[must_use]
pub fn should_admit_auto_learn(
    score: f32,
    f0_match_value: f32,
    continuous_speech_ms: f32,
    config: &GateConfig,
) -> bool {
    let theta_learn = if config.use_as_norm {
        config.theta_learn_as_norm
    } else {
        config.theta_learn
    };
    score >= theta_learn
        && f0_match_value >= config.theta_f0
        && continuous_speech_ms >= config.min_continuous_speech_sec * 1000.0
}

/// Apply the attack/release envelope to `audio` given the gate
/// `decisions` produced by [`GateState::update`].
///
/// `decisions` is a list of `(start_sample, is_on)` tuples in increasing
/// order. The first tuple's `start_sample` must be 0; every entry's
/// region runs to the next entry's start (or the end of the buffer).
///
/// # Errors
/// Returns [`ApplyEnvelopeError::DecisionsMustStartAtZero`] when the
/// invariant on the first decision is violated.
pub fn apply_envelope(
    audio: &[f32],
    decisions: &[(usize, bool)],
    sample_rate: u32,
    config: GateConfig,
) -> Result<Vec<f32>, ApplyEnvelopeError> {
    if decisions.is_empty() || decisions[0].0 != 0 {
        return Err(ApplyEnvelopeError::DecisionsMustStartAtZero);
    }
    let mut env = EnvelopeState::new(config, sample_rate);
    let n = audio.len();
    let mut out = vec![0.0_f32; n];
    let mut boundaries: Vec<usize> = decisions.iter().map(|d| d.0).collect();
    boundaries.push(n);
    for (i, &(start, is_on)) in decisions.iter().enumerate() {
        let end = boundaries[i + 1];
        if end < start || end > n {
            // Out-of-order decisions are caller bugs; clamp to keep us
            // memory-safe and let the test detect the misuse via the
            // mis-shaped output.
            continue;
        }
        let gain = env.advance(is_on, end - start);
        for (k, &g) in gain.iter().enumerate() {
            out[start + k] = audio[start + k] * g;
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn cos_similarity_basic() {
        assert!(approx_eq(
            cos_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]),
            1.0,
            1e-6
        ));
        assert!(approx_eq(
            cos_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            0.0,
            1e-6
        ));
        assert!(approx_eq(
            cos_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]),
            -1.0,
            1e-6
        ));
    }

    #[test]
    fn cos_similarity_zero_vector() {
        assert!(approx_eq(
            cos_similarity(&[0.0, 0.0], &[1.0, 0.0]),
            0.0,
            0.0
        ));
    }

    #[test]
    fn cos_sim_max_picks_best() {
        let target = [1.0_f32, 0.0];
        let pool: Vec<Vec<f32>> = vec![vec![0.5, 0.5], vec![0.99, 0.01], vec![0.0, 1.0]];
        let best = cos_sim_max(&target, &pool);
        assert!(best > 0.99 && best <= 1.0, "got {best}");
    }

    #[test]
    fn cos_sim_max_empty_pool() {
        let pool: Vec<Vec<f32>> = vec![];
        assert!(approx_eq(cos_sim_max(&[1.0_f32], &pool), 0.0, 0.0));
    }

    #[test]
    fn f0_match_gaussian() {
        assert!(approx_eq(f0_match(120.0, 120.0, 10.0), 1.0, 1e-6));
        assert!(approx_eq(
            f0_match(130.0, 120.0, 10.0),
            (-0.5_f32).exp(),
            1e-6
        ));
        assert!(f0_match(180.0, 120.0, 10.0) < 1e-3);
    }

    #[test]
    fn f0_match_neutral_when_sigma_zero() {
        assert!(approx_eq(f0_match(120.0, 120.0, 0.0), 1.0, 0.0));
        assert!(approx_eq(f0_match(0.0, 120.0, 0.0), 1.0, 0.0));
    }

    #[test]
    fn as_norm_returns_raw_when_cohort_empty() {
        let cohort: Vec<Vec<f32>> = vec![];
        let out = as_norm_score(&[1.0_f32, 0.0], 0.7, &cohort, 10);
        assert!(approx_eq(out, 0.7, 1e-6));
    }

    #[test]
    fn as_norm_returns_raw_for_zero_query() {
        let emb = [0.0_f32; 8];
        let cohort: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                let mut row = vec![0.0; 8];
                row[i] = 1.0;
                row
            })
            .collect();
        let out = as_norm_score(&emb, 0.4, &cohort, 4);
        assert!(approx_eq(out, 0.4, 1e-6));
    }

    #[test]
    fn as_norm_normalises_to_z_score() {
        // Mirrors poc/tests/test_gating_as_norm.py::test_as_norm_normalises_to_z_score.
        let cohort: Vec<Vec<f32>> = vec![
            vec![0.4, 0.0, 0.0],
            vec![0.5, 0.0, 0.0],
            vec![0.6, 0.0, 0.0],
            vec![0.7, 0.0, 0.0],
            vec![0.0, 1.0, 0.0], // orthogonal — score 0
        ];
        let emb = [1.0_f32, 0.0, 0.0];
        let raw: f32 = 0.8;

        // Top-K=4 impostor scores: [0.4, 0.5, 0.6, 0.7]
        let top = [0.4_f32, 0.5, 0.6, 0.7];
        let mu: f32 = top.iter().copied().sum::<f32>() / top.len() as f32;
        let var: f32 = top.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / top.len() as f32;
        let sigma = var.sqrt();
        let expected = (raw - mu) / sigma;

        let out = as_norm_score(&emb, raw, &cohort, 4);
        assert!(
            approx_eq(out, expected, 1e-5),
            "out={out} expected={expected}"
        );
    }

    #[test]
    fn as_norm_handles_zero_sigma_topk() {
        let cohort: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0_f32, 0.0, 0.0]).collect();
        let emb = [1.0_f32, 0.0, 0.0];
        let out = as_norm_score(&emb, 0.9, &cohort, 4);
        assert!(approx_eq(out, 0.9 - 1.0, 1e-5));
    }

    #[test]
    fn as_norm_clamps_top_k_to_cohort_size() {
        // 3 unit basis vectors — request top-K=99
        let cohort: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let emb = [1.0_f32, 0.0, 0.0];
        let out = as_norm_score(&emb, 0.5, &cohort, 99);
        // impostor scores: [1.0, 0.0, 0.0]
        let scores = [1.0_f32, 0.0, 0.0];
        let mu: f32 = scores.iter().copied().sum::<f32>() / 3.0;
        let var: f32 = scores.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / 3.0;
        let expected = (0.5 - mu) / var.sqrt();
        assert!(
            approx_eq(out, expected, 1e-5),
            "out={out} expected={expected}"
        );
    }

    #[test]
    fn target_score_as_norm_uses_cos_sim_max() {
        // Mirrors poc/tests/test_gating_as_norm.py::test_target_score_as_norm_uses_cos_sim_max.
        let pool: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0]];
        let cohort_raw: Vec<Vec<f32>> = vec![
            vec![0.5, 0.5, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let cohort: Vec<Vec<f32>> = cohort_raw
            .into_iter()
            .map(|row| {
                let n: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                row.into_iter().map(|v| v / n).collect()
            })
            .collect();
        let emb = [1.0_f32, 0.0, 0.0];

        let impostor_scores: Vec<f32> = cohort
            .iter()
            .map(|row| {
                row.iter()
                    .zip(emb.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
            })
            .collect();
        // top-2 of [√2/2, 0.0, 0.0] = [√2/2, 0.0]
        let mut sorted = impostor_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top = &sorted[sorted.len() - 2..];
        let mu = top.iter().copied().sum::<f32>() / top.len() as f32;
        let var: f32 = top.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / top.len() as f32;
        let expected = (1.0 - mu) / var.sqrt();

        let out = target_score_as_norm(&emb, &pool, &cohort, 2);
        assert!(
            approx_eq(out, expected, 1e-5),
            "out={out} expected={expected}"
        );
    }

    // -----------------------------------------------------------------
    // Stateful streaming primitives (mirrors poc/tests/test_gating.py)
    // -----------------------------------------------------------------

    #[test]
    fn gate_state_pass_above_threshold() {
        let mut gate = GateState::new(GateConfig::default());
        assert!(gate.update(0.9, 20.0));
        assert!(gate.is_on());
    }

    #[test]
    fn gate_state_hangover_keeps_pass() {
        let mut gate = GateState::new(GateConfig {
            hangover_ms: 200.0,
            ..GateConfig::default()
        });
        gate.update(0.9, 20.0);
        assert!(gate.update(0.0, 100.0));
        assert!(gate.update(0.0, 99.0));
        assert!(!gate.update(0.0, 10.0));
    }

    #[test]
    fn gate_state_no_hangover_when_off() {
        let mut gate = GateState::new(GateConfig::default());
        assert!(!gate.update(0.0, 20.0));
        assert!(!gate.update(0.0, 20.0));
    }

    #[test]
    fn envelope_attack_release_monotonic() {
        let cfg = GateConfig {
            attack_ms: 15.0,
            release_ms: 100.0,
            ..GateConfig::default()
        };
        let sr: u32 = 48_000;
        let mut env = EnvelopeState::new(cfg, sr);
        let n_attack = (0.1 * f64::from(sr)) as usize;
        let on = env.advance(true, n_attack);
        assert!(on[0] < *on.last().unwrap());
        assert!(*on.last().unwrap() > 0.95);
        let n_release = (0.5 * f64::from(sr)) as usize;
        let off = env.advance(false, n_release);
        assert!(off[0] > *off.last().unwrap());
        assert!(*off.last().unwrap() < 0.05);
    }

    #[test]
    fn envelope_attack_faster_than_release() {
        let cfg = GateConfig {
            attack_ms: 15.0,
            release_ms: 100.0,
            ..GateConfig::default()
        };
        let sr: u32 = 48_000;
        let mut env_a = EnvelopeState::new(cfg, sr);
        let mut env_b = EnvelopeState::new(cfg, sr);
        let attack_curve = env_a.advance(true, sr as usize);
        env_b.set_value(1.0);
        let release_curve = env_b.advance(false, sr as usize);
        let to_half_attack = attack_curve
            .iter()
            .position(|&v| v >= 0.5)
            .expect("attack must cross 0.5");
        let to_half_release = release_curve
            .iter()
            .position(|&v| v <= 0.5)
            .expect("release must cross 0.5");
        assert!(to_half_attack > 0);
        assert!(to_half_release > 0);
        assert!(to_half_attack < to_half_release);
    }

    #[test]
    fn apply_envelope_alignment() {
        let cfg = GateConfig {
            attack_ms: 15.0,
            release_ms: 100.0,
            ..GateConfig::default()
        };
        let sr: u32 = 48_000;
        let n = sr as usize;
        let audio = vec![1.0_f32; n];
        let decisions = [(0, true), (n / 2, false)];
        let out = apply_envelope(&audio, &decisions, sr, cfg).unwrap();
        assert_eq!(out.len(), audio.len());
        let mid = n / 2;
        assert!(out[mid - 100] > 0.95);
        assert!(out[n - 100] < 0.05);
    }

    #[test]
    fn apply_envelope_requires_zero_start() {
        let audio = vec![1.0_f32; 10];
        let res = apply_envelope(&audio, &[(1, true)], 48_000, GateConfig::default());
        assert_eq!(
            res.unwrap_err(),
            ApplyEnvelopeError::DecisionsMustStartAtZero
        );
    }

    #[test]
    fn apply_envelope_rejects_empty_decisions() {
        let audio = vec![1.0_f32; 10];
        let res = apply_envelope(&audio, &[], 48_000, GateConfig::default());
        assert_eq!(
            res.unwrap_err(),
            ApplyEnvelopeError::DecisionsMustStartAtZero
        );
    }

    // -----------------------------------------------------------------
    // Auto-learn admission rule
    // -----------------------------------------------------------------

    #[test]
    fn should_admit_auto_learn_all_pass() {
        let cfg = GateConfig::default();
        assert!(should_admit_auto_learn(0.85, 0.75, 1500.0, &cfg));
    }

    #[test]
    fn should_admit_auto_learn_low_score_blocked() {
        let cfg = GateConfig::default();
        assert!(!should_admit_auto_learn(0.79, 0.9, 2000.0, &cfg));
    }

    #[test]
    fn should_admit_auto_learn_low_f0_match_blocked() {
        let cfg = GateConfig::default();
        assert!(!should_admit_auto_learn(0.85, 0.5, 2000.0, &cfg));
    }

    #[test]
    fn should_admit_auto_learn_short_run_blocked() {
        let cfg = GateConfig::default();
        assert!(!should_admit_auto_learn(0.95, 0.9, 500.0, &cfg));
    }

    #[test]
    fn should_admit_auto_learn_boundary_inclusive() {
        let cfg = GateConfig {
            theta_learn: 0.80,
            theta_f0: 0.7,
            min_continuous_speech_sec: 1.0,
            ..GateConfig::default()
        };
        assert!(should_admit_auto_learn(0.80, 0.7, 1000.0, &cfg));
        assert!(!should_admit_auto_learn(0.799, 0.7, 1000.0, &cfg));
        assert!(!should_admit_auto_learn(0.80, 0.699, 1000.0, &cfg));
        assert!(!should_admit_auto_learn(0.80, 0.7, 999.0, &cfg));
    }

    #[test]
    fn should_admit_auto_learn_uses_as_norm_threshold() {
        let cfg = GateConfig {
            use_as_norm: true,
            theta_pass_as_norm: 1.0,
            theta_learn_as_norm: 3.0,
            theta_f0: 0.5,
            min_continuous_speech_sec: 0.1,
            ..GateConfig::default()
        };
        assert!(should_admit_auto_learn(3.5, 0.9, 200.0, &cfg));
        assert!(!should_admit_auto_learn(2.5, 0.9, 200.0, &cfg));
    }
}
