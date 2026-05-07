//! Gating logic ported from `poc/mellonella_poc/gating.py`.
//!
//! Pure-math primitives operating on already-computed embeddings — no
//! dependency on the ECAPA ONNX wrapper. Mirrors:
//!
//! * [`cos_similarity`] / [`cos_sim_max`] — cosine similarity helpers
//! * [`f0_match`]                          — Gaussian F0 match
//! * [`as_norm_score`]                     — Adaptive S-Norm z-score
//! * [`target_score_as_norm`]              — pool-vs-cohort integrated score
//!
//! Stateful pieces (gate hangover, attack/release envelope, auto-learn
//! admission) are intentionally not yet ported — they will follow once the
//! streaming buffer model in [`crate::pipeline`] (TBD) is settled.

// `usize → f32` for top-K mean / variance: cohort top-K is bounded by
// literature 10–50, far below f32's 2^23 mantissa, so the loss clippy
// warns about cannot occur in practice.
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

#[cfg(test)]
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
}
