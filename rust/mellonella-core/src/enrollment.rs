//! Two-compartment embedding pool ported from
//! `poc/mellonella_poc/enrollment.py`.
//!
//! `EmbeddingPool` holds:
//!
//! * `anchors`     — written once by explicit enrollment, never deleted
//! * `auto_learn`  — `VecDeque` of high-confidence runtime embeddings
//!
//! Drift safeguards mirror `docs/gating.md` D-004:
//!
//! * candidates must clear a minimum cosine similarity to existing
//!   anchors (the `theta_learn` score check is the *caller's* job)
//! * the pool's median anchor distance is monitored and a soft reset
//!   clears the auto-learn FIFO when it exceeds `anchor_reset_threshold`
//!
//! Persistence (JSON round-trip) is intentionally deferred — `serde` /
//! `serde_json` are not yet workspace deps and the algorithm path needs
//! to stabilise before the on-disk schema is fixed.

use std::collections::VecDeque;

use crate::gating::cos_similarity;

/// Subset of the Python `GatingConfig` consumed by the pool. More fields
/// will be added as stateful gating gets ported; until then the pool
/// only needs the drift-safety knobs.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingPoolConfig {
    /// Maximum `1 - max_cos(anchors)` allowed for auto-learn admission.
    pub anchor_distance_threshold: f32,
    /// Median anchor distance over the auto-learn pool that triggers
    /// a soft reset of the FIFO.
    pub anchor_reset_threshold: f32,
    /// FIFO capacity for the auto-learn compartment.
    pub auto_learn_max_size: usize,
}

impl Default for EmbeddingPoolConfig {
    fn default() -> Self {
        // Mirrors `mellonella_poc.config.GatingConfig` defaults.
        Self {
            anchor_distance_threshold: 0.4,
            anchor_reset_threshold: 0.5,
            auto_learn_max_size: 20,
        }
    }
}

/// F0 statistics captured from the explicit enrollment recording.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnrollmentMetadata {
    pub f0_mu: f32,
    pub f0_sigma: f32,
}

/// Two-compartment pool with anchor protection.
#[derive(Debug, Clone)]
pub struct EmbeddingPool {
    config: EmbeddingPoolConfig,
    anchors: Vec<Vec<f32>>,
    auto_learn: VecDeque<Vec<f32>>,
    metadata: EnrollmentMetadata,
}

impl EmbeddingPool {
    #[must_use]
    pub fn new(config: EmbeddingPoolConfig) -> Self {
        Self {
            config,
            anchors: Vec::new(),
            auto_learn: VecDeque::new(),
            metadata: EnrollmentMetadata::default(),
        }
    }

    #[must_use]
    pub fn config(&self) -> EmbeddingPoolConfig {
        self.config
    }

    #[must_use]
    pub fn anchors(&self) -> &[Vec<f32>] {
        &self.anchors
    }

    #[must_use]
    pub fn auto_learn(&self) -> &VecDeque<Vec<f32>> {
        &self.auto_learn
    }

    #[must_use]
    pub fn metadata(&self) -> EnrollmentMetadata {
        self.metadata
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.anchors.len() + self.auto_learn.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty() && self.auto_learn.is_empty()
    }

    /// Iterate anchors first, then the auto-learn FIFO (insertion order).
    /// Mirrors Python's `__iter__`.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.anchors
            .iter()
            .map(Vec::as_slice)
            .chain(self.auto_learn.iter().map(Vec::as_slice))
    }

    pub fn add_anchors<I, V>(&mut self, embeddings: I)
    where
        I: IntoIterator<Item = V>,
        V: Into<Vec<f32>>,
    {
        for emb in embeddings {
            self.anchors.push(emb.into());
        }
    }

    pub fn set_f0_stats(&mut self, mu: f32, sigma: f32) {
        self.metadata = EnrollmentMetadata {
            f0_mu: mu,
            f0_sigma: sigma,
        };
    }

    /// `1 - max_cos(anchors)`. Returns `None` if there are no anchors.
    #[must_use]
    pub fn anchor_distance(&self, emb: &[f32]) -> Option<f32> {
        if self.anchors.is_empty() {
            return None;
        }
        let best = self
            .anchors
            .iter()
            .map(|a| cos_similarity(emb, a))
            .fold(f32::NEG_INFINITY, f32::max);
        Some(1.0 - best)
    }

    /// Drift-safety check before accepting an auto-learn candidate.
    /// Returns `false` when there are no anchors yet.
    #[must_use]
    pub fn can_auto_learn(&self, emb: &[f32]) -> bool {
        match self.anchor_distance(emb) {
            Some(d) => d <= self.config.anchor_distance_threshold,
            None => false,
        }
    }

    /// Try to admit `emb` to the auto-learn FIFO. Returns `true` on
    /// accept. Equivalent to Python's `add_auto_learn`.
    pub fn add_auto_learn<V: Into<Vec<f32>>>(&mut self, emb: V) -> bool {
        let emb = emb.into();
        if !self.can_auto_learn(&emb) {
            return false;
        }
        self.auto_learn.push_back(emb);
        while self.auto_learn.len() > self.config.auto_learn_max_size {
            self.auto_learn.pop_front();
        }
        true
    }

    /// Median of `1 - max_cos(anchors)` over every auto-learn entry, or
    /// `0.0` when the pool is empty.
    #[must_use]
    pub fn median_anchor_distance(&self) -> f32 {
        if self.auto_learn.is_empty() {
            return 0.0;
        }
        let mut distances: Vec<f32> = self
            .auto_learn
            .iter()
            .filter_map(|e| self.anchor_distance(e))
            .collect();
        if distances.is_empty() {
            return 0.0;
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = distances.len();
        if n % 2 == 0 {
            (distances[n / 2 - 1] + distances[n / 2]) * 0.5
        } else {
            distances[n / 2]
        }
    }

    /// Reset the auto-learn FIFO if median anchor distance is too high.
    /// Returns `true` when a reset happened.
    pub fn maybe_reset(&mut self) -> bool {
        if self.median_anchor_distance() > self.config.anchor_reset_threshold {
            self.auto_learn.clear();
            return true;
        }
        false
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn anchor_distance_zero_for_identical() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchor = unit(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        pool.add_anchors([anchor.clone()]);
        let d = pool.anchor_distance(&anchor).unwrap();
        assert!(d.abs() < 1e-6, "d={d}");
    }

    #[test]
    fn anchor_distance_none_when_empty() {
        let pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert!(pool.anchor_distance(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn can_auto_learn_rejects_far_embedding() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 0.4,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        let near = unit(&[0.95, 0.05, 0.0]);
        let far = unit(&[0.0, 0.0, 1.0]);
        assert!(pool.can_auto_learn(&near));
        assert!(!pool.can_auto_learn(&far));
    }

    #[test]
    fn add_auto_learn_respects_fifo_bound() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            auto_learn_max_size: 3,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        let candidates: Vec<Vec<f32>> = (1..=5)
            .map(|i| unit(&[1.0, 0.01 * i as f32, 0.0]))
            .collect();
        for c in &candidates {
            pool.add_auto_learn(c.clone());
        }
        assert_eq!(pool.auto_learn().len(), 3);
        let last = pool.auto_learn().back().unwrap();
        for (a, b) in last.iter().zip(candidates[4].iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn no_auto_learn_without_anchor() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert!(!pool.add_auto_learn(unit(&[1.0, 0.0])));
    }

    #[test]
    fn maybe_reset_clears_drifted_pool() {
        // Admission threshold 1.0 lets the orthogonal vector in (the
        // Python test bypasses admission by extending the deque directly;
        // here we go through the public API and pick a threshold that
        // admits the same drift candidate).
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            anchor_reset_threshold: 0.5,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        for _ in 0..5 {
            assert!(pool.add_auto_learn(unit(&[0.0, 1.0, 0.0])));
        }
        assert!(pool.maybe_reset());
        assert!(pool.auto_learn().is_empty());
    }

    #[test]
    fn maybe_reset_keeps_healthy_pool() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 0.5,
            anchor_reset_threshold: 0.5,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        for i in 1..=3 {
            pool.add_auto_learn(unit(&[1.0, 0.001 * i as f32, 0.0]));
        }
        assert!(!pool.maybe_reset());
        assert_eq!(pool.auto_learn().len(), 3);
    }

    #[test]
    fn iter_yields_anchors_then_autolearn() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let a = unit(&[1.0, 0.0]);
        let b = unit(&[0.0, 1.0]);
        pool.add_anchors([a.clone()]);
        pool.auto_learn.push_back(b.clone());
        let seen: Vec<Vec<f32>> = pool.iter().map(<[f32]>::to_vec).collect();
        assert_eq!(seen.len(), 2);
        for (got, want) in seen[0].iter().zip(a.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
        for (got, want) in seen[1].iter().zip(b.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn set_f0_stats_round_trip() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.set_f0_stats(180.0, 25.0);
        let m = pool.metadata();
        assert_eq!(m.f0_mu, 180.0);
        assert_eq!(m.f0_sigma, 25.0);
    }
}
