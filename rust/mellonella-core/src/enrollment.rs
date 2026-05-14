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
//! JSON persistence ([`EmbeddingPool::to_json`] / [`from_json`] /
//! [`save`](Self::save) / [`load`](Self::load)) uses the same `version: 1`
//! layout as `mellonella_poc.enrollment.EmbeddingPool.to_dict`, so a
//! pool serialised by Python can be deserialised by Rust and vice versa.

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::gating::{cos_sim_max_iter, cos_similarity};

/// Persistence-schema version. Incremented if the on-disk shape ever
/// changes; readers reject any payload whose `version` field differs.
pub const PERSISTENCE_VERSION: u32 = 1;

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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EnrollmentMetadata {
    #[serde(default)]
    pub f0_mu: f32,
    #[serde(default)]
    pub f0_sigma: f32,
}

/// Two-compartment pool with anchor protection.
#[derive(Debug, Clone)]
pub struct EmbeddingPool {
    config: EmbeddingPoolConfig,
    anchors: Vec<Vec<f32>>,
    auto_learn: VecDeque<Vec<f32>>,
    metadata: EnrollmentMetadata,
    /// Element-wise mean of `anchors`, recomputed whenever the anchor
    /// set changes. `None` while there are no anchors. Cached because
    /// anchors are immutable after enrollment, so the centroid only
    /// has to be recomputed on `add_anchors` / deserialisation rather
    /// than on every score lookup. See [`Self::match_score`] (#117).
    anchor_centroid: Option<Vec<f32>>,
}

/// Element-wise mean of `anchors`, or `None` for an empty slice.
///
/// For a single anchor the mean is that anchor verbatim (multiplying
/// each f32 by `1.0` is exact), so a single-anchor pool scores
/// byte-identically to the pre-#117 `cos_sim_max` path.
fn compute_anchor_centroid(anchors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dim = anchors.first()?.len();
    let mut sum = vec![0.0f32; dim];
    for a in anchors {
        debug_assert_eq!(a.len(), dim, "anchor embeddings must share a dimension");
        for (s, &x) in sum.iter_mut().zip(a.iter()) {
            *s += x;
        }
    }
    // Anchor count is a handful (5–10 enrollment windows), exact in f32.
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / anchors.len() as f32;
    for s in &mut sum {
        *s *= inv;
    }
    Some(sum)
}

impl EmbeddingPool {
    #[must_use]
    pub fn new(config: EmbeddingPoolConfig) -> Self {
        Self {
            config,
            anchors: Vec::new(),
            auto_learn: VecDeque::new(),
            metadata: EnrollmentMetadata::default(),
            anchor_centroid: None,
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
        self.anchor_centroid = compute_anchor_centroid(&self.anchors);
    }

    /// Element-wise mean of the anchor embeddings, or `None` when the
    /// pool has no anchors yet. The reference used by
    /// [`Self::match_score`].
    #[must_use]
    pub fn anchor_centroid(&self) -> Option<&[f32]> {
        self.anchor_centroid.as_deref()
    }

    /// Pool match score for `emb` (#117): cosine similarity against the
    /// anchor centroid, maxed with the per-entry max over the
    /// auto-learn FIFO.
    ///
    /// Replaces the previous "max cosine over every anchor + auto-learn
    /// entry" rule. The anchors all come from one enrollment recording
    /// of the same speaker, so their centroid is a stabler reference
    /// than the single best-matching anchor (one outlier anchor or a
    /// noisy refresh window no longer swings the score). The auto-learn
    /// FIFO keeps the max so distinct runtime vocal modes are still
    /// captured. Floors at `0.0`, matching the old `cos_sim_max`
    /// behaviour; returns `0.0` for an empty pool.
    #[must_use]
    pub fn match_score(&self, emb: &[f32]) -> f32 {
        let anchor_score = self
            .anchor_centroid
            .as_deref()
            .map_or(0.0, |c| cos_similarity(emb, c));
        let auto_score = cos_sim_max_iter(emb, self.auto_learn.iter().map(Vec::as_slice));
        anchor_score.max(auto_score)
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

    // ---- JSON persistence ------------------------------------------

    /// Serialise the pool to a JSON string. Schema mirrors Python's
    /// `EmbeddingPool.to_dict` (version 1).
    ///
    /// # Errors
    /// Returns [`PersistenceError::Json`] if `serde_json` rejects the
    /// payload — should not happen for the in-memory shape, but the
    /// error path is preserved for symmetry with [`Self::from_json`].
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        let payload = PoolPayload::from_pool(self);
        serde_json::to_string(&payload).map_err(PersistenceError::Json)
    }

    /// Deserialise a pool from a JSON string. The pool's `config` is
    /// supplied by the caller — it controls drift thresholds and FIFO
    /// capacity, neither of which is part of the on-disk payload.
    ///
    /// # Errors
    /// * [`PersistenceError::Json`]            — malformed JSON
    /// * [`PersistenceError::UnsupportedVersion`] — `version` field
    ///   does not equal [`PERSISTENCE_VERSION`]
    pub fn from_json(s: &str, config: EmbeddingPoolConfig) -> Result<Self, PersistenceError> {
        let payload: PoolPayload = serde_json::from_str(s).map_err(PersistenceError::Json)?;
        if payload.version != PERSISTENCE_VERSION {
            return Err(PersistenceError::UnsupportedVersion(payload.version));
        }
        Ok(payload.into_pool(config))
    }

    /// Persist the pool as JSON to `path`. Convenience wrapper around
    /// [`Self::to_json`] + `fs::write`.
    ///
    /// # Errors
    /// Returns [`PersistenceError::Io`] for filesystem failures or
    /// [`PersistenceError::Json`] if serialisation fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PersistenceError> {
        let body = self.to_json()?;
        fs::write(path, body).map_err(PersistenceError::Io)
    }

    /// Load a pool from a JSON file at `path`.
    ///
    /// # Errors
    /// Returns [`PersistenceError::Io`] for filesystem failures or
    /// [`PersistenceError::Json`] / [`PersistenceError::UnsupportedVersion`]
    /// for malformed payloads.
    pub fn load(
        path: impl AsRef<Path>,
        config: EmbeddingPoolConfig,
    ) -> Result<Self, PersistenceError> {
        let body = fs::read_to_string(path).map_err(PersistenceError::Io)?;
        Self::from_json(&body, config)
    }
}

/// On-disk representation of [`EmbeddingPool`]. Kept private to the
/// module so the public type remains free to evolve.
#[derive(Debug, Serialize, Deserialize)]
struct PoolPayload {
    version: u32,
    #[serde(default)]
    anchors: Vec<Vec<f32>>,
    #[serde(default)]
    auto_learn: Vec<Vec<f32>>,
    #[serde(default)]
    metadata: EnrollmentMetadata,
}

impl PoolPayload {
    fn from_pool(pool: &EmbeddingPool) -> Self {
        Self {
            version: PERSISTENCE_VERSION,
            anchors: pool.anchors.clone(),
            auto_learn: pool.auto_learn.iter().cloned().collect(),
            metadata: pool.metadata,
        }
    }

    fn into_pool(self, config: EmbeddingPoolConfig) -> EmbeddingPool {
        let mut auto_learn: VecDeque<Vec<f32>> = self.auto_learn.into_iter().collect();
        // Honour the FIFO bound on load — older saves with a larger
        // capacity must not overflow a tighter live config.
        while auto_learn.len() > config.auto_learn_max_size {
            auto_learn.pop_front();
        }
        let anchor_centroid = compute_anchor_centroid(&self.anchors);
        EmbeddingPool {
            config,
            anchors: self.anchors,
            auto_learn,
            metadata: self.metadata,
            anchor_centroid,
        }
    }
}

/// Errors returned by the [`EmbeddingPool`] JSON persistence path.
#[derive(Debug)]
pub enum PersistenceError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported enrollment version: {v}")
            }
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::UnsupportedVersion(_) => None,
        }
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
    fn anchor_centroid_none_when_empty() {
        let pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert!(pool.anchor_centroid().is_none());
    }

    #[test]
    fn anchor_centroid_single_anchor_is_verbatim() {
        // Centroid of one anchor is that anchor byte-for-byte (each
        // f32 multiplied by 1.0). This is what keeps `pipeline_parity`
        // byte-equal under #117.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchor = unit(&[0.3, -0.7, 0.1, 0.64]);
        pool.add_anchors([anchor.clone()]);
        assert_eq!(pool.anchor_centroid().unwrap(), anchor.as_slice());
    }

    #[test]
    fn anchor_centroid_is_elementwise_mean() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.add_anchors([vec![1.0_f32, 3.0, -1.0], vec![3.0_f32, 1.0, 1.0]]);
        assert_eq!(pool.anchor_centroid().unwrap(), &[2.0, 2.0, 0.0]);
    }

    #[test]
    fn add_anchors_recomputes_centroid() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        pool.add_anchors([vec![0.0_f32, 2.0]]);
        assert_eq!(pool.anchor_centroid().unwrap(), &[0.0, 2.0]);
        pool.add_anchors([vec![2.0_f32, 0.0]]);
        assert_eq!(pool.anchor_centroid().unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn match_score_single_anchor_equals_cos_similarity() {
        // Single anchor, no auto-learn: match_score reduces to
        // cos_similarity against that anchor (the pre-#117 path).
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let anchor = unit(&[1.0, 0.0, 0.0]);
        pool.add_anchors([anchor.clone()]);
        let emb = unit(&[0.8, 0.6, 0.0]);
        assert!((pool.match_score(&emb) - cos_similarity(&emb, &anchor)).abs() < 1e-6);
    }

    #[test]
    fn match_score_uses_centroid_not_best_anchor() {
        // Two opposed anchors → centroid near the bisector. A probe
        // aligned with one anchor scores against the *centroid*, which
        // is lower than the old max-over-anchors would have given.
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        let a = unit(&[1.0, 1.0, 0.0]);
        let b = unit(&[1.0, -1.0, 0.0]);
        pool.add_anchors([a.clone(), b]);
        let probe = a.clone();
        let centroid = pool.anchor_centroid().unwrap();
        let score = pool.match_score(&probe);
        assert!((score - cos_similarity(&probe, centroid)).abs() < 1e-6);
        // Old max-over-anchors would have returned cos(probe, a) = 1.0;
        // the centroid score is strictly below that.
        assert!(
            score < 0.999,
            "centroid score should be below the per-anchor max: {score}"
        );
    }

    #[test]
    fn match_score_maxes_centroid_with_auto_learn() {
        let mut pool = EmbeddingPool::new(EmbeddingPoolConfig {
            anchor_distance_threshold: 1.0,
            ..EmbeddingPoolConfig::default()
        });
        pool.add_anchors([unit(&[1.0, 0.0, 0.0])]);
        let probe = unit(&[0.0, 1.0, 0.0]);
        // Orthogonal to the anchor → centroid score ~0.
        assert!(pool.match_score(&probe) < 1e-6);
        // Admit an auto-learn entry that matches the probe → score jumps.
        assert!(pool.add_auto_learn(probe.clone()));
        assert!((pool.match_score(&probe) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn match_score_zero_for_empty_pool() {
        let pool = EmbeddingPool::new(EmbeddingPoolConfig::default());
        assert_eq!(pool.match_score(&[1.0, 0.0, 0.0]), 0.0);
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

    // -----------------------------------------------------------------
    // JSON persistence (mirrors poc/tests/test_enrollment.py)
    // -----------------------------------------------------------------

    #[test]
    fn json_round_trip_preserves_state() {
        let cfg = EmbeddingPoolConfig::default();
        let mut pool = EmbeddingPool::new(cfg);
        let anchor = unit(&[1.0, 0.0]);
        pool.add_anchors([anchor.clone()]);
        pool.set_f0_stats(180.0, 25.0);
        assert!(pool.add_auto_learn(unit(&[0.99, 0.01])));

        let body = pool.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["version"], 1);
        assert_eq!(parsed["anchors"].as_array().unwrap().len(), 1);

        let restored = EmbeddingPool::from_json(&body, cfg).unwrap();
        assert_eq!(restored.anchors().len(), 1);
        assert_eq!(restored.metadata().f0_mu, 180.0);
        assert_eq!(restored.metadata().f0_sigma, 25.0);
        assert_eq!(restored.auto_learn().len(), 1);
    }

    #[test]
    fn save_load_through_filesystem() {
        let cfg = EmbeddingPoolConfig::default();
        let mut pool = EmbeddingPool::new(cfg);
        pool.add_anchors([unit(&[1.0, 0.0])]);

        let dir = std::env::temp_dir().join(format!(
            "mellonella-pool-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("enrollment.json");

        pool.save(&path).unwrap();
        let restored = EmbeddingPool::load(&path, cfg).unwrap();
        assert_eq!(restored.anchors().len(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_json_rejects_unknown_version() {
        let cfg = EmbeddingPoolConfig::default();
        let body = r#"{"version": 99, "anchors": [], "auto_learn": [], "metadata": {}}"#;
        match EmbeddingPool::from_json(body, cfg) {
            Err(PersistenceError::UnsupportedVersion(99)) => {}
            other => panic!("expected UnsupportedVersion(99), got {other:?}"),
        }
    }

    #[test]
    fn from_json_clamps_auto_learn_to_config() {
        // Saved with 5 auto-learn entries; config caps at 2 — load
        // drops the oldest.
        let saved = r#"{
            "version": 1,
            "anchors": [],
            "auto_learn": [[1.0], [2.0], [3.0], [4.0], [5.0]],
            "metadata": {"f0_mu": 0.0, "f0_sigma": 0.0}
        }"#;
        let cfg = EmbeddingPoolConfig {
            auto_learn_max_size: 2,
            ..EmbeddingPoolConfig::default()
        };
        let pool = EmbeddingPool::from_json(saved, cfg).unwrap();
        assert_eq!(pool.auto_learn().len(), 2);
        assert_eq!(pool.auto_learn()[0][0], 4.0);
        assert_eq!(pool.auto_learn()[1][0], 5.0);
    }
}
