"""EmbeddingPool tests. Uses synthetic 8-d embeddings."""

from __future__ import annotations

import json

import numpy as np

from mellonella_poc.config import GatingConfig
from mellonella_poc.enrollment import EmbeddingPool


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def test_anchor_distance_zero_for_identical():
    cfg = GatingConfig()
    pool = EmbeddingPool(config=cfg)
    anchor = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    pool.add_anchors([anchor])
    assert pool.anchor_distance(anchor) < 1e-6


def test_can_auto_learn_rejects_far_embedding():
    cfg = GatingConfig(anchor_distance_threshold=0.4)
    pool = EmbeddingPool(config=cfg)
    pool.add_anchors([_unit(np.array([1.0, 0.0, 0.0]))])
    near = _unit(np.array([0.95, 0.05, 0.0]))
    far = _unit(np.array([0.0, 0.0, 1.0]))
    assert pool.can_auto_learn(near)
    assert not pool.can_auto_learn(far)


def test_add_auto_learn_respects_fifo_bound():
    cfg = GatingConfig(auto_learn_max_size=3)
    pool = EmbeddingPool(config=cfg)
    pool.add_anchors([_unit(np.array([1.0, 0.0, 0.0]))])
    candidates = [_unit(np.array([1.0, 0.01 * i, 0.0])) for i in range(1, 6)]
    for c in candidates:
        pool.add_auto_learn(c)
    assert len(pool.auto_learn) == 3
    np.testing.assert_allclose(pool.auto_learn[-1], candidates[-1], atol=1e-6)


def test_no_auto_learn_without_anchor():
    cfg = GatingConfig()
    pool = EmbeddingPool(config=cfg)
    assert not pool.add_auto_learn(_unit(np.array([1.0, 0.0])))


def test_maybe_reset_clears_drifted_pool():
    cfg = GatingConfig(anchor_distance_threshold=0.99, anchor_reset_threshold=0.5)
    pool = EmbeddingPool(config=cfg)
    pool.add_anchors([_unit(np.array([1.0, 0.0, 0.0]))])
    pool.auto_learn.extend(_unit(np.array([0.0, 1.0, 0.0])) for _ in range(5))
    assert pool.maybe_reset()
    assert len(pool.auto_learn) == 0


def test_maybe_reset_keeps_healthy_pool():
    cfg = GatingConfig(anchor_distance_threshold=0.5, anchor_reset_threshold=0.5)
    pool = EmbeddingPool(config=cfg)
    pool.add_anchors([_unit(np.array([1.0, 0.0, 0.0]))])
    pool.auto_learn.extend(_unit(np.array([1.0, 0.001 * i, 0.0])) for i in range(1, 4))
    assert not pool.maybe_reset()
    assert len(pool.auto_learn) == 3


def test_round_trip_to_json(tmp_path):
    cfg = GatingConfig()
    pool = EmbeddingPool(config=cfg)
    pool.add_anchors([_unit(np.array([1.0, 0.0]))])
    pool.set_f0_stats(180.0, 25.0)
    pool.add_auto_learn(_unit(np.array([0.99, 0.01])))
    path = tmp_path / "enrollment.json"
    pool.save(path)

    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert len(payload["anchors"]) == 1

    restored = EmbeddingPool.load(path, cfg)
    assert len(restored.anchors) == 1
    assert restored.metadata.f0_mu == 180.0
    assert restored.metadata.f0_sigma == 25.0
    assert len(restored.auto_learn) == 1


def test_iter_yields_anchors_then_autolearn():
    cfg = GatingConfig()
    pool = EmbeddingPool(config=cfg)
    a = _unit(np.array([1.0, 0.0]))
    b = _unit(np.array([0.0, 1.0]))
    pool.add_anchors([a])
    pool.auto_learn.append(b)
    seen = list(iter(pool))
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], a)
    np.testing.assert_allclose(seen[1], b)
