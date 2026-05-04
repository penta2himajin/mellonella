"""Unit tests for the AS-Norm helpers added in PR (D-010 Phase 2).

The cohort matrices are tiny synthetic numpy arrays so the suite stays
free of HF / torch and runs in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_poc.config import GatingConfig
from mellonella_poc.gating import (
    GateState,
    as_norm_score,
    load_cohort,
    should_admit_auto_learn,
    target_score_as_norm,
)


def _orthonormal_basis(n: int, dim: int = 192, seed: int = 0) -> np.ndarray:
    """Return ``n`` mutually-orthogonal unit vectors in ``dim`` dims via QR."""
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((max(n, dim), dim))
    q, _ = np.linalg.qr(rand)
    return q[:n].astype(np.float32)


def test_as_norm_returns_raw_when_cohort_empty():
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cohort = np.zeros((0, 2), dtype=np.float32)
    out = as_norm_score(emb, raw_score=0.7, cohort=cohort, top_k=10)
    assert out == pytest.approx(0.7)


def test_as_norm_returns_raw_for_zero_query():
    emb = np.zeros(8, dtype=np.float32)
    cohort = np.eye(8, dtype=np.float32)
    out = as_norm_score(emb, raw_score=0.4, cohort=cohort, top_k=4)
    assert out == pytest.approx(0.4)


def test_as_norm_normalises_to_z_score():
    """With a cohort spread of impostor scores, AS-Norm should subtract
    the mean and divide by the std of the top-K scores."""
    # Query is unit vector e1; cohort scores will equal cohort[:, 0].
    cohort = np.array(
        [
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # orthogonal — score 0
        ],
        dtype=np.float32,
    )
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    raw = 0.8

    # Pick top-K=4 impostor scores: [0.4, 0.5, 0.6, 0.7]
    top = np.array([0.4, 0.5, 0.6, 0.7])
    expected = (raw - top.mean()) / top.std()
    out = as_norm_score(emb, raw_score=raw, cohort=cohort, top_k=4)
    assert out == pytest.approx(float(expected), abs=1e-5)


def test_as_norm_handles_zero_sigma_topk():
    """All top-K scores identical → σ=0 → fall back to (raw - μ) without divide."""
    cohort = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = as_norm_score(emb, raw_score=0.9, cohort=cohort, top_k=4)
    assert out == pytest.approx(0.9 - 1.0, abs=1e-5)


def test_as_norm_clamps_top_k_to_cohort_size():
    """top_k larger than cohort size should silently use the whole cohort."""
    cohort = np.eye(3, dtype=np.float32)  # 3 vectors
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out = as_norm_score(emb, raw_score=0.5, cohort=cohort, top_k=99)
    # impostor scores: [1.0, 0.0, 0.0]; mean=1/3, std≈0.471
    expected = (0.5 - (1.0 / 3.0)) / np.std([1.0, 0.0, 0.0])
    assert out == pytest.approx(float(expected), abs=1e-5)


def test_target_score_as_norm_uses_cos_sim_max(tmp_path):
    """target_score_as_norm should pull max cos sim against the pool first,
    then z-normalise it against the cohort."""
    pool = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    cohort = (
        _orthonormal_basis(8, dim=3)
        if False
        else np.array(
            [
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    cohort = cohort / np.linalg.norm(cohort, axis=1, keepdims=True)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cfg = GatingConfig(use_as_norm=True, as_norm_top_k=2)

    # Manually compute expected: cs_max = 1.0 (perfect match with pool[0])
    # impostor scores against e1 = [0.707..., 0.0, 0.0]; top-2 = [0.707, 0.0]
    impostor_scores = cohort @ emb
    top = np.partition(impostor_scores, -2)[-2:]
    expected = (1.0 - top.mean()) / top.std()
    out = target_score_as_norm(emb, pool, cohort, cfg)
    assert out == pytest.approx(float(expected), abs=1e-5)


def test_load_cohort_renormalises(tmp_path):
    """Cohort vectors are re-normalised on load even if the .npz held
    un-normalised inputs."""
    raw = np.array(
        [
            [3.0, 0.0, 0.0],  # |v|=3 → expected unit [1,0,0]
            [0.0, 4.0, 0.0],  # |v|=4 → expected unit [0,1,0]
        ],
        dtype=np.float32,
    )
    out_path = tmp_path / "cohort.npz"
    np.savez(
        out_path,
        embeddings=raw,
        languages=np.asarray(["en", "de"], dtype=object),
        speaker_ids=np.asarray(["A", "B"], dtype=object),
    )
    loaded = load_cohort(out_path)
    norms = np.linalg.norm(loaded, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert loaded.shape == (2, 3)


def test_load_cohort_handles_zero_row(tmp_path):
    """A degenerate all-zero row should not blow up (norms get clamped to 1)."""
    raw = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    out_path = tmp_path / "cohort.npz"
    np.savez(
        out_path,
        embeddings=raw,
        languages=np.asarray(["en", "en"], dtype=object),
        speaker_ids=np.asarray(["zero", "v"], dtype=object),
    )
    loaded = load_cohort(out_path)
    # First row stays zero; second row becomes [0.6, 0.8].
    assert np.allclose(loaded[0], 0.0)
    assert np.allclose(np.linalg.norm(loaded[1]), 1.0, atol=1e-5)


def test_gate_state_uses_as_norm_threshold_when_enabled():
    """When use_as_norm=True the GateState compares against
    theta_pass_as_norm rather than theta_pass."""
    cfg = GatingConfig(
        use_as_norm=True,
        theta_pass_as_norm=2.0,
        theta_learn_as_norm=3.0,
    )
    state = GateState(config=cfg)
    # 1.5 < theta_pass_as_norm=2.0 → must mute
    assert state.update(score=1.5, dt_ms=10.0) is False
    # 2.5 >= 2.0 → must pass
    assert state.update(score=2.5, dt_ms=10.0) is True


def test_gate_state_legacy_threshold_still_used_when_disabled():
    cfg = GatingConfig(use_as_norm=False, theta_pass=0.30)
    state = GateState(config=cfg)
    # 0.20 < 0.30 → mute
    assert state.update(score=0.20, dt_ms=10.0) is False
    # 0.40 >= 0.30 → pass
    assert state.update(score=0.40, dt_ms=10.0) is True


def test_should_admit_auto_learn_uses_as_norm_threshold():
    cfg = GatingConfig(
        use_as_norm=True,
        theta_pass_as_norm=1.0,
        theta_learn_as_norm=3.0,
        theta_f0=0.5,
        min_continuous_speech_sec=0.1,
    )
    # Score above theta_learn_as_norm + good F0 + long enough run → admit
    assert should_admit_auto_learn(
        score=3.5, f0_match_value=0.9, continuous_speech_ms=200.0, config=cfg
    )
    # Score below theta_learn_as_norm → reject
    assert not should_admit_auto_learn(
        score=2.5, f0_match_value=0.9, continuous_speech_ms=200.0, config=cfg
    )


def test_gating_config_rejects_bad_as_norm_threshold_order():
    with pytest.raises(ValueError):
        GatingConfig(theta_pass_as_norm=3.0, theta_learn_as_norm=2.0)


def test_gating_config_rejects_zero_top_k():
    with pytest.raises(ValueError):
        GatingConfig(as_norm_top_k=0)
