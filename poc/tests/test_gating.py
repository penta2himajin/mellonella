"""Tests for the pure gating logic. No model dependencies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mellonella_poc.config import GatingConfig
from mellonella_poc.gating import (
    EnvelopeState,
    GateState,
    apply_envelope,
    cos_sim_max,
    cos_similarity,
    f0_match,
    target_score,
)


def test_cos_similarity_basic():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cos_similarity(a, b) == pytest.approx(1.0)
    c = np.array([0.0, 1.0, 0.0])
    assert cos_similarity(a, c) == pytest.approx(0.0)
    assert cos_similarity(a, -a) == pytest.approx(-1.0)


def test_cos_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cos_similarity(a, b) == 0.0


def test_cos_sim_max_picks_best():
    target = np.array([1.0, 0.0])
    pool = [np.array([0.5, 0.5]), np.array([0.99, 0.01]), np.array([0.0, 1.0])]
    best = cos_sim_max(target, pool)
    assert 0.99 < best <= 1.0


def test_cos_sim_max_empty_pool():
    assert cos_sim_max(np.array([1.0]), []) == 0.0


def test_f0_match_gaussian():
    assert f0_match(120.0, 120.0, 10.0) == pytest.approx(1.0)
    decay_one_sigma = f0_match(130.0, 120.0, 10.0)
    assert decay_one_sigma == pytest.approx(math.exp(-0.5))
    assert f0_match(180.0, 120.0, 10.0) < 1e-3


def test_f0_match_neutral_when_sigma_zero():
    assert f0_match(120.0, 120.0, 0.0) == 1.0
    assert f0_match(0.0, 120.0, 0.0) == 1.0


def test_target_score_weights():
    cfg = GatingConfig(alpha=0.8, beta=0.2)
    pool = [np.array([1.0, 0.0])]
    score_match = target_score(np.array([1.0, 0.0]), pool, 120.0, 120.0, 10.0, cfg)
    assert score_match == pytest.approx(0.8 * 1.0 + 0.2 * 1.0)


def test_target_score_orthogonal_embedding():
    cfg = GatingConfig()
    pool = [np.array([1.0, 0.0])]
    score = target_score(np.array([0.0, 1.0]), pool, 120.0, 120.0, 10.0, cfg)
    assert score == pytest.approx(cfg.beta)


def test_gate_state_pass_above_threshold():
    cfg = GatingConfig()
    gate = GateState(config=cfg)
    assert gate.update(score=0.9, dt_ms=20.0) is True
    assert gate.is_on


def test_gate_state_hangover_keeps_pass():
    cfg = GatingConfig(hangover_ms=200.0)
    gate = GateState(config=cfg)
    gate.update(score=0.9, dt_ms=20.0)
    assert gate.update(score=0.0, dt_ms=100.0) is True
    assert gate.update(score=0.0, dt_ms=99.0) is True
    assert gate.update(score=0.0, dt_ms=10.0) is False


def test_gate_state_no_hangover_when_off():
    cfg = GatingConfig()
    gate = GateState(config=cfg)
    assert gate.update(score=0.0, dt_ms=20.0) is False
    assert gate.update(score=0.0, dt_ms=20.0) is False


def test_envelope_attack_release_monotonic():
    cfg = GatingConfig(attack_ms=15.0, release_ms=100.0)
    sr = 48_000
    env = EnvelopeState(config=cfg, sample_rate=sr)
    n_attack = int(0.1 * sr)
    on = env.advance(target_on=True, n_samples=n_attack)
    assert on[0] < on[-1]
    assert on[-1] > 0.95
    n_release = int(0.5 * sr)
    off = env.advance(target_on=False, n_samples=n_release)
    assert off[0] > off[-1]
    assert off[-1] < 0.05


def test_envelope_attack_faster_than_release():
    cfg = GatingConfig(attack_ms=15.0, release_ms=100.0)
    sr = 48_000
    env_a = EnvelopeState(config=cfg, sample_rate=sr)
    env_b = EnvelopeState(config=cfg, sample_rate=sr)
    attack_curve = env_a.advance(True, sr)
    env_b.value = 1.0
    release_curve = env_b.advance(False, sr)
    samples_to_half_attack = int(np.argmax(attack_curve >= 0.5))
    samples_to_half_release = int(np.argmax(release_curve <= 0.5))
    assert samples_to_half_attack > 0
    assert samples_to_half_release > 0
    assert samples_to_half_attack < samples_to_half_release


def test_apply_envelope_alignment():
    cfg = GatingConfig(attack_ms=15.0, release_ms=100.0)
    sr = 48_000
    audio = np.ones(sr, dtype=np.float32)
    decisions = [(0, True), (sr // 2, False)]
    out = apply_envelope(audio, decisions, sample_rate=sr, config=cfg)
    assert out.shape == audio.shape
    mid = sr // 2
    assert out[mid - 100] > 0.95
    assert out[-100] < 0.05


def test_apply_envelope_requires_zero_start():
    cfg = GatingConfig()
    audio = np.ones(10, dtype=np.float32)
    with pytest.raises(ValueError):
        apply_envelope(audio, [(1, True)], sample_rate=48_000, config=cfg)


def test_gating_config_validates_threshold_order():
    with pytest.raises(ValueError):
        GatingConfig(theta_pass=0.9, theta_learn=0.5)


def test_gating_config_validates_weight_sum():
    with pytest.raises(ValueError):
        GatingConfig(alpha=0.5, beta=0.3)


def test_should_admit_auto_learn_all_pass():
    from mellonella_poc.gating import should_admit_auto_learn

    cfg = GatingConfig(theta_learn=0.80, theta_f0=0.7, min_continuous_speech_sec=1.0)
    assert should_admit_auto_learn(0.85, 0.75, 1500.0, cfg)


def test_should_admit_auto_learn_low_score_blocked():
    from mellonella_poc.gating import should_admit_auto_learn

    cfg = GatingConfig()
    assert not should_admit_auto_learn(0.79, 0.9, 2000.0, cfg)


def test_should_admit_auto_learn_low_f0_match_blocked():
    from mellonella_poc.gating import should_admit_auto_learn

    cfg = GatingConfig(theta_learn=0.80, theta_f0=0.7)
    assert not should_admit_auto_learn(0.85, 0.5, 2000.0, cfg)


def test_should_admit_auto_learn_short_run_blocked():
    from mellonella_poc.gating import should_admit_auto_learn

    cfg = GatingConfig(min_continuous_speech_sec=1.0)
    assert not should_admit_auto_learn(0.95, 0.9, 500.0, cfg)


def test_should_admit_auto_learn_boundary_inclusive():
    from mellonella_poc.gating import should_admit_auto_learn

    cfg = GatingConfig(theta_learn=0.80, theta_f0=0.7, min_continuous_speech_sec=1.0)
    # exactly-at-threshold passes
    assert should_admit_auto_learn(0.80, 0.7, 1000.0, cfg)
    assert not should_admit_auto_learn(0.799, 0.7, 1000.0, cfg)
    assert not should_admit_auto_learn(0.80, 0.699, 1000.0, cfg)
    assert not should_admit_auto_learn(0.80, 0.7, 999.0, cfg)
