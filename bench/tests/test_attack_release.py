"""Tests for attack/release time-constant fitting."""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_bench.metrics.attack_release import (
    fit_first_order,
    measure_attack_release_from_step,
)


def _exp_curve(
    initial: float, final: float, tau_ms: float, sr: int, duration_ms: float
) -> np.ndarray:
    t = np.arange(int(sr * duration_ms / 1000)) / sr
    return final + (initial - final) * np.exp(-t / (tau_ms / 1000.0))


@pytest.mark.parametrize("tau_ms", [10.0, 50.0, 200.0])
def test_fit_first_order_recovers_tau(tau_ms: float):
    sr = 48_000
    response = _exp_curve(initial=0.0, final=1.0, tau_ms=tau_ms, sr=sr, duration_ms=1000.0)
    fit = fit_first_order(response, sample_rate=sr, initial=0.0, final=1.0)
    assert fit.tau_ms == pytest.approx(tau_ms, rel=0.05)
    assert fit.rms_residual < 1e-3


def test_fit_release_curve():
    sr = 48_000
    response = _exp_curve(initial=1.0, final=0.0, tau_ms=100.0, sr=sr, duration_ms=1000.0)
    fit = fit_first_order(response, sample_rate=sr, initial=1.0, final=0.0)
    assert fit.tau_ms == pytest.approx(100.0, rel=0.05)


def test_measure_attack_release_from_step():
    sr = 48_000
    attack = _exp_curve(0.0, 1.0, tau_ms=15.0, sr=sr, duration_ms=200.0)
    release = _exp_curve(1.0, 0.0, tau_ms=100.0, sr=sr, duration_ms=500.0)
    envelope = np.concatenate([attack, release])
    a_fit, r_fit = measure_attack_release_from_step(envelope, attack.size, sample_rate=sr)
    assert a_fit.tau_ms == pytest.approx(15.0, rel=0.10)
    assert r_fit.tau_ms == pytest.approx(100.0, rel=0.10)


def test_fit_rejects_constant_signal():
    sr = 48_000
    flat = np.full(int(sr * 0.1), 0.5, dtype=np.float64)
    with pytest.raises(ValueError):
        fit_first_order(flat, sample_rate=sr, initial=0.5, final=0.5)


def test_fit_rejects_short_signal():
    with pytest.raises(ValueError):
        fit_first_order(np.array([0.0, 0.5, 1.0]), sample_rate=48_000)


def test_step_index_validation():
    envelope = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError):
        measure_attack_release_from_step(envelope, step_index=0, sample_rate=48_000)
    with pytest.raises(ValueError):
        measure_attack_release_from_step(envelope, step_index=99, sample_rate=48_000)
