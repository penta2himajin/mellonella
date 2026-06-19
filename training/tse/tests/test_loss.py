"""Tests for the composite TSE training loss.

Covers the multi-resolution STFT term, the mixture-consistency penalty, and
the `composite_loss` wrapper that combines them with SI-SDR. The headline
contract is that `composite_loss` with both extra weights zero is *exactly*
`neg_si_sdr_loss`, so enabling the new terms is opt-in and existing runs are
byte-identical.
"""

from __future__ import annotations

import pytest
import torch

from tse.loss import (
    composite_loss,
    mixture_consistency_loss,
    multi_resolution_stft_loss,
    neg_si_sdr_loss,
)


def _signals(seed: int = 0, batch: int = 3, length: int = 8000):
    torch.manual_seed(seed)
    target = torch.randn(batch, length)
    mixture = target + 0.5 * torch.randn(batch, length)
    estimate = target + 0.1 * torch.randn(batch, length)
    return estimate, target, mixture


def test_composite_zero_weights_equals_neg_si_sdr() -> None:
    est, tgt, mix = _signals()
    a = neg_si_sdr_loss(est, tgt)
    b = composite_loss(est, tgt, mix, mr_stft_weight=0.0, mix_consist_weight=0.0)
    assert torch.allclose(a, b, atol=0.0, rtol=0.0)


def test_mr_stft_is_zero_for_identical_signals() -> None:
    _, tgt, _ = _signals()
    # Perfect estimate → spectral convergence 0, log-mag L1 0.
    loss = multi_resolution_stft_loss(tgt, tgt)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_mr_stft_positive_and_increases_with_error() -> None:
    _, tgt, _ = _signals()
    small = multi_resolution_stft_loss(tgt + 0.01 * torch.randn_like(tgt), tgt)
    large = multi_resolution_stft_loss(tgt + 0.5 * torch.randn_like(tgt), tgt)
    assert small.item() > 0.0
    assert large.item() > small.item()


def test_mixture_consistency_zero_when_residual_orthogonal() -> None:
    # Construct estimate and residual that are exactly orthogonal: estimate is
    # a cosine, residual is a sine at the same frequency (zero inner product
    # over an integer number of periods).
    n = 8000
    t = torch.arange(n, dtype=torch.float32)
    est = torch.cos(2 * torch.pi * 8 * t / n).unsqueeze(0)
    residual = torch.sin(2 * torch.pi * 8 * t / n).unsqueeze(0)
    mixture = est + residual
    loss = mixture_consistency_loss(est, mixture)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_mixture_consistency_penalises_correlated_residual() -> None:
    # Residual is a scaled copy of the estimate → maximal correlation → loss
    # near 1.0.
    n = 8000
    est = torch.randn(1, n)
    mixture = est + 0.5 * est  # residual = 0.5 * est, perfectly correlated
    loss = mixture_consistency_loss(est, mixture)
    assert loss.item() == pytest.approx(1.0, abs=1e-3)


def test_composite_gradients_flow() -> None:
    est, tgt, mix = _signals()
    est = est.detach().requires_grad_(True)
    loss = composite_loss(est, tgt, mix, mr_stft_weight=0.3, mix_consist_weight=0.1)
    loss.backward()
    assert est.grad is not None
    assert torch.isfinite(est.grad).all()
    assert est.grad.abs().sum() > 0.0


def test_composite_requires_mixture_when_consistency_enabled() -> None:
    est, tgt, _ = _signals()
    with pytest.raises(ValueError, match="mixture is None"):
        composite_loss(est, tgt, None, mix_consist_weight=0.1)


def test_shape_mismatch_raises() -> None:
    a = torch.randn(2, 100)
    b = torch.randn(2, 200)
    with pytest.raises(ValueError, match="shape mismatch"):
        multi_resolution_stft_loss(a, b)
    with pytest.raises(ValueError, match="shape mismatch"):
        mixture_consistency_loss(a, b)
