"""Tests for training utilities (LR scheduler, enrollment-embedding device pick)."""

from __future__ import annotations

import pytest
import torch

from tse.train import _build_scheduler


def _opt(lr: float = 0.01) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def test_build_scheduler_none() -> None:
    assert _build_scheduler(_opt(), "none", epochs=10) is None


def test_build_scheduler_cosine_anneals_to_min() -> None:
    opt = _opt(lr=0.01)
    sched = _build_scheduler(opt, "cosine", epochs=10, min_lr_ratio=0.01)
    assert sched is not None
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    for _ in range(10):
        sched.step()
    # CosineAnnealingLR ends at eta_min after T_max steps.
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01 * 0.01, rel=1e-3)


def test_build_scheduler_step_halves_third() -> None:
    opt = _opt(lr=0.01)
    sched = _build_scheduler(opt, "step", epochs=9)  # step_size = 3
    assert sched is not None
    # After 3 steps, LR halved
    for _ in range(3):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.005)
    # After 6 total steps, LR halved again
    for _ in range(3):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0025)


def test_build_scheduler_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown lr schedule"):
        _build_scheduler(_opt(), "bogus", epochs=10)


# ---------------------------------------------------------------------------
# Enrollment-embedding provider selection
# ---------------------------------------------------------------------------


def test_resolve_providers_cpu_explicit() -> None:
    pytest.importorskip("onnxruntime")
    from tse.prepare_enrollment_embeddings import _resolve_providers

    assert _resolve_providers("cpu") == ["CPUExecutionProvider"]


def test_resolve_providers_auto_falls_back_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    ort = pytest.importorskip("onnxruntime")
    from tse.prepare_enrollment_embeddings import _resolve_providers

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert _resolve_providers("auto") == ["CPUExecutionProvider"]


def test_resolve_providers_auto_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    from tse.prepare_enrollment_embeddings import _resolve_providers

    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert _resolve_providers("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_cuda_strict_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = pytest.importorskip("onnxruntime")
    from tse.prepare_enrollment_embeddings import _resolve_providers

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        _resolve_providers("cuda")
