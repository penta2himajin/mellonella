"""Tests for training utilities (LR scheduler, enrollment-embedding device pick)."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from tse.train import (
    ExponentialMovingAverage,
    MuonHybrid,
    _build_optimizer,
    _build_parser,
    _build_scheduler,
    _newton_schulz_orthogonalize,
    _resolve_amp_dtype,
    train_epoch,
)


def _w(linear: nn.Module) -> torch.Tensor:
    """Typed accessor for ``Linear.weight`` (mypy treats it as Tensor | Module)."""
    return cast(nn.Linear, linear).weight


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


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 4)


def test_build_optimizer_adam() -> None:
    opt = _build_optimizer(_tiny_model(), "adam", lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.Adam)
    assert not isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_adamw_carries_weight_decay() -> None:
    opt = _build_optimizer(_tiny_model(), "adamw", lr=1e-3, weight_decay=0.05)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)


def test_build_optimizer_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown optimizer"):
        _build_optimizer(_tiny_model(), "lookahead", lr=1e-3, weight_decay=0.0)


# ---------------------------------------------------------------------------
# Warmup scheduler
# ---------------------------------------------------------------------------


def test_scheduler_warmup_then_cosine_ramps_up_then_anneals() -> None:
    opt = _opt(lr=0.01)
    sched = _build_scheduler(opt, "cosine", epochs=10, warmup_epochs=3, min_lr_ratio=0.01)
    assert sched is not None
    # Warmup starts at lr * start_factor (0.01)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01 * 0.01)
    # After warmup (3 steps), LR should be back at the peak.
    for _ in range(3):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    # Then anneal — final LR should approach eta_min.
    for _ in range(7):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01 * 0.01, rel=1e-3)


def test_scheduler_warmup_only_when_schedule_is_none() -> None:
    opt = _opt(lr=0.01)
    sched = _build_scheduler(opt, "none", epochs=10, warmup_epochs=4)
    assert sched is not None
    assert isinstance(sched, torch.optim.lr_scheduler.LinearLR)


# ---------------------------------------------------------------------------
# AMP dtype resolver
# ---------------------------------------------------------------------------


def test_resolve_amp_off_returns_none() -> None:
    assert _resolve_amp_dtype("off", "cuda") is None
    assert _resolve_amp_dtype("off", "cpu") is None


def test_resolve_amp_auto_cuda_fp16() -> None:
    assert _resolve_amp_dtype("auto", "cuda") is torch.float16


def test_resolve_amp_auto_cpu_returns_none() -> None:
    assert _resolve_amp_dtype("auto", "cpu") is None


def test_resolve_amp_on_cpu_bf16() -> None:
    assert _resolve_amp_dtype("on", "cpu") is torch.bfloat16


def test_resolve_amp_on_cuda_fp16() -> None:
    assert _resolve_amp_dtype("on", "cuda") is torch.float16


# ---------------------------------------------------------------------------
# Exponential moving average
# ---------------------------------------------------------------------------


def test_ema_decay_one_keeps_initial_weights() -> None:
    model = _tiny_model()
    initial = _w(model).detach().clone()
    ema = ExponentialMovingAverage(model, decay=0.999)
    # Mutate the live model — EMA shadow should barely move at decay=0.999.
    with torch.no_grad():
        _w(model).add_(torch.ones_like(_w(model)))
    ema.update(model)
    # Shadow shifted by (1 - 0.999) = 0.001 in the direction of the new weights.
    expected = initial + (1.0 - 0.999) * 1.0
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)


def test_ema_decay_zero_tracks_live_weights() -> None:
    model = _tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.0)
    with torch.no_grad():
        _w(model).fill_(42.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], _w(model))


def test_ema_invalid_decay_raises() -> None:
    with pytest.raises(ValueError, match="decay"):
        ExponentialMovingAverage(_tiny_model(), decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        ExponentialMovingAverage(_tiny_model(), decay=-0.1)


def test_ema_state_dict_roundtrips() -> None:
    model = _tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        _w(model).fill_(3.0)
    ema.update(model)
    saved = ema.state_dict()
    # Mutate shadow, then restore.
    with torch.no_grad():
        ema.shadow["weight"].fill_(0.0)
    ema.load_state_dict(saved)
    assert torch.allclose(ema.shadow["weight"], saved["weight"])


# ---------------------------------------------------------------------------
# Muon (Newton-Schulz + hybrid)
# ---------------------------------------------------------------------------


def test_newton_schulz_singular_values_in_quintic_envelope() -> None:
    """The quintic NS iteration (Keller Jordan / Muon) produces *approximate*
    orthogonalisation — the output's singular values land in roughly
    ``[0.5, 1.5]`` rather than exactly 1. This is by design (the coefficients
    maximise slope-at-zero, not exact convergence to 1) and turns out not
    to hurt model performance in published Muon results.
    """
    torch.manual_seed(0)
    g = torch.randn(8, 16)
    o = _newton_schulz_orthogonalize(g, steps=8)
    assert o.shape == g.shape
    sigma = torch.linalg.svdvals(o)
    # Looser bounds than the docstring's nominal [0.5, 1.5] to absorb tail
    # cases; the key property is "no singular value blows up or vanishes".
    assert sigma.min() > 0.3, f"NS collapsed a direction: σ_min={sigma.min().item():.4f}"
    assert sigma.max() < 2.0, f"NS exploded: σ_max={sigma.max().item():.4f}"


def test_newton_schulz_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        _newton_schulz_orthogonalize(torch.zeros(3, 4, 5))


def test_muon_hybrid_routes_matrix_vs_1d_params_correctly() -> None:
    """Linear weights → Muon, 1×1 conv → Muon, depthwise conv + biases → AdamW."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=True)
            self.pointwise = nn.Conv1d(4, 4, kernel_size=1)
            self.depthwise = nn.Conv1d(4, 4, kernel_size=3, groups=4)
            self.gamma = nn.Parameter(torch.ones(4))  # 1-D scale

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    model = Tiny()
    opt = MuonHybrid(model, lr=0.02, adamw_lr=1e-3, weight_decay=0.0)
    muon_ids = opt._muon_param_ids
    assert id(model.linear.weight) in muon_ids
    assert id(model.pointwise.weight) in muon_ids
    assert id(model.depthwise.weight) not in muon_ids  # depthwise → AdamW
    assert id(model.linear.bias) not in muon_ids  # bias → AdamW
    assert id(model.gamma) not in muon_ids  # 1-D scale → AdamW
    # Two groups, kinds == ['muon', 'adamw']
    assert [g["kind"] for g in opt.param_groups] == ["muon", "adamw"]


def test_muon_hybrid_step_updates_both_groups() -> None:
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return self.linear(x)

    model = Tiny()
    opt = MuonHybrid(model, lr=0.02, adamw_lr=1e-3, weight_decay=0.0)
    w0 = model.linear.weight.detach().clone()
    b0 = model.linear.bias.detach().clone()
    # Inject gradients and step.
    model.linear.weight.grad = torch.randn_like(model.linear.weight)
    model.linear.bias.grad = torch.ones_like(model.linear.bias)
    opt.step()
    assert not torch.allclose(model.linear.weight.detach(), w0)
    assert not torch.allclose(model.linear.bias.detach(), b0)


def test_build_optimizer_muon() -> None:
    opt = _build_optimizer(_tiny_model(), "muon", lr=0.02, weight_decay=0.0)
    assert isinstance(opt, MuonHybrid)


def test_build_optimizer_adamw_fused_works_on_cpu() -> None:
    # On CPU, fused=False is auto-selected; the call should not raise.
    opt = _build_optimizer(_tiny_model(), "adamw-fused", lr=1e-3, weight_decay=0.01)
    assert isinstance(opt, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# Gradient-norm clipping plumbing (--clip-grad-norm)
# ---------------------------------------------------------------------------


def test_train_epoch_clip_grad_norm_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """train_epoch must pass its ``clip_grad_norm`` arg to ``clip_grad_norm_``.

    The default is ``5.0`` (preserving the old v3/v4 behaviour); a CLI
    setting like ``--clip-grad-norm 0.5`` is the standard fp16-AMP stability
    knob. We capture every call into ``torch.nn.utils.clip_grad_norm_`` and
    assert the max-norm threshold it received matches what we passed.
    """
    from tse.config import TSEConfig
    from tse.data import synthetic_fixture_dataset
    from tse.model import CausalConvTasNetTSE

    config = TSEConfig.poc_16k()
    model = CausalConvTasNetTSE(config)
    ds = synthetic_fixture_dataset(n=2, sample_rate=config.sample_rate, duration_sec=0.5, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    captured: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def _spy(params, max_norm, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(float(max_norm))
        return original(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy)

    train_epoch(model, loader, optimizer, "cpu", clip_grad_norm=0.5)

    assert captured, "clip_grad_norm_ was never called"
    assert all(c == pytest.approx(0.5) for c in captured)


def test_train_epoch_default_clip_grad_norm_is_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default clip_grad_norm preserves the v3/v4 PoC value of 5.0."""
    from tse.config import TSEConfig
    from tse.data import synthetic_fixture_dataset
    from tse.model import CausalConvTasNetTSE

    config = TSEConfig.poc_16k()
    model = CausalConvTasNetTSE(config)
    ds = synthetic_fixture_dataset(n=2, sample_rate=config.sample_rate, duration_sec=0.5, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    captured: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def _spy(params, max_norm, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(float(max_norm))
        return original(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy)

    # Omit clip_grad_norm — exercise the default branch.
    train_epoch(model, loader, optimizer, "cpu")

    assert captured
    assert all(c == pytest.approx(5.0) for c in captured)


def test_cli_clip_grad_norm_flag() -> None:
    """The new --clip-grad-norm flag is wired into the argparser."""
    parser = _build_parser()
    args = parser.parse_args(["--clip-grad-norm", "0.5"])
    assert args.clip_grad_norm == pytest.approx(0.5)
    # Default preserves v3/v4 behaviour.
    args_default = parser.parse_args([])
    assert args_default.clip_grad_norm == pytest.approx(5.0)
