"""Training loop for the causal Conv-TasNet TSE model.

Runs in two modes from the same code path:

* **overfit** (``--overfit-steps N``): repeatedly fit a *single* synthetic
  mixture for N steps. This is the local-CPU sanity gate — the SI-SDR loss
  must drop substantially toward its floor. No datasets needed.
* **full** (``--epochs E``): the real training loop over a
  :class:`~tse.data.TSEMixtureDataset`. On Kaggle (Phase 3) this is pointed
  at LibriSpeech/LibriMix + MUSAN via ``--data-dir``; the structure here is
  identical, only the source list changes.

Both modes write per-epoch checkpoints and a ``metrics.json`` under
``--out``. ``--resume`` continues from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TSEConfig
from .data import TSEMixtureDataset, synthetic_fixture_dataset
from .loss import neg_si_sdr_loss, si_sdr
from .model import CausalConvTasNetTSE, count_parameters


class OverfitResult(TypedDict):
    """Return type of :func:`overfit_one_batch`."""

    loss_curve: list[float]
    start_loss: float
    end_loss: float
    final_si_sdr: float


# ---------------------------------------------------------------------------
# Exponential moving average of weights
# ---------------------------------------------------------------------------


class ExponentialMovingAverage:
    """Shadow copy of model parameters updated each step.

    EMA weights frequently match or beat the raw final weights at zero
    additional training cost. Updated as
    ``ema = decay * ema + (1 - decay) * param`` after every optimizer
    step. ``decay`` is held in ``[0, 1)`` — closer to 1 means slower
    averaging. The shadow tensors live on the same device as the model.

    Saved alongside the model in checkpoints; downstream inference and
    ONNX export should load the EMA snapshot.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: t.detach().cpu() for name, t in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, t in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(t.to(self.shadow[name].device))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: CausalConvTasNetTSE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    *,
    ema: ExponentialMovingAverage | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": vars(model.config),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
        payload["ema_decay"] = ema.decay
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: CausalConvTasNetTSE,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0))


def _latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Overfit-one-batch mode (local sanity gate)
# ---------------------------------------------------------------------------


def overfit_one_batch(
    model: CausalConvTasNetTSE,
    *,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 25,
    seed: int = 0,
) -> OverfitResult:
    """Overfit a single synthetic mixture for ``steps`` steps.

    Returns a dict with the loss curve and start/end SI-SDR — the smoke
    test asserts the loss drops substantially.
    """
    torch.manual_seed(seed)
    model = model.to(device).train()
    ds = synthetic_fixture_dataset(
        n=1, sample_rate=model.config.sample_rate, duration_sec=1.0, seed=seed
    )
    mixture, cond, target = ds[0]
    mixture = mixture.unsqueeze(0).to(device)
    cond = cond.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curve: list[float] = []
    for step in range(steps):
        optimizer.zero_grad()
        est = model(mixture, cond)
        loss = neg_si_sdr_loss(est, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        curve.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[overfit] step {step:4d}  loss {loss.item():+.4f} dB", file=sys.stderr)

    model.eval()
    with torch.no_grad():
        est = model(mixture, cond)
        final_sisdr = float(si_sdr(target, est).mean().item())
    return {
        "loss_curve": curve,
        "start_loss": curve[0],
        "end_loss": curve[-1],
        "final_si_sdr": final_sisdr,
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> dict[str, float]:
    """One training epoch.

    ``amp_dtype`` enables :func:`torch.autocast` for the forward + loss
    (``torch.float16`` or ``torch.bfloat16``). ``scaler`` is only needed
    with fp16 on CUDA — bf16 trains stably without loss scaling. ``ema``,
    when given, is updated after every successful optimizer step.

    Accumulators stay on the model's device; the per-batch ``.item()``
    sync that v1 paid is consolidated into a single sync at epoch end.
    """
    model.train()
    is_cuda = str(device).startswith("cuda")
    autocast_device = "cuda" if is_cuda else "cpu"
    total_loss = torch.zeros((), device=device)
    total_sisdr = torch.zeros((), device=device)
    n = 0
    for mixture, cond, target in loader:
        mixture = mixture.to(device, non_blocking=is_cuda)
        cond = cond.to(device, non_blocking=is_cuda)
        target = target.to(device, non_blocking=is_cuda)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            est = model(mixture, cond)
            loss = neg_si_sdr_loss(est, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = mixture.shape[0]
        total_loss = total_loss + loss.detach() * bs
        with torch.no_grad():
            total_sisdr = total_sisdr + si_sdr(target, est).sum()
        n += bs
    return {
        "loss": float((total_loss / max(n, 1)).item()),
        "si_sdr": float((total_sisdr / max(n, 1)).item()),
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
    *,
    min_lr_ratio: float = 0.01,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a per-epoch LR scheduler, optionally with linear warmup.

    ``schedule`` is one of:

    * ``"none"`` — no scheduling, returns ``None`` (or warmup-only when
      ``warmup_epochs > 0``).
    * ``"cosine"`` — cosine anneal from ``lr`` to ``lr * min_lr_ratio`` over
      ``epochs - warmup_epochs`` steps.
    * ``"step"`` — halve the LR every ``max((epochs - warmup_epochs) // 3, 1)``
      epochs after warmup.

    With ``warmup_epochs > 0`` the LR linearly ramps from ``lr * 0.01`` to
    ``lr`` over those epochs, then switches to the chosen main schedule.
    Returns ``None`` only when both ``schedule == "none"`` and
    ``warmup_epochs == 0``.

    The scheduler is stepped once per epoch by :func:`run_training`.
    """
    main_epochs = max(epochs - warmup_epochs, 1)
    main: torch.optim.lr_scheduler.LRScheduler | None
    if schedule == "none":
        main = None
    elif schedule == "cosine":
        eta_min = optimizer.param_groups[0]["lr"] * min_lr_ratio
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=eta_min
        )
    elif schedule == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(main_epochs // 3, 1), gamma=0.5
        )
    else:
        raise ValueError(f"unknown lr schedule: {schedule!r}")

    if warmup_epochs <= 0:
        return main
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    if main is None:
        return warmup
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, main], milestones=[warmup_epochs]
    )


def _newton_schulz_orthogonalize(
    g: torch.Tensor, *, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the orthogonal factor of ``g`` via the quintic Newton-Schulz iteration.

    Returns a matrix ``U V^T`` (where ``g = U Σ V^T`` is the SVD) without
    the singular values. Coefficients ``(a, b, c) = (3.4445, -4.7750,
    2.0315)`` are the standard quintic from Bernstein & Newhouse / Keller
    Jordan's Muon.

    The iteration is run in ``bfloat16`` to match Keller Jordan's reference
    implementation — the quintic is range-bounded so bf16 precision is
    sufficient, and on modern Xeons / GPUs the matmul throughput is ~2x
    fp32. The result is cast back to ``g.dtype`` before return.
    """
    if g.ndim != 2:
        raise ValueError(f"_newton_schulz_orthogonalize requires 2D input, got {g.ndim}D")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.detach().to(torch.bfloat16)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.T
        transposed = True
    x = x / (x.norm() + eps)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class MuonHybrid(torch.optim.Optimizer):
    """Muon (Newton-Schulz orthogonalisation) for 2D weight matrices, AdamW for the rest.

    Following Keller Jordan's 2024 recipe: 2D-or-higher ``nn.Linear``-like
    weights have their momentum-smoothed gradient *orthogonalised* before
    application, producing a faster-converging update direction. 1-D
    parameters (norm scale/bias, PReLU, embeddings), depthwise conv
    weights, and biases stay with AdamW.

    The split is decided at construction time by inspecting the model's
    modules:

    * ``nn.Linear.weight`` → Muon.
    * ``nn.Conv1d.weight`` / ``nn.ConvTranspose1d.weight`` with
      ``groups == 1`` → Muon. Depthwise / grouped convs (groups > 1) →
      AdamW. (Newton-Schulz on a stack of independent 1-D filters has no
      meaning.)
    * Everything else (biases, 1-D params) → AdamW.

    Untested for Conv-TasNet TSE — the published evidence is on
    transformers. Use as an experiment, not a default.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 0.02,
        adamw_lr: float = 1e-3,
        momentum: float = 0.95,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        muon_ids, muon_params, adamw_params = self._split_params(model)
        if not muon_params:
            raise ValueError("MuonHybrid: no matrix params found — use AdamW directly")
        param_groups = [
            {
                "params": muon_params,
                "lr": lr,
                "kind": "muon",
                "momentum": momentum,
                "weight_decay": weight_decay,
                "ns_steps": ns_steps,
            },
            {
                "params": adamw_params,
                "lr": adamw_lr,
                "kind": "adamw",
                "betas": adamw_betas,
                "weight_decay": weight_decay,
                "eps": 1e-8,
            },
        ]
        super().__init__(param_groups, defaults={})
        # Stash the ids so test code can verify the routing without poking state.
        self._muon_param_ids: set[int] = muon_ids

    @staticmethod
    def _split_params(
        model: nn.Module,
    ) -> tuple[set[int], list[torch.Tensor], list[torch.Tensor]]:
        muon_ids: set[int] = set()
        for mod in model.modules():
            # Linear → matrix; Conv1d/ConvTranspose1d with groups==1 → matrix-like
            # (depthwise grouped convs route to AdamW — orthogonalising a stack
            # of independent 1-D filters has no meaning).
            is_matrix = isinstance(mod, nn.Linear) or (
                isinstance(mod, nn.Conv1d | nn.ConvTranspose1d) and mod.groups == 1
            )
            if is_matrix and mod.weight.requires_grad:
                muon_ids.add(id(mod.weight))
        muon_params: list[torch.Tensor] = []
        adamw_params: list[torch.Tensor] = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (muon_params if id(p) in muon_ids else adamw_params).append(p)
        return muon_ids, muon_params, adamw_params

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]  # noqa: ANN001, ANN201
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            kind = group["kind"]
            lr = group["lr"]
            wd = group["weight_decay"]
            if kind == "muon":
                mom = group["momentum"]
                ns = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(p.grad)
                    update = p.grad.add(buf, alpha=mom)  # Nesterov-style lookahead
                    orig_shape = update.shape
                    update_2d = update.reshape(orig_shape[0], -1)
                    update_2d = _newton_schulz_orthogonalize(update_2d, steps=ns)
                    # Preserve the per-row scale: the orthogonal matrix has
                    # ~unit norm per row, so scale by sqrt(fan_out / fan_in).
                    scale = max(1.0, orig_shape[0] / max(update_2d.size(1), 1)) ** 0.5
                    update = (update_2d * scale).reshape(orig_shape)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)
            elif kind == "adamw":
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    m_, v_ = state["exp_avg"], state["exp_avg_sq"]
                    g = p.grad
                    m_.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v_.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]
                    step_size = lr / bc1
                    denom = (v_.sqrt() / (bc2**0.5)).add_(eps)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(m_, denom, value=-step_size)
            else:  # pragma: no cover - constructor restricts kind
                raise RuntimeError(f"unknown param-group kind: {kind!r}")
        return loss


def _build_optimizer(
    model: nn.Module, name: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Construct the optimizer by name.

    Supported: ``adam`` | ``adamw`` | ``adamw-fused`` (CUDA fused kernels;
    falls back to non-fused on CPU) | ``muon`` (experimental hybrid; uses
    ``lr`` for the Muon group and a fixed ``adamw_lr = lr / 20`` for the
    1-D / depthwise AdamW group, mirroring the Muon paper's recipe).
    """
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw-fused":
        on_cuda = any(p.is_cuda for p in model.parameters())
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, fused=on_cuda
        )
    if name == "muon":
        return MuonHybrid(model, lr=lr, adamw_lr=lr / 20.0, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name!r}")


def _resolve_amp_dtype(amp: str, device: str) -> torch.dtype | None:
    """Pick the autocast dtype.

    * ``amp == "off"`` → ``None`` (no autocast).
    * ``amp == "auto"`` (default) → ``torch.float16`` on CUDA, ``None`` on CPU.
      The CPU path is opt-in because bf16 autocast quality can vary by op
      coverage in older PyTorch builds.
    * ``amp == "on"`` → ``torch.float16`` on CUDA, ``torch.bfloat16`` on CPU
      (AVX-512 BF16 on modern Xeons gives a real win; opt-in).
    """
    if amp == "off":
        return None
    if str(device).startswith("cuda"):
        return torch.float16
    if amp == "on":
        return torch.bfloat16
    return None


def run_training(
    config: TSEConfig,
    dataset: TSEMixtureDataset,
    out_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    lr_schedule: str = "none",
    warmup_epochs: int = 0,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    ema_decay: float = 0.0,
    amp: str = "auto",
    compile_model: bool = False,
    device: str = "cpu",
    resume: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Run the full training loop, writing checkpoints + ``metrics.json``."""
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        # Fixed input shapes per batch — let cuDNN pick the best conv kernels.
        torch.backends.cudnn.benchmark = True

    model = CausalConvTasNetTSE(config).to(device)
    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = _build_scheduler(optimizer, lr_schedule, epochs, warmup_epochs=warmup_epochs)
    amp_dtype = _resolve_amp_dtype(amp, device)
    # GradScaler is only useful for fp16 on CUDA; bf16 (CPU or CUDA) trains
    # stably without loss scaling.
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 and is_cuda else None
    ema = ExponentialMovingAverage(model, decay=ema_decay) if ema_decay > 0.0 else None

    start_epoch = 0
    global_step = 0
    if resume:
        latest = _latest_checkpoint(out_dir)
        if latest is not None:
            start_epoch, global_step = load_checkpoint(latest, model, optimizer, ema=ema)
            print(f"[train] resumed from {latest} (epoch {start_epoch})", file=sys.stderr)
            if scheduler is not None:
                # Advance the schedule to match the resumed epoch.
                for _ in range(start_epoch):
                    scheduler.step()

    # torch.compile *after* state is restored. We keep the un-compiled
    # ``model`` reference for state_dict / EMA / save_checkpoint (the
    # compiled wrapper's keys are prefixed and don't round-trip cleanly);
    # ``forward_model`` is what the training loop actually calls.
    forward_model: nn.Module = model
    if compile_model:
        try:
            forward_model = torch.compile(model)  # type: ignore[assignment]
            print("[train] torch.compile enabled", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - depends on torch build
            print(f"[train] torch.compile unavailable ({exc!r}); continuing eager", file=sys.stderr)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=is_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    print(
        f"[train] params={count_parameters(model):,}  epochs={epochs}  "
        f"batch={batch_size}  lr={lr}  schedule={lr_schedule}+warmup{warmup_epochs}  "
        f"opt={optimizer_name}(wd={weight_decay})  amp={amp}({amp_dtype})  "
        f"ema={ema_decay if ema is not None else 'off'}  device={device}",
        file=sys.stderr,
    )
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        stats = train_epoch(
            forward_model,
            loader,
            optimizer,
            device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            ema=ema,
        )
        global_step += len(loader)
        elapsed = time.perf_counter() - t0
        stats["epoch"] = float(epoch)
        stats["elapsed_sec"] = elapsed
        stats["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(stats)
        print(
            f"[train] epoch {epoch:3d}  loss {stats['loss']:+.4f}  "
            f"si_sdr {stats['si_sdr']:+.4f} dB  lr={stats['lr']:.2e}  ({elapsed:.1f}s)",
            file=sys.stderr,
        )
        save_checkpoint(
            out_dir / f"ckpt_epoch{epoch:04d}.pt",
            model,
            optimizer,
            epoch,
            global_step,
            ema=ema,
        )
        if scheduler is not None:
            scheduler.step()

    metrics: dict[str, object] = {
        "config": vars(config),
        "params": count_parameters(model),
        "history": history,
        "options": {
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "lr_schedule": lr_schedule,
            "warmup_epochs": warmup_epochs,
            "ema_decay": ema_decay if ema is not None else 0.0,
            "amp": amp,
            "amp_dtype": str(amp_dtype),
            "compile": compile_model,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote {out_dir / 'metrics.json'}", file=sys.stderr)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        choices=("poc_16k", "prod_48k"),
        default="poc_16k",
        help="model config preset",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=0,
        help="if > 0, run overfit-one-batch sanity mode for this many steps",
    )
    parser.add_argument("--epochs", type=int, default=10, help="full-mode training epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine", "step"),
        default="none",
        help="per-epoch LR schedule (cosine anneals to lr*0.01 over `epochs`)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="linear LR warmup from lr*0.01 over the first N epochs",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "adamw-fused", "muon"),
        default="adam",
        help=(
            "adamw: decoupled WD; adamw-fused: CUDA-fused kernels; "
            "muon: experimental hybrid (Muon for 2D weights + AdamW elsewhere). "
            "muon untested for TSE — experimental."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay (adamw recommended when > 0)",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="if > 0, maintain EMA shadow weights (e.g. 0.999); saved alongside model",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "on", "off"),
        default="auto",
        help="autocast: auto = fp16 on CUDA / off on CPU, on = also enable bf16 on CPU",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="wrap the model in torch.compile (falls back silently if unsupported)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="root of LibriSpeech/MUSAN data (full mode). Omit to use the synthetic fixture set.",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        default=None,
        help=(
            "path to the .npz of frozen ECAPA enrollment embeddings produced "
            "by tse.prepare_enrollment_embeddings. Required for real training. "
            "Without it, librispeech_musan_sources falls back to a deterministic "
            "per-speaker placeholder vector (plumbing-only)."
        ),
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help="cap the number of (target, interferer) source items built from --data-dir",
    )
    parser.add_argument("--out", type=Path, default=Path("build/tse"), help="output directory")
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixture-n",
        type=int,
        default=16,
        help="number of synthetic fixture items when --data-dir is omitted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = TSEConfig.poc_16k() if args.config == "poc_16k" else TSEConfig.prod_48k()

    if args.overfit_steps > 0:
        model = CausalConvTasNetTSE(config)
        print(f"[train] overfit mode — params={count_parameters(model):,}", file=sys.stderr)
        result = overfit_one_batch(
            model, steps=args.overfit_steps, lr=args.lr, device=args.device, seed=args.seed
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "overfit_metrics.json").write_text(json.dumps(result, indent=2))
        print(
            f"[train] overfit done — start {result['start_loss']:+.4f} dB  "
            f"end {result['end_loss']:+.4f} dB  final SI-SDR {result['final_si_sdr']:+.4f} dB",
            file=sys.stderr,
        )
        return 0

    if args.data_dir is not None:
        from .data import librispeech_musan_sources

        sources = librispeech_musan_sources(
            args.data_dir,
            sample_rate=config.sample_rate,
            embeddings_npz=args.embeddings_npz,
            n_pairs=args.n_pairs,
            seed=args.seed,
        )
        print(f"[train] {len(sources)} source items from {args.data_dir}", file=sys.stderr)
        dataset = TSEMixtureDataset(
            sources,
            sample_rate=config.sample_rate,
            segment_samples=config.sample_rate,  # 1 s segments
            random_crop=True,
            seed=args.seed,
        )
    else:
        print(
            "[train] no --data-dir given; training on the synthetic fixture set "
            "(scaffold validation only — not a real model)",
            file=sys.stderr,
        )
        dataset = synthetic_fixture_dataset(
            n=args.fixture_n, sample_rate=config.sample_rate, duration_sec=1.0, seed=args.seed
        )

    run_training(
        config,
        dataset,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        amp=args.amp,
        compile_model=args.compile,
        device=args.device,
        resume=args.resume,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
