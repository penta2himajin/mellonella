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
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: CausalConvTasNetTSE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": vars(model.config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CausalConvTasNetTSE,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
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
    model: CausalConvTasNetTSE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_sisdr = 0.0
    n = 0
    for mixture, cond, target in loader:
        mixture = mixture.to(device)
        cond = cond.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        est = model(mixture, cond)
        loss = neg_si_sdr_loss(est, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = mixture.shape[0]
        total_loss += float(loss.item()) * bs
        with torch.no_grad():
            total_sisdr += float(si_sdr(target, est).sum().item())
        n += bs
    return {"loss": total_loss / max(n, 1), "si_sdr": total_sisdr / max(n, 1)}


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: str,
    epochs: int,
    *,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a per-epoch LR scheduler.

    ``schedule`` is one of:

    * ``"none"`` — no scheduling, returns ``None``.
    * ``"cosine"`` — cosine anneal from ``lr`` to ``lr * min_lr_ratio`` over
      ``epochs`` steps. Matches the warmup-free standard for short-to-medium
      training runs.
    * ``"step"`` — halve the LR every ``max(epochs // 3, 1)`` epochs.

    The scheduler is stepped once per epoch by :func:`run_training`.
    """
    if schedule == "none":
        return None
    if schedule == "cosine":
        eta_min = optimizer.param_groups[0]["lr"] * min_lr_ratio
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    if schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)
    raise ValueError(f"unknown lr schedule: {schedule!r}")


def run_training(
    config: TSEConfig,
    dataset: TSEMixtureDataset,
    out_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    lr_schedule: str = "none",
    device: str = "cpu",
    resume: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Run the full training loop, writing checkpoints + ``metrics.json``."""
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CausalConvTasNetTSE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _build_scheduler(optimizer, lr_schedule, epochs)

    start_epoch = 0
    global_step = 0
    if resume:
        latest = _latest_checkpoint(out_dir)
        if latest is not None:
            start_epoch, global_step = load_checkpoint(latest, model, optimizer)
            print(f"[train] resumed from {latest} (epoch {start_epoch})", file=sys.stderr)
            if scheduler is not None:
                # Advance the schedule to match the resumed epoch.
                for _ in range(start_epoch):
                    scheduler.step()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    print(
        f"[train] params={count_parameters(model):,}  epochs={epochs}  "
        f"batch={batch_size}  lr={lr}  schedule={lr_schedule}  device={device}",
        file=sys.stderr,
    )
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs):
        t0 = time.perf_counter()
        stats = train_epoch(model, loader, optimizer, device)
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
        save_checkpoint(out_dir / f"ckpt_epoch{epoch:04d}.pt", model, optimizer, epoch, global_step)
        if scheduler is not None:
            scheduler.step()

    metrics = {
        "config": vars(config),
        "params": count_parameters(model),
        "history": history,
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
        device=args.device,
        resume=args.resume,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
