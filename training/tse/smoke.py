"""Local-CPU validation gate for the Stage C TSE scaffold.

Run as ``python training/tse/smoke.py`` (from the repo root) or
``python -m tse.smoke`` (with ``training`` installed / on the path).
Exits non-zero on any failure.

Four checks:

1. **forward+backward** — instantiate the model, print the parameter count,
   run one forward + backward on a synthetic batch, assert the loss is
   finite and gradients flow.
2. **full == streaming** — assert the full-sequence forward and the
   chunked streaming forward produce the same output (<= 1e-4).
3. **overfit** — overfit ONE synthetic mixture for ~200 steps and assert the
   negative-SI-SDR loss drops substantially toward its floor.
4. **ONNX round-trip** — export the (untrained) model to ONNX via
   ``scripts/export_tse_onnx.py`` and thread a test clip through ONNX
   Runtime, asserting per-chunk parity vs the PyTorch streaming mode (1e-4).

This is a *scaffold* gate: it validates the architecture, the
conditioning, the training path, and the export — not model quality.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Make the `tse` package + `scripts/` importable regardless of how this
# file is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT / "training", _REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tse.config import TSEConfig  # noqa: E402
from tse.data import synthetic_fixture_dataset  # noqa: E402
from tse.loss import neg_si_sdr_loss  # noqa: E402
from tse.model import CausalConvTasNetTSE, count_parameters  # noqa: E402
from tse.train import overfit_one_batch  # noqa: E402

STREAM_TOL = 1e-4
ONNX_TOL = 1e-4
OVERFIT_STEPS = 200
# The confirmed architecture (R=2, X=6, B=128, H=256, N=256) lands a touch
# under the 1.5-2.5 M design target; keep a band that brackets it but still
# catches gross structural mistakes.
PARAM_LOW = 1_000_000
PARAM_HIGH = 3_000_000


def _section(title: str) -> None:
    print(f"\n=== {title} ===", file=sys.stderr)


def check_forward_backward(config: TSEConfig) -> CausalConvTasNetTSE:
    _section("check 1: forward + backward, param count")
    torch.manual_seed(0)
    model = CausalConvTasNetTSE(config)
    n_params = count_parameters(model)
    print(f"[smoke] param count: {n_params:,}", file=sys.stderr)
    if not (PARAM_LOW <= n_params <= PARAM_HIGH):
        raise AssertionError(
            f"param count {n_params:,} outside expected band [{PARAM_LOW:,}, {PARAM_HIGH:,}]"
        )

    ds = synthetic_fixture_dataset(n=3, sample_rate=config.sample_rate, duration_sec=1.0)
    mix = torch.stack([ds[i][0] for i in range(3)])
    cond = torch.stack([ds[i][1] for i in range(3)])
    target = torch.stack([ds[i][2] for i in range(3)])

    model.train()
    est = model(mix, cond)
    if est.shape != target.shape:
        raise AssertionError(f"output shape {tuple(est.shape)} != target {tuple(target.shape)}")
    loss = neg_si_sdr_loss(est, target)
    if not torch.isfinite(loss):
        raise AssertionError(f"loss is not finite: {loss.item()}")
    loss.backward()
    grad_norm = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    if grad_norm == 0.0 or not torch.isfinite(torch.tensor(grad_norm)):
        raise AssertionError(f"bad gradient norm: {grad_norm}")
    print(
        f"[smoke] forward+backward OK  loss={loss.item():+.4f} dB  grad_norm={grad_norm:.3e}",
        file=sys.stderr,
    )
    return model


def check_full_vs_streaming(config: TSEConfig) -> None:
    _section("check 2: full-sequence == streaming")
    torch.manual_seed(1)
    model = CausalConvTasNetTSE(config).eval()
    batch = 2
    chunk_len = config.enc_stride * 10
    n_chunks = 6
    total = chunk_len * n_chunks
    mix = torch.randn(batch, total)
    cond = torch.randn(batch, config.cond_dim)

    with torch.no_grad():
        full = model(mix, cond)
        state = model.make_initial_state(batch_size=batch)
        outs = []
        for i in range(n_chunks):
            chunk = mix[:, i * chunk_len : (i + 1) * chunk_len]
            out, state = model.forward_streaming(chunk, cond, state)
            outs.append(out)
        streamed = torch.cat(outs, dim=1)

    if full.shape != streamed.shape:
        raise AssertionError(
            f"shape mismatch: full {tuple(full.shape)} vs stream {tuple(streamed.shape)}"
        )
    delta = float((full - streamed).abs().max())
    print(
        f"[smoke] full-vs-streaming max|Δ| = {delta:.3e}  (tol {STREAM_TOL:.0e})", file=sys.stderr
    )
    if delta > STREAM_TOL:
        raise AssertionError(f"full vs streaming diverged: {delta:.3e} > {STREAM_TOL:.0e}")
    print("[smoke] full == streaming OK", file=sys.stderr)


def check_overfit(config: TSEConfig) -> None:
    _section("check 3: overfit one synthetic mixture")
    torch.manual_seed(2)
    model = CausalConvTasNetTSE(config)
    result = overfit_one_batch(model, steps=OVERFIT_STEPS, lr=1e-3, log_every=50, seed=2)
    start = float(result["start_loss"])
    end = float(result["end_loss"])
    final_sisdr = float(result["final_si_sdr"])
    print(
        f"[smoke] overfit: start {start:+.4f} dB  end {end:+.4f} dB  "
        f"final SI-SDR {final_sisdr:+.4f} dB  (drop {start - end:+.4f} dB)",
        file=sys.stderr,
    )
    # The loss is -SI-SDR; a working model drives it well below the start.
    if not (end < start - 5.0):
        raise AssertionError(
            f"overfit loss did not drop substantially: start {start:.4f}, end {end:.4f}"
        )
    if final_sisdr < 5.0:
        raise AssertionError(
            f"overfit final SI-SDR too low ({final_sisdr:.4f} dB) — model is not fitting"
        )
    print("[smoke] overfit OK", file=sys.stderr)


def check_onnx_roundtrip(config: TSEConfig) -> None:
    _section("check 4: ONNX export + per-chunk round-trip parity")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise AssertionError(
            "onnxruntime not installed — install the `onnx` extra: pip install onnx onnxruntime"
        ) from None

    import export_tse_onnx as exporter

    chunk_len = config.enc_stride * 10
    n_chunks = 8
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "tse_smoke.onnx"
        export_args = exporter._build_parser().parse_args(
            [
                "export",
                "--config",
                "poc_16k" if config.sample_rate == 16_000 else "prod_48k",
                "--chunk",
                str(chunk_len),
                "--output",
                str(onnx_path),
            ]
        )
        rc = exporter.cmd_export(export_args)
        if rc != 0:
            raise AssertionError(f"ONNX export failed with rc={rc}")

        verify_args = exporter._build_parser().parse_args(
            [
                "verify",
                "--config",
                "poc_16k" if config.sample_rate == 16_000 else "prod_48k",
                "--chunk",
                str(chunk_len),
                "--onnx",
                str(onnx_path),
                "--n-chunks",
                str(n_chunks),
                "--tol",
                str(ONNX_TOL),
            ]
        )
        rc = exporter.cmd_verify(verify_args)
        if rc != 0:
            raise AssertionError(f"ONNX per-chunk parity check failed with rc={rc}")
    print("[smoke] ONNX round-trip OK", file=sys.stderr)


def main() -> int:
    config = TSEConfig.poc_16k()
    print(
        f"[smoke] Stage C TSE scaffold gate — config poc_16k "
        f"(sr={config.sample_rate}, enc {config.enc_kernel}/{config.enc_stride})",
        file=sys.stderr,
    )
    checks = [
        ("forward+backward", lambda: check_forward_backward(config)),
        ("full==streaming", lambda: check_full_vs_streaming(config)),
        ("overfit", lambda: check_overfit(config)),
        ("onnx-roundtrip", lambda: check_onnx_roundtrip(config)),
    ]
    failures: list[str] = []
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - gate must catch + report all
            failures.append(name)
            print(f"[smoke] FAIL ({name}): {exc}", file=sys.stderr)

    print("", file=sys.stderr)
    if failures:
        print(
            f"[smoke] FAILED — {len(failures)}/{len(checks)} checks: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    print(f"[smoke] PASS — all {len(checks)} checks green", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
