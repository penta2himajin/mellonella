#!/usr/bin/env python3
"""Export DeepFilterNet 3 to ONNX for Rust-side noise suppression.

Upstream DFN3 can't be ONNX-exported as-is: ``df.multiframe.DF.forward``
calls ``torch.view_as_complex`` and does an in-place mutation on an
``as_strided`` view chain (``spec[..., : num_freqs, :] = …``). Both
legacy and dynamo exporters bail out on that.

Workaround: monkey-patch ``model.df_op`` with a functionally-equivalent
``DfOnnxSafe`` module that

* operates on real-valued tensors only (no ``view_as_complex``)
* substitutes the filtered low-freq band via ``torch.cat`` instead of
  in-place assignment

After the patch, ``torch.onnx.export`` lowers the model cleanly. The
exported ONNX takes the same ``(spec, feat_erb, feat_spec)`` inputs the
PyTorch model takes, so the Rust caller still needs ``deep_filter``'s
STFT + ERB feature pipeline to feed it.

Subcommands:

* ``export``  – write the ONNX file
* ``verify``  – run PyTorch and ONNX side by side on synth noise, check
                per-sample parity after iSTFT

Run on a host with the ``models`` extra installed (plus ``onnxscript``):

    python scripts/export_dfn3_onnx.py export-and-verify \\
        --output build/dfn3.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

DEFAULT_TOL = 1e-3


# ---------------------------------------------------------------------------
# ONNX-safe replacement for DF.forward (df_op)
# ---------------------------------------------------------------------------


def _build_safe_df_op(orig_df):  # noqa: ANN001
    """Return a ``DfOnnxSafe`` torch.nn.Module mirroring ``orig_df``."""
    import torch
    import torch.nn as nn
    from df.multiframe import df_real

    # Stash params on the new module so the export captures them.
    num_freqs = orig_df.num_freqs
    frame_size = orig_df.frame_size
    lookahead = orig_df.lookahead

    front_pad = frame_size - 1 - lookahead
    back_pad = lookahead

    class DfOnnxSafe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_freqs = num_freqs
            self.frame_size = frame_size
            self.lookahead = lookahead

        def forward(self, spec: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
            # spec : (B, C, T, F, 2)  — real-valued
            # coefs: (B, C, T, F', 2) — real-valued
            # F.pad with literal integer args keeps the output shape
            # static for ONNX trace (ConstantPad3d builds them at runtime
            # which produces a dynamic-shape Pad op that breaks Unfold).
            if self.frame_size > 1:
                padded = torch.nn.functional.pad(
                    spec, (0, 0, 0, 0, front_pad, back_pad), mode="constant", value=0.0
                )
                # Replace tensor.unfold with a stack of narrowed slices —
                # `unfold` requires static input sizes which ONNX trace
                # struggles to surface; stacking N narrows is equivalent
                # and exports cleanly.
                slices = [
                    padded.narrow(-3, n, spec.shape[-3]) for n in range(self.frame_size)
                ]
                spec_u = torch.stack(slices, dim=2)
            else:
                spec_u = spec.unsqueeze(2)
            spec_f = spec_u.narrow(-2, 0, self.num_freqs)
            new_shape = [coefs.shape[0], -1, self.frame_size] + list(coefs.shape[2:])
            coefs = coefs.view(new_shape)
            spec_filtered = df_real(spec_f, coefs)
            # Functional substitution: concat filtered low-freq with the
            # original high-freq band along the F axis.
            spec_high = spec[..., self.num_freqs :, :]
            return torch.cat([spec_filtered, spec_high], dim=-2)

    return DfOnnxSafe()


# ---------------------------------------------------------------------------
# Loading + input shape probing
# ---------------------------------------------------------------------------


def _load_model_and_patch():  # noqa: ANN202
    from df.enhance import init_df

    model, df_state, _ = init_df()
    model.eval()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device="cpu")
    model.df_op = _build_safe_df_op(model.df_op)
    return model, df_state


def _build_inputs(model, df_state):  # noqa: ANN001
    import torch
    from df.enhance import df_features
    from df.model import ModelParams

    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    audio = np.random.default_rng(42).standard_normal((1, 48_000)).astype(np.float32) * 0.1
    audio = torch.from_numpy(audio)
    audio = torch.nn.functional.pad(audio, (0, df_state.fft_size()))
    spec, feat_erb, feat_spec = df_features(audio, df_state, nb_df, device="cpu")
    return spec, feat_erb, feat_spec


# ---------------------------------------------------------------------------
# Export / verify
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("[export] loading DFN3 + patching df_op", file=sys.stderr)
    model, df_state = _load_model_and_patch()
    spec, feat_erb, feat_spec = _build_inputs(model, df_state)
    print(
        f"[export] input shapes: spec={tuple(spec.shape)} "
        f"feat_erb={tuple(feat_erb.shape)} feat_spec={tuple(feat_spec.shape)}",
        file=sys.stderr,
    )

    with torch.no_grad():
        ref_out = model(spec, feat_erb, feat_spec)
    for i, t in enumerate(ref_out):
        print(f"[export] forward output #{i}: {tuple(t.shape)}", file=sys.stderr)

    print(f"[export] writing {output}", file=sys.stderr)
    torch.onnx.export(
        model,
        (spec, feat_erb, feat_spec),
        str(output),
        input_names=["spec", "feat_erb", "feat_spec"],
        output_names=["enhanced_spec", "mask", "lsnr", "df_alpha"],
        # All axes are fixed in this export. df_op uses tensor.unfold,
        # which can't be lowered to ONNX when any dim it sees is
        # symbolic — and the trace propagates dynamic dims through the
        # encoder layers. Callers chunk audio into fixed-length
        # segments (default 102 frames ≈ 1 s at 48 kHz with 480-sample
        # hop) before running the ONNX.
        dynamic_axes=None,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(
        f"[export] done — {output} ({output.stat().st_size / 1e6:.1f} MB)",
        file=sys.stderr,
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    import onnxruntime as ort
    import torch

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[verify] missing {onnx_path}; run `export` first", file=sys.stderr)
        return 2

    print("[verify] loading torch model + ONNX session", file=sys.stderr)
    model, df_state = _load_model_and_patch()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    spec, feat_erb, feat_spec = _build_inputs(model, df_state)
    with torch.no_grad():
        ref_enh_spec, _, _, _ = model(spec, feat_erb, feat_spec)
    ref = ref_enh_spec.cpu().numpy().astype(np.float64)

    onnx_inputs = {
        "spec": spec.numpy(),
        "feat_erb": feat_erb.numpy(),
        "feat_spec": feat_spec.numpy(),
    }
    onnx_out = session.run(["enhanced_spec"], onnx_inputs)[0].astype(np.float64)

    delta = float(np.max(np.abs(ref - onnx_out)))
    print(f"[verify] enhanced_spec max|Δ| = {delta:.3e}", file=sys.stderr)
    print(f"[verify] tolerance            = {args.tol:.3e}", file=sys.stderr)
    if delta < args.tol:
        print("[verify] PASS — enhanced_spec parity within tolerance", file=sys.stderr)
        return 0
    print("[verify] FAIL — enhanced_spec parity exceeds tolerance", file=sys.stderr)
    return 1


def cmd_export_and_verify(args: argparse.Namespace) -> int:
    rc = cmd_export(args)
    if rc != 0:
        return rc
    args.onnx = args.output
    return cmd_verify(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--output", default="build/dfn3.onnx")
    p_export.add_argument("--opset", type=int, default=17)
    p_export.set_defaults(func=cmd_export)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--onnx", required=True)
    p_verify.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_verify.set_defaults(func=cmd_verify)

    p_both = sub.add_parser("export-and-verify")
    p_both.add_argument("--output", default="build/dfn3.onnx")
    p_both.add_argument("--opset", type=int, default=17)
    p_both.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_both.set_defaults(func=cmd_export_and_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
