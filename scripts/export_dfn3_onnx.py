#!/usr/bin/env python3
"""Export DeepFilterNet 3 to ONNX for Rust-side noise suppression.

**Stateful per-frame variant.** The exported ONNX takes one STFT
frame at a time plus the three GRU hidden states, and returns one
enhanced spectrum frame plus the updated states. Live callers
maintain a rolling ``conv_lookahead``-frame feature buffer and
thread the GRU states across calls, yielding an algorithmic latency
of roughly 30 ms (model's intrinsic budget — see
``docs/architecture.md``) instead of the 1.02 s the previous
102-frame export forced on streaming consumers.

Upstream DFN3 isn't directly ONNX-exportable as-is. The historic
issues:

1. ``df.multiframe.DF.forward`` uses ``torch.view_as_complex`` and
   does an in-place mutation on an ``as_strided`` view chain
   (``spec[..., : num_freqs, :] = …``). Both legacy and dynamo
   exporters bail out on that.
2. ``DfNet`` calls the inner GRU layers with ``h=None``, dropping
   the hidden state every batch. The 102-frame export hides this by
   processing everything in one shot; for streaming we have to
   thread state explicitly.
3. ``DfNet.pad_feat`` / ``pad_spec`` use ``ConstantPad2d`` to do an
   in-batch lookahead-shift, which is fundamentally per-batch and
   breaks chunked streaming.

This script applies three patches before tracing:

* Replace ``model.df_op`` with a functionally-equivalent
  ``DfOnnxSafe`` that operates on real-valued tensors and uses
  ``torch.cat`` instead of in-place mutation.
* Replace ``model.pad_feat`` / ``pad_spec`` with ``nn.Identity()`` —
  the live wrapper aligns inputs externally by maintaining a
  ``conv_lookahead``-frame future-feature ring.
* Monkey-patch each ``SqueezedGRU_S.forward`` to read/write hidden
  state via instance attributes that the outer wrapper sets before
  each forward call. ``torch.onnx.export`` traces these attribute
  re-binds as direct Tensor flow through the graph.

After the patches, the wrapping module's ``forward`` signature is::

    (spec, feat_erb, feat_spec, enc_h, erb_h, df_h)
      -> (enhanced_spec, new_enc_h, new_erb_h, new_df_h)

All tensors have time dim 1 (one STFT frame per call).

Subcommands:

* ``export``  – write the ONNX file
* ``verify``  – run PyTorch and ONNX side-by-side, threading state
                across the test clip, checking per-frame parity.

Run on a host with the ``models`` extra installed (plus
``onnxscript``)::

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

DEFAULT_TOL = 5e-3


# ---------------------------------------------------------------------------
# Stateful + ONNX-safe wrapping
# ---------------------------------------------------------------------------


def _build_safe_df_op(orig_df):  # noqa: ANN001
    """Return a ``DfOnnxSafe`` torch.nn.Module mirroring ``orig_df``.

    Operates on real-valued tensors only (no ``view_as_complex``) and
    substitutes the filtered low-freq band via ``torch.cat`` instead
    of in-place assignment. Shape-equivalent to the upstream forward
    for any time-dimension length.
    """
    import torch
    import torch.nn as nn
    from df.multiframe import df_real

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
            # spec : (B, C, T, F, 2) — real-valued
            # coefs: (B, C, T, F', 2) — real-valued
            if self.frame_size > 1:
                padded = torch.nn.functional.pad(
                    spec, (0, 0, 0, 0, front_pad, back_pad), mode="constant", value=0.0
                )
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
            spec_high = spec[..., self.num_freqs:, :]
            return torch.cat([spec_filtered, spec_high], dim=-2)

    return DfOnnxSafe()


def _build_stateful_wrapper(base_model):  # noqa: ANN001
    """Wrap ``base_model`` so its forward signature accepts/returns
    the three GRU hidden states explicitly.

    Side-effects on ``base_model``: ``pad_feat`` / ``pad_spec`` are
    replaced with ``Identity`` (caller does manual alignment), the
    three ``SqueezedGRU_S.forward`` methods are monkey-patched to
    read state from ``base_model._mell_*_in`` and write to
    ``base_model._mell_*_out``. The outer wrapper.forward sets the
    ``_in`` slots from its arguments before invoking base.forward,
    and returns the ``_out`` slots.

    ``torch.onnx.export`` traces the actual Tensor data flow inside
    each forward, so the slot mechanics don't generate graph ops —
    the state flows directly from wrapper inputs through the GRU and
    out to wrapper outputs.
    """
    import torch
    import torch.nn as nn

    base_model.pad_feat = nn.Identity()
    base_model.pad_spec = nn.Identity()

    # Initial slots so attribute reads succeed during the first trace.
    base_model._mell_enc_in = torch.zeros((1, 1, 256))
    base_model._mell_erb_in = torch.zeros((2, 1, 256))
    base_model._mell_df_in = torch.zeros((2, 1, 256))
    base_model._mell_enc_out = None
    base_model._mell_erb_out = None
    base_model._mell_df_out = None

    def make_patched(squeezed_gru, slot_in: str, slot_out: str):  # noqa: ANN001
        def patched(input: "torch.Tensor", h=None):  # noqa: ANN001, A002
            x = squeezed_gru.linear_in(input)
            x, new_h = squeezed_gru.gru(x, getattr(base_model, slot_in))
            x = squeezed_gru.linear_out(x)
            if squeezed_gru.gru_skip is not None:
                x = x + squeezed_gru.gru_skip(input)
            setattr(base_model, slot_out, new_h)
            return x, new_h

        return patched

    base_model.enc.emb_gru.forward = make_patched(
        base_model.enc.emb_gru, "_mell_enc_in", "_mell_enc_out"
    )
    base_model.erb_dec.emb_gru.forward = make_patched(
        base_model.erb_dec.emb_gru, "_mell_erb_in", "_mell_erb_out"
    )
    base_model.df_dec.df_gru.forward = make_patched(
        base_model.df_dec.df_gru, "_mell_df_in", "_mell_df_out"
    )

    class StatefulWrapper(nn.Module):
        def __init__(self, m):  # noqa: ANN001
            super().__init__()
            self.base = m

        def forward(
            self,
            spec: "torch.Tensor",
            feat_erb: "torch.Tensor",
            feat_spec: "torch.Tensor",
            enc_h: "torch.Tensor",
            erb_h: "torch.Tensor",
            df_h: "torch.Tensor",
        ):
            self.base._mell_enc_in = enc_h
            self.base._mell_erb_in = erb_h
            self.base._mell_df_in = df_h
            out = self.base(spec, feat_erb, feat_spec)
            return (
                out[0],
                self.base._mell_enc_out,
                self.base._mell_erb_out,
                self.base._mell_df_out,
            )

    return StatefulWrapper(base_model)


def _load_model_and_patch():  # noqa: ANN202
    from df.enhance import init_df

    model, df_state, _ = init_df()
    model.eval()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device="cpu")
    model.df_op = _build_safe_df_op(model.df_op)
    wrapper = _build_stateful_wrapper(model)
    return wrapper, model, df_state


def _conv_lookahead() -> int:
    from df.model import ModelParams

    return ModelParams().conv_lookahead


def _manual_align(t, conv_la: int):  # noqa: ANN001
    """Replicate the original ``pad_feat`` shift (drop first
    ``conv_la`` frames, append ``conv_la`` zero frames) outside the
    model. Used by ``verify`` to feed inputs the same way the live
    Rust wrapper will."""
    import torch

    if conv_la == 0:
        return t
    shifted = t[:, :, conv_la:]
    pad_shape = list(t.shape)
    pad_shape[2] = conv_la
    zeros = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([shifted, zeros], dim=2)


def _build_inputs(model, df_state):  # noqa: ANN001
    """Build a 1 s test clip's STFT features at the unpadded shape
    (i.e. before the model's now-disabled pad_feat). Returns
    ``(spec, feat_erb, feat_spec)`` with time dim 102."""
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

    print("[export] loading DFN3 + patching df_op + stateful wrapper", file=sys.stderr)
    wrapper, _model, df_state = _load_model_and_patch()
    spec, feat_erb, feat_spec = _build_inputs(_model, df_state)

    # Build single-frame example inputs for the trace.
    conv_la = _conv_lookahead()
    aligned_erb = _manual_align(feat_erb, conv_la)
    aligned_spec = _manual_align(feat_spec, conv_la)
    example = (
        spec[:, :, 0:1],
        aligned_erb[:, :, 0:1],
        aligned_spec[:, :, 0:1],
        torch.zeros((1, 1, 256)),
        torch.zeros((2, 1, 256)),
        torch.zeros((2, 1, 256)),
    )
    print(
        "[export] example shapes: "
        f"spec={tuple(example[0].shape)} "
        f"feat_erb={tuple(example[1].shape)} "
        f"feat_spec={tuple(example[2].shape)} "
        f"enc_h={tuple(example[3].shape)} "
        f"erb_h={tuple(example[4].shape)} "
        f"df_h={tuple(example[5].shape)}",
        file=sys.stderr,
    )

    print(f"[export] writing {output}", file=sys.stderr)
    torch.onnx.export(
        wrapper,
        example,
        str(output),
        input_names=["spec", "feat_erb", "feat_spec", "enc_h", "erb_h", "df_h"],
        output_names=["enhanced_spec", "new_enc_h", "new_erb_h", "new_df_h"],
        # Time dim is fixed at 1: tensor.unfold in the safe df_op
        # still wants a static length for the time axis. The wrapper's
        # caller invokes the ONNX once per STFT frame.
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
    # Build a parallel reference by loading a fresh unpatched model
    # (only df_op patched, not stateful) and feeding the full 102
    # frames. This is the "ground truth" the streaming ONNX must
    # approximate.
    from df.enhance import init_df

    ref_model, df_state, _ = init_df()
    ref_model.eval()
    if hasattr(ref_model, "reset_h0"):
        ref_model.reset_h0(batch_size=1, device="cpu")
    ref_model.df_op = _build_safe_df_op(ref_model.df_op)

    spec, feat_erb, feat_spec = _build_inputs(ref_model, df_state)
    T = spec.shape[2]
    with torch.no_grad():
        ref_enh_spec, _, _, _ = ref_model(spec, feat_erb, feat_spec)
    ref = ref_enh_spec.cpu().numpy().astype(np.float64)

    # Now run the streaming ONNX frame-by-frame and concatenate.
    conv_la = _conv_lookahead()
    aligned_erb = _manual_align(feat_erb, conv_la)
    aligned_spec = _manual_align(feat_spec, conv_la)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    enc_h = np.zeros((1, 1, 256), dtype=np.float32)
    erb_h = np.zeros((2, 1, 256), dtype=np.float32)
    df_h = np.zeros((2, 1, 256), dtype=np.float32)
    chunks = []
    for t in range(T):
        inputs = {
            "spec": spec[:, :, t:t + 1].numpy(),
            "feat_erb": aligned_erb[:, :, t:t + 1].numpy(),
            "feat_spec": aligned_spec[:, :, t:t + 1].numpy(),
            "enc_h": enc_h,
            "erb_h": erb_h,
            "df_h": df_h,
        }
        out = session.run(
            ["enhanced_spec", "new_enc_h", "new_erb_h", "new_df_h"], inputs
        )
        chunks.append(out[0])
        enc_h, erb_h, df_h = out[1], out[2], out[3]
    onnx_out = np.concatenate(chunks, axis=2).astype(np.float64)

    delta = float(np.max(np.abs(ref - onnx_out)))
    print(f"[verify] enhanced_spec max|Δ| = {delta:.3e}", file=sys.stderr)
    print(f"[verify] tolerance            = {args.tol:.3e}", file=sys.stderr)
    if delta < args.tol:
        print("[verify] PASS — streaming ONNX parity within tolerance", file=sys.stderr)
        return 0
    print("[verify] FAIL — streaming ONNX parity exceeds tolerance", file=sys.stderr)
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
