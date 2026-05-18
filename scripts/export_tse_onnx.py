#!/usr/bin/env python3
"""Export the causal Conv-TasNet TSE model to ONNX for Rust-side inference.

**Stateful per-chunk variant.** Mirrors ``scripts/export_dfn3_onnx.py``: the
exported ONNX consumes one fixed-size audio chunk plus the model's causal
conv-state tensors, and returns the extracted chunk plus the updated state.
Live callers thread the state across calls. The exported graph has a fixed
chunk length and ``dynamic_axes=None`` (every causal dilated depthwise conv
needs a static time axis for the trace).

The model itself (:class:`tse.model.CausalConvTasNetTSE`) already exposes a
:meth:`~tse.model.CausalConvTasNetTSE.forward_streaming` that threads a flat
``list[Tensor]`` of conv states. This script wraps that list into a flat
positional signature so ``torch.onnx.export`` can trace it::

    (audio_chunk, cond_embedding, state_0, ..., state_{K-1})
      -> (extracted_chunk, new_state_0, ..., new_state_{K-1})

where ``K == model.n_state_tensors``. The state tensor order is exactly the
one documented in ``tse/model.py`` (encoder overlap, input-norm running
stats, per-block depthwise ring buffer + two cumulative-LN triples, decoder
overlap-add tail).

Subcommands:

* ``export``            – write the ONNX file
* ``verify``            – thread state across a test clip in both PyTorch
                          (streaming mode) and ONNX Runtime, check per-chunk
                          parity at ``--tol`` (default 1e-4)
* ``export-and-verify`` – both

Run with ``onnx`` + ``onnxruntime`` installed (the ``onnx`` extra of
``mellonella-tse``)::

    python scripts/export_tse_onnx.py export-and-verify --output build/tse.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the in-repo `tse` package importable when run as a script.
_TRAINING_ROOT = Path(__file__).resolve().parent.parent / "training"
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

DEFAULT_TOL = 1e-4
DEFAULT_CHUNK = 160  # samples; multiple of the PoC encoder stride (16)
DEFAULT_N_CHUNKS = 12  # test-clip length, in chunks, for `verify`


def _load_config(name: str):  # noqa: ANN202
    from tse.config import TSEConfig

    if name == "poc_16k":
        return TSEConfig.poc_16k()
    if name == "prod_48k":
        return TSEConfig.prod_48k()
    raise ValueError(f"unknown config preset: {name!r}")


def _build_model(config, checkpoint: Path | None):  # noqa: ANN001, ANN202
    import torch

    from tse.model import CausalConvTasNetTSE

    model = CausalConvTasNetTSE(config)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"[export] loaded weights from {checkpoint}", file=sys.stderr)
    else:
        print(
            "[export] no --checkpoint given — exporting randomly-initialised weights",
            file=sys.stderr,
        )
    model.eval()
    return model


class _FlatStreamingWrapper:
    """Adapter exposing ``forward_streaming`` with a flat positional signature.

    ``torch.onnx.export`` traces a module whose ``forward`` takes
    ``(audio_chunk, cond, *state_tensors)`` and returns
    ``(extracted_chunk, *new_state_tensors)`` — the model's
    ``list[Tensor]`` state is just unpacked / repacked here.
    """

    def __new__(cls, model):  # noqa: ANN001, ANN204 - build a real nn.Module
        import torch.nn as nn

        n_state = model.n_state_tensors

        class FlatWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.n_state = n_state

            def forward(self, audio_chunk, cond, *states):  # noqa: ANN001, ANN201
                out, new_states = self.model.forward_streaming(
                    audio_chunk, cond, list(states)
                )
                return (out, *new_states)

        wrapper = FlatWrapper()
        wrapper.eval()
        return wrapper


def _state_names(n_state: int) -> tuple[list[str], list[str]]:
    in_names = [f"state_in_{i}" for i in range(n_state)]
    out_names = [f"state_out_{i}" for i in range(n_state)]
    return in_names, out_names


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)
    if args.chunk % config.enc_stride != 0:
        print(
            f"[export] chunk {args.chunk} is not a multiple of enc_stride "
            f"{config.enc_stride}",
            file=sys.stderr,
        )
        return 2

    model = _build_model(config, args.checkpoint)
    wrapper = _FlatStreamingWrapper(model)
    n_state = model.n_state_tensors

    chunk = torch.zeros(1, args.chunk, dtype=torch.float32)
    cond = torch.zeros(1, config.cond_dim, dtype=torch.float32)
    states = model.make_initial_state(batch_size=1, device="cpu")
    example = (chunk, cond, *states)

    in_names, out_names = _state_names(n_state)
    input_names = ["audio_chunk", "cond_embedding", *in_names]
    output_names = ["extracted_chunk", *out_names]

    print(
        f"[export] config={args.config} chunk={args.chunk} "
        f"n_state_tensors={n_state} params={sum(p.numel() for p in model.parameters()):,}",
        file=sys.stderr,
    )
    print(f"[export] writing {output}", file=sys.stderr)
    torch.onnx.export(
        wrapper,
        example,
        str(output),
        input_names=input_names,
        output_names=output_names,
        # Fixed chunk length — the causal dilated depthwise convs need a
        # static time axis for the trace, exactly like the DFN3 export.
        dynamic_axes=None,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    # Always drop a weights sidecar next to the ONNX. `verify` loads it so
    # the PyTorch reference uses *exactly* the weights that were traced —
    # otherwise a fresh `CausalConvTasNetTSE()` would be a different random
    # net and parity is meaningless. With a real `--checkpoint` the sidecar
    # is just a copy of those weights.
    weights_path = output.with_suffix(output.suffix + ".weights.pt")
    torch.save({"model": model.state_dict(), "config": vars(config)}, weights_path)
    print(
        f"[export] done — {output} ({output.stat().st_size / 1e6:.2f} MB); "
        f"weights sidecar {weights_path.name}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    import onnxruntime as ort
    import torch

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[verify] missing {onnx_path}; run `export` first", file=sys.stderr)
        return 2

    config = _load_config(args.config)
    # Prefer an explicit --checkpoint; otherwise fall back to the weights
    # sidecar written by `export` so the PyTorch reference matches the ONNX.
    checkpoint = args.checkpoint
    if checkpoint is None:
        sidecar = onnx_path.with_suffix(onnx_path.suffix + ".weights.pt")
        if sidecar.exists():
            checkpoint = sidecar
            print(f"[verify] using weights sidecar {sidecar.name}", file=sys.stderr)
        else:
            print(
                "[verify] WARNING: no --checkpoint and no weights sidecar — the "
                "PyTorch reference will be a fresh random net and parity is "
                "meaningless",
                file=sys.stderr,
            )
    model = _build_model(config, checkpoint)
    n_state = model.n_state_tensors
    chunk_len = args.chunk
    n_chunks = args.n_chunks

    # Deterministic test clip.
    rng = np.random.default_rng(42)
    total = chunk_len * n_chunks
    audio = (rng.standard_normal(total) * 0.1).astype(np.float32)
    cond_np = rng.standard_normal((1, config.cond_dim)).astype(np.float32)
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    cond_t = torch.from_numpy(cond_np)

    # --- PyTorch streaming reference ---
    torch_chunks: list[np.ndarray] = []
    state = model.make_initial_state(batch_size=1, device="cpu")
    with torch.no_grad():
        for i in range(n_chunks):
            ch = audio_t[:, i * chunk_len : (i + 1) * chunk_len]
            out, state = model.forward_streaming(ch, cond_t, state)
            torch_chunks.append(out.cpu().numpy())
    torch_out = np.concatenate(torch_chunks, axis=1).astype(np.float64)

    # --- ONNX Runtime, threading state the same way ---
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names, out_names = _state_names(n_state)
    onnx_state = [
        s.numpy() for s in model.make_initial_state(batch_size=1, device="cpu")
    ]
    onnx_chunks: list[np.ndarray] = []
    per_chunk_delta = 0.0
    for i in range(n_chunks):
        ch = audio[i * chunk_len : (i + 1) * chunk_len][None, :].astype(np.float32)
        feeds = {"audio_chunk": ch, "cond_embedding": cond_np}
        for name, val in zip(in_names, onnx_state, strict=True):
            feeds[name] = val.astype(np.float32)
        results = session.run(["extracted_chunk", *out_names], feeds)
        onnx_chunks.append(results[0])
        onnx_state = list(results[1:])
        chunk_delta = float(
            np.max(
                np.abs(
                    results[0].astype(np.float64) - torch_chunks[i].astype(np.float64)
                )
            )
        )
        per_chunk_delta = max(per_chunk_delta, chunk_delta)
        print(f"[verify] chunk {i:3d}  max|Δ| = {chunk_delta:.3e}", file=sys.stderr)
    onnx_out = np.concatenate(onnx_chunks, axis=1).astype(np.float64)

    overall = float(np.max(np.abs(torch_out - onnx_out)))
    print("", file=sys.stderr)
    print(f"[verify] n_chunks={n_chunks} chunk_len={chunk_len}", file=sys.stderr)
    print(f"[verify] per-chunk max|Δ|  = {per_chunk_delta:.3e}", file=sys.stderr)
    print(f"[verify] overall  max|Δ|   = {overall:.3e}", file=sys.stderr)
    print(f"[verify] tolerance         = {args.tol:.3e}", file=sys.stderr)
    if overall < args.tol:
        print(
            "[verify] PASS — PyTorch↔ONNX per-chunk parity within tolerance",
            file=sys.stderr,
        )
        return 0
    print("[verify] FAIL — parity exceeds tolerance", file=sys.stderr)
    return 1


def cmd_export_and_verify(args: argparse.Namespace) -> int:
    rc = cmd_export(args)
    if rc != 0:
        return rc
    args.onnx = args.output
    return cmd_verify(args)


# ---------------------------------------------------------------------------
# dump-fixture
# ---------------------------------------------------------------------------


def cmd_dump_fixture(args: argparse.Namespace) -> int:
    """Stream a caller-supplied clip through ONNX Runtime and dump the
    extracted-audio output to a flat ``float32`` binary file.

    Inputs (all little-endian ``float32`` binary, written by the caller —
    typically a Rust parity test):

    * ``--clip``  : ``chunk_len * n_chunks`` samples
    * ``--cond``  : ``cond_dim`` (192 for ECAPA) cond embedding

    Output: ``--output`` ``float32`` binary, same length as the clip
    (per-chunk extracted audio concatenated).

    This is the lower-friction path for the Rust ↔ ONNX parity test —
    Rust writes its own deterministic clip + cond, calls this command,
    reads back the ONNX-side expected output and compares.
    """
    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    clip_path = Path(args.clip)
    cond_path = Path(args.cond)
    out_path = Path(args.output)
    if not onnx_path.exists():
        print(f"[dump-fixture] missing {onnx_path}", file=sys.stderr)
        return 2
    if not clip_path.exists():
        print(f"[dump-fixture] missing {clip_path}", file=sys.stderr)
        return 2
    if not cond_path.exists():
        print(f"[dump-fixture] missing {cond_path}", file=sys.stderr)
        return 2

    config = _load_config(args.config)
    cond_dim = config.cond_dim

    audio = np.fromfile(clip_path, dtype=np.float32)
    cond = np.fromfile(cond_path, dtype=np.float32)
    if cond.size != cond_dim:
        print(
            f"[dump-fixture] cond length {cond.size} != cond_dim {cond_dim}",
            file=sys.stderr,
        )
        return 2
    chunk_len = args.chunk
    if audio.size == 0 or audio.size % chunk_len != 0:
        print(
            f"[dump-fixture] clip length {audio.size} is not a positive multiple "
            f"of chunk {chunk_len}",
            file=sys.stderr,
        )
        return 2
    if chunk_len % config.enc_stride != 0:
        print(
            f"[dump-fixture] chunk {chunk_len} not a multiple of enc_stride "
            f"{config.enc_stride}",
            file=sys.stderr,
        )
        return 2
    n_chunks = audio.size // chunk_len

    # Build initial state from a zero-init model. We only need the
    # *shapes* of make_initial_state — the values are zero — so we don't
    # have to load the weights sidecar here.
    from tse.model import CausalConvTasNetTSE  # local import keeps `torch`
    # off the import path when `--onnx` is the only thing we touch.

    model = CausalConvTasNetTSE(config)
    state = [s.numpy() for s in model.make_initial_state(batch_size=1)]
    in_names, out_names = _state_names(len(state))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cond_np = cond.reshape(1, cond_dim).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_chunks: list[np.ndarray] = []
    for i in range(n_chunks):
        ch = audio[i * chunk_len : (i + 1) * chunk_len][None, :].astype(np.float32)
        feeds = {"audio_chunk": ch, "cond_embedding": cond_np}
        for name, val in zip(in_names, state, strict=True):
            feeds[name] = val.astype(np.float32)
        results = session.run(["extracted_chunk", *out_names], feeds)
        out_chunks.append(results[0].astype(np.float32))
        state = list(results[1:])
        if args.verbose:
            print(
                f"[dump-fixture] chunk {i:3d} max|x|={float(np.max(np.abs(results[0]))):.3e}",
                file=sys.stderr,
            )

    concatenated = np.concatenate(out_chunks, axis=1).astype(np.float32).reshape(-1)
    concatenated.tofile(out_path)
    print(
        f"[dump-fixture] wrote {out_path} ({concatenated.size} f32 samples)",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", choices=("poc_16k", "prod_48k"), default="poc_16k")
    common.add_argument(
        "--chunk", type=int, default=DEFAULT_CHUNK, help="chunk length in samples"
    )
    common.add_argument(
        "--checkpoint", type=Path, default=None, help="optional trained-weights .pt"
    )

    p_export = sub.add_parser("export", parents=[common])
    p_export.add_argument("--output", default="build/tse.onnx")
    p_export.add_argument("--opset", type=int, default=18)
    p_export.set_defaults(func=cmd_export)

    p_verify = sub.add_parser("verify", parents=[common])
    p_verify.add_argument("--onnx", required=True)
    p_verify.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    p_verify.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_verify.set_defaults(func=cmd_verify)

    p_both = sub.add_parser("export-and-verify", parents=[common])
    p_both.add_argument("--output", default="build/tse.onnx")
    p_both.add_argument("--opset", type=int, default=18)
    p_both.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    p_both.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_both.set_defaults(func=cmd_export_and_verify)

    # Stream a caller-supplied clip through ONNX Runtime and dump the
    # output as a flat float32 binary. Used by the Rust parity test.
    p_dump = sub.add_parser("dump-fixture", parents=[common])
    p_dump.add_argument("--onnx", required=True)
    p_dump.add_argument(
        "--clip", required=True, help="float32 binary, length n_chunks * chunk"
    )
    p_dump.add_argument(
        "--cond", required=True, help="float32 binary, length cond_dim (192)"
    )
    p_dump.add_argument(
        "--output", required=True, help="destination float32 binary"
    )
    p_dump.add_argument("--verbose", action="store_true")
    p_dump.set_defaults(func=cmd_dump_fixture)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
