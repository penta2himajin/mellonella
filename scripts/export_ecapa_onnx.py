#!/usr/bin/env python3
"""Export SpeechBrain ECAPA-TDNN to ONNX and verify PyTorch ↔ ONNX parity.

First concrete deliverable for implementation.md Phase 3 (Rust port). The
goal is a single ONNX graph that takes a 16 kHz mono waveform and emits the
192-dim speaker embedding, so the Rust side only needs the `ort` crate plus
audio resampling — no SpeechBrain / torchaudio dependency at runtime.

Modes
-----
``export`` (default)
    Build a torch.nn.Module that wraps the SpeechBrain pipeline
    (compute_features → mean_var_norm → embedding_model) and call
    ``torch.onnx.export`` on it.

``verify``
    Load both the PyTorch model and the exported ONNX, run them on a small
    set of deterministic synthetic waveforms (and optional `--audio` files),
    and report:

      * max |Δ| in raw 192-dim embeddings (component-wise)
      * max |Δ| in pairwise cosine similarity matrix
      * pass/fail against ``--tol`` (default 1e-4 to match issue #64)

Usage
-----
    python scripts/export_ecapa_onnx.py export \\
        --output build/ecapa_tdnn.onnx
    python scripts/export_ecapa_onnx.py verify \\
        --onnx build/ecapa_tdnn.onnx --tol 1e-4

    # Both in one go (export then verify the freshly produced file):
    python scripts/export_ecapa_onnx.py export-and-verify \\
        --output build/ecapa_tdnn.onnx

Notes
-----
* Requires the `models` extra (`pip install -e poc[models]`) for torch,
  speechbrain, and onnxruntime.
* The exported graph fixes the sample rate to 16 kHz (SpeechBrain's
  ``compute_features`` pre-built pipeline is hard-coded to that rate). The
  time axis is dynamic so the same .onnx works for any clip length ≥ 1 s.
* This is intentionally a _verification_ tool, not a production builder.
  The Rust integration in ``mellonella-core`` will likely re-export with
  graph-level optimisation (e.g. constant-fold the mean/var norm stats).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_TOL = 1e-4
EMBEDDING_DIM = 192


def _synth_waveform(seed: int, sr: int, duration_sec: float, f0: float) -> np.ndarray:
    """Deterministic harmonic-stack waveform with light AM modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration_sec)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for harmonic in range(1, 6):
        phase = rng.uniform(0, 2 * np.pi)
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t + phase).astype(np.float32)
    am = 0.5 * (1.0 + 0.3 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
    wave = (wave * am).astype(np.float32)
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave = (wave / peak) * 0.9
    return wave.astype(np.float32)


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        try:
            import librosa  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"audio at {path} is {sr} Hz, need {target_sr} Hz; install librosa for resample"
            ) from exc
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def _build_wrapper(source: str, savedir: str | None) -> "torch.nn.Module":
    """Return a torch.nn.Module: (B, T) float32 → (B, 192) float32 embedding."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import-not-found]

    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": "cpu"},
    )

    compute_features = classifier.mods.compute_features
    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model

    class EcapaOnnxWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compute_features = compute_features
            self.mean_var_norm = mean_var_norm
            self.embedding_model = embedding_model

        def forward(self, wav: "torch.Tensor") -> "torch.Tensor":
            feats = self.compute_features(wav)
            wav_lens = torch.ones(wav.shape[0], device=wav.device)
            feats = self.mean_var_norm(feats, wav_lens)
            emb = self.embedding_model(feats, wav_lens)
            return emb.squeeze(1)

    wrapper = EcapaOnnxWrapper()
    wrapper.eval()
    return wrapper


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading {args.source}", file=sys.stderr)
    wrapper = _build_wrapper(args.source, args.savedir)

    dummy = torch.zeros(1, args.sample_rate * 3, dtype=torch.float32)
    with torch.no_grad():
        out = wrapper(dummy)
    if out.shape[-1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"unexpected embedding dim {out.shape[-1]}, expected {EMBEDDING_DIM}; "
            f"the SpeechBrain checkpoint may have changed"
        )

    print(f"[export] writing {output}", file=sys.stderr)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output),
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "embedding": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[export] done — {output} ({output.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return 0


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_n @ b_n.T


def cmd_verify(args: argparse.Namespace) -> int:
    import onnxruntime as ort  # type: ignore[import-not-found]
    import torch

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"[verify] missing {onnx_path}; run `export` first", file=sys.stderr)
        return 2

    print(f"[verify] loading torch model {args.source}", file=sys.stderr)
    wrapper = _build_wrapper(args.source, args.savedir)

    print(f"[verify] loading onnx {onnx_path}", file=sys.stderr)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    waves: list[np.ndarray] = []
    labels: list[str] = []

    f0s = (140.0, 180.0, 220.0)
    for i, f0 in enumerate(f0s):
        wav = _synth_waveform(seed=42 + i, sr=args.sample_rate, duration_sec=3.0, f0=f0)
        waves.append(wav)
        labels.append(f"synth_{i}_f{int(f0)}")

    for path_str in args.audio or []:
        wav = _load_audio(Path(path_str), args.sample_rate)
        if wav.size < args.sample_rate:
            print(f"[verify] skip {path_str}: shorter than 1 s", file=sys.stderr)
            continue
        waves.append(wav)
        labels.append(Path(path_str).stem)

    if not waves:
        print("[verify] no inputs", file=sys.stderr)
        return 2

    torch_embs: list[np.ndarray] = []
    onnx_embs: list[np.ndarray] = []
    raw_max_delta = 0.0
    for wav, label in zip(waves, labels, strict=True):
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            emb_t = wrapper(wav_t).cpu().numpy().astype(np.float64)
        emb_o = session.run(["embedding"], {"waveform": wav[None, :]})[0].astype(np.float64)
        delta = float(np.max(np.abs(emb_t - emb_o)))
        raw_max_delta = max(raw_max_delta, delta)
        print(f"[verify] {label:24s} raw max|Δ|={delta:.3e}", file=sys.stderr)
        torch_embs.append(emb_t[0])
        onnx_embs.append(emb_o[0])

    torch_arr = np.stack(torch_embs, axis=0)
    onnx_arr = np.stack(onnx_embs, axis=0)
    cos_t = _cosine_matrix(torch_arr, torch_arr)
    cos_o = _cosine_matrix(onnx_arr, onnx_arr)
    cos_max_delta = float(np.max(np.abs(cos_t - cos_o)))

    print("", file=sys.stderr)
    print(f"[verify] N={len(waves)} clips", file=sys.stderr)
    print(f"[verify] raw embedding max|Δ|       = {raw_max_delta:.3e}", file=sys.stderr)
    print(f"[verify] cosine-similarity max|Δ|   = {cos_max_delta:.3e}", file=sys.stderr)
    print(f"[verify] tolerance                  = {args.tol:.3e}", file=sys.stderr)

    cos_pass = cos_max_delta < args.tol
    if cos_pass:
        print("[verify] PASS — cosine parity within tolerance", file=sys.stderr)
        return 0
    print("[verify] FAIL — cosine parity exceeds tolerance", file=sys.stderr)
    return 1


def cmd_export_and_verify(args: argparse.Namespace) -> int:
    rc = cmd_export(args)
    if rc != 0:
        return rc
    args.onnx = args.output
    return cmd_verify(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", default=DEFAULT_SOURCE)
    common.add_argument("--savedir", default=None)
    common.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)

    p_export = sub.add_parser("export", parents=[common])
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--opset", type=int, default=17)
    p_export.set_defaults(func=cmd_export)

    p_verify = sub.add_parser("verify", parents=[common])
    p_verify.add_argument("--onnx", required=True)
    p_verify.add_argument("--audio", nargs="*", help="optional WAV/FLAC files for additional parity checks")
    p_verify.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_verify.set_defaults(func=cmd_verify)

    p_both = sub.add_parser("export-and-verify", parents=[common])
    p_both.add_argument("--output", required=True)
    p_both.add_argument("--opset", type=int, default=17)
    p_both.add_argument("--audio", nargs="*")
    p_both.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_both.set_defaults(func=cmd_export_and_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
