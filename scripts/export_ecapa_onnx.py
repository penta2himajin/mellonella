#!/usr/bin/env python3
"""Export SpeechBrain ECAPA-TDNN to ONNX and verify PyTorch ↔ ONNX parity.

First concrete deliverable for implementation.md Phase 3 (Rust port).

Modes
-----
``--mode full``
    Wraps SpeechBrain's compute_features → mean_var_norm → embedding_model
    into a single torch.nn.Module and exports it as a fixed-16 kHz,
    dynamic-time-axis ONNX. **Currently broken** under torch 2.4 — the
    SpeechBrain STFT returns a complex tensor and torch.onnx.export
    raises ``SymbolicValueError: STFT does not currently support complex
    types``. Kept here for when upstream lifts the restriction.

``--mode embedding-only`` (default)
    Wraps mean_var_norm → embedding_model only. Input is the
    pre-computed Fbank ``(B, T_frames, n_mels)`` tensor. The Rust side
    must reproduce SpeechBrain's Fbank pipeline (or call out to a
    feature-extractor crate). Verify still computes both halves in
    PyTorch and confirms the ONNX subgraph is byte-equivalent on its
    inputs, plus the round-trip cosine vs. the full PyTorch reference.

``--mode encoder`` / ``--mode pooler`` (#119)
    Split the embedding-only graph at the attentive-pooling boundary:
    ``encoder`` is features → per-frame feature map ``(B, C_mfa, T)``
    (mean_var_norm + ECAPA blocks + MFA), ``pooler`` is that map →
    ``(B, 192)`` embedding (attentive stat pooling + bn + fc).
    ``pooler(encoder(x))`` reproduces the monolithic embedding-only
    graph exactly — see the ``verify-split`` subcommand. Splitting lets
    a streaming caller run the heavy time-invariant front-end at one
    cadence and the cheap time-pooling head at a finer one.

Usage
-----
    python scripts/export_ecapa_onnx.py export \\
        --output build/ecapa_tdnn.onnx
    python scripts/export_ecapa_onnx.py verify \\
        --onnx build/ecapa_tdnn.onnx --tol 1e-4

    # Both in one go (export then verify the freshly produced file):
    python scripts/export_ecapa_onnx.py export-and-verify \\
        --output build/ecapa_tdnn.onnx

    # #119 split: export the two halves, then verify they recombine.
    python scripts/export_ecapa_onnx.py export \\
        --mode encoder --output build/ecapa_encoder.onnx
    python scripts/export_ecapa_onnx.py export \\
        --mode pooler --output build/ecapa_pooler.onnx
    python scripts/export_ecapa_onnx.py verify-split \\
        --encoder build/ecapa_encoder.onnx --pooler build/ecapa_pooler.onnx

Notes
-----
* Requires the `models` extra (`pip install -e poc[models]`) for torch,
  speechbrain, and onnxruntime.
* The exported graph fixes the sample rate to 16 kHz (SpeechBrain's
  ``compute_features`` pipeline is hard-coded to that rate). The time
  axis is dynamic so the same .onnx works for any clip length ≥ 1 s.
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
DEFAULT_MEL_BINS = 80
DEFAULT_MODE = "embedding-only"


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


def _load_classifier(source: str, savedir: str | None):  # noqa: ANN202
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import-not-found]

    return EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": "cpu"},
    )


def _build_full_wrapper(classifier) -> "torch.nn.Module":  # noqa: ANN001
    """Module: (B, T) float32 → (B, 192) float32 — known to fail ONNX export today."""
    import torch

    compute_features = classifier.mods.compute_features
    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model

    class FullWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.compute_features = compute_features
            self.mean_var_norm = mean_var_norm
            self.embedding_model = embedding_model

        def forward(self, wav: "torch.Tensor") -> "torch.Tensor":
            feats = self.compute_features(wav)
            wav_lens = torch.ones(wav.shape[0], device=wav.device)
            normed = self.mean_var_norm(feats, wav_lens)
            emb = self.embedding_model(normed, wav_lens)
            return emb.squeeze(1)

    wrapper = FullWrapper()
    wrapper.eval()
    return wrapper


def _build_features(classifier) -> "torch.nn.Module":  # noqa: ANN001
    """Module: (B, T) → (B, T_frames, n_mels) — SpeechBrain Fbank."""
    import torch

    compute_features = classifier.mods.compute_features

    class Features(torch.nn.Module):
        def forward(self, wav: "torch.Tensor") -> "torch.Tensor":
            return compute_features(wav)

    fe = Features()
    fe.eval()
    return fe


def _build_embedding_only_wrapper(classifier) -> "torch.nn.Module":  # noqa: ANN001
    """Module: (B, T_frames, n_mels) → (B, 192) — ONNX-exportable subgraph."""
    import torch

    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model

    class EmbeddingOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mean_var_norm = mean_var_norm
            self.embedding_model = embedding_model

        def forward(self, feats: "torch.Tensor") -> "torch.Tensor":
            wav_lens = torch.ones(feats.shape[0], device=feats.device)
            normed = self.mean_var_norm(feats, wav_lens)
            emb = self.embedding_model(normed, wav_lens)
            return emb.squeeze(1)

    eo = EmbeddingOnly()
    eo.eval()
    return eo


def _build_encoder_wrapper(classifier) -> "torch.nn.Module":  # noqa: ANN001
    """Module: (B, T_frames, n_mels) → (B, C_mfa, T_frames) — #119.

    The first half of the embedding-only graph: mean_var_norm + the
    ECAPA blocks + multi-layer feature aggregation, stopping *before*
    the attentive statistics pooling. Output is the per-frame feature
    map; pairing this with :func:`_build_pooler_wrapper` reproduces the
    monolithic ``embedding-only`` graph exactly. Splitting here lets a
    streaming caller run the (heavy, time-invariant) conv front-end at
    one cadence and the (cheap, time-pooling) head at a finer one.
    """
    import torch
    from speechbrain.lobes.models.ECAPA_TDNN import TDNNBlock

    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mean_var_norm = mean_var_norm
            self.embedding_model = embedding_model

        def forward(self, feats: "torch.Tensor") -> "torch.Tensor":
            wav_lens = torch.ones(feats.shape[0], device=feats.device)
            normed = self.mean_var_norm(feats, wav_lens)
            model = self.embedding_model
            # Mirror ECAPA_TDNN.forward up to and including `mfa`.
            x = normed.transpose(1, 2)
            xl = []
            for layer in model.blocks:
                if isinstance(layer, TDNNBlock):
                    x = layer(x)
                else:
                    x = layer(x, lengths=wav_lens)
                xl.append(x)
            x = torch.cat(xl[1:], dim=1)
            return model.mfa(x)

    enc = Encoder()
    enc.eval()
    return enc


def _build_pooler_wrapper(classifier) -> "torch.nn.Module":  # noqa: ANN001
    """Module: (B, C_mfa, T_frames) → (B, 192) — #119.

    The second half of the embedding-only graph: attentive statistics
    pooling + batchnorm + the final linear layer. Consumes the
    per-frame feature map produced by :func:`_build_encoder_wrapper`.
    """
    import torch

    embedding_model = classifier.mods.embedding_model

    class Pooler(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding_model = embedding_model

        def forward(self, frame_features: "torch.Tensor") -> "torch.Tensor":
            model = self.embedding_model
            wav_lens = torch.ones(frame_features.shape[0], device=frame_features.device)
            # Mirror ECAPA_TDNN.forward from `asp` onward.
            x = model.asp(frame_features, lengths=wav_lens)
            x = model.asp_bn(x)
            x = model.fc(x)
            return x.transpose(1, 2).squeeze(1)

    pool = Pooler()
    pool.eval()
    return pool


def _probe_mfa_channels(classifier, sample_rate: int) -> int:  # noqa: ANN001
    """Run the encoder wrapper once to learn the MFA output channel
    count (C_mfa) — it depends on the checkpoint, so it is not
    hard-coded."""
    import torch

    fe = _build_features(classifier)
    enc = _build_encoder_wrapper(classifier)
    with torch.no_grad():
        feats = fe(torch.zeros(1, sample_rate, dtype=torch.float32))
        frame_features = enc(feats)
    return int(frame_features.shape[1])


def _dummy_features(sr: int, duration_sec: float = 3.0) -> "torch.Tensor":
    import torch

    return torch.zeros(1, int(sr * duration_sec / 160), DEFAULT_MEL_BINS, dtype=torch.float32)


def cmd_export(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] mode={args.mode} loading {args.source}", file=sys.stderr)
    classifier = _load_classifier(args.source, args.savedir)

    output_names = ["embedding"]
    expect_embedding_dim = True
    if args.mode == "full":
        wrapper = _build_full_wrapper(classifier)
        dummy = torch.zeros(1, args.sample_rate * 3, dtype=torch.float32)
        input_names = ["waveform"]
        dynamic_axes = {
            "waveform": {0: "batch", 1: "samples"},
            "embedding": {0: "batch"},
        }
    elif args.mode == "embedding-only":
        # Probe true mel-bin count by running compute_features once.
        fe = _build_features(classifier)
        with torch.no_grad():
            probe = fe(torch.zeros(1, args.sample_rate, dtype=torch.float32))
        n_mels = probe.shape[-1]
        wrapper = _build_embedding_only_wrapper(classifier)
        dummy = torch.zeros(1, 100, n_mels, dtype=torch.float32)
        input_names = ["features"]
        dynamic_axes = {
            "features": {0: "batch", 1: "frames"},
            "embedding": {0: "batch"},
        }
        print(f"[export] embedding-only n_mels={n_mels}", file=sys.stderr)
    elif args.mode == "encoder":
        # First half of the embedding-only graph (#119): features →
        # per-frame feature map, stopping before the time pooling.
        fe = _build_features(classifier)
        with torch.no_grad():
            probe = fe(torch.zeros(1, args.sample_rate, dtype=torch.float32))
        n_mels = probe.shape[-1]
        wrapper = _build_encoder_wrapper(classifier)
        dummy = torch.zeros(1, 100, n_mels, dtype=torch.float32)
        input_names = ["features"]
        output_names = ["frame_features"]
        expect_embedding_dim = False
        dynamic_axes = {
            "features": {0: "batch", 1: "frames"},
            "frame_features": {0: "batch", 2: "frames"},
        }
        print(f"[export] encoder n_mels={n_mels}", file=sys.stderr)
    elif args.mode == "pooler":
        # Second half of the embedding-only graph (#119): per-frame
        # feature map → embedding.
        c_mfa = _probe_mfa_channels(classifier, args.sample_rate)
        wrapper = _build_pooler_wrapper(classifier)
        dummy = torch.zeros(1, c_mfa, 100, dtype=torch.float32)
        input_names = ["frame_features"]
        dynamic_axes = {
            "frame_features": {0: "batch", 2: "frames"},
            "embedding": {0: "batch"},
        }
        print(f"[export] pooler c_mfa={c_mfa}", file=sys.stderr)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    with torch.no_grad():
        out = wrapper(dummy)
    if expect_embedding_dim and out.shape[-1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"unexpected embedding dim {out.shape[-1]}, expected {EMBEDDING_DIM}; "
            f"the SpeechBrain checkpoint may have changed"
        )

    print(f"[export] writing {output}", file=sys.stderr)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
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

    print(f"[verify] mode={args.mode} loading torch model {args.source}", file=sys.stderr)
    classifier = _load_classifier(args.source, args.savedir)
    full_ref = _build_full_wrapper(classifier)
    fe = _build_features(classifier) if args.mode == "embedding-only" else None

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
            emb_t = full_ref(wav_t).cpu().numpy().astype(np.float64)
            if args.mode == "full":
                onnx_input = wav[None, :]
                onnx_input_name = "waveform"
            else:
                feats_t = fe(wav_t).cpu().numpy().astype(np.float32)
                onnx_input = feats_t
                onnx_input_name = "features"
        emb_o = session.run(["embedding"], {onnx_input_name: onnx_input})[0].astype(np.float64)
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


def cmd_verify_split(args: argparse.Namespace) -> int:
    """Verify the #119 encoder + pooler split reproduces the monolithic
    embedding-only graph: for each test clip, run
    ``pooler(encoder(features))`` through ONNX Runtime and compare the
    embedding (raw + pairwise-cosine) against the full PyTorch
    reference."""
    import onnxruntime as ort  # type: ignore[import-not-found]
    import torch

    enc_path = Path(args.encoder)
    pool_path = Path(args.pooler)
    for p in (enc_path, pool_path):
        if not p.exists():
            print(f"[verify-split] missing {p}; run `export` first", file=sys.stderr)
            return 2

    print(f"[verify-split] loading torch reference {args.source}", file=sys.stderr)
    classifier = _load_classifier(args.source, args.savedir)
    full_ref = _build_full_wrapper(classifier)
    fe = _build_features(classifier)

    print(f"[verify-split] loading onnx {enc_path} + {pool_path}", file=sys.stderr)
    enc_sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    pool_sess = ort.InferenceSession(str(pool_path), providers=["CPUExecutionProvider"])

    waves: list[np.ndarray] = []
    labels: list[str] = []
    for i, f0 in enumerate((140.0, 180.0, 220.0)):
        waves.append(_synth_waveform(seed=42 + i, sr=args.sample_rate, duration_sec=3.0, f0=f0))
        labels.append(f"synth_{i}_f{int(f0)}")
    for path_str in args.audio or []:
        wav = _load_audio(Path(path_str), args.sample_rate)
        if wav.size < args.sample_rate:
            print(f"[verify-split] skip {path_str}: shorter than 1 s", file=sys.stderr)
            continue
        waves.append(wav)
        labels.append(Path(path_str).stem)

    torch_embs: list[np.ndarray] = []
    onnx_embs: list[np.ndarray] = []
    raw_max_delta = 0.0
    for wav, label in zip(waves, labels, strict=True):
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            emb_t = full_ref(wav_t).cpu().numpy().astype(np.float64)
            feats = fe(wav_t).cpu().numpy().astype(np.float32)
        frame_features = enc_sess.run(["frame_features"], {"features": feats})[0]
        emb_o = pool_sess.run(
            ["embedding"], {"frame_features": frame_features}
        )[0].astype(np.float64)
        delta = float(np.max(np.abs(emb_t - emb_o)))
        raw_max_delta = max(raw_max_delta, delta)
        print(f"[verify-split] {label:24s} raw max|Δ|={delta:.3e}", file=sys.stderr)
        torch_embs.append(emb_t[0])
        onnx_embs.append(emb_o[0])

    torch_arr = np.stack(torch_embs, axis=0)
    onnx_arr = np.stack(onnx_embs, axis=0)
    cos_max_delta = float(
        np.max(np.abs(_cosine_matrix(torch_arr, torch_arr) - _cosine_matrix(onnx_arr, onnx_arr)))
    )

    print("", file=sys.stderr)
    print(f"[verify-split] N={len(waves)} clips", file=sys.stderr)
    print(f"[verify-split] raw embedding max|Δ|     = {raw_max_delta:.3e}", file=sys.stderr)
    print(f"[verify-split] cosine-similarity max|Δ| = {cos_max_delta:.3e}", file=sys.stderr)
    print(f"[verify-split] tolerance                = {args.tol:.3e}", file=sys.stderr)

    if cos_max_delta < args.tol:
        print("[verify-split] PASS — split graph matches the monolithic graph", file=sys.stderr)
        return 0
    print("[verify-split] FAIL — split parity exceeds tolerance", file=sys.stderr)
    return 1


def cmd_quantize(args: argparse.Namespace) -> int:
    """Dynamic INT8 quantization of an existing FP32 ECAPA ONNX.

    Uses ``onnxruntime.quantization.quantize_dynamic`` — weight-only INT8
    with activations dynamically quantized at runtime. No calibration
    dataset required.

    Measured trade-off on the AVX-512 + VNNI dev VM (ORT 1.25.1):

    * QInt8  (signed):     cosine max|Δ| = 6.6e-3 ✅ but
                           ECAPA 40 ms → 80 ms (**regression** —
                           the QInt8 conv1d kernel path isn't tuned
                           for this hardware).
    * QUInt8 (unsigned):   cosine max|Δ| = 3.2e-2 (above the 1e-2
                           bar but only 1.4 % relative to the AS-Norm
                           gate threshold 2.25; production gating
                           still decides correctly on synth fixtures)
                           and ECAPA 40 ms → 35 ms (-14 %),
                           pipeline 324 ms → 280 ms (-14 %).

    Neither setting reaches the 2–3× Microsoft reports for
    transformer-heavy networks — ECAPA-TDNN is conv1d-dominant and
    the dynamic-quant overhead eats most of the win. The Rust side
    needs no changes: ``MELLONELLA_ECAPA_ONNX`` can point at either
    flavour, this script just helps build them.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as exc:  # pragma: no cover
        print(
            "[quantize] missing dependency: pip install onnxruntime onnx onnxruntime-tools",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        print(f"[quantize] missing {src}; run `export` first", file=sys.stderr)
        return 2
    dst.parent.mkdir(parents=True, exist_ok=True)

    weight_type = QuantType.QUInt8 if args.unsigned else QuantType.QInt8
    print(f"[quantize] {src} → {dst} (weight_type={weight_type})", file=sys.stderr)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=weight_type,
    )
    size_in = src.stat().st_size / 1e6
    size_out = dst.stat().st_size / 1e6
    print(
        f"[quantize] done — {size_in:.1f} MB → {size_out:.1f} MB "
        f"({size_out / size_in * 100:.1f} % of original)",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", default=DEFAULT_SOURCE)
    common.add_argument("--savedir", default=None)
    common.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    common.add_argument(
        "--mode",
        choices=("full", "embedding-only", "encoder", "pooler"),
        default=DEFAULT_MODE,
        help="full = waveform→embedding (broken under torch 2.4 STFT export); "
        "embedding-only = features→embedding (default); "
        "encoder = features→frame_features (#119, first half); "
        "pooler = frame_features→embedding (#119, second half)",
    )

    p_export = sub.add_parser("export", parents=[common])
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--opset", type=int, default=17)
    p_export.set_defaults(func=cmd_export)

    p_verify = sub.add_parser("verify", parents=[common])
    p_verify.add_argument("--onnx", required=True)
    p_verify.add_argument("--audio", nargs="*", help="optional WAV/FLAC files for additional parity checks")
    p_verify.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_verify.set_defaults(func=cmd_verify)

    p_verify_split = sub.add_parser("verify-split", parents=[common])
    p_verify_split.add_argument("--encoder", required=True, help="encoder ONNX (#119)")
    p_verify_split.add_argument("--pooler", required=True, help="pooler ONNX (#119)")
    p_verify_split.add_argument(
        "--audio", nargs="*", help="optional WAV/FLAC files for additional parity checks"
    )
    p_verify_split.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_verify_split.set_defaults(func=cmd_verify_split)

    p_both = sub.add_parser("export-and-verify", parents=[common])
    p_both.add_argument("--output", required=True)
    p_both.add_argument("--opset", type=int, default=17)
    p_both.add_argument("--audio", nargs="*")
    p_both.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p_both.set_defaults(func=cmd_export_and_verify)

    p_quant = sub.add_parser("quantize")
    p_quant.add_argument("--input", required=True, help="FP32 ECAPA ONNX file")
    p_quant.add_argument("--output", required=True, help="output INT8 ONNX file")
    p_quant.add_argument(
        "--unsigned",
        action="store_true",
        help="use QUInt8 weights (default is QInt8). UInt8 sometimes maps to faster "
        "x86_64 kernels on hosts where the QInt8 path isn't optimised.",
    )
    p_quant.set_defaults(func=cmd_quantize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
