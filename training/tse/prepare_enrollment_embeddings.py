"""Pre-compute per-utterance ECAPA enrollment embeddings into an ``.npz``.

The TSE model is conditioned on a *frozen* 192-dim ECAPA enrollment
embedding (see :class:`~tse.model.FiLMConditioner`). The ECAPA model is
**not** part of the TSE model and is not trained — so the embeddings are
computed once, offline, and cached. The training data loader then just
looks each one up by utterance id (no ECAPA model in the training loop).

This script reuses the **existing** ECAPA ONNX (the one
``scripts/export_ecapa_onnx.py`` produces), located via the
``MELLONELLA_ECAPA_ONNX`` environment variable, and runs it through
``onnxruntime``. The default ECAPA ONNX is the ``embedding-only`` graph:
its input is the SpeechBrain Fbank feature tensor ``(B, T_frames, n_mels)``,
not raw audio (see the ``export_ecapa_onnx.py`` docstring and
``poc/mellonella_poc/embedding.py``). The Fbank front-end is reproduced
here via SpeechBrain's ``compute_features`` so the input contract matches
exactly.

.. note::

   This module **cannot be run-validated in Phase 2** — there is no ECAPA
   ONNX file in the local environment and SpeechBrain is not installed. It
   is written to the verified input contract and is exercised in Phase 3
   (Kaggle), where the ECAPA ONNX and SpeechBrain are available. It fails
   loudly if either dependency is missing.

Usage
-----
::

    export MELLONELLA_ECAPA_ONNX=build/ecapa_tdnn.onnx
    python -m tse.prepare_enrollment_embeddings \\
        --audio-dir data/librispeech/train-clean-100 \\
        --out build/tse/enroll_embeddings.npz

The output ``.npz`` maps ``utterance_id -> float32[192]``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ECAPA_ENV_VAR = "MELLONELLA_ECAPA_ONNX"
SAMPLE_RATE = 16_000
EMBEDDING_DIM = 192


def _resolve_ecapa_onnx(explicit: str | None) -> Path:
    """Find the ECAPA ONNX: explicit ``--ecapa-onnx`` else ``$MELLONELLA_ECAPA_ONNX``."""
    raw = explicit or os.environ.get(ECAPA_ENV_VAR)
    if not raw:
        raise RuntimeError(
            f"no ECAPA ONNX given — set ${ECAPA_ENV_VAR} or pass --ecapa-onnx "
            f"(produce one with scripts/export_ecapa_onnx.py)"
        )
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ECAPA ONNX not found: {path}")
    return path


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load ``path`` as mono float32 at ``target_sr`` (resampling if needed)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        try:
            import librosa  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"{path} is {sr} Hz, need {target_sr} Hz; install librosa to resample"
            ) from exc
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _build_fbank_extractor():  # noqa: ANN202
    """Return SpeechBrain's Fbank ``compute_features`` callable.

    Mirrors ``scripts/export_ecapa_onnx.py::_build_features`` — the ECAPA
    ``embedding-only`` ONNX consumes this Fbank tensor, not raw audio.
    """
    try:
        from speechbrain.inference.speaker import (  # type: ignore[import-not-found]
            EncoderClassifier,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "speechbrain not installed — needed to reproduce the ECAPA Fbank "
            "front-end. Install with `pip install -e poc[models]`."
        ) from exc
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    return classifier.mods.compute_features


def compute_embedding(
    audio: np.ndarray,
    compute_features,  # noqa: ANN001
    session,  # noqa: ANN001 - onnxruntime.InferenceSession
) -> np.ndarray:
    """Compute the 192-dim ECAPA embedding for one mono 16 kHz clip.

    ``audio`` -> SpeechBrain Fbank ``(1, T_frames, n_mels)`` -> ECAPA ONNX
    -> ``float32[192]``.
    """
    import torch

    if audio.ndim != 1:
        raise ValueError("compute_embedding expects a 1-D mono buffer")
    if audio.size < SAMPLE_RATE:
        raise ValueError(f"need >= 1 s of audio for ECAPA, got {audio.size / SAMPLE_RATE:.2f}s")
    with torch.no_grad():
        feats = compute_features(torch.from_numpy(audio).unsqueeze(0))
    feats_np = feats.cpu().numpy().astype(np.float32)
    emb = session.run(["embedding"], {"features": feats_np})[0]
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    if emb.shape[0] != EMBEDDING_DIM:
        raise RuntimeError(f"ECAPA ONNX returned dim {emb.shape[0]}, expected {EMBEDDING_DIM}")
    return emb


def _utterance_id(path: Path, audio_dir: Path) -> str:
    """Stable id for an utterance: its path relative to the audio root, no suffix."""
    rel = path.relative_to(audio_dir)
    return rel.with_suffix("").as_posix()


def prepare(
    audio_dir: Path,
    out_npz: Path,
    *,
    ecapa_onnx: Path,
    pattern: str = "*.flac",
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute embeddings for every ``pattern``-matching file under ``audio_dir``.

    Writes ``out_npz`` mapping ``utterance_id -> float32[192]`` and also
    returns the dict.
    """
    import onnxruntime as ort  # type: ignore[import-not-found]

    files = sorted(audio_dir.rglob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"no files matching {pattern!r} under {audio_dir}")

    print(f"[enroll] ECAPA ONNX: {ecapa_onnx}", file=sys.stderr)
    session = ort.InferenceSession(str(ecapa_onnx), providers=["CPUExecutionProvider"])
    compute_features = _build_fbank_extractor()

    embeddings: dict[str, np.ndarray] = {}
    for i, path in enumerate(files):
        audio = _load_audio(path, SAMPLE_RATE)
        uid = _utterance_id(path, audio_dir)
        embeddings[uid] = compute_embedding(audio, compute_features, session)
        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"[enroll] {i + 1}/{len(files)} embeddings", file=sys.stderr)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # numpy's savez stub mistypes **kwds; the call is correct at runtime.
    np.savez(out_npz, **embeddings)  # type: ignore[arg-type]
    print(f"[enroll] wrote {len(embeddings)} embeddings -> {out_npz}", file=sys.stderr)
    return embeddings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--audio-dir", type=Path, required=True, help="root of enrollment utterances"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output .npz (utterance_id -> float32[192])"
    )
    parser.add_argument(
        "--ecapa-onnx",
        default=None,
        help=f"ECAPA ONNX path (default: ${ECAPA_ENV_VAR})",
    )
    parser.add_argument(
        "--pattern", default="*.flac", help="glob for audio files (LibriSpeech: *.flac)"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the number of files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ecapa_onnx = _resolve_ecapa_onnx(args.ecapa_onnx)
    prepare(
        args.audio_dir,
        args.out,
        ecapa_onnx=ecapa_onnx,
        pattern=args.pattern,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
