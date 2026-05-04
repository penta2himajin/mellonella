"""Emilia-YODAS subset preparation.

License (Emilia-YODAS):
    CC-BY 4.0 (commercial use OK). Source dataset:
    https://huggingface.co/datasets/amphion/Emilia-Dataset

Emilia-YODAS is the YouTube-CC-BY-3.0-derived subset of the Emilia
collection (the parent dataset also contains a CC BY-NC-4.0 ``Emilia``
half — we explicitly load only the YODAS shards). It covers six
languages — **including ja, ko and zh** that MLS lacks — and ships a
real per-clip ``speaker`` field, making it the FOSS-licensed Asian
language story for Scenario 5.

Access requirements
-------------------
The HuggingFace repo is **gated** — callers must agree to the dataset
ToS once on the HF site, then provide an HF read token. Pass it via
``--hf-token`` on the CLI or the ``HF_TOKEN`` env var. CI workflows that
don't have ``HF_TOKEN`` configured will see the prep step fail loudly,
which is intentional: silent skipping would mask coverage regressions.

Loading interface
-----------------
Per the upstream loading example, Emilia-YODAS is filtered via
``data_files`` glob patterns (no HF config names): the language-specific
subset lives under ``Emilia-YODAS/<UPPER>/*.tar``. The samples come out
of the WebDataset loader as dicts with ``mp3``/``json`` keys; we accept
both shapes in :func:`_extract_audio` / :func:`_extract_metadata` so the
module survives minor upstream layout drift.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
from collections import Counter
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .common import default_data_dir
from .commonvoice import CommonVoiceClip, write_manifest

SAMPLE_RATE = 16_000
DEFAULT_TOP_SPEAKERS = 3
DEFAULT_CLIPS_PER_SPEAKER = 4

LANGUAGE_TO_EMILIA_DIR: dict[str, str] = {
    "en": "EN",
    "zh-CN": "ZH",
    "de": "DE",
    "fr": "FR",
    "ja": "JA",
    "ko": "KO",
}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_TO_EMILIA_DIR)
DATASET_REPO = "amphion/Emilia-Dataset"


def _resample_to_target(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if src_sr == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    g = gcd(int(src_sr), int(SAMPLE_RATE))
    return resample_poly(audio, SAMPLE_RATE // g, src_sr // g).astype(np.float32)


def _extract_audio(sample: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Pull (array, sample_rate) from one Emilia WebDataset sample.

    The HF WebDataset loader sometimes auto-decodes the mp3 into
    ``{"array", "sampling_rate"}`` and sometimes hands back raw bytes;
    accept both. Falls back to ``audio`` for the JSONL-shape mock used
    by tests.
    """
    blob = sample.get("mp3") or sample.get("audio") or sample.get("flac")
    if blob is None:
        raise KeyError("emilia sample missing audio (mp3/audio/flac)")
    if isinstance(blob, dict) and "array" in blob:
        return np.asarray(blob["array"], dtype=np.float32), int(blob["sampling_rate"])
    if isinstance(blob, (bytes, bytearray)):
        audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr)
    raise TypeError(f"unsupported emilia audio payload type: {type(blob)!r}")


def _extract_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON metadata dict for a sample (handles both shapes)."""
    if isinstance(sample.get("json"), dict):
        return sample["json"]
    return sample


def prepare(
    language: str,
    output_dir: Path,
    *,
    top_speakers: int = DEFAULT_TOP_SPEAKERS,
    clips_per_speaker: int = DEFAULT_CLIPS_PER_SPEAKER,
    max_stream: int = 5_000,
    hf_token: str | None = None,
) -> Path:
    """Stream Emilia-YODAS for ``language``, write top-K speaker subset.

    Idempotent on ``output_dir/manifest.csv``. ``hf_token`` defaults to
    ``$HF_TOKEN`` so CI workflows can wire it via secret without changing
    the call site.
    """
    if language not in LANGUAGE_TO_EMILIA_DIR:
        raise ValueError(
            f"language {language!r} not in supported Emilia-YODAS set {SUPPORTED_LANGUAGES}"
        )
    if top_speakers <= 0:
        raise ValueError("top_speakers must be > 0")
    if clips_per_speaker <= 0:
        raise ValueError("clips_per_speaker must be > 0")

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        return output_dir

    token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Emilia-YODAS is a gated HuggingFace dataset; pass hf_token=... "
            "or set $HF_TOKEN to a read-scoped token after agreeing to the "
            f"dataset ToS at https://huggingface.co/datasets/{DATASET_REPO}"
        )

    from datasets import load_dataset

    pattern = f"Emilia-YODAS/{LANGUAGE_TO_EMILIA_DIR[language]}/*.tar"
    ds = load_dataset(
        DATASET_REPO,
        data_files={"train": pattern},
        split="train",
        streaming=True,
        token=token,
    )

    by_speaker: dict[str, list[dict]] = {}
    n_seen = 0
    for sample in ds:
        if n_seen >= max_stream:
            break
        n_seen += 1
        meta = _extract_metadata(sample)
        speaker = meta.get("speaker")
        if not speaker:
            continue
        spk = str(speaker)
        bucket = by_speaker.setdefault(spk, [])
        if len(bucket) >= clips_per_speaker:
            continue
        try:
            audio_array, sr = _extract_audio(sample)
        except (KeyError, TypeError):
            continue
        bucket.append(
            {
                "audio": audio_array,
                "sr": sr,
                "text": str(meta.get("text") or ""),
            }
        )
        full = sum(1 for b in by_speaker.values() if len(b) >= clips_per_speaker)
        if full >= top_speakers:
            break

    counts = Counter({spk: len(clips) for spk, clips in by_speaker.items()})
    chosen_ids = [spk for spk, _ in counts.most_common(top_speakers)]
    chosen = [(spk, by_speaker[spk]) for spk in chosen_ids if by_speaker[spk]]
    if len(chosen) < top_speakers:
        raise RuntimeError(
            f"Emilia-YODAS({language}) yielded only {len(chosen)} speaker(s) "
            f"after {n_seen} samples; need {top_speakers}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[CommonVoiceClip] = []
    for speaker_idx, (raw_speaker, clips) in enumerate(chosen, start=1):
        speaker_label = f"speaker{speaker_idx:02d}"
        spk_dir = output_dir / speaker_label
        spk_dir.mkdir(parents=True, exist_ok=True)
        for clip_idx, clip in enumerate(clips):
            audio = _resample_to_target(clip["audio"], clip["sr"])
            rel = Path(speaker_label) / f"{clip_idx:03d}.wav"
            sf.write(str(output_dir / rel), audio, SAMPLE_RATE)
            manifest.append(
                CommonVoiceClip(
                    language=language,
                    speaker_id=speaker_label,
                    clip_path=rel,
                    sentence=clip["text"],
                )
            )
        (spk_dir / "_emilia_speaker.txt").write_text(str(raw_speaker))

    write_manifest(manifest, manifest_path)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emilia-YODAS subset preparation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="stream Emilia-YODAS + materialise per-speaker subset")
    p_prep.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="ISO language code; mapped to the Emilia-YODAS subdirectory",
    )
    p_prep.add_argument("--data-dir", type=Path, default=None)
    p_prep.add_argument("--top-speakers", type=int, default=DEFAULT_TOP_SPEAKERS)
    p_prep.add_argument("--clips-per-speaker", type=int, default=DEFAULT_CLIPS_PER_SPEAKER)
    p_prep.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace read token; falls back to $HF_TOKEN",
    )

    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        root = args.data_dir if args.data_dir is not None else default_data_dir() / "emilia_yodas"
        output_dir = root / args.language
        prepare(
            args.language,
            output_dir,
            top_speakers=args.top_speakers,
            clips_per_speaker=args.clips_per_speaker,
            hf_token=args.hf_token,
        )
        print(f"  output: {output_dir}")
        print(f"  manifest: {output_dir / 'manifest.csv'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
