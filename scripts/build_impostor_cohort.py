#!/usr/bin/env python3
"""Build an impostor embedding cohort for AS-Norm score normalization.

Background
----------
Per ``docs/decisions.md`` D-010, mellonella's gating layer adopts
**Adaptive S-Norm (AS-Norm)** to remove per-language / per-condition score
distribution drift instead of maintaining per-language θ_pass overrides.
AS-Norm needs a small cohort of impostor embeddings (random non-target
speakers) against which the runtime cosine similarity is normalised:

    S_norm = (S_target - μ_top-K(S_impostor)) / σ_top-K(S_impostor)

This script builds that cohort offline by walking one or more
manifest.csv files (the same format produced by
``mellonella_bench.datasets.{commonvoice,mls,emilia}``), running each
selected speaker's concatenated audio through ECAPA-TDNN, and packing
the resulting 192-dim embeddings into an ``.npz`` file.

Usage
-----
::

    python scripts/build_impostor_cohort.py \
        --manifest en=$MELLONELLA_DATA_DIR/emilia_yodas/en/manifest.csv \
        --manifest de=$MELLONELLA_DATA_DIR/mls/de/manifest.csv \
        --manifest ja=$MELLONELLA_DATA_DIR/emilia_yodas/ja/manifest.csv \
        --per-language 5 \
        --output bench/data/cohorts/impostor_cohort_v1.npz

The output ``.npz`` carries:

* ``embeddings``  — ``(N, 192) float32``, L2-normalised so cosine sim
                    reduces to a dot product at runtime
* ``languages``   — ``(N,)`` array of ISO codes (object dtype)
* ``speaker_ids`` — ``(N,)`` array of upstream speaker labels (object dtype)

The file is small (50 speakers × 192 dims × 4 B ≈ 38 KB) and is
intended to be checked in alongside the bench artifacts so the runtime
pipeline can load it without a network round-trip.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT / "poc"))

from mellonella_bench.datasets.commonvoice import (  # noqa: E402
    load_speakers_from_manifest,
)

SAMPLE_RATE = 16_000
DEFAULT_PER_LANGUAGE = 5
DEFAULT_MIN_SECONDS = 5.0
EMBEDDING_DIM = 192
EPS = 1e-12


@dataclass
class ManifestSpec:
    """One ``LANG=PATH`` argument from the CLI."""

    language: str
    manifest: Path

    @classmethod
    def parse(cls, raw: str) -> ManifestSpec:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"--manifest expects 'LANG=PATH', got {raw!r}"
            )
        lang, path = raw.split("=", 1)
        lang = lang.strip()
        path = path.strip()
        if not lang or not path:
            raise argparse.ArgumentTypeError(
                f"--manifest expects non-empty 'LANG=PATH', got {raw!r}"
            )
        return cls(language=lang, manifest=Path(path))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit L2 norm (zero-vector safe)."""
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32)


def select_speakers_for_language(
    manifest_path: Path,
    *,
    per_language: int,
    min_seconds: float,
) -> list[tuple[str, np.ndarray]]:
    """Return up to ``per_language`` (speaker_id, audio) pairs for one manifest.

    Speakers are ranked by total concatenated audio length (longest
    first). ``load_speakers_from_manifest`` already drops speakers below
    ``min_seconds``; the longest few are then taken from the survivors.
    """
    raw = load_speakers_from_manifest(
        manifest_path, SAMPLE_RATE, min_seconds=min_seconds
    )
    if not raw:
        raise RuntimeError(
            f"manifest {manifest_path} yielded zero usable speakers "
            f"after the {min_seconds}s filter"
        )
    items = [(spk, np.asarray(audio, dtype=np.float32)) for spk, audio in raw.items()]
    items.sort(key=lambda kv: kv[1].size, reverse=True)
    return items[:per_language]


def embed_speakers(
    selections: list[tuple[str, np.ndarray]],
    *,
    embed_fn,
) -> np.ndarray:
    """Apply ``embed_fn`` to each selected audio buffer, L2-normalise the result."""
    embeddings: list[np.ndarray] = []
    for _, audio in selections:
        raw_emb = embed_fn(audio)
        embeddings.append(l2_normalize(np.asarray(raw_emb, dtype=np.float32)))
    if not embeddings:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.stack(embeddings, axis=0).astype(np.float32)


def build_cohort(
    specs: list[ManifestSpec],
    *,
    per_language: int = DEFAULT_PER_LANGUAGE,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    embed_fn=None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Aggregate per-language selections + their embeddings into a flat cohort.

    Returns ``(embeddings, languages, speaker_ids)`` where ``embeddings``
    is shape ``(N, EMBEDDING_DIM)`` and the metadata lists have length N.
    ``embed_fn`` defaults to a freshly-instantiated
    :class:`mellonella_poc.embedding.EcapaTdnn`; tests inject a mock.
    """
    if per_language <= 0:
        raise ValueError("per_language must be > 0")
    if embed_fn is None:
        from mellonella_poc.embedding import EcapaTdnn

        embed_fn = EcapaTdnn(sample_rate=SAMPLE_RATE).embed

    all_embeddings: list[np.ndarray] = []
    all_languages: list[str] = []
    all_speakers: list[str] = []

    for spec in specs:
        selections = select_speakers_for_language(
            spec.manifest,
            per_language=per_language,
            min_seconds=min_seconds,
        )
        embeddings = embed_speakers(selections, embed_fn=embed_fn)
        all_embeddings.append(embeddings)
        for spk_id, _ in selections:
            all_languages.append(spec.language)
            all_speakers.append(spk_id)

    if not all_embeddings or all(e.size == 0 for e in all_embeddings):
        raise RuntimeError("cohort build produced zero embeddings; check manifests")

    matrix = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    return matrix, all_languages, all_speakers


def save_cohort(
    embeddings: np.ndarray,
    languages: list[str],
    speaker_ids: list[str],
    output_path: Path,
) -> None:
    """Persist the cohort under ``output_path`` as ``.npz`` + a sidecar manifest."""
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"expected {EMBEDDING_DIM}-dim embeddings, got shape {embeddings.shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        languages=np.asarray(languages, dtype=object),
        speaker_ids=np.asarray(speaker_ids, dtype=object),
    )
    summary = {
        "n_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "languages": sorted(set(languages)),
        "per_language_counts": {
            lang: int(sum(1 for x in languages if x == lang))
            for lang in sorted(set(languages))
        },
        # Per-language ordered list of upstream speaker IDs that fed the
        # cohort, for run-to-run diff inspection. Without this, AS-Norm
        # variance investigations (per `docs/decisions.md` D-010 Phase 2
        # follow-up) need to crack open the .npz to see what changed —
        # making the artifact upload from CI low-signal.
        "selected_speakers": {
            lang: [
                speaker_ids[i] for i in range(len(languages)) if languages[i] == lang
            ]
            for lang in sorted(set(languages))
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))


def load_cohort(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Inverse of :func:`save_cohort` — handy for the runtime AS-Norm path."""
    with np.load(path, allow_pickle=True) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        languages = [str(x) for x in data["languages"]]
        speaker_ids = [str(x) for x in data["speaker_ids"]]
    return embeddings, languages, speaker_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=ManifestSpec.parse,
        required=True,
        help="LANG=PATH ; repeat per language (e.g. --manifest ja=/data/.../manifest.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output .npz path (parent dir is created if missing)",
    )
    parser.add_argument(
        "--per-language",
        type=int,
        default=DEFAULT_PER_LANGUAGE,
        help=f"speakers to keep per language (default: {DEFAULT_PER_LANGUAGE})",
    )
    parser.add_argument(
        "--min-seconds-per-speaker",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help=f"drop speakers with less than N s of audio (default: {DEFAULT_MIN_SECONDS})",
    )
    args = parser.parse_args(argv)

    embeddings, languages, speaker_ids = build_cohort(
        args.manifest,
        per_language=args.per_language,
        min_seconds=args.min_seconds_per_speaker,
    )
    save_cohort(embeddings, languages, speaker_ids, args.output)

    print(
        json.dumps(
            {
                "n_embeddings": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "per_language_counts": {
                    lang: int(sum(1 for x in languages if x == lang))
                    for lang in sorted(set(languages))
                },
                "selected_speakers": {
                    lang: [
                        speaker_ids[i]
                        for i in range(len(languages))
                        if languages[i] == lang
                    ]
                    for lang in sorted(set(languages))
                },
                "output": str(args.output),
                "summary": str(args.output.with_suffix(".json")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
