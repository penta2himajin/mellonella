"""Multilingual LibriSpeech (MLS) subset preparation.

License (MLS):
    CC-BY 4.0 (re-distributable, attribution required). Source dataset:
    https://huggingface.co/datasets/facebook/multilingual_librispeech

MLS covers eight European languages with real per-clip ``speaker_id``
labels (~60+ speakers per language in the test split), making it a
clean fit for Scenario 5's target/other pair selection. The dataset is
ungated on HuggingFace, so the prep step needs no token.

The module emits the same flat per-speaker manifest format as
:mod:`mellonella_bench.datasets.commonvoice` so downstream code
(calibration, scenario_5 runner, ``scripts/scenario_5_from_manifest.py``)
consumes both interchangeably.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import default_data_dir
from .commonvoice import CommonVoiceClip, write_manifest

SAMPLE_RATE = 16_000
DEFAULT_TOP_SPEAKERS = 10
DEFAULT_CLIPS_PER_SPEAKER = 4
DEFAULT_SPLIT = "test"
# We over-collect per speaker (up to OVERSAMPLE_FACTOR × clips_per_speaker)
# so the post-stream deterministic clip sort has real material to choose
# from. 4× is enough to absorb streaming-order shuffle for most MLS test
# splits without blowing memory on 5 000-sample windows.
OVERSAMPLE_FACTOR = 4

# MLS HF config names spell out the language. We expose short ISO codes
# at the public API and resolve to the long names internally so the
# manifest ``language`` column stays compact and stable.
#
# NOTE: English was DROPPED from facebook/multilingual_librispeech when the
# repo migrated to parquet. The currently-published configs are exactly the
# seven below — `english` is no longer accessible on HF. Use Emilia-YODAS
# (datasets/emilia.py) for English instead.
LANGUAGE_TO_MLS_CONFIG: dict[str, str] = {
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_TO_MLS_CONFIG)


def _resample_to_target(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Convert audio to mono float32 at :data:`SAMPLE_RATE`."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if src_sr == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    g = gcd(int(src_sr), int(SAMPLE_RATE))
    return resample_poly(audio, SAMPLE_RATE // g, src_sr // g).astype(np.float32)


def _extract_audio(sample: dict) -> tuple[np.ndarray, int]:
    """Pull the (array, sample_rate) pair from one MLS sample."""
    audio = sample["audio"]
    return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])


def _clip_hash_key(clip: dict) -> str:
    """Deterministic per-clip sort key based on the audio content itself."""
    audio = np.ascontiguousarray(clip["audio"], dtype=np.float32)
    return hashlib.sha1(audio.tobytes()).hexdigest()


def _stable_pick_clips(clips: list[dict], k: int) -> list[dict]:
    """Pick the first ``k`` clips after sorting by content hash."""
    return sorted(clips, key=_clip_hash_key)[:k]


def _extract_text(sample: dict) -> str:
    """Pull the transcript from one MLS sample (field name varies by version)."""
    for key in ("text", "transcript", "transcription"):
        value = sample.get(key)
        if value:
            return str(value)
    return ""


def prepare(
    language: str,
    output_dir: Path,
    *,
    top_speakers: int = DEFAULT_TOP_SPEAKERS,
    clips_per_speaker: int = DEFAULT_CLIPS_PER_SPEAKER,
    split: str = DEFAULT_SPLIT,
    max_stream: int = 5_000,
) -> Path:
    """Stream MLS for ``language``, materialise top-K speakers + manifest.csv.

    Idempotent: returns immediately if ``output_dir/manifest.csv`` exists.

    The function pulls up to ``max_stream`` samples from the requested
    split (default ``test``), buckets them by ``speaker_id``, picks the
    ``top_speakers`` most-clipped speakers and keeps the first
    ``clips_per_speaker`` clips each. Audio is resampled to 16 kHz and
    written as wav alongside a CV-shaped ``manifest.csv``.
    """
    if language not in LANGUAGE_TO_MLS_CONFIG:
        raise ValueError(f"language {language!r} not in supported MLS set {SUPPORTED_LANGUAGES}")
    if top_speakers <= 0:
        raise ValueError("top_speakers must be > 0")
    if clips_per_speaker <= 0:
        raise ValueError("clips_per_speaker must be > 0")

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        return output_dir

    config = LANGUAGE_TO_MLS_CONFIG[language]

    from datasets import load_dataset

    # MLS on HF was migrated to parquet; the legacy ``trust_remote_code`` opt-in
    # is no longer accepted (and not needed). The English config was also dropped
    # in the migration — see LANGUAGE_TO_MLS_CONFIG below.
    ds = load_dataset(
        "facebook/multilingual_librispeech",
        config,
        split=split,
        streaming=True,
    )

    # Scan a fixed window and over-collect clips per speaker. We don't
    # early-break on "first N speakers full" — that made the manifest
    # depend on HF streaming-arrival order, which is the root cause of
    # the cohort-drift observed in D-010 Phase 4.
    per_speaker_cap = clips_per_speaker * OVERSAMPLE_FACTOR
    by_speaker: dict[str, list[dict]] = {}
    n_seen = 0
    for sample in ds:
        if n_seen >= max_stream:
            break
        n_seen += 1
        speaker_raw = sample.get("speaker_id")
        if speaker_raw is None:
            continue
        spk = str(speaker_raw)
        bucket = by_speaker.setdefault(spk, [])
        if len(bucket) >= per_speaker_cap:
            continue
        bucket.append(
            {
                "audio": np.asarray(sample["audio"]["array"], dtype=np.float32),
                "sr": int(sample["audio"]["sampling_rate"]),
                "text": _extract_text(sample),
            }
        )

    # Speaker selection: most-clipped first with lex tiebreak so that two
    # streaming runs that see the same set of speakers + per-speaker
    # populations produce the same top-N selection.
    ranked = sorted(
        ((spk, clips) for spk, clips in by_speaker.items() if clips),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    selected = ranked[:top_speakers]
    if len(selected) < top_speakers:
        raise RuntimeError(
            f"MLS({config}, {split}) yielded only {len(selected)} speaker(s) "
            f"after {n_seen} samples; need {top_speakers}"
        )

    # Stable label assignment: re-sort the chosen speakers by upstream
    # speaker_id so ``speaker01`` always maps to the same upstream id given
    # the same selection set, regardless of clip-count tiebreaks.
    chosen = sorted(selected, key=lambda kv: kv[0])
    # Each speaker's clips are sorted by a content-addressed hash; this
    # decouples the kept-K-of-N decision from streaming-arrival order.
    chosen = [(spk, _stable_pick_clips(clips, clips_per_speaker)) for spk, clips in chosen]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[CommonVoiceClip] = []
    for speaker_idx, (raw_speaker_id, clips) in enumerate(chosen, start=1):
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
        # Stash the upstream MLS speaker id alongside as a sidecar so the
        # mapping back to the source dataset isn't lost (debugging aid only).
        (spk_dir / "_mls_speaker_id.txt").write_text(str(raw_speaker_id))

    write_manifest(manifest, manifest_path)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLS subset preparation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="stream MLS + materialise per-speaker subset")
    p_prep.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="ISO language code; mapped to the matching MLS HF config",
    )
    p_prep.add_argument("--data-dir", type=Path, default=None)
    p_prep.add_argument("--top-speakers", type=int, default=DEFAULT_TOP_SPEAKERS)
    p_prep.add_argument("--clips-per-speaker", type=int, default=DEFAULT_CLIPS_PER_SPEAKER)
    p_prep.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "validation", "test"],
        help=f"MLS split to stream (default: {DEFAULT_SPLIT})",
    )

    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        root = args.data_dir if args.data_dir is not None else default_data_dir() / "mls"
        output_dir = root / args.language
        prepare(
            args.language,
            output_dir,
            top_speakers=args.top_speakers,
            clips_per_speaker=args.clips_per_speaker,
            split=args.split,
        )
        print(f"  output: {output_dir}")
        print(f"  manifest: {output_dir / 'manifest.csv'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
