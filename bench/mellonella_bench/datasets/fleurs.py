"""Google FLEURS subset preparation.

License (FLEURS):
    CC-BY 4.0 (re-distributable). Source dataset:
    https://huggingface.co/datasets/google/fleurs

FLEURS covers 102 languages (including ja, zh-CN, ko, ar that are
absent from MLS) and is distributed as parquet on HuggingFace, so no
short-lived signed URL or trust_remote_code dance is required.

Speaker-id caveat
-----------------
FLEURS does NOT expose a per-clip ``speaker_id`` field — only ``gender``
(binary). Scenario 5 needs a *target* and an *other* speaker, so we use
gender as a coarse speaker proxy:

* speaker01 ← all clips with ``gender == 0`` (typically male)
* speaker02 ← all clips with ``gender == 1`` (typically female)

This validates "ECAPA-TDNN gates target-vs-non-target across languages"
but does NOT isolate to specific speakers — the enrollment is an average
of a same-gender voice pool. For finer per-speaker calibration the user
should fall back to CommonVoice locally (see
``mellonella_bench.datasets.commonvoice``).

The module emits the same flat per-speaker manifest format as
``commonvoice.py`` so downstream code (calibration, scenario_5 runner,
``scripts/scenario_5_from_manifest.py``) consumes both interchangeably.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import default_data_dir
from .commonvoice import CommonVoiceClip, write_manifest

SAMPLE_RATE = 16_000
DEFAULT_CLIPS_PER_SPEAKER = 5
DEFAULT_SPLIT = "test"

# FLEURS HF config names follow ``<iso>_<region>``. We expose the short ISO
# code as the public "language" key and resolve to the full HF config
# internally — keeps CLI / manifest fields short and stable across versions.
LANGUAGE_TO_FLEURS_CONFIG: dict[str, str] = {
    "en": "en_us",
    "ja": "ja_jp",
    "de": "de_de",
    "fr": "fr_fr",
    "zh-CN": "cmn_hans_cn",
    "es": "es_419",
    "ko": "ko_kr",
    "ar": "ar_eg",
    "nl": "nl_nl",
    "it": "it_it",
    "pt": "pt_br",
    "pl": "pl_pl",
}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_TO_FLEURS_CONFIG)

# Synthetic speaker labels — one per gender bucket. The Scenario 5
# pipeline pairs ``speaker01`` (target enrollment) against ``speaker02``
# (other) so the labels here intentionally match the
# ``select_top_speakers`` output of ``commonvoice.py``.
SPEAKER_LABEL_BY_GENDER: dict[int, str] = {0: "speaker01", 1: "speaker02"}


def _resample_to_target(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample to :data:`SAMPLE_RATE` if needed, return float32 mono."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if src_sr == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    g = gcd(int(src_sr), int(SAMPLE_RATE))
    return resample_poly(audio, SAMPLE_RATE // g, src_sr // g).astype(np.float32)


def prepare(
    language: str,
    output_dir: Path,
    *,
    clips_per_speaker: int = DEFAULT_CLIPS_PER_SPEAKER,
    split: str = DEFAULT_SPLIT,
    max_stream: int = 2000,
) -> Path:
    """Stream FLEURS for ``language``, write a per-gender subset + manifest.

    Idempotent: returns immediately if ``output_dir/manifest.csv`` exists.

    The function loads up to ``max_stream`` samples from the requested
    split (default ``test``), buckets them by ``gender`` (0 → speaker01,
    1 → speaker02), and keeps the first ``clips_per_speaker`` clips per
    bucket. Audio is resampled to 16 kHz and written as wav alongside a
    ``manifest.csv`` that uses :class:`CommonVoiceClip` rows for full
    schema compatibility with the calibrate / scenario_5 tooling.
    """
    if language not in LANGUAGE_TO_FLEURS_CONFIG:
        raise ValueError(
            f"language {language!r} not in supported FLEURS set "
            f"{tuple(LANGUAGE_TO_FLEURS_CONFIG)}"
        )
    if clips_per_speaker <= 0:
        raise ValueError("clips_per_speaker must be > 0")

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        return output_dir

    config = LANGUAGE_TO_FLEURS_CONFIG[language]

    # Lazy import — `datasets` is heavy and only needed on the prep path.
    from datasets import load_dataset

    ds = load_dataset(
        "google/fleurs",
        config,
        split=split,
        streaming=True,
    )

    by_gender: dict[int, list[dict]] = defaultdict(list)
    target_buckets = set(SPEAKER_LABEL_BY_GENDER)
    n_seen = 0
    for sample in ds:
        if n_seen >= max_stream:
            break
        n_seen += 1
        gender = sample.get("gender")
        if gender not in target_buckets:
            continue
        if len(by_gender[gender]) >= clips_per_speaker:
            continue
        audio_blob = sample["audio"]
        by_gender[gender].append(
            {
                "audio": np.asarray(audio_blob["array"], dtype=np.float32),
                "sr": int(audio_blob["sampling_rate"]),
                "text": sample.get("transcription") or sample.get("raw_transcription") or "",
            }
        )
        if all(len(by_gender[b]) >= clips_per_speaker for b in target_buckets):
            break

    # FLEURS guarantees both genders are present per language; fail loudly
    # if that invariant breaks (e.g. on a custom split or a heavily filtered
    # mirror) so callers can fall back to a different source.
    missing = [b for b in target_buckets if not by_gender[b]]
    if missing:
        labels = sorted(SPEAKER_LABEL_BY_GENDER[b] for b in missing)
        raise RuntimeError(
            f"FLEURS({config}, {split}) yielded zero clips for {labels}; "
            f"saw {n_seen} samples"
        )

    # Materialise the wavs under output_dir/<speakerNN>/.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[CommonVoiceClip] = []
    for gender_value, clips in sorted(by_gender.items()):
        speaker_label = SPEAKER_LABEL_BY_GENDER[gender_value]
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

    write_manifest(manifest, manifest_path)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FLEURS subset preparation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare", help="stream FLEURS + materialise a per-gender subset"
    )
    p_prep.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="ISO language code; mapped to the matching FLEURS HF config",
    )
    p_prep.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="root data dir; defaults to $MELLONELLA_DATA_DIR/fleurs",
    )
    p_prep.add_argument(
        "--clips-per-speaker",
        type=int,
        default=DEFAULT_CLIPS_PER_SPEAKER,
        help=f"clips per gender bucket (default: {DEFAULT_CLIPS_PER_SPEAKER})",
    )
    p_prep.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "validation", "test"],
        help=f"FLEURS split to stream (default: {DEFAULT_SPLIT})",
    )

    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        root = args.data_dir if args.data_dir is not None else default_data_dir() / "fleurs"
        output_dir = root / args.language
        prepare(
            args.language,
            output_dir,
            clips_per_speaker=args.clips_per_speaker,
            split=args.split,
        )
        print(f"  output: {output_dir}")
        print(f"  manifest: {output_dir / 'manifest.csv'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
