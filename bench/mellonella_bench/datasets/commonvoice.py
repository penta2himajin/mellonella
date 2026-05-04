"""Mozilla Common Voice subset preparation.

License (CommonVoice):
    CC0-1.0 (public domain). Reuse is unrestricted but **redistribution
    of the corpus itself is forbidden by Mozilla's terms** — keep the
    archive download local. Speaker identity inference is also
    explicitly prohibited.

CommonVoice ships per-language tarballs under
``https://commonvoice.mozilla.org/api/v1/cv-corpus-...``; URLs require a
short-lived signed token, so this module does NOT bake a fixed URL into
the source. Pass ``--archive PATH`` after downloading the language tarball
manually (Mozilla page → "Download" → save the .tar.gz):

    python -m mellonella_bench.datasets.commonvoice prepare \
        --language ja --archive ~/Downloads/cv-corpus-19.0-2024-09-13-ja.tar.gz

The script then:

* extracts the archive under ``$MELLONELLA_DATA_DIR/commonvoice/<lang>``
* parses ``validated.tsv`` and groups clips by ``client_id``
* picks the top-K speakers (by clip count) and copies the first N clips
  per speaker into ``$MELLONELLA_DATA_DIR/commonvoice/<lang>/subset/``
* writes a ``manifest.csv`` (one row per clip) for downstream calibration

Bench / calibration scripts can then enumerate the manifest to assemble
multi-speaker, multi-language test pools without re-extracting the full
corpus each time.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .common import DatasetSpec, default_data_dir, ensure_clean_dir

SPEC = DatasetSpec(
    name="commonvoice",
    url="",  # CommonVoice URLs are signed; the user provides the archive locally.
    archive_sha256="",
    license="CC0-1.0",
    notes=(
        "Mozilla Common Voice corpus, per-language tarballs. "
        "Redistribution forbidden; keep archive downloads local."
    ),
)

DEFAULT_TOP_SPEAKERS = 10
DEFAULT_CLIPS_PER_SPEAKER = 20
SUPPORTED_LANGUAGES = (
    "en",
    "ja",
    "de",
    "fr",
    "zh-CN",
    "es",
    "ko",
    "ar",
)


@dataclass(frozen=True)
class CommonVoiceClip:
    """One row of the prepared subset manifest."""

    language: str
    speaker_id: str
    clip_path: Path  # relative to the subset root
    sentence: str


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract a CommonVoice .tar.gz into ``dest``. Idempotent on warm dest."""
    if dest.exists() and any(dest.iterdir()):
        return dest
    ensure_clean_dir(dest)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    return dest


def _find_validated_tsv(extracted_root: Path, language: str) -> Path:
    """Locate ``validated.tsv`` for the given language inside the extracted tree.

    CommonVoice tarballs nest under a top-level ``cv-corpus-<version>`` dir
    and then a per-language subdir, but exact names change per release —
    walk the tree to find any ``<language>/validated.tsv`` match.
    """
    for tsv in extracted_root.rglob("validated.tsv"):
        if tsv.parent.name == language:
            return tsv
    raise FileNotFoundError(
        f"validated.tsv for language {language!r} not found under {extracted_root}"
    )


def _read_validated(tsv_path: Path) -> list[dict[str, str]]:
    """Parse a CommonVoice ``validated.tsv`` into a list of row dicts."""
    with tsv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return [dict(row) for row in reader]


def select_top_speakers(
    rows: list[dict[str, str]],
    top_k: int,
    clips_per_speaker: int,
) -> dict[str, list[dict[str, str]]]:
    """Group rows by ``client_id`` and pick the top-K most-clipped speakers.

    For each picked speaker, return the first ``clips_per_speaker`` rows
    in source order (deterministic when the input TSV order is stable).
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if clips_per_speaker <= 0:
        raise ValueError("clips_per_speaker must be > 0")
    counts = Counter(r["client_id"] for r in rows if r.get("client_id"))
    chosen_ids = [cid for cid, _ in counts.most_common(top_k)]
    selected: dict[str, list[dict[str, str]]] = {cid: [] for cid in chosen_ids}
    for r in rows:
        cid = r.get("client_id")
        if cid in selected and len(selected[cid]) < clips_per_speaker:
            selected[cid].append(r)
    return selected


def build_subset(
    extracted_root: Path,
    language: str,
    subset_dir: Path,
    *,
    top_k: int = DEFAULT_TOP_SPEAKERS,
    clips_per_speaker: int = DEFAULT_CLIPS_PER_SPEAKER,
) -> list[CommonVoiceClip]:
    """Materialise a flat per-speaker subset under ``subset_dir``."""
    tsv_path = _find_validated_tsv(extracted_root, language)
    rows = _read_validated(tsv_path)
    selected = select_top_speakers(rows, top_k, clips_per_speaker)

    clips_root = tsv_path.parent / "clips"
    if not clips_root.exists():
        raise FileNotFoundError(f"clips/ directory missing under {tsv_path.parent}")

    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[CommonVoiceClip] = []
    for speaker_counter, picked in enumerate(selected.values(), start=1):
        speaker_label = f"speaker{speaker_counter:02d}"
        speaker_dir = subset_dir / speaker_label
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for clip_idx, row in enumerate(picked):
            src_name = row.get("path") or ""
            if not src_name:
                continue
            src = clips_root / src_name
            if not src.exists():
                continue
            dst_name = f"{clip_idx:03d}_{src_name}"
            dst = speaker_dir / dst_name
            shutil.copy2(src, dst)
            manifest.append(
                CommonVoiceClip(
                    language=language,
                    speaker_id=speaker_label,
                    clip_path=Path(speaker_label) / dst_name,
                    sentence=row.get("sentence", ""),
                )
            )

    write_manifest(manifest, subset_dir / "manifest.csv")
    return manifest


def write_manifest(manifest: list[CommonVoiceClip], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["language", "speaker_id", "clip_path", "sentence"])
        for c in manifest:
            writer.writerow([c.language, c.speaker_id, str(c.clip_path), c.sentence])


def read_manifest(path: Path) -> list[CommonVoiceClip]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [
            CommonVoiceClip(
                language=row["language"],
                speaker_id=row["speaker_id"],
                clip_path=Path(row["clip_path"]),
                sentence=row.get("sentence", ""),
            )
            for row in reader
        ]


def load_speakers_from_manifest(
    manifest_path: Path,
    sample_rate: int,
    *,
    min_seconds: float = 5.0,
) -> dict[str, object]:
    """Group manifest clips by ``speaker_id`` and return one concatenated buffer
    per speaker, resampled to ``sample_rate``.

    Speakers whose total audio is shorter than ``min_seconds`` are dropped —
    calibration needs at least an enrollment-half plus a test-half worth of
    material per speaker.

    Audio loading uses ``soundfile`` and ``scipy.signal.resample_poly``;
    both are pulled in lazily so the test suite can import this module
    without librosa / soundfile (only the heavy-dep code path imports them).
    """
    from math import gcd

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    clips = read_manifest(manifest_path)
    base = manifest_path.parent
    by_speaker: dict[str, list] = {}
    for clip in clips:
        clip_path = base / clip.clip_path
        if not clip_path.exists():
            continue
        audio, sr = sf.read(str(clip_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if sr != sample_rate:
            g = gcd(int(sr), int(sample_rate))
            audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
        by_speaker.setdefault(clip.speaker_id, []).append(audio)

    out: dict[str, object] = {}
    min_samples = int(min_seconds * sample_rate)
    for speaker, parts in by_speaker.items():
        merged = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
        if merged.size >= min_samples:
            out[speaker] = merged
    return out


def prepare(
    archive: Path,
    language: str,
    *,
    data_dir: Path | None = None,
    top_k: int = DEFAULT_TOP_SPEAKERS,
    clips_per_speaker: int = DEFAULT_CLIPS_PER_SPEAKER,
) -> Path:
    """End-to-end: extract -> select speakers -> materialise subset.

    Returns the path to the populated subset root (which contains a
    ``manifest.csv`` plus one directory per speaker).
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"language {language!r} not in supported set {SUPPORTED_LANGUAGES}")
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")
    root = data_dir if data_dir is not None else default_data_dir() / "commonvoice"
    lang_root = root / language
    extracted_root = lang_root / "extracted"
    subset_dir = lang_root / "subset"
    extract_archive(archive, extracted_root)
    build_subset(
        extracted_root,
        language,
        subset_dir,
        top_k=top_k,
        clips_per_speaker=clips_per_speaker,
    )
    return subset_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CommonVoice subset preparation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="extract + build the per-speaker subset")
    p_prep.add_argument("--archive", type=Path, required=True, help="path to the .tar.gz")
    p_prep.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="ISO language code matching the archive",
    )
    p_prep.add_argument("--data-dir", type=Path, default=None)
    p_prep.add_argument("--top-k", type=int, default=DEFAULT_TOP_SPEAKERS)
    p_prep.add_argument("--clips-per-speaker", type=int, default=DEFAULT_CLIPS_PER_SPEAKER)

    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        subset_dir = prepare(
            args.archive,
            args.language,
            data_dir=args.data_dir,
            top_k=args.top_k,
            clips_per_speaker=args.clips_per_speaker,
        )
        print(f"  subset: {subset_dir}")
        print(f"  manifest: {subset_dir / 'manifest.csv'}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
