"""Tests for the CommonVoice subset preparation helpers.

These tests exercise the pure-Python parts of
``mellonella_bench.datasets.commonvoice`` (TSV parsing, speaker
selection, manifest IO, archive extraction) against tiny synthesised
fixtures. We intentionally do NOT hit the real CommonVoice CDN — the
URLs are signed and the corpus is multi-GB.
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import pytest

from mellonella_bench.datasets.commonvoice import (
    DEFAULT_CLIPS_PER_SPEAKER,
    SUPPORTED_LANGUAGES,
    CommonVoiceClip,
    build_subset,
    extract_archive,
    prepare,
    read_manifest,
    select_top_speakers,
    write_manifest,
)


def _make_validated_rows(
    speaker_clip_counts: dict[str, int],
) -> list[dict[str, str]]:
    """Synthesise CommonVoice ``validated.tsv`` rows for the given counts."""
    rows: list[dict[str, str]] = []
    for client_id, n in speaker_clip_counts.items():
        for i in range(n):
            rows.append(
                {
                    "client_id": client_id,
                    "path": f"{client_id}_{i:03d}.mp3",
                    "sentence": f"clip {i} for {client_id}",
                }
            )
    return rows


def test_select_top_speakers_picks_most_clipped():
    rows = _make_validated_rows({"A": 5, "B": 30, "C": 15, "D": 1, "E": 8})
    selected = select_top_speakers(rows, top_k=3, clips_per_speaker=10)
    assert set(selected) == {"B", "C", "E"}
    assert all(len(v) <= 10 for v in selected.values())
    # Top speaker capped to clips_per_speaker
    assert len(selected["B"]) == 10


def test_select_top_speakers_respects_clips_per_speaker():
    rows = _make_validated_rows({"A": 50})
    selected = select_top_speakers(rows, top_k=1, clips_per_speaker=4)
    assert len(selected["A"]) == 4


def test_select_top_speakers_validates_args():
    with pytest.raises(ValueError):
        select_top_speakers([], top_k=0, clips_per_speaker=1)
    with pytest.raises(ValueError):
        select_top_speakers([], top_k=1, clips_per_speaker=0)


def test_select_top_speakers_skips_blank_client_ids():
    rows = [
        {"client_id": "", "path": "a.mp3", "sentence": ""},
        {"client_id": "X", "path": "b.mp3", "sentence": ""},
    ]
    selected = select_top_speakers(rows, top_k=5, clips_per_speaker=10)
    assert "" not in selected
    assert "X" in selected


def test_manifest_round_trip(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    clips = [
        CommonVoiceClip(
            language="en",
            speaker_id="speaker01",
            clip_path=Path("speaker01") / "000_a.mp3",
            sentence="hello world",
        ),
        CommonVoiceClip(
            language="en",
            speaker_id="speaker01",
            clip_path=Path("speaker01") / "001_b.mp3",
            sentence="another clip",
        ),
    ]
    write_manifest(clips, manifest_path)
    loaded = read_manifest(manifest_path)
    assert loaded == clips


def test_extract_archive_skips_when_warm(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "hello.txt").write_text("hi")
    archive = tmp_path / "src.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src_dir / "hello.txt", arcname="hello.txt")

    dest = tmp_path / "dest"
    extract_archive(archive, dest)
    assert (dest / "hello.txt").exists()

    # Touch a sentinel and re-call: the warm path must not nuke the contents.
    sentinel = dest / "warm.txt"
    sentinel.write_text("preserved")
    extract_archive(archive, dest)
    assert sentinel.exists()
    assert sentinel.read_text() == "preserved"


def test_build_subset_against_synthetic_extraction(tmp_path):
    """Construct a fake CommonVoice extraction tree and run the subset builder."""
    # Layout: extracted_root / cv-corpus-test / en / { validated.tsv, clips/ }
    extracted_root = tmp_path / "extracted"
    lang_root = extracted_root / "cv-corpus-test" / "en"
    clips_root = lang_root / "clips"
    clips_root.mkdir(parents=True)

    speaker_counts = {"alpha": 6, "beta": 4, "gamma": 2}
    rows = _make_validated_rows(speaker_counts)
    # Write the audio files (just empty placeholders — the builder copies bytes).
    for r in rows:
        (clips_root / r["path"]).write_bytes(b"\x00" * 32)

    tsv_path = lang_root / "validated.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["client_id", "path", "sentence"], delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    subset_dir = tmp_path / "subset"
    manifest = build_subset(extracted_root, "en", subset_dir, top_k=2, clips_per_speaker=3)
    assert len(manifest) == 6  # 2 speakers × 3 clips
    assert {clip.speaker_id for clip in manifest} == {"speaker01", "speaker02"}
    # Every manifest path must exist on disk under subset_dir
    for clip in manifest:
        assert (subset_dir / clip.clip_path).exists()
    # Manifest CSV is written
    assert (subset_dir / "manifest.csv").exists()


def test_build_subset_rejects_missing_validated_tsv(tmp_path):
    extracted_root = tmp_path / "extracted"
    extracted_root.mkdir()
    with pytest.raises(FileNotFoundError):
        build_subset(extracted_root, "en", tmp_path / "subset")


def test_prepare_rejects_unknown_language(tmp_path):
    archive = tmp_path / "fake.tar.gz"
    archive.write_bytes(b"")
    with pytest.raises(ValueError):
        prepare(archive, "klingon", data_dir=tmp_path)  # type: ignore[arg-type]


def test_prepare_rejects_missing_archive(tmp_path):
    archive = tmp_path / "missing.tar.gz"
    with pytest.raises(FileNotFoundError):
        prepare(archive, "en", data_dir=tmp_path)


def test_supported_languages_cover_docs_targets():
    """docs/benchmarks.md scenario_5 lists 8 candidate languages; verify we
    expose at least those codes via SUPPORTED_LANGUAGES."""
    docs_targets = {"en", "ja", "de", "fr", "zh-CN", "es", "ko", "ar"}
    assert docs_targets <= set(SUPPORTED_LANGUAGES)


def test_default_clips_per_speaker_is_reasonable():
    # docs/benchmarks.md PoC subset suggests ~50 utt per language; default 20
    # leaves room for top-K speakers without bloating the bundle.
    assert 5 <= DEFAULT_CLIPS_PER_SPEAKER <= 100
