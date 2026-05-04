"""Tests for ``scripts/scenario_5_from_manifest.py``.

The script lives outside the ``bench`` package so it can be invoked
standalone (it tweaks ``sys.path`` to pick up both ``poc`` and ``bench``);
the tests load it via ``importlib`` so they don't depend on it being on
``$PYTHONPATH``.

The fixtures synthesise a CommonVoice-shaped manifest tree per language —
no real CommonVoice tarball is required, so this exercises the full
manifest → load_speakers → materialise → scenario_5.run → failures.json
chain in CI without external data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.datasets.commonvoice import (
    CommonVoiceClip,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "scenario_5_from_manifest.py"


def _import_script():
    """Load the helper script as a module without polluting global state."""
    spec = importlib.util.spec_from_file_location("scenario_5_from_manifest", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("scenario_5_from_manifest", module)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_manifest(
    manifest_dir: Path,
    *,
    language: str,
    n_speakers: int = 2,
    seconds_per_clip: float = 4.0,
    sr: int = 16_000,
) -> Path:
    """Synthesise a CommonVoice-shape subset directory.

    Each speaker gets a pair of WAV clips of distinct base frequencies so
    ``load_speakers_from_manifest`` returns >= 2 speakers each long enough
    to clear the ``min_seconds`` filter.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(seconds_per_clip * sr)
    rng = np.random.default_rng(hash(language) % 2**16)
    clips: list[CommonVoiceClip] = []
    for spk_idx in range(n_speakers):
        speaker_id = f"{language}_spk{spk_idx:02d}"
        spk_dir = manifest_dir / speaker_id
        spk_dir.mkdir(parents=True, exist_ok=True)
        for clip_idx in range(2):  # two clips per speaker → > min_seconds=5s default
            t = np.arange(n_samples) / sr
            base = 220.0 + 110.0 * spk_idx
            audio = (
                0.4 * np.sin(2 * np.pi * base * t) + 0.05 * rng.standard_normal(n_samples)
            ).astype(np.float32)
            rel = Path(speaker_id) / f"{clip_idx:03d}.wav"
            sf.write(str(manifest_dir / rel), audio, sr)
            clips.append(
                CommonVoiceClip(
                    language=language,
                    speaker_id=speaker_id,
                    clip_path=rel,
                    sentence=f"clip {clip_idx} for {speaker_id}",
                )
            )
    manifest_path = manifest_dir / "manifest.csv"
    write_manifest(clips, manifest_path)
    return manifest_path


def test_manifest_spec_parses_lang_path():
    mod = _import_script()
    spec = mod.ManifestSpec.parse("ja=/tmp/foo/manifest.csv")
    assert spec.language == "ja"
    assert spec.manifest == Path("/tmp/foo/manifest.csv")


def test_manifest_spec_rejects_missing_separator():
    mod = _import_script()
    with pytest.raises(argparse.ArgumentTypeError):
        mod.ManifestSpec.parse("ja:/tmp/foo")


def test_manifest_spec_rejects_blank_parts():
    mod = _import_script()
    with pytest.raises(argparse.ArgumentTypeError):
        mod.ManifestSpec.parse("=/tmp/foo")
    with pytest.raises(argparse.ArgumentTypeError):
        mod.ManifestSpec.parse("ja=")


def test_build_items_materialises_target_other_per_language(tmp_path):
    mod = _import_script()
    ja = _write_synthetic_manifest(tmp_path / "ja_subset", language="ja")
    en = _write_synthetic_manifest(tmp_path / "en_subset", language="en")
    specs = [
        mod.ManifestSpec(language="ja", manifest=ja),
        mod.ManifestSpec(language="en", manifest=en),
    ]
    items = mod.build_items(specs, work_dir=tmp_path / "work", top_speakers=2)
    assert len(items) == 2  # 1 (target, other) pair per language
    assert {it.language for it in items} == {"ja", "en"}
    for it in items:
        assert it.target_path.exists()
        assert it.other_path.exists()
        assert it.noise_path.exists()
        assert it.target_speaker.startswith(it.language)
        assert it.other_speaker.startswith(it.language)
        assert it.target_speaker != it.other_speaker
        assert it.enrollment_path == it.target_path


def test_build_items_rejects_top_speakers_lt_2(tmp_path):
    mod = _import_script()
    ja = _write_synthetic_manifest(tmp_path / "ja_subset", language="ja")
    with pytest.raises(ValueError):
        mod.build_items(
            [mod.ManifestSpec(language="ja", manifest=ja)],
            work_dir=tmp_path / "work",
            top_speakers=1,
        )


def test_build_items_rejects_single_speaker_manifest(tmp_path):
    mod = _import_script()
    only_one = _write_synthetic_manifest(tmp_path / "ja_subset", language="ja", n_speakers=1)
    with pytest.raises(ValueError):
        mod.build_items(
            [mod.ManifestSpec(language="ja", manifest=only_one)],
            work_dir=tmp_path / "work",
            top_speakers=2,
        )


def test_collect_failures_flags_below_tpr_min():
    """A row in target mode with TPR below the minimum is reported."""
    mod = _import_script()
    from mellonella_bench.scenarios.base import SnrSweepEntry

    entries = [
        SnrSweepEntry(
            sample_id="x",
            language="ja",
            snr_db=0.0,
            gate_tpr=0.10,
            notes="mode=target",
        ),
        SnrSweepEntry(
            sample_id="x",
            language="ja",
            snr_db=10.0,
            gate_tpr=0.95,
            notes="mode=target",
        ),
    ]
    failures = mod.collect_failures(entries, tpr_min=0.5, fpr_max=0.5)
    assert len(failures) == 1
    assert failures[0]["violation"] == "below_tpr_min"
    assert failures[0]["snr_db"] == 0.0


def test_collect_failures_flags_above_fpr_max():
    mod = _import_script()
    from mellonella_bench.scenarios.base import SnrSweepEntry

    entries = [
        SnrSweepEntry(
            sample_id="x",
            language="en",
            snr_db=5.0,
            gate_fpr=0.95,
            notes="mode=other",
        ),
    ]
    failures = mod.collect_failures(entries, tpr_min=0.5, fpr_max=0.5)
    assert len(failures) == 1
    assert failures[0]["violation"] == "above_fpr_max"


def test_main_end_to_end_with_stub_provider(tmp_path):
    """The default stub passes everything → TPR=1, FPR=1.

    With ``--fpr-max=0.5`` every other-mode row is a failure, so the script
    should write summary/failures and exit 1.
    """
    mod = _import_script()
    ja = _write_synthetic_manifest(tmp_path / "ja_subset", language="ja")
    en = _write_synthetic_manifest(tmp_path / "en_subset", language="en")
    output = tmp_path / "out"
    rc = mod.main(
        [
            "--manifest",
            f"ja={ja}",
            "--manifest",
            f"en={en}",
            "--output",
            str(output),
            "--snrs-db",
            "10.0",
            "--tpr-min",
            "0.5",
            "--fpr-max",
            "0.5",
        ]
    )
    assert rc == 1  # FPR=1 with stub → every other-mode row violates fpr_max
    assert (output / "scenario_5.csv").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_items"] == 2
    assert set(summary["languages"]) == {"ja", "en"}
    failures = json.loads((output / "failures.json").read_text())
    assert failures["n_failures"] >= 2
    assert all(f["violation"] == "above_fpr_max" for f in failures["failures"])


def test_main_passes_when_thresholds_satisfied(tmp_path):
    """With permissive thresholds (fpr_max=1.1) the stub passes → exit 0."""
    mod = _import_script()
    ja = _write_synthetic_manifest(tmp_path / "ja_subset", language="ja")
    output = tmp_path / "out"
    rc = mod.main(
        [
            "--manifest",
            f"ja={ja}",
            "--output",
            str(output),
            "--snrs-db",
            "10.0",
            "--tpr-min",
            "0.0",
            "--fpr-max",
            "1.1",
        ]
    )
    assert rc == 0
    failures = json.loads((output / "failures.json").read_text())
    assert failures["n_failures"] == 0
