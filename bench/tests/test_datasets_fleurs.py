"""Tests for the FLEURS subset preparation module.

The HF ``datasets`` library is mocked end-to-end so the test suite never
hits the network. Each fake sample mirrors the real FLEURS schema
(``audio: {array, sampling_rate}``, ``transcription``, ``gender``) and
exercises the gender-proxy bucketing behaviour.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest


def _install_fake_datasets(monkeypatch, samples: Iterable[dict]):
    """Inject a fake ``datasets`` module before fleurs.py imports it."""
    materialised = list(samples)

    def fake_load_dataset(*_args, **_kwargs) -> Iterator[dict]:
        return iter(materialised)

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)


def _make_fleurs_sample(
    *, gender: int, freq: float, duration: float = 1.0, sr: int = 16_000, text: str = "x"
) -> dict:
    t = np.arange(int(duration * sr)) / sr
    arr = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return {
        "audio": {"array": arr, "sampling_rate": sr},
        "transcription": text,
        "gender": gender,
    }


def test_prepare_writes_manifest_with_two_speakers(tmp_path, monkeypatch):
    samples = [
        _make_fleurs_sample(gender=0, freq=120.0, text="m1"),
        _make_fleurs_sample(gender=1, freq=240.0, text="f1"),
        _make_fleurs_sample(gender=0, freq=130.0, text="m2"),
        _make_fleurs_sample(gender=1, freq=260.0, text="f2"),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.fleurs import prepare

    out = tmp_path / "ja_out"
    prepare("ja", out, clips_per_speaker=2)
    manifest = out / "manifest.csv"
    assert manifest.exists()

    # Per-speaker subdirs materialised as wav.
    assert (out / "speaker01" / "000.wav").exists()
    assert (out / "speaker02" / "000.wav").exists()
    # Both buckets reach the cap.
    assert (out / "speaker01" / "001.wav").exists()
    assert (out / "speaker02" / "001.wav").exists()


def test_prepare_idempotent_on_warm_dir(tmp_path, monkeypatch):
    samples = [
        _make_fleurs_sample(gender=0, freq=120.0),
        _make_fleurs_sample(gender=1, freq=240.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.fleurs import prepare

    out = tmp_path / "en_out"
    prepare("en", out, clips_per_speaker=1)
    sentinel = out / "warm.txt"
    sentinel.write_text("preserved")

    # Re-call should short-circuit and not nuke the directory.
    prepare("en", out, clips_per_speaker=1)
    assert sentinel.exists()
    assert sentinel.read_text() == "preserved"


def test_prepare_resamples_non_target_rate(tmp_path, monkeypatch):
    src_sr = 8_000
    samples = [
        _make_fleurs_sample(gender=0, freq=120.0, duration=2.0, sr=src_sr),
        _make_fleurs_sample(gender=1, freq=240.0, duration=2.0, sr=src_sr),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.fleurs import prepare, SAMPLE_RATE

    out = tmp_path / "fr_out"
    prepare("fr", out, clips_per_speaker=1)

    # Resampled wav length should match the target rate (allowing ±1 sample).
    import soundfile as sf

    for spk in ("speaker01", "speaker02"):
        audio, sr = sf.read(str(out / spk / "000.wav"))
        assert sr == SAMPLE_RATE
        assert abs(audio.size - 2 * SAMPLE_RATE) <= 3


def test_prepare_rejects_unknown_language(tmp_path, monkeypatch):
    _install_fake_datasets(monkeypatch, [])
    from mellonella_bench.datasets.fleurs import prepare

    with pytest.raises(ValueError):
        prepare("klingon", tmp_path / "out")


def test_prepare_rejects_zero_clips_per_speaker(tmp_path, monkeypatch):
    _install_fake_datasets(monkeypatch, [])
    from mellonella_bench.datasets.fleurs import prepare

    with pytest.raises(ValueError):
        prepare("en", tmp_path / "out", clips_per_speaker=0)


def test_prepare_raises_when_a_gender_bucket_is_empty(tmp_path, monkeypatch):
    samples = [
        # All male — no female clips → raise so callers can pivot dataset.
        _make_fleurs_sample(gender=0, freq=120.0),
        _make_fleurs_sample(gender=0, freq=130.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.fleurs import prepare

    with pytest.raises(RuntimeError):
        prepare("en", tmp_path / "out", clips_per_speaker=1)


def test_prepare_caps_at_clips_per_speaker(tmp_path, monkeypatch):
    samples = [_make_fleurs_sample(gender=0, freq=120.0) for _ in range(10)]
    samples.extend(_make_fleurs_sample(gender=1, freq=240.0) for _ in range(10))
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.fleurs import prepare

    out = tmp_path / "en_out"
    prepare("en", out, clips_per_speaker=3)
    # Exactly three clips per bucket — no overflow.
    assert sorted((out / "speaker01").iterdir()) == sorted(
        [out / "speaker01" / f"{i:03d}.wav" for i in range(3)]
    )
    assert sorted((out / "speaker02").iterdir()) == sorted(
        [out / "speaker02" / f"{i:03d}.wav" for i in range(3)]
    )


def test_manifest_rows_use_iso_language_code(tmp_path, monkeypatch):
    samples = [
        _make_fleurs_sample(gender=0, freq=120.0, text="ja-male"),
        _make_fleurs_sample(gender=1, freq=240.0, text="ja-female"),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.commonvoice import read_manifest
    from mellonella_bench.datasets.fleurs import prepare

    out = tmp_path / "ja_out"
    prepare("ja", out, clips_per_speaker=1)
    rows = read_manifest(out / "manifest.csv")
    assert {r.language for r in rows} == {"ja"}
    assert {r.speaker_id for r in rows} == {"speaker01", "speaker02"}
    # Sentence is preserved from the FLEURS transcription field.
    assert any(r.sentence == "ja-male" for r in rows)
    assert any(r.sentence == "ja-female" for r in rows)


def test_supported_languages_cover_docs_targets():
    """docs/benchmarks.md scenario_5 lists 8 candidate languages."""
    from mellonella_bench.datasets.fleurs import SUPPORTED_LANGUAGES

    docs_targets = {"en", "ja", "de", "fr", "zh-CN", "es", "ko", "ar"}
    assert docs_targets <= set(SUPPORTED_LANGUAGES)
