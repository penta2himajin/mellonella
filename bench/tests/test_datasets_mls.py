"""Tests for the MLS subset preparation module.

The HF ``datasets`` library is mocked end-to-end so the test suite never
hits the network. Each fake sample mirrors the real MLS schema
(``audio: {array, sampling_rate}``, ``text``, ``speaker_id``).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Iterator

import numpy as np
import pytest


def _install_fake_datasets(monkeypatch, samples: Iterable[dict]) -> None:
    """Inject a fake ``datasets`` module before mls.py imports it."""
    materialised = list(samples)

    def fake_load_dataset(*_args, **_kwargs) -> Iterator[dict]:
        return iter(materialised)

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)


def _make_mls_sample(
    *, speaker_id: int, freq: float, duration: float = 1.0, sr: int = 16_000, text: str = "x"
) -> dict:
    t = np.arange(int(duration * sr)) / sr
    arr = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return {
        "audio": {"array": arr, "sampling_rate": sr},
        "text": text,
        "speaker_id": speaker_id,
    }


def test_prepare_writes_manifest_with_top_speakers(tmp_path, monkeypatch):
    samples = [
        _make_mls_sample(speaker_id=10, freq=120.0, text="s10-1"),
        _make_mls_sample(speaker_id=10, freq=120.0, text="s10-2"),
        _make_mls_sample(speaker_id=20, freq=240.0, text="s20-1"),
        _make_mls_sample(speaker_id=20, freq=240.0, text="s20-2"),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.mls import prepare

    out = tmp_path / "de"
    prepare("de", out, top_speakers=2, clips_per_speaker=2)
    assert (out / "manifest.csv").exists()
    assert (out / "speaker01" / "000.wav").exists()
    assert (out / "speaker01" / "001.wav").exists()
    assert (out / "speaker02" / "000.wav").exists()
    # Sidecar mapping back to the upstream speaker id is stashed for debugging.
    assert (out / "speaker01" / "_mls_speaker_id.txt").exists()


def test_prepare_idempotent_on_warm_dir(tmp_path, monkeypatch):
    samples = [
        _make_mls_sample(speaker_id=1, freq=120.0),
        _make_mls_sample(speaker_id=2, freq=240.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.mls import prepare

    out = tmp_path / "de"
    prepare("de", out, top_speakers=2, clips_per_speaker=1)
    sentinel = out / "warm.txt"
    sentinel.write_text("preserved")

    prepare("de", out, top_speakers=2, clips_per_speaker=1)
    assert sentinel.exists()
    assert sentinel.read_text() == "preserved"


def test_prepare_resamples_non_target_rate(tmp_path, monkeypatch):
    src_sr = 8_000
    samples = [
        _make_mls_sample(speaker_id=1, freq=120.0, duration=2.0, sr=src_sr),
        _make_mls_sample(speaker_id=2, freq=240.0, duration=2.0, sr=src_sr),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.mls import SAMPLE_RATE, prepare

    out = tmp_path / "fr"
    prepare("fr", out, top_speakers=2, clips_per_speaker=1)

    import soundfile as sf

    for spk in ("speaker01", "speaker02"):
        audio, sr = sf.read(str(out / spk / "000.wav"))
        assert sr == SAMPLE_RATE
        assert abs(audio.size - 2 * SAMPLE_RATE) <= 3


def test_prepare_rejects_unknown_language(tmp_path, monkeypatch):
    _install_fake_datasets(monkeypatch, [])
    from mellonella_bench.datasets.mls import prepare

    with pytest.raises(ValueError):
        prepare("klingon", tmp_path / "out")


def test_prepare_rejects_zero_top_speakers(tmp_path, monkeypatch):
    _install_fake_datasets(monkeypatch, [])
    from mellonella_bench.datasets.mls import prepare

    with pytest.raises(ValueError):
        prepare("de", tmp_path / "out", top_speakers=0)


def test_prepare_raises_when_not_enough_speakers(tmp_path, monkeypatch):
    samples = [
        # Only one distinct speaker available.
        _make_mls_sample(speaker_id=1, freq=120.0),
        _make_mls_sample(speaker_id=1, freq=130.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.mls import prepare

    with pytest.raises(RuntimeError):
        prepare("de", tmp_path / "out", top_speakers=2, clips_per_speaker=1)


def test_prepare_caps_at_clips_per_speaker(tmp_path, monkeypatch):
    samples = [_make_mls_sample(speaker_id=1, freq=120.0) for _ in range(10)]
    samples.extend(_make_mls_sample(speaker_id=2, freq=240.0) for _ in range(10))
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.mls import prepare

    out = tmp_path / "de"
    prepare("de", out, top_speakers=2, clips_per_speaker=3)
    speaker1_wavs = sorted(p for p in (out / "speaker01").iterdir() if p.suffix == ".wav")
    assert len(speaker1_wavs) == 3
    speaker2_wavs = sorted(p for p in (out / "speaker02").iterdir() if p.suffix == ".wav")
    assert len(speaker2_wavs) == 3


def test_manifest_uses_iso_language_code(tmp_path, monkeypatch):
    samples = [
        _make_mls_sample(speaker_id=10, freq=120.0, text="hello"),
        _make_mls_sample(speaker_id=20, freq=240.0, text="world"),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.commonvoice import read_manifest
    from mellonella_bench.datasets.mls import prepare

    out = tmp_path / "de"
    prepare("de", out, top_speakers=2, clips_per_speaker=1)
    rows = read_manifest(out / "manifest.csv")
    assert {r.language for r in rows} == {"de"}
    assert {r.speaker_id for r in rows} == {"speaker01", "speaker02"}
    assert {r.sentence for r in rows} == {"hello", "world"}


def test_supported_languages_match_currently_published_configs():
    """English was dropped from the MLS HF repo at the parquet migration; the
    other seven non-English configs remain."""
    from mellonella_bench.datasets.mls import SUPPORTED_LANGUAGES

    assert set(SUPPORTED_LANGUAGES) == {"de", "fr", "es", "it", "nl", "pl", "pt"}


def test_extract_text_falls_back_across_field_names(tmp_path, monkeypatch):
    """``text``, ``transcript``, ``transcription`` are all handled."""
    samples = [
        {
            "audio": {"array": np.full(16_000, 0.1, dtype=np.float32), "sampling_rate": 16_000},
            "transcript": "fallback-1",
            "speaker_id": 1,
        },
        {
            "audio": {"array": np.full(16_000, 0.1, dtype=np.float32), "sampling_rate": 16_000},
            "transcription": "fallback-2",
            "speaker_id": 2,
        },
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.commonvoice import read_manifest
    from mellonella_bench.datasets.mls import prepare

    out = tmp_path / "de"
    prepare("de", out, top_speakers=2, clips_per_speaker=1)
    rows = read_manifest(out / "manifest.csv")
    assert {r.sentence for r in rows} == {"fallback-1", "fallback-2"}
