"""Tests for the Emilia-YODAS subset preparation module.

The HF ``datasets`` library is mocked end-to-end. Each fake sample uses
the ``{json: {...}}`` WebDataset shape that Emilia-YODAS produces with
the upstream ``data_files=Emilia-YODAS/<LANG>/*.tar`` glob.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Iterator

import numpy as np
import pytest


def _install_fake_datasets(monkeypatch, samples: Iterable[dict]) -> None:
    materialised = list(samples)

    def fake_load_dataset(*_args, **_kwargs) -> Iterator[dict]:
        return iter(materialised)

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)


def _make_emilia_sample(
    *,
    speaker: str,
    language: str,
    freq: float,
    duration: float = 1.0,
    sr: int = 16_000,
    text: str = "x",
) -> dict:
    """Mirror the WebDataset ``{mp3, json}`` shape with auto-decoded mp3 array."""
    t = np.arange(int(duration * sr)) / sr
    arr = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return {
        "mp3": {"array": arr, "sampling_rate": sr},
        "json": {"speaker": speaker, "language": language, "text": text},
    }


def test_prepare_requires_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _install_fake_datasets(monkeypatch, [])
    from mellonella_bench.datasets.emilia import prepare

    with pytest.raises(RuntimeError, match="gated"):
        prepare("ja", tmp_path / "ja")


def test_prepare_writes_manifest_with_top_speakers(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    samples = [
        _make_emilia_sample(speaker="JA_001", language="ja", freq=120.0, text="ja-1"),
        _make_emilia_sample(speaker="JA_001", language="ja", freq=120.0, text="ja-2"),
        _make_emilia_sample(speaker="JA_002", language="ja", freq=240.0, text="ja-3"),
        _make_emilia_sample(speaker="JA_002", language="ja", freq=240.0, text="ja-4"),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.emilia import prepare

    out = tmp_path / "ja"
    prepare("ja", out, top_speakers=2, clips_per_speaker=2)
    assert (out / "manifest.csv").exists()
    assert (out / "speaker01" / "000.wav").exists()
    assert (out / "speaker02" / "000.wav").exists()
    # Sidecar mapping back to upstream Emilia speaker id.
    assert (out / "speaker01" / "_emilia_speaker.txt").exists()


def test_prepare_idempotent_on_warm_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    samples = [
        _make_emilia_sample(speaker="EN_a", language="en", freq=120.0),
        _make_emilia_sample(speaker="EN_b", language="en", freq=240.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.emilia import prepare

    out = tmp_path / "en"
    prepare("en", out, top_speakers=2, clips_per_speaker=1)
    sentinel = out / "warm.txt"
    sentinel.write_text("preserved")

    prepare("en", out, top_speakers=2, clips_per_speaker=1)
    assert sentinel.exists()
    assert sentinel.read_text() == "preserved"


def test_prepare_resamples_non_target_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    src_sr = 24_000
    samples = [
        _make_emilia_sample(speaker="ZH_001", language="zh", freq=120.0, duration=2.0, sr=src_sr),
        _make_emilia_sample(speaker="ZH_002", language="zh", freq=240.0, duration=2.0, sr=src_sr),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.emilia import SAMPLE_RATE, prepare

    out = tmp_path / "zh-CN"
    prepare("zh-CN", out, top_speakers=2, clips_per_speaker=1)

    import soundfile as sf

    for spk in ("speaker01", "speaker02"):
        audio, sr = sf.read(str(out / spk / "000.wav"))
        assert sr == SAMPLE_RATE
        assert abs(audio.size - 2 * SAMPLE_RATE) <= 5


def test_prepare_rejects_unknown_language(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    _install_fake_datasets(monkeypatch, [])
    from mellonella_bench.datasets.emilia import prepare

    with pytest.raises(ValueError):
        prepare("klingon", tmp_path / "out")


def test_prepare_raises_when_not_enough_speakers(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    samples = [
        _make_emilia_sample(speaker="JA_solo", language="ja", freq=120.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.emilia import prepare

    with pytest.raises(RuntimeError):
        prepare("ja", tmp_path / "out", top_speakers=2, clips_per_speaker=1)


def test_supported_languages_cover_asian_targets():
    from mellonella_bench.datasets.emilia import SUPPORTED_LANGUAGES

    assert {"ja", "ko", "zh-CN"} <= set(SUPPORTED_LANGUAGES)


def test_extract_audio_handles_raw_bytes(tmp_path, monkeypatch):
    """A raw mp3 bytes blob must round-trip through soundfile."""
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    import io

    import soundfile as sf

    sr = 16_000
    arr = np.full(sr, 0.1, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, arr, sr, format="WAV")  # WAV bytes are also valid soundfile input
    buffer.seek(0)
    sample_bytes = buffer.read()

    samples = [
        {"mp3": sample_bytes, "json": {"speaker": "X", "language": "en", "text": "raw-bytes"}},
        {
            "mp3": {"array": arr, "sampling_rate": sr},
            "json": {"speaker": "Y", "language": "en", "text": "decoded"},
        },
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.commonvoice import read_manifest
    from mellonella_bench.datasets.emilia import prepare

    out = tmp_path / "en"
    prepare("en", out, top_speakers=2, clips_per_speaker=1)
    rows = read_manifest(out / "manifest.csv")
    assert {r.sentence for r in rows} == {"raw-bytes", "decoded"}


def test_prepare_accepts_explicit_token_param(tmp_path, monkeypatch):
    """``hf_token`` arg overrides the env var (and unblocks the gate)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    samples = [
        _make_emilia_sample(speaker="A", language="en", freq=120.0),
        _make_emilia_sample(speaker="B", language="en", freq=240.0),
    ]
    _install_fake_datasets(monkeypatch, samples)
    from mellonella_bench.datasets.emilia import prepare

    out = tmp_path / "en"
    prepare("en", out, top_speakers=2, clips_per_speaker=1, hf_token="explicit-token")
    assert (out / "manifest.csv").exists()
