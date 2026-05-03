"""Tests for the PipelineProvider abstraction and stub end-to-end flow."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.scenarios.base import (
    PipelineCallable,
    StubPipelineProvider,
)
from mellonella_bench.scenarios.scenario_1 import Scenario1Item, run


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_stub_provider_returns_callable():
    provider = StubPipelineProvider()
    callable_ = provider.for_item(object())
    mixture = _sine(220.0, 16_000, 1.0)
    result = callable_(mixture, 16_000)
    assert result.audio.shape == mixture.shape
    assert result.gate_per_frame.dtype == bool
    assert result.gate_per_frame.all()
    # Stub leaves auto-learn fields at defaults.
    assert result.auto_learn_admissions == 0
    assert result.auto_learn_resets == 0
    assert result.anchor_distance_final is None


def test_stub_provider_gate_is_frame_rate():
    provider = StubPipelineProvider()
    mixture = _sine(220.0, 16_000, 1.0)  # 1 s
    result = provider.for_item(None)(mixture, 16_000)
    # 1 s at 16 kHz / 512 samples-per-frame = 31 frames
    assert result.gate_per_frame.size == 31


def test_run_with_synthetic_items(tmp_path):
    sr = 16_000
    target = _sine(220.0, sr, 1.0)
    noise = 0.01 * np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    target_path = tmp_path / "target.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(target_path), target, sr)
    sf.write(str(noise_path), noise, sr)

    voiced_mask = np.ones(target.size // 512, dtype=bool)

    item = Scenario1Item(
        sample_id="utt_synthetic_001",
        target_path=target_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )

    csv_path = tmp_path / "scenario_1.csv"
    result = run(
        items=[item],
        provider=StubPipelineProvider(),
        sample_rate=sr,
        output_csv=csv_path,
        snrs_db=(0.0, 10.0),
        pesq_mode=None,  # skip PESQ — extras may not be installed
    )

    assert result.n_samples == 1
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "utt_synthetic_001"
    # Stub passes everything through, so TPR should be 1.0
    assert float(rows[0]["gate_tpr"]) == pytest.approx(1.0)


def test_run_rejects_sample_rate_mismatch(tmp_path):
    sr = 16_000
    target = _sine(220.0, sr, 0.5)
    noise = 0.01 * np.random.default_rng(0).standard_normal(sr // 2).astype(np.float32)
    target_path = tmp_path / "target.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(target_path), target, sr)
    sf.write(str(noise_path), noise, 8_000)

    voiced_mask = np.ones(target.size // 512, dtype=bool)
    item = Scenario1Item(
        sample_id="utt_x",
        target_path=target_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )

    with pytest.raises(ValueError):
        run(items=[item], provider=StubPipelineProvider(), sample_rate=sr, snrs_db=(0.0,))


def test_real_provider_imports_only_when_used():
    """Constructing :class:`RealPipelineProvider` must not pull in torch."""
    from mellonella_bench.scenarios.pipeline_provider import RealPipelineProvider

    provider = RealPipelineProvider()
    assert provider.config is None
    assert provider.components is None


def test_real_provider_requires_enrollment_path():
    from mellonella_bench.scenarios.pipeline_provider import RealPipelineProvider

    class _FakeItem:
        pass

    provider = RealPipelineProvider()
    with pytest.raises(ValueError):
        provider.for_item(_FakeItem())


def test_pipeline_callable_alias():
    """The ``PipelineCallable`` alias is callable-shaped and re-exported."""
    cb: PipelineCallable = StubPipelineProvider().for_item(None)
    result = cb(np.zeros(512, dtype=np.float32), 16_000)
    assert result.audio.shape == (512,)
    assert result.gate_per_frame.dtype == bool
