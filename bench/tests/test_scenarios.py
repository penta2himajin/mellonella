"""Tests for the scenario harness and CSV writer."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from mellonella_bench.scenarios.base import (
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    mix_at_snr,
)


def test_mix_at_snr_matches_target_snr():
    rng = np.random.default_rng(0)
    sr = 16_000
    speech = rng.standard_normal(sr).astype(np.float32)
    noise = rng.standard_normal(sr).astype(np.float32)
    for snr in (-5.0, 0.0, 5.0, 10.0):
        mixture = mix_at_snr(speech, noise, snr)
        # SNR(speech / (mixture - speech)) should equal `snr` to good precision.
        residual = mixture - speech
        ratio = float(np.mean(speech**2) / np.mean(residual**2))
        observed = 10.0 * np.log10(ratio)
        assert observed == pytest.approx(snr, abs=0.5)


def test_mix_at_snr_tiles_short_noise():
    sr = 16_000
    speech = np.ones(sr, dtype=np.float32)
    noise = np.full(sr // 4, 0.5, dtype=np.float32)
    out = mix_at_snr(speech, noise, snr_db=10.0)
    assert out.shape == speech.shape


def test_mix_at_snr_rejects_zero_energy():
    speech = np.zeros(100, dtype=np.float32)
    noise = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        mix_at_snr(speech, noise, snr_db=0.0)


def test_mix_at_snr_rejects_multidim():
    speech = np.zeros((2, 100), dtype=np.float32)
    noise = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError):
        mix_at_snr(speech, noise, snr_db=0.0)


def test_snr_sweep_csv_round_trip(tmp_path):
    sweep = SnrSweep(scenario="scenario_1")
    sweep.append(
        SnrSweepEntry(
            sample_id="utt_001",
            snr_db=5.0,
            gate_tpr=0.92,
            gate_tnr=0.85,
            si_sdr=10.5,
        )
    )
    sweep.append(
        SnrSweepEntry(
            sample_id="utt_001",
            snr_db=10.0,
            gate_tpr=0.95,
            gate_tnr=0.88,
            si_sdr=15.2,
        )
    )
    out = tmp_path / "scenario_1.csv"
    sweep.write_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert rows[0]["scenario"] == "scenario_1"
    assert rows[0]["sample_id"] == "utt_001"
    assert float(rows[1]["snr_db"]) == 10.0
    assert float(rows[1]["si_sdr"]) == pytest.approx(15.2)


def test_scenario_result_holds_aggregate_metrics():
    result = ScenarioResult(
        scenario="scenario_1",
        n_samples=42,
        metrics={"gate_tpr_mean": 0.91},
    )
    assert result.metrics["gate_tpr_mean"] == 0.91
    assert result.scenario == "scenario_1"
