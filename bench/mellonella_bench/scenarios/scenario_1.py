"""Scenario 1: solo target speaker + noise.

    INPUT:  target_speaker_audio (clean) + noise (MUSAN/DEMAND, SNR sweep)
    EXPECT: gate pass, NS-cleaned target speech audible

Phase 1 PoC scope:
- SNR sweep: -5, 0, 5, 10, 15, 20 dB
- TPR (frame-level pass rate during voiced regions)
- PESQ / STOI / SI-SDR vs the ground-truth clean speech
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..metrics.gating import confusion_from_frames
from ..metrics.ns_quality import (
    MissingDependencyError,
    pesq_score,
    si_sdr,
    stoi_score,
)
from .base import ScenarioResult, SnrSweep, SnrSweepEntry, mix_at_snr

DEFAULT_SNRS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


@dataclass
class Scenario1Item:
    """One target × noise pair to evaluate."""

    sample_id: str
    target_path: Path
    noise_path: Path
    voiced_mask: np.ndarray
    """Frame-level ground-truth voicing (True == speech) at the SV frame rate."""


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def evaluate_one(
    item: Scenario1Item,
    pipeline_callable,
    sample_rate: int,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    pesq_mode: str | None = "wb",
    rng: np.random.Generator | None = None,
) -> list[SnrSweepEntry]:
    """Run ``pipeline_callable`` over each SNR for one item.

    ``pipeline_callable(mixture: np.ndarray, sr: int) -> (output: np.ndarray,
    gate_decisions: np.ndarray[bool])`` is the integration point with
    :func:`mellonella_poc.pipeline.process_offline`. Tests substitute a
    deterministic stub.
    """
    target, target_sr = _load_mono(item.target_path)
    noise, noise_sr = _load_mono(item.noise_path)
    if target_sr != sample_rate or noise_sr != sample_rate:
        raise ValueError(
            f"sample-rate mismatch (target={target_sr}, noise={noise_sr}, expected={sample_rate})"
        )

    rows: list[SnrSweepEntry] = []
    for snr in snrs_db:
        mixture = mix_at_snr(target, noise, snr, rng=rng)
        t0 = time.perf_counter()
        out, gate_decisions = pipeline_callable(mixture, sample_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        confusion = confusion_from_frames(item.voiced_mask, gate_decisions)
        sisdr = si_sdr(target, out[: target.size])

        pesq_val: float | None = None
        if pesq_mode is not None:
            try:
                pesq_val = pesq_score(target, out[: target.size], sample_rate, mode=pesq_mode)
            except (MissingDependencyError, ValueError):
                pesq_val = None

        try:
            stoi_val: float | None = stoi_score(target, out[: target.size], sample_rate)
        except (MissingDependencyError, ValueError):
            stoi_val = None

        rows.append(
            SnrSweepEntry(
                sample_id=item.sample_id,
                snr_db=snr,
                gate_tpr=confusion.tpr,
                gate_tnr=confusion.tnr,
                gate_fpr=confusion.fpr,
                gate_fnr=confusion.fnr,
                pesq=pesq_val,
                stoi=stoi_val,
                si_sdr=sisdr,
                processing_time_ms=elapsed_ms,
            )
        )
    return rows


def run(
    items: list[Scenario1Item],
    pipeline_callable,
    sample_rate: int,
    output_csv: Path | None = None,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    pesq_mode: str | None = "wb",
    seed: int = 0,
) -> ScenarioResult:
    """Evaluate every item and return aggregated metrics."""
    sweep = SnrSweep(scenario="scenario_1")
    rng = np.random.default_rng(seed)
    for item in items:
        for row in evaluate_one(
            item, pipeline_callable, sample_rate, snrs_db, pesq_mode=pesq_mode, rng=rng
        ):
            sweep.append(row)

    if output_csv is not None:
        sweep.write_csv(output_csv)

    aggregate: dict[str, float] = {}
    if sweep.entries:
        for field_name in ("gate_tpr", "gate_tnr", "si_sdr", "pesq", "stoi"):
            values = [
                getattr(e, field_name) for e in sweep.entries if getattr(e, field_name) is not None
            ]
            if values:
                aggregate[f"{field_name}_mean"] = float(np.mean(values))

    return ScenarioResult(
        scenario="scenario_1",
        n_samples=len(items),
        metrics=aggregate,
        sweep=sweep,
    )
