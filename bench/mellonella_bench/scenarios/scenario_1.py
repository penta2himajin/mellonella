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
from .base import (
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
    mix_at_snr,
)

DEFAULT_SNRS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


@dataclass
class Scenario1Item:
    """One target × noise pair to evaluate.

    ``enrollment_path`` is consumed by the real pipeline provider to build
    a per-item :class:`EmbeddingPool`. It can be left as ``None`` when
    running with the deterministic stub.
    """

    sample_id: str
    target_path: Path
    noise_path: Path
    voiced_mask: np.ndarray
    """Frame-level ground-truth voicing (True == speech) at the SV frame rate."""
    enrollment_path: Path | None = None


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _truncate_to_match(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Trim or right-pad ``candidate`` so it lines up with ``reference``."""
    if candidate.size >= reference.size:
        return candidate[: reference.size]
    pad = np.zeros(reference.size - candidate.size, dtype=candidate.dtype)
    return np.concatenate([candidate, pad])


def evaluate_one(
    item: Scenario1Item,
    provider: PipelineProvider,
    sample_rate: int,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    pesq_mode: str | None = "wb",
    rng: np.random.Generator | None = None,
) -> list[SnrSweepEntry]:
    """Run the per-item pipeline over each SNR.

    The ``provider`` builds a :data:`PipelineCallable` for this item; the
    callable returns ``(output_audio, gate_per_frame_bool)`` aligned with
    ``item.voiced_mask``.
    """
    target, target_sr = _load_mono(item.target_path)
    noise, noise_sr = _load_mono(item.noise_path)
    if target_sr != sample_rate or noise_sr != sample_rate:
        raise ValueError(
            f"sample-rate mismatch (target={target_sr}, noise={noise_sr}, expected={sample_rate})"
        )

    pipeline = provider.for_item(item)

    rows: list[SnrSweepEntry] = []
    for snr in snrs_db:
        mixture = mix_at_snr(target, noise, snr, rng=rng)
        t0 = time.perf_counter()
        output_audio, gate_per_frame = pipeline(mixture, sample_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gate_aligned = _truncate_to_match(item.voiced_mask, gate_per_frame.astype(bool))
        confusion = confusion_from_frames(item.voiced_mask, gate_aligned)
        out_aligned = _truncate_to_match(target, output_audio)
        sisdr = si_sdr(target, out_aligned)

        pesq_val: float | None = None
        if pesq_mode is not None:
            try:
                pesq_val = pesq_score(target, out_aligned, sample_rate, mode=pesq_mode)
            except (MissingDependencyError, ValueError):
                pesq_val = None

        try:
            stoi_val: float | None = stoi_score(target, out_aligned, sample_rate)
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
    provider: PipelineProvider | None = None,
    sample_rate: int = 16_000,
    output_csv: Path | None = None,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    pesq_mode: str | None = "wb",
    seed: int = 0,
) -> ScenarioResult:
    """Evaluate every item and return aggregated metrics.

    ``provider`` defaults to :class:`StubPipelineProvider` — useful for
    smoke tests and to verify the harness layout end-to-end without
    pulling in heavy ML dependencies.
    """
    if provider is None:
        provider = StubPipelineProvider()

    sweep = SnrSweep(scenario="scenario_1")
    rng = np.random.default_rng(seed)
    for item in items:
        rows = evaluate_one(item, provider, sample_rate, snrs_db, pesq_mode=pesq_mode, rng=rng)
        for row in rows:
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
