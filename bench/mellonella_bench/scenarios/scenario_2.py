"""Scenario 2: solo other speaker + noise.

    INPUT:  other_speaker_audio + noise (MUSAN/DEMAND, SNR sweep)
    EXPECT: gate mute everywhere; output ≈ silence

The mirror of Scenario 1: the enrollment pool is built from the *target*
speaker, but the audio fed through the pipeline is from a DIFFERENT
speaker. Anything the gate lets through is a false-acceptance — the
score we want to drive down.

Phase 1 PoC scope:
- SNR sweep: -5, 0, 5, 10, 15, 20 dB (same as Scenario 1)
- TNR (frame-level mute rate during voiced frames of the other speaker)
- FPR (frame-level pass rate during voiced frames; complement of TNR)
- output_rms_db (overall energy of the gated output, dBFS — lower = better
  attenuation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import (
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
    mix_at_snr,
)

DEFAULT_SNRS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
EPS = 1e-12


@dataclass
class Scenario2Item:
    """One non-target × noise pair to evaluate.

    ``enrollment_path`` should point at a recording of the *target*
    speaker (different from ``other_speaker_path``). The pipeline builds
    an enrollment pool from it and we then check that the gate mutes
    ``other_speaker_path`` despite the noise.
    """

    sample_id: str
    other_speaker_path: Path
    noise_path: Path
    voiced_mask: np.ndarray
    """Frame-level mask of where the *other* speaker is voiced (True ==
    speech). The TNR/FPR ratios are restricted to these frames so that
    silence between utterances does not inflate the apparent TNR."""
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


def _rms_db(audio: np.ndarray) -> float:
    """Return 20·log10(rms) in dBFS. Empty / zero input returns -inf."""
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms + EPS))


def evaluate_one(
    item: Scenario2Item,
    provider: PipelineProvider,
    sample_rate: int,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    rng: np.random.Generator | None = None,
) -> list[SnrSweepEntry]:
    """Run the per-item pipeline over each SNR.

    Returns one entry per SNR with TNR/FPR computed *within voiced frames
    of the other speaker*; output_rms_db reports the overall output
    energy (lower = better attenuation).
    """
    other, other_sr = _load_mono(item.other_speaker_path)
    noise, noise_sr = _load_mono(item.noise_path)
    if other_sr != sample_rate or noise_sr != sample_rate:
        raise ValueError(
            f"sample-rate mismatch (other={other_sr}, noise={noise_sr}, expected={sample_rate})"
        )

    pipeline = provider.for_item(item)
    voiced = item.voiced_mask.astype(bool)

    rows: list[SnrSweepEntry] = []
    for snr in snrs_db:
        mixture = mix_at_snr(other, noise, snr, rng=rng)
        t0 = time.perf_counter()
        result = pipeline(mixture, sample_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gate_aligned = _truncate_to_match(voiced, result.gate_per_frame.astype(bool))
        voiced_count = int(voiced.sum())
        if voiced_count > 0:
            pass_voiced = int((gate_aligned & voiced).sum())
            mute_voiced = voiced_count - pass_voiced
            fpr = pass_voiced / voiced_count
            tnr = mute_voiced / voiced_count
        else:
            fpr = 0.0
            tnr = 0.0

        rms_db = _rms_db(result.audio)

        rows.append(
            SnrSweepEntry(
                sample_id=item.sample_id,
                snr_db=snr,
                gate_tnr=tnr,
                gate_fpr=fpr,
                output_rms_db=rms_db,
                processing_time_ms=elapsed_ms,
            )
        )
    return rows


def run(
    items: list[Scenario2Item],
    provider: PipelineProvider | None = None,
    sample_rate: int = 16_000,
    output_csv: Path | None = None,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    seed: int = 0,
) -> ScenarioResult:
    """Evaluate every item and return aggregated metrics.

    ``provider`` defaults to :class:`StubPipelineProvider` — useful for
    smoke tests and to verify the harness layout end-to-end without
    pulling in heavy ML dependencies.
    """
    if provider is None:
        provider = StubPipelineProvider()

    sweep = SnrSweep(scenario="scenario_2")
    rng = np.random.default_rng(seed)
    for item in items:
        rows = evaluate_one(item, provider, sample_rate, snrs_db, rng=rng)
        for row in rows:
            sweep.append(row)

    if output_csv is not None:
        sweep.write_csv(output_csv)

    aggregate: dict[str, float] = {}
    if sweep.entries:
        for field_name in ("gate_tnr", "gate_fpr", "output_rms_db"):
            values = [
                getattr(e, field_name)
                for e in sweep.entries
                if getattr(e, field_name) is not None and np.isfinite(getattr(e, field_name))
            ]
            if values:
                aggregate[f"{field_name}_mean"] = float(np.mean(values))

    return ScenarioResult(
        scenario="scenario_2",
        n_samples=len(items),
        metrics=aggregate,
        sweep=sweep,
    )
