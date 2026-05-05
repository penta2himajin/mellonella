"""Scenario 4: simultaneous target + other speech (FP-tolerant verification).

    INPUT:  target_speaker_audio + other_speaker_audio mixed at a range of
            target-to-other ratios
    EXPECT: gate passes whenever the target component is audible (FP-tolerant policy)
            even if the other speaker leaks through

Per ``docs/benchmarks.md`` Scenario 4 / ``docs/evaluation.md`` §Scenario 4
the goal is to verify that the FP-tolerant policy actually keeps target
speech audible during overlapping speech. We sweep the target-to-other
power ratio (``target_to_other_ratios_db``) instead of SNR vs noise:

    +inf dB  → target only        (TPR upper bound; matches scenario_1 idea)
    +9.5 dB  → target dominant
    +3.0 dB  → target slightly louder
    0.0 dB   → equal
    -3.0 dB  → other slightly louder
    -9.5 dB  → other dominant
    -inf dB  → other only         (FPR test; matches scenario_2 idea)

For each ratio we record:

* ``gate_tpr``    pass rate during voiced frames (we *want* this high
                  except at -inf where the target is absent)
* ``si_sdr``      SI-SDR of the gated output vs the clean target
                  ground-truth — drops as the gate leaks more "other"

``snr_db`` in :class:`SnrSweepEntry` is repurposed as the per-row
target-to-other ratio in dB (positive = target louder); the scenario
column on the CSV row distinguishes it from the noise SNR used by
scenario_1 / scenario_2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..metrics.gating import confusion_from_frames
from ..metrics.ns_quality import si_sdr
from .base import (
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
)

DEFAULT_RATIOS_DB: tuple[float, ...] = (
    float("inf"),  # target only
    9.0,
    3.0,
    0.0,
    -3.0,
    -9.0,
    float("-inf"),  # other only
)


@dataclass
class Scenario4Item:
    """One simultaneous-speech evaluation item.

    ``target_path`` and ``other_path`` are mixed together at each ratio in
    ``target_to_other_ratios_db``. The runner truncates both to the
    shorter recording before mixing so the two signals are perfectly
    aligned.

    ``voiced_mask`` should be the frame-level mask of where the *target*
    is voiced (not the mix). TPR is restricted to those frames so silence
    or tail padding cannot dilute the metric.
    """

    sample_id: str
    target_path: Path
    other_path: Path
    voiced_mask: np.ndarray
    enrollment_path: Path | None = None
    target_to_other_ratios_db: tuple[float, ...] = field(default_factory=lambda: DEFAULT_RATIOS_DB)


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _truncate_to_match(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if candidate.size >= reference.size:
        return candidate[: reference.size]
    pad = np.zeros(reference.size - candidate.size, dtype=candidate.dtype)
    return np.concatenate([candidate, pad])


def mix_at_ratio(
    target: np.ndarray,
    other: np.ndarray,
    ratio_db: float,
) -> np.ndarray:
    """Mix ``target`` + ``other`` at the requested target-to-other power ratio.

    * ``ratio_db = +inf`` returns ``target`` unchanged
    * ``ratio_db = -inf`` returns ``other`` (truncated to ``target`` length)
    * Otherwise both signals are truncated to the same length, ``other`` is
      scaled to make ``power(target) / power(scale * other) == 10**(ratio_db/10)``,
      and the sum is returned.
    """
    if target.ndim != 1 or other.ndim != 1:
        raise ValueError("mix_at_ratio expects 1-D arrays")
    if target.size == 0:
        raise ValueError("target must be non-empty")
    other_aligned = _truncate_to_match(target, other)

    if ratio_db == float("inf"):
        return target.astype(np.float32, copy=False)
    if ratio_db == float("-inf"):
        return other_aligned.astype(np.float32, copy=False)

    target_power = float(np.mean(target.astype(np.float64) ** 2))
    other_power = float(np.mean(other_aligned.astype(np.float64) ** 2))
    if target_power == 0.0 or other_power == 0.0:
        raise ValueError("zero-energy target or other; cannot mix at finite ratio")

    target_other_power = target_power / (10.0 ** (ratio_db / 10.0))
    scale = float(np.sqrt(target_other_power / other_power))
    return (target.astype(np.float32) + scale * other_aligned.astype(np.float32)).astype(np.float32)


def evaluate_one(
    item: Scenario4Item,
    provider: PipelineProvider,
    sample_rate: int,
    *,
    output_sr: int = 48_000,
) -> list[SnrSweepEntry]:
    """Run the per-item pipeline at every target-to-other ratio."""
    target, target_sr = _load_mono(item.target_path)
    other, other_sr = _load_mono(item.other_path)
    if target_sr != sample_rate or other_sr != sample_rate:
        raise ValueError(
            f"sample-rate mismatch (target={target_sr}, other={other_sr}, expected={sample_rate})"
        )

    pipeline = provider.for_item(item)
    voiced = item.voiced_mask.astype(bool)
    rows: list[SnrSweepEntry] = []

    for ratio_db in item.target_to_other_ratios_db:
        mixture = mix_at_ratio(target, other, ratio_db)
        t0 = time.perf_counter()
        result = pipeline(mixture, sample_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gate_aligned = _truncate_to_match(voiced, result.gate_per_frame.astype(bool))
        confusion = confusion_from_frames(voiced, gate_aligned)

        # SI-SDR is computed at the input sample rate (== sample_rate),
        # using the clean target as reference. Result.audio comes back at
        # ``output_sr`` (typically 48 kHz); we trust the scenario harness
        # to be told what that is via the kwarg.
        if result.audio.size == 0:
            sisdr_value: float | None = None
        else:
            from math import gcd

            from scipy.signal import resample_poly

            if output_sr == sample_rate:
                output_at_target_sr = result.audio
            else:
                g = gcd(output_sr, sample_rate)
                output_at_target_sr = resample_poly(
                    result.audio, sample_rate // g, output_sr // g
                ).astype(np.float32)
            n = min(target.size, output_at_target_sr.size)
            sisdr_value = si_sdr(target[:n], output_at_target_sr[:n])

        # Encode +/- inf as None so the CSV stays well-formed.
        snr_field: float | None
        if ratio_db == float("inf") or ratio_db == float("-inf"):
            snr_field = None
            note_suffix = "target_only" if ratio_db == float("inf") else "other_only"
        else:
            snr_field = ratio_db
            note_suffix = ""

        rows.append(
            SnrSweepEntry(
                sample_id=item.sample_id,
                snr_db=snr_field,
                gate_tpr=confusion.tpr,
                gate_tnr=confusion.tnr,
                gate_fpr=confusion.fpr,
                gate_fnr=confusion.fnr,
                si_sdr=sisdr_value,
                processing_time_ms=elapsed_ms,
                notes=note_suffix,
            )
        )
    return rows


def run(
    items: list[Scenario4Item],
    provider: PipelineProvider | None = None,
    sample_rate: int = 16_000,
    output_csv: Path | None = None,
    *,
    output_sr: int = 48_000,
) -> ScenarioResult:
    """Evaluate every item and return aggregated metrics."""
    if provider is None:
        provider = StubPipelineProvider()

    sweep = SnrSweep(scenario="scenario_4")
    for item in items:
        for row in evaluate_one(item, provider, sample_rate, output_sr=output_sr):
            sweep.append(row)

    if output_csv is not None:
        sweep.write_csv(output_csv)

    aggregate: dict[str, float] = {}
    if sweep.entries:
        for field_name in ("gate_tpr", "gate_tnr", "si_sdr"):
            values = [
                getattr(e, field_name)
                for e in sweep.entries
                if getattr(e, field_name) is not None and np.isfinite(getattr(e, field_name))
            ]
            if values:
                aggregate[f"{field_name}_mean"] = float(np.mean(values))

    return ScenarioResult(
        scenario="scenario_4",
        n_samples=len(items),
        metrics=aggregate,
        sweep=sweep,
    )
