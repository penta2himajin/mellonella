"""Scenario 6: drift verification (auto-learn long-running behaviour).

    INPUT:  one explicit-enrollment recording + an ordered sequence of
            voice-variant recordings (e.g. normal / hoarse / tired)
    EXPECT: gate accuracy holds across variants because the auto-learn
            FIFO absorbs each variant; drift is bounded and any reset
            event is logged

Per ``docs/benchmarks.md`` and ``docs/evaluation.md`` §Scenario 6.

Per-item metrics:

* ``frame_accuracy``           gate-on rate across the concatenated test track
                               (we assume target is speaking everywhere)
* ``gate_tpr``                 same as ``frame_accuracy`` here
* ``auto_learn_admissions``    number of admit events from the wired pipeline
* ``auto_learn_resets``        number of drift-reset events
* ``anchor_distance_final``    pool's median anchor distance at end of run

The runner does NOT itself simulate noise; if you want noise-perturbed
drift use Scenario 1 inputs as the variant_paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import (
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
)


@dataclass
class Scenario6Item:
    """One drift-verification evaluation item."""

    sample_id: str
    enrollment_path: Path
    variant_paths: tuple[Path, ...]
    """Ordered list of voice-variant recordings of the SAME target speaker."""
    variant_durations_sec: tuple[float | None, ...] = field(default_factory=tuple)
    """Per-variant truncation in seconds. ``None`` or shorter than the
    available audio means use the full recording. Empty tuple = use full
    duration for every variant."""


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def assemble_track(
    item: Scenario6Item,
    sample_rate: int,
) -> np.ndarray:
    """Load every variant in order, truncate, and concatenate into one buffer."""
    parts: list[np.ndarray] = []
    durations = list(item.variant_durations_sec) + [None] * max(
        0, len(item.variant_paths) - len(item.variant_durations_sec)
    )
    for path, duration_sec in zip(item.variant_paths, durations, strict=False):
        audio, sr = _load_mono(path)
        if sr != sample_rate:
            raise ValueError(f"{path}: expected {sample_rate} Hz, got {sr} Hz")
        if duration_sec is not None:
            n = int(round(duration_sec * sample_rate))
            audio = audio[:n]
        parts.append(audio)
    if not parts:
        raise ValueError(f"{item.sample_id}: variant_paths is empty")
    return np.concatenate(parts).astype(np.float32)


def evaluate_one(
    item: Scenario6Item,
    provider: PipelineProvider,
    sample_rate: int,
) -> SnrSweepEntry:
    """Run the concatenated variant track through the pipeline once.

    Auto-learn admission is enabled by default in :class:`GatingConfig`;
    the real provider will populate ``auto_learn_*`` fields. The stub
    provider records 0 admissions / 0 resets, which is also a valid
    smoke-mode output.
    """
    audio = assemble_track(item, sample_rate)
    pipeline = provider.for_item(item)
    t0 = time.perf_counter()
    result = pipeline(audio, sample_rate)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    gate_tpr = 0.0 if result.gate_per_frame.size == 0 else float(result.gate_per_frame.mean())

    return SnrSweepEntry(
        sample_id=item.sample_id,
        gate_tpr=gate_tpr,
        frame_accuracy=gate_tpr,
        auto_learn_admissions=result.auto_learn_admissions,
        auto_learn_resets=result.auto_learn_resets,
        anchor_distance_final=result.anchor_distance_final,
        processing_time_ms=elapsed_ms,
    )


def run(
    items: list[Scenario6Item],
    provider: PipelineProvider | None = None,
    sample_rate: int = 16_000,
    output_csv: Path | None = None,
) -> ScenarioResult:
    """Evaluate every item and return aggregated metrics."""
    if provider is None:
        provider = StubPipelineProvider()

    sweep = SnrSweep(scenario="scenario_6")
    for item in items:
        sweep.append(evaluate_one(item, provider, sample_rate))

    if output_csv is not None:
        sweep.write_csv(output_csv)

    aggregate: dict[str, float] = {}
    if sweep.entries:
        for field_name in (
            "gate_tpr",
            "frame_accuracy",
            "auto_learn_admissions",
            "auto_learn_resets",
            "anchor_distance_final",
        ):
            values = [
                getattr(e, field_name) for e in sweep.entries if getattr(e, field_name) is not None
            ]
            if values:
                aggregate[f"{field_name}_mean"] = float(np.mean(values))

    return ScenarioResult(
        scenario="scenario_6",
        n_samples=len(items),
        metrics=aggregate,
        sweep=sweep,
    )
