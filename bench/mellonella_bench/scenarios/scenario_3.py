"""Scenario 3: alternating speech (target → silence → other → silence → target …).

    INPUT:  concatenated audio of target/other/silence segments
    EXPECT: gate passes only the target segments

Per ``docs/benchmarks.md`` and ``docs/evaluation.md``, this scenario
quantifies the gate's *transition* behaviour: how quickly it opens when
the target starts speaking and how quickly it closes when the target
stops. Output metrics:

* ``frame_accuracy``     binary (target vs not-target) per-frame accuracy
* ``gate_tpr / tnr / fpr / fnr``
* ``onset_latency_ms``   mean delay from a "→ target" transition to the
                         first ``pass`` decision in that segment
* ``offset_latency_ms``  mean delay from a "target → not-target" transition
                         to the first ``mute`` decision in that segment
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..metrics.gating import confusion_from_frames, frame_accuracy
from .base import (
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
)

SegmentLabel = Literal["target", "other", "silence"]
DEFAULT_SEGMENTS: tuple[tuple[SegmentLabel, float], ...] = (
    ("target", 4.0),
    ("silence", 1.0),
    ("other", 4.0),
    ("silence", 1.0),
    ("target", 4.0),
    ("silence", 1.0),
    ("other", 4.0),
)


@dataclass
class Scenario3Item:
    """One alternating-speech evaluation item."""

    sample_id: str
    target_path: Path
    other_path: Path
    enrollment_path: Path | None = None
    segments: tuple[tuple[SegmentLabel, float], ...] = field(
        default_factory=lambda: DEFAULT_SEGMENTS
    )


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _take_or_tile(audio: np.ndarray, n: int) -> np.ndarray:
    """Return ``n`` samples from ``audio``, tiling if shorter."""
    if audio.size >= n:
        return audio[:n]
    repeats = (n + audio.size - 1) // audio.size
    return np.tile(audio, repeats)[:n]


def assemble_audio(
    segments: tuple[tuple[SegmentLabel, float], ...],
    target: np.ndarray,
    other: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Concatenate per-segment audio according to ``segments``."""
    parts: list[np.ndarray] = []
    for label, duration_sec in segments:
        n = int(round(duration_sec * sample_rate))
        if label == "silence":
            parts.append(np.zeros(n, dtype=np.float32))
        elif label == "target":
            parts.append(_take_or_tile(target, n).astype(np.float32))
        elif label == "other":
            parts.append(_take_or_tile(other, n).astype(np.float32))
        else:
            raise ValueError(f"unknown segment label: {label!r}")
    return np.concatenate(parts)


def voicing_mask_at_frame_rate(
    segments: tuple[tuple[SegmentLabel, float], ...],
    sample_rate: int,
    samples_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(target_mask, frame_label)``.

    * ``target_mask``  per-frame bool: True iff the frame's whole window
      lies inside a "target" segment.
    * ``frame_label``  per-frame int: 0=silence, 1=target, 2=other.

    The two arrays are aligned and have the same length, computed by walking
    the segment list and counting full ``samples_per_frame``-wide frames per
    segment.
    """
    target_mask: list[bool] = []
    label_codes: list[int] = []
    code = {"silence": 0, "target": 1, "other": 2}
    for label, duration_sec in segments:
        n_samples = int(round(duration_sec * sample_rate))
        n_frames = n_samples // samples_per_frame
        target_mask.extend([label == "target"] * n_frames)
        label_codes.extend([code[label]] * n_frames)
    return np.asarray(target_mask, dtype=bool), np.asarray(label_codes, dtype=np.int8)


def latencies_per_transition(
    target_mask: np.ndarray,
    predicted: np.ndarray,
    frame_ms: float,
) -> tuple[list[float], list[float]]:
    """Walk the ground-truth target_mask and collect per-transition latencies.

    For each rising edge in ``target_mask`` (silence/other → target) compute
    the onset latency = time until ``predicted`` first goes True within the
    target run. For each falling edge compute the offset latency = time until
    ``predicted`` first goes False after the target run ends. Latencies are
    only emitted when the prediction actually catches up (otherwise the
    transition is skipped).
    """
    if target_mask.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {target_mask.shape} vs {predicted.shape}")
    onset_lats: list[float] = []
    offset_lats: list[float] = []
    n = target_mask.size
    if n == 0:
        return onset_lats, offset_lats

    # Walk the segments by detecting (start, end] runs where target_mask is True.
    in_target = bool(target_mask[0])
    run_start = 0 if in_target else -1
    for i in range(1, n + 1):
        cur = bool(target_mask[i]) if i < n else False
        if not in_target and cur:
            run_start = i
            in_target = True
        elif in_target and not cur:
            # Compute onset latency in [run_start, i)
            window = predicted[run_start:i]
            on_idx = np.where(window)[0]
            if on_idx.size:
                onset_lats.append(on_idx[0] * frame_ms)
            # Compute offset latency in [i, n)
            tail = predicted[i:]
            off_idx = np.where(~tail)[0]
            if off_idx.size:
                offset_lats.append(off_idx[0] * frame_ms)
            in_target = False
    return onset_lats, offset_lats


def evaluate_one(
    item: Scenario3Item,
    provider: PipelineProvider,
    sample_rate: int,
    *,
    frame_ms: float = 32.0,
) -> SnrSweepEntry:
    """Run one alternating-speech item through the pipeline.

    ``frame_ms`` should match :attr:`AudioConfig.vad_frame_ms` from the
    pipeline (32 ms by default for silero-vad >= 6).
    """
    target, target_sr = _load_mono(item.target_path)
    other, other_sr = _load_mono(item.other_path)
    if target_sr != sample_rate or other_sr != sample_rate:
        raise ValueError(
            f"sample-rate mismatch (target={target_sr}, other={other_sr}, expected={sample_rate})"
        )

    samples_per_frame = int(round(frame_ms * sample_rate / 1000.0))
    if samples_per_frame <= 0:
        raise ValueError(f"frame_ms={frame_ms} yields zero samples at {sample_rate} Hz")

    audio = assemble_audio(item.segments, target, other, sample_rate)
    target_mask, _label_codes = voicing_mask_at_frame_rate(
        item.segments, sample_rate, samples_per_frame
    )

    pipeline = provider.for_item(item)
    t0 = time.perf_counter()
    _output_audio, gate_per_frame = pipeline(audio, sample_rate)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Trim/pad gate_per_frame to align with target_mask.
    if gate_per_frame.size >= target_mask.size:
        gate_aligned = gate_per_frame[: target_mask.size].astype(bool)
    else:
        pad = np.zeros(target_mask.size - gate_per_frame.size, dtype=bool)
        gate_aligned = np.concatenate([gate_per_frame.astype(bool), pad])

    confusion = confusion_from_frames(target_mask, gate_aligned)
    accuracy = frame_accuracy(target_mask, gate_aligned)
    onset_lats, offset_lats = latencies_per_transition(target_mask, gate_aligned, frame_ms)

    return SnrSweepEntry(
        sample_id=item.sample_id,
        gate_tpr=confusion.tpr,
        gate_tnr=confusion.tnr,
        gate_fpr=confusion.fpr,
        gate_fnr=confusion.fnr,
        frame_accuracy=accuracy,
        onset_latency_ms=float(np.mean(onset_lats)) if onset_lats else None,
        offset_latency_ms=float(np.mean(offset_lats)) if offset_lats else None,
        processing_time_ms=elapsed_ms,
    )


def run(
    items: list[Scenario3Item],
    provider: PipelineProvider | None = None,
    sample_rate: int = 16_000,
    output_csv: Path | None = None,
    *,
    frame_ms: float = 32.0,
) -> ScenarioResult:
    """Evaluate every item and return aggregated metrics."""
    if provider is None:
        provider = StubPipelineProvider()

    sweep = SnrSweep(scenario="scenario_3")
    for item in items:
        sweep.append(evaluate_one(item, provider, sample_rate, frame_ms=frame_ms))

    if output_csv is not None:
        sweep.write_csv(output_csv)

    aggregate: dict[str, float] = {}
    if sweep.entries:
        for field_name in (
            "gate_tpr",
            "gate_tnr",
            "frame_accuracy",
            "onset_latency_ms",
            "offset_latency_ms",
        ):
            values = [
                getattr(e, field_name) for e in sweep.entries if getattr(e, field_name) is not None
            ]
            if values:
                aggregate[f"{field_name}_mean"] = float(np.mean(values))

    return ScenarioResult(
        scenario="scenario_3",
        n_samples=len(items),
        metrics=aggregate,
        sweep=sweep,
    )
