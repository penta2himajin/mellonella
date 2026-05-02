"""Common scenario plumbing: SNR mixing, result records, CSV emission."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

PipelineCallable = Callable[[np.ndarray, int], "tuple[np.ndarray, np.ndarray]"]
"""``(mixture, sample_rate) -> (output_audio, gate_per_frame_bool)``.

``gate_per_frame_bool`` is the boolean gate decision per SV frame
(``Config.audio.frame_ms`` cadence at ``Config.audio.sv_sr``); it is what
:class:`mellonella_bench.metrics.gating.confusion_from_frames` consumes.
"""


class PipelineProvider(Protocol):
    """Returns a per-item :data:`PipelineCallable`.

    Implementations either ignore the item entirely (the deterministic stub
    used by tests) or build a fresh enrollment pool from the item's
    enrollment recording (the real-pipeline path).
    """

    def for_item(self, item: object) -> PipelineCallable:  # pragma: no cover - protocol
        ...


@dataclass
class SnrSweepEntry:
    """One row's worth of metrics. Used by Scenario 1 (per-SNR) and other
    scenarios that emit one or many rows per item; non-applicable fields
    stay ``None`` and the CSV writer renders them as empty cells.
    """

    sample_id: str
    snr_db: float | None = None
    target_speaker: str = ""
    other_speaker: str = ""
    language: str = ""
    pesq: float | None = None
    stoi: float | None = None
    si_sdr: float | None = None
    gate_tpr: float | None = None
    gate_tnr: float | None = None
    gate_fpr: float | None = None
    gate_fnr: float | None = None
    frame_accuracy: float | None = None
    onset_latency_ms: float | None = None
    offset_latency_ms: float | None = None
    attack_ms: float | None = None
    release_ms: float | None = None
    processing_time_ms: float | None = None
    notes: str = ""


@dataclass
class SnrSweep:
    """Container for a sweep across SNR conditions."""

    scenario: str
    entries: list[SnrSweepEntry] = field(default_factory=list)

    def append(self, entry: SnrSweepEntry) -> None:
        self.entries.append(entry)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.entries:
            path.write_text("")
            return
        fieldnames = [
            "sample_id",
            "scenario",
            "language",
            "snr_db",
            "target_speaker",
            "other_speaker",
            "gate_tpr",
            "gate_tnr",
            "gate_fpr",
            "gate_fnr",
            "pesq",
            "stoi",
            "si_sdr",
            "frame_accuracy",
            "onset_latency_ms",
            "offset_latency_ms",
            "attack_ms",
            "release_ms",
            "processing_time_ms",
            "notes",
        ]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                writer.writerow(
                    {
                        "sample_id": e.sample_id,
                        "scenario": self.scenario,
                        "language": e.language,
                        "snr_db": e.snr_db,
                        "target_speaker": e.target_speaker,
                        "other_speaker": e.other_speaker,
                        "gate_tpr": e.gate_tpr,
                        "gate_tnr": e.gate_tnr,
                        "gate_fpr": e.gate_fpr,
                        "gate_fnr": e.gate_fnr,
                        "pesq": e.pesq,
                        "stoi": e.stoi,
                        "si_sdr": e.si_sdr,
                        "frame_accuracy": e.frame_accuracy,
                        "onset_latency_ms": e.onset_latency_ms,
                        "offset_latency_ms": e.offset_latency_ms,
                        "attack_ms": e.attack_ms,
                        "release_ms": e.release_ms,
                        "processing_time_ms": e.processing_time_ms,
                        "notes": e.notes,
                    }
                )


@dataclass
class ScenarioResult:
    """Aggregate result emitted by a scenario runner."""

    scenario: str
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    sweep: SnrSweep | None = None


def _identity_pipeline(mixture: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Pass-through pipeline: returns the mixture unchanged with all-on gate.

    The gate array length matches the SV frame cadence (50 Hz at 16 kHz),
    so the output is shape-compatible with what real scenarios feed into
    :class:`mellonella_bench.metrics.gating.confusion_from_frames`.
    """
    del sample_rate
    n_frames = max(1, mixture.size // 320)  # 20 ms frames at 16 kHz
    return mixture, np.ones(n_frames, dtype=bool)


@dataclass
class StubPipelineProvider:
    """Deterministic identity provider. Ignores the item; used by tests / dry runs."""

    def for_item(self, item: object) -> PipelineCallable:
        del item
        return _identity_pipeline


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix ``speech`` and ``noise`` at the requested SNR (dB).

    The shorter array is tiled to match. Output is the same length as
    ``speech``.
    """
    if speech.ndim != 1 or noise.ndim != 1:
        raise ValueError("speech and noise must be 1-D")
    if speech.size == 0 or noise.size == 0:
        raise ValueError("speech and noise must be non-empty")

    if noise.size < speech.size:
        repeats = int(np.ceil(speech.size / noise.size))
        noise_long = np.tile(noise, repeats)[: speech.size]
    else:
        offset = 0 if rng is None else int(rng.integers(0, noise.size - speech.size + 1))
        noise_long = noise[offset : offset + speech.size]

    speech_power = float(np.mean(speech.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise_long.astype(np.float64) ** 2))
    if speech_power == 0.0 or noise_power == 0.0:
        raise ValueError("zero-energy speech or noise; cannot mix at SNR")

    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return (speech.astype(np.float32) + scale * noise_long.astype(np.float32)).astype(np.float32)
