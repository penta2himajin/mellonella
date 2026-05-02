"""Gating accuracy metrics.

Operates on aligned frame-level boolean / int arrays:

- ``ground_truth``  per-frame label (0 = silence/other, 1 = target)
- ``predicted``     per-frame gate decision (0 = mute, 1 = pass)

Latency metrics quantify the delay between a ground-truth onset/offset and
the gate's response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    """Frame-level confusion matrix counts."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def tpr(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def tnr(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def fnr(self) -> float:
        denom = self.fn + self.tp
        return self.fn / denom if denom else 0.0


def confusion_from_frames(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> ConfusionCounts:
    """Build a confusion-counts object from boolean / 0-1 frame arrays."""
    gt = np.asarray(ground_truth).astype(bool)
    pr = np.asarray(predicted).astype(bool)
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pr.shape}")
    tp = int(np.count_nonzero(gt & pr))
    fp = int(np.count_nonzero(~gt & pr))
    tn = int(np.count_nonzero(~gt & ~pr))
    fn = int(np.count_nonzero(gt & ~pr))
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def frame_accuracy(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of frames where ``predicted == ground_truth``."""
    gt = np.asarray(ground_truth).astype(bool)
    pr = np.asarray(predicted).astype(bool)
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pr.shape}")
    if gt.size == 0:
        return 0.0
    return float(np.mean(gt == pr))


def onset_offset_latency(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    frame_ms: float,
) -> tuple[float | None, float | None]:
    """Return (onset_latency_ms, offset_latency_ms) for the first transition.

    - onset = first ground-truth False→True index. Latency = first frame at or
      after that index where ``predicted`` is True.
    - offset = first ground-truth True→False index after the onset.

    Either value is ``None`` if the transition or the matching prediction
    isn't found.
    """
    gt = np.asarray(ground_truth).astype(bool)
    pr = np.asarray(predicted).astype(bool)
    if gt.shape != pr.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pr.shape}")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be > 0")

    onset_idx = _first_rising_edge(gt)
    onset_lat: float | None
    if onset_idx is None:
        onset_lat = None
    else:
        match = _first_index_where(pr[onset_idx:], True)
        onset_lat = match * frame_ms if match is not None else None

    offset_lat: float | None
    if onset_idx is None:
        offset_lat = None
    else:
        falling = _first_falling_edge(gt[onset_idx:])
        if falling is None:
            offset_lat = None
        else:
            offset_idx = onset_idx + falling
            match = _first_index_where(pr[offset_idx:], False)
            offset_lat = match * frame_ms if match is not None else None

    return onset_lat, offset_lat


def _first_rising_edge(arr: np.ndarray) -> int | None:
    if arr.size == 0:
        return None
    if arr[0]:
        return 0
    diff = np.diff(arr.astype(np.int8))
    rising = np.where(diff == 1)[0]
    if rising.size == 0:
        return None
    return int(rising[0]) + 1


def _first_falling_edge(arr: np.ndarray) -> int | None:
    if arr.size == 0:
        return None
    diff = np.diff(arr.astype(np.int8))
    falling = np.where(diff == -1)[0]
    if falling.size == 0:
        return None
    return int(falling[0]) + 1


def _first_index_where(arr: np.ndarray, value: bool) -> int | None:
    matches = np.where(arr == value)[0]
    if matches.size == 0:
        return None
    return int(matches[0])
