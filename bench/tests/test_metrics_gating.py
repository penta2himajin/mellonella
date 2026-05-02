"""Tests for frame-level gating metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_bench.metrics.gating import (
    confusion_from_frames,
    frame_accuracy,
    onset_offset_latency,
)


def test_confusion_perfect_match():
    gt = np.array([0, 0, 1, 1, 1, 0, 0])
    pr = gt.copy()
    cm = confusion_from_frames(gt, pr)
    assert cm.tp == 3
    assert cm.tn == 4
    assert cm.fp == 0
    assert cm.fn == 0
    assert cm.tpr == 1.0
    assert cm.tnr == 1.0
    assert cm.fpr == 0.0
    assert cm.fnr == 0.0


def test_confusion_all_wrong():
    gt = np.array([1, 1, 0, 0])
    pr = np.array([0, 0, 1, 1])
    cm = confusion_from_frames(gt, pr)
    assert cm.tp == 0
    assert cm.tn == 0
    assert cm.fpr == 1.0
    assert cm.fnr == 1.0


def test_confusion_mixed():
    gt = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pr = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    cm = confusion_from_frames(gt, pr)
    assert cm.tp == 2
    assert cm.fn == 2
    assert cm.tn == 2
    assert cm.fp == 2
    assert cm.tpr == 0.5
    assert cm.fpr == 0.5
    assert cm.total == 8


def test_confusion_shape_mismatch():
    with pytest.raises(ValueError):
        confusion_from_frames(np.array([1, 0]), np.array([1]))


def test_confusion_empty_classes():
    gt = np.array([0, 0, 0])
    pr = np.array([0, 0, 0])
    cm = confusion_from_frames(gt, pr)
    assert cm.tpr == 0.0
    assert cm.tnr == 1.0


def test_frame_accuracy():
    gt = np.array([1, 1, 0, 0])
    pr = np.array([1, 0, 0, 0])
    assert frame_accuracy(gt, pr) == pytest.approx(0.75)


def test_frame_accuracy_empty_returns_zero():
    assert frame_accuracy(np.array([]), np.array([])) == 0.0


def test_onset_offset_latency_immediate():
    gt = np.array([0, 0, 1, 1, 1, 0, 0])
    pr = np.array([0, 0, 1, 1, 1, 0, 0])
    onset, offset = onset_offset_latency(gt, pr, frame_ms=20.0)
    assert onset == pytest.approx(0.0)
    assert offset == pytest.approx(0.0)


def test_onset_offset_latency_delayed():
    gt = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0])
    pr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])  # gate is 2 frames late on, 2 late off
    onset, offset = onset_offset_latency(gt, pr, frame_ms=10.0)
    assert onset == pytest.approx(20.0)
    assert offset == pytest.approx(20.0)


def test_onset_offset_latency_no_speech_in_gt():
    gt = np.zeros(5, dtype=bool)
    pr = np.zeros(5, dtype=bool)
    onset, offset = onset_offset_latency(gt, pr, frame_ms=20.0)
    assert onset is None
    assert offset is None


def test_onset_offset_latency_no_offset():
    gt = np.array([0, 1, 1, 1, 1])
    pr = np.array([0, 1, 1, 1, 1])
    onset, offset = onset_offset_latency(gt, pr, frame_ms=10.0)
    assert onset == pytest.approx(0.0)
    assert offset is None


def test_onset_offset_latency_bad_frame_ms():
    with pytest.raises(ValueError):
        onset_offset_latency(np.array([1]), np.array([1]), frame_ms=0.0)
