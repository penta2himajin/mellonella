"""Pure-NumPy tests for the run-length helpers exposed by `pipeline`.

The full `process_offline` orchestration pulls in DFN3 / VAD / ECAPA, so it
lives behind the `models` marker. These tests only exercise the helpers
that bench needs to translate gate run-lengths into per-sample arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_poc.pipeline import ProcessResult, expand_gate_decisions


def test_expand_gate_decisions_simple():
    out = expand_gate_decisions([(0, True), (5, False), (8, True)], n_samples=10)
    assert out.dtype == bool
    np.testing.assert_array_equal(
        out,
        [True, True, True, True, True, False, False, False, True, True],
    )


def test_expand_gate_decisions_starts_off():
    out = expand_gate_decisions([(0, False), (3, True)], n_samples=6)
    np.testing.assert_array_equal(out, [False, False, False, True, True, True])


def test_expand_gate_decisions_empty_returns_zeros():
    out = expand_gate_decisions([], n_samples=4)
    assert out.shape == (4,)
    assert not out.any()


def test_expand_gate_decisions_zero_samples():
    out = expand_gate_decisions([(0, True)], n_samples=0)
    assert out.shape == (0,)


def test_expand_gate_decisions_requires_zero_start():
    with pytest.raises(ValueError):
        expand_gate_decisions([(2, True)], n_samples=4)


def test_expand_gate_decisions_rejects_negative():
    with pytest.raises(ValueError):
        expand_gate_decisions([(0, True)], n_samples=-1)


def test_process_result_defaults_are_isolated():
    a = ProcessResult(audio=np.zeros(4, dtype=np.float32))
    b = ProcessResult(audio=np.zeros(4, dtype=np.float32))
    a.gate_decisions.append((0, True))
    a.gate_per_frame = np.array([True, False])
    assert b.gate_decisions == []
    assert b.gate_per_frame.size == 0
