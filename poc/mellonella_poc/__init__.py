"""mellonella PoC — Phase 1 Python implementation."""

from .config import Config, GatingConfig
from .enrollment import EmbeddingPool
from .gating import (
    EnvelopeState,
    GateState,
    apply_envelope,
    should_admit_auto_learn,
    target_score,
    update_gate,
)
from .pipeline import AutoLearnEvent, ProcessResult, expand_gate_decisions

__all__ = [
    "AutoLearnEvent",
    "Config",
    "EmbeddingPool",
    "EnvelopeState",
    "GateState",
    "GatingConfig",
    "ProcessResult",
    "apply_envelope",
    "expand_gate_decisions",
    "should_admit_auto_learn",
    "target_score",
    "update_gate",
]
