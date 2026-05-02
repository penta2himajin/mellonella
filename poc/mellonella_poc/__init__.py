"""mellonella PoC — Phase 1 Python implementation."""

from .config import Config, GatingConfig
from .enrollment import EmbeddingPool
from .gating import EnvelopeState, GateState, apply_envelope, target_score, update_gate
from .pipeline import ProcessResult, expand_gate_decisions

__all__ = [
    "Config",
    "EmbeddingPool",
    "EnvelopeState",
    "GateState",
    "GatingConfig",
    "ProcessResult",
    "apply_envelope",
    "expand_gate_decisions",
    "target_score",
    "update_gate",
]
