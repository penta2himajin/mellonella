"""Causal time-domain target speaker extraction (Stage C).

A small causal Conv-TasNet TSE network conditioned on a frozen 192-dim
ECAPA enrollment embedding via SpeakerBeam-style FiLM. See ``README.md``.
"""

from __future__ import annotations

from .config import TSEConfig
from .loss import neg_si_sdr_loss, si_sdr
from .model import CausalConvTasNetTSE, count_parameters

__all__ = [
    "TSEConfig",
    "CausalConvTasNetTSE",
    "count_parameters",
    "neg_si_sdr_loss",
    "si_sdr",
]
