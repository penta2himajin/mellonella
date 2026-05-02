"""Scenario runners. Each module exposes a ``run`` function returning :class:`ScenarioResult`."""

from .base import (
    PipelineCallable,
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
    mix_at_snr,
)
from .pipeline_provider import RealPipelineProvider

__all__ = [
    "PipelineCallable",
    "PipelineProvider",
    "RealPipelineProvider",
    "ScenarioResult",
    "SnrSweep",
    "SnrSweepEntry",
    "StubPipelineProvider",
    "mix_at_snr",
]
# Scenario modules are imported on demand; keep the registry here for discovery.
SCENARIOS = ("scenario_1", "scenario_3")
