"""Scenario runners. Each module exposes a ``run`` function returning :class:`ScenarioResult`."""

from .base import (
    PipelineCallable,
    PipelineCallResult,
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
    "PipelineCallResult",
    "PipelineProvider",
    "RealPipelineProvider",
    "ScenarioResult",
    "SnrSweep",
    "SnrSweepEntry",
    "StubPipelineProvider",
    "mix_at_snr",
]
# Scenario modules are imported on demand; keep the registry here for discovery.
SCENARIOS = (
    "scenario_1",
    "scenario_2",
    "scenario_3",
    "scenario_4",
    "scenario_5",
    "scenario_6",
)
