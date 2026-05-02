"""Top-level evaluation orchestrator.

Phase 1 supports only Scenario 1 (solo target + noise). Items have to be
discovered locally (the dataset downloaders write under
``$MELLONELLA_DATA_DIR``); until that discovery code lands, the runner
emits well-formed empty CSVs / a summary so the harness wiring stays
exercised in CI.

A real evaluation can already be assembled programmatically by importing
:mod:`mellonella_bench.scenarios.scenario_1` directly and passing a list
of :class:`Scenario1Item` into ``run`` — this module just wraps that
behind the ``mellonella-bench`` CLI.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..scenarios.base import PipelineProvider, StubPipelineProvider

SCENARIOS = ("scenario_1", "scenario_3")


@dataclass
class RunnerConfig:
    """Settings parsed from the CLI."""

    output_dir: Path
    scenarios: tuple[str, ...] = SCENARIOS
    quick: bool = False
    use_real_pipeline: bool = False
    """When False, scenarios run with :class:`StubPipelineProvider`. The real
    provider depends on torch/speechbrain via ``mellonella_poc``."""


@dataclass
class RunSummary:
    """Aggregate emitted to ``summary.json``."""

    eval_id: str
    git_commit: str
    system_info: dict[str, str]
    scenarios: dict[str, dict] = field(default_factory=dict)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _system_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }


def _build_eval_id() -> str:
    return time.strftime("eval_%Y%m%d_%H%M%S")


def _resolve_provider(use_real: bool) -> PipelineProvider:
    """Pick a :class:`PipelineProvider` based on the CLI flag.

    The real provider is only imported when explicitly requested so that
    the lightweight CI path stays free of torch/speechbrain.
    """
    if not use_real:
        return StubPipelineProvider()
    from ..scenarios.pipeline_provider import RealPipelineProvider

    return RealPipelineProvider()


def run(config: RunnerConfig) -> RunSummary:
    """Execute the configured scenarios and emit ``summary.json`` + per-scenario CSVs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(
        eval_id=_build_eval_id(),
        git_commit=_git_commit(),
        system_info=_system_info(),
    )

    provider = _resolve_provider(config.use_real_pipeline)

    if "scenario_1" in config.scenarios:
        from ..scenarios.scenario_1 import run as run_scenario_1

        # Phase 1 PoC: items are not yet discovered automatically (the
        # dataset prep script runs locally only). The harness still emits
        # an empty CSV + zero-sample summary so the output directory layout
        # is exercised end-to-end.
        result = run_scenario_1(
            items=[],
            provider=provider,
            sample_rate=16_000,
            output_csv=config.output_dir / "scenario_1.csv",
        )
        summary.scenarios["scenario_1"] = {
            "n_samples": result.n_samples,
            "metrics": result.metrics,
        }

    if "scenario_3" in config.scenarios:
        from ..scenarios.scenario_3 import run as run_scenario_3

        result = run_scenario_3(
            items=[],
            provider=provider,
            sample_rate=16_000,
            output_csv=config.output_dir / "scenario_3.csv",
        )
        summary.scenarios["scenario_3"] = {
            "n_samples": result.n_samples,
            "metrics": result.metrics,
        }

    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "eval_id": summary.eval_id,
                "git_commit": summary.git_commit,
                "system_info": summary.system_info,
                "scenarios": summary.scenarios,
            },
            indent=2,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mellonella-bench")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory for CSVs and summary.json",
    )
    parser.add_argument(
        "--scenarios",
        type=lambda s: tuple(s.split(",")),
        default=SCENARIOS,
        help="comma-separated subset of scenarios to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use the minimal evaluation set (Phase 1 PoC default)",
    )
    parser.add_argument(
        "--real-pipeline",
        action="store_true",
        help="use the real mellonella-poc pipeline (requires `pip install -e poc[models]`)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunnerConfig(
        output_dir=args.output,
        scenarios=args.scenarios,
        quick=args.quick,
        use_real_pipeline=args.real_pipeline,
    )
    summary = run(config)
    print(json.dumps({"eval_id": summary.eval_id, "output": str(config.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
