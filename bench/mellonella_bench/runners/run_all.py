"""Top-level evaluation orchestrator.

Phase 1 supports only Scenario 1 (solo target + noise) with a stub pipeline
fallback so the harness can exercise its CSV / JSON wiring without the
heavy ML dependencies installed. Other scenarios will be wired in here as
they land.
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

SCENARIOS = ("scenario_1",)


@dataclass
class RunnerConfig:
    """Settings parsed from the CLI."""

    output_dir: Path
    scenarios: tuple[str, ...] = SCENARIOS
    quick: bool = False
    use_real_pipeline: bool = True
    """When False, scenarios run with a deterministic identity-pipeline stub.
    Used by tests and dry-runs."""


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


def _identity_pipeline(mixture, sr):  # type: ignore[no-untyped-def]
    """Stub pipeline: returns the mixture unchanged with a 'pass everything' gate.

    Useful for harness tests and as a baseline number ('do nothing' floor).
    """
    import numpy as np

    return mixture, np.ones(mixture.size, dtype=bool)


def _resolve_pipeline(use_real: bool):  # type: ignore[no-untyped-def]
    """Return the callable used by scenarios to filter a mixture.

    The real pipeline path is left as a TODO until the per-item enrollment
    plumbing lands. For now we always return the identity stub so the
    harness produces well-formed CSVs/JSON even on a fresh checkout.
    """
    del use_real
    return _identity_pipeline


def run(config: RunnerConfig) -> RunSummary:
    """Execute the configured scenarios and emit ``summary.json`` + per-scenario CSVs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(
        eval_id=_build_eval_id(),
        git_commit=_git_commit(),
        system_info=_system_info(),
    )

    pipeline = _resolve_pipeline(config.use_real_pipeline)

    if "scenario_1" in config.scenarios:
        from ..scenarios.scenario_1 import run as run_scenario_1

        # Phase 1 PoC: no items wired up yet (datasets only fetched locally).
        # The runner still emits an empty CSV + zero-sample summary so the
        # output directory layout is exercised end-to-end.
        result = run_scenario_1(
            items=[],
            pipeline_callable=pipeline,
            sample_rate=16_000,
            output_csv=config.output_dir / "scenario_1.csv",
        )
        summary.scenarios["scenario_1"] = {
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
        "--stub-pipeline",
        action="store_true",
        help="force the identity-pipeline stub (used by tests and dry runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunnerConfig(
        output_dir=args.output,
        scenarios=args.scenarios,
        quick=args.quick,
        use_real_pipeline=not args.stub_pipeline,
    )
    summary = run(config)
    print(json.dumps({"eval_id": summary.eval_id, "output": str(config.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
