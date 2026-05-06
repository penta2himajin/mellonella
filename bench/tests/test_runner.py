"""End-to-end tests for the run_all orchestrator using the stub pipeline."""

from __future__ import annotations

import json

from mellonella_bench.runners.run_all import RunnerConfig, build_parser, run


def test_run_with_stub_pipeline_emits_summary(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_1",),
        quick=True,
        use_real_pipeline=False,
    )
    summary = run(config)

    summary_path = tmp_path / "eval" / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["scenarios"]["scenario_1"]["n_samples"] == 0
    assert payload["eval_id"].startswith("eval_")
    assert summary.git_commit  # any string, even "unknown"


def test_scenario_1_csv_is_present_even_when_empty(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_1",),
        use_real_pipeline=False,
    )
    run(config)
    assert (tmp_path / "eval" / "scenario_1.csv").exists()


def test_scenario_2_csv_is_present_even_when_empty(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_2",),
        use_real_pipeline=False,
    )
    summary = run(config)
    assert (tmp_path / "eval" / "scenario_2.csv").exists()
    assert summary.scenarios["scenario_2"]["n_samples"] == 0


def test_scenario_3_csv_is_present_even_when_empty(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_3",),
        use_real_pipeline=False,
    )
    summary = run(config)
    assert (tmp_path / "eval" / "scenario_3.csv").exists()
    assert summary.scenarios["scenario_3"]["n_samples"] == 0


def test_scenario_4_csv_is_present_even_when_empty(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_4",),
        use_real_pipeline=False,
    )
    summary = run(config)
    assert (tmp_path / "eval" / "scenario_4.csv").exists()
    assert summary.scenarios["scenario_4"]["n_samples"] == 0


def test_scenario_5_csv_is_present_even_when_empty(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_5",),
        use_real_pipeline=False,
    )
    summary = run(config)
    assert (tmp_path / "eval" / "scenario_5.csv").exists()
    assert summary.scenarios["scenario_5"]["n_samples"] == 0


def test_scenario_6_csv_is_present_even_when_empty(tmp_path):
    config = RunnerConfig(
        output_dir=tmp_path / "eval",
        scenarios=("scenario_6",),
        use_real_pipeline=False,
    )
    summary = run(config)
    assert (tmp_path / "eval" / "scenario_6.csv").exists()
    assert summary.scenarios["scenario_6"]["n_samples"] == 0


def test_parser_defaults():
    args = build_parser().parse_args(["--output", "/tmp/out"])
    assert args.scenarios == (
        "scenario_1",
        "scenario_2",
        "scenario_3",
        "scenario_4",
        "scenario_5",
        "scenario_6",
    )
    assert not args.quick
    assert not args.real_pipeline
    assert args.as_norm_cohort is None


def test_parser_scenarios_split():
    args = build_parser().parse_args(["--output", "/tmp/out", "--scenarios", "scenario_1"])
    assert args.scenarios == ("scenario_1",)


def test_parser_as_norm_cohort_path_passes_through():
    from pathlib import Path

    args = build_parser().parse_args(
        ["--output", "/tmp/out", "--as-norm-cohort", "/tmp/cohort.npz"]
    )
    assert args.as_norm_cohort == Path("/tmp/cohort.npz")
