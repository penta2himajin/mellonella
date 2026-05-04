"""Light-weight tests for ``scripts/calibrate.py`` D-010 Phase 3 additions.

The full pipeline sweep is too heavy for CI (loads ECAPA + DFN3 + runs
~108 cells), so we only cover the new pure-numpy helpers in this file:
the threshold-grid switch, the post-hoc gate replay under AS-Norm mode,
and the recommendation logic with overridable budgets. The script is
loaded via ``importlib`` to avoid relying on it being on ``$PYTHONPATH``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibrate.py"


def _import_script():
    spec = importlib.util.spec_from_file_location("calibrate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("calibrate", module)
    spec.loader.exec_module(module)
    return module


def test_legacy_theta_grid_lives_in_cosine_range():
    """Legacy sweep stays inside [0, 1] cosine-similarity scale."""
    mod = _import_script()
    assert 0.0 <= min(mod.THETA_GRID) <= max(mod.THETA_GRID) <= 1.0


def test_as_norm_theta_grid_lives_in_z_score_range():
    """AS-Norm sweep covers a sensible z-score range (literature 0.5-3.0)."""
    mod = _import_script()
    assert min(mod.THETA_GRID_AS_NORM) >= 0.5
    assert max(mod.THETA_GRID_AS_NORM) <= 3.0
    # Coarse sanity: the grid is monotonic and has at least 5 entries.
    assert len(mod.THETA_GRID_AS_NORM) >= 5
    assert list(mod.THETA_GRID_AS_NORM) == sorted(mod.THETA_GRID_AS_NORM)


def test_simulate_gate_legacy_threshold_branch():
    """Legacy branch compares against ``theta_pass`` (cosine-scale)."""
    mod = _import_script()
    scores = np.array([0.1, 0.4, 0.7, 0.2], dtype=np.float32)
    gate = mod._simulate_gate(scores, vad_dt_ms=10.0, theta=0.30, hangover_ms=0.0)
    # 0.10 < 0.30 → mute, 0.40/0.70 ≥ 0.30 → pass, 0.20 < 0.30 → mute (no hangover)
    assert gate.tolist() == [False, True, True, False]


def test_simulate_gate_as_norm_threshold_branch():
    """AS-Norm branch compares against ``theta_pass_as_norm`` (z-score scale).

    Same numeric scores, but the threshold semantics changes — confirm
    the GateState routes through the AS-Norm path.
    """
    mod = _import_script()
    scores = np.array([0.5, 1.5, 2.5, 0.0], dtype=np.float32)
    gate = mod._simulate_gate(scores, vad_dt_ms=10.0, theta=2.0, hangover_ms=0.0, use_as_norm=True)
    # 0.5 < 2.0 → mute, 1.5 < 2.0 → mute, 2.5 ≥ 2.0 → pass, 0.0 < 2.0 → mute
    assert gate.tolist() == [False, False, True, False]


def test_recommend_theta_picks_smallest_qualifier():
    """Smallest θ whose fpr_mean ≤ budget AND tpr_median ≥ floor wins."""
    mod = _import_script()
    per_theta = {
        0.20: {"tpr_median": 0.95, "fpr_mean": 0.20, "tpr_mean": 0.95, "fpr_mean_": 0.20},
        0.30: {"tpr_median": 0.85, "fpr_mean": 0.04, "tpr_mean": 0.85, "fpr_mean_": 0.04},
        0.40: {"tpr_median": 0.70, "fpr_mean": 0.02, "tpr_mean": 0.70, "fpr_mean_": 0.02},
    }
    # 0.20 has too-high FPR, 0.30 and 0.40 both qualify → pick 0.30 (loosest).
    chosen = mod.recommend_theta(per_theta, max_mean_fpr=0.05, min_tpr_floor=0.50)
    assert chosen == 0.30


def test_recommend_theta_falls_back_to_fpr_only_when_tpr_floor_unreachable():
    """If no θ clears the TPR floor, fall back to smallest FPR-qualifier."""
    mod = _import_script()
    per_theta = {
        0.20: {"tpr_median": 0.20, "fpr_mean": 0.04, "tpr_mean": 0.20, "fpr_mean_": 0.04},
        0.30: {"tpr_median": 0.10, "fpr_mean": 0.02, "tpr_mean": 0.10, "fpr_mean_": 0.02},
        0.40: {"tpr_median": 0.05, "fpr_mean": 0.50, "tpr_mean": 0.05, "fpr_mean_": 0.50},
    }
    chosen = mod.recommend_theta(per_theta, max_mean_fpr=0.05, min_tpr_floor=0.50)
    # 0.20 and 0.30 both meet the FPR budget; 0.20 is smaller.
    assert chosen == 0.20


def test_recommend_theta_falls_back_to_max_when_nothing_qualifies():
    """If every θ exceeds the FPR budget, return the strictest one."""
    mod = _import_script()
    per_theta = {
        0.20: {"tpr_median": 0.95, "fpr_mean": 0.50, "tpr_mean": 0.95, "fpr_mean_": 0.50},
        0.40: {"tpr_median": 0.70, "fpr_mean": 0.20, "tpr_mean": 0.70, "fpr_mean_": 0.20},
    }
    chosen = mod.recommend_theta(per_theta, max_mean_fpr=0.05, min_tpr_floor=0.50)
    assert chosen == 0.40


def test_recommend_theta_honours_as_norm_budget_override():
    """AS-Norm needs a looser FP budget; verify the override actually changes the pick."""
    mod = _import_script()
    per_theta = {
        1.0: {"tpr_median": 0.85, "fpr_mean": 0.30, "tpr_mean": 0.85, "fpr_mean_": 0.30},
        1.5: {"tpr_median": 0.80, "fpr_mean": 0.08, "tpr_mean": 0.80, "fpr_mean_": 0.08},
        2.0: {"tpr_median": 0.70, "fpr_mean": 0.03, "tpr_mean": 0.70, "fpr_mean_": 0.03},
    }
    # With legacy budget (0.05) → 2.0 wins.
    assert mod.recommend_theta(per_theta, max_mean_fpr=0.05, min_tpr_floor=0.50) == 2.0
    # With AS-Norm budget (0.10) → 1.5 wins (looser, higher TPR).
    assert mod.recommend_theta(per_theta, max_mean_fpr=0.10, min_tpr_floor=0.50) == 1.5


def test_main_rejects_use_as_norm_without_cohort(tmp_path):
    """Programmatic CLI invocation should error out cleanly."""
    mod = _import_script()
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--use-as-norm",
                "--results-csv",
                str(tmp_path / "x.csv"),
                "--summary-json",
                str(tmp_path / "x.json"),
            ]
        )


def test_summary_paths_default_to_as_norm_when_flag_set(tmp_path, monkeypatch):
    """Default --results-csv / --summary-json paths route to *_as_norm files
    when --use-as-norm is set. Use --from-csv to short-circuit the heavy
    pipeline run; we just need the path-resolution branch.
    """
    mod = _import_script()
    # Pre-create a minimal AS-Norm results CSV at the default location so
    # --from-csv doesn't trip on FileNotFoundError. We monkeypatch the
    # module-level constant to a tmp path so the test is hermetic.
    fake_results = tmp_path / "calibration_as_norm_results.csv"
    fake_results.write_text(
        "language,enroll,test,noise,snr_db,theta_pass,kind,rate,mode\n"
        "en,A,A,white,10.0,1.5,tpr,0.8,as_norm\n"
        "en,A,B,white,10.0,1.5,fpr,0.04,as_norm\n"
    )
    monkeypatch.setattr(mod, "RESULTS_CSV_AS_NORM", fake_results)
    monkeypatch.setattr(mod, "SUMMARY_JSON_AS_NORM", tmp_path / "calibration_as_norm_summary.json")

    rc = mod.main(["--use-as-norm", "--cohort", str(tmp_path / "cohort.npz"), "--from-csv"])
    assert rc == 0
    assert (tmp_path / "calibration_as_norm_summary.json").exists()
