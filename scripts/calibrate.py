#!/usr/bin/env python3
"""θ_pass / θ_learn calibration sweep (legacy + AS-Norm).

Runs the real ``mellonella_poc`` pipeline across a small grid of
speakers, noise types and SNRs. For each (enrollment, test, noise, SNR)
combination we run the pipeline ONCE and capture the per-frame target
score series; threshold sweeps are then done post-hoc in NumPy, which
keeps the wall-clock cost down to a single ECAPA + DFN3 pass per cell.

Two calibration modes share this driver (D-010 Phase 3):

* **Legacy** (default) — the ``α·cs + β·f0`` integrated score is
  swept over :data:`THETA_GRID` (0.20-0.55, 0.025 step). Outputs to
  ``docs/benchmarks/calibration_{results.csv,summary.json}``.
* **AS-Norm** (``--use-as-norm --cohort PATH``) — pipeline is run
  with the cohort-normalised z-score, swept over
  :data:`THETA_GRID_AS_NORM` (0.5-3.0, 0.25 step). Outputs to
  ``docs/benchmarks/calibration_as_norm_{results.csv,summary.json}``.

Recommendation policy (per ``docs/gating.md`` D-004, the FP-tolerant policy):
the docs explicitly accept FP > FN for the single-target case, so we
pick the **smallest θ whose mean FPR across all (speaker pair, noise,
SNR ≥ 5dB) cells stays at or below the budget**. This maximises
target-pass rate (TPR) under the FP budget rather than the other way
around. A minimum TPR floor is applied as a sanity check.

The AS-Norm sweep uses a slightly looser FPR budget
(:data:`MAX_MEAN_FPR_AS_NORM` = 0.10 vs 0.05) because PR #23's
post-Phase-4 baseline showed wider per-language FPR spread under
AS-Norm — see ``docs/decisions.md`` D-010 Phase 3 for the reasoning.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from mellonella_poc.config import AudioConfig, Config, GatingConfig
from mellonella_poc.gating import GateState
from mellonella_poc.pipeline import (
    PipelineComponents,
    enroll_from_recording,
    process_offline,
)
from scipy.signal import lfilter, resample_poly

SAMPLE_RATE = 16_000
SPEAKERS: tuple[str, ...] = ("libri1", "libri2", "libri3")
NOISE_TYPES: tuple[str, ...] = ("white", "pink")
SNRS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
THETA_GRID: tuple[float, ...] = tuple(round(0.20 + 0.025 * i, 3) for i in range(15))
# AS-Norm sweep operates on a z-score scale (typical -3..3); D-010 Phase 3.
# 0.5-3.0 step 0.25 = 11 points, covers conservative (3.0) → permissive (0.5).
THETA_GRID_AS_NORM: tuple[float, ...] = tuple(
    round(0.5 + 0.25 * i, 3) for i in range(11)
)
SEED = 0
MIN_REPRESENTATIVE_SNR_DB = 5.0
MAX_MEAN_FPR = 0.05
MIN_TPR_FLOOR = 0.50
# AS-Norm has empirically wider per-language FPR spread (PR #23 baseline:
# zh-CN/ko ≈ 0.31, others ≤ 0.03) so the grand-mean budget is loosened to
# ~0.10 to keep any θ qualifying. The TPR floor stays at 0.50.
MAX_MEAN_FPR_AS_NORM = 0.10
MIN_TPR_FLOOR_AS_NORM = 0.50

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "docs" / "benchmarks" / "calibration_results.csv"
SUMMARY_JSON = REPO_ROOT / "docs" / "benchmarks" / "calibration_summary.json"
# Separate paths for AS-Norm so the legacy and z-score sweeps don't overwrite
# each other during back-to-back local runs.
RESULTS_CSV_AS_NORM = (
    REPO_ROOT / "docs" / "benchmarks" / "calibration_as_norm_results.csv"
)
SUMMARY_JSON_AS_NORM = (
    REPO_ROOT / "docs" / "benchmarks" / "calibration_as_norm_summary.json"
)


def _to_target_sr(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _load_speaker(name: str) -> np.ndarray:
    import librosa

    audio, sr = sf.read(librosa.example(name), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return _to_target_sr(np.asarray(audio, dtype=np.float32), int(sr), SAMPLE_RATE)


def _load_speakers_from_librosa() -> tuple[dict[str, np.ndarray], str]:
    """Default speaker source: librosa libri1/2/3 (English LibriSpeech)."""
    return {name: _load_speaker(name) for name in SPEAKERS}, "en"


def _load_speakers_from_manifest(
    manifest_path: Path,
    language: str,
) -> tuple[dict[str, np.ndarray], str]:
    """CommonVoice manifest source: per-speaker concatenated buffers."""
    from mellonella_bench.datasets.commonvoice import load_speakers_from_manifest

    raw = load_speakers_from_manifest(manifest_path, SAMPLE_RATE)
    speakers: dict[str, np.ndarray] = {
        name: np.asarray(audio, dtype=np.float32) for name, audio in raw.items()
    }
    if not speakers:
        raise ValueError(
            f"manifest {manifest_path} yielded zero usable speakers "
            f"(after the min-duration filter)"
        )
    return speakers, language


def _make_noise(kind: str, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize ``n_samples`` of either white or pink noise (deterministic)."""
    white = rng.standard_normal(n_samples).astype(np.float32)
    if kind == "white":
        return white
    if kind == "pink":
        # Voss-McCartney style 1/f filter via Paul Kellet's approximate IIR.
        b = np.array([0.049922, -0.095993, 0.050612, -0.004408], dtype=np.float64)
        a = np.array([1.0, -2.494956, 2.017265, -0.522189], dtype=np.float64)
        pink = lfilter(b, a, white).astype(np.float32)
        # Renormalize to unit RMS so SNR mixing math stays consistent.
        rms = float(np.sqrt(np.mean(pink.astype(np.float64) ** 2))) or 1.0
        return (pink / rms).astype(np.float32)
    raise ValueError(f"unknown noise kind: {kind!r}")


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(speech.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if speech_power == 0.0 or noise_power == 0.0:
        raise ValueError("zero-energy speech or noise; cannot mix at SNR")
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return (speech + scale * noise[: speech.size]).astype(np.float32)


def _simulate_gate(
    score_per_frame: np.ndarray,
    vad_dt_ms: float,
    theta: float,
    hangover_ms: float,
    *,
    use_as_norm: bool = False,
) -> np.ndarray:
    """Replay :class:`GateState` post-hoc with a different threshold.

    When ``use_as_norm`` is True, ``theta`` is interpreted as
    ``theta_pass_as_norm`` (z-score scale) and the GateState routes its
    comparison through :attr:`GatingConfig.theta_pass_as_norm`. Otherwise
    the legacy ``α·cs + β·f0`` threshold is used.
    """
    if use_as_norm:
        cfg = GatingConfig(
            use_as_norm=True,
            theta_pass_as_norm=theta,
            # theta_learn_as_norm only gates auto-learn admission, which
            # is disabled in calibration runs anyway. Keep it strictly
            # greater than theta to satisfy the validator.
            theta_learn_as_norm=max(theta + 0.1, 2.5),
            hangover_ms=hangover_ms,
        )
    else:
        cfg = GatingConfig(
            theta_pass=theta,
            theta_learn=max(theta + 0.01, 0.80),
            hangover_ms=hangover_ms,
        )
    state = GateState(config=cfg)
    return np.fromiter(
        (state.update(float(s), vad_dt_ms) for s in score_per_frame),
        dtype=bool,
        count=score_per_frame.size,
    )


def _build_pools(
    speakers: dict[str, np.ndarray],
    config: Config,
    components: PipelineComponents,
) -> dict[str, object]:
    """Build one EmbeddingPool per speaker from the first half of each recording."""
    pools: dict[str, object] = {}
    for name, audio in speakers.items():
        half = audio.size // 2
        pools[name] = enroll_from_recording(
            audio[:half], SAMPLE_RATE, config, components
        )
    return pools


def _test_segments(speakers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: audio[audio.size // 2 :] for name, audio in speakers.items()}


def measure_cells(
    speakers: dict[str, np.ndarray] | None = None,
    language: str = "en",
    *,
    use_as_norm: bool = False,
    cohort_path: Path | None = None,
) -> list[dict[str, float | str]]:
    """Run the pipeline once per cell and return the per-(cell, θ) rows.

    ``speakers`` lets callers inject a custom speaker pool (e.g.
    CommonVoice manifest output); when ``None`` we fall back to the
    librosa libri1/2/3 default. ``language`` is recorded verbatim into
    every output row for downstream analysis.

    When ``use_as_norm=True`` the pipeline is configured with the AS-Norm
    cohort from ``cohort_path`` so ``score_per_frame`` is in z-score
    scale, and the threshold sweep uses :data:`THETA_GRID_AS_NORM`. The
    output rows tag the score scale via the ``mode`` field (``legacy`` or
    ``as_norm``) so downstream aggregations can route on it.
    """
    if speakers is None:
        speakers, language = _load_speakers_from_librosa()
    if use_as_norm and cohort_path is None:
        raise ValueError("use_as_norm=True requires cohort_path")
    speaker_names = list(speakers)
    print(
        f"[calibrate] using {len(speaker_names)} speakers "
        f"(language={language!r}, mode={'as_norm' if use_as_norm else 'legacy'}): "
        f"{', '.join(speaker_names)}"
    )
    for name, a in speakers.items():
        print(f"  {name}: {a.size / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz")

    audio_cfg = AudioConfig()
    # Use a calibration config with auto-learn DISABLED — we want the
    # cosine-similarity score to depend only on the original anchor pool.
    cal_config = Config(
        audio=audio_cfg,
        gating=GatingConfig(
            theta_pass=0.50,  # legacy threshold; sweep happens post-hoc
            theta_learn=0.80,
            enable_auto_learn=False,
            use_as_norm=use_as_norm,
            as_norm_cohort_path=str(cohort_path) if cohort_path is not None else None,
        ),
    )
    components = PipelineComponents.build_default(cal_config)
    pools = _build_pools(speakers, cal_config, components)
    test_segs = _test_segments(speakers)
    rng = np.random.default_rng(SEED)

    theta_grid = THETA_GRID_AS_NORM if use_as_norm else THETA_GRID
    mode_tag = "as_norm" if use_as_norm else "legacy"

    rows: list[dict[str, float | str]] = []
    cells: list[tuple[str, str, str, float]] = list(
        itertools.product(speaker_names, speaker_names, NOISE_TYPES, SNRS_DB)
    )
    n_cells = len(cells)
    print(
        f"[calibrate] running {n_cells} cells × {len(theta_grid)} thresholds "
        f"(mode={mode_tag})..."
    )
    t_start = time.perf_counter()

    for idx, (enroll_name, test_name, noise_kind, snr_db) in enumerate(cells, 1):
        pool = pools[enroll_name]
        test_audio = test_segs[test_name]
        noise = _make_noise(noise_kind, test_audio.size, rng)
        mix = _mix_at_snr(test_audio, noise, snr_db)
        result = process_offline(mix, SAMPLE_RATE, pool, cal_config, components)  # type: ignore[arg-type]

        scores = result.score_per_frame
        kind = "tpr" if enroll_name == test_name else "fpr"
        for theta in theta_grid:
            gate = _simulate_gate(
                scores,
                audio_cfg.vad_frame_ms,
                theta,
                cal_config.gating.hangover_ms,
                use_as_norm=use_as_norm,
            )
            rate = float(gate.mean()) if gate.size else 0.0
            rows.append(
                {
                    "language": language,
                    "enroll": enroll_name,
                    "test": test_name,
                    "noise": noise_kind,
                    "snr_db": snr_db,
                    "theta_pass": theta,
                    "kind": kind,
                    "rate": round(rate, 4),
                    "mode": mode_tag,
                }
            )

        if idx % 6 == 0 or idx == n_cells:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / idx * (n_cells - idx)
            print(
                f"  cell {idx}/{n_cells}  elapsed={elapsed:5.1f}s  eta={eta:5.1f}s  "
                f"({enroll_name}→{test_name} {noise_kind}@{snr_db:+.0f}dB)"
            )
    return rows


def write_results_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "language",
        "enroll",
        "test",
        "noise",
        "snr_db",
        "theta_pass",
        "kind",
        "rate",
        "mode",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def aggregate(rows: list[dict[str, float | str]]) -> dict[float, dict[str, float]]:
    """Per-θ aggregation: ``{theta: {tpr_median, fpr_median, ...}}``."""
    by_theta: dict[float, dict[str, list[float]]] = {}
    for r in rows:
        if float(r["snr_db"]) < MIN_REPRESENTATIVE_SNR_DB:
            continue
        theta = float(r["theta_pass"])
        bucket = by_theta.setdefault(theta, {"tpr": [], "fpr": []})
        bucket[str(r["kind"])].append(float(r["rate"]))

    summary: dict[float, dict[str, float]] = {}
    for theta, bucket in sorted(by_theta.items()):
        summary[theta] = {
            "tpr_median": float(np.median(bucket["tpr"])) if bucket["tpr"] else 0.0,
            "tpr_mean": float(np.mean(bucket["tpr"])) if bucket["tpr"] else 0.0,
            "fpr_median": float(np.median(bucket["fpr"])) if bucket["fpr"] else 0.0,
            "fpr_mean": float(np.mean(bucket["fpr"])) if bucket["fpr"] else 0.0,
        }
    return summary


def recommend_theta(
    per_theta: dict[float, dict[str, float]],
    *,
    max_mean_fpr: float = MAX_MEAN_FPR,
    min_tpr_floor: float = MIN_TPR_FLOOR,
) -> float:
    """Pick the smallest θ that keeps mean FPR within ``max_mean_fpr``.

    FP-tolerant by design: the docs explicitly accept FP > FN for the
    single-target case, so we maximise TPR (= pick the loosest gate)
    under an FPR budget rather than the other way around. Among the
    qualifying θ values we additionally require ``TPR_median`` to clear
    ``min_tpr_floor`` as a sanity check; if that fails we relax to
    "smallest θ meeting just the FPR budget", and if even that fails we
    return the strictest θ in the grid (caller logs a warning).

    The ``max_mean_fpr`` / ``min_tpr_floor`` knobs are surfaced as
    arguments so the legacy and AS-Norm sweeps can use different budgets
    (PR #24 / D-010 Phase 3 — AS-Norm has wider per-language FPR spread).
    """
    qualifying = [
        t
        for t, m in per_theta.items()
        if m["fpr_mean"] <= max_mean_fpr and m["tpr_median"] >= min_tpr_floor
    ]
    if qualifying:
        return min(qualifying)
    # Fallback: smallest θ among those meeting the FPR budget alone.
    fpr_only = [t for t, m in per_theta.items() if m["fpr_mean"] <= max_mean_fpr]
    if fpr_only:
        return min(fpr_only)
    # Nothing qualifies — every θ exceeds the FPR budget. Pick the largest.
    return max(per_theta)


def _help_relative(path: Path) -> str:
    """Render ``path`` relative to the repo root for --help text, with a
    fallback to absolute when a test monkey-patches the constant to a tmp
    location outside the repo tree."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="θ calibration sweep")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help=(
            "output CSV path. Defaults to "
            f"{_help_relative(RESULTS_CSV)} for legacy sweeps, "
            f"{_help_relative(RESULTS_CSV_AS_NORM)} when --use-as-norm is set."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "output summary path. Defaults to "
            f"{_help_relative(SUMMARY_JSON)} for legacy sweeps, "
            f"{_help_relative(SUMMARY_JSON_AS_NORM)} when --use-as-norm is set."
        ),
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="skip the pipeline sweep; re-aggregate the existing results CSV",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "CommonVoice subset manifest.csv to drive speaker selection. "
            "When omitted, uses the librosa libri1/2/3 default."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language tag recorded in every output row (default: en)",
    )
    parser.add_argument(
        "--use-as-norm",
        action="store_true",
        help=(
            "Calibrate the AS-Norm gating path (D-010 Phase 3). Requires "
            "--cohort. Switches the threshold sweep to z-score scale and "
            "writes to the *_as_norm output paths."
        ),
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help=(
            "Path to the impostor cohort .npz produced by "
            "scripts/build_impostor_cohort.py. Required when --use-as-norm."
        ),
    )
    parser.add_argument(
        "--max-mean-fpr",
        type=float,
        default=None,
        help=(
            f"FP budget for theta recommendation. Defaults to {MAX_MEAN_FPR} "
            f"for legacy sweeps, {MAX_MEAN_FPR_AS_NORM} when --use-as-norm is set."
        ),
    )
    parser.add_argument(
        "--min-tpr-floor",
        type=float,
        default=None,
        help=(
            f"TPR floor for theta recommendation. Defaults to {MIN_TPR_FLOOR} "
            f"(both modes — AS-Norm doesn't require relaxation here)."
        ),
    )
    args = parser.parse_args(argv)

    if args.use_as_norm and args.cohort is None:
        parser.error("--use-as-norm requires --cohort PATH")

    # Resolve mode-aware defaults.
    if args.results_csv is None:
        args.results_csv = RESULTS_CSV_AS_NORM if args.use_as_norm else RESULTS_CSV
    if args.summary_json is None:
        args.summary_json = SUMMARY_JSON_AS_NORM if args.use_as_norm else SUMMARY_JSON
    if args.max_mean_fpr is None:
        args.max_mean_fpr = MAX_MEAN_FPR_AS_NORM if args.use_as_norm else MAX_MEAN_FPR
    if args.min_tpr_floor is None:
        args.min_tpr_floor = (
            MIN_TPR_FLOOR_AS_NORM if args.use_as_norm else MIN_TPR_FLOOR
        )

    if args.from_csv:
        if not args.results_csv.exists():
            print(
                f"[calibrate] {args.results_csv} not found; cannot --from-csv",
                file=sys.stderr,
            )
            return 1
        with args.results_csv.open() as fh:
            rows = [
                {
                    "language": r.get("language", "en"),
                    "enroll": r["enroll"],
                    "test": r["test"],
                    "noise": r["noise"],
                    "snr_db": float(r["snr_db"]),
                    "theta_pass": float(r["theta_pass"]),
                    "kind": r["kind"],
                    "rate": float(r["rate"]),
                    # mode is optional for backwards compatibility with
                    # pre-Phase-3 result CSVs that don't carry the column.
                    "mode": r.get("mode", "as_norm" if args.use_as_norm else "legacy"),
                }
                for r in csv.DictReader(fh)
            ]
        print(f"[calibrate] re-aggregating {len(rows)} rows from {args.results_csv}")
    else:
        if args.manifest is not None:
            speakers, language = _load_speakers_from_manifest(
                args.manifest, args.language
            )
        else:
            speakers, language = _load_speakers_from_librosa()
        rows = measure_cells(
            speakers=speakers,
            language=language,
            use_as_norm=args.use_as_norm,
            cohort_path=args.cohort,
        )
        write_results_csv(rows, args.results_csv)
        print(f"\n[calibrate] wrote {len(rows)} rows to {args.results_csv}")

    per_theta = aggregate(rows)
    recommended = recommend_theta(
        per_theta,
        max_mean_fpr=args.max_mean_fpr,
        min_tpr_floor=args.min_tpr_floor,
    )
    print(f"\n[calibrate] per-θ aggregate (snr ≥ {MIN_REPRESENTATIVE_SNR_DB:.0f} dB):")
    print(
        f"  {'theta':>6} {'tpr_med':>9} {'fpr_med':>9} {'tpr_mean':>10} {'fpr_mean':>10}"
    )
    for theta, m in per_theta.items():
        marker = " ← recommended" if theta == recommended else ""
        print(
            f"  {theta:>6.3f} {m['tpr_median']:>9.3f} {m['fpr_median']:>9.3f}  "
            f"{m['tpr_mean']:>9.3f}  {m['fpr_mean']:>9.3f}{marker}"
        )

    summary = {
        "schema_version": 2,
        "mode": "as_norm" if args.use_as_norm else "legacy",
        "config": {
            "speakers": list(SPEAKERS),
            "noise_types": list(NOISE_TYPES),
            "snrs_db": list(SNRS_DB),
            "theta_grid": list(THETA_GRID_AS_NORM if args.use_as_norm else THETA_GRID),
            "min_representative_snr_db": MIN_REPRESENTATIVE_SNR_DB,
            "max_mean_fpr": args.max_mean_fpr,
            "min_tpr_floor": args.min_tpr_floor,
            "seed": SEED,
        },
        "recommended_theta_pass": recommended,
        "per_theta": {
            f"{theta:.3f}": {k: round(v, 4) for k, v in m.items()}
            for theta, m in per_theta.items()
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[calibrate] recommended θ_pass = {recommended:.3f}")
    print(f"[calibrate] wrote summary to {args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
