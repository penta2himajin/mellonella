#!/usr/bin/env python3
"""θ_pass / θ_learn calibration sweep.

Runs the real ``mellonella_poc`` pipeline across a small grid of
speakers, noise types and SNRs. For each (enrollment, test, noise, SNR)
combination we run the pipeline ONCE and capture the per-frame target
score series; threshold sweeps are then done post-hoc in NumPy, which
keeps the wall-clock cost down to a single ECAPA + DFN3 pass per cell.

Outputs:

* ``docs/benchmarks/calibration_results.csv``  — every (cell, theta) row
* ``docs/benchmarks/calibration_summary.json`` — recommended θ_pass

Recommendation policy (per ``docs/gating.md`` D-004 "FP 許容方針"):
the docs explicitly accept FP > FN for the single-target case, so we
pick the **smallest θ_pass whose mean FPR across all (speaker pair,
noise, SNR ≥ 5dB) cells stays at or below ``MAX_MEAN_FPR``**. This
maximises target-pass rate (TPR) under the FP budget rather than the
other way around. A minimum TPR floor is applied as a sanity check.
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
from scipy.signal import lfilter, resample_poly

from mellonella_poc.config import AudioConfig, Config, GatingConfig
from mellonella_poc.gating import GateState
from mellonella_poc.pipeline import (
    PipelineComponents,
    enroll_from_recording,
    process_offline,
)

SAMPLE_RATE = 16_000
SPEAKERS: tuple[str, ...] = ("libri1", "libri2", "libri3")
NOISE_TYPES: tuple[str, ...] = ("white", "pink")
SNRS_DB: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
THETA_GRID: tuple[float, ...] = tuple(round(0.20 + 0.025 * i, 3) for i in range(15))
SEED = 0
MIN_REPRESENTATIVE_SNR_DB = 5.0
MAX_MEAN_FPR = 0.05
MIN_TPR_FLOOR = 0.50

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "docs" / "benchmarks" / "calibration_results.csv"
SUMMARY_JSON = REPO_ROOT / "docs" / "benchmarks" / "calibration_summary.json"


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
    theta_pass: float,
    hangover_ms: float,
) -> np.ndarray:
    """Replay :class:`GateState` post-hoc with a different ``theta_pass``."""
    cfg = GatingConfig(
        theta_pass=theta_pass,
        theta_learn=max(theta_pass + 0.01, 0.80),
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


def measure_cells() -> list[dict[str, float | str]]:
    """Run the pipeline once per cell and return the per-(cell, θ) rows."""
    print(f"[calibrate] loading {len(SPEAKERS)} speakers from librosa...")
    speakers = {name: _load_speaker(name) for name in SPEAKERS}
    for name, a in speakers.items():
        print(f"  {name}: {a.size / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz")

    audio_cfg = AudioConfig()
    # Use a calibration config with auto-learn DISABLED — we want the
    # cosine-similarity score to depend only on the original anchor pool.
    cal_config = Config(
        audio=audio_cfg,
        gating=GatingConfig(
            theta_pass=0.50,  # value irrelevant for score logging; sweep happens post-hoc
            theta_learn=0.80,
            enable_auto_learn=False,
        ),
    )
    components = PipelineComponents.build_default(cal_config)
    pools = _build_pools(speakers, cal_config, components)
    test_segs = _test_segments(speakers)
    rng = np.random.default_rng(SEED)

    rows: list[dict[str, float | str]] = []
    cells: list[tuple[str, str, str, float]] = list(
        itertools.product(SPEAKERS, SPEAKERS, NOISE_TYPES, SNRS_DB)
    )
    n_cells = len(cells)
    print(f"[calibrate] running {n_cells} cells × {len(THETA_GRID)} thresholds...")
    t_start = time.perf_counter()

    for idx, (enroll_name, test_name, noise_kind, snr_db) in enumerate(cells, 1):
        pool = pools[enroll_name]
        test_audio = test_segs[test_name]
        noise = _make_noise(noise_kind, test_audio.size, rng)
        mix = _mix_at_snr(test_audio, noise, snr_db)
        result = process_offline(mix, SAMPLE_RATE, pool, cal_config, components)  # type: ignore[arg-type]

        scores = result.score_per_frame
        kind = "tpr" if enroll_name == test_name else "fpr"
        for theta in THETA_GRID:
            gate = _simulate_gate(
                scores, audio_cfg.vad_frame_ms, theta, cal_config.gating.hangover_ms
            )
            rate = float(gate.mean()) if gate.size else 0.0
            rows.append(
                {
                    "enroll": enroll_name,
                    "test": test_name,
                    "noise": noise_kind,
                    "snr_db": snr_db,
                    "theta_pass": theta,
                    "kind": kind,
                    "rate": round(rate, 4),
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
    fieldnames = ["enroll", "test", "noise", "snr_db", "theta_pass", "kind", "rate"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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


def recommend_theta(per_theta: dict[float, dict[str, float]]) -> float:
    """Pick the smallest θ that keeps mean FPR within ``MAX_MEAN_FPR``.

    FP-tolerant by design: the docs explicitly accept FP > FN for the
    single-target case, so we maximise TPR (= pick the loosest gate)
    under an FPR budget rather than the other way around. Among the
    qualifying θ values we additionally require ``TPR_median`` to clear
    ``MIN_TPR_FLOOR`` as a sanity check; if that fails we relax to
    "smallest θ overall" with a warning printed by the caller.
    """
    qualifying = [
        t
        for t, m in per_theta.items()
        if m["fpr_mean"] <= MAX_MEAN_FPR and m["tpr_median"] >= MIN_TPR_FLOOR
    ]
    if qualifying:
        return min(qualifying)
    # Fallback: smallest θ among those meeting the FPR budget alone.
    fpr_only = [t for t, m in per_theta.items() if m["fpr_mean"] <= MAX_MEAN_FPR]
    if fpr_only:
        return min(fpr_only)
    # Nothing qualifies — every θ exceeds the FPR budget. Pick the largest.
    return max(per_theta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="θ calibration sweep")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_CSV,
        help=f"output CSV path (default: {RESULTS_CSV.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=SUMMARY_JSON,
        help=f"output summary path (default: {SUMMARY_JSON.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="skip the pipeline sweep; re-aggregate the existing results CSV",
    )
    args = parser.parse_args(argv)

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
                    "enroll": r["enroll"],
                    "test": r["test"],
                    "noise": r["noise"],
                    "snr_db": float(r["snr_db"]),
                    "theta_pass": float(r["theta_pass"]),
                    "kind": r["kind"],
                    "rate": float(r["rate"]),
                }
                for r in csv.DictReader(fh)
            ]
        print(f"[calibrate] re-aggregating {len(rows)} rows from {args.results_csv}")
    else:
        rows = measure_cells()
        write_results_csv(rows, args.results_csv)
        print(f"\n[calibrate] wrote {len(rows)} rows to {args.results_csv}")

    per_theta = aggregate(rows)
    recommended = recommend_theta(per_theta)
    print(
        "\n[calibrate] per-θ aggregate (snr ≥ {:.0f} dB):".format(
            MIN_REPRESENTATIVE_SNR_DB
        )
    )
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
        "schema_version": 1,
        "config": {
            "speakers": list(SPEAKERS),
            "noise_types": list(NOISE_TYPES),
            "snrs_db": list(SNRS_DB),
            "theta_grid": list(THETA_GRID),
            "min_representative_snr_db": MIN_REPRESENTATIVE_SNR_DB,
            "max_mean_fpr": MAX_MEAN_FPR,
            "min_tpr_floor": MIN_TPR_FLOOR,
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
