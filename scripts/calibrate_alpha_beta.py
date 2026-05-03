#!/usr/bin/env python3
"""Joint α / θ_pass sweep for the F0-aux ablation (D-005 verification).

Mirrors ``scripts/calibrate.py`` but captures the per-frame
``cos_sim_max`` and ``f0_match`` series from each pipeline run, so we
can post-hoc reconstruct the integrated score for any α (and β = 1 - α)
and replay the gate at any θ_pass — no extra pipeline runs needed.

Outputs:

* ``docs/benchmarks/calibration_alpha_beta_results.csv`` — one row per
  (cell, α, θ, kind) with the gate-on rate
* ``docs/benchmarks/calibration_alpha_beta_summary.json`` — per-(α, θ)
  median TPR / mean FPR over SNR ≥ MIN_REPRESENTATIVE_SNR_DB cells, and
  the recommended (α, θ) pair under the same FP-tolerant policy as
  ``calibrate.py``

Recommendation policy: smallest θ_pass whose mean FPR (over SNR ≥ 5 dB)
stays at or below ``MAX_MEAN_FPR``, with TPR_median ≥ ``MIN_TPR_FLOOR``.
Among the qualifying rows, prefer the α that maximises TPR_median.
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
ALPHA_GRID: tuple[float, ...] = tuple(
    round(0.0 + 0.1 * i, 2) for i in range(11)
)  # 0.0..1.0
THETA_GRID: tuple[float, ...] = tuple(
    round(0.20 + 0.05 * i, 3) for i in range(8)
)  # 0.20..0.55
SEED = 0
MIN_REPRESENTATIVE_SNR_DB = 5.0
MAX_MEAN_FPR = 0.05
MIN_TPR_FLOOR = 0.50

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "docs" / "benchmarks" / "calibration_alpha_beta_results.csv"
SUMMARY_JSON = REPO_ROOT / "docs" / "benchmarks" / "calibration_alpha_beta_summary.json"


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
    white = rng.standard_normal(n_samples).astype(np.float32)
    if kind == "white":
        return white
    if kind == "pink":
        b = np.array([0.049922, -0.095993, 0.050612, -0.004408], dtype=np.float64)
        a = np.array([1.0, -2.494956, 2.017265, -0.522189], dtype=np.float64)
        pink = lfilter(b, a, white).astype(np.float32)
        rms = float(np.sqrt(np.mean(pink.astype(np.float64) ** 2))) or 1.0
        return (pink / rms).astype(np.float32)
    raise ValueError(f"unknown noise kind: {kind!r}")


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(speech.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if speech_power == 0.0 or noise_power == 0.0:
        raise ValueError("zero-energy speech or noise")
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return (speech + scale * noise[: speech.size]).astype(np.float32)


def _simulate_gate(
    score_per_frame: np.ndarray,
    vad_dt_ms: float,
    theta_pass: float,
    hangover_ms: float,
) -> np.ndarray:
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


def measure_cells() -> list[dict[str, float | str]]:
    """Run every cell once, sweep (α, θ) post-hoc, return one row per combination."""
    print(f"[alpha-beta] loading {len(SPEAKERS)} speakers from librosa...")
    speakers = {name: _load_speaker(name) for name in SPEAKERS}
    for name, a in speakers.items():
        print(f"  {name}: {a.size / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz")

    audio_cfg = AudioConfig()
    cal_config = Config(
        audio=audio_cfg,
        gating=GatingConfig(
            theta_pass=0.50,  # value irrelevant for score logging
            theta_learn=0.80,
            enable_auto_learn=False,
        ),
    )
    components = PipelineComponents.build_default(cal_config)
    pools: dict[str, object] = {}
    for name, audio in speakers.items():
        pools[name] = enroll_from_recording(
            audio[: audio.size // 2], SAMPLE_RATE, cal_config, components
        )
    test_segs = {name: audio[audio.size // 2 :] for name, audio in speakers.items()}
    rng = np.random.default_rng(SEED)

    rows: list[dict[str, float | str]] = []
    cells: list[tuple[str, str, str, float]] = list(
        itertools.product(SPEAKERS, SPEAKERS, NOISE_TYPES, SNRS_DB)
    )
    n_cells = len(cells)
    print(
        f"[alpha-beta] running {n_cells} cells × {len(ALPHA_GRID)} α × {len(THETA_GRID)} θ..."
    )
    t_start = time.perf_counter()

    for idx, (enroll_name, test_name, noise_kind, snr_db) in enumerate(cells, 1):
        pool = pools[enroll_name]
        test_audio = test_segs[test_name]
        noise = _make_noise(noise_kind, test_audio.size, rng)
        mix = _mix_at_snr(test_audio, noise, snr_db)
        result = process_offline(mix, SAMPLE_RATE, pool, cal_config, components)  # type: ignore[arg-type]

        cs = result.cos_sim_max_per_frame
        fm = result.f0_match_per_frame
        kind = "tpr" if enroll_name == test_name else "fpr"

        for alpha in ALPHA_GRID:
            beta = 1.0 - alpha
            scores = (alpha * cs + beta * fm).astype(np.float32)
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
                        "alpha": alpha,
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
    fieldnames = [
        "enroll",
        "test",
        "noise",
        "snr_db",
        "alpha",
        "theta_pass",
        "kind",
        "rate",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def aggregate(
    rows: list[dict[str, float | str]],
) -> dict[tuple[float, float], dict[str, float]]:
    """Per-(α, θ) aggregation restricted to SNR ≥ MIN_REPRESENTATIVE_SNR_DB."""
    by_pair: dict[tuple[float, float], dict[str, list[float]]] = {}
    for r in rows:
        if float(r["snr_db"]) < MIN_REPRESENTATIVE_SNR_DB:
            continue
        key = (float(r["alpha"]), float(r["theta_pass"]))
        bucket = by_pair.setdefault(key, {"tpr": [], "fpr": []})
        bucket[str(r["kind"])].append(float(r["rate"]))

    summary: dict[tuple[float, float], dict[str, float]] = {}
    for key in sorted(by_pair):
        bucket = by_pair[key]
        summary[key] = {
            "tpr_median": float(np.median(bucket["tpr"])) if bucket["tpr"] else 0.0,
            "tpr_mean": float(np.mean(bucket["tpr"])) if bucket["tpr"] else 0.0,
            "fpr_median": float(np.median(bucket["fpr"])) if bucket["fpr"] else 0.0,
            "fpr_mean": float(np.mean(bucket["fpr"])) if bucket["fpr"] else 0.0,
        }
    return summary


def recommend(
    summary: dict[tuple[float, float], dict[str, float]],
) -> tuple[float, float] | None:
    """Largest TPR_median that respects FPR + TPR floor; tie-break: smallest θ then largest α."""
    qualifying = [
        (a, t, m["tpr_median"])
        for (a, t), m in summary.items()
        if m["fpr_mean"] <= MAX_MEAN_FPR and m["tpr_median"] >= MIN_TPR_FLOOR
    ]
    if not qualifying:
        return None
    best_tpr = max(q[2] for q in qualifying)
    near_best = [
        (a, t)
        for (a, t, tpr) in qualifying
        if tpr >= best_tpr - 0.01  # tolerate 1 % wobble
    ]
    # tie-break: smallest θ (loosest gate that still meets bar) then largest α
    near_best.sort(key=lambda at: (at[1], -at[0]))
    return near_best[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="α / θ_pass joint sweep")
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON)
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="skip the pipeline sweep; re-aggregate the existing results CSV",
    )
    args = parser.parse_args(argv)

    if args.from_csv:
        if not args.results_csv.exists():
            print(f"[alpha-beta] {args.results_csv} not found", file=sys.stderr)
            return 1
        with args.results_csv.open() as fh:
            rows = [
                {
                    "enroll": r["enroll"],
                    "test": r["test"],
                    "noise": r["noise"],
                    "snr_db": float(r["snr_db"]),
                    "alpha": float(r["alpha"]),
                    "theta_pass": float(r["theta_pass"]),
                    "kind": r["kind"],
                    "rate": float(r["rate"]),
                }
                for r in csv.DictReader(fh)
            ]
        print(f"[alpha-beta] re-aggregating {len(rows)} rows from {args.results_csv}")
    else:
        rows = measure_cells()
        write_results_csv(rows, args.results_csv)
        print(f"\n[alpha-beta] wrote {len(rows)} rows to {args.results_csv}")

    summary = aggregate(rows)
    recommended = recommend(summary)
    print(
        f"\n[alpha-beta] per-(α, θ) aggregate (snr ≥ {MIN_REPRESENTATIVE_SNR_DB:.0f} dB):"
    )
    print(
        f"  {'α':>5} {'θ':>6} {'tpr_med':>9} {'fpr_med':>9} {'tpr_mean':>10} {'fpr_mean':>10}"
    )
    for (alpha, theta), m in sorted(summary.items()):
        marker = ""
        if recommended is not None and (alpha, theta) == recommended:
            marker = " ← recommended"
        print(
            f"  {alpha:>5.2f} {theta:>6.3f} {m['tpr_median']:>9.3f} {m['fpr_median']:>9.3f}  "
            f"{m['tpr_mean']:>9.3f}  {m['fpr_mean']:>9.3f}{marker}"
        )

    summary_payload = {
        "schema_version": 1,
        "config": {
            "speakers": list(SPEAKERS),
            "noise_types": list(NOISE_TYPES),
            "snrs_db": list(SNRS_DB),
            "alpha_grid": list(ALPHA_GRID),
            "theta_grid": list(THETA_GRID),
            "min_representative_snr_db": MIN_REPRESENTATIVE_SNR_DB,
            "max_mean_fpr": MAX_MEAN_FPR,
            "min_tpr_floor": MIN_TPR_FLOOR,
            "seed": SEED,
        },
        "recommended": (
            {
                "alpha": recommended[0],
                "beta": round(1.0 - recommended[0], 4),
                "theta_pass": recommended[1],
            }
            if recommended is not None
            else None
        ),
        "per_pair": {
            f"a={alpha:.2f},t={theta:.3f}": {k: round(v, 4) for k, v in m.items()}
            for (alpha, theta), m in sorted(summary.items())
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary_payload, indent=2) + "\n")
    if recommended is not None:
        print(
            f"\n[alpha-beta] recommended α={recommended[0]:.2f}, "
            f"β={1.0 - recommended[0]:.2f}, θ_pass={recommended[1]:.3f}"
        )
    else:
        print("\n[alpha-beta] no (α, θ) pair satisfied both FP budget and TPR floor")
    print(f"[alpha-beta] wrote summary to {args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
