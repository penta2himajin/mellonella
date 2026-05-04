#!/usr/bin/env python3
"""CI accuracy regression check (case A + simultaneous-speech extension).

Drives the real ``mellonella_poc`` pipeline through deterministic mini
scenarios and compares the resulting metrics against a committed
baseline. Fails (exit 1) the moment any metric drops more than the
configured tolerance below baseline; improvements are silently accepted.

Inputs
------
* enrollment + target test = ``librosa.example("libri1")`` halves
* other speaker             = ``librosa.example("libri2")``
* noise                     = synthetic Gaussian white noise (seed-fixed)
* SNR sweep                 = 5, 10, 15 dB                  (case A / case C)
* simultaneous mix sweep    = target+other at +9, 0 dB      (case B)

Per measurement key we record the metrics that apply:

    snr_<S>_db (target speaker + noise at <S> dB)
        tpr             — gate-on rate during target speech
        fpr             — gate-on rate when the input is the other speaker
        si_sdr_db       — SI-SDR vs the clean target reference
        other_rms_db    — overall RMS (dBFS) of the gated output when
                          the input is the other speaker; lower = better
                          attenuation. Catches "gate stays open"
                          regressions even when FPR happens to land at 0.

    sim4_<R>db (target + other speech at target-to-other ratio <R>)
        tpr             — gate-on rate during target voiced frames; the
                          FP-tolerant policy in docs/gating.md D-001 says
                          this should stay high even at 0 dB mix
        si_sdr_db       — SI-SDR of the gated output vs the clean target

Tolerances (worse-side only; improvements are ignored):

    TPR           :  current >= baseline * 0.95           (relative -5%)
    FPR           :  current <= baseline * 1.05           (relative +5%)
                      — when baseline is 0, allow current up to 0.05 absolute
    SI-SDR        :  current >= baseline - 1.0 dB         (absolute, dB)
    other_rms_db  :  current <= baseline + 3.0 dB         (absolute, dB up = worse)

Usage
-----
    python scripts/ci_accuracy.py                    # compare to committed baseline
    python scripts/ci_accuracy.py --update-baseline  # write the baseline file

The first form is what CI runs. The second is for the maintainer to
refresh the baseline after an intentional, validated change.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from mellonella_poc.config import Config
from mellonella_poc.pipeline import (
    PipelineComponents,
    enroll_from_recording,
    process_offline,
)
from scipy.signal import resample_poly

from mellonella_bench.metrics.ns_quality import si_sdr

SAMPLE_RATE = 16_000
SNRS_DB: tuple[float, ...] = (5.0, 10.0, 15.0)
SIM4_RATIOS_DB: tuple[float, ...] = (9.0, 0.0)
SEED = 0
TPR_REL_TOL = 0.05
FPR_REL_TOL = 0.05
FPR_ABS_FLOOR = 0.05  # used when baseline FPR is exactly 0
SI_SDR_ABS_TOL_DB = 1.0
OTHER_RMS_ABS_TOL_DB = 3.0  # output_rms_db growing by > 3 dB = mute weakening

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "docs" / "benchmarks" / "ci_baseline.json"


def _to_target_sr(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _load_mono(path: Path | str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(speech.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if speech_power == 0.0 or noise_power == 0.0:
        raise ValueError("zero-energy speech or noise; cannot mix at SNR")
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return (speech.astype(np.float32) + scale * noise.astype(np.float32)).astype(np.float32)


def _mix_target_other(target: np.ndarray, other: np.ndarray, ratio_db: float) -> np.ndarray:
    """Mix target + other at the requested target-to-other power ratio (dB).

    Lengths are aligned to ``target.size`` (truncate-or-pad ``other``).
    """
    if other.size >= target.size:
        other_aligned = other[: target.size]
    else:
        other_aligned = np.concatenate(
            [other, np.zeros(target.size - other.size, dtype=other.dtype)]
        )
    target_power = float(np.mean(target.astype(np.float64) ** 2))
    other_power = float(np.mean(other_aligned.astype(np.float64) ** 2))
    if target_power == 0.0 or other_power == 0.0:
        raise ValueError("zero-energy target or other; cannot mix at finite ratio")
    target_other_power = target_power / (10.0 ** (ratio_db / 10.0))
    scale = float(np.sqrt(target_other_power / other_power))
    return (target.astype(np.float32) + scale * other_aligned.astype(np.float32)).astype(np.float32)


def _gate_on_rate(gate_per_frame: np.ndarray) -> float:
    if gate_per_frame.size == 0:
        return 0.0
    return float(gate_per_frame.mean())


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return -120.0
    return 20.0 * float(np.log10(rms + 1e-12))


def measure() -> dict[str, dict[str, float]]:
    """Run the pipeline at every SNR + simultaneous mix and aggregate metrics."""
    import librosa  # lazy: only needed when the script actually runs

    target_path = librosa.example("libri1")
    other_path = librosa.example("libri2")

    target_audio, sr_t = _load_mono(target_path)
    other_audio, sr_o = _load_mono(other_path)
    target_audio = _to_target_sr(target_audio, sr_t, SAMPLE_RATE)
    other_audio = _to_target_sr(other_audio, sr_o, SAMPLE_RATE)

    half = target_audio.size // 2
    enrollment_audio = target_audio[:half]
    target_test = target_audio[half:]

    # Project defaults (post-calibration); see scripts/calibrate.py.
    config = Config()
    components = PipelineComponents.build_default(config)
    pool = enroll_from_recording(enrollment_audio, SAMPLE_RATE, config, components)

    rng = np.random.default_rng(SEED)
    metrics: dict[str, dict[str, float]] = {}

    # --- Case A / C: target+noise / other+noise at each SNR ------------------
    for snr in SNRS_DB:
        target_noise = rng.standard_normal(target_test.size).astype(np.float32)
        other_noise = rng.standard_normal(other_audio.size).astype(np.float32)

        target_mix = _mix_at_snr(target_test, target_noise, snr)
        target_result = process_offline(target_mix, SAMPLE_RATE, pool, config, components)
        tpr = _gate_on_rate(target_result.gate_per_frame)
        out_at_16k = _to_target_sr(target_result.audio, config.audio.output_sr, SAMPLE_RATE)
        n = min(target_test.size, out_at_16k.size)
        sisdr = si_sdr(target_test[:n], out_at_16k[:n])

        other_mix = _mix_at_snr(other_audio, other_noise, snr)
        other_result = process_offline(other_mix, SAMPLE_RATE, pool, config, components)
        fpr = _gate_on_rate(other_result.gate_per_frame)
        other_rms_db_value = _rms_db(other_result.audio)

        metrics[f"snr_{int(snr)}_db"] = {
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "si_sdr_db": round(sisdr, 2),
            "other_rms_db": round(other_rms_db_value, 2),
        }

    # --- Case B: simultaneous target + other at each ratio -------------------
    for ratio_db in SIM4_RATIOS_DB:
        sim_mix = _mix_target_other(target_test, other_audio, ratio_db)
        sim_result = process_offline(sim_mix, SAMPLE_RATE, pool, config, components)
        sim_tpr = _gate_on_rate(sim_result.gate_per_frame)
        sim_out_at_16k = _to_target_sr(sim_result.audio, config.audio.output_sr, SAMPLE_RATE)
        n_sim = min(target_test.size, sim_out_at_16k.size)
        sim_sisdr = si_sdr(target_test[:n_sim], sim_out_at_16k[:n_sim])

        metrics[f"sim4_{int(ratio_db)}db"] = {
            "tpr": round(sim_tpr, 4),
            "si_sdr_db": round(sim_sisdr, 2),
        }

    return metrics


def _check_metric_pair(
    failures: list[str],
    key: str,
    cur: dict[str, float],
    base: dict[str, float],
) -> None:
    """Compare ``cur`` and ``base`` for one measurement key. Skip metrics that
    are absent in either side; fail per-metric on regression."""
    if "tpr" in cur and "tpr" in base and cur["tpr"] < base["tpr"] * (1.0 - TPR_REL_TOL):
        failures.append(
            f"{key} TPR regressed: {cur['tpr']:.3f} < " f"{base['tpr']:.3f} * (1 - {TPR_REL_TOL})"
        )

    if "fpr" in cur and "fpr" in base:
        if base["fpr"] == 0.0:
            if cur["fpr"] > FPR_ABS_FLOOR:
                failures.append(
                    f"{key} FPR regressed: {cur['fpr']:.3f} > {FPR_ABS_FLOOR} "
                    "(baseline=0; absolute floor)"
                )
        elif cur["fpr"] > base["fpr"] * (1.0 + FPR_REL_TOL):
            failures.append(
                f"{key} FPR regressed: {cur['fpr']:.3f} > "
                f"{base['fpr']:.3f} * (1 + {FPR_REL_TOL})"
            )

    if (
        "si_sdr_db" in cur
        and "si_sdr_db" in base
        and cur["si_sdr_db"] < base["si_sdr_db"] - SI_SDR_ABS_TOL_DB
    ):
        failures.append(
            f"{key} SI-SDR regressed: {cur['si_sdr_db']:.2f} dB < "
            f"{base['si_sdr_db']:.2f} dB - {SI_SDR_ABS_TOL_DB} dB"
        )

    if (
        "other_rms_db" in cur
        and "other_rms_db" in base
        and cur["other_rms_db"] > base["other_rms_db"] + OTHER_RMS_ABS_TOL_DB
    ):
        failures.append(
            f"{key} other_rms_db regressed: {cur['other_rms_db']:.2f} dB > "
            f"{base['other_rms_db']:.2f} dB + {OTHER_RMS_ABS_TOL_DB} dB"
        )


def _check_against_baseline(
    current: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> list[str]:
    failures: list[str] = []
    for key, cur in current.items():
        if key not in baseline:
            failures.append(f"{key}: missing from baseline")
            continue
        _check_metric_pair(failures, key, cur, baseline[key])
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CI accuracy regression check")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite docs/benchmarks/ci_baseline.json with the current measurement",
    )
    args = parser.parse_args(argv)

    print(
        f"[ci-accuracy] running mini scenario_1 at {SAMPLE_RATE} Hz, SNRs={SNRS_DB} dB; "
        f"sim4 ratios={SIM4_RATIOS_DB} dB"
    )
    metrics = measure()
    print("[ci-accuracy] measurements:")
    print(json.dumps(metrics, indent=2))

    if args.update_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        defaults = Config()
        payload = {
            "schema_version": 2,
            "config": {
                "sample_rate": SAMPLE_RATE,
                "snrs_db": list(SNRS_DB),
                "sim4_ratios_db": list(SIM4_RATIOS_DB),
                "seed": SEED,
                "theta_pass": defaults.gating.theta_pass,
                "theta_learn": defaults.gating.theta_learn,
                "alpha": defaults.gating.alpha,
                "beta": defaults.gating.beta,
            },
            "tolerances": {
                "tpr_relative": TPR_REL_TOL,
                "fpr_relative": FPR_REL_TOL,
                "fpr_absolute_floor_when_zero": FPR_ABS_FLOOR,
                "si_sdr_absolute_db": SI_SDR_ABS_TOL_DB,
                "other_rms_absolute_db": OTHER_RMS_ABS_TOL_DB,
            },
            "metrics": metrics,
        }
        BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[ci-accuracy] wrote baseline to {BASELINE_PATH}")
        return 0

    if not BASELINE_PATH.exists():
        print(
            f"[ci-accuracy] no baseline at {BASELINE_PATH}; "
            "re-run with --update-baseline to seed it.",
            file=sys.stderr,
        )
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    failures = _check_against_baseline(metrics, baseline.get("metrics", {}))
    if failures:
        print("\n[ci-accuracy] REGRESSION DETECTED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\n[ci-accuracy] OK — all metrics within tolerance vs baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
