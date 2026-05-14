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
    python scripts/ci_accuracy.py                      # python engine vs baseline
    python scripts/ci_accuracy.py --engine rust        # rust core (live) vs baseline
    python scripts/ci_accuracy.py --update-baseline    # refresh the baseline file

``--engine`` selects the implementation under test: ``python`` is the
``mellonella_poc`` reference, ``rust`` shells out to the ``mellonella``
CLI — the live Rust core that actually ships (#121). Each engine + mode
tracks its own baseline file (see ``_baseline_path``). The rust engine
needs a release-built CLI and the ONNX env vars
(``MELLONELLA_ECAPA_ONNX`` / ``MELLONELLA_VAD_ONNX`` / ``ORT_DYLIB_PATH``).

CI runs the compare form; ``--update-baseline`` is for the maintainer to
refresh a baseline after an intentional, validated change.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mellonella_bench.metrics.ns_quality import si_sdr

# ``mellonella_poc`` (torch / speechbrain) is imported lazily inside the
# Python runner so ``--engine rust`` can run without the heavy ML deps.

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


def _baseline_path(use_as_norm: bool, engine: str) -> Path:
    """Pick which baseline file to read/write for the active engine + mode.

    Four distributions, four files under ``docs/benchmarks/``:

    * ``ci_baseline.json``               — Python, legacy ``α·cs + β·f0``
    * ``ci_baseline_as_norm.json``       — Python, AS-Norm z-score
    * ``ci_baseline_rust.json``          — Rust core, raw-cs + EMA path
    * ``ci_baseline_rust_as_norm.json``  — Rust core, AS-Norm z-score

    The Python ``α·cs + β·f0`` path and the AS-Norm z-score path produce
    metric values on different scales, so they cannot share a file. The
    Rust core is a separate implementation again (raw cosine vs the
    anchor centroid, plus EMA smoothing — see #117) and is what the
    live product actually runs, so it tracks its own baseline rather
    than being force-fit to the Python reference's numbers (#121).
    """
    name = "ci_baseline"
    if engine == "rust":
        name += "_rust"
    if use_as_norm:
        name += "_as_norm"
    return REPO_ROOT / "docs" / "benchmarks" / f"{name}.json"


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


# A runner takes a 16 kHz mixture and returns
# ``(gate_per_frame, output_audio, output_sr)`` — the per-VAD-frame gate
# decision, the gated output waveform, and that waveform's sample rate.
# Both engines expose the same shape so :func:`measure` is engine-agnostic.
PipelineRun = "tuple[np.ndarray, np.ndarray, int]"


def _python_runner(enrollment_audio: np.ndarray, as_norm_cohort: Path | None):
    """Build a runner backed by ``mellonella_poc.pipeline.process_offline``."""
    from mellonella_poc.config import Config, GatingConfig
    from mellonella_poc.pipeline import (
        PipelineComponents,
        enroll_from_recording,
        process_offline,
    )

    if as_norm_cohort is not None:
        config = Config(
            gating=GatingConfig(
                use_as_norm=True,
                as_norm_cohort_path=str(as_norm_cohort),
            ),
        )
    else:
        config = Config()
    components = PipelineComponents.build_default(config)
    pool = enroll_from_recording(enrollment_audio, SAMPLE_RATE, config, components)
    output_sr = int(config.audio.output_sr)

    def run(mixture: np.ndarray):
        result = process_offline(mixture, SAMPLE_RATE, pool, config, components)
        return (
            np.asarray(result.gate_per_frame, dtype=bool),
            np.asarray(result.audio, dtype=np.float32),
            output_sr,
        )

    return run


def _rust_runner(enrollment_audio: np.ndarray, as_norm_cohort: Path | None):
    """Build a runner backed by the ``mellonella`` CLI (the live Rust core).

    Enrolls once into a temp JSON, then per mixture writes a 16-bit WAV,
    runs ``mellonella process … --gate-decisions`` and reads back the
    gated output WAV + the per-frame gate diagnostics. The CLI needs
    ``MELLONELLA_ECAPA_ONNX`` / ``MELLONELLA_VAD_ONNX`` / ``ORT_DYLIB_PATH``
    in the environment. The binary path defaults to the release build and
    can be overridden with ``MELLONELLA_RUST_BIN``.
    """
    if as_norm_cohort is not None:
        raise SystemExit(
            "[ci-accuracy] --engine rust does not support --as-norm-cohort yet "
            "(the CLI has no --cohort flag — tracked as a #121 follow-up)"
        )

    rust_bin = Path(
        os.environ.get(
            "MELLONELLA_RUST_BIN",
            REPO_ROOT / "rust" / "target" / "release" / "mellonella",
        )
    )
    if not rust_bin.exists():
        raise SystemExit(
            f"[ci-accuracy] Rust CLI missing at {rust_bin}; run "
            "`cargo build --release -p mellonella-cli` or set MELLONELLA_RUST_BIN"
        )

    workdir = Path(tempfile.mkdtemp(prefix="ci-accuracy-rust-"))
    enroll_wav = workdir / "enroll.wav"
    enroll_json = workdir / "enroll.json"
    sf.write(str(enroll_wav), enrollment_audio, SAMPLE_RATE, subtype="PCM_16")
    subprocess.run(
        [str(rust_bin), "enroll", str(enroll_wav), str(enroll_json)],
        check=True,
        env=os.environ.copy(),
    )

    counter = itertools.count()

    def run(mixture: np.ndarray):
        idx = next(counter)
        in_wav = workdir / f"mix_{idx}.wav"
        out_wav = workdir / f"out_{idx}.wav"
        diag_json = workdir / f"diag_{idx}.json"
        sf.write(str(in_wav), mixture, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(
            [
                str(rust_bin),
                "process",
                str(in_wav),
                str(enroll_json),
                str(out_wav),
                "--gate-decisions",
                str(diag_json),
            ],
            check=True,
            env=os.environ.copy(),
        )
        audio, sr = sf.read(str(out_wav), dtype="float32", always_2d=False)
        diag = json.loads(diag_json.read_text())
        return (
            np.asarray(diag["gate_per_frame"], dtype=bool),
            np.asarray(audio, dtype=np.float32),
            int(sr),
        )

    return run


def _make_runner(engine: str, enrollment_audio: np.ndarray, as_norm_cohort: Path | None):
    if engine == "python":
        return _python_runner(enrollment_audio, as_norm_cohort)
    if engine == "rust":
        return _rust_runner(enrollment_audio, as_norm_cohort)
    raise SystemExit(f"[ci-accuracy] unknown engine: {engine!r} (expected python|rust)")


def measure(
    as_norm_cohort: Path | None = None,
    engine: str = "python",
) -> dict[str, dict[str, float]]:
    """Run the pipeline at every SNR + simultaneous mix and aggregate metrics.

    ``engine`` selects which implementation drives the measurement:

    * ``python`` — ``mellonella_poc.pipeline.process_offline`` (the
      reference; ``α·cs + β·f0`` or AS-Norm scoring).
    * ``rust``   — the ``mellonella`` CLI, i.e. the live Rust core. This
      is what ships, so its numbers are the ones that matter for the
      gating-stability work (#117–#120). See #121.

    When ``as_norm_cohort`` points at an impostor cohort ``.npz`` (built
    by :mod:`scripts/build_impostor_cohort`), the pipeline switches to
    the AS-Norm gating path. Only supported for ``engine="python"`` so
    far.
    """
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

    run = _make_runner(engine, enrollment_audio, as_norm_cohort)

    rng = np.random.default_rng(SEED)
    metrics: dict[str, dict[str, float]] = {}

    # --- Case A / C: target+noise / other+noise at each SNR ------------------
    for snr in SNRS_DB:
        target_noise = rng.standard_normal(target_test.size).astype(np.float32)
        other_noise = rng.standard_normal(other_audio.size).astype(np.float32)

        target_mix = _mix_at_snr(target_test, target_noise, snr)
        target_gate, target_audio_out, target_out_sr = run(target_mix)
        tpr = _gate_on_rate(target_gate)
        out_at_16k = _to_target_sr(target_audio_out, target_out_sr, SAMPLE_RATE)
        n = min(target_test.size, out_at_16k.size)
        sisdr = si_sdr(target_test[:n], out_at_16k[:n])

        other_mix = _mix_at_snr(other_audio, other_noise, snr)
        other_gate, other_audio_out, _ = run(other_mix)
        fpr = _gate_on_rate(other_gate)
        other_rms_db_value = _rms_db(other_audio_out)

        metrics[f"snr_{int(snr)}_db"] = {
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "si_sdr_db": round(sisdr, 2),
            "other_rms_db": round(other_rms_db_value, 2),
        }

    # --- Case B: simultaneous target + other at each ratio -------------------
    for ratio_db in SIM4_RATIOS_DB:
        sim_mix = _mix_target_other(target_test, other_audio, ratio_db)
        sim_gate, sim_audio_out, sim_out_sr = run(sim_mix)
        sim_tpr = _gate_on_rate(sim_gate)
        sim_out_at_16k = _to_target_sr(sim_audio_out, sim_out_sr, SAMPLE_RATE)
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


def _config_payload(engine: str, use_as_norm: bool) -> dict[str, object]:
    """Metadata block stored in the baseline file. Engine-aware: the
    Python theta/alpha/beta defaults only exist for ``engine="python"``;
    the Rust core's gate defaults live in its own source (`GateConfig` /
    `PipelineConfig`) and are not duplicated here."""
    payload: dict[str, object] = {
        "engine": engine,
        "sample_rate": SAMPLE_RATE,
        "snrs_db": list(SNRS_DB),
        "sim4_ratios_db": list(SIM4_RATIOS_DB),
        "seed": SEED,
    }
    if engine == "python":
        from mellonella_poc.config import Config

        defaults = Config()
        payload["theta_pass"] = defaults.gating.theta_pass
        payload["theta_learn"] = defaults.gating.theta_learn
        payload["alpha"] = defaults.gating.alpha
        payload["beta"] = defaults.gating.beta
        if use_as_norm:
            payload["use_as_norm"] = True
            payload["theta_pass_as_norm"] = defaults.gating.theta_pass_as_norm
            payload["theta_learn_as_norm"] = defaults.gating.theta_learn_as_norm
    elif use_as_norm:
        payload["use_as_norm"] = True
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CI accuracy regression check")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "overwrite the baseline file for the active engine + mode with "
            "the current measurement (see --engine / --as-norm-cohort)"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("python", "rust"),
        default="python",
        help=(
            "which implementation to measure. 'python' = the "
            "mellonella_poc reference (default); 'rust' = the mellonella "
            "CLI, i.e. the live Rust core that actually ships (#121). The "
            "two track separate baseline files because they are distinct "
            "implementations."
        ),
    )
    parser.add_argument(
        "--as-norm-cohort",
        type=Path,
        default=None,
        help=(
            "path to an impostor cohort .npz "
            "(see scripts/build_impostor_cohort.py); enables AS-Norm in the "
            "real pipeline (D-010 Phase 6 Part 2 step 3). When set, the "
            "script reads / writes the AS-Norm baseline variant instead of "
            "the legacy one — the two metric distributions are "
            "incompatible (mixed cosine+pitch vs cohort z-score). "
            "Only supported with --engine python so far."
        ),
    )
    args = parser.parse_args(argv)

    use_as_norm = args.as_norm_cohort is not None
    baseline_path = _baseline_path(use_as_norm, args.engine)

    print(
        f"[ci-accuracy] engine={args.engine}; running mini scenario_1 at "
        f"{SAMPLE_RATE} Hz, SNRs={SNRS_DB} dB; sim4 ratios={SIM4_RATIOS_DB} dB"
        + (f"; AS-Norm cohort={args.as_norm_cohort}" if use_as_norm else "")
    )
    metrics = measure(as_norm_cohort=args.as_norm_cohort, engine=args.engine)
    print("[ci-accuracy] measurements:")
    print(json.dumps(metrics, indent=2))

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        config_payload = _config_payload(args.engine, use_as_norm)
        payload = {
            "schema_version": 2,
            "config": config_payload,
            "tolerances": {
                "tpr_relative": TPR_REL_TOL,
                "fpr_relative": FPR_REL_TOL,
                "fpr_absolute_floor_when_zero": FPR_ABS_FLOOR,
                "si_sdr_absolute_db": SI_SDR_ABS_TOL_DB,
                "other_rms_absolute_db": OTHER_RMS_ABS_TOL_DB,
            },
            "metrics": metrics,
        }
        baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[ci-accuracy] wrote baseline to {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(
            f"[ci-accuracy] no baseline at {baseline_path}; "
            "re-run with --update-baseline to seed it.",
            file=sys.stderr,
        )
        return 1

    baseline = json.loads(baseline_path.read_text())
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
