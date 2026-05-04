#!/usr/bin/env python3
"""End-to-end smoke test for the mellonella-poc CLI.

Generates two short synthetic recordings, drives them through ``enroll``
and ``process``, and verifies the outputs are well-formed. Exit code 0
means the pipeline can load every model and run the full path without
exceptions; exit code != 0 means a regression worth investigating.

Usage:
    scripts/smoke.py                    # writes outputs under $TMPDIR
    SMOKE_KEEP_OUTPUT=1 scripts/smoke.py /custom/dir

The script intentionally does NOT assert on perceptual quality (gate
decisions, SI-SDR, etc.); silero-vad can legitimately reject the
synthetic input. The goal is to catch import / shape / runtime breakages
across torch, speechbrain, silero-vad, and deepfilternet.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ENROLL_SR = 16_000
ENROLL_DURATION_SEC = 5.0
INPUT_DURATION_SEC = 3.0
OUTPUT_SR = 48_000


def _synth_voiced(duration_sec: float, sr: int, *, fundamental_hz: float = 180.0) -> np.ndarray:
    """Sum of low-order harmonics with light AM — rough speech-like waveform."""
    t = np.arange(int(sr * duration_sec)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for harmonic in range(1, 6):
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * fundamental_hz * harmonic * t)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    out = wave * am * 0.3
    return out.astype(np.float32)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _check_audio(path: Path, expected_sr: int, expected_min_duration: float) -> None:
    info = sf.info(str(path))
    if info.samplerate != expected_sr:
        raise SystemExit(f"{path}: expected {expected_sr} Hz, got {info.samplerate} Hz")
    duration = info.frames / info.samplerate
    if duration < expected_min_duration * 0.95:
        raise SystemExit(
            f"{path}: expected ~{expected_min_duration:.2f}s, got {duration:.2f}s "
            f"({info.frames} frames)"
        )
    print(f"  {path.name}: {duration:.2f}s @ {info.samplerate} Hz, {info.frames} frames")


def _check_enrollment(path: Path) -> None:
    payload = json.loads(path.read_text())
    if payload.get("version") != 1:
        raise SystemExit(f"{path}: unsupported version {payload.get('version')}")
    n_anchors = len(payload.get("anchors", []))
    if n_anchors == 0:
        raise SystemExit(f"{path}: no anchors present")
    f0_mu = payload.get("metadata", {}).get("f0_mu", 0.0)
    print(f"  {path.name}: {n_anchors} anchors, f0_mu={f0_mu:.1f} Hz")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    work_dir = Path(args[0]) if args else Path.cwd() / ".smoke"
    work_dir.mkdir(parents=True, exist_ok=True)

    enrollment_wav = work_dir / "enrollment.wav"
    input_wav = work_dir / "input.wav"
    enrollment_json = work_dir / "enrollment.json"
    output_wav = work_dir / "filtered.wav"

    print(f"[smoke] generating synthetic recordings under {work_dir}")
    sf.write(str(enrollment_wav), _synth_voiced(ENROLL_DURATION_SEC, ENROLL_SR), ENROLL_SR)
    rng = np.random.default_rng(0)
    mixture = _synth_voiced(INPUT_DURATION_SEC, ENROLL_SR) + 0.02 * rng.standard_normal(
        int(INPUT_DURATION_SEC * ENROLL_SR)
    ).astype(np.float32)
    sf.write(str(input_wav), mixture, ENROLL_SR)

    print("[smoke] enroll")
    _run(
        [
            "mellonella-poc",
            "enroll",
            "--input",
            str(enrollment_wav),
            "--output",
            str(enrollment_json),
        ]
    )
    _check_enrollment(enrollment_json)

    print("[smoke] process")
    _run(
        [
            "mellonella-poc",
            "process",
            "--enrollment",
            str(enrollment_json),
            "--input",
            str(input_wav),
            "--output",
            str(output_wav),
        ]
    )
    _check_audio(output_wav, expected_sr=OUTPUT_SR, expected_min_duration=INPUT_DURATION_SEC)

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
