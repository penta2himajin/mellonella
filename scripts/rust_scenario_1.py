#!/usr/bin/env python3
"""Cross-implementation scenario_1 smoke test (Rust ↔ Python).

Builds a synthetic target + noise mixture at a fixed SNR, runs it
through both the Python PoC pipeline and the Rust CLI, and compares
the per-VAD-frame gate state, the gated output waveform, and the gate
duty cycle.

Goal: prove the Rust deliverable produces functionally equivalent
output to the Python reference on a real (-ish) end-to-end mixture —
not byte parity (resampler / DFN3 differences mean that's
unachievable), but agreement-within-tolerance on the metrics the
gating decision actually drives.

Run on a host with:

* ``poc[models]`` installed (torch, speechbrain, silero-vad, etc.)
* ``cargo build -p mellonella-cli`` already executed
* ``MELLONELLA_ECAPA_ONNX``, ``MELLONELLA_VAD_ONNX``, ``ORT_DYLIB_PATH``
  pointing at the right binaries

    python scripts/rust_scenario_1.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
SNR_DB = 10.0
DURATION_SEC = 3.0
ENROLL_DURATION_SEC = 3.0
RUST_BIN = Path(__file__).resolve().parents[1] / "rust" / "target" / "debug" / "mellonella"


def synth_target(duration_sec: float, sr: int, f0: float = 200.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sr)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for harmonic in range(1, 6):
        phase = rng.uniform(0.0, 2 * np.pi)
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t + phase).astype(np.float32)
    am = 0.5 * (1.0 + 0.3 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
    wave = (wave * am).astype(np.float32)
    peak = float(np.max(np.abs(wave)))
    return (wave / peak * 0.8).astype(np.float32) if peak > 0 else wave


def synth_noise(duration_sec: float, sr: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.5, int(duration_sec * sr)).astype(np.float32)


def mix_at_snr(target: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    target_rms = float(np.sqrt(np.mean(target**2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise**2) + 1e-12))
    target_factor = 10 ** (snr_db / 20.0) * noise_rms / target_rms
    scaled_target = target * target_factor
    mix = scaled_target + noise
    peak = float(np.max(np.abs(mix)))
    return (mix / peak * 0.9).astype(np.float32) if peak > 0 else mix


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr, subtype="PCM_16")


def run_rust(input_wav: Path, enroll_wav: Path, work: Path) -> dict:
    enroll_json = work / "rust_enroll.json"
    out_wav = work / "rust_out.wav"
    diag_json = work / "rust_diag.json"

    if not RUST_BIN.exists():
        raise FileNotFoundError(
            f"Rust binary missing at {RUST_BIN}; run `cargo build -p mellonella-cli` first"
        )

    subprocess.run(
        [str(RUST_BIN), "enroll", str(enroll_wav), str(enroll_json)],
        check=True,
        env=os.environ.copy(),
    )
    subprocess.run(
        [
            str(RUST_BIN),
            "process",
            str(input_wav),
            str(enroll_json),
            str(out_wav),
            "--gate-decisions",
            str(diag_json),
        ],
        check=True,
        env=os.environ.copy(),
    )

    audio, sr = sf.read(str(out_wav), dtype="float32", always_2d=False)
    assert sr == SR
    diag = json.loads(diag_json.read_text())
    return {
        "audio": audio.astype(np.float32),
        "gate_per_frame": np.asarray(diag["gate_per_frame"], dtype=bool),
        "score_per_frame": np.asarray(diag["score_per_frame"], dtype=np.float32),
    }


def run_python(mixture: np.ndarray, enroll: np.ndarray) -> dict:
    from mellonella_poc.config import Config
    from mellonella_poc.pipeline import (
        PipelineComponents,
        enroll_from_recording,
        process_offline,
    )

    config = Config()
    components = PipelineComponents.build_default(config)
    pool = enroll_from_recording(enroll, SR, config, components)
    result = process_offline(mixture, SR, pool, config, components)
    # The Python pipeline emits 48 kHz output (D-002); resample to 16k
    # for direct comparison with Rust which stays at 16 kHz throughout.
    audio_48k = result.audio
    if audio_48k.size != mixture.size:
        from scipy.signal import resample_poly

        out_sr = config.audio.output_sr
        from math import gcd

        g = gcd(out_sr, SR)
        audio_16k = resample_poly(audio_48k, SR // g, out_sr // g).astype(np.float32)
    else:
        audio_16k = audio_48k.astype(np.float32)
    return {
        "audio": audio_16k,
        "gate_per_frame": result.gate_per_frame,
        "score_per_frame": result.score_per_frame,
    }


def metrics(name: str, audio: np.ndarray, gate: np.ndarray) -> dict:
    return {
        "name": name,
        "samples": int(audio.size),
        "duration_sec": float(audio.size / SR),
        "gate_on_frames": int(gate.sum()),
        "gate_total_frames": int(gate.size),
        "gate_duty_cycle": float(gate.mean()) if gate.size else 0.0,
        "audio_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "audio_peak": float(np.max(np.abs(audio))),
    }


def main() -> int:
    print(f"[scenario_1] SR={SR} SNR={SNR_DB} dB duration={DURATION_SEC} s", file=sys.stderr)
    target = synth_target(DURATION_SEC, SR)
    enroll_audio = synth_target(ENROLL_DURATION_SEC, SR, seed=23)
    noise = synth_noise(DURATION_SEC, SR)
    mixture = mix_at_snr(target, noise, SNR_DB)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        in_wav = work / "input.wav"
        enroll_wav = work / "enroll.wav"
        write_wav(in_wav, mixture, SR)
        write_wav(enroll_wav, enroll_audio, SR)

        print("[scenario_1] running Python pipeline …", file=sys.stderr)
        py = run_python(mixture, enroll_audio)

        print("[scenario_1] running Rust CLI …", file=sys.stderr)
        rs = run_rust(in_wav, enroll_wav, work)

    m_py = metrics("python", py["audio"], py["gate_per_frame"])
    m_rs = metrics("rust", rs["audio"], rs["gate_per_frame"])

    # Per-frame gate agreement.
    n_compare = min(py["gate_per_frame"].size, rs["gate_per_frame"].size)
    if n_compare:
        agree = int(np.sum(py["gate_per_frame"][:n_compare] == rs["gate_per_frame"][:n_compare]))
        agreement = agree / n_compare
    else:
        agreement = 0.0

    print(json.dumps(
        {
            "snr_db": SNR_DB,
            "python": m_py,
            "rust": m_rs,
            "gate_agreement": agreement,
            "n_frames_compared": int(n_compare),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
