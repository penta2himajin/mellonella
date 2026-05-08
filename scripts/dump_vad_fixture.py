#!/usr/bin/env python3
"""Dump silero-vad reference probabilities for the Rust port's parity test.

Drives a deterministic synth waveform through the Python silero-vad
ONNX wrapper one 512-sample chunk at a time and records the resulting
speech probability sequence. Output goes to:

* ``rust/mellonella-core/tests/fixtures/vad_input.bin``
* ``rust/mellonella-core/tests/fixtures/vad_expected.bin``

Run:

    python scripts/dump_vad_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rust" / "mellonella-core" / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 512
SR = 16_000


def synth(seed: int, sr: int, duration_sec: float, f0: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration_sec)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for harmonic in range(1, 6):
        phase = rng.uniform(0, 2 * np.pi)
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t + phase).astype(np.float32)
    am = 0.5 * (1.0 + 0.3 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
    wave = (wave * am).astype(np.float32)
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave = (wave / peak) * 0.9
    return wave.astype(np.float32)


def main() -> int:
    from silero_vad import load_silero_vad  # type: ignore[import-not-found]
    import torch

    model = load_silero_vad(onnx=True)

    # 1 s of synthesised "speech" at 180 Hz harmonic stack.
    audio = synth(seed=42, sr=SR, duration_sec=1.0, f0=180.0)
    n_chunks = audio.size // CHUNK
    probs = np.zeros(n_chunks, dtype=np.float32)
    for i in range(n_chunks):
        chunk = audio[i * CHUNK : (i + 1) * CHUNK]
        prob = float(model(torch.from_numpy(chunk), SR).item())
        probs[i] = prob

    (OUT / "vad_input.bin").write_bytes(audio.tobytes())
    (OUT / "vad_expected.bin").write_bytes(probs.tobytes())
    meta = {
        "sample_rate": SR,
        "chunk_samples": CHUNK,
        "n_chunks": int(n_chunks),
        "input_shape": list(audio.shape),
        "expected_shape": list(probs.shape),
    }
    (OUT / "vad_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {n_chunks} chunks @ {SR} Hz; first 5 probs = {probs[:5].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
