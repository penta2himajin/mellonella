#!/usr/bin/env python3
"""Dump pipeline parity reference data for the Rust integration test.

Runs a deliberately Rust-equivalent pipeline in Python so the orchestrator
output (per-VAD-frame score + gate state) can be compared byte-for-byte
on the Rust side. The Rust pipeline currently:

* Skips DFN3 (audio assumed already clean)
* Operates entirely at 16 kHz (no output-rate resample)
* Uses raw `cos_sim_max` instead of `α·cs + β·f0` in the non-AS-Norm
  path (the alpha/beta fields aren't on Rust's GateConfig yet)

To get an exact parity oracle, this script reproduces the *Rust* path
in Python rather than calling `mellonella_poc.pipeline.process_offline`
directly. Doing it the other way would require monkey-patching DFN3 +
resample out of the PoC code.

Writes to ``rust/mellonella-core/tests/fixtures/``:

* ``pipeline_input.bin``         — 2 s synth audio @ 16 kHz (f32)
* ``pipeline_anchor.bin``        — 192-dim anchor embedding (f32)
* ``pipeline_score_per_frame.bin``  — N-frame score (f32)
* ``pipeline_gate_per_frame.bin``   — N-frame gate state (u8, 0/1)
* ``pipeline_meta.json``         — shapes + cadence parameters

The audio is 2 s of 180 Hz harmonic stack so the SV update (≥ 1 s speech
buffer) actually fires inside the run.

Run:

    python scripts/dump_pipeline_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rust" / "mellonella-core" / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

SR = 16_000
VAD_FRAME = 512
SV_WINDOW = 16_000
SV_UPDATE = 4_000
# A 180 Hz harmonic stack doesn't look enough like speech for silero-vad
# to clear 0.5 on most frames, so the speech buffer never fills and the
# SV-refresh path never fires. Setting the threshold below the smallest
# probability the model returns forces every frame to contribute, which
# exercises the whole orchestrator including ECAPA embedding refresh.
VAD_THRESHOLD = -1.0
FRAME_MS = 1000.0 * VAD_FRAME / SR  # 32 ms

# Mirrors GateConfig defaults in rust/mellonella-core/src/gating.rs.
THETA_PASS = 0.30
HANGOVER_MS = 300.0


def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    from speechbrain.inference.speaker import (  # type: ignore[import-not-found]
        EncoderClassifier,
    )

    audio = synth(seed=42, sr=SR, duration_sec=2.0, f0=180.0)

    print("[parity] loading silero-vad ONNX")
    vad = load_silero_vad(onnx=True)

    print("[parity] loading SpeechBrain ECAPA")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    def embed(window: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            return ecapa.encode_batch(t).squeeze().cpu().numpy().astype(np.float32)

    # Synthetic anchor = embedding of the audio itself, padded out to
    # 1 s so ECAPA accepts it. This gives `cos_sim_max` something
    # non-trivial to score against without needing a separate enrollment
    # clip — the Rust test will load this anchor verbatim.
    anchor = embed(audio)
    print(f"[parity] anchor shape = {anchor.shape}, |anchor| = {np.linalg.norm(anchor):.3f}")

    # Reset VAD state before the loop (load_silero_vad starts fresh, but
    # being explicit makes the Rust reproduction trivially exact).
    vad.reset_states()

    # State variables — mirror process_offline in Rust line-for-line.
    speech_buffer: list[float] = []
    samples_since_update = 0
    consecutive_speech_ms = 0.0
    last_score = 0.0
    gate_is_on = False
    elapsed_off_ms = 0.0

    score_per_frame: list[float] = []
    gate_per_frame: list[int] = []

    n_frames = audio.size // VAD_FRAME
    for frame_idx in range(n_frames):
        frame = audio[frame_idx * VAD_FRAME : (frame_idx + 1) * VAD_FRAME]
        prob = float(vad(torch.from_numpy(frame), SR).item())
        if prob > VAD_THRESHOLD:
            speech_buffer.extend(frame.tolist())
            if len(speech_buffer) > SV_WINDOW:
                speech_buffer = speech_buffer[-SV_WINDOW:]
            consecutive_speech_ms += FRAME_MS
        else:
            consecutive_speech_ms = 0.0
        samples_since_update += VAD_FRAME

        if samples_since_update >= SV_UPDATE and len(speech_buffer) >= SV_WINDOW:
            samples_since_update = 0
            window = np.asarray(speech_buffer[-SV_WINDOW:], dtype=np.float32)
            emb = embed(window)
            cs = cos_similarity(emb, anchor)
            last_score = cs  # mirrors Rust non-AS-Norm path (raw cs)

        # GateState::update — mirrors rust/mellonella-core/src/gating.rs.
        if last_score >= THETA_PASS:
            gate_is_on = True
            elapsed_off_ms = 0.0
            is_on = True
        elif gate_is_on:
            elapsed_off_ms += FRAME_MS
            if elapsed_off_ms < HANGOVER_MS:
                is_on = True
            else:
                gate_is_on = False
                is_on = False
        else:
            is_on = False

        score_per_frame.append(last_score)
        gate_per_frame.append(1 if is_on else 0)

    (OUT / "pipeline_input.bin").write_bytes(audio.astype(np.float32).tobytes())
    (OUT / "pipeline_anchor.bin").write_bytes(anchor.astype(np.float32).tobytes())
    (OUT / "pipeline_score_per_frame.bin").write_bytes(
        np.asarray(score_per_frame, dtype=np.float32).tobytes()
    )
    (OUT / "pipeline_gate_per_frame.bin").write_bytes(
        np.asarray(gate_per_frame, dtype=np.uint8).tobytes()
    )

    meta = {
        "sample_rate": SR,
        "vad_frame_samples": VAD_FRAME,
        "sv_window_samples": SV_WINDOW,
        "sv_update_samples": SV_UPDATE,
        "vad_threshold": VAD_THRESHOLD,
        "theta_pass": THETA_PASS,
        "hangover_ms": HANGOVER_MS,
        "n_frames": int(n_frames),
        "anchor_dim": int(anchor.size),
        "audio_source": "fixtures/vad_input.bin",
    }
    (OUT / "pipeline_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    on_count = sum(gate_per_frame)
    print(f"[parity] {n_frames} frames, {on_count} ON, score range "
          f"[{min(score_per_frame):.3f}, {max(score_per_frame):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
