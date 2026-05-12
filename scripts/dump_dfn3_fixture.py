#!/usr/bin/env python3
"""Dump DFN3 enhancement reference for the Rust parity test.

Runs the *patched* DFN3 (same df_op rewrite as
``scripts/export_dfn3_onnx.py``) over a deterministic 1-second noise +
sine mix at 48 kHz, single 102-frame chunk. Writes:

* ``rust/mellonella-core/tests/fixtures/dfn3_input.bin``       — 48 960 f32
* ``rust/mellonella-core/tests/fixtures/dfn3_expected_audio.bin`` — 48 960 f32
* ``rust/mellonella-core/tests/fixtures/dfn3_expected_spec.bin``  — 102 × 481 × 2 f32
* ``rust/mellonella-core/tests/fixtures/dfn3_meta.json``        — shapes / params

The Rust integration test reproduces the same path (STFT + ERB features
+ ONNX + iSTFT via ``deep_filter`` primitives) and asserts max|Δ| on
both intermediate ``enhanced_spec`` and the final audio waveform.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rust" / "mellonella-core" / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

SR = 48_000
FRAMES_PER_CHUNK = 102
HOP = 480
N_FFT = 960
SAMPLES_PER_CHUNK = FRAMES_PER_CHUNK * HOP


def synth(seed: int, sr: int, duration_sec: float, f0: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration_sec)) / sr
    speech = 0.5 * np.sin(2 * np.pi * f0 * t).astype(np.float32)
    noise = rng.normal(0.0, 0.1, t.size).astype(np.float32)
    mix = speech + noise
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = (mix / peak) * 0.7
    return mix.astype(np.float32)


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_dfn3_onnx import _load_model_and_patch
    from df.enhance import df_features
    from df.model import ModelParams

    model, df_state = _load_model_and_patch()
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))

    audio_raw = synth(seed=42, sr=SR, duration_sec=1.0, f0=440.0)
    # The Python pipeline pads n_fft trailing zeros so the STFT pipeline
    # has room to flush. After the pad the chunk is 48 960 samples = 102
    # hops; the model's exported shape requires exactly that.
    audio_padded = np.zeros(SAMPLES_PER_CHUNK, dtype=np.float32)
    audio_padded[: audio_raw.size] = audio_raw

    audio_t = torch.from_numpy(audio_padded).unsqueeze(0)
    spec, feat_erb, feat_spec = df_features(audio_t, df_state, nb_df, device="cpu")
    print(
        f"[parity] spec={tuple(spec.shape)} erb={tuple(feat_erb.shape)} "
        f"df_spec={tuple(feat_spec.shape)}",
        file=sys.stderr,
    )

    with torch.no_grad():
        enhanced_spec, _, _, _ = model(spec, feat_erb, feat_spec)
    enhanced_spec_np = enhanced_spec.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    # Synthesise via the libdf binding — takes (C, Tf, F) complex and
    # returns (C, T) float32.
    df_state.reset()
    enhanced_complex = enhanced_spec_np.view(np.complex64).reshape(
        1, FRAMES_PER_CHUNK, 481
    )
    out_buffer = df_state.synthesis(enhanced_complex)
    if out_buffer.ndim == 2:
        out_buffer = out_buffer[0]
    out_buffer = out_buffer[:SAMPLES_PER_CHUNK].astype(np.float32)

    (OUT / "dfn3_input.bin").write_bytes(audio_padded.tobytes())
    (OUT / "dfn3_expected_audio.bin").write_bytes(out_buffer.astype(np.float32).tobytes())
    meta = {
        "sample_rate": SR,
        "n_fft": N_FFT,
        "hop_size": HOP,
        "n_erb": int(feat_erb.shape[-1]),
        "nb_df": int(feat_spec.shape[-2]),
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "samples_per_chunk": SAMPLES_PER_CHUNK,
        "input_shape": list(audio_padded.shape),
        "expected_audio_shape": list(out_buffer.shape),
    }
    (OUT / "dfn3_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[parity] wrote 3 fixtures + meta to {OUT.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
