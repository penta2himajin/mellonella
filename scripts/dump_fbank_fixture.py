#!/usr/bin/env python3
"""Dump SpeechBrain Fbank reference data for the Rust port's parity tests.

Writes raw little-endian f32 binaries under
``rust/mellonella-core/tests/fixtures/``:

* ``fbank_filterbank.bin``  — 201 × 80 triangular filterbank matrix
* ``fbank_input.bin``       — synth 16 kHz waveform (3 s, 180 Hz harmonic stack)
* ``fbank_expected.bin``    — expected log-mel output (n_frames × 80)
* ``fbank_meta.json``       — shapes / dtype / parameters

Run on a machine with ``poc[models]`` installed:

    python scripts/dump_fbank_fixture.py

The Rust side embeds these via ``include_bytes!`` inside an integration
test that compares its own Fbank against ``fbank_expected.bin``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rust" / "mellonella-core" / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


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


def write_f32(path: Path, arr: np.ndarray) -> None:
    flat = np.ascontiguousarray(arr, dtype=np.float32)
    path.write_bytes(flat.tobytes())
    print(f"  wrote {path.relative_to(ROOT)}: shape={arr.shape}, bytes={path.stat().st_size}")


def main() -> int:
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import-not-found]

    print("[fbank] loading SpeechBrain ECAPA")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    fb = classifier.mods.compute_features
    fbk = fb.compute_fbanks

    # Build the filterbank matrix the same way Filterbank.forward does,
    # *before* taking the log so we can re-apply log_mel in Rust.
    f_central_mat = fbk.f_central.repeat(fbk.all_freqs_mat.shape[1], 1).transpose(0, 1)
    band_mat = fbk.band.repeat(fbk.all_freqs_mat.shape[1], 1).transpose(0, 1)
    fbank_matrix = fbk._create_fbank_matrix(f_central_mat, band_mat)  # noqa: SLF001
    fbank_matrix = fbank_matrix.detach().cpu().numpy().astype(np.float32)
    print(f"[fbank] filterbank matrix shape = {fbank_matrix.shape}")

    sr = 16_000
    # 1 s clip is enough to validate per-frame Fbank parity and keeps the
    # fixture size under ~150 kB total.
    wav = synth(seed=42, sr=sr, duration_sec=1.0, f0=180.0)

    with torch.no_grad():
        feats = fb(torch.from_numpy(wav).unsqueeze(0))
    feats = feats.squeeze(0).cpu().numpy().astype(np.float32)
    print(f"[fbank] expected output shape = {feats.shape}")

    write_f32(OUT / "fbank_filterbank.bin", fbank_matrix)
    write_f32(OUT / "fbank_input.bin", wav)
    write_f32(OUT / "fbank_expected.bin", feats)

    meta = {
        "sample_rate": sr,
        "n_fft": int(fb.compute_STFT.n_fft),
        "hop_length": int(fb.compute_STFT.hop_length),
        "win_length": int(fb.compute_STFT.win_length),
        "window": "hamming",
        "center": True,
        "pad_mode": "constant",
        "n_mels": int(fbk.n_mels),
        "f_min": float(fbk.f_min),
        "f_max": float(fbk.f_max),
        "log_mel": True,
        "power": int(fbk.power_spectrogram),
        "amin": float(fbk.amin),
        "top_db": float(fbk.top_db),
        "multiplier": int(fbk.multiplier),
        "fbank_matrix_shape": list(fbank_matrix.shape),
        "input_shape": list(wav.shape),
        "expected_shape": list(feats.shape),
    }
    (OUT / "fbank_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {(OUT / 'fbank_meta.json').relative_to(ROOT)}")

    print("[fbank] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
