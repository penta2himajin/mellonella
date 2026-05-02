"""YIN pitch estimator (NumPy implementation).

Reference: De Cheveigné & Kawahara, *YIN, a fundamental frequency estimator
for speech and music*, JASA 2002.

Used as the lightweight F0 path. CREPE is reserved for the optional
high-precision branch (`docs/architecture.md` Stage 4).
"""

from __future__ import annotations

import numpy as np


def _difference(frame: np.ndarray, max_tau: int) -> np.ndarray:
    diff = np.zeros(max_tau, dtype=np.float64)
    for tau in range(1, max_tau):
        d = frame[: frame.size - tau] - frame[tau:]
        diff[tau] = float(np.dot(d, d))
    return diff


def _cumulative_mean_normalized(diff: np.ndarray) -> np.ndarray:
    cmnd = np.empty_like(diff)
    cmnd[0] = 1.0
    running = 0.0
    for tau in range(1, diff.size):
        running += diff[tau]
        cmnd[tau] = diff[tau] * tau / running if running > 0 else 1.0
    return cmnd


def yin_frame(
    frame: np.ndarray,
    sample_rate: int,
    f_min: float = 50.0,
    f_max: float = 500.0,
    threshold: float = 0.1,
) -> float | None:
    """Estimate F0 for a single frame. Returns None for unvoiced/unstable frames."""
    if frame.ndim != 1:
        raise ValueError("frame must be 1-D")
    if frame.size < 64:
        return None

    tau_min = max(2, int(sample_rate / f_max))
    tau_max = min(frame.size // 2, int(sample_rate / f_min))
    if tau_max <= tau_min:
        return None

    diff = _difference(frame, tau_max + 1)
    cmnd = _cumulative_mean_normalized(diff)

    tau = tau_min
    while tau < tau_max:
        if cmnd[tau] < threshold:
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            break
        tau += 1
    else:
        return None

    if 1 <= tau < tau_max - 1:
        s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = s0 + s2 - 2.0 * s1
        tau_f = tau + 0.5 * (s0 - s2) / denom if denom != 0.0 else float(tau)
    else:
        tau_f = float(tau)

    if tau_f <= 0:
        return None
    return float(sample_rate) / tau_f


def estimate_f0_track(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    f_min: float = 50.0,
    f_max: float = 500.0,
) -> np.ndarray:
    """Compute a F0 track over `audio`. Unvoiced frames yield NaN."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("audio must be 1-D")
    n_frames = max(0, 1 + (audio.size - frame_size) // hop_size)
    track = np.full(n_frames, np.nan, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start : start + frame_size]
        est = yin_frame(frame, sample_rate, f_min=f_min, f_max=f_max)
        if est is not None:
            track[i] = est
    return track


def f0_statistics(track: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of voiced frames in `track`. NaN-safe."""
    voiced = track[np.isfinite(track)]
    if voiced.size == 0:
        return 0.0, 0.0
    return float(np.mean(voiced)), float(np.std(voiced))
