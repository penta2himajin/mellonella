"""Noise-suppression quality metrics.

- ``si_sdr``      pure NumPy implementation, always available.
- ``pesq_score``  thin wrapper around the ``pesq`` package (extra dep).
- ``stoi_score``  thin wrapper around the ``pystoi`` package (extra dep).

The PESQ/STOI wrappers raise :class:`MissingDependencyError` when the
underlying packages are not installed, so callers can decide whether to
skip or fail.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


class MissingDependencyError(RuntimeError):
    """Raised when an optional metric backend is not installed."""


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB.

    Definition (Le Roux et al. 2019):

        s_target = <s_hat, s> / ||s||^2 * s
        e_noise  = s_hat - s_target
        SI-SDR   = 10 * log10(||s_target||^2 / ||e_noise||^2)
    """
    s = np.asarray(reference, dtype=np.float64).flatten()
    s_hat = np.asarray(estimate, dtype=np.float64).flatten()
    if s.shape != s_hat.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {s_hat.shape}")
    if s.size == 0:
        raise ValueError("empty signal")

    s = s - s.mean()
    s_hat = s_hat - s_hat.mean()
    denom = float(np.dot(s, s)) + EPS
    alpha = float(np.dot(s_hat, s)) / denom
    s_target = alpha * s
    e_noise = s_hat - s_target
    num = float(np.dot(s_target, s_target)) + EPS
    den = float(np.dot(e_noise, e_noise)) + EPS
    return 10.0 * float(np.log10(num / den))


def pesq_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    mode: str = "wb",
) -> float:
    """Wide-band (16 kHz) or narrow-band (8 kHz) PESQ score."""
    try:
        from pesq import pesq as _pesq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "pesq not installed; install with `pip install -e bench[metrics]`"
        ) from exc

    if mode not in {"wb", "nb"}:
        raise ValueError(f"mode must be 'wb' or 'nb', got {mode!r}")
    expected_sr = 16_000 if mode == "wb" else 8_000
    if sample_rate != expected_sr:
        raise ValueError(f"PESQ {mode} expects {expected_sr} Hz, got {sample_rate} Hz")
    return float(
        _pesq(
            sample_rate,
            np.asarray(reference, dtype=np.float32),
            np.asarray(estimate, dtype=np.float32),
            mode,
        )
    )


def stoi_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    *,
    extended: bool = False,
) -> float:
    """STOI (Short-Time Objective Intelligibility), in [0, 1]."""
    try:
        from pystoi import stoi as _stoi  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "pystoi not installed; install with `pip install -e bench[metrics]`"
        ) from exc

    return float(
        _stoi(
            np.asarray(reference, dtype=np.float32),
            np.asarray(estimate, dtype=np.float32),
            sample_rate,
            extended=extended,
        )
    )
