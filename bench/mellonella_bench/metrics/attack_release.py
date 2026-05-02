"""Empirical attack/release time-constant fitting.

Feeds a step input through the gate envelope and fits

    y(t) = target + (initial - target) * exp(-(t - t0) / tau)

via least squares to recover ``tau``. ``attack_ms`` corresponds to a 0→1
step; ``release_ms`` to a 1→0 step. The 'time constant' is the canonical
1/e settling time, matching :class:`mellonella_poc.gating.EnvelopeState`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttackReleaseFit:
    """Result of a single attack-or-release fit."""

    tau_ms: float
    initial: float
    final: float
    rms_residual: float


def fit_first_order(
    response: np.ndarray,
    sample_rate: int,
    *,
    initial: float | None = None,
    final: float | None = None,
) -> AttackReleaseFit:
    """Fit ``y(t) = final + (initial - final) * exp(-t / tau)`` to ``response``.

    ``initial`` defaults to ``response[0]`` and ``final`` defaults to
    ``response[-1]``. Both can be supplied explicitly when the asymptote is
    known a priori (e.g. 0.0 and 1.0 for an attack step).
    """
    response = np.asarray(response, dtype=np.float64).flatten()
    if response.size < 4:
        raise ValueError("need at least 4 samples to fit a first-order curve")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    a = float(initial if initial is not None else response[0])
    b = float(final if final is not None else response[-1])
    if a == b:
        raise ValueError("initial and final must differ")

    # y(t) = b + (a - b) e^(-t/tau)  ⇒  log((y - b) / (a - b)) = -t / tau
    delta = response - b
    sign = np.sign(a - b)
    y_norm = sign * delta / abs(a - b)
    valid = y_norm > 1e-6
    if valid.sum() < 4:
        raise ValueError("not enough monotonic samples to fit tau")
    t = np.arange(response.size, dtype=np.float64) / sample_rate
    log_y = np.log(y_norm[valid])
    inv_tau, _ = np.polyfit(t[valid], log_y, 1)
    tau = -1.0 / inv_tau if inv_tau < 0 else float("inf")
    fitted = b + (a - b) * np.exp(-t / tau)
    rms = float(np.sqrt(np.mean((fitted - response) ** 2)))
    return AttackReleaseFit(
        tau_ms=tau * 1000.0,
        initial=a,
        final=b,
        rms_residual=rms,
    )


def measure_attack_release_from_step(
    envelope: np.ndarray,
    step_index: int,
    sample_rate: int,
) -> tuple[AttackReleaseFit, AttackReleaseFit]:
    """Convenience helper: fit attack and release halves around a single step.

    ``envelope`` is the full per-sample envelope produced by
    :class:`mellonella_poc.gating.EnvelopeState`. ``step_index`` is the
    boundary between the two halves; the function inspects the direction of
    each half independently and returns ``(attack_fit, release_fit)``.
    """
    if not 1 <= step_index < envelope.size - 1:
        raise ValueError("step_index must lie strictly inside the envelope")
    pre = envelope[:step_index]
    post = envelope[step_index:]

    def _fit_segment(segment: np.ndarray) -> AttackReleaseFit:
        if segment[-1] >= segment[0]:
            return fit_first_order(segment, sample_rate, initial=float(segment[0]), final=1.0)
        return fit_first_order(segment, sample_rate, initial=float(segment[0]), final=0.0)

    pre_fit = _fit_segment(pre)
    post_fit = _fit_segment(post)
    if pre_fit.final >= post_fit.final:
        return pre_fit, post_fit
    return post_fit, pre_fit
