"""Gating logic: integrated score, hangover, attack/release envelope.

Pure NumPy / Python — no model dependencies. This is the algorithmic core
that Phase 1 needs to validate before any Rust port.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .config import GatingConfig


def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0 for zero vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cos_sim_max(emb: np.ndarray, pool: Iterable[np.ndarray]) -> float:
    """Maximum cosine similarity between `emb` and any vector in `pool`."""
    best = 0.0
    found = False
    for ref in pool:
        found = True
        score = cos_similarity(emb, ref)
        if score > best:
            best = score
    return best if found else 0.0


def f0_match(f0_mean: float, mu: float, sigma: float) -> float:
    """Gaussian fit of `f0_mean` against the enrollment F0 distribution.

    `sigma` <= 0 disables the contribution (returns 1.0 — neutral).
    """
    if sigma <= 0.0 or not math.isfinite(f0_mean):
        return 1.0
    z = (f0_mean - mu) / sigma
    return float(math.exp(-0.5 * z * z))


def target_score(
    embedding: np.ndarray,
    pool: Iterable[np.ndarray],
    f0_mean: float,
    f0_mu: float,
    f0_sigma: float,
    config: GatingConfig,
) -> float:
    """Integrated target score per `docs/gating.md`.

    score = alpha * cos_sim_max + beta * f0_match
    """
    cs = cos_sim_max(embedding, pool)
    fm = f0_match(f0_mean, f0_mu, f0_sigma)
    return config.alpha * cs + config.beta * fm


@dataclass
class GateState:
    """Mutable hangover state for the binary gate.

    `update` returns the binary decision (True == pass). The caller feeds
    that decision into `EnvelopeState.advance` to derive a smooth gain.
    """

    config: GatingConfig
    is_on: bool = False
    elapsed_off_ms: float = 0.0

    def update(self, score: float, dt_ms: float) -> bool:
        if score >= self.config.theta_pass:
            self.is_on = True
            self.elapsed_off_ms = 0.0
            return True

        if self.is_on:
            self.elapsed_off_ms += dt_ms
            if self.elapsed_off_ms < self.config.hangover_ms:
                return True
            self.is_on = False
            return False

        return False


@dataclass
class EnvelopeState:
    """Attack/release envelope follower.

    Computes a per-sample gain in [0, 1]. Use `advance(target_on, n_samples)`
    to step forward by `n_samples` audio frames; the returned ndarray of
    length `n_samples` is the gain to multiply against the audio.
    """

    config: GatingConfig
    sample_rate: int
    value: float = 0.0

    def _coef(self, ms: float) -> float:
        if ms <= 0:
            return 1.0
        tau_samples = ms * self.sample_rate / 1000.0
        return 1.0 - math.exp(-1.0 / tau_samples)

    def advance(self, target_on: bool, n_samples: int) -> np.ndarray:
        coef = self._coef(self.config.attack_ms if target_on else self.config.release_ms)
        target = 1.0 if target_on else 0.0
        out = np.empty(n_samples, dtype=np.float32)
        v = self.value
        for i in range(n_samples):
            v += coef * (target - v)
            out[i] = v
        self.value = v
        return out


def apply_envelope(
    audio: np.ndarray,
    gate_decisions: list[tuple[int, bool]],
    sample_rate: int,
    config: GatingConfig,
) -> np.ndarray:
    """Apply the attack/release envelope to `audio` given gate decisions.

    `gate_decisions` is a list of `(start_sample, is_on)` tuples in increasing
    order. The first tuple's `start_sample` should be 0.

    Returns a new array of the same shape as `audio`.
    """
    if audio.ndim != 1:
        raise ValueError("audio must be a 1-D array")
    if not gate_decisions or gate_decisions[0][0] != 0:
        raise ValueError("gate_decisions must start with sample 0")

    env = EnvelopeState(config=config, sample_rate=sample_rate)
    out = np.empty_like(audio, dtype=np.float32)
    n = audio.shape[0]
    boundaries = [start for start, _ in gate_decisions] + [n]
    for (start, is_on), end in zip(gate_decisions, boundaries[1:], strict=False):
        gain = env.advance(is_on, end - start)
        out[start:end] = audio[start:end].astype(np.float32) * gain
    return out


def update_gate(
    state: GateState,
    score: float,
    dt_ms: float,
) -> bool:
    """Convenience wrapper around `GateState.update` for symmetry with `apply_envelope`."""
    return state.update(score, dt_ms)
