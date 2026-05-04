"""Gating logic: integrated score, hangover, attack/release envelope.

Pure NumPy / Python — no model dependencies. This is the algorithmic core
that Phase 1 needs to validate before any Rust port.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GatingConfig

EPS = 1e-12


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


def load_cohort(path: str | Path) -> np.ndarray:
    """Load an impostor cohort (.npz) and return the L2-normalised matrix.

    The file format matches what ``scripts/build_impostor_cohort.py``
    writes: an ``embeddings`` key holding a ``(N, 192) float32`` array.
    Vectors are re-normalised on load so that downstream cosine-vs-cohort
    work reduces to a plain dot product.
    """
    with np.load(path, allow_pickle=True) as data:
        raw = np.asarray(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return (raw / norms).astype(np.float32)


def as_norm_score(
    embedding: np.ndarray,
    raw_score: float,
    cohort: np.ndarray,
    top_k: int,
) -> float:
    """Apply Adaptive S-Norm to ``raw_score``.

        z = (raw_score - μ_topK(impostor_scores)) / σ_topK(impostor_scores)

    where ``impostor_scores`` is the cosine similarity between the L2-
    normalised query ``embedding`` and each row of ``cohort`` (also
    L2-normalised — see :func:`load_cohort`). The top-K largest scores
    define the "tough impostor" baseline so the normalisation target
    matches what the pool is actually competing against.

    Empty / degenerate inputs fall back gracefully:

    * empty ``cohort`` → return ``raw_score`` unchanged
    * zero-length ``embedding`` → return ``raw_score`` unchanged
    * ``σ`` close to zero → return ``raw_score - μ`` (skip the divide)

    Per :data:`docs/decisions.md` D-010 we accept that the resulting
    score is on a different scale (typical z-score range: -3 to +3) than
    the legacy ``α·cs + β·f0`` formula and pair it with the dedicated
    :attr:`GatingConfig.theta_pass_as_norm` threshold.
    """
    if cohort.shape[0] == 0:
        return float(raw_score)
    norm = float(np.linalg.norm(embedding))
    if norm < EPS:
        return float(raw_score)
    query = embedding.astype(np.float32) / norm
    impostor_scores = cohort @ query
    k = int(min(top_k, impostor_scores.shape[0]))
    if k <= 0:
        return float(raw_score)
    if k == impostor_scores.shape[0]:
        top = impostor_scores
    else:
        top = np.partition(impostor_scores, -k)[-k:]
    mu = float(top.mean())
    sigma = float(top.std())
    if sigma < EPS:
        return float(raw_score - mu)
    return float((raw_score - mu) / sigma)


def target_score_as_norm(
    embedding: np.ndarray,
    pool: Iterable[np.ndarray],
    cohort: np.ndarray,
    config: GatingConfig,
) -> float:
    """AS-Norm-only variant of :func:`target_score`.

    Used by :func:`mellonella_poc.pipeline.process_offline` when
    :attr:`GatingConfig.use_as_norm` is True. F0 is intentionally NOT
    folded into the gate-decision score under AS-Norm — the cohort
    cancels per-language drift in the embedding similarity directly,
    and the F0 channel still gates auto-learn admission via
    :func:`should_admit_auto_learn`.
    """
    cs = cos_sim_max(embedding, pool)
    return as_norm_score(embedding, cs, cohort, config.as_norm_top_k)


@dataclass
class GateState:
    """Mutable hangover state for the binary gate.

    `update` returns the binary decision (True == pass). The caller feeds
    that decision into `EnvelopeState.advance` to derive a smooth gain.
    """

    config: GatingConfig
    is_on: bool = False
    elapsed_off_ms: float = 0.0

    @property
    def _theta_pass(self) -> float:
        return self.config.theta_pass_as_norm if self.config.use_as_norm else self.config.theta_pass

    def update(self, score: float, dt_ms: float) -> bool:
        if score >= self._theta_pass:
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


def should_admit_auto_learn(
    score: float,
    f0_match_value: float,
    continuous_speech_ms: float,
    config: GatingConfig,
) -> bool:
    """Check the auto-learn admission rule from `docs/gating.md` D-004.

    Returns True iff every required guard is satisfied:

    * ``score >= theta_learn``                     high cosine confidence
    * ``f0_match_value >= theta_f0``               F0 inside the enrollment range
    * ``continuous_speech_ms >= min_continuous_speech_sec * 1000``
                                                   long enough run that the
                                                   embedding represents stable
                                                   speech, not a transient

    This function is the gatekeeper *before* :meth:`EmbeddingPool.add_auto_learn`,
    which then applies the anchor-distance check to decide whether the
    candidate is actually a drift-safe addition.
    """
    theta_learn = config.theta_learn_as_norm if config.use_as_norm else config.theta_learn
    return (
        score >= theta_learn
        and f0_match_value >= config.theta_f0
        and continuous_speech_ms >= config.min_continuous_speech_sec * 1000.0
    )
