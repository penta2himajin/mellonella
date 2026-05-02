"""Embedding pool for explicit enrollment + auto-learning.

`EmbeddingPool` holds two compartments:

* `anchors`  — written once by explicit enrollment, never deleted.
* `auto_learn` — FIFO of high-confidence embeddings observed at runtime.

The pool enforces drift safeguards described in `docs/gating.md` D-004:
- candidates must clear `theta_learn` (handled by caller) AND a minimum
  cosine similarity to existing anchors;
- the pool's median anchor distance is monitored and a soft reset clears
  the auto-learn FIFO when it exceeds `anchor_reset_threshold`.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import GatingConfig
from .gating import cos_similarity


@dataclass
class EnrollmentMetadata:
    """F0 statistics captured from the explicit enrollment recording."""

    f0_mu: float = 0.0
    f0_sigma: float = 0.0


@dataclass
class EmbeddingPool:
    """Two-compartment pool with anchor protection."""

    config: GatingConfig
    anchors: list[np.ndarray] = field(default_factory=list)
    auto_learn: deque[np.ndarray] = field(default_factory=deque)
    metadata: EnrollmentMetadata = field(default_factory=EnrollmentMetadata)

    # ---- iteration ----------------------------------------------------
    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self.anchors
        yield from self.auto_learn

    def __len__(self) -> int:
        return len(self.anchors) + len(self.auto_learn)

    # ---- enrollment ---------------------------------------------------
    def add_anchors(self, embeddings: Iterable[np.ndarray]) -> None:
        for emb in embeddings:
            self.anchors.append(np.asarray(emb, dtype=np.float32))

    def set_f0_stats(self, mu: float, sigma: float) -> None:
        self.metadata = EnrollmentMetadata(f0_mu=float(mu), f0_sigma=float(sigma))

    # ---- auto-learning -------------------------------------------------
    def anchor_distance(self, emb: np.ndarray) -> float:
        """1 - max cosine similarity to any anchor."""
        if not self.anchors:
            raise RuntimeError("anchor_distance requires at least one anchor")
        best = max(cos_similarity(emb, a) for a in self.anchors)
        return 1.0 - best

    def can_auto_learn(self, emb: np.ndarray) -> bool:
        """Apply the drift-safety check before accepting an auto-learn candidate."""
        if not self.anchors:
            return False
        return self.anchor_distance(emb) <= self.config.anchor_distance_threshold

    def add_auto_learn(self, emb: np.ndarray) -> bool:
        """Try to admit `emb` to the auto-learn FIFO. Returns True on accept."""
        emb = np.asarray(emb, dtype=np.float32)
        if not self.can_auto_learn(emb):
            return False
        self.auto_learn.append(emb)
        while len(self.auto_learn) > self.config.auto_learn_max_size:
            self.auto_learn.popleft()
        return True

    # ---- drift monitoring ----------------------------------------------
    def median_anchor_distance(self) -> float:
        if not self.auto_learn:
            return 0.0
        distances = [self.anchor_distance(e) for e in self.auto_learn]
        return float(np.median(distances))

    def maybe_reset(self) -> bool:
        """Reset the auto-learn FIFO if median anchor distance is too high."""
        if self.median_anchor_distance() > self.config.anchor_reset_threshold:
            self.auto_learn.clear()
            return True
        return False

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "anchors": [a.tolist() for a in self.anchors],
            "auto_learn": [a.tolist() for a in self.auto_learn],
            "metadata": {
                "f0_mu": self.metadata.f0_mu,
                "f0_sigma": self.metadata.f0_sigma,
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: dict, config: GatingConfig) -> EmbeddingPool:
        if payload.get("version") != 1:
            raise ValueError(f"unsupported enrollment version: {payload.get('version')}")
        pool = cls(config=config)
        pool.anchors = [np.asarray(a, dtype=np.float32) for a in payload.get("anchors", [])]
        pool.auto_learn = deque(
            (np.asarray(a, dtype=np.float32) for a in payload.get("auto_learn", [])),
            maxlen=config.auto_learn_max_size,
        )
        meta = payload.get("metadata", {})
        pool.metadata = EnrollmentMetadata(
            f0_mu=float(meta.get("f0_mu", 0.0)),
            f0_sigma=float(meta.get("f0_sigma", 0.0)),
        )
        return pool

    @classmethod
    def load(cls, path: str | Path, config: GatingConfig) -> EmbeddingPool:
        return cls.from_dict(json.loads(Path(path).read_text()), config)
