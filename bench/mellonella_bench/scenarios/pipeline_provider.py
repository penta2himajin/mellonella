"""Real pipeline provider that builds an :class:`EmbeddingPool` per item.

Imports from :mod:`mellonella_poc` are deferred until ``for_item`` is
actually invoked, so unit tests that only construct the provider do not
trigger torch / speechbrain loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import PipelineCallable


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@dataclass
class RealPipelineProvider:
    """Build a per-item callable backed by :func:`mellonella_poc.pipeline.process_offline`.

    The provider expects each scenario item to expose an ``enrollment_path``
    attribute pointing at a clean recording of the target speaker; that
    recording is fed through :func:`enroll_from_recording` to populate the
    :class:`EmbeddingPool` used during evaluation.
    """

    config: Any = None  # Config; left untyped to keep the bench import lightweight.
    components: Any = None  # PipelineComponents; lazy-built on first use.

    def _ensure_runtime(self) -> tuple[Any, Any]:
        from mellonella_poc.config import Config
        from mellonella_poc.pipeline import PipelineComponents

        if self.config is None:
            self.config = Config()
        if self.components is None:
            self.components = PipelineComponents.build_default(self.config)
        return self.config, self.components

    def for_item(self, item: object) -> PipelineCallable:
        from mellonella_poc.pipeline import enroll_from_recording, process_offline

        enrollment_path = getattr(item, "enrollment_path", None)
        if enrollment_path is None:
            raise ValueError(
                "RealPipelineProvider requires items to expose an `enrollment_path` attribute"
            )

        config, components = self._ensure_runtime()
        enrollment_audio, enrollment_sr = _load_mono(Path(enrollment_path))
        pool = enroll_from_recording(enrollment_audio, enrollment_sr, config, components)

        def _call(mixture: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
            result = process_offline(mixture, sample_rate, pool, config, components)
            return result.audio, result.gate_per_frame

        return _call
