"""ECAPA-TDNN speaker embedding wrapper (SpeechBrain).

The 192-dim output is the primary signal feeding `gating.target_score`.
Heavy import is deferred until first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EcapaTdnn:
    sample_rate: int = 16_000
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str | None = None
    _model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from speechbrain.inference.speaker import (  # type: ignore[import-not-found]
                EncoderClassifier,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "speechbrain not installed; reinstall with `pip install -e poc[models]`"
            ) from exc
        self._model = EncoderClassifier.from_hparams(
            source=self.source,
            savedir=self.savedir,
            run_opts={"device": "cpu"},
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Compute a 192-dim speaker embedding for `audio` at `sample_rate`."""
        if audio.ndim != 1:
            raise ValueError("EcapaTdnn expects a 1-D mono buffer")
        if audio.size < self.sample_rate:
            raise ValueError(
                f"need at least 1 s of audio for ECAPA-TDNN, got {audio.size / self.sample_rate:.2f}s"
            )
        self._ensure_model()
        import torch  # type: ignore[import-not-found]

        with torch.no_grad():
            tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            emb = self._model.encode_batch(tensor)
        return emb.squeeze().cpu().numpy().astype(np.float32)
