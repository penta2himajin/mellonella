"""silero-vad wrapper. Heavy import is deferred until first use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SileroVAD:
    sample_rate: int = 16_000
    _model: Any = None
    _utils: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "silero-vad not installed; reinstall with `pip install -e poc[models]`"
            ) from exc
        self._model = load_silero_vad(onnx=True)

    def score(self, frame: np.ndarray) -> float:
        """Return speech probability in [0, 1] for a single 30 ms frame."""
        if frame.ndim != 1:
            raise ValueError("SileroVAD expects a 1-D mono frame")
        self._ensure_model()
        import torch  # type: ignore[import-not-found]

        with torch.no_grad():
            tensor = torch.from_numpy(frame.astype(np.float32))
            return float(self._model(tensor, self.sample_rate).item())
