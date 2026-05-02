"""DeepFilterNet 3 (full-band noise suppression) wrapper.

This module *deliberately* does the heavy import lazily so that the
algorithmic core of the package can be loaded without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DeepFilterNet3:
    """Thin wrapper around the `deepfilternet` package.

    Construction defers model loading; the first call to `process` materialises
    the underlying network. This keeps unit tests fast when no audio is
    actually processed.
    """

    sample_rate: int = 48_000
    _model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from df.enhance import enhance, init_df  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only with extras
            raise RuntimeError(
                "deepfilternet not installed; reinstall with `pip install -e poc[models]`"
            ) from exc
        model, df_state, _ = init_df()
        self._model = (model, df_state, enhance)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Run NS over a 48 kHz mono buffer and return the enhanced waveform."""
        if audio.ndim != 1:
            raise ValueError("DeepFilterNet3 expects a 1-D mono buffer")
        self._ensure_model()
        import torch  # type: ignore[import-not-found]

        model, df_state, enhance = self._model
        tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        enhanced = enhance(model, df_state, tensor)
        return enhanced.squeeze(0).cpu().numpy().astype(np.float32)
