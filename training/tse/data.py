"""On-the-fly target speaker extraction mixture dataset.

A :class:`TSEMixtureDataset` item is a ``(target, interferer, noise,
cond_embedding)`` source bundle. Each ``__getitem__`` mixes the target with
the interferer (and optional noise) at a sampled power ratio and returns
``(mixture, cond_embedding, clean_target)`` ready for the training loop.

The dataset deliberately does *not* hard-code where the audio comes from:
it takes an explicit list of :class:`TSESourceItem`. This lets the smoke /
overfit tests run with a tiny in-memory synthetic fixture set
(:func:`synthetic_fixture_dataset`) with **no datasets downloaded**, while
the real LibriSpeech / LibriMix + MUSAN loading path
(:func:`librispeech_musan_sources` — currently a documented stub) plugs the
same :class:`TSESourceItem` list in from disk.

Mixing reuses the ratio convention from
``bench/mellonella_bench/scenarios/scenario_4.py`` (``mix_at_ratio``):
positive dB == target louder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Source bundle
# ---------------------------------------------------------------------------


@dataclass
class TSESourceItem:
    """One mixing source bundle.

    Attributes
    ----------
    target:
        Clean target-speaker waveform, 1-D float32 array at the dataset rate.
    interferer:
        Competing-speaker waveform, 1-D float32.
    cond_embedding:
        Frozen 192-dim ECAPA enrollment embedding for the target speaker,
        1-D float32. In the synthetic fixtures this is just a fixed random
        vector — no ECAPA model is needed.
    noise:
        Optional background noise waveform, 1-D float32. ``None`` disables
        the noise term.
    sample_id:
        Human-readable identifier (for logging / debugging).
    """

    target: np.ndarray
    interferer: np.ndarray
    cond_embedding: np.ndarray
    noise: np.ndarray | None = None
    sample_id: str = ""


# ---------------------------------------------------------------------------
# Mixing primitives (consistent with bench scenario_4.mix_at_ratio)
# ---------------------------------------------------------------------------


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2))


def _scale_to_ratio(target: np.ndarray, other: np.ndarray, ratio_db: float) -> np.ndarray:
    """Scale ``other`` so ``power(target) / power(scaled) == 10**(ratio_db/10)``."""
    tp = _power(target)
    op = _power(other)
    if tp == 0.0 or op == 0.0:
        return other.astype(np.float32, copy=False)
    target_other_power = tp / (10.0 ** (ratio_db / 10.0))
    scale = float(np.sqrt(target_other_power / op))
    return (scale * other).astype(np.float32)


def _fit_length(x: np.ndarray, length: int) -> np.ndarray:
    """Truncate or zero-pad ``x`` to exactly ``length`` samples."""
    if x.size >= length:
        return x[:length].astype(np.float32, copy=False)
    return np.concatenate([x, np.zeros(length - x.size, dtype=np.float32)])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TSEMixtureDataset(Dataset):
    """On-the-fly TSE mixture dataset.

    Parameters
    ----------
    sources:
        Sequence of :class:`TSESourceItem` bundles.
    segment_samples:
        Fixed output segment length. Each item is cropped (deterministically
        per index, unless ``random_crop``) to this many samples.
    target_interferer_ratio_db:
        ``(low, high)`` range the target-to-interferer ratio is sampled from.
    target_noise_ratio_db:
        ``(low, high)`` range the target-to-noise ratio is sampled from
        (only used when a source has a ``noise`` track).
    random_crop:
        When ``True``, crop offsets and mix ratios are sampled randomly per
        access (training). When ``False`` they are derived deterministically
        from the index (smoke / overfit / eval reproducibility).
    seed:
        Base RNG seed.
    """

    def __init__(
        self,
        sources: Sequence[TSESourceItem],
        *,
        segment_samples: int = 16_000,
        target_interferer_ratio_db: tuple[float, float] = (-5.0, 5.0),
        target_noise_ratio_db: tuple[float, float] = (5.0, 20.0),
        random_crop: bool = True,
        seed: int = 0,
    ) -> None:
        if len(sources) == 0:
            raise ValueError("TSEMixtureDataset needs at least one source item")
        self.sources = list(sources)
        self.segment_samples = segment_samples
        self.ti_ratio_db = target_interferer_ratio_db
        self.tn_ratio_db = target_noise_ratio_db
        self.random_crop = random_crop
        self.seed = seed

    def __len__(self) -> int:
        return len(self.sources)

    def _rng(self, index: int) -> np.random.Generator:
        if self.random_crop:
            # Fresh entropy per access so epochs see different crops/ratios.
            return np.random.default_rng()
        return np.random.default_rng(self.seed + index)

    def _crop(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        seg = self.segment_samples
        if x.size <= seg:
            return _fit_length(x, seg)
        start = int(rng.integers(0, x.size - seg + 1)) if self.random_crop else 0
        return x[start : start + seg].astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self.sources[index]
        rng = self._rng(index)

        target = self._crop(np.asarray(src.target, dtype=np.float32), rng)
        interferer = _fit_length(
            np.asarray(src.interferer, dtype=np.float32), self.segment_samples
        )

        ti_db = float(rng.uniform(*self.ti_ratio_db))
        mixture = target + _scale_to_ratio(target, interferer, ti_db)

        if src.noise is not None:
            noise = _fit_length(np.asarray(src.noise, dtype=np.float32), self.segment_samples)
            tn_db = float(rng.uniform(*self.tn_ratio_db))
            mixture = mixture + _scale_to_ratio(target, noise, tn_db)

        cond = np.asarray(src.cond_embedding, dtype=np.float32)
        return (
            torch.from_numpy(mixture.astype(np.float32)),
            torch.from_numpy(cond),
            torch.from_numpy(target.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Synthetic fixture set — no downloads required
# ---------------------------------------------------------------------------


def _synth_voice(
    rng: np.random.Generator, n: int, sr: int, f0: float
) -> np.ndarray:
    """Deterministic harmonic-stack 'voice' with light AM modulation."""
    t = np.arange(n) / sr
    wave = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 6):
        phase = rng.uniform(0, 2 * np.pi)
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t + phase)
    am = 0.5 * (1.0 + 0.3 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi)))
    wave = wave * am
    peak = float(np.max(np.abs(wave))) or 1.0
    return (wave / peak * 0.9).astype(np.float32)


def synthetic_fixture_dataset(
    n: int = 4,
    *,
    sample_rate: int = 16_000,
    duration_sec: float = 1.0,
    cond_dim: int = 192,
    with_noise: bool = True,
    segment_samples: int | None = None,
    seed: int = 1234,
) -> TSEMixtureDataset:
    """Build an in-memory synthetic :class:`TSEMixtureDataset`.

    Each item has a distinct-f0 harmonic 'target', a different-f0
    'interferer', optional white 'noise', and a fixed random 192-dim
    conditioning vector. Nothing is downloaded — this is what the smoke and
    overfit tests run on. ``random_crop`` is off so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate * duration_sec)
    sources: list[TSESourceItem] = []
    for i in range(n):
        f0_t = 110.0 + 25.0 * i
        f0_i = 190.0 + 25.0 * i
        target = _synth_voice(rng, n_samples, sample_rate, f0_t)
        interferer = _synth_voice(rng, n_samples, sample_rate, f0_i)
        noise = (
            (rng.standard_normal(n_samples) * 0.1).astype(np.float32) if with_noise else None
        )
        cond = rng.standard_normal(cond_dim).astype(np.float32)
        sources.append(
            TSESourceItem(
                target=target,
                interferer=interferer,
                cond_embedding=cond,
                noise=noise,
                sample_id=f"synth_{i}",
            )
        )
    return TSEMixtureDataset(
        sources,
        segment_samples=segment_samples or n_samples,
        random_crop=False,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Real-data loading path — documented stub (Phase 3)
# ---------------------------------------------------------------------------


def librispeech_musan_sources(
    data_dir: Path | None = None,
    *,
    split: str = "train-clean-100",
    n_pairs: int | None = None,
    sample_rate: int = 16_000,
    embeddings_npz: Path | None = None,
) -> list[TSESourceItem]:
    """Build :class:`TSESourceItem` bundles from local LibriSpeech + MUSAN.

    **Phase 3 stub.** The structure is wired; the actual disk loading is a
    TODO. The intended implementation:

    1. ``data_dir`` defaults to ``bench``'s ``default_data_dir()`` (honours
       ``$MELLONELLA_DATA_DIR``). LibriSpeech ``train-clean-100`` lives under
       ``data_dir / "librispeech" / split``; MUSAN noise under
       ``data_dir / "musan"`` (see ``bench/mellonella_bench/datasets/musan.py``
       for the fetch + subset helpers to reuse).
    2. Enumerate per-speaker utterances. For each *target* utterance pick a
       different-speaker *interferer* utterance and (optionally) a MUSAN
       noise clip — reuse ``bench``'s loaders rather than re-implementing.
    3. The frozen 192-dim ECAPA enrollment embedding per target utterance is
       precomputed offline by ``prepare_enrollment_embeddings.py`` into an
       ``.npz``; ``embeddings_npz`` points at it and we look each one up by
       utterance id. (No ECAPA model is loaded here.)
    4. Resample to ``sample_rate`` if needed (LibriSpeech is 16 kHz natively;
       the 48 kHz production path uses VCTK + DEMAND instead — a config swap,
       not a code change here).

    Until that lands this raises :class:`NotImplementedError` so callers
    fail loudly rather than silently training on nothing.
    """
    raise NotImplementedError(
        "librispeech_musan_sources is a Phase 3 stub — wire LibriSpeech/MUSAN "
        "loading via bench dataset infra, then look up ECAPA embeddings from "
        "the prepare_enrollment_embeddings.py .npz. For now use "
        "synthetic_fixture_dataset() for smoke/overfit."
    )
