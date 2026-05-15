"""On-the-fly target speaker extraction mixture dataset.

A :class:`TSEMixtureDataset` item is a ``(target, interferer, noise,
cond_embedding)`` source bundle. Each ``__getitem__`` mixes the target with
the interferer (and optional noise) at a sampled power ratio and returns
``(mixture, cond_embedding, clean_target)`` ready for the training loop.

The dataset deliberately does *not* hard-code where the audio comes from:
it takes an explicit list of :class:`TSESourceItem`, whose audio fields may
be either pre-loaded ``np.ndarray``\\ s (synthetic / smoke fixtures) or
:class:`pathlib.Path`\\ s loaded lazily on access (real data, so a 6 GB
LibriSpeech subset never has to live in RAM).

The two paths in:

* :func:`synthetic_fixture_dataset` — tiny in-memory harmonic-stack
  'voices'. Nothing is downloaded; this is what the smoke and overfit
  tests run on.
* :func:`librispeech_musan_sources` — real LibriSpeech + MUSAN from disk,
  with per-target ECAPA enrollment embeddings looked up from the
  ``prepare_enrollment_embeddings.py`` ``.npz``.

Mixing reuses the ratio convention from
``bench/mellonella_bench/scenarios/scenario_4.py`` (``mix_at_ratio``):
positive dB == target louder.
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
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
        Clean target-speaker audio. Either a 1-D ``float32`` array at the
        dataset rate, or a :class:`~pathlib.Path` to an audio file decoded
        lazily on access.
    interferer:
        Competing-speaker audio, same type rules as ``target``.
    cond_embedding:
        Frozen 192-dim ECAPA enrollment embedding for the target speaker,
        1-D float32. In the synthetic fixtures this is just a fixed random
        vector — no ECAPA model is needed.
    noise:
        Optional background noise (array or Path). ``None`` disables the
        noise term.
    sample_id:
        Human-readable identifier (for logging / debugging).
    """

    target: np.ndarray | Path
    interferer: np.ndarray | Path
    cond_embedding: np.ndarray
    noise: np.ndarray | Path | None = None
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


def _load_audio_field(field: np.ndarray | Path, target_sr: int) -> np.ndarray:
    """Materialise an audio field to a mono ``float32`` array.

    Arrays are returned as-is (no copy). Paths are decoded with
    ``soundfile`` and resampled to ``target_sr`` if needed.
    """
    if isinstance(field, np.ndarray):
        return field
    if not isinstance(field, Path):
        field = Path(field)
    audio, sr = sf.read(str(field), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        # librosa is an optional dep — only required when actually resampling.
        try:
            import librosa  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - librosa is in the onnx extra
            raise RuntimeError(
                f"{field}: {sr} Hz, need {target_sr} Hz; install librosa to resample"
            ) from exc
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


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
        sample_rate: int = 16_000,
        segment_samples: int = 16_000,
        target_interferer_ratio_db: tuple[float, float] = (-5.0, 5.0),
        target_noise_ratio_db: tuple[float, float] = (5.0, 20.0),
        random_crop: bool = True,
        seed: int = 0,
    ) -> None:
        if len(sources) == 0:
            raise ValueError("TSEMixtureDataset needs at least one source item")
        self.sources = list(sources)
        self.sample_rate = sample_rate
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

        target_raw = _load_audio_field(src.target, self.sample_rate)
        interferer_raw = _load_audio_field(src.interferer, self.sample_rate)
        target = self._crop(target_raw.astype(np.float32, copy=False), rng)
        interferer = _fit_length(interferer_raw, self.segment_samples)

        ti_db = float(rng.uniform(*self.ti_ratio_db))
        mixture = target + _scale_to_ratio(target, interferer, ti_db)

        if src.noise is not None:
            noise_raw = _load_audio_field(src.noise, self.sample_rate)
            noise = _fit_length(noise_raw, self.segment_samples)
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


def _synth_voice(rng: np.random.Generator, n: int, sr: int, f0: float) -> np.ndarray:
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
        noise = (rng.standard_normal(n_samples) * 0.1).astype(np.float32) if with_noise else None
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
        sample_rate=sample_rate,
        segment_samples=segment_samples or n_samples,
        random_crop=False,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Real-data loading path — LibriSpeech + MUSAN
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    """Mirror ``bench``'s ``default_data_dir`` without importing bench.

    Honours ``$MELLONELLA_DATA_DIR``; defaults to ``./data`` relative to
    the current working directory.
    """
    raw = os.environ.get("MELLONELLA_DATA_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd() / "data"


def _scan_musan_noise(data_dir: Path, musan_subset: str) -> list[Path]:
    """Find MUSAN noise ``.wav`` files under standard ``bench`` layouts.

    ``bench/mellonella_bench/datasets/musan.py`` extracts under
    ``$MELLONELLA_DATA_DIR/musan/extracted/musan/<category>``; it also builds
    a deterministic ``subset/`` directory. We look in both, plus a flat
    ``musan/<category>`` layout for ad-hoc setups.
    """
    musan_root = data_dir / "musan"
    candidates = [
        musan_root / "extracted" / "musan" / musan_subset,
        musan_root / "subset" / musan_subset,
        musan_root / musan_subset,
    ]
    for c in candidates:
        if c.is_dir():
            files = sorted(c.rglob("*.wav"))
            if files:
                return files
    return []


def _deterministic_cond(speaker_id: str, cond_dim: int) -> np.ndarray:
    """Deterministic per-speaker random conditioning vector (no ECAPA needed).

    Used as the fallback when ``embeddings_npz`` is omitted, so the data
    plumbing can be smoke-tested without the ECAPA ONNX. Seeded from a
    stable SHA-256 of the speaker id so the same speaker always gets the
    same vector — distinct speakers get distinct, deterministic vectors.
    This vector carries **no** speaker-identity information, so a model
    trained on it cannot actually condition on the target — use a real
    ``embeddings_npz`` for proper training.
    """
    seed = int(hashlib.sha256(speaker_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(cond_dim).astype(np.float32)


def librispeech_musan_sources(
    data_dir: Path | None = None,
    *,
    split: str = "train-clean-100",
    librispeech_root: str = "LibriSpeech",
    musan_subset: str | None = "noise",
    n_pairs: int | None = None,
    sample_rate: int = 16_000,
    embeddings_npz: Path | None = None,
    cond_dim: int = 192,
    seed: int = 0,
) -> list[TSESourceItem]:
    """Build :class:`TSESourceItem` bundles from local LibriSpeech + MUSAN.

    Layout expected under ``data_dir`` (defaults to ``$MELLONELLA_DATA_DIR``
    or ``./data``):

    ::

        <data_dir>/<librispeech_root>/<split>/<speaker>/<chapter>/<utt>.flac
        <data_dir>/musan/...                       # any of the layouts
                                                   # bench.datasets.musan produces

    For each LibriSpeech utterance the function picks a different-speaker
    interferer utterance and (when ``musan_subset`` is given and files are
    found) a random MUSAN noise clip. Audio is **not** loaded here — only
    paths — so the source list stays small; :class:`TSEMixtureDataset`
    decodes lazily per access.

    ``embeddings_npz``
        Path to the ``.npz`` written by
        :mod:`tse.prepare_enrollment_embeddings`, keyed by utterance id
        (the LibriSpeech relative path without the ``.flac`` suffix). When
        provided, only utterances with a precomputed embedding are kept.
        When omitted, a deterministic per-speaker placeholder is used
        (plumbing only — carries no speaker information; a model trained
        on it cannot generalise).

    ``n_pairs``
        Optional cap on the number of returned items.

    ``sample_rate``
        Stored on the eventual :class:`TSEMixtureDataset`; LibriSpeech is
        natively 16 kHz so no resampling is performed at this rate.
    """
    data_dir = data_dir if data_dir is not None else _default_data_dir()
    libri_split = data_dir / librispeech_root / split
    if not libri_split.is_dir():
        raise FileNotFoundError(f"LibriSpeech split not found: {libri_split}")

    # LibriSpeech filename: <speaker>-<chapter>-<utt>.flac → speaker = stem[:3].
    by_speaker: dict[str, list[Path]] = {}
    for flac in libri_split.rglob("*.flac"):
        speaker_id = flac.stem.split("-", 1)[0]
        by_speaker.setdefault(speaker_id, []).append(flac)
    if len(by_speaker) < 2:
        raise RuntimeError(f"need >= 2 speakers under {libri_split}, found {len(by_speaker)}")

    embeddings: dict[str, np.ndarray] | None = None
    if embeddings_npz is not None:
        loaded = np.load(embeddings_npz)
        embeddings = {k: loaded[k].astype(np.float32) for k in loaded.files}
        print(
            f"[data] loaded {len(embeddings)} enrollment embeddings from {embeddings_npz}",
            file=sys.stderr,
        )
    else:
        warnings.warn(
            "librispeech_musan_sources called without embeddings_npz — using a "
            "deterministic per-speaker placeholder vector. This is for plumbing "
            "tests only; a model trained against it cannot condition on the "
            "target. Pass embeddings_npz= for real training.",
            stacklevel=2,
        )

    noise_files: list[Path] = []
    if musan_subset is not None:
        noise_files = _scan_musan_noise(data_dir, musan_subset)
        if not noise_files:
            warnings.warn(
                f"no MUSAN noise files found under {data_dir / 'musan'!s}; "
                f"training without noise augmentation.",
                stacklevel=2,
            )

    rng = np.random.default_rng(seed)
    speakers = sorted(by_speaker.keys())
    for s in speakers:
        by_speaker[s].sort()

    # Flat shuffled (speaker, target_path) list, then pair each with a
    # different-speaker interferer.
    all_utts: list[tuple[str, Path]] = [(s, p) for s in speakers for p in by_speaker[s]]
    rng.shuffle(all_utts)  # numpy default_rng shuffles lists in place

    items: list[TSESourceItem] = []
    for target_speaker, target_path in all_utts:
        # Look up cond embedding by utterance id (relative path no suffix).
        utt_id = target_path.relative_to(libri_split).with_suffix("").as_posix()
        if embeddings is not None:
            if utt_id not in embeddings:
                continue
            cond = embeddings[utt_id]
        else:
            cond = _deterministic_cond(target_speaker, cond_dim)

        # Pick a different-speaker interferer.
        other_speaker = target_speaker
        while other_speaker == target_speaker:
            other_speaker = speakers[int(rng.integers(0, len(speakers)))]
        interferer_choices = by_speaker[other_speaker]
        interferer_path = interferer_choices[int(rng.integers(0, len(interferer_choices)))]

        noise_path: Path | None = None
        if noise_files:
            noise_path = noise_files[int(rng.integers(0, len(noise_files)))]

        items.append(
            TSESourceItem(
                target=target_path,
                interferer=interferer_path,
                cond_embedding=cond,
                noise=noise_path,
                sample_id=utt_id,
            )
        )
        if n_pairs is not None and len(items) >= n_pairs:
            break

    if not items:
        msg = f"no source items built from {libri_split}"
        if embeddings is not None:
            msg += " (utterance ids in embeddings_npz did not match any audio file)"
        raise RuntimeError(msg)
    # sample_rate is plumbed through for documentation and is consumed by the
    # downstream TSEMixtureDataset (which does the actual resampling).
    _ = sample_rate
    return items
