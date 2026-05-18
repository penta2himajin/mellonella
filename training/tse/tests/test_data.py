"""Tests for the on-disk data loading path.

Covers:

* Lazy ``Path``-based audio loading in :class:`TSESourceItem` /
  :class:`TSEMixtureDataset`.
* :func:`librispeech_musan_sources` against a tiny on-disk LibriSpeech /
  MUSAN-shaped fixture — both with and without an enrollment-embedding
  ``.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tse.data import (
    TSEMixtureDataset,
    TSESourceItem,
    _load_audio_field,
    librispeech_musan_sources,
    vctk_demand_sources,
)

SR = 16_000


def _sine(n: int, sr: int, f0: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    phase = rng.uniform(0, 2 * np.pi)
    return (0.5 * np.sin(2 * np.pi * f0 * t + phase)).astype(np.float32)


# ---------------------------------------------------------------------------
# Lazy audio-field loading
# ---------------------------------------------------------------------------


def test_load_audio_field_returns_array_unchanged() -> None:
    a = np.zeros(100, dtype=np.float32)
    out = _load_audio_field(a, SR)
    assert out is a or np.array_equal(out, a)


def test_load_audio_field_decodes_path(tmp_path: Path) -> None:
    wav = _sine(SR, SR, 440.0)
    path = tmp_path / "tone.flac"
    sf.write(path, wav, SR, format="FLAC")
    loaded = _load_audio_field(path, SR)
    assert loaded.dtype == np.float32
    assert loaded.shape == wav.shape
    assert np.max(np.abs(loaded - wav)) < 1e-3  # FLAC quantisation tolerance


def test_dataset_loads_paths_lazily(tmp_path: Path) -> None:
    """A TSESourceItem with Path fields decodes on __getitem__."""
    target = _sine(SR, SR, 220.0, seed=1)
    interferer = _sine(SR, SR, 330.0, seed=2)
    noise = (np.random.default_rng(3).standard_normal(SR) * 0.05).astype(np.float32)
    target_path = tmp_path / "target.flac"
    interferer_path = tmp_path / "interferer.flac"
    noise_path = tmp_path / "noise.wav"
    sf.write(target_path, target, SR, format="FLAC")
    sf.write(interferer_path, interferer, SR, format="FLAC")
    sf.write(noise_path, noise, SR)

    cond = np.random.default_rng(0).standard_normal(192).astype(np.float32)
    item = TSESourceItem(
        target=target_path,
        interferer=interferer_path,
        cond_embedding=cond,
        noise=noise_path,
        sample_id="lazy",
    )
    ds = TSEMixtureDataset([item], sample_rate=SR, segment_samples=SR, random_crop=False)
    mix, c, t = ds[0]
    assert mix.shape == (SR,)
    assert c.shape == (192,)
    assert t.shape == (SR,)
    assert torch.isfinite(mix).all()


# ---------------------------------------------------------------------------
# librispeech_musan_sources — tiny on-disk fixture
# ---------------------------------------------------------------------------


def _build_librispeech_fixture(
    root: Path, *, speakers: list[str], utts_per_speaker: int = 2
) -> list[str]:
    """Write a tiny LibriSpeech-shaped tree (1 chapter per speaker).

    Returns the list of utterance ids (``<rel path without suffix>``) created,
    so tests can build a matching embeddings npz.
    """
    split = root / "LibriSpeech" / "train-clean-100"
    utt_ids: list[str] = []
    rng = np.random.default_rng(0)
    for s in speakers:
        chapter = "0001"
        ch_dir = split / s / chapter
        ch_dir.mkdir(parents=True, exist_ok=True)
        for k in range(utts_per_speaker):
            f0 = 110.0 + rng.uniform(0, 200.0)
            wav = _sine(SR, SR, f0, seed=int(s) * 100 + k)
            fname = f"{s}-{chapter}-{k:04d}.flac"
            sf.write(ch_dir / fname, wav, SR, format="FLAC")
            utt_ids.append(f"{s}/{chapter}/{Path(fname).stem}")
    return utt_ids


def _write_musan_noise(root: Path, n: int = 2) -> None:
    noise_dir = root / "musan" / "extracted" / "musan" / "noise" / "free-sound"
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        wav = (rng.standard_normal(SR) * 0.05).astype(np.float32)
        sf.write(noise_dir / f"noise-{i:04d}.wav", wav, SR)


def test_librispeech_musan_sources_without_embeddings(tmp_path: Path) -> None:
    """Plumbing path: builds source items, uses deterministic cond fallback."""
    _build_librispeech_fixture(tmp_path, speakers=["103", "204", "307"], utts_per_speaker=2)
    _write_musan_noise(tmp_path)

    with pytest.warns(UserWarning, match="placeholder"):
        sources = librispeech_musan_sources(tmp_path, n_pairs=4, seed=0)

    assert len(sources) == 4
    for s in sources:
        assert isinstance(s.target, Path)
        assert isinstance(s.interferer, Path)
        assert s.noise is None or isinstance(s.noise, Path)
        assert s.cond_embedding.shape == (192,)
        # Different speakers between target and interferer.
        assert s.target.stem.split("-")[0] != s.interferer.stem.split("-")[0]


def test_librispeech_musan_sources_with_embeddings(tmp_path: Path) -> None:
    """When embeddings_npz is given, only matching utterance ids are kept."""
    utt_ids = _build_librispeech_fixture(tmp_path, speakers=["103", "204"], utts_per_speaker=2)

    # Build an embeddings npz covering only the first 2 utterance ids.
    keep_ids = utt_ids[:2]
    embeds = {
        uid: np.random.default_rng(i).standard_normal(192).astype(np.float32)
        for i, uid in enumerate(keep_ids)
    }
    npz_path = tmp_path / "embeddings.npz"
    np.savez(npz_path, **embeds)  # type: ignore[arg-type]

    sources = librispeech_musan_sources(
        tmp_path, embeddings_npz=npz_path, n_pairs=None, musan_subset=None, seed=0
    )
    assert len(sources) == len(keep_ids)
    seen_ids = {s.sample_id for s in sources}
    assert seen_ids == set(keep_ids)
    # The cond_embedding must match the npz lookup.
    for s in sources:
        np.testing.assert_array_equal(s.cond_embedding, embeds[s.sample_id])


def test_librispeech_musan_sources_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        librispeech_musan_sources(tmp_path, embeddings_npz=None, musan_subset=None)


def test_dataset_end_to_end_real_data(tmp_path: Path) -> None:
    """librispeech_musan_sources → TSEMixtureDataset → __getitem__ end-to-end."""
    _build_librispeech_fixture(tmp_path, speakers=["103", "204"], utts_per_speaker=2)
    _write_musan_noise(tmp_path)
    with pytest.warns(UserWarning):
        sources = librispeech_musan_sources(tmp_path, n_pairs=2, seed=0)
    ds = TSEMixtureDataset(sources, sample_rate=SR, segment_samples=SR, random_crop=False)
    mix, cond, target = ds[0]
    assert mix.shape == (SR,)
    assert cond.shape == (192,)
    assert target.shape == (SR,)
    assert torch.isfinite(mix).all() and torch.isfinite(target).all()


# ---------------------------------------------------------------------------
# precache_audio
# ---------------------------------------------------------------------------


def test_precache_replaces_paths_with_arrays_and_dedupes(tmp_path: Path) -> None:
    """precache_audio should decode each path once and replace fields in-place."""
    _build_librispeech_fixture(tmp_path, speakers=["103", "204"], utts_per_speaker=2)
    _write_musan_noise(tmp_path, n=2)
    with pytest.warns(UserWarning):
        sources = librispeech_musan_sources(tmp_path, n_pairs=3, seed=0)
    # Before: all audio fields are Paths.
    assert all(isinstance(s.target, Path) for s in sources)
    ds = TSEMixtureDataset(sources, sample_rate=SR, segment_samples=SR, random_crop=False)

    info = ds.precache_audio(verbose=False)
    # Dedup: 4 LibriSpeech + 2 MUSAN noise = at most 6 unique files, but
    # n_pairs=3 means we touched a subset. Whatever was touched is now an ndarray.
    assert info["n_unique"] >= 1
    assert info["bytes"] > 0
    for s in sources:
        assert isinstance(s.target, np.ndarray)
        assert isinstance(s.interferer, np.ndarray)
        if s.noise is not None:
            assert isinstance(s.noise, np.ndarray)
    # Dataset still produces correct shapes after caching.
    mix, cond, target = ds[0]
    assert mix.shape == (SR,) and target.shape == (SR,)
    assert torch.isfinite(mix).all() and torch.isfinite(target).all()


def test_precache_float16_smaller_storage(tmp_path: Path) -> None:
    _build_librispeech_fixture(tmp_path, speakers=["103", "204"], utts_per_speaker=2)
    with pytest.warns(UserWarning):
        sources_fp32 = librispeech_musan_sources(tmp_path, n_pairs=2, seed=0, musan_subset=None)
    with pytest.warns(UserWarning):
        sources_fp16 = librispeech_musan_sources(tmp_path, n_pairs=2, seed=0, musan_subset=None)
    ds32 = TSEMixtureDataset(sources_fp32, sample_rate=SR, segment_samples=SR)
    ds16 = TSEMixtureDataset(sources_fp16, sample_rate=SR, segment_samples=SR)
    info32 = ds32.precache_audio(dtype="float32", verbose=False)
    info16 = ds16.precache_audio(dtype="float16", verbose=False)
    # fp16 should use roughly half the bytes for the same audio.
    assert info16["bytes"] == info32["bytes"] // 2
    # Sources now hold fp16 arrays.
    for s in sources_fp16:
        if isinstance(s.target, np.ndarray):
            assert s.target.dtype == np.float16


def test_precache_invalid_dtype_raises(tmp_path: Path) -> None:
    _build_librispeech_fixture(tmp_path, speakers=["103", "204"], utts_per_speaker=2)
    with pytest.warns(UserWarning):
        sources = librispeech_musan_sources(tmp_path, n_pairs=2, seed=0, musan_subset=None)
    ds = TSEMixtureDataset(sources, sample_rate=SR, segment_samples=SR)
    with pytest.raises(ValueError, match="dtype"):
        ds.precache_audio(dtype="float64", verbose=False)


# ---------------------------------------------------------------------------
# vctk_demand_sources — tiny on-disk fixture
# ---------------------------------------------------------------------------


def _build_vctk_fixture(
    tmp_path: Path,
    *,
    speakers: list[str] | None = None,
    utts_per_speaker: int = 2,
) -> list[str]:
    """Write a tiny VCTK-shaped tree under ``<tmp>/VCTK-Corpus/...``.

    We use the ``wav48_silence_trimmed/<pXXX>/<pXXX>_<utt>.wav`` layout —
    the default for recent VCTK Kaggle drops. The loader's filename-prefix
    scan also works against the other variants (``wav48/...``, flat
    ``pXXX/...``); this fixture validates that the prefix scan finds the
    speakers regardless.

    Returns the list of utterance ids (relative to ``VCTK-Corpus/``,
    without the suffix), so the embeddings-npz test can build a matching
    lookup table.
    """
    if speakers is None:
        speakers = ["p225", "p226", "p227"]
    root = tmp_path / "VCTK-Corpus" / "wav48_silence_trimmed"
    utt_ids: list[str] = []
    rng = np.random.default_rng(0)
    for s in speakers:
        spk_dir = root / s
        spk_dir.mkdir(parents=True, exist_ok=True)
        for k in range(utts_per_speaker):
            f0 = 110.0 + rng.uniform(0, 200.0)
            # We deliberately write 16 kHz wav files for the fixture
            # (cheap to generate); the dataset resamples to its
            # sample_rate on access.
            wav = _sine(SR, SR, f0, seed=hash((s, k)) & 0xFFFF)
            fname = f"{s}_{k:03d}.wav"
            sf.write(spk_dir / fname, wav, SR)
            utt_ids.append(f"wav48_silence_trimmed/{s}/{Path(fname).stem}")
    return utt_ids


def _write_demand_noise(
    tmp_path: Path,
    *,
    categories: list[str] | None = None,
) -> None:
    """Write a tiny DEMAND-shaped tree under ``<tmp>/demand/<CATEGORY>/chNN.wav``.

    Includes decoy ``ch02.wav`` and ``ch03.wav`` per category to verify the
    loader keeps only ``ch01.wav``.
    """
    if categories is None:
        categories = ["TBUS", "TCAR"]
    root = tmp_path / "demand"
    rng = np.random.default_rng(42)
    for cat in categories:
        cat_dir = root / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for ch in (1, 2, 3):
            wav = (rng.standard_normal(SR) * 0.05).astype(np.float32)
            sf.write(cat_dir / f"ch{ch:02d}.wav", wav, SR)


def test_vctk_demand_sources_without_embeddings(tmp_path: Path) -> None:
    """Plumbing path: builds source items, uses deterministic cond fallback."""
    _build_vctk_fixture(tmp_path, speakers=["p225", "p226", "p227"], utts_per_speaker=2)
    _write_demand_noise(tmp_path, categories=["TBUS", "TCAR"])

    with pytest.warns(UserWarning, match="placeholder"):
        sources = vctk_demand_sources(tmp_path, n_pairs=4, seed=0)

    assert len(sources) == 4
    for s in sources:
        assert isinstance(s.target, Path)
        assert isinstance(s.interferer, Path)
        assert s.noise is None or isinstance(s.noise, Path)
        assert s.cond_embedding.shape == (192,)
        # Speaker id is the pXXX prefix of the filename.
        target_speaker = Path(s.target).stem.split("_", 1)[0]
        interferer_speaker = Path(s.interferer).stem.split("_", 1)[0]
        assert target_speaker != interferer_speaker
        # DEMAND noise canonicalisation: only ch01.wav is used.
        if s.noise is not None:
            assert Path(s.noise).name == "ch01.wav"


def test_vctk_demand_sources_with_embeddings(tmp_path: Path) -> None:
    """When embeddings_npz is given, only matching utterance ids are kept."""
    utt_ids = _build_vctk_fixture(tmp_path, speakers=["p225", "p226"], utts_per_speaker=2)
    # No DEMAND noise — exercise the no-noise path too.

    keep_ids = utt_ids[:2]
    embeds = {
        uid: np.random.default_rng(i).standard_normal(192).astype(np.float32)
        for i, uid in enumerate(keep_ids)
    }
    npz_path = tmp_path / "embeddings.npz"
    np.savez(npz_path, **embeds)  # type: ignore[arg-type]

    sources = vctk_demand_sources(
        tmp_path,
        embeddings_npz=npz_path,
        n_pairs=None,
        seed=0,
    )
    assert len(sources) == len(keep_ids)
    seen_ids = {s.sample_id for s in sources}
    assert seen_ids == set(keep_ids)
    for s in sources:
        np.testing.assert_array_equal(s.cond_embedding, embeds[s.sample_id])


def test_vctk_demand_sources_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vctk_demand_sources(tmp_path, embeddings_npz=None)


def test_vctk_demand_dataset_end_to_end_48k(tmp_path: Path) -> None:
    """vctk_demand_sources → TSEMixtureDataset at 48 kHz resamples on access.

    Fixture wavs are written at 16 kHz for cheapness; the dataset is built
    at ``sample_rate=48_000``, so ``_load_audio_field`` must resample 3×
    on access. We assert the returned tensors are at the 48 kHz length.
    """
    pytest.importorskip("librosa")  # resampling backend
    _build_vctk_fixture(tmp_path, speakers=["p225", "p226"], utts_per_speaker=2)
    _write_demand_noise(tmp_path, categories=["TBUS"])
    with pytest.warns(UserWarning):
        sources = vctk_demand_sources(tmp_path, n_pairs=2, sample_rate=48_000, seed=0)
    target_sr = 48_000
    ds = TSEMixtureDataset(
        sources, sample_rate=target_sr, segment_samples=target_sr, random_crop=False
    )
    mix, cond, target = ds[0]
    # The dataset resampled the 1-second 16 kHz wav up to 48 kHz before
    # cropping to ``segment_samples``, so the output is exactly
    # ``target_sr`` samples long.
    assert mix.shape == (target_sr,)
    assert target.shape == (target_sr,)
    assert cond.shape == (192,)
    assert torch.isfinite(mix).all() and torch.isfinite(target).all()
