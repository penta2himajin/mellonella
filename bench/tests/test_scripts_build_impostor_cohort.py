"""Tests for ``scripts/build_impostor_cohort.py``.

The script lives outside the bench package and adjusts ``sys.path`` to
pick up both ``poc`` and ``bench``; tests load it via ``importlib`` to
side-step that dance and inject a mock embedder so the suite never has
to load ECAPA-TDNN. Synthetic CommonVoice-shape manifests are written
to ``tmp_path`` per case — no real HF download required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.datasets.commonvoice import CommonVoiceClip, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_impostor_cohort.py"
EMBEDDING_DIM = 192


def _import_script():
    spec = importlib.util.spec_from_file_location("build_impostor_cohort", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_impostor_cohort", module)
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    manifest_dir: Path,
    *,
    language: str,
    n_speakers: int = 3,
    seconds_per_clip: float = 4.0,
    clips_per_speaker: int = 2,
    sr: int = 16_000,
) -> Path:
    """Synthesise a CommonVoice-shape subset directory."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(seconds_per_clip * sr)
    rng = np.random.default_rng(hash(language) % 2**16)
    clips: list[CommonVoiceClip] = []
    for spk_idx in range(n_speakers):
        speaker_id = f"{language}_spk{spk_idx:02d}"
        spk_dir = manifest_dir / speaker_id
        spk_dir.mkdir(parents=True, exist_ok=True)
        for clip_idx in range(clips_per_speaker):
            t = np.arange(n_samples) / sr
            base = 220.0 + 110.0 * spk_idx
            audio = (
                0.4 * np.sin(2 * np.pi * base * t) + 0.05 * rng.standard_normal(n_samples)
            ).astype(np.float32)
            rel = Path(speaker_id) / f"{clip_idx:03d}.wav"
            sf.write(str(manifest_dir / rel), audio, sr)
            clips.append(
                CommonVoiceClip(
                    language=language,
                    speaker_id=speaker_id,
                    clip_path=rel,
                    sentence="",
                )
            )
    manifest_path = manifest_dir / "manifest.csv"
    write_manifest(clips, manifest_path)
    return manifest_path


def _deterministic_embed(audio: np.ndarray) -> np.ndarray:
    """Produce a 192-dim embedding from audio's first few summary statistics.

    Two different audio buffers will land on different embeddings, which
    is enough for the cohort tests (we just need distinct vectors per
    speaker, not realistic ECAPA distances).
    """
    rng = np.random.default_rng(int(abs(audio.mean()) * 1e6) % 2**32)
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


def test_manifest_spec_parses_lang_path():
    mod = _import_script()
    spec = mod.ManifestSpec.parse("ja=/tmp/foo/manifest.csv")
    assert spec.language == "ja"
    assert spec.manifest == Path("/tmp/foo/manifest.csv")


def test_l2_normalize_returns_unit_vector():
    mod = _import_script()
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    out = mod.l2_normalize(v)
    assert pytest.approx(float(np.linalg.norm(out)), abs=1e-6) == 1.0


def test_l2_normalize_handles_zero_vector():
    mod = _import_script()
    v = np.zeros(8, dtype=np.float32)
    out = mod.l2_normalize(v)
    assert out.shape == v.shape
    assert float(np.linalg.norm(out)) == 0.0


def test_select_speakers_picks_top_k_by_audio_length(tmp_path):
    mod = _import_script()
    manifest = _write_manifest(tmp_path / "en", language="en", n_speakers=3, clips_per_speaker=3)
    selections = mod.select_speakers_for_language(manifest, per_language=2, min_seconds=5.0)
    assert len(selections) == 2
    # All survivors should have the same audio length (3 clips × 4 s each)
    for spk_id, audio in selections:
        assert spk_id.startswith("en_spk")
        assert audio.size == int(3 * 4.0 * 16_000)


def test_select_speakers_raises_on_empty_manifest(tmp_path):
    mod = _import_script()
    # 1-speaker manifest with too-short clips → load_speakers_from_manifest
    # filters everything out → script should raise.
    manifest = _write_manifest(
        tmp_path / "en", language="en", n_speakers=1, clips_per_speaker=1, seconds_per_clip=1.0
    )
    with pytest.raises(RuntimeError):
        mod.select_speakers_for_language(manifest, per_language=2, min_seconds=5.0)


def test_build_cohort_concatenates_languages(tmp_path):
    mod = _import_script()
    en = _write_manifest(tmp_path / "en", language="en")
    de = _write_manifest(tmp_path / "de", language="de")
    specs = [
        mod.ManifestSpec(language="en", manifest=en),
        mod.ManifestSpec(language="de", manifest=de),
    ]
    embeddings, languages, speakers = mod.build_cohort(
        specs, per_language=2, embed_fn=_deterministic_embed
    )
    assert embeddings.shape == (4, EMBEDDING_DIM)
    assert languages == ["en", "en", "de", "de"]
    assert all(s.startswith(("en_spk", "de_spk")) for s in speakers)
    # All vectors are L2-normalised
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_cohort_rejects_zero_per_language(tmp_path):
    mod = _import_script()
    en = _write_manifest(tmp_path / "en", language="en")
    with pytest.raises(ValueError):
        mod.build_cohort(
            [mod.ManifestSpec(language="en", manifest=en)],
            per_language=0,
            embed_fn=_deterministic_embed,
        )


def test_save_and_load_roundtrip(tmp_path):
    mod = _import_script()
    embeddings = np.random.default_rng(0).standard_normal((4, EMBEDDING_DIM)).astype(np.float32)
    languages = ["en", "en", "de", "ja"]
    speaker_ids = ["A", "B", "C", "D"]
    out = tmp_path / "cohort.npz"
    mod.save_cohort(embeddings, languages, speaker_ids, out)
    assert out.exists()
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    summary = json.loads(sidecar.read_text())
    assert summary["n_embeddings"] == 4
    assert summary["embedding_dim"] == EMBEDDING_DIM
    assert summary["per_language_counts"] == {"de": 1, "en": 2, "ja": 1}
    # New diagnostic field: per-language ordered speaker_id list.
    assert summary["selected_speakers"] == {
        "de": ["C"],
        "en": ["A", "B"],
        "ja": ["D"],
    }

    loaded_emb, loaded_langs, loaded_spks = mod.load_cohort(out)
    assert np.allclose(loaded_emb, embeddings)
    assert loaded_langs == languages
    assert loaded_spks == speaker_ids


def test_save_cohort_rejects_wrong_embedding_dim(tmp_path):
    mod = _import_script()
    bad = np.zeros((2, EMBEDDING_DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="192"):
        mod.save_cohort(bad, ["en", "en"], ["A", "B"], tmp_path / "bad.npz")


def test_main_end_to_end(tmp_path):
    """Exercise the CLI path with a fake embedder injected via monkeypatching.

    main() doesn't accept an embed_fn arg (that's a code-path optimisation
    for the build_cohort helper). For the CLI test we monkey-patch the
    EcapaTdnn import inside the script.
    """
    mod = _import_script()
    en = _write_manifest(tmp_path / "en", language="en")
    de = _write_manifest(tmp_path / "de", language="de")
    output = tmp_path / "cohort.npz"

    # Patch EcapaTdnn so main() doesn't try to load real ECAPA weights.
    class _FakeEcapa:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed(self, audio):
            return _deterministic_embed(audio)

    import types as _types

    fake_module = _types.ModuleType("mellonella_poc.embedding")
    fake_module.EcapaTdnn = _FakeEcapa  # type: ignore[attr-defined]
    sys.modules["mellonella_poc.embedding"] = fake_module

    rc = mod.main(
        [
            "--manifest",
            f"en={en}",
            "--manifest",
            f"de={de}",
            "--output",
            str(output),
            "--per-language",
            "2",
        ]
    )
    assert rc == 0
    assert output.exists()
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["n_embeddings"] == 4
    assert summary["per_language_counts"] == {"de": 2, "en": 2}
