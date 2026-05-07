#!/usr/bin/env python3
"""Run Scenario 6 (drift verification / auto-learn long-running behaviour)
against per-language manifests.

Mirrors ``scripts/scenario_4_from_manifest.py`` for the auto-learn drift
test (D-010 Phase 7 step 2). For each ``--manifest LANG=PATH`` argument it:

1. reads the per-clip rows via :func:`mellonella_bench.datasets.commonvoice.read_manifest`
   (the manifest format used by both MLS and Emilia-YODAS prep).
2. groups clips by ``speaker_id`` and picks the speaker with the longest
   total concatenated audio per language (top-1 by total duration —
   same ranking criterion as scenario_4 / scenario_5 use, gives the
   speaker most likely to have an enrollment-grade clip).
3. sorts that speaker's clips deterministically by ``clip_path`` (lex)
   and uses ``clip[0]`` as the enrollment, ``clip[1..variants_per_speaker]``
   as the ordered drift variants.
4. assembles a list of :class:`Scenario6Item` (1 per language) and invokes
   :func:`mellonella_bench.scenarios.scenario_6.run`.
5. emits ``scenario_6.csv`` (per-row metrics) and ``summary.json``
   (per-language aggregates: ``gate_tpr_mean__<lang>``,
   ``auto_learn_admissions_mean__<lang>``, ``auto_learn_resets_mean__<lang>``,
   ``anchor_distance_final_mean__<lang>``).

Phase 7 step 2 first activation is observation-only — no ``failures.json``
and no hard thresholds. The PoC closeout deliverable is "auto-learn pool
is connected end-to-end and admission counters / anchor_distance values
come back finite from the real pipeline" (implementation.md Phase 2).
Drift signal richness is currently capped by ``clips_per_speaker = 4``
in dataset prep (3 variants per language); a follow-up may bump it to
8-12 if the observed signal is too thin.

Manifest format is the one written by
``mellonella_bench.datasets.commonvoice.write_manifest``
(`language`, `speaker_id`, `clip_path`, `sentence`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT / "poc"))

from mellonella_bench.datasets.commonvoice import (  # noqa: E402
    CommonVoiceClip,
    read_manifest,
)
from mellonella_bench.scenarios.base import (  # noqa: E402
    PipelineProvider,
    StubPipelineProvider,
)
from mellonella_bench.scenarios.scenario_6 import (  # noqa: E402
    Scenario6Item,
    run as run_scenario_6,
)

SAMPLE_RATE = 16_000
DEFAULT_VARIANTS_PER_SPEAKER = 3


@dataclass
class ManifestSpec:
    """One ``LANG=PATH`` argument from the CLI."""

    language: str
    manifest: Path

    @classmethod
    def parse(cls, raw: str) -> "ManifestSpec":
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                f"--manifest expects 'LANG=PATH', got {raw!r}"
            )
        lang, path = raw.split("=", 1)
        lang = lang.strip()
        path = path.strip()
        if not lang or not path:
            raise argparse.ArgumentTypeError(
                f"--manifest expects non-empty 'LANG=PATH', got {raw!r}"
            )
        return cls(language=lang, manifest=Path(path))


def _clip_duration_s(absolute_path: Path) -> float:
    """Read a wav header and return its duration in seconds (cheap, no decode)."""
    info = sf.info(str(absolute_path))
    return info.frames / info.samplerate


def _pick_speaker_clips(
    manifest_path: Path,
    language: str,
) -> tuple[str, list[Path]]:
    """Return ``(speaker_id, sorted_clip_paths)`` for the longest-audio speaker.

    Reads the manifest (clip-level rows), groups clips by speaker, ranks
    speakers by total audio duration (longest first; lex tiebreak on
    speaker_id), picks the top speaker and returns their absolute clip
    paths sorted lex by ``clip_path``.
    """
    rows: list[CommonVoiceClip] = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"manifest {manifest_path} is empty (language={language!r})")
    base = manifest_path.parent

    by_speaker: dict[str, list[CommonVoiceClip]] = defaultdict(list)
    for r in rows:
        by_speaker[r.speaker_id].append(r)

    # Rank speakers by total clip duration (sum across their clips).
    ranked: list[tuple[float, str, list[CommonVoiceClip]]] = []
    for spk, clips in by_speaker.items():
        total_s = 0.0
        for c in clips:
            try:
                total_s += _clip_duration_s(base / c.clip_path)
            except Exception:
                # If any clip fails to read its header, skip this speaker
                # rather than silently produce a corrupt item — the
                # manifest is meant to point at materialised wavs.
                total_s = -1.0
                break
        if total_s > 0:
            ranked.append((total_s, spk, clips))
    if not ranked:
        raise ValueError(
            f"manifest {manifest_path} (language={language!r}) "
            "yielded zero speakers with readable audio"
        )
    ranked.sort(key=lambda t: (-t[0], t[1]))
    _, top_speaker, top_clips = ranked[0]
    sorted_paths = sorted((base / c.clip_path for c in top_clips), key=str)
    return top_speaker, sorted_paths


def build_items(
    specs: list[ManifestSpec],
    *,
    variants_per_speaker: int = DEFAULT_VARIANTS_PER_SPEAKER,
) -> list[tuple[str, str, Scenario6Item]]:
    """Build one ``(language, speaker_id, item)`` per manifest.

    ``clip[0]`` is the enrollment, ``clip[1..variants_per_speaker]`` are
    the ordered drift variants. The function fails loud if a manifest
    has fewer than ``1 + variants_per_speaker`` readable clips for the
    top speaker — that signals the dataset prep emitted thinner data
    than expected and the caller should bump ``clips_per_speaker`` in
    the dataset prep step rather than silently drop variants.
    """
    if variants_per_speaker < 1:
        raise ValueError("variants_per_speaker must be >= 1")
    out: list[tuple[str, str, Scenario6Item]] = []
    for spec in specs:
        speaker_id, paths = _pick_speaker_clips(spec.manifest, spec.language)
        needed = 1 + variants_per_speaker
        if len(paths) < needed:
            raise ValueError(
                f"manifest {spec.manifest} (language={spec.language!r}, "
                f"speaker={speaker_id!r}) has only {len(paths)} clip(s); "
                f"need >= {needed} (1 enrollment + {variants_per_speaker} variants). "
                f"Bump clips_per_speaker in the dataset prep."
            )
        enrollment = paths[0]
        variants = tuple(paths[1 : 1 + variants_per_speaker])
        sample_id = f"{spec.language}_{speaker_id}"
        out.append(
            (
                spec.language,
                speaker_id,
                Scenario6Item(
                    sample_id=sample_id,
                    enrollment_path=enrollment,
                    variant_paths=variants,
                ),
            )
        )
    return out


def _aggregate_by_language(
    csv_path: Path,
    item_languages: dict[str, str],
) -> dict[str, float]:
    """Compute per-language aggregates from the scenario_6 CSV.

    scenario_6 emits one row per item; ``item_languages`` maps
    ``sample_id -> language`` so we can break the row-level metrics
    down per language. Aggregates produced:

    * ``gate_tpr_mean__<lang>``
    * ``frame_accuracy_mean__<lang>``
    * ``auto_learn_admissions_mean__<lang>``
    * ``auto_learn_resets_mean__<lang>``
    * ``anchor_distance_final_mean__<lang>``
    """
    import csv

    by_lang: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample_id = row.get("sample_id", "")
            lang = item_languages.get(sample_id)
            if lang is None:
                continue
            for field in (
                "gate_tpr",
                "frame_accuracy",
                "auto_learn_admissions",
                "auto_learn_resets",
                "anchor_distance_final",
            ):
                raw = row.get(field, "")
                if not raw:
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if math.isfinite(val):
                    by_lang[lang][field].append(val)

    out: dict[str, float] = {}
    for lang, fields in by_lang.items():
        for field, values in fields.items():
            if values:
                out[f"{field}_mean__{lang}"] = float(np.mean(values))
    return out


def _resolve_provider(
    use_real: bool, *, as_norm_cohort: Path | None = None
) -> PipelineProvider:
    if not use_real:
        return StubPipelineProvider()
    from mellonella_bench.scenarios.pipeline_provider import RealPipelineProvider

    return RealPipelineProvider(as_norm_cohort_path=as_norm_cohort)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=ManifestSpec.parse,
        required=True,
        help="LANG=PATH ; repeat per language",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory for scenario_6.csv + summary.json",
    )
    parser.add_argument(
        "--variants-per-speaker",
        type=int,
        default=DEFAULT_VARIANTS_PER_SPEAKER,
        help=(
            "number of drift variants per language (default: "
            f"{DEFAULT_VARIANTS_PER_SPEAKER}). The driver picks the speaker with "
            "the most concatenated audio per language and uses clip[0] as "
            "enrollment + clip[1..N] as variants."
        ),
    )
    parser.add_argument(
        "--real-pipeline",
        action="store_true",
        help="use the real mellonella-poc pipeline (requires `pip install -e poc[models]`)",
    )
    parser.add_argument(
        "--as-norm-cohort",
        type=Path,
        default=None,
        help=(
            "path to an impostor cohort .npz; enables AS-Norm in the real "
            "pipeline. Ignored when --real-pipeline is not set."
        ),
    )
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    # Working dir is unused right now (scenario_6 reads enrollment +
    # variants directly from the prepared manifest's tree) but keep the
    # plumbing for parity with scenario_4 / scenario_5 in case future
    # work needs to materialise truncated variants here.
    work_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    Path(tempfile.mkdtemp(prefix="scenario6_workdir_", dir=str(work_root)))

    triples = build_items(
        args.manifest,
        variants_per_speaker=args.variants_per_speaker,
    )
    if not triples:
        print("error: built zero scenario_6 items from manifests", file=sys.stderr)
        return 2
    items = [t[2] for t in triples]
    item_languages = {t[2].sample_id: t[0] for t in triples}
    languages = sorted({t[0] for t in triples})

    provider = _resolve_provider(args.real_pipeline, as_norm_cohort=args.as_norm_cohort)
    csv_path = args.output / "scenario_6.csv"
    result = run_scenario_6(
        items,
        provider=provider,
        sample_rate=SAMPLE_RATE,
        output_csv=csv_path,
    )

    per_lang = _aggregate_by_language(csv_path, item_languages)
    summary = {
        "n_items": result.n_samples,
        "languages": languages,
        "variants_per_speaker": args.variants_per_speaker,
        "metrics": {**result.metrics, **per_lang},
        "speakers_picked": [
            {"language": t[0], "speaker_id": t[1], "sample_id": t[2].sample_id}
            for t in triples
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
