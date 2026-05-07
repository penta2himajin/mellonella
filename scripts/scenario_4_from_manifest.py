#!/usr/bin/env python3
"""Run Scenario 4 (simultaneous target + other speech) against per-language manifests.

Mirrors ``scripts/scenario_5_from_manifest.py`` for the FP-tolerant overlap
test (D-010 Phase 7 step 1). For each ``--manifest LANG=PATH`` argument it:

1. loads the per-speaker concatenated buffers via
   :func:`mellonella_bench.datasets.commonvoice.load_speakers_from_manifest`
   (the manifest format used by both MLS and Emilia-YODAS prep).
2. picks the top-2 speakers by audio length per language; designates
   speaker[0] as the *target* and speaker[1] as the *other*.
3. materialises target / other clips to a working directory as 16 kHz wav.
4. assembles a list of :class:`Scenario4Item` (1 per language) and invokes
   :func:`mellonella_bench.scenarios.scenario_4.run`.
5. emits ``scenario_4.csv`` (per-row sweep) and ``summary.json``
   (per-ratio aggregates: ``gate_tpr_mean_at_<ratio>``,
   ``si_sdr_mean_at_<ratio>``, plus the aggregate scenario_4.run already
   computes).

This first PR runs in observation-only mode — no ``failures.json`` and no
hard thresholds. Phase 7 step 1 closeout will add per-ratio thresholds
once we have a stable baseline observed across two main runs.

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
    load_speakers_from_manifest,
)
from mellonella_bench.scenarios.base import (  # noqa: E402
    PipelineProvider,
    StubPipelineProvider,
)
from mellonella_bench.scenarios.scenario_4 import (  # noqa: E402
    DEFAULT_RATIOS_DB,
    Scenario4Item,
    run as run_scenario_4,
)

SAMPLE_RATE = 16_000


@dataclass
class ManifestSpec:
    """One ``LANG=PATH`` argument from the CLI (same format as scenario_5)."""

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


def _materialise_speaker_wav(
    audio: np.ndarray, dest: Path, *, max_seconds: float
) -> Path:
    n = min(audio.size, int(max_seconds * SAMPLE_RATE))
    if n <= 0:
        raise ValueError(f"speaker buffer is empty for {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio[:n].astype(np.float32), SAMPLE_RATE)
    return dest


def build_items(
    specs: list[ManifestSpec],
    work_dir: Path,
    *,
    max_seconds_per_speaker: float = 8.0,
) -> list[tuple[str, Scenario4Item]]:
    """Materialise per-language target/other wavs and build Scenario4 items.

    Per language: pick the 2 speakers with the most concatenated audio.
    Designate speaker[0] as ``target`` and speaker[1] as ``other``. Both
    clips are trimmed to ``max_seconds_per_speaker`` so frame counts are
    predictable across languages.

    Returns a list of ``(language, item)`` tuples; the language tag is
    not stored on :class:`Scenario4Item` itself but the driver needs it
    to break per-ratio aggregates down by language for the summary.
    """
    items: list[tuple[str, Scenario4Item]] = []
    n_samples_clip = int(max_seconds_per_speaker * SAMPLE_RATE)
    n_frames = n_samples_clip // 512

    for spec in specs:
        raw = load_speakers_from_manifest(spec.manifest, SAMPLE_RATE)
        if not raw:
            raise ValueError(
                f"manifest {spec.manifest} (language={spec.language!r}) "
                "yielded zero usable speakers after the min-duration filter"
            )
        ranked = sorted(raw.items(), key=lambda kv: kv[1].size, reverse=True)
        if len(ranked) < 2:
            raise ValueError(
                f"manifest {spec.manifest} (language={spec.language!r}) "
                f"has only {len(ranked)} speaker(s); need >= 2"
            )
        target_id, target_audio = ranked[0]
        other_id, other_audio = ranked[1]
        target_audio = np.asarray(target_audio, dtype=np.float32)
        other_audio = np.asarray(other_audio, dtype=np.float32)

        lang_dir = work_dir / spec.language
        target_path = _materialise_speaker_wav(
            target_audio,
            lang_dir / f"{target_id}_target.wav",
            max_seconds=max_seconds_per_speaker,
        )
        other_path = _materialise_speaker_wav(
            other_audio,
            lang_dir / f"{other_id}_other.wav",
            max_seconds=max_seconds_per_speaker,
        )
        sample_id = f"{spec.language}_{target_id}_vs_{other_id}"
        items.append(
            (
                spec.language,
                Scenario4Item(
                    sample_id=sample_id,
                    target_path=target_path,
                    other_path=other_path,
                    voiced_mask=np.ones(n_frames, dtype=bool),
                    enrollment_path=target_path,
                ),
            )
        )
    return items


def _aggregate_by_ratio(
    csv_path: Path,
    languages: list[str],
) -> dict[str, float]:
    """Compute per-ratio aggregates from the scenario_4 CSV.

    The scenario_4 CSV has one row per (item, ratio). ``snr_db`` is the
    ratio in dB for finite ratios, or empty when ``notes`` says
    ``target_only`` / ``other_only``. We bucket by the canonical ratio
    label (``+inf`` / ``+9.0`` / … / ``-9.0`` / ``-inf``) and emit:

    * ``gate_tpr_mean_at_<label>``
    * ``si_sdr_mean_at_<label>``
    * ``gate_tpr_mean__<lang>_at_<label>`` for per-language breakdown
    """
    import csv

    by_ratio_tpr: dict[str, list[float]] = defaultdict(list)
    by_ratio_sisdr: dict[str, list[float]] = defaultdict(list)
    by_lang_ratio_tpr: dict[tuple[str, str], list[float]] = defaultdict(list)

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            note = row.get("notes", "")
            snr_str = row.get("snr_db", "")
            if note == "target_only":
                label = "target_only"
            elif note == "other_only":
                label = "other_only"
            elif snr_str:
                # Canonicalise to one decimal so +9 → "+9.0"
                label = f"{float(snr_str):+.1f}db"
            else:
                continue

            tpr_str = row.get("gate_tpr", "")
            if tpr_str:
                tpr = float(tpr_str)
                if math.isfinite(tpr):
                    by_ratio_tpr[label].append(tpr)
                    sample_id = row.get("sample_id", "")
                    lang = sample_id.split("_", 1)[0] if sample_id else ""
                    if lang in languages:
                        by_lang_ratio_tpr[(lang, label)].append(tpr)

            sisdr_str = row.get("si_sdr", "")
            if sisdr_str:
                sisdr = float(sisdr_str)
                if math.isfinite(sisdr):
                    by_ratio_sisdr[label].append(sisdr)

    out: dict[str, float] = {}
    for label, values in by_ratio_tpr.items():
        out[f"gate_tpr_mean_at_{label}"] = float(np.mean(values))
    for label, values in by_ratio_sisdr.items():
        out[f"si_sdr_mean_at_{label}"] = float(np.mean(values))
    for (lang, label), values in by_lang_ratio_tpr.items():
        out[f"gate_tpr_mean__{lang}_at_{label}"] = float(np.mean(values))
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
        help="output directory for scenario_4.csv + summary.json",
    )
    parser.add_argument(
        "--max-seconds-per-speaker",
        type=float,
        default=8.0,
        help="trim each per-speaker buffer to this many seconds before mixing",
    )
    parser.add_argument(
        "--ratios-db",
        type=str,
        default=None,
        help=(
            "comma-separated target-to-other ratios in dB; use 'inf' / '-inf' "
            "for the target-only / other-only endpoints (default: scenario_4 "
            f"DEFAULT_RATIOS_DB = {DEFAULT_RATIOS_DB})"
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

    if args.ratios_db is not None:
        ratios = tuple(float(x) for x in args.ratios_db.split(","))
    else:
        ratios = DEFAULT_RATIOS_DB

    args.output.mkdir(parents=True, exist_ok=True)
    work_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    work_dir = Path(tempfile.mkdtemp(prefix="scenario4_workdir_", dir=str(work_root)))

    lang_items = build_items(
        args.manifest,
        work_dir=work_dir,
        max_seconds_per_speaker=args.max_seconds_per_speaker,
    )
    if not lang_items:
        print("error: built zero scenario_4 items from manifests", file=sys.stderr)
        return 2

    # Apply the requested ratios uniformly (scenario_4 picks them up off
    # the item, not as a run() kwarg).
    items = []
    for _lang, item in lang_items:
        item.target_to_other_ratios_db = ratios
        items.append(item)

    languages = sorted({lang for lang, _ in lang_items})

    provider = _resolve_provider(args.real_pipeline, as_norm_cohort=args.as_norm_cohort)
    csv_path = args.output / "scenario_4.csv"
    result = run_scenario_4(
        items,
        provider=provider,
        sample_rate=SAMPLE_RATE,
        output_csv=csv_path,
    )

    per_ratio = _aggregate_by_ratio(csv_path, languages)
    summary = {
        "n_items": result.n_samples,
        "languages": languages,
        "ratios_db": [
            "inf" if r == float("inf") else "-inf" if r == float("-inf") else r
            for r in ratios
        ],
        "metrics": {**result.metrics, **per_ratio},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
