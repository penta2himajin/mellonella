#!/usr/bin/env python3
"""Phase 7 step 3 — per-language theta_pass_as_norm sweep.

For each ``--manifest LANG=PATH`` argument, runs the AS-Norm calibration
sweep that ``scripts/calibrate.py`` would on the librosa libri1/2/3
default speakers, but using the top-N speakers (by total concatenated
audio duration) from the language's prepared manifest. Produces a
single combined summary JSON with one ``per_language`` entry per
manifest plus a ``spread_analysis`` block that compares the per-language
recommendations against the current global θ_pass_as_norm = 2.25 to
decide whether to (a) keep the global θ or (b) adopt a per-language
table.

The scenario_5 results for D-010 Phase 6 / Phase 7 step 2 consistently
show zh-CN as a per-language outlier (FPR mean 0.084-0.184 vs ≤0.062
for the other 5 languages; scenario_6 TPR 0.317 vs ≥0.93). Running
calibrate.py against libri1/2/3 — the current scenario_5.yml step —
anchors the recommended θ on English statistics and cannot speak to
whether zh-CN wants a different θ. This script closes that gap.

Reuses ``measure_cells`` / ``aggregate`` / ``recommend_theta`` from
``scripts/calibrate.py`` per language. Designed to be triggered
manually via ``workflow_dispatch`` rather than per-PR — the cost is
~6:26 × N_languages on a warm cohort cache, observation-only output,
and downstream consumers (GatingConfig propagation, scenario_5 baseline
re-capture) follow as separate PRs after we read the recommendations.

Manifest format is the one written by
``mellonella_bench.datasets.commonvoice.write_manifest``
(`language`, `speaker_id`, `clip_path`, `sentence`).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT / "poc"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mellonella_bench.datasets.commonvoice import (  # noqa: E402
    load_speakers_from_manifest,
)

# Reuse the calibrate.py plumbing rather than duplicating the sweep
# logic. The script is at module level (no package), so a sys.path
# insert above and a flat import is the cleanest way in.
import calibrate  # noqa: E402  (used for NOISE_TYPES monkey-patching below)
from calibrate import (  # noqa: E402
    MAX_MEAN_FPR_AS_NORM,
    MIN_TPR_FLOOR_AS_NORM,
    NOISE_TYPES,
    SNRS_DB,
    THETA_GRID_AS_NORM,
    aggregate,
    measure_cells,
    recommend_theta,
)

SAMPLE_RATE = 16_000
DEFAULT_TOP_SPEAKERS_PER_LANG = 3  # matches libri1/2/3 cell count
DEFAULT_MAX_SECONDS_PER_SPEAKER = 6.0  # matches libri1/2/3 typical audio length
DEFAULT_GLOBAL_THETA = 2.25  # current GatingConfig default
DEFAULT_SPREAD_DECISION_THRESHOLD = 0.5


@dataclass
class ManifestSpec:
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


def _select_top_speakers(
    speakers: dict[str, np.ndarray], n: int
) -> dict[str, np.ndarray]:
    """Pick the ``n`` speakers with the longest concatenated audio.

    Lex tiebreak on speaker_id keeps selection deterministic across
    runs on the same manifest (mirrors the ranking criterion in
    ``scripts/build_impostor_cohort.py`` so target / cohort selection
    stay aligned conceptually).
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    items = list(speakers.items())
    items.sort(key=lambda kv: (-kv[1].size, kv[0]))
    return dict(items[:n])


def _trim_speakers(
    speakers: dict[str, np.ndarray], max_seconds: float
) -> dict[str, np.ndarray]:
    """Trim each speaker buffer to ``max_seconds`` so per-cell pipeline cost
    matches the libri1/2/3 footprint scripts/calibrate.py was originally sized
    for.

    ``measure_cells`` half-splits each speaker (enroll = audio[:half],
    test = audio[half:]) and runs ``process_offline`` on the test half
    once per (enroll, test, noise, snr) cell. Pipeline cost scales with
    audio length, so a 40 s manifest concat (4 clips × ~10 s, current
    Phase 6 v10 dataset prep default) is ~8× the libri1/2/3 ~5 s
    samples — enough that the global calibrate.py budget (~6:26 per run
    on 108 cells) explodes to multi-hour territory at 6 languages and
    blows the 60-min job-level timeout.

    Trimming to ``max_seconds`` (default 6 s, matching libri1/2/3
    typical) keeps the per-cell cost comparable. The accuracy cost is
    bounded — the sweep only measures TPR / FPR rates that already
    converge well at the libri1/2/3 audio scale; longer buffers just
    give marginally more stable rate estimates per cell.
    """
    n_max = int(max_seconds * SAMPLE_RATE)
    if n_max <= 0:
        raise ValueError("max_seconds must be > 0")
    out: dict[str, np.ndarray] = {}
    for name, audio in speakers.items():
        if audio.size > n_max:
            out[name] = audio[:n_max]
        else:
            out[name] = audio
    return out


def sweep_one_language(
    manifest_path: Path,
    language: str,
    *,
    cohort_path: Path,
    top_speakers: int,
    max_seconds_per_speaker: float,
    noise_types: tuple[str, ...] | None = None,
) -> dict:
    """Run the AS-Norm sweep for one language and return its per-θ + recommended.

    ``noise_types`` (e.g. ``("white",)``) overrides ``calibrate.NOISE_TYPES``
    for the duration of this call so a diagnostic sweep can isolate the
    contribution of one noise type. Default ``None`` keeps the module-level
    setting (``("white", "pink")``).
    """
    raw = load_speakers_from_manifest(manifest_path, SAMPLE_RATE)
    if not raw:
        raise RuntimeError(
            f"manifest {manifest_path} (language={language!r}) yielded "
            "zero usable speakers after the min-duration filter"
        )
    if len(raw) < top_speakers:
        raise RuntimeError(
            f"manifest {manifest_path} (language={language!r}) has only "
            f"{len(raw)} speaker(s); need >= {top_speakers}"
        )
    speakers = _select_top_speakers(raw, top_speakers)
    speakers = _trim_speakers(speakers, max_seconds_per_speaker)

    saved_noise_types = calibrate.NOISE_TYPES
    if noise_types is not None:
        calibrate.NOISE_TYPES = tuple(noise_types)
    try:
        rows = measure_cells(
            speakers=speakers,
            language=language,
            use_as_norm=True,
            cohort_path=cohort_path,
        )
    finally:
        calibrate.NOISE_TYPES = saved_noise_types
    per_theta = aggregate(rows)
    recommended = recommend_theta(
        per_theta,
        max_mean_fpr=MAX_MEAN_FPR_AS_NORM,
        min_tpr_floor=MIN_TPR_FLOOR_AS_NORM,
    )
    return {
        "speakers_used": list(speakers),
        "n_speakers": len(speakers),
        "n_cells": len(speakers) ** 2 * len(NOISE_TYPES) * len(SNRS_DB),
        "n_rows": len(rows),
        "per_theta": {f"{theta:.3f}": metrics for theta, metrics in per_theta.items()},
        "recommended_theta_pass": recommended,
    }


def _spread_analysis(
    per_lang: dict[str, dict],
    *,
    global_theta: float,
    decision_threshold: float,
) -> dict:
    """Compare per-language recommendations against the current global θ.

    Decision rule (Phase 7 step 3 closeout criterion):

    * range_width = max(recommended) - min(recommended)
    * If range_width <= decision_threshold (default 0.5): keep global θ
      (per-language complexity not worth the spread).
    * Otherwise: adopt a per-language table.

    Also flags any language whose recommended θ is more than
    ``decision_threshold`` away from the current global setting — those
    are the ones the global θ is currently mis-serving.
    """
    recommendations = {
        lang: float(payload["recommended_theta_pass"])
        for lang, payload in per_lang.items()
    }
    if not recommendations:
        return {"decision": "no_data", "recommendations": {}}
    values = list(recommendations.values())
    min_t = min(values)
    max_t = max(values)
    width = max_t - min_t
    decision = (
        "adopt_per_language_table"
        if width > decision_threshold
        else "keep_global"
    )
    deltas_from_global = {
        lang: round(theta - global_theta, 3)
        for lang, theta in recommendations.items()
    }
    outliers_vs_global = sorted(
        lang for lang, delta in deltas_from_global.items()
        if abs(delta) > decision_threshold
    )
    return {
        "global_theta": global_theta,
        "decision_threshold": decision_threshold,
        "recommendations": recommendations,
        "min": min_t,
        "max": max_t,
        "range_width": round(width, 3),
        "deltas_from_global": deltas_from_global,
        "outliers_vs_global": outliers_vs_global,
        "decision": decision,
    }


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
        "--cohort",
        type=Path,
        required=True,
        help="path to the AS-Norm impostor cohort .npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output path for the combined per-language summary JSON",
    )
    parser.add_argument(
        "--top-speakers",
        type=int,
        default=DEFAULT_TOP_SPEAKERS_PER_LANG,
        help=(
            f"speakers per language (default: {DEFAULT_TOP_SPEAKERS_PER_LANG} — "
            "matches the libri1/2/3 cell count = 108 cells/lang)"
        ),
    )
    parser.add_argument(
        "--noise-types",
        type=str,
        default=None,
        help=(
            "comma-separated noise types to mix during the sweep "
            "(default: keep the calibrate.py module-level NOISE_TYPES, "
            f"currently {NOISE_TYPES}). Diagnostic flag — set "
            "to e.g. 'white' to isolate one noise type's contribution. "
            "Phase 7 step 3 follow-up: ja was hitting recommend_theta's "
            "fallback path (TPR_median < 0.5 floor at every theta) and we "
            "wanted to test whether pink noise's low-frequency energy was "
            "the cause."
        ),
    )
    parser.add_argument(
        "--max-seconds-per-speaker",
        type=float,
        default=DEFAULT_MAX_SECONDS_PER_SPEAKER,
        help=(
            f"trim each per-speaker buffer to this many seconds before the "
            f"sweep (default: {DEFAULT_MAX_SECONDS_PER_SPEAKER} — matches the "
            "libri1/2/3 footprint calibrate.py was sized for; raising it "
            "makes per-cell cost grow linearly and blows the 60-min job-level "
            "timeout at 6 languages)"
        ),
    )
    parser.add_argument(
        "--global-theta",
        type=float,
        default=DEFAULT_GLOBAL_THETA,
        help=(
            f"current global theta_pass_as_norm (default: {DEFAULT_GLOBAL_THETA}) — "
            "used only to compute deltas_from_global in the spread analysis"
        ),
    )
    parser.add_argument(
        "--spread-decision-threshold",
        type=float,
        default=DEFAULT_SPREAD_DECISION_THRESHOLD,
        help=(
            f"range_width threshold for the per-language adoption decision "
            f"(default: {DEFAULT_SPREAD_DECISION_THRESHOLD})"
        ),
    )
    args = parser.parse_args(argv)

    noise_types_override: tuple[str, ...] | None = None
    if args.noise_types is not None:
        noise_types_override = tuple(
            s.strip() for s in args.noise_types.split(",") if s.strip()
        )
        if not noise_types_override:
            parser.error("--noise-types must list at least one type")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    per_lang: dict[str, dict] = {}
    for spec in args.manifest:
        print(f"\n[per-lang-calibrate] === language={spec.language!r} ===")
        per_lang[spec.language] = sweep_one_language(
            spec.manifest,
            spec.language,
            cohort_path=args.cohort,
            top_speakers=args.top_speakers,
            max_seconds_per_speaker=args.max_seconds_per_speaker,
            noise_types=noise_types_override,
        )

    spread = _spread_analysis(
        per_lang,
        global_theta=args.global_theta,
        decision_threshold=args.spread_decision_threshold,
    )

    summary = {
        "schema_version": 1,
        "config": {
            "snrs_db": list(SNRS_DB),
            "noise_types": list(NOISE_TYPES),
            "theta_grid": list(THETA_GRID_AS_NORM),
            "max_mean_fpr": MAX_MEAN_FPR_AS_NORM,
            "min_tpr_floor": MIN_TPR_FLOOR_AS_NORM,
            "top_speakers_per_lang": args.top_speakers,
            "max_seconds_per_speaker": args.max_seconds_per_speaker,
            "noise_types_used": list(noise_types_override) if noise_types_override is not None else list(NOISE_TYPES),
            "cohort_path": str(args.cohort),
        },
        "per_language": per_lang,
        "spread_analysis": spread,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(json.dumps(spread, indent=2))
    print(f"\n[per-lang-calibrate] decision: {spread['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
