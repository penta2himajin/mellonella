#!/usr/bin/env python3
"""Run Scenario 5 (multilingual robustness) against CommonVoice manifests.

Bridges the gap between the CommonVoice subset prep step and the
``scenario_5`` runner. For each ``--manifest LANG=PATH`` argument it:

1. loads the per-speaker concatenated buffers via
   :func:`mellonella_bench.datasets.commonvoice.load_speakers_from_manifest`
2. picks the top-N speakers (by audio length); designates speaker[0] as
   the *target* and pairs it against speaker[1..] as *other* speakers
3. materialises target/other clips to a working directory as 16 kHz wav
4. assembles a list of :class:`Scenario5Item` and invokes
   :func:`mellonella_bench.scenarios.scenario_5.run`
5. emits ``scenario_5.csv`` (per-row sweep), ``summary.json`` (per-language
   aggregates + cross-language stddev) and ``failures.json`` (any row that
   violates the configured ``--tpr-min`` / ``--fpr-max`` thresholds)

Exit code is 1 when any threshold violation is recorded — wire the script
into a workflow (or call it from ``ci_accuracy.py`` once real CommonVoice
data is in place) to hard-fail regressions.

Manifest format is the one written by
``mellonella_bench.datasets.commonvoice.write_manifest`` (`language`,
`speaker_id`, `clip_path`, `sentence`).

Noise: ``--noise PATH`` consumes a wav; otherwise the script generates
deterministic white noise (seeded) sized to the longest target buffer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
from mellonella_bench.scenarios.scenario_5 import (  # noqa: E402
    DEFAULT_SNRS_DB,
    Scenario5Item,
    run as run_scenario_5,
)

SAMPLE_RATE = 16_000
DEFAULT_TPR_MIN = 0.5
DEFAULT_FPR_MAX = 0.5
DEFAULT_TOP_SPEAKERS = 2


@dataclass
class ManifestSpec:
    """One ``LANG=PATH`` argument from the CLI."""

    language: str
    manifest: Path

    @classmethod
    def parse(cls, raw: str) -> ManifestSpec:
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


def _trim_or_pad(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim ``audio`` to ``n_samples`` or right-pad with zeros."""
    if audio.size >= n_samples:
        return audio[:n_samples]
    pad = np.zeros(n_samples - audio.size, dtype=audio.dtype)
    return np.concatenate([audio, pad])


def _build_noise(reference_n: int, noise_path: Path | None, seed: int) -> np.ndarray:
    """Either load ``noise_path`` (and tile/trim to ``reference_n``) or synthesise white noise."""
    if noise_path is not None:
        audio, sr = sf.read(str(noise_path), dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"--noise wav must be {SAMPLE_RATE} Hz, got {sr} Hz at {noise_path}"
            )
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < reference_n:
            repeats = int(np.ceil(reference_n / max(audio.size, 1)))
            audio = np.tile(audio, repeats)[:reference_n]
        else:
            audio = audio[:reference_n]
        return audio.astype(np.float32)
    rng = np.random.default_rng(seed)
    return (0.05 * rng.standard_normal(reference_n)).astype(np.float32)


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
    top_speakers: int = DEFAULT_TOP_SPEAKERS,
    max_seconds_per_speaker: float = 8.0,
    noise_path: Path | None = None,
    seed: int = 0,
) -> list[Scenario5Item]:
    """Materialise per-language target/other/noise wavs and build Scenario5 items.

    Per language:

    * Pick the ``top_speakers`` speakers with the most concatenated audio.
    * Designate speaker[0] as ``target`` and pair it against each of
      speaker[1:top_speakers] as a separate ``other``.
    * Both clips are trimmed to ``max_seconds_per_speaker`` so SNR mixing
      and frame counts are predictable across languages.
    """
    if top_speakers < 2:
        raise ValueError(
            "top_speakers must be >= 2 (need a target and at least one other)"
        )

    items: list[Scenario5Item] = []
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
        target_audio = np.asarray(target_audio, dtype=np.float32)

        lang_dir = work_dir / spec.language
        target_path = _materialise_speaker_wav(
            target_audio,
            lang_dir / f"{target_id}_target.wav",
            max_seconds=max_seconds_per_speaker,
        )
        # Build a per-language noise file so downstream loaders can read it.
        noise_local = _build_noise(
            n_samples_clip,
            noise_path=noise_path,
            seed=seed + hash(spec.language) % 2**16,
        )
        noise_local_path = lang_dir / "noise.wav"
        noise_local_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(noise_local_path), noise_local, SAMPLE_RATE)

        for other_id, other_audio in ranked[1:top_speakers]:
            other_audio = np.asarray(other_audio, dtype=np.float32)
            other_path = _materialise_speaker_wav(
                other_audio,
                lang_dir / f"{other_id}_other.wav",
                max_seconds=max_seconds_per_speaker,
            )
            sample_id = f"{spec.language}_{target_id}_vs_{other_id}"
            items.append(
                Scenario5Item(
                    sample_id=sample_id,
                    language=spec.language,
                    target_path=target_path,
                    other_path=other_path,
                    noise_path=noise_local_path,
                    target_voiced_mask=np.ones(n_frames, dtype=bool),
                    other_voiced_mask=np.ones(n_frames, dtype=bool),
                    target_speaker=target_id,
                    other_speaker=other_id,
                    enrollment_path=target_path,
                )
            )
    return items


def collect_failures(
    sweep_entries: list,
    *,
    tpr_min: float,
    fpr_max: float,
) -> list[dict]:
    """Return one dict per (item, snr, mode) that violated a threshold."""
    failures: list[dict] = []
    for entry in sweep_entries:
        if entry.notes == "mode=target" and entry.gate_tpr is not None:
            if entry.gate_tpr < tpr_min:
                failures.append(
                    {
                        "language": entry.language,
                        "sample_id": entry.sample_id,
                        "target_speaker": entry.target_speaker,
                        "other_speaker": entry.other_speaker,
                        "snr_db": entry.snr_db,
                        "mode": "target",
                        "metric": "gate_tpr",
                        "value": entry.gate_tpr,
                        "threshold": tpr_min,
                        "violation": "below_tpr_min",
                    }
                )
        elif entry.notes == "mode=other" and entry.gate_fpr is not None:
            if entry.gate_fpr > fpr_max:
                failures.append(
                    {
                        "language": entry.language,
                        "sample_id": entry.sample_id,
                        "target_speaker": entry.target_speaker,
                        "other_speaker": entry.other_speaker,
                        "snr_db": entry.snr_db,
                        "mode": "other",
                        "metric": "gate_fpr",
                        "value": entry.gate_fpr,
                        "threshold": fpr_max,
                        "violation": "above_fpr_max",
                    }
                )
    return failures


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
        help="LANG=PATH ; repeat per language (e.g. --manifest ja=/data/.../manifest.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output directory for scenario_5.csv + summary.json + failures.json",
    )
    parser.add_argument(
        "--noise",
        type=Path,
        default=None,
        help="optional 16 kHz wav to use as the noise source (default: synthesised white)",
    )
    parser.add_argument(
        "--top-speakers",
        type=int,
        default=DEFAULT_TOP_SPEAKERS,
        help=f"how many speakers to consume per language (default: {DEFAULT_TOP_SPEAKERS})",
    )
    parser.add_argument(
        "--max-seconds-per-speaker",
        type=float,
        default=8.0,
        help="trim each per-speaker buffer to this many seconds before mixing",
    )
    parser.add_argument(
        "--snrs-db",
        type=lambda s: tuple(float(x) for x in s.split(",")),
        default=DEFAULT_SNRS_DB,
        help=f"comma-separated SNR list in dB (default: {','.join(map(str, DEFAULT_SNRS_DB))})",
    )
    parser.add_argument(
        "--tpr-min",
        type=float,
        default=DEFAULT_TPR_MIN,
        help=f"per-row minimum target TPR; rows below count as failures (default: {DEFAULT_TPR_MIN})",
    )
    parser.add_argument(
        "--fpr-max",
        type=float,
        default=DEFAULT_FPR_MAX,
        help=f"per-row maximum other FPR; rows above count as failures (default: {DEFAULT_FPR_MAX})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for noise + SNR mixing (default: 0)",
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
            "path to an impostor cohort .npz "
            "(see scripts/build_impostor_cohort.py); enables AS-Norm in the "
            "real pipeline. Ignored when --real-pipeline is not set."
        ),
    )
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    # Keep intermediate per-speaker wavs out of args.output so the directory
    # stays artifact-friendly (only csv / json results land there).
    work_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    work_dir = Path(tempfile.mkdtemp(prefix="scenario5_workdir_", dir=str(work_root)))

    items = build_items(
        args.manifest,
        work_dir=work_dir,
        top_speakers=args.top_speakers,
        max_seconds_per_speaker=args.max_seconds_per_speaker,
        noise_path=args.noise,
        seed=args.seed,
    )
    if not items:
        print("error: built zero scenario_5 items from manifests", file=sys.stderr)
        return 2

    provider = _resolve_provider(args.real_pipeline, as_norm_cohort=args.as_norm_cohort)
    result = run_scenario_5(
        items,
        provider=provider,
        sample_rate=SAMPLE_RATE,
        output_csv=args.output / "scenario_5.csv",
        snrs_db=args.snrs_db,
        seed=args.seed,
    )

    summary = {
        "n_items": result.n_samples,
        "languages": sorted({s.language for s in args.manifest}),
        "snrs_db": list(args.snrs_db),
        "metrics": result.metrics,
        "thresholds": {"tpr_min": args.tpr_min, "fpr_max": args.fpr_max},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    sweep_entries = result.sweep.entries if result.sweep is not None else []
    failures = collect_failures(
        sweep_entries, tpr_min=args.tpr_min, fpr_max=args.fpr_max
    )
    failures_payload = {
        "thresholds": {"tpr_min": args.tpr_min, "fpr_max": args.fpr_max},
        "n_failures": len(failures),
        "n_total_rows": len(sweep_entries),
        "failures": failures,
    }
    (args.output / "failures.json").write_text(json.dumps(failures_payload, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"failures: {len(failures)} / {len(sweep_entries)} rows")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
