"""Scenario 5: multilingual robustness.

    INPUT:  per-language (target speaker, other speaker, noise) triples
    EXPECT: gating accuracy stable across languages

ECAPA-TDNN is trained on VoxCeleb (multilingual) and is in principle
language-agnostic, but speaker-verification thresholds calibrated on a
single language can still drift when applied to phonetically distinct
material. This scenario fans the standard SNR sweep out across the
CommonVoice language set and reports per-language TPR/FPR plus the
cross-language standard deviation.

Per item, we run the pipeline twice at every SNR:

* target + noise   → TPR (frame-level pass rate during target voicing)
* other  + noise   → FPR (frame-level pass rate during other-speaker voicing)

The CSV gets two rows per ``(item, SNR)`` pair (one ``mode='target'`` and
one ``mode='other'``); aggregate metrics are organised per language and
include a cross-language stddev to surface language-specific regressions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..metrics.gating import confusion_from_frames
from .base import (
    PipelineProvider,
    ScenarioResult,
    SnrSweep,
    SnrSweepEntry,
    StubPipelineProvider,
    mix_at_snr,
)

DEFAULT_SNRS_DB: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)


@dataclass
class Scenario5Item:
    """One per-language evaluation triple.

    The caller is responsible for pre-materialising audio paths (e.g. via
    :func:`mellonella_bench.datasets.commonvoice.load_speakers_from_manifest`
    + ``soundfile.write`` to a tmp dir). The scenario itself just consumes
    the wav paths.
    """

    sample_id: str
    language: str
    target_path: Path
    other_path: Path
    noise_path: Path
    target_voiced_mask: np.ndarray
    """Per-frame voicing mask of ``target_path`` at the SV frame rate."""
    other_voiced_mask: np.ndarray
    """Per-frame voicing mask of ``other_path`` at the SV frame rate."""
    target_speaker: str = ""
    other_speaker: str = ""
    enrollment_path: Path | None = None


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _truncate_to_match(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Trim or right-pad ``candidate`` so it lines up with ``reference``."""
    if candidate.size >= reference.size:
        return candidate[: reference.size]
    pad = np.zeros(reference.size - candidate.size, dtype=candidate.dtype)
    return np.concatenate([candidate, pad])


def evaluate_one(
    item: Scenario5Item,
    provider: PipelineProvider,
    sample_rate: int,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    rng: np.random.Generator | None = None,
) -> list[SnrSweepEntry]:
    """Run the per-item pipeline twice per SNR (target+noise and other+noise).

    Returns two rows per SNR — the ``notes`` field carries ``"mode=target"``
    or ``"mode=other"`` so downstream code can split them, and ``language``
    is populated from ``item.language``.
    """
    target, target_sr = _load_mono(item.target_path)
    other, other_sr = _load_mono(item.other_path)
    noise, noise_sr = _load_mono(item.noise_path)
    if target_sr != sample_rate or other_sr != sample_rate or noise_sr != sample_rate:
        raise ValueError(
            "sample-rate mismatch "
            f"(target={target_sr}, other={other_sr}, noise={noise_sr}, "
            f"expected={sample_rate})"
        )

    pipeline = provider.for_item(item)

    target_voiced = item.target_voiced_mask.astype(bool)
    other_voiced = item.other_voiced_mask.astype(bool)

    rows: list[SnrSweepEntry] = []
    for snr in snrs_db:
        # ---- target + noise → TPR ----
        mixture_t = mix_at_snr(target, noise, snr, rng=rng)
        t0 = time.perf_counter()
        result_t = pipeline(mixture_t, sample_rate)
        elapsed_t = (time.perf_counter() - t0) * 1000.0
        gate_t = _truncate_to_match(target_voiced, result_t.gate_per_frame.astype(bool))
        confusion_t = confusion_from_frames(target_voiced, gate_t)

        rows.append(
            SnrSweepEntry(
                sample_id=item.sample_id,
                language=item.language,
                target_speaker=item.target_speaker,
                other_speaker=item.other_speaker,
                snr_db=snr,
                gate_tpr=confusion_t.tpr,
                gate_fnr=confusion_t.fnr,
                processing_time_ms=elapsed_t,
                notes="mode=target",
            )
        )

        # ---- other + noise → FPR ----
        mixture_o = mix_at_snr(other, noise, snr, rng=rng)
        t0 = time.perf_counter()
        result_o = pipeline(mixture_o, sample_rate)
        elapsed_o = (time.perf_counter() - t0) * 1000.0
        gate_o = _truncate_to_match(other_voiced, result_o.gate_per_frame.astype(bool))
        voiced_count = int(other_voiced.sum())
        if voiced_count > 0:
            pass_voiced = int((gate_o & other_voiced).sum())
            fpr = pass_voiced / voiced_count
            tnr = 1.0 - fpr
        else:
            fpr = 0.0
            tnr = 0.0

        rows.append(
            SnrSweepEntry(
                sample_id=item.sample_id,
                language=item.language,
                target_speaker=item.target_speaker,
                other_speaker=item.other_speaker,
                snr_db=snr,
                gate_tnr=tnr,
                gate_fpr=fpr,
                processing_time_ms=elapsed_o,
                notes="mode=other",
            )
        )
    return rows


def _aggregate_by_language(entries: list[SnrSweepEntry]) -> dict[str, float]:
    """Compute per-language means + cross-language stddev for TPR/FPR.

    The aggregate dict layout:

    * ``gate_tpr_mean__<lang>``   mean TPR over target rows for ``<lang>``
    * ``gate_fpr_mean__<lang>``   mean FPR over other rows for ``<lang>``
    * ``gate_tpr_mean``           grand mean across all per-language means
    * ``gate_tpr_std_across_languages``  cross-language stddev of TPR means
    * ``gate_fpr_mean``           grand mean across all per-language means
    * ``gate_fpr_std_across_languages``  cross-language stddev of FPR means

    Cross-language stddev surfaces language-specific regressions even when
    the grand mean looks healthy.
    """
    out: dict[str, float] = {}
    if not entries:
        return out

    languages = sorted({e.language for e in entries if e.language})
    tpr_means: list[float] = []
    fpr_means: list[float] = []
    for lang in languages:
        tpr_vals = [
            e.gate_tpr
            for e in entries
            if e.language == lang and e.gate_tpr is not None and e.notes == "mode=target"
        ]
        fpr_vals = [
            e.gate_fpr
            for e in entries
            if e.language == lang and e.gate_fpr is not None and e.notes == "mode=other"
        ]
        if tpr_vals:
            m = float(np.mean(tpr_vals))
            out[f"gate_tpr_mean__{lang}"] = m
            tpr_means.append(m)
        if fpr_vals:
            m = float(np.mean(fpr_vals))
            out[f"gate_fpr_mean__{lang}"] = m
            fpr_means.append(m)

    if tpr_means:
        out["gate_tpr_mean"] = float(np.mean(tpr_means))
        out["gate_tpr_std_across_languages"] = float(np.std(tpr_means))
    if fpr_means:
        out["gate_fpr_mean"] = float(np.mean(fpr_means))
        out["gate_fpr_std_across_languages"] = float(np.std(fpr_means))
    return out


def run(
    items: list[Scenario5Item],
    provider: PipelineProvider | None = None,
    sample_rate: int = 16_000,
    output_csv: Path | None = None,
    snrs_db: tuple[float, ...] = DEFAULT_SNRS_DB,
    *,
    seed: int = 0,
) -> ScenarioResult:
    """Evaluate every item and return per-language aggregated metrics.

    ``provider`` defaults to :class:`StubPipelineProvider` so the harness
    can be exercised end-to-end in CI without torch / speechbrain.
    """
    if provider is None:
        provider = StubPipelineProvider()

    sweep = SnrSweep(scenario="scenario_5")
    rng = np.random.default_rng(seed)
    for item in items:
        rows = evaluate_one(item, provider, sample_rate, snrs_db, rng=rng)
        for row in rows:
            sweep.append(row)

    if output_csv is not None:
        sweep.write_csv(output_csv)

    aggregate = _aggregate_by_language(sweep.entries)
    return ScenarioResult(
        scenario="scenario_5",
        n_samples=len(items),
        metrics=aggregate,
        sweep=sweep,
    )
