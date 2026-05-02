"""Offline orchestrator that wires DFN3, VAD, ECAPA, F0, gating into one pass.

This is the Phase 1 deliverable described in `docs/implementation.md` §Phase 1.
The implementation is intentionally batch / non-streaming — Phase 1 only needs
to demonstrate algorithmic correctness on pre-recorded audio.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .dfn3 import DeepFilterNet3
from .embedding import EcapaTdnn
from .enrollment import EmbeddingPool
from .f0 import estimate_f0_track, f0_statistics
from .gating import GateState, apply_envelope, target_score
from .vad import SileroVAD


@dataclass
class ProcessResult:
    """Output of :func:`process_offline`.

    - ``audio`` is the gated, NS-cleaned waveform at ``Config.audio.output_sr``.
    - ``gate_decisions`` is the run-length list of ``(start_sample, is_on)``
      tuples consumed by :func:`mellonella_poc.gating.apply_envelope`. The
      first tuple's ``start_sample`` is always 0.
    - ``gate_per_frame`` is the per-SV-frame boolean gate array (one entry
      per ``Config.audio.frame_ms`` window), aligned with what bench's
      ground-truth ``voiced_mask`` uses for confusion analysis.
    """

    audio: np.ndarray
    gate_decisions: list[tuple[int, bool]] = field(default_factory=list)
    gate_per_frame: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


def expand_gate_decisions(
    decisions: list[tuple[int, bool]],
    n_samples: int,
) -> np.ndarray:
    """Expand run-length ``(start, is_on)`` decisions into a per-sample bool array."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    out = np.zeros(n_samples, dtype=bool)
    if not decisions:
        return out
    if decisions[0][0] != 0:
        raise ValueError("decisions must start at sample 0")
    boundaries = [start for start, _ in decisions] + [n_samples]
    for (start, is_on), end in zip(decisions, boundaries[1:], strict=False):
        if is_on:
            out[start:end] = True
    return out


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resample backed by `scipy.signal.resample_poly`."""
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly  # type: ignore[import-not-found]

    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


@dataclass
class PipelineComponents:
    """Container for the heavy model wrappers. Built lazily by `build_default`."""

    dfn3: DeepFilterNet3
    vad: SileroVAD
    ecapa: EcapaTdnn

    @classmethod
    def build_default(cls, config: Config) -> PipelineComponents:
        return cls(
            dfn3=DeepFilterNet3(sample_rate=config.audio.output_sr),
            vad=SileroVAD(sample_rate=config.audio.sv_sr),
            ecapa=EcapaTdnn(sample_rate=config.audio.sv_sr),
        )


def enroll_from_recording(
    audio: np.ndarray,
    sample_rate: int,
    config: Config,
    components: PipelineComponents,
    chunk_sec: float = 3.0,
    shift_sec: float = 1.5,
) -> EmbeddingPool:
    """Build an `EmbeddingPool` from a clean enrollment recording.

    Per `docs/gating.md`:
    - 5–10 anchor embeddings via sliding 3 s / 1.5 s chunks
    - F0 statistics from the voiced portion of the whole recording
    """
    sv = resample(audio, sample_rate, config.audio.sv_sr)
    chunk = int(chunk_sec * config.audio.sv_sr)
    shift = int(shift_sec * config.audio.sv_sr)
    anchors: list[np.ndarray] = []
    start = 0
    while start + chunk <= sv.size:
        anchors.append(components.ecapa.embed(sv[start : start + chunk]))
        start += shift

    if not anchors:
        if sv.size < config.audio.sv_sr:
            raise ValueError("enrollment recording shorter than 1 s; need at least 1 s of speech")
        anchors.append(components.ecapa.embed(sv[: min(sv.size, chunk)]))

    track = estimate_f0_track(sv, config.audio.sv_sr)
    mu, sigma = f0_statistics(track)

    pool = EmbeddingPool(config=config.gating)
    pool.add_anchors(anchors)
    pool.set_f0_stats(mu, sigma)
    return pool


def _frame_chunks(n_samples: int, frame_size: int) -> Iterable[tuple[int, int]]:
    start = 0
    while start < n_samples:
        end = min(start + frame_size, n_samples)
        yield start, end
        start = end


def process_offline(
    audio: np.ndarray,
    sample_rate: int,
    pool: EmbeddingPool,
    config: Config,
    components: PipelineComponents,
) -> ProcessResult:
    """Run the full pipeline end-to-end on a finite buffer.

    Returns a :class:`ProcessResult` containing the NS-cleaned, gated
    waveform along with the gate-decision artefacts needed by bench
    metrics.
    """
    if audio.ndim != 1:
        raise ValueError("audio must be a 1-D mono buffer")

    out_sr = config.audio.output_sr
    sv_sr = config.audio.sv_sr
    out48 = resample(audio, sample_rate, out_sr)
    enhanced48 = components.dfn3.process(out48)
    sv16 = resample(enhanced48, out_sr, sv_sr)

    frame_size_sv = int(config.audio.frame_ms * sv_sr / 1000)
    sv_window = int(config.audio.sv_window_sec * sv_sr)
    sv_update = int(config.audio.sv_update_ms * sv_sr / 1000)
    speech_buffer: list[np.ndarray] = []
    samples_since_update = 0
    last_score = 0.0
    last_f0 = 0.0

    gate_state = GateState(config=config.gating)
    decisions: list[tuple[int, bool]] = []
    current_decision: bool | None = None
    per_frame: list[bool] = []

    for start_sv, end_sv in _frame_chunks(sv16.size, frame_size_sv):
        frame = sv16[start_sv:end_sv]
        if frame.size < frame_size_sv:
            break

        speech_prob = components.vad.score(frame)
        if speech_prob > 0.5:
            speech_buffer.append(frame)
            total = sum(b.size for b in speech_buffer)
            while total > sv_window and speech_buffer:
                drop = speech_buffer.pop(0)
                total -= drop.size
        samples_since_update += frame.size

        if samples_since_update >= sv_update and sum(b.size for b in speech_buffer) >= sv_window:
            samples_since_update = 0
            window = np.concatenate(speech_buffer)[-sv_window:]
            embedding = components.ecapa.embed(window)
            f0_track = estimate_f0_track(window, sv_sr)
            f0_mean, _ = f0_statistics(f0_track)
            last_f0 = f0_mean
            last_score = target_score(
                embedding=embedding,
                pool=pool,
                f0_mean=last_f0,
                f0_mu=pool.metadata.f0_mu,
                f0_sigma=pool.metadata.f0_sigma,
                config=config.gating,
            )

        is_on = gate_state.update(last_score, dt_ms=config.audio.frame_ms)
        per_frame.append(is_on)

        out_start = int(start_sv * out_sr / sv_sr)
        if current_decision is None or current_decision != is_on:
            decisions.append((out_start, is_on))
            current_decision = is_on

    if not decisions:
        decisions = [(0, False)]
    elif decisions[0][0] != 0:
        decisions.insert(0, (0, decisions[0][1]))

    output = apply_envelope(enhanced48, decisions, sample_rate=out_sr, config=config.gating)
    return ProcessResult(
        audio=output,
        gate_decisions=decisions,
        gate_per_frame=np.asarray(per_frame, dtype=bool),
    )
