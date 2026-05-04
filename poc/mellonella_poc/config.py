"""Default thresholds and tunable parameters.

Initial values follow `docs/gating.md` and `docs/architecture.md`. They are
subject to revision after Phase 1 measurements; treat any number here as a
starting point, not a contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GatingConfig:
    """Thresholds and weights for the integrated speaker gate."""

    alpha: float = 0.9
    """Weight applied to cosine similarity in the integrated score.

    Recalibrated from the docs/decisions.md D-005 initial value of 0.8
    via the joint α / θ_pass sweep in
    ``docs/benchmarks/calibration_alpha_beta_summary.json``. On the
    librosa libri1/2/3 + white/pink noise grid, α=0.9 reaches the same
    TPR_median (~0.84) as α=0.8 at θ_pass=0.30 with FPR_mean reduced
    from 0.046 to 0.017. Caveat: the calibration speakers all share an
    English LibriSpeech distribution; a wider F0 spread across speakers
    will likely raise the optimal β again.
    """

    beta: float = 0.1
    """Weight applied to F0 match in the integrated score. alpha + beta == 1.0."""

    theta_pass: float = 0.30
    """Output gate threshold. Below this, frames are muted.

    Recalibrated from the docs/gating.md initial value of 0.50 against
    real recordings + noise (white + pink, SNRs -5..20 dB, librosa
    libri1/2/3) — see ``docs/benchmarks/calibration_summary.json`` and
    ``scripts/calibrate.py``. The 0.50 default closed the gate
    completely under any realistic noise; 0.30 yields a median TPR of
    ~0.84 with mean FPR ~4.6 % across the calibration grid, in line
    with the FP-tolerant policy in D-004.
    """

    theta_learn: float = 0.80
    """Auto-learning threshold. Strictly greater than theta_pass."""

    theta_f0: float = 0.7
    """Minimum f0_match required before an embedding can join auto-learn."""

    hangover_ms: float = 300.0
    """How long the gate stays ON after target_score drops below theta_pass."""

    attack_ms: float = 15.0
    """Envelope attack time in milliseconds (fast, avoids click)."""

    release_ms: float = 100.0
    """Envelope release time in milliseconds (slow, avoids cutoff)."""

    anchor_distance_threshold: float = 0.4
    """Maximum 1 - max_cos(anchors) allowed before a candidate is rejected."""

    anchor_reset_threshold: float = 0.5
    """Median anchor distance over auto-learn pool that triggers reset."""

    auto_learn_max_size: int = 20
    """FIFO bound on auto-learning embeddings."""

    min_continuous_speech_sec: float = 1.0
    """Minimum speech run length before auto-learn admission."""

    enable_auto_learn: bool = True
    """Master switch for auto-learn admission inside `process_offline`.

    When False, embeddings are never added to the auto-learn FIFO and the
    pool's drift-reset hook is never invoked. Used for ablation runs and
    tests that want a deterministic, time-invariant pool.
    """

    use_as_norm: bool = False
    """Adaptive S-Norm (`docs/decisions.md` D-010, `docs/gating.md`).

    When True, the gating decision uses
    ``z = (cos_sim_max - μ_topK(S_impostor)) / σ_topK(S_impostor)``
    against :attr:`theta_pass_as_norm` instead of the
    ``α · cs + β · f0_match`` mixture against :attr:`theta_pass`. F0 stops
    contributing to the per-frame gate decision (it is still consulted
    for auto-learn admission via :attr:`theta_f0`). Default False to keep
    every existing PoC + bench test path bit-identical until a cohort and
    fresh calibration are wired up.
    """

    as_norm_top_k: int = 10
    """Number of top-scoring impostor cohort entries used in the AS-Norm
    z-score numerator/denominator. Standard literature picks 10–50; we
    default to 10 because the shipped cohort is small (50 speakers across
    6 languages, see `bench/data/cohorts/`)."""

    as_norm_cohort_path: str | None = None
    """Filesystem path to the impostor cohort ``.npz`` produced by
    :mod:`scripts/build_impostor_cohort`. Loaded once when
    :class:`PipelineComponents` is built; ignored when
    :attr:`use_as_norm` is False."""

    theta_pass_as_norm: float = 1.5
    """Output gate threshold when :attr:`use_as_norm` is True (z-score scale).

    Initial value is heuristic — proper calibration with the real cohort
    happens in AS-Norm Phase 3. ``1.5`` corresponds to "target embedding
    is ~1.5 σ above the top-K impostor distribution", which is the
    classic operating point in the AS-Norm literature.
    """

    theta_learn_as_norm: float = 2.5
    """Auto-learn admission threshold under AS-Norm (z-score scale). Higher
    than :attr:`theta_pass_as_norm` so only confidently-target frames feed
    the pool, mirroring the ``theta_learn > theta_pass`` invariant from
    the legacy mixed-score path."""

    def __post_init__(self) -> None:
        if self.theta_pass >= self.theta_learn:
            raise ValueError("theta_pass must be strictly less than theta_learn")
        if self.theta_pass_as_norm >= self.theta_learn_as_norm:
            raise ValueError("theta_pass_as_norm must be strictly less than theta_learn_as_norm")
        if not 0.99 <= self.alpha + self.beta <= 1.01:
            raise ValueError("alpha + beta must equal 1.0")
        if self.as_norm_top_k <= 0:
            raise ValueError("as_norm_top_k must be > 0")


@dataclass(frozen=True)
class AudioConfig:
    """Sampling-rate plumbing matching `docs/architecture.md`."""

    output_sr: int = 48_000
    """DFN3 native rate; also the final pipeline output rate."""

    sv_sr: int = 16_000
    """ECAPA-TDNN native rate; used for VAD/SV/F0 stages."""

    frame_ms: float = 20.0
    """Streaming frame size at output_sr (envelope / chunking cadence)."""

    vad_frame_samples: int = 512
    """Number of samples per silero-vad call.

    silero-vad >= 6.0 enforces a hard 512-sample chunk at 16 kHz (32 ms)
    and 256 samples at 8 kHz; older versions accepted arbitrary lengths.
    The pipeline's gate / hangover / envelope cadence is driven by this
    same step, so any change here also changes ``dt_ms`` for the gate.
    """

    sv_window_sec: float = 1.0
    """Window length used to compute speaker embeddings."""

    sv_update_ms: float = 250.0
    """How often a new embedding is computed during continuous speech."""

    @property
    def vad_frame_ms(self) -> float:
        """Convenience: VAD frame duration in milliseconds (read-only)."""
        return 1000.0 * self.vad_frame_samples / self.sv_sr


@dataclass(frozen=True)
class Config:
    """Top-level configuration container."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
