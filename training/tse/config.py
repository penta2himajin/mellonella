"""Configuration for the causal Conv-TasNet target speaker extraction model.

The same code path serves a cheap 16 kHz proof-of-concept and a 48 kHz
production model. The *only* differences are the sample rate and the
encoder kernel/stride — and those are scaled so the latent frame rate
(``sample_rate / stride``) stays constant at 1 kHz across both. Everything
downstream of the encoder (the TCN separator, the FiLM conditioning, the
mask, the decoder) is rate-agnostic and identical for PoC and production.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TSEConfig:
    """Frozen hyper-parameter bundle for :class:`~tse.model.CausalConvTasNetTSE`.

    Attributes
    ----------
    sample_rate:
        Audio sample rate in Hz.
    enc_kernel / enc_stride:
        1-D conv encoder kernel size and hop. ``enc_stride`` sets the latent
        frame rate (``sample_rate / enc_stride``). PoC: 32/16 @ 16 kHz.
        Prod: 96/48 @ 48 kHz — same 1 kHz latent rate.
    n_basis:
        Encoder basis channel count ``N``.
    bottleneck:
        TCN bottleneck channel count ``B`` (also the FiLM gamma/beta size).
    hidden:
        TCN depthwise-separable conv hidden channels ``H``.
    tcn_kernel:
        Depthwise conv kernel size ``P``.
    n_blocks:
        Number of conv blocks per repeat ``X`` (dilations 1,2,4,...,2**(X-1)).
    n_repeats:
        Number of TCN repeats ``R``.
    cond_dim:
        Dimensionality of the frozen enrollment embedding (ECAPA = 192).
    film_hidden:
        Hidden width of the 2-layer FiLM conditioning MLP.
    mask_act:
        Activation applied to the estimated mask ("sigmoid" or "relu").
    ln_eps:
        Epsilon for the cumulative layer norm.
    """

    sample_rate: int = 16_000
    enc_kernel: int = 32
    enc_stride: int = 16
    n_basis: int = 256
    bottleneck: int = 128
    hidden: int = 256
    tcn_kernel: int = 3
    n_blocks: int = 6
    n_repeats: int = 2
    cond_dim: int = 192
    film_hidden: int = 256
    mask_act: str = "sigmoid"
    ln_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.enc_kernel % self.enc_stride != 0:
            raise ValueError(
                f"enc_kernel ({self.enc_kernel}) must be a multiple of "
                f"enc_stride ({self.enc_stride}) for clean streaming overlap"
            )
        if self.enc_kernel < self.enc_stride:
            raise ValueError("enc_kernel must be >= enc_stride")
        if self.mask_act not in ("sigmoid", "relu"):
            raise ValueError(f"mask_act must be 'sigmoid' or 'relu', got {self.mask_act!r}")

    # -- derived quantities -------------------------------------------------

    @property
    def latent_hop(self) -> int:
        """Samples advanced per latent frame (== ``enc_stride``)."""
        return self.enc_stride

    @property
    def latent_rate_hz(self) -> float:
        """Latent frame rate in Hz. Held constant across PoC and prod."""
        return self.sample_rate / self.enc_stride

    @property
    def enc_overlap(self) -> int:
        """Encoder/decoder analysis-window overlap (``kernel - stride``)."""
        return self.enc_kernel - self.enc_stride

    @property
    def dilations(self) -> tuple[int, ...]:
        """Per-repeat dilation schedule: 1, 2, 4, ..., 2**(n_blocks-1)."""
        return tuple(2**i for i in range(self.n_blocks))

    def dilation_for(self, repeat: int, block: int) -> int:
        """Dilation for the ``block``-th conv in the ``repeat``-th TCN repeat."""
        return self.dilations[block]

    # -- presets ------------------------------------------------------------

    @classmethod
    def poc_16k(cls) -> TSEConfig:
        """The default proof-of-concept config: 16 kHz, kernel 32 / stride 16."""
        return cls(
            sample_rate=16_000,
            enc_kernel=32,
            enc_stride=16,
        )

    @classmethod
    def prod_48k(cls) -> TSEConfig:
        """Production config: 48 kHz, kernel 96 / stride 48.

        Encoder kernel/stride are tripled vs. the PoC so the latent frame
        rate stays at 1 kHz; the separator is byte-identical to the PoC.
        """
        return cls(
            sample_rate=48_000,
            enc_kernel=96,
            enc_stride=48,
        )

    def with_overrides(self, **kwargs: object) -> TSEConfig:
        """Return a copy with selected fields replaced (e.g. for tests)."""
        return replace(self, **kwargs)  # type: ignore[arg-type]
