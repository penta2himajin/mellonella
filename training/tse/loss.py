"""Negative SI-SDR training loss for target speaker extraction.

The loss is the negative scale-invariant signal-to-distortion ratio. It is
numerically consistent with the NumPy reference in
``bench/mellonella_bench/metrics/ns_quality.py`` (``si_sdr``): mean-centre
both signals, project the estimate onto the reference, and take
``10 * log10(||s_target||^2 / ||e_noise||^2)``. Minimising
``-SI-SDR`` maximises SI-SDR.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR in dB, per item in the batch.

    Parameters
    ----------
    reference:
        Clean target waveform, shape ``(B, T)`` or ``(T,)``.
    estimate:
        Extracted waveform, same shape as ``reference``.

    Returns
    -------
    Tensor of shape ``(B,)`` (or scalar for a 1-D input) with the SI-SDR
    in dB for each example.
    """
    if reference.shape != estimate.shape:
        raise ValueError(f"shape mismatch: {tuple(reference.shape)} vs {tuple(estimate.shape)}")
    squeeze = reference.dim() == 1
    if squeeze:
        reference = reference.unsqueeze(0)
        estimate = estimate.unsqueeze(0)

    ref = reference - reference.mean(dim=-1, keepdim=True)
    est = estimate - estimate.mean(dim=-1, keepdim=True)

    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + EPS
    proj = (est * ref).sum(dim=-1, keepdim=True) / ref_energy
    s_target = proj * ref
    e_noise = est - s_target

    num = (s_target * s_target).sum(dim=-1) + EPS
    den = (e_noise * e_noise).sum(dim=-1) + EPS
    value = 10.0 * torch.log10(num / den)
    return value.squeeze(0) if squeeze else value


def neg_si_sdr_loss(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative SI-SDR loss.

    Note the argument order ``(estimate, reference)`` — the convention for a
    loss function. ``reduction`` is one of ``"mean"``, ``"sum"``, ``"none"``.
    """
    per_item = -si_sdr(reference, estimate)
    if reduction == "none":
        return per_item
    if reduction == "sum":
        return per_item.sum()
    if reduction == "mean":
        return per_item.mean()
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


# ---------------------------------------------------------------------------
# Multi-resolution STFT loss and mixture-consistency penalty.
#
# These exist to attack the musical-noise / broadband-jitter artefacts that
# pure SI-SDR training leaves on Conv-TasNet outputs. The literature (Kolbæk
# 2019; Yamamoto 2020 for MR-STFT; Wisdom 2019 for mixture consistency) is
# clear that SI-SDR is deaf to the time-varying spectral structure humans
# hear as grit, and that combining a waveform-level loss with a multi-
# resolution log-magnitude STFT term reliably reduces those artefacts at
# essentially the same SI-SDR. Mixture consistency adds a physical
# constraint — the residual ``mixture - est`` should be uncorrelated with
# ``est`` — which discourages the over-suppression bursts that produce
# musical noise during overlapping-speech segments.


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Magnitude STFT, shape ``(B, F, T)``. Hann window, centred frames."""
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
    )
    return spec.abs()


def multi_resolution_stft_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    n_ffts: tuple[int, ...] = (512, 1024, 2048),
    hops: tuple[int, ...] = (128, 256, 512),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Mean over resolutions of ``spectral_convergence + log_mag_L1``.

    Yamamoto et al. (Parallel WaveGAN, 2020) formulation, widely used as a
    waveform-level perceptual regulariser. The two terms are
    complementary: spectral convergence weights spectral peaks, log-mag L1
    weights the valleys / noise floor — so the loss penalises broadband
    artefacts evenly across the dynamic range.

    `estimate` and `target` must share shape ``(B, T)``.
    """
    if estimate.shape != target.shape:
        raise ValueError(
            f"shape mismatch: {tuple(estimate.shape)} vs {tuple(target.shape)}"
        )
    if len(n_ffts) != len(hops):
        raise ValueError("n_ffts and hops must have the same length")
    losses = []
    for n_fft, hop in zip(n_ffts, hops, strict=True):
        e = _stft_mag(estimate, n_fft, hop)
        t = _stft_mag(target, n_fft, hop)
        sc = torch.linalg.norm(t - e, dim=(1, 2)) / (
            torch.linalg.norm(t, dim=(1, 2)) + eps
        )
        log_l1 = torch.mean(
            torch.abs(torch.log(t + eps) - torch.log(e + eps)),
            dim=(1, 2),
        )
        losses.append((sc + log_l1).mean())
    return torch.stack(losses).mean()


def mixture_consistency_loss(
    estimate: torch.Tensor,
    mixture: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalise correlation between the extracted target and its residual.

    Single-source form of Wisdom et al. (2019). If the residual
    ``r = mixture - estimate`` is uncorrelated with ``estimate``, the
    separator has cleanly partitioned the mixture's energy. When the
    separator over-suppresses, the residual picks up target-correlated
    bursts (= musical noise that's a distorted copy of the target).
    Penalising the normalised cross-correlation between ``estimate`` and
    ``r`` discourages that failure mode without forcing a specific
    energy partition.

    Returns the mean squared normalised correlation (per item, then mean).
    """
    if estimate.shape != mixture.shape:
        raise ValueError(
            f"shape mismatch: {tuple(estimate.shape)} vs {tuple(mixture.shape)}"
        )
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    residual = mixture - estimate
    residual = residual - residual.mean(dim=-1, keepdim=True)
    cross = (est * residual).sum(dim=-1)
    denom = (
        torch.sqrt((est * est).sum(dim=-1) + eps)
        * torch.sqrt((residual * residual).sum(dim=-1) + eps)
    )
    return ((cross / denom) ** 2).mean()


def composite_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor | None = None,
    *,
    mr_stft_weight: float = 0.0,
    mix_consist_weight: float = 0.0,
) -> torch.Tensor:
    """SI-SDR + optional MR-STFT + optional mixture-consistency.

    With both extra weights zero this is byte-identical to
    [`neg_si_sdr_loss`] so existing runs are unaffected. Set
    ``mr_stft_weight`` (typical 0.1-0.5) to attack musical noise via
    spectral-domain supervision; add ``mix_consist_weight`` (typical
    0.05-0.2) when ``mixture`` is available to additionally discourage
    over-suppression bursts.

    The MR-STFT term is in roughly the 1-10 range during training, while
    ``-SI-SDR`` is in the −20 to +20 dB range; the weight scales the
    contribution relative to that.
    """
    loss = neg_si_sdr_loss(estimate, target)
    if mr_stft_weight > 0.0:
        loss = loss + mr_stft_weight * multi_resolution_stft_loss(estimate, target)
    if mix_consist_weight > 0.0:
        if mixture is None:
            raise ValueError(
                "mix_consist_weight > 0 but mixture is None — pass the mixture tensor"
            )
        loss = loss + mix_consist_weight * mixture_consistency_loss(estimate, mixture)
    return loss
