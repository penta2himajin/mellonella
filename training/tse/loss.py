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
