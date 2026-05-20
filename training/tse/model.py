"""Causal Conv-TasNet target speaker extraction model.

SpeakerBeam-style FiLM conditioning on a frozen 192-dim ECAPA enrollment
embedding. The model supports two numerically-equivalent forward modes:

* :meth:`CausalConvTasNetTSE.forward` — full-sequence, for training.
* :meth:`CausalConvTasNetTSE.forward_streaming` — one fixed chunk at a time
  with explicit causal-conv state buffers threaded in and out. This mode is
  what exports to ONNX.

Both modes share the same weights and produce identical output (within
~1e-4) for the same input. Causality everywhere is enforced by *left*
padding only, cumulative (causal) layer norm, and explicit ring-buffer
state for every dilated depthwise conv plus the encoder/decoder overlap.

State layout
------------
``forward_streaming`` threads a flat ``list[Tensor]`` of conv states. The
order is fixed (see :meth:`CausalConvTasNetTSE.make_initial_state`):

    [enc_overlap]                                 1 tensor
    [input_norm_sum, input_norm_sqsum, input_norm_count]
                                                  3 tensors
    per TCN block, in (repeat, block) order:
      [dw_ringbuffer,
       cln1_sum, cln1_sqsum, cln1_count,
       cln2_sum, cln2_sqsum, cln2_count]          7 tensors / block
    [dec_overlap]                                 1 tensor

Total: ``1 + 3 + 7 * n_repeats * n_blocks + 1`` tensors.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .config import TSEConfig

# ---------------------------------------------------------------------------
# Cumulative (causal) layer norm
# ---------------------------------------------------------------------------


class CumulativeLayerNorm(nn.Module):
    """Channel-wise cumulative layer norm — causal-safe.

    Normalises each time step by the running mean/variance over *all
    channels and all time steps up to and including* that step. This is
    the cumulative layer norm from the Conv-TasNet paper; unlike global
    LN it never peeks at the future, so the full-sequence and streaming
    paths produce identical results.

    Input/output shape: ``(B, C, T)``.
    """

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        chan_sum = x.sum(dim=1, keepdim=True)  # (B,1,T)
        chan_sqsum = (x * x).sum(dim=1, keepdim=True)  # (B,1,T)
        cum_sum = torch.cumsum(chan_sum, dim=2)
        cum_sqsum = torch.cumsum(chan_sqsum, dim=2)
        count = torch.arange(1, t + 1, device=x.device, dtype=x.dtype).view(1, 1, t) * c
        mean = cum_sum / count
        var = (cum_sqsum / count - mean * mean).clamp(min=0.0)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.gamma + self.beta

    def forward_streaming(
        self,
        x: torch.Tensor,
        run_mean: torch.Tensor,
        run_sqmean: torch.Tensor,
        run_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming cumulative LN over one chunk.

        State is carried as the *running mean* and *running mean-of-squares*
        (plus the element count), all shape ``(B,1,1)``. Carrying means
        rather than raw sums keeps the state O(1) regardless of how long the
        stream has run — raw cumulative sums grow unboundedly and lose
        float32 precision (and ONNX accumulates them in a different order),
        which otherwise breaks per-chunk parity. Returns the normalised
        chunk plus the updated ``(mean, sqmean, count)`` triple.
        """
        b, c, t = x.shape
        chan_sum = x.sum(dim=1, keepdim=True)
        chan_sqsum = (x * x).sum(dim=1, keepdim=True)
        # Cumulative within-chunk sums, offset by the running totals
        # reconstructed from the carried means.
        cum_sum = torch.cumsum(chan_sum, dim=2) + run_mean * run_count
        cum_sqsum = torch.cumsum(chan_sqsum, dim=2) + run_sqmean * run_count
        steps = torch.arange(1, t + 1, device=x.device, dtype=x.dtype).view(1, 1, t) * c
        count = steps + run_count
        mean = cum_sum / count
        sqmean = cum_sqsum / count
        var = (sqmean - mean * mean).clamp(min=0.0)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        out = normed * self.gamma + self.beta
        return out, mean[:, :, -1:], sqmean[:, :, -1:], count[:, :, -1:]


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------


class FiLMConditioner(nn.Module):
    """2-layer MLP mapping the frozen enrollment embedding to FiLM (gamma, beta).

    ``cond_dim -> film_hidden -> 2 * bottleneck``. The same (gamma, beta)
    pair is applied (broadcast over time) inside every TCN block. The ECAPA
    model itself is *not* part of this module — the embedding is a plain
    input tensor of shape ``(B, cond_dim)``.
    """

    def __init__(self, cond_dim: int, film_hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.net = nn.Sequential(
            nn.Linear(cond_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * bottleneck),
        )

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(gamma, beta)``, each shaped ``(B, bottleneck, 1)``."""
        gb = self.net(cond)
        gamma, beta = gb.split(self.bottleneck, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Causal TCN block
# ---------------------------------------------------------------------------


class CausalTCNBlock(nn.Module):
    """One depthwise-separable causal conv block of the TCN separator.

    Pipeline: 1x1 conv (B->H) -> PReLU -> cumulative LN -> causal dilated
    depthwise conv (H) -> PReLU -> cumulative LN -> 1x1 conv (H->B) ->
    FiLM(gamma, beta) -> residual + skip. FiLM modulates the B-dimensional
    projected feature (SpeakerBeam-style conditioning at the bottleneck);
    gamma/beta are size ``B``, which is why FiLM is applied *after* the
    H->B projection rather than on the H-dim depthwise output. The dilated
    depthwise conv is made causal by left-padding only ``(kernel - 1) *
    dilation`` samples; in streaming mode that left context comes from a
    per-block ring buffer instead.
    """

    def __init__(self, config: TSEConfig, dilation: int) -> None:
        super().__init__()
        b = config.bottleneck
        h = config.hidden
        p = config.tcn_kernel
        self.dilation = dilation
        self.kernel = p
        self.pad = (p - 1) * dilation
        self.hidden = h

        self.conv1x1_in = nn.Conv1d(b, h, kernel_size=1)
        self.prelu1 = nn.PReLU(h)
        self.norm1 = CumulativeLayerNorm(h, eps=config.ln_eps)
        self.dw_conv = nn.Conv1d(h, h, kernel_size=p, dilation=dilation, groups=h, padding=0)
        self.prelu2 = nn.PReLU(h)
        self.norm2 = CumulativeLayerNorm(h, eps=config.ln_eps)
        self.conv1x1_res = nn.Conv1d(h, b, kernel_size=1)
        self.conv1x1_skip = nn.Conv1d(h, b, kernel_size=1)

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-sequence forward. Returns ``(residual_out, skip_out)``."""
        y = self.conv1x1_in(x)
        y = self.prelu1(y)
        y = self.norm1(y)
        y = F.pad(y, (self.pad, 0))  # causal: left-pad only
        y = self.dw_conv(y)
        y = self.prelu2(y)
        y = self.norm2(y)
        residual = self.conv1x1_res(y) * gamma + beta  # FiLM at the bottleneck
        skip = self.conv1x1_skip(y) * gamma + beta
        return x + residual, skip

    def forward_streaming(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        dw_ring: torch.Tensor,
        cln1: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cln2: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Streaming forward for one chunk.

        ``dw_ring`` is the ``(B, H, pad)`` ring buffer of the depthwise
        conv's most recent past inputs. ``cln1`` / ``cln2`` are the
        ``(sum, sqsum, count)`` triples for the two cumulative LNs.

        Returns ``(residual_out, skip_out, new_dw_ring, new_cln1, new_cln2)``.
        """
        y = self.conv1x1_in(x)
        y = self.prelu1(y)
        y, *new_cln1 = self.norm1.forward_streaming(y, *cln1)
        # Prepend ring-buffer context instead of zero-padding.
        if self.pad > 0:
            y_ctx = torch.cat([dw_ring, y], dim=2)
            new_ring = y_ctx[:, :, -self.pad :]
        else:  # pragma: no cover - tcn_kernel is always > 1 in practice
            y_ctx = y
            new_ring = dw_ring
        y = self.dw_conv(y_ctx)
        y = self.prelu2(y)
        y, *new_cln2 = self.norm2.forward_streaming(y, *cln2)
        residual = self.conv1x1_res(y) * gamma + beta
        skip = self.conv1x1_skip(y) * gamma + beta
        return (
            x + residual,
            skip,
            new_ring,
            (new_cln1[0], new_cln1[1], new_cln1[2]),
            (new_cln2[0], new_cln2[1], new_cln2[2]),
        )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class CausalConvTasNetTSE(nn.Module):
    """Causal Conv-TasNet target speaker extraction network.

    See the module docstring for the two forward modes and the streaming
    state layout.
    """

    def __init__(self, config: TSEConfig | None = None) -> None:
        super().__init__()
        self.config = config or TSEConfig.poc_16k()
        cfg = self.config

        # Encoder: 1-D conv + ReLU. Made causal by left-padding the overlap.
        self.encoder = nn.Conv1d(
            1, cfg.n_basis, kernel_size=cfg.enc_kernel, stride=cfg.enc_stride, bias=False
        )
        # Pre-separator: cumulative LN + 1x1 bottleneck projection.
        self.input_norm = CumulativeLayerNorm(cfg.n_basis, eps=cfg.ln_eps)
        self.bottleneck_proj = nn.Conv1d(cfg.n_basis, cfg.bottleneck, kernel_size=1)

        # FiLM conditioning MLP.
        self.film = FiLMConditioner(cfg.cond_dim, cfg.film_hidden, cfg.bottleneck)

        # TCN separator: R repeats x X blocks.
        self.blocks = nn.ModuleList()
        for _r in range(cfg.n_repeats):
            for d in cfg.dilations:
                self.blocks.append(CausalTCNBlock(cfg, dilation=d))

        # Mask head: PReLU -> 1x1 conv -> activation.
        self.mask_prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(cfg.bottleneck, cfg.n_basis, kernel_size=1)

        # Decoder: transposed conv, mirror of the encoder.
        self.decoder = nn.ConvTranspose1d(
            cfg.n_basis, 1, kernel_size=cfg.enc_kernel, stride=cfg.enc_stride, bias=False
        )

    # -- shared helpers -----------------------------------------------------

    def _apply_mask_act(self, m: torch.Tensor) -> torch.Tensor:
        if self.config.mask_act == "sigmoid":
            return torch.sigmoid(m)
        return F.relu(m)

    @property
    def n_state_tensors(self) -> int:
        """Number of tensors in the flat streaming-state list.

        Layout: 1 enc-overlap + 3 input-norm + 7 per TCN block + 1
        dec-overlap + 1 mask-EMA carry. The trailing mask-EMA tensor
        is always present (it threads the temporal mask smoother's
        `prev` frame across chunks); with ``mask_smoothing_beta == 0``
        the smoother is the identity, so the tensor is inert but kept
        so the state layout doesn't depend on the beta value.
        """
        return 1 + 3 + 7 * len(self.blocks) + 1 + 1

    def receptive_field_samples(self) -> int:
        """Total causal receptive field of the separator, in input samples."""
        cfg = self.config
        latent = 1 + sum(cast(CausalTCNBlock, block).pad for block in self.blocks)
        # Each latent frame covers enc_stride input samples; plus the
        # encoder analysis window overlap.
        return latent * cfg.enc_stride + cfg.enc_overlap

    # -- full-sequence forward ---------------------------------------------

    def forward(self, mixture: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward (training mode).

        Parameters
        ----------
        mixture:
            Mixture waveform, shape ``(B, T)`` or ``(B, 1, T)``.
        cond:
            Frozen enrollment embedding, shape ``(B, cond_dim)``.

        Returns
        -------
        Extracted target waveform, shape ``(B, T)`` (same length as input).
        """
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)
        cfg = self.config
        in_len = mixture.shape[-1]

        # Left-pad for causality; right-pad so length is a stride multiple.
        x = F.pad(mixture, (cfg.enc_overlap, 0))
        rem = (x.shape[-1] - cfg.enc_kernel) % cfg.enc_stride
        if rem != 0:
            x = F.pad(x, (0, cfg.enc_stride - rem))

        enc = F.relu(self.encoder(x))  # (B, N, L)
        feat = self.input_norm(enc)
        feat = self.bottleneck_proj(feat)  # (B, B, L)

        gamma, beta = self.film(cond)

        skip_sum = torch.zeros(
            feat.shape[0], cfg.bottleneck, feat.shape[2], device=feat.device, dtype=feat.dtype
        )
        h = feat
        for block in self.blocks:
            h, skip = block(h, gamma, beta)
            skip_sum = skip_sum + skip

        m = self.mask_prelu(skip_sum)
        m = self.mask_conv(m)
        mask = self._apply_mask_act(m)  # (B, N, L)

        # Temporal mask smoothing (matches `forward_streaming`), seeded
        # from a zero carry so the full-sequence and chunked-streaming
        # masks agree frame-for-frame (the `verify` parity check relies
        # on this). `beta` here is `mask_smoothing_beta`, not the FiLM
        # beta bound above.
        smooth_beta = cfg.mask_smoothing_beta
        smoothed_frames: list[torch.Tensor] = []
        cur_mask = torch.zeros(
            mask.shape[0], mask.shape[1], 1, device=mask.device, dtype=mask.dtype
        )
        for t in range(mask.shape[2]):
            cur_mask = smooth_beta * cur_mask + (1.0 - smooth_beta) * mask[:, :, t : t + 1]
            smoothed_frames.append(cur_mask)
        mask = torch.cat(smoothed_frames, dim=2)

        masked = enc * mask
        out = self.decoder(masked)  # (B, 1, T_dec)
        # The transposed-conv decoded stream starts at decoded-index 0; this
        # is the same alignment the streaming path emits (the encoder's
        # ``enc_overlap`` left-pad is the analysis-window context, not an
        # output offset). Trim to the original input length.
        out = out[:, :, :in_len]
        if out.shape[-1] < in_len:
            out = F.pad(out, (0, in_len - out.shape[-1]))
        return out.squeeze(1)

    # -- streaming forward --------------------------------------------------

    def make_initial_state(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> list[torch.Tensor]:
        """Build a zero-initialised flat streaming-state list.

        Order: ``[enc_overlap]``, ``[input_norm sum/sqsum/count]``, then per
        block ``[dw_ring, cln1 sum/sqsum/count, cln2 sum/sqsum/count]``, then
        ``[dec_overlap]``. See the module docstring.
        """
        cfg = self.config
        dev = torch.device(device)
        z11 = lambda: torch.zeros(batch_size, 1, 1, device=dev)  # noqa: E731
        state: list[torch.Tensor] = [
            torch.zeros(batch_size, 1, cfg.enc_overlap, device=dev),  # enc overlap
            z11(),
            z11(),
            z11(),  # input_norm sum/sqsum/count
        ]
        for block in self.blocks:
            block = cast(CausalTCNBlock, block)
            state.append(torch.zeros(batch_size, cfg.hidden, block.pad, device=dev))
            state.extend([z11(), z11(), z11()])  # cln1
            state.extend([z11(), z11(), z11()])  # cln2
        state.append(torch.zeros(batch_size, 1, cfg.enc_overlap, device=dev))  # dec overlap
        # Mask-EMA carry: last smoothed mask frame, shape (B, n_basis, 1).
        state.append(torch.zeros(batch_size, cfg.n_basis, 1, device=dev))
        return state

    def forward_streaming(
        self,
        chunk: torch.Tensor,
        cond: torch.Tensor,
        conv_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process one fixed-size chunk, threading causal-conv state.

        Parameters
        ----------
        chunk:
            One mixture chunk, shape ``(B, chunk_len)`` or ``(B, 1, chunk_len)``.
            ``chunk_len`` must be a positive multiple of ``enc_stride``.
        cond:
            Frozen enrollment embedding, shape ``(B, cond_dim)``.
        conv_states:
            Flat state list from :meth:`make_initial_state` (or the previous
            call's return value).

        Returns
        -------
        ``(extracted_chunk, new_conv_states)`` — ``extracted_chunk`` has shape
        ``(B, chunk_len)``.
        """
        if chunk.dim() == 2:
            chunk = chunk.unsqueeze(1)
        cfg = self.config
        chunk_len = chunk.shape[-1]
        if chunk_len <= 0 or chunk_len % cfg.enc_stride != 0:
            raise ValueError(
                f"chunk length {chunk_len} must be a positive multiple of "
                f"enc_stride ({cfg.enc_stride})"
            )
        expected = self.n_state_tensors
        if len(conv_states) != expected:
            raise ValueError(f"conv_states has {len(conv_states)} tensors, expected {expected}")

        states = list(conv_states)
        idx = 0

        # --- encoder (overlap state prepended) ---
        enc_overlap_state = states[idx]
        idx += 1
        x = torch.cat([enc_overlap_state, chunk], dim=2)
        new_enc_overlap = x[:, :, -cfg.enc_overlap :] if cfg.enc_overlap > 0 else enc_overlap_state
        enc = F.relu(self.encoder(x))  # (B, N, L), L = chunk_len / stride

        # --- input cumulative LN (streaming) ---
        in_sum, in_sqsum, in_count = states[idx], states[idx + 1], states[idx + 2]
        idx += 3
        feat, n_in_sum, n_in_sqsum, n_in_count = self.input_norm.forward_streaming(
            enc, in_sum, in_sqsum, in_count
        )
        feat = self.bottleneck_proj(feat)

        gamma, beta = self.film(cond)

        skip_sum = torch.zeros(
            feat.shape[0], cfg.bottleneck, feat.shape[2], device=feat.device, dtype=feat.dtype
        )
        h = feat
        new_states: list[torch.Tensor] = [new_enc_overlap, n_in_sum, n_in_sqsum, n_in_count]
        for block in self.blocks:
            block = cast(CausalTCNBlock, block)
            dw_ring = states[idx]
            cln1 = (states[idx + 1], states[idx + 2], states[idx + 3])
            cln2 = (states[idx + 4], states[idx + 5], states[idx + 6])
            idx += 7
            h, skip, new_ring, n1, n2 = block.forward_streaming(h, gamma, beta, dw_ring, cln1, cln2)
            skip_sum = skip_sum + skip
            new_states.extend([new_ring, n1[0], n1[1], n1[2], n2[0], n2[1], n2[2]])

        m = self.mask_prelu(skip_sum)
        m = self.mask_conv(m)
        mask = self._apply_mask_act(m)
        # Temporal mask smoothing (Option B): low-pass the per-frame
        # mask along time with a one-pole EMA, threading the last
        # frame across chunks via the trailing state tensor. This
        # suppresses the frame-to-frame mask flicker that Conv-TasNet
        # produces in ambiguous regions (the "ジャギジャギ" musical
        # noise). `beta == 0` makes the recurrence the identity, so
        # the state is inert and the output is byte-identical to the
        # unsmoothed model.
        prev_mask_ema = states[-1]  # (B, n_basis, 1)
        beta = cfg.mask_smoothing_beta
        smoothed_frames: list[torch.Tensor] = []
        cur_mask = prev_mask_ema
        for t in range(mask.shape[2]):
            cur_mask = beta * cur_mask + (1.0 - beta) * mask[:, :, t : t + 1]
            smoothed_frames.append(cur_mask)
        mask = torch.cat(smoothed_frames, dim=2)
        new_mask_ema = cur_mask
        masked = enc * mask
        dec = self.decoder(masked)  # (B, 1, chunk_len + enc_overlap)

        # --- decoder overlap-add ---
        dec_overlap_state = states[idx]
        idx += 1
        ov = cfg.enc_overlap
        if ov > 0:
            head = dec[:, :, :ov] + dec_overlap_state
            body = dec[:, :, ov:chunk_len]
            new_dec_overlap = dec[:, :, chunk_len:]  # length ov
            out = torch.cat([head, body], dim=2)
        else:  # pragma: no cover - kernel == stride is unusual
            out = dec[:, :, :chunk_len]
            new_dec_overlap = dec_overlap_state
        new_states.append(new_dec_overlap)
        new_states.append(new_mask_ema)
        return out.squeeze(1), new_states


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Total parameter count of ``model``."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
