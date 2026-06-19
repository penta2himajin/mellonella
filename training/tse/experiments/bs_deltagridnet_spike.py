"""BS-DeltaGridNet trainability spike — stable Gated-DeltaNet time-axis core.

This is an **exploratory spike**, not production code. It validates that the
time-axis core of the north-star BS-DeltaGridNet architecture
(``docs/bs-deltagridnet.md``) can be implemented in **pure PyTorch (no Triton)**
in a numerically stable, trainable, ONNX-exportable form.

Findings captured here (see ``docs/bs-deltagridnet-trainability.md`` for the
full write-up):

* A naive Gated-DeltaNet-2 recurrence with a *channel-wise* erase gate
  ``e = b ⊙ k`` is **not a contraction** (the rank-1 term is asymmetric) and the
  state diverges (|state| → 1e37 in ~5 steps) across seeds, on both CPU and GPU.
* The fix is the standard stabilisation recipe from the DeltaNet / Gated
  DeltaNet / Mamba-2 / flash-linear-attention literature:
    - **L2-normalise q and k** (unit keys are the contraction precondition),
    - **symmetric Householder erase** ``I − β·kkᵀ`` with a *scalar* per-head
      ``β ∈ (0, 2)`` (eigenvalue ``1 − β`` stays in ``(−1, 1)``),
    - **channel-wise decay** ``α ∈ (0, 1)`` in **log space** (Mamba-2
      ``exp(−softplus(·))``),
    - **post-cell RMSNorm + SiLU output gate**.
  With this, ``‖state‖`` stays ≈ 1 across seeds and the core trains cleanly.
* This stable form keeps Gated-DeltaNet-2's *channel-wise write gate* and
  *channel-wise decay*, but drops the *channel-wise erase* (scalar β instead) —
  the channel-wise erase is the part that genuinely needs the WY chunkwise
  algorithm to stay stable, i.e. the validated NVlabs / flash-linear-attention
  kernel path.

Throughput: the per-timestep Python recurrence here is **launch-bound** on GPU
(~28k lane-tok/s on a T4 → ~114 h/epoch). The **chunkwise matmul (WY)** form
(``chunkwise_gated_delta``) re-expresses the same recurrence as batched matmuls +
one triangular solve per chunk and measured ~2.3M lane-tok/s on a T4 (~84×
faster, ~5.5 h/epoch, parity-exact) — making a free-tier T4 viable.

Run (CPU, no datasets, no GPU)::

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime
    python -m tse.experiments.bs_deltagridnet_spike
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain


class StableGatedDeltaCore(nn.Module):
    """Numerically-stable Gated-DeltaNet-style causal time-axis core.

    Per-head state ``state`` is ``[d_k, d_v]``. The single-step recurrence is::

        decayed = diag(alpha) @ state          # channel-wise log-space decay
        pred    = kᵀ decayed                    # current value stored at key k
        state   = decayed − beta·k·predᵀ + beta·k·(w⊙v)ᵀ     # symmetric erase + gated write
        o       = stateᵀ q

    The key-side operator ``(I − beta·kkᵀ)·diag(alpha)`` is non-expansive for
    unit ``k`` and ``beta ∈ (0, 2)``, ``alpha ∈ (0, 1)`` — so ``state`` stays
    bounded. ``forward`` runs the per-step recurrence; ``forward_parallel`` is an
    exact affine-scan reference for parity; ``step`` is the ONNX streaming path.
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.n_heads, self.dk, self.dv = n_heads, d_k, d_v
        self.q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.wg = nn.Linear(d_model, n_heads * d_v)  # channel-wise write gate
        self.decay = nn.Linear(d_model, n_heads * d_k)  # channel-wise decay (log-space)
        self.beta = nn.Linear(d_model, n_heads)  # scalar erase strength / head
        self.g = nn.Linear(d_model, n_heads * d_v)  # output gate
        self.norm = RMSNorm(n_heads * d_v)
        self.out = nn.Linear(n_heads * d_v, d_model, bias=False)
        nn.init.constant_(self.decay.bias, 2.0)

    def _proj(self, x: torch.Tensor):
        n, t, _ = x.shape
        h, dk, dv = self.n_heads, self.dk, self.dv
        q = F.normalize(self.q(x).view(n, t, h, dk), dim=-1)
        k = F.normalize(self.k(x).view(n, t, h, dk), dim=-1)
        v = self.v(x).view(n, t, h, dv)
        w = torch.sigmoid(self.wg(x).view(n, t, h, dv))
        alpha = torch.exp(-F.softplus(self.decay(x).view(n, t, h, dk)))  # (0, 1)
        beta = 2.0 * torch.sigmoid(self.beta(x).view(n, t, h))  # (0, 2)
        return q, k, v, w, alpha, beta, self.g(x)

    def _cell(self, q, k, v, w, alpha, beta, state):
        outs = []
        for t in range(q.shape[1]):
            kt, qt = k[:, t], q[:, t]
            zt = w[:, t] * v[:, t]
            bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            decayed = alpha[:, t].unsqueeze(-1) * state
            pred = torch.einsum("nhkv,nhk->nhv", decayed, kt)
            state = (
                decayed
                - bt * torch.einsum("nhk,nhv->nhkv", kt, pred)
                + bt * torch.einsum("nhk,nhv->nhkv", kt, zt)
            )
            outs.append(torch.einsum("nhkv,nhk->nhv", state, qt))
        return torch.stack(outs, 1), state

    def _readout(self, o, g, n, t):
        o = self.norm(o.reshape(n, t, self.n_heads * self.dv)) * F.silu(g)
        return self.out(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, _ = x.shape
        q, k, v, w, alpha, beta, g = self._proj(x)
        state = x.new_zeros(n, self.n_heads, self.dk, self.dv)
        o, _ = self._cell(q, k, v, w, alpha, beta, state)
        return self._readout(o, g, n, t)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Exact affine (Hillis-Steele) scan — parity reference only.

        ``state_t = a_t state_{t-1} + c_t`` with ``a_t = (I − beta·kkᵀ)·diag(alpha)``
        and ``c_t = beta·k·(w⊙v)ᵀ``. Materialises ``[d_k, d_k]`` maps, so it is
        for small-shape correctness checks, not training.
        """
        n, t, _ = x.shape
        q, k, v, w, alpha, beta, g = self._proj(x)
        eye = torch.eye(self.dk, device=x.device)
        amap = (eye - beta[..., None, None] * torch.einsum("bthk,bthj->bthkj", k, k)) * alpha[
            :, :, :, None, :
        ]
        cmap = beta[..., None, None] * torch.einsum("bthk,bthv->bthkv", k, w * v)
        amap = amap.transpose(1, 2).contiguous()
        cmap = cmap.transpose(1, 2).contiguous()
        d = 1
        while d < t:
            left = amap[:, :, d:]
            amap = torch.cat([amap[:, :, :d], left @ amap[:, :, : t - d]], 2)
            cmap = torch.cat([cmap[:, :, :d], left @ cmap[:, :, : t - d] + cmap[:, :, d:]], 2)
            d *= 2
        o = torch.einsum("bhtkv,bthk->bthv", cmap, q).reshape(n, t, self.n_heads * self.dv)
        return self._readout(o, g, n, t)

    def step(self, state: torch.Tensor, x_t: torch.Tensor):
        """One streaming step (ONNX export path). ``x_t`` is ``[N, 1, d_model]``."""
        q, k, v, w, alpha, beta, g = self._proj(x_t)
        kt, qt = k[:, 0], q[:, 0]
        zt = w[:, 0] * v[:, 0]
        bt = beta[:, 0].unsqueeze(-1).unsqueeze(-1)
        decayed = alpha[:, 0].unsqueeze(-1) * state
        pred = torch.einsum("nhkv,nhk->nhv", decayed, kt)
        state = (
            decayed
            - bt * torch.einsum("nhk,nhv->nhkv", kt, pred)
            + bt * torch.einsum("nhk,nhv->nhkv", kt, zt)
        )
        o = torch.einsum("nhkv,nhk->nhv", state, qt).reshape(
            x_t.shape[0], 1, self.n_heads * self.dv
        )
        return state, self._readout(o, g, x_t.shape[0], 1)


# ---------------------------------------------------------------------------
# Chunkwise (WY / UT-transform) training-throughput path
# ---------------------------------------------------------------------------
#
# The per-step ``StableGatedDeltaCore`` recurrence above is launch-bound on a
# GPU (~28k lane-tok/s on a T4). The chunkwise form below is the *same* gated
# delta rule re-expressed as batched matmuls + one triangular solve per chunk,
# with only T/chunk sequential steps — it hits tensor cores and measured
# ~2.3M lane-tok/s on a T4 (~84x faster, ~5.5 h/epoch vs ~114 h).
#
# To keep the chunk algebra a clean matmul, the decay here is **scalar per head**
# (Gated-DeltaNet style) rather than the channel-wise decay of the per-step core;
# a channel-wise erase (Gated-DeltaNet-2's novelty) would need the full WY kernel.
# Decay is folded via log-cumsum *ratios* (always ≤ 1, so no underflow blow-up).
#
# These functions operate directly on projected tensors of shape ``[L, T, ·]``
# (``L`` = batch·heads lanes), with ``alpha``/``beta`` scalar ``[L, T]``.


def recurrent_gated_delta(q, k, v, w, alpha, beta):
    """Reference per-step gated delta rule (scalar decay) — parity oracle."""
    lanes, t_len, d = k.shape
    state = k.new_zeros(lanes, d, v.shape[-1])
    outs = []
    for t in range(t_len):
        kt, qt = k[:, t], q[:, t]
        zt = w[:, t] * v[:, t]
        bt = beta[:, t, None, None]
        decayed = alpha[:, t, None, None] * state
        pred = torch.einsum("ld,ldv->lv", kt, decayed)
        state = (
            decayed
            - bt * torch.einsum("ld,lv->ldv", kt, pred)
            + bt * torch.einsum("ld,lv->ldv", kt, zt)
        )
        outs.append(torch.einsum("ldv,ld->lv", state, qt))
    return torch.stack(outs, 1), state


def chunkwise_gated_delta(q, k, v, w, alpha, beta, chunk: int = 64):
    """Chunkwise WY form of :func:`recurrent_gated_delta` (matmul, tensor-core).

    Within a chunk the writes ``u_i = beta_i (z_i − Ŝ_{i-1}ᵀ k̄_i)`` form a
    unit-lower-triangular system ``a u = rhs`` (the UT transform); solving it once
    per chunk and applying the carried state with decay ratios reproduces the
    recurrence exactly. Sequential work is only ``T // chunk`` chunk steps.
    """
    lanes, t_len, d = k.shape
    dv = v.shape[-1]
    z = w * v
    state = k.new_zeros(lanes, d, dv)
    log_a = torch.log(alpha.clamp_min(1e-30))
    eye = torch.eye(chunk, device=k.device)
    tri = torch.tril(torch.ones(chunk, chunk, device=k.device))
    strict = torch.tril(torch.ones(chunk, chunk, device=k.device), -1)
    out = []
    for c0 in range(0, t_len, chunk):
        cs = slice(c0, c0 + chunk)
        kc, qc, zc, bc = k[:, cs], q[:, cs], z[:, cs], beta[:, cs]
        n = kc.shape[1]
        log_g = torch.cumsum(log_a[:, cs], 1)  # cumulative log-decay (≤ 0)
        gamma = torch.exp(log_g)  # γ_i ∈ (0, 1]
        ratio = torch.exp(log_g[:, :, None] - log_g[:, None, :]) * tri[:n, :n]  # γ_i/γ_j ≤ 1
        kk = torch.einsum("lid,ljd->lij", kc, kc)
        qk = torch.einsum("lid,ljd->lij", qc, kc)
        a_mat = eye[:n, :n].expand(lanes, n, n) + bc[:, :, None] * (ratio * kk) * strict[:n, :n]
        ks = torch.einsum("lid,ldv->liv", kc, state)
        rhs = bc[:, :, None] * (zc - gamma[:, :, None] * ks)
        u = torch.linalg.solve_triangular(a_mat, rhs, upper=False, unitriangular=True)
        qs = torch.einsum("lid,ldv->liv", qc, state)
        out.append(
            gamma[:, :, None] * qs + torch.einsum("lij,ljv->liv", (ratio * qk) * tri[:n, :n], u)
        )
        g_last = gamma[:, -1]
        state = g_last[:, None, None] * state + torch.einsum(
            "lid,liv->ldv", kc * (g_last[:, None] / gamma)[:, :, None], u
        )
    return torch.cat(out, 1), state


# ---------------------------------------------------------------------------
# Spike checks (CPU, no datasets)
# ---------------------------------------------------------------------------


def check_stability(seeds: int = 8) -> None:
    worst = 0.0
    for seed in range(seeds):
        torch.manual_seed(seed)
        core = StableGatedDeltaCore(24, 2, 16, 16)
        x = torch.randn(1, 64, 24)
        with torch.no_grad():
            o = core(x)
        assert torch.isfinite(o).all(), f"non-finite output at seed {seed}"
        worst = max(worst, float(o.abs().max()))
    print(f"[stability] {seeds} seeds finite, max|out| = {worst:.2f}")


def check_overfit(seeds: int = 5) -> None:
    first = last = 0.0
    for seed in range(seeds):
        torch.manual_seed(seed)
        core = StableGatedDeltaCore(24, 2, 16, 16)
        x = torch.randn(1, 48, 24)
        y = torch.zeros_like(x)
        y[:, 2:] = x[:, :-2]  # delay-2 copy: needs recurrent memory
        opt = torch.optim.Adam(core.parameters(), 3e-3)
        for i in range(400):
            opt.zero_grad()
            loss = F.mse_loss(core(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        assert last < first * 0.05, f"seed {seed} did not converge ({first:.3f} -> {last:.3f})"
    print(f"[overfit]   {seeds} seeds delay-2 copy converged (e.g. {first:.3f} -> {last:.5f})")


def check_parity() -> None:
    torch.manual_seed(0)
    core = StableGatedDeltaCore(24, 2, 16, 16).eval()
    x = torch.randn(2, 40, 24)
    with torch.no_grad():
        err = (core(x) - core.forward_parallel(x)).abs().max().item()
    print(f"[parity]    recurrent vs affine-scan  max|Δ| = {err:.2e}")
    assert err < 1e-4


def check_onnx() -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[onnx]      skipped (onnx/onnxruntime not installed)")
        return

    class _Step(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, state, x_t):
            return self.core.step(state, x_t)

    torch.manual_seed(0)
    core = StableGatedDeltaCore(24, 2, 16, 16).eval()
    state = torch.zeros(1, 2, 16, 16)
    x = torch.randn(1, 1, 24)
    path = "/tmp/bs_deltagridnet_step.onnx"
    torch.onnx.export(
        _Step(core),
        (state, x),
        path,
        input_names=["state_in", "x_t"],
        output_names=["state_out", "o_t"],
        dynamo=False,
        opset_version=17,
    )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    state_pt, state_ort = state.clone(), state.clone().numpy()
    err = 0.0
    for _ in range(10):
        x = torch.randn(1, 1, 24)
        with torch.no_grad():
            state_pt, o_pt = core.step(state_pt, x)
        state_ort, o_ort = sess.run(["state_out", "o_t"], {"state_in": state_ort, "x_t": x.numpy()})
        err = max(err, float(np.abs(o_pt.numpy() - o_ort).max()))
    print(f"[onnx]      10-step streaming round-trip max|torch-ort| = {err:.2e}")
    assert err < 1e-4


def check_chunkwise() -> None:
    torch.manual_seed(0)
    lanes, t_len, d, dv = 4, 40, 8, 8
    q = F.normalize(torch.randn(lanes, t_len, d), dim=-1)
    k = F.normalize(torch.randn(lanes, t_len, d), dim=-1)
    v = torch.randn(lanes, t_len, dv)
    w = torch.sigmoid(torch.randn(lanes, t_len, dv))
    alpha = torch.sigmoid(torch.randn(lanes, t_len) + 2.0)  # scalar decay ∈ (0, 1)
    beta = 2.0 * torch.sigmoid(torch.randn(lanes, t_len))  # ∈ (0, 2)
    o_ref, s_ref = recurrent_gated_delta(q, k, v, w, alpha, beta)
    worst = 0.0
    for chunk in (8, 16, 40):
        o_c, s_c = chunkwise_gated_delta(q, k, v, w, alpha, beta, chunk=chunk)
        worst = max(worst, (o_ref - o_c).abs().max().item(), (s_ref - s_c).abs().max().item())
    print(f"[chunkwise] WY vs recurrent (chunks 8/16/40)  max|Δ| = {worst:.2e}")
    assert worst < 1e-4


def main() -> None:
    check_stability()
    check_overfit()
    check_parity()
    check_onnx()
    check_chunkwise()
    print("[ok] stable Gated-DeltaNet core: stable + trainable + ONNX + chunkwise-WY")


if __name__ == "__main__":
    main()
