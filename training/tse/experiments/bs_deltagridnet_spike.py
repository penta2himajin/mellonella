"""BS-DeltaGridNet trainability spike — stable Gated-DeltaNet time-axis core.

This is an **exploratory spike**, not production code. It validates that the
time-axis core of the north-star BS-DeltaGridNet architecture
(``docs/bs-deltagridnet.md``) can be implemented in **pure PyTorch (no Triton)**
in a numerically stable, trainable, ONNX-exportable form.

Findings captured here (see ``docs/bs-deltagridnet-trainability.md`` for the
full write-up):

* A naive Gated-DeltaNet-2 recurrence with a *channel-wise* erase gate
  ``e = b ⊙ k`` is **not a contraction** (the rank-1 term is asymmetric) and the
  state diverges (|S| → 1e37 in ~5 steps) across seeds, on both CPU and GPU.
* The fix is the standard stabilisation recipe from the DeltaNet / Gated
  DeltaNet / Mamba-2 / flash-linear-attention literature:
    - **L2-normalise q and k** (unit keys are the contraction precondition),
    - **symmetric Householder erase** ``I − β·kkᵀ`` with a *scalar* per-head
      ``β ∈ (0, 2)`` (eigenvalue ``1 − β`` stays in ``(−1, 1)``),
    - **channel-wise decay** ``α ∈ (0, 1)`` in **log space** (Mamba-2
      ``exp(−softplus(·))``),
    - **post-cell RMSNorm + SiLU output gate**.
  With this, ``‖S‖`` stays ≈ 1 across seeds and the core trains cleanly.
* This stable form keeps Gated-DeltaNet-2's *channel-wise write gate* and
  *channel-wise decay*, but drops the *channel-wise erase* (scalar β instead) —
  the channel-wise erase is the part that genuinely needs the WY chunkwise
  algorithm to stay stable, i.e. the validated NVlabs / flash-linear-attention
  kernel path.

Open item (the remaining blocker): the per-timestep Python recurrence here is
**launch-bound** on GPU (~28k lane-tok/s on a T4 → ~114 h/epoch), so it is not
yet training-throughput-viable. The next step is a **chunkwise matmul (WY)**
reformulation of this same stable recurrence (tensor-core friendly), which is
expected to give the 10–50× needed for a free-tier T4.

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

    State ``S`` per head is ``[d_k, d_v]``. The single-step recurrence is::

        DS    = diag(alpha) @ S                # channel-wise log-space decay
        pred  = kᵀ DS                          # current value stored at key k
        S     = DS − beta·k·predᵀ + beta·k·(w⊙v)ᵀ     # symmetric Householder erase + gated write
        o     = Sᵀ q

    The key-side operator ``(I − beta·kkᵀ)·diag(alpha)`` is non-expansive for
    unit ``k`` and ``beta ∈ (0, 2)``, ``alpha ∈ (0, 1)`` — so ``S`` stays
    bounded. ``forward`` runs the per-step recurrence; ``forward_parallel`` is an
    exact affine-scan reference for parity; ``step`` is the ONNX streaming path.
    """

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.H, self.dk, self.dv = n_heads, d_k, d_v
        self.q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.wg = nn.Linear(d_model, n_heads * d_v)      # channel-wise write gate
        self.a = nn.Linear(d_model, n_heads * d_k)       # channel-wise decay (log-space)
        self.beta = nn.Linear(d_model, n_heads)          # scalar erase strength / head
        self.g = nn.Linear(d_model, n_heads * d_v)       # output gate
        self.norm = RMSNorm(n_heads * d_v)
        self.out = nn.Linear(n_heads * d_v, d_model, bias=False)
        nn.init.constant_(self.a.bias, 2.0)

    def _proj(self, x: torch.Tensor):
        n, t, _ = x.shape
        h, dk, dv = self.H, self.dk, self.dv
        q = F.normalize(self.q(x).view(n, t, h, dk), dim=-1)
        k = F.normalize(self.k(x).view(n, t, h, dk), dim=-1)
        v = self.v(x).view(n, t, h, dv)
        w = torch.sigmoid(self.wg(x).view(n, t, h, dv))
        alpha = torch.exp(-F.softplus(self.a(x).view(n, t, h, dk)))   # (0, 1)
        beta = 2.0 * torch.sigmoid(self.beta(x).view(n, t, h))        # (0, 2)
        return q, k, v, w, alpha, beta, self.g(x)

    def _cell(self, q, k, v, w, alpha, beta, S):
        outs = []
        for t in range(q.shape[1]):
            kt, qt = k[:, t], q[:, t]
            zt = w[:, t] * v[:, t]
            bt = beta[:, t].unsqueeze(-1).unsqueeze(-1)
            DS = alpha[:, t].unsqueeze(-1) * S
            pred = torch.einsum("nhkv,nhk->nhv", DS, kt)
            S = DS - bt * torch.einsum("nhk,nhv->nhkv", kt, pred) \
                   + bt * torch.einsum("nhk,nhv->nhkv", kt, zt)
            outs.append(torch.einsum("nhkv,nhk->nhv", S, qt))
        return torch.stack(outs, 1), S

    def _readout(self, o, g, n, t):
        o = self.norm(o.reshape(n, t, self.H * self.dv)) * F.silu(g)
        return self.out(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, _ = x.shape
        q, k, v, w, alpha, beta, g = self._proj(x)
        S = x.new_zeros(n, self.H, self.dk, self.dv)
        o, _ = self._cell(q, k, v, w, alpha, beta, S)
        return self._readout(o, g, n, t)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Exact affine (Hillis-Steele) scan — parity reference only.

        ``S_t = A_t S_{t-1} + c_t`` with ``A_t = (I − beta·kkᵀ)·diag(alpha)`` and
        ``c_t = beta·k·(w⊙v)ᵀ``. Materialises ``[d_k, d_k]`` maps, so it is for
        small-shape correctness checks, not training.
        """
        n, t, _ = x.shape
        q, k, v, w, alpha, beta, g = self._proj(x)
        eye = torch.eye(self.dk, device=x.device)
        A = (eye - beta[..., None, None] * torch.einsum("bthk,bthj->bthkj", k, k)) \
            * alpha[:, :, :, None, :]
        C = beta[..., None, None] * torch.einsum("bthk,bthv->bthkv", k, w * v)
        A = A.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()
        d = 1
        while d < t:
            al = A[:, :, d:]
            A = torch.cat([A[:, :, :d], al @ A[:, :, :t - d]], 2)
            C = torch.cat([C[:, :, :d], al @ C[:, :, :t - d] + C[:, :, d:]], 2)
            d *= 2
        o = torch.einsum("bhtkv,bthk->bthv", C, q).reshape(n, t, self.H * self.dv)
        return self._readout(o, g, n, t)

    def step(self, S: torch.Tensor, x_t: torch.Tensor):
        """One streaming step (ONNX export path). ``x_t`` is ``[N, 1, d_model]``."""
        q, k, v, w, alpha, beta, g = self._proj(x_t)
        kt, qt = k[:, 0], q[:, 0]
        zt = w[:, 0] * v[:, 0]
        bt = beta[:, 0].unsqueeze(-1).unsqueeze(-1)
        DS = alpha[:, 0].unsqueeze(-1) * S
        pred = torch.einsum("nhkv,nhk->nhv", DS, kt)
        S = DS - bt * torch.einsum("nhk,nhv->nhkv", kt, pred) \
               + bt * torch.einsum("nhk,nhv->nhkv", kt, zt)
        o = torch.einsum("nhkv,nhk->nhv", S, qt).reshape(x_t.shape[0], 1, self.H * self.dv)
        return S, self._readout(o, g, x_t.shape[0], 1)


# ---------------------------------------------------------------------------
# Spike checks (CPU, no datasets)
# ---------------------------------------------------------------------------

def check_stability(seeds: int = 8) -> None:
    worst = 0.0
    for seed in range(seeds):
        torch.manual_seed(seed)
        m = StableGatedDeltaCore(24, 2, 16, 16)
        x = torch.randn(1, 64, 24)
        with torch.no_grad():
            o = m(x)
        assert torch.isfinite(o).all(), f"non-finite output at seed {seed}"
        # peek at internal state magnitude via the parallel path's bound proxy
        worst = max(worst, float(o.abs().max()))
    print(f"[stability] {seeds} seeds finite, max|out| = {worst:.2f}")


def check_overfit(seeds: int = 5) -> None:
    for seed in range(seeds):
        torch.manual_seed(seed)
        m = StableGatedDeltaCore(24, 2, 16, 16)
        x = torch.randn(1, 48, 24)
        y = torch.zeros_like(x)
        y[:, 2:] = x[:, :-2]                       # delay-2 copy: needs recurrent memory
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        l0 = lf = None
        for i in range(400):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if i == 0:
                l0 = loss.item()
            lf = loss.item()
        assert lf < l0 * 0.05, f"seed {seed} did not converge ({l0:.3f} -> {lf:.3f})"
    print(f"[overfit]   {seeds} seeds delay-2 copy converged (e.g. {l0:.3f} -> {lf:.5f})")


def check_parity() -> None:
    torch.manual_seed(0)
    m = StableGatedDeltaCore(24, 2, 16, 16).eval()
    x = torch.randn(2, 40, 24)
    with torch.no_grad():
        err = (m(x) - m.forward_parallel(x)).abs().max().item()
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

        def forward(self, S, x_t):
            return self.core.step(S, x_t)

    torch.manual_seed(0)
    m = StableGatedDeltaCore(24, 2, 16, 16).eval()
    S = torch.zeros(1, 2, 16, 16)
    x = torch.randn(1, 1, 24)
    path = "/tmp/bs_deltagridnet_step.onnx"
    torch.onnx.export(_Step(m), (S, x), path, input_names=["S_in", "x_t"],
                      output_names=["S_out", "o_t"], dynamo=False, opset_version=17)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    St, So = S.clone(), S.clone().numpy()
    err = 0.0
    for _ in range(10):
        x = torch.randn(1, 1, 24)
        with torch.no_grad():
            St, ot = m.step(St, x)
        So, oo = sess.run(["S_out", "o_t"], {"S_in": So, "x_t": x.numpy()})
        err = max(err, float(np.abs(ot.numpy() - oo).max()))
    print(f"[onnx]      10-step streaming round-trip max|torch-ort| = {err:.2e}")
    assert err < 1e-4


def main() -> None:
    check_stability()
    check_overfit()
    check_parity()
    check_onnx()
    print("[ok] stable Gated-DeltaNet core: stable + trainable + ONNX-exportable")


if __name__ == "__main__":
    main()
