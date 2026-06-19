# BS-DeltaGridNet — Trainability Spike (Stage C TSE north star)

Companion to [`bs-deltagridnet.md`](bs-deltagridnet.md). This records the
trainability investigation of the north-star architecture's **only genuinely
novel/risky component — the Gated-DeltaNet-2 time-axis core** — and what it
means for *where* and *how* we could train it. The other components (STFT,
ERB band-split, cross-band attention, complex masking) are well-trodden.

Reproducible spike: [`training/tse/experiments/bs_deltagridnet_spike.py`](../training/tse/experiments/bs_deltagridnet_spike.py)
(CPU, no datasets). GPU numbers below were measured on Kaggle T4.

## TL;DR

| Question | Verdict |
|---|---|
| Does the gated-delta mechanism train in **pure PyTorch (no Triton)**? | **Yes** — once stabilised, it trains across seeds on CPU and T4. |
| Does the streaming step **export to ONNX** (deployment gate ①)? | **Yes** — single-step recurrence round-trips at ~5e-7. |
| Is it **numerically stable**? | **Yes, with the standard recipe** (see §3). A naive channel-wise-erase form diverges. |
| Is it **training-throughput-viable on a free T4** (gate ②)? | **Yes, with the chunkwise-WY reformulation** — ~2.3M tok/s (≈84× the per-step loop), ~5.5 h/epoch. The per-step loop alone was launch-bound (~28k tok/s → ~114 h/epoch). |
| Can we use the **validated Triton kernels** on Kaggle instead? | **No** — Kaggle free tier exposes only T4 / P100 (pre-Ampere, CC < 8.0); flash-linear-attention needs CC ≥ 8.0. The chunkwise-WY path above is pure PyTorch, so it does not need them. |

## 1. Method

- **CPU spike** (this repo): a self-contained Gated-DeltaNet core with two
  numerically-equivalent forward modes (recurrent + parallel affine-scan) plus a
  streaming `step` for ONNX. Checks: multi-seed stability, delay-2-copy overfit
  (requires recurrent memory), recurrent-vs-parallel parity, ONNX round-trip.
- **Kaggle T4**: GPU stability + overfit (no NaN at scale) and forward+backward
  throughput with a chunked + gradient-checkpointed loop, plus a 1-epoch
  extrapolation for a Libri-scale dataset.

## 2. Results

**Platform-independent (pass):**

| Check | Result |
|---|---|
| Multi-seed stability (8 seeds) | all finite, `max|S| ≈ 0.75–1.16` |
| Delay-2-copy overfit (5 seeds) | `~1.0 → ~6e-5` every seed |
| Recurrent vs parallel parity | `max|Δ| = 2.4e-7` |
| ONNX 10-step streaming round-trip | `max|torch-ort| = 4.6e-7` |
| One time-core params (C=128, H=8, dk=dv=128) | ~0.92 M |

**GPU (Kaggle T4):**

| Config | ms/step | peak | throughput |
|---|---|---|---|
| ceiling H=8, dk=dv=128, chunk=64 | — | — | **OOM** |
| ceiling, chunk=16 | 6598 | 8.17 GB | 7.4k tok/s |
| practical H=4, dk=dv=64, N=64, T=1000 | 2300 | 1.74 GB | 27.8k tok/s |
| practical, N=64, T=2000 | 4643 | 2.91 GB | 27.6k tok/s |
| **1-epoch** (B=2, K=32, 1 s clips, 100 h) | — | — | **≈114 h/epoch (100 ep ≈ 11,400 h)** |

The stabilisation fix did **not** change throughput (same per-step-loop
structure): ~28k tok/s before and after. Reducing dims/bands/clip length only
helps linearly; the gap to a usable epoch time is ~100×, so it needs the
algorithmic (chunkwise-matmul) change, not parameter tweaks.

**Chunkwise-WY (the fix), Kaggle T4, fwd+bwd, L = 256 lanes, d=dv=64:**

| Config | ms/step | peak | throughput |
|---|---|---|---|
| per-step loop (baseline) | 2300 | 1.46 GB | 27.8k tok/s |
| chunkwise-WY, chunk=32 | 164 | 1.04 GB | 1.56M tok/s |
| chunkwise-WY, chunk=64 | 110 | 1.05 GB | **2.32M tok/s** |
| chunkwise-WY, chunk=128 | 95 | 1.19 GB | 2.70M tok/s |
| **1-epoch** (B=2, K=32, 1 s, 100 h, chunk=64) | — | — | **≈5.5 h/epoch** |

The chunkwise form (`chunkwise_gated_delta` in the spike) is the *same* gated
delta rule re-expressed as batched matmuls + one triangular solve per chunk,
with only `T/chunk` sequential steps — it hits tensor cores and is **~84× faster**
than the per-step loop, parity-exact (`max|Δ| ≈ 3e-7` vs the recurrent
reference), and memory-light (~1 GB). This moves the time core from
~114 h/epoch to ~5.5 h/epoch — **viable on a free T4**. To keep the chunk algebra
a clean matmul the decay is **scalar per head** (Gated-DeltaNet style) rather
than channel-wise; decay is folded via log-cumsum *ratios* (≤ 1, no underflow).

## 3. Why a naive recurrence diverged, and the stabilisation recipe

A naive Gated-DeltaNet-2 step with a **channel-wise erase gate**
`e = b ⊙ k` uses an *asymmetric* rank-1 term `I − k·eᵀ`, whose operator norm can
exceed 1 → the state grows ~10×/step to `1e37 → inf` within ~5 steps, on every
seed (CPU and GPU). (A contributing bug in the first spike — `F.normalize(x, -1)`
sets the *norm order* `p=-1` instead of `dim=-1`, leaving keys un-normalised —
amplified this; both must be correct.)

The fix is the standard recipe from the DeltaNet / Gated DeltaNet / Mamba-2 /
flash-linear-attention literature:

1. **L2-normalise q and k** (unit keys are the contraction precondition).
2. **Symmetric Householder erase** `I − β·kkᵀ` with a *scalar* per-head
   `β ∈ (0, 2)` → eigenvalue `1 − β ∈ (−1, 1)`, non-expansive.
3. **Channel-wise decay** `α ∈ (0, 1)` in **log space** (Mamba-2
   `exp(−softplus(·))`).
4. **Post-cell RMSNorm + SiLU output gate**.

With this, `‖S‖ ≈ 1` across seeds and the core trains cleanly. This stable form
keeps Gated-DeltaNet-2's **channel-wise write gate** and **channel-wise decay**
but uses a **scalar erase** — the *channel-wise erase* (GDN-2's headline novelty)
is exactly the part that needs the WY chunkwise algorithm to stay stable, i.e.
the validated NVlabs / flash-linear-attention kernel.

References: Gated DeltaNet (ICLR 2025); Gated DeltaNet-2 (arXiv 2605.22791);
DeltaNet (NeurIPS 2024); Mamba-2 / SSD (ICML 2024); flash-linear-attention.

## 4. GPU availability (Kaggle)

Kaggle's accepted accelerator list includes A100 / L4 / H100, but they are
competition/admin-gated. Requesting `NvidiaL4` on this account **scheduled a
P100** (CC 6.0). Effective free-tier GPUs are **T4 (CC 7.5) and P100 (CC 6.0)**,
both pre-Ampere. flash-linear-attention's fused delta kernels target CC ≥ 8.0,
so the **validated Triton path is not runnable on Kaggle** — it would need Colab
Pro+ (A100/L4) or a cloud A100 (paid).

## 5. Conclusion & next steps

- **Stability and ONNX-exportability are solved** in pure PyTorch; the mechanism
  trains. The earlier "trainable ✅ (CPU)" claim was over-stated due to a
  lucky-seed + the normalise bug — now corrected and verified across seeds.
- **Throughput is solved too.** The chunkwise-WY reformulation
  (`chunkwise_gated_delta`) is ~84× faster than the per-step loop (~2.3M tok/s,
  ~5.5 h/epoch) and parity-exact, so **path A (Kaggle-native, free T4) is
  viable** — no Triton, no paid Ampere required.

What this stable+fast core is, precisely: scalar-β Householder erase, **scalar**
per-head decay, channel-wise write gate, post-cell RMSNorm. It is essentially
**Gated DeltaNet**. Two things are *not* yet included and remain optional future
work:

- **Channel-wise decay** (KDA) — needs a `[chunk, chunk, d]` ratio tensor in the
  chunk algebra (more expensive); scalar decay was chosen for the clean fast path.
- **Channel-wise erase** (Gated-DeltaNet-2's headline novelty) — needs the full
  gate-aware WY kernel; best recovered via the validated NVlabs /
  flash-linear-attention kernels on **paid Ampere** (Colab Pro+ / cloud A100), if
  an ablation shows it earns its keep for TSE.

Remaining decisions / next steps:

- Ad-hoc throughput tuning of the pure-PyTorch chunkwise (`torch.compile`,
  fused projections, bf16/AMP, chunk size) to push epoch time down further.
- A small **real-data** training run (VCTK + DEMAND) to confirm SI-SDRi moves.
- Decide whether channel-wise erase/decay is worth the kernel/hardware cost.
