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

**Ad-hoc speedups on top of chunkwise-WY (T4, same shape, fwd+bwd):**

| Variant | ms/step | throughput | vs eager |
|---|---|---|---|
| eager fp32, chunk=64 | 111 | 2.30M tok/s | 1.0× |
| eager fp32, chunk=128 | 95 | 2.69M tok/s | 1.17× |
| **`torch.compile`, chunk=64** | 46 | **5.56M tok/s** | **2.4×** |
| fp16 autocast (eager), chunk=64 | 117 | 2.20M tok/s | ~1.0× |
| `torch.compile` + fp16 autocast, chunk=64 | 37 | 6.99M tok/s | 3.0× |

- **`torch.compile` is the safe ~2.4× win** (Inductor fuses the per-chunk
  elementwise ops + loop overhead; output stays fp32, all finite). Just wrap the
  function. Note the optimum flips to **chunk=64** under compile (the per-chunk
  fixed cost is fused away, so more-but-cheaper chunks win).
- **fp16 autocast alone barely helps** on a T4 (small `d`, and `solve_triangular`
  doesn't benefit); it only adds value *under* compile (+25%) and carries fp16
  numerics risk — treat as opt-in pending a convergence/parity check.

Cumulatively: per-step loop ~28k → chunkwise-WY ~2.3M → +`torch.compile` ~5.6M
lane-tok/s (~200×), i.e. ~2.3 h/epoch on the conservative estimate above.

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

> **Formal proof.** The `‖S‖ ≈ 1` stability above is now formalised in Lean 4 +
> Mathlib: the key-side operator `A = (I − β·k·kᵀ)·diag(α)` (`‖k‖ = 1`,
> `β ∈ [0, 2]`, `α ∈ (0, 1]`) is proved non-expansive, `‖A‖₂ ≤ 1`, so the
> recurrent state stays bounded. See [`formal/`](../formal/) →
> `Formal/BSDeltaGridNet/Stability.lean` (`gatedDeltaCLM_opNorm_le_one`), proved
> with no `sorry`.

References: Gated DeltaNet (ICLR 2025); Gated DeltaNet-2 (arXiv 2605.22791);
DeltaNet (NeurIPS 2024); Mamba-2 / SSD (ICML 2024); flash-linear-attention.

## 4. GPU availability (Kaggle)

Kaggle's accepted accelerator list includes A100 / L4 / H100, but they are
competition/admin-gated. Requesting `NvidiaL4` on this account **scheduled a
P100** (CC 6.0). Effective free-tier GPUs are **T4 (CC 7.5) and P100 (CC 6.0)**,
both pre-Ampere. flash-linear-attention's fused delta kernels target CC ≥ 8.0,
so the **validated Triton path is not runnable on Kaggle** — it would need Colab
Pro+ (A100/L4) or a cloud A100 (paid).

## 4b. Real-data sanity (LibriSpeech)

The stable core was dropped into a minimal end-to-end STFT TSE — STFT encoder →
FiLM conditioning on a jointly-learned enrollment embedding → chunkwise core →
complex-mask decoder → iSTFT, trained with SI-SDR on on-the-fly LibriSpeech
dev-clean 2-speaker mixtures (target + interferer, separate enrollment clip).
A 0.3 M-param model over 600 steps (~75 s on a T4) moved held-out SI-SDR from the
**0 dB mixture baseline to +1.3 dB**, training stably (no NaN). The absolute
number is small (tiny model, naive mask, weak ref encoder, no tuning), but it
confirms the core **learns to extract the enrolled target end-to-end on real
audio** with a waveform loss — the integration works.

Two numerical-safety fixes to the chunkwise were required and are now in the
spike (with a strengthened parity test that exercises strong decay over multiple
chunks): (1) mask the decay-ratio's strict-upper triangle to `-inf` *before*
`exp` (otherwise `exp(+large)·0 = nan` under strong decay); (2) compute the
inter-chunk carry ratio `γ_last/γ_j` in log space (a plain division underflows
to `γ_j = 0` → `inf`). The mild-decay parity test did not catch these.

## 4c. Scaling the end-to-end model: depth needs ReZero; data is the next wall

Pushing past the 1-block smoke (LibriSpeech dev-clean, 16 kHz, on-the-fly
2-speaker mixtures, SI-SDR loss) surfaced two things — one solved, one open.

**Depth collapse → fixed with ReZero.** Stacking the time core into a deeper
residual network made *training* collapse to ~passthrough. A controlled ablation
from the working 1-block baseline (changing one variable at a time) isolated the
cause unambiguously — it is **depth**, not the core / conditioning / mask / band
split / segment length:

| Variant (in-pool unless noted) | train SI-SDR | eval |
|---|---|---|
| V0 reproduce (1 block, 2.0 s) | +0.97 | +1.04 |
| V1 + speaker split (unseen eval) | +1.38 | +0.43 (generalisation gap) |
| **V2 + depth (N=4 residual)** | **+0.14** | **+0.04** (collapses) |
| V3 + 2.5 s segments | +1.01 | +0.88 |

The fix is the standard deep-residual remedy — **ReZero / LayerScale**: a
learnable per-branch scalar initialised at 0, so the network starts as identity
and grows the residual contributions during training. With it, depth trains and
*helps*:

| Config (in-pool) | train | eval |
|---|---|---|
| N=4 plain | +0.28 | +0.65 |
| **N=4 + ReZero** | +1.06 | **+1.48** |
| **N=8 + ReZero** | +1.01 | +1.42 |

**Data scale is the next wall.** A faithful mini-grid (mel band-split → N×[causal
gated-delta time core + bidirectional cross-band GRU + FFN], all residual
branches ReZero-gated → bounded complex mask), conditioned on a **frozen ECAPA**
192-dim embedding (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0, not in the
training graph), now trains without collapsing (train climbs to ~+2 dB) — but
**overfits**: seen-speaker SI-SDR +1.4 while unseen degrades to −1.2 and gets
*worse* as training proceeds. dev-clean is far too small for a speaker-conditioned
extractor (40 speakers, ~5 h; we train on 32). The ReZero scales also stayed near
0, i.e. the grid blocks were barely engaged. Both point to needing real training
scale (e.g. train-clean-100: 251 speakers / 100 h, or LibriMix), regularisation,
an MR-STFT auxiliary loss, and more steps — standard "scale + tune" work, not a
fundamental blocker.

**Scaling up confirms the approach generalises and is genuinely
target-conditioned.** Training the same mini-grid on **train-clean-100 (251
speakers)** with `torch.compile`, MR-STFT aux loss, weight decay, and a
warmup+cosine schedule for 15 k steps (~1.75 h on a T4), evaluated on **disjoint
dev-clean speakers**:

| Metric (unseen unless noted) | SI-SDR |
|---|---|
| unseen, 0 dB mix | **+2.12 dB** (vs 0 dB passthrough) |
| unseen, +5 dB SIR | +5.84 dB (vs +5) |
| seen, 0 dB | +2.42 dB |

The overfitting is gone — **seen ≈ unseen**, confirming the dev-clean failure was
a *data-scale* problem, not architecture/conditioning. So the north-star approach
**does extract on unseen speakers and generalises**.

A wrong-enrollment + per-branch ablation (on a 5 k-step snapshot) then checked
*how* it works — and refuted two earlier worries:

| Ablation (unseen, 0 dB) | SI-SDR | reading |
|---|---|---|
| correct enrollment | +0.71 | baseline |
| **wrong** enrollment (other speaker) | −1.41 | **−2.1 dB → conditioning is target-specific**, not generic enhancement |
| random enrollment | −0.68 | −1.4 dB → embedding genuinely drives selection |
| kill time core (rt=0) | −0.70 | **−1.4 dB → the time core is *not* idle** |
| kill cross-band (rb=0) | −2.45 | −3.2 dB → cross-band GRU is the biggest contributor |
| kill FFN (rf=0) | +0.35 | −0.4 dB |
| kill all blocks | −11.35 | the bare band-split/merge linear mask is useless; the blocks do the work |

Two corrections to first impressions: (1) the small learned **ReZero scalars are
a misleading proxy** — `rt ≈ 0.01` looks "idle" but ablating the time branch
still costs 1.4 dB (the branch output magnitude compensates for the small
scalar); (2) the model is **genuinely target-conditioned** (wrong/random
enrollment costs 1.4–2.1 dB), not doing speaker-agnostic enhancement. So there is
**no structural defect** — all components contribute (cross-band > time core >
FFN) and the conditioning works.

The remaining gap to competitive SI-SDRi is therefore **capacity + scale**, not a
bug: the model is tiny (~1.3 M) and trained briefly (~1.75 h) versus, e.g.,
TF-GridNet (~20 M, trained for days on LibriMix). Quality levers from here:
more capacity (C / N / heads), longer training, a higher-ceiling output stage
(deep filtering), and strengthening the dominant cross-band path.

## 5. Conclusion & next steps

- **Stability and ONNX-exportability are solved** in pure PyTorch; the mechanism
  trains. The earlier "trainable ✅ (CPU)" claim was over-stated due to a
  lucky-seed + the normalise bug — now corrected and verified across seeds.
- **Throughput is solved too.** The chunkwise-WY reformulation
  (`chunkwise_gated_delta`) is ~84× faster than the per-step loop (~2.3M tok/s,
  ~5.5 h/epoch), `torch.compile` adds a further ~2.4×, and it is parity-exact, so
  **path A (Kaggle-native, free T4) is viable** — no Triton, no paid Ampere.
- **Real-data smoke passes**: SI-SDR moves 0 → +1.3 dB on held-out LibriSpeech
  mixtures (§4b).

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

- **End-to-end training mechanics are understood** (§4c): the core trains in a
  real STFT TSE; deeper models need **ReZero/LayerScale** on every residual
  branch (without it, depth collapses to passthrough). With ReZero, depth helps.
- **At scale it generalises and is target-conditioned** (§4c): train-clean-100
  gives unseen +2.1 dB (seen ≈ unseen), and an ablation confirms the conditioning
  is target-specific and every block contributes — no structural defect.
- **Remaining quality gap is capacity + scale, not architecture**: the model is
  tiny (~1.3 M) and briefly trained. Real quality needs more capacity, longer
  training, a higher-ceiling output stage, and/or LibriMix-scale data.

Remaining decisions / next steps:

- ✅ Ad-hoc throughput tuning — `torch.compile` gives a safe ~2.4× (fp16-AMP an
  opt-in further +25%); see the table in §2.
- ✅ Real-data integration + end-to-end training mechanics (§4b, §4c) — the core
  trains end-to-end; depth needs ReZero; data scale is the next wall.
- A **larger-scale training run** (train-clean-100 / LibriMix, regularisation,
  MR-STFT, more steps) to turn the working mechanics into competitive SI-SDRi.
- Decide whether channel-wise erase/decay is worth the kernel/hardware cost.
