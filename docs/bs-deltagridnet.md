# BS-DeltaGridNet — North-Star Architecture for Stage C TSE

> **Status: aspirational design (north star), not a build target yet.** This
> document specifies the *highest-quality* causal target-speaker-extraction
> (TSE) architecture we would build for `mellonella`'s Stage C overlap path
> **if implementation difficulty, training budget, and parameter count were
> no object**. It deliberately ignores those constraints so we have a fixed
> ceiling to measure against. The separate question of *where we can actually
> train this* and *which parts to drop for the edge build* is future work,
> tracked as a follow-up (see [§9](#9-relationship-to-the-current-stage-c-and-next-steps)).

The name decodes as **B**and-**S**plit + **Delta** (the Gated-DeltaNet family
of delta-rule linear RNNs) + **GridNet** (the TF-GridNet time–frequency grid).
The design fuses the best current component for each axis of the problem.

## 1. Design principle

One observation drives every choice below:

> **Causality constrains only the time axis. The frequency axis, the richness
> of the recurrent state, and the speaker conditioning can absorb unlimited
> compute without adding any algorithmic latency.**

Separation quality is dominated by (a) capturing harmonic / phase structure
*across frequency* and (b) retaining the *target speaker's* subspace in the
recurrent state while erasing interferers. Neither needs future time context:
cross-frequency mixing happens within a single STFT frame, and conditioning is
just *how* the state is written. So we can chase the quality ceiling and still
keep strict causal, streaming, ≤100 ms operation — the property that makes a
band-split / grid formulation a better fit here than a time-domain TasNet.

## 2. Architecture overview

```
mixture (48 kHz, mono)
  │  STFT  (20 ms window / 10 ms hop, complex)              ── matches DFN3 framing
  ▼
[Band-Split]  F bins → K ERB subbands → per-band LN + Linear → C-dim     (BSRNN)
  │
  ├─ coarse conditioning: global FiLM(e) biases band features toward the target
  ▼
┌─ N × Grid Block ─────────────────────────────────────────────────────────┐
│ (1) Sub-band temporal : CAUSAL multi-head Gated DeltaNet-2                 │ time axis (causal)
│        └ ECAPA conditions the channel-wise erase b_t / write w_t / decay α_t │
│ (2) Cross-band fusion : BIDIRECTIONAL self-attention over the K bands      │ freq axis (non-causal OK)
│ (3) Cross-frame attn  : CAUSAL sliding-window self-attention (past only)   │ time axis (no lookahead)
└───────────────────────────────────────────────────────────────────────────┘
  │
  ▼
[Band-Merge]  C-dim → complex bins
  │  Complex Spectral Mapping  (+ optional Deep Filtering taps)             (cIRM / DFN)
  ▼  applied to the mixture STFT → iSTFT
target waveform (48 kHz, mono)
```

Conditioning input `e ∈ ℝ¹⁹²` is a **frozen** ECAPA-TDNN enrollment embedding
— the same vector `mellonella` already enrolls. The ECAPA network is never in
the training loop; only *how the embedding is consumed* is learned.

## 3. Components and rationale

### 3.1 Input — complex STFT (do not retreat to the time domain)

SpeakerBeam-SS shows a time-domain Conv-TasNet frontend can win on real-time
factor, but the *quality ceiling* belongs to the time–frequency domain
(TF-GridNet lineage): complex masking/mapping can correct phase, and Band-Split
state compression only makes sense in frequency. Framing (20 ms / 10 ms)
matches DeepFilterNet 3 so the existing STFT machinery and latency budget carry
over.

### 3.2 Band-Split — ERB subbands (BSRNN)

Split the `F` frequency bins into `K` non-uniform subbands on an **ERB** scale
(fine in the low end, tens of bins per band in the high end). Each subband is
LayerNorm'd and linearly projected to a hidden width `C`. This is what turns
"one recurrent state per frequency bin (thousands)" into "one per band (≈ tens)"
— the memory lever that makes per-band recurrence tractable.

### 3.3 Sub-band temporal model — causal Gated DeltaNet-2 (the core)

The per-band time-axis sequence model is a **causal, multi-head Gated
DeltaNet-2** linear RNN. Single-step (streaming) recurrence:

```
S_t = (I − k_t e_tᵀ) · D_t · S_{t-1} + k_t z_tᵀ ,   e_t = b_t ⊙ k_t,  z_t = w_t ⊙ v_t
o_t = S_tᵀ q_t
```

- `b_t` — **channel-wise erase gate** (key axis): which state coordinates to clear.
- `w_t` — **channel-wise write gate** (value axis): which coordinates to commit.
- `D_t = Diag(α_t)` — **channel-wise decay**.

Why this specific layer, over plain DeltaNet / Gated DeltaNet / Mamba-2:
**Gated DeltaNet-2 decouples *erase* from *write* with independent channel-wise
gates.** That is the exact mechanism TSE needs — "erase the interferer's
coordinates, write only the target's" — and it is the principled home for
speaker conditioning (§4). Plain DeltaNet has no forget gate; Gated DeltaNet
couples erase and write into one scalar. The inference recurrence is a pure
fixed-size matrix recurrence (two mat-vecs + elementwise gates), so it exports
to standard ONNX operators with no custom kernel — the streaming-state design
mirrors the explicit conv-state threading the current Stage C model already
uses.

State per block: `[batch, heads, d_k, d_v]`, fixed size, independent of stream
length.

*Max-expressivity knob:* take multiple delta steps per frame in the
DeltaProduct style (products of Householder reflections) for stronger
state-tracking on hard overlaps, at higher compute.

### 3.4 Cross-band fusion — bidirectional attention over K (the free lunch)

The frequency-mixing module runs **within a single time frame**, so it cannot
leak future time context and is therefore allowed to be **fully bidirectional**.
We use a full self-attention across the `K` (≈ tens) bands. Because `K` is
small, this is cheap, and it captures long-range harmonic coupling
(low-band ↔ high-band) far better than the flattened FFN of the original
BS-DeltaGridNet sketch (which also caused its ~60–70 M parameter blow-up) or a
band-RNN. **This is where the design buys the most quality for zero latency.**

### 3.5 Cross-frame attention — causal sliding-window self-attention

Per the Gated DeltaNet-2 recipe, the best configuration is a **hybrid**: the
linear RNN (compressed, unbounded memory) plus a **causal sliding-window
softmax attention** over recent frames (exact local memory). The window looks
**only at the past**, so it adds a KV-cache memory cost but **zero algorithmic
latency**.

### 3.6 Output — complex spectral mapping (+ deep filtering)

Band-Merge projects each band's `C`-dim feature back to complex bins. For the
quality ceiling, **predict the target's real/imaginary spectrum directly
(complex spectral mapping)** rather than a bounded mask — TF-GridNet finds
mapping beats masking at high SDR. Optionally add a few **deep-filtering** taps
(a short complex FIR across neighbouring frames/bins, à la DeepFilterNet) to
fix transients and phase in the hardest cases. A bounded complex ratio mask
(cIRM, tanh-compressed) is the conservative fallback when artifact-minimisation
matters more than peak SDR. Finish with iSTFT.

## 4. Speaker conditioning (the differentiator)

The frozen 192-dim ECAPA embedding `e` is injected **hierarchically**, coarse
to fine:

1. **Coarse** — a global FiLM(`e`) right after Band-Split biases every band's
   features toward the target direction.
2. **Fine (the core)** — in every Grid Block, a **per-block, per-gate MLP(`e`)**
   modulates the Gated DeltaNet-2 gates *channel-wise* just before their
   nonlinearity: it steers `b_t` (erase the interferer-correlated coordinates),
   `w_t` (write the target-correlated coordinates), and `α_t` (decay). This is
   the spec's original "inject the speaker into the forget/input gates so only
   the target's information is written to the state," realised with the gate
   structure that actually makes it separable.
3. **Attention adaptation (optional)** — multiplicatively adapt the cross-frame
   attention's query/key by `e`, SpeakerBeam-style.

Conditioning lineage: FiLM (Perez et al., 2018) and SpeakerBeam
(Delcroix / Žmolíková et al.) generalised onto a delta-rule linear RNN's gates.

## 5. Training objective

Tuned for extraction quality **and** the gating / overlap-routing use case:

- **Primary** — SI-SDR on the iSTFT waveform.
- **Aux 1** — multi-resolution STFT magnitude L1 (convergence stability, high-band quality).
- **Aux 2 (TSE)** — speaker-consistency: maximise `cos(ECAPA(output), e)` so the
  extraction is pinned to the target identity.
- **Aux 3 (mellonella)** — **target-absent suppression**: include mixtures with
  no target speaker and train the output toward silence. This is what makes the
  model cooperate with hard-gating / overlap routing and the FP-tolerant policy
  (cf. Personal VAD).

**Data must stay commercially clean even at the ceiling** — VCTK + DEMAND,
DNS5, MUSAN, FSD50K-style noise, and LibriSpeech-derived mixtures, **without
WHAM!** (CC BY-NC). This is a `mellonella` invariant (decision D-007), not a
difficulty knob.

## 6. Reference configuration (ceiling-leaning)

| Symbol | Meaning | Value |
|---|---|---|
| SR | sample rate | 48 kHz |
| window / hop | STFT framing | 20 ms / 10 ms |
| K | ERB subbands | ≈ 48 |
| C | per-band hidden width | ≈ 128 |
| N | grid blocks | ≈ 6 |
| heads | GDN-2 heads | ≈ 8 |
| d_k, d_v | per-head key/value dim | ≈ 128 |
| SWA window | causal attention span | ≈ 1–2 s of past |

Order-of-magnitude parameter count: **~10–30 M** — large, but with no structural
blow-up (the band-fusion attention over a small `K` is cheap; the 60–70 M of the
original flattened-FFN sketch is gone).

## 7. Why this still streams under 100 ms

| Element | Latency contribution |
|---|---|
| STFT framing | ~20–30 ms (20 ms window + hop) |
| Sub-band temporal (GDN-2) | single-frame recurrence — no wait |
| Cross-band fusion (bidirectional) | within one frame — **0 ms** |
| Cross-frame attention | past only — **0 ms** (memory only) |
| **Total algorithmic latency** | **~20–30 ms**, inside the DFN3 budget |

Optional quality-for-latency trade, still under budget: allow a small **20–40 ms
future lookahead** on the time axis (TF-GridNet gains from limited future
context).

## 8. Optional max-expressivity knobs (all latency-free or in-budget)

- DeltaProduct-style multi-step delta updates inside the GDN-2 core.
- Deep-filtering taps on the output stage.
- 20–40 ms time-axis lookahead.
- A dual cross-band path (attention + band-conv).

## 9. Relationship to the current Stage C and next steps

`mellonella`'s current Stage C (`training/tse/`) is a ~1.41 M causal
Conv-TasNet with global FiLM conditioning — a deliberately shippable, commodity-
GPU-trainable baseline. **BS-DeltaGridNet is the north star that baseline aims
at**, not a replacement to be merged blindly.

> **Trainability spike (done):** the time-axis core has since been prototyped in
> pure PyTorch — it is numerically stable, trains across seeds, and exports to
> ONNX, but the per-step recurrence is throughput-bound on a free T4. Full
> findings and the remaining chunkwise-matmul work are in
> [`bs-deltagridnet-trainability.md`](bs-deltagridnet-trainability.md).

The two open questions, to be answered in a follow-up after this spec lands:

1. **Where can we train it?** The efficient Gated DeltaNet-2 training path leans
   on fused Triton kernels (WY triangular solve + gate-aware backward) tuned for
   Ampere/Hopper. `mellonella`'s TSE works at a ~1 kHz latent rate over short
   (seconds-long) windows, so a pure-PyTorch autograd implementation of the
   chunkwise recurrence may be viable on commodity GPUs — this needs an
   empirical trainability spike (1-epoch wall-clock on a T4, plus ONNX
   round-trip parity).
2. **What do we drop for the edge build?** A de-risking table mapping each knob
   (frequency-domain front/back, GDN-2 vs. Gated DeltaNet vs. Mamba-3 time core,
   bidirectional vs. lightweight band fusion, complex mapping vs. cIRM) to its
   quality / parameter / training-cost / ONNX-complexity delta.

## References

- TF-GridNet — Wang et al., *Integrating Full- and Sub-Band Modeling for Speech Separation*, IEEE/ACM TASLP 2023.
- BSRNN — Luo & Yu, *Music Source Separation with Band-Split RNN*, IEEE/ACM TASLP 2023.
- Gated DeltaNet-2 — *Decoupling Erase and Write in Linear Attention*, arXiv 2605.22791 (NVIDIA, 2026); code under CC BY 4.0.
- Gated DeltaNet — Yang, Kautz, Hatamizadeh, *Improving Mamba2 with Delta Rule*, ICLR 2025.
- DeltaNet — Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, NeurIPS 2024.
- DeltaProduct — Siems et al., *Improving State-Tracking in Linear RNNs via Householder Products*, 2025.
- SSD / Mamba-2 — Dao & Gu, *Transformers are SSMs: ... Structured State Space Duality*, ICML 2024.
- SpeakerBeam-SS — Sato et al., *Real-time TSE with Lightweight Conv-TasNet and State Space Modeling*, Interspeech 2024.
- ECAPA-TDNN — Desplanques et al., Interspeech 2020.
- FiLM — Perez et al., *Feature-wise Linear Modulation*, AAAI 2018.
- SpeakerBeam — Delcroix / Žmolíková et al., *Target Speaker Extraction*.
- Complex Ratio Masking — Williamson et al., IEEE/ACM TASLP 2016.
- DeepFilterNet — Schröter et al., ICASSP 2022.
- Personal VAD — Ding et al., 2019.
