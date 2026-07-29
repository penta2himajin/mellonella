# formal — Lean 4 formal verification of BS-DeltaGridNet

Lean 4 + Mathlib formalisation of the mathematical properties of the
BS-DeltaGridNet stable gated-delta core (see `docs/bs-deltagridnet.md` and
`docs/bs-deltagridnet-trainability.md`, and the session handoff in issue #186).

## Toolchain

- Lean: `leanprover/lean4:v4.31.0` (pinned in `lean-toolchain`)
- Mathlib: `v4.31.0` (pinned in `lakefile.toml` / `lake-manifest.json`)

## Build

```sh
# from this directory
lake exe cache get   # fetch prebuilt Mathlib oleans (first time only)
lake build
```

`lake build` is green and every theorem type-checks with **no `sorry`**
(verified with `#print axioms`: only `propext`, `Classical.choice`, `Quot.sound`).

## Results

### `Stability.lean` — non-expansiveness (handoff theorem #1)

**Non-expansiveness of the stable gated-delta key-side operator.** For
`A = (I − β·k·kᵀ)·diag(α)` with `‖k‖₂ = 1`, `β ∈ [0, 2]`, `α ∈ (0, 1]`
(more generally `|αᵢ| ≤ 1`):

| Lemma | Statement |
|---|---|
| `householder_norm_le` | `‖x − (β⟪k,x⟫)•k‖ ≤ ‖x‖` (symmetric Householder erase is non-expansive) |
| `diagLM_norm_le` | `‖diag(α) x‖ ≤ ‖x‖` (per-channel decay is non-expansive) |
| `gatedDelta_nonexpansive` | the composite `A` is non-expansive, pointwise |
| `gatedDeltaCLM_opNorm_le_one` | bundled operator-norm bound `‖A‖₂ ≤ 1` |

This is the precise statement behind the empirical stability finding `‖S‖ ≈ 1`
in `docs/bs-deltagridnet-trainability.md` §3: because `A` is non-expansive, the
recurrent state stays bounded.

### `Chunkwise.lean` — chunkwise ⇔ recurrent (handoff theorem #2)

The scalar-decay gated-delta state update is a first-order **affine matrix
recurrence** `Sₜ = Aₜ Sₜ₋₁ + Cₜ` with `Aₜ = αₜ(I − βₜ kₜ kₜᵀ)`, `Cₜ = βₜ kₜ zₜᵀ`
(the `amap`/`cmap` of `forward_parallel`). Two facts make "chunkwise = recurrent"
precise:

| Theorem | Statement |
|---|---|
| `recurrent_flatten` / `chunkwise_eq_recurrent` | chunk-carry exactness: arbitrary chunking with carried boundary state = per-step recurrence |
| `recurrent_linear` | superposition: `recurrent S₀ ps = linPart ps * S₀ + recurrent 0 ps` (single-matmul carry + chunk-local-from-zero) |
| `gatedDelta_isAffineStep` | the spike's gated-delta update is an instance of the affine recurrence |

Together these are the algebraic identity the WY/UT-transform kernel relies on:
each chunk = a linear carry of the boundary state (`linPart * S₀`, the γ-decay
carry) plus a chunk-local term computed from zero state (the triangular solve),
and re-stitching the chunks is exact. The per-step `Aₜ` is the same operator
proved non-expansive in `Stability.lean`, so the carry stays bounded.

### `Causality.lean` — causality (handoff theorem #3)

The output `oₜ = Sₜᵀ qₜ` depends only on the inputs at times `≤ t`. Modelling the
per-step data as `(Aₜ, Cₜ, qₜ)` (state update `+` read-out), with
`outputs S₀ steps` the list of outputs:

| Theorem | Statement |
|---|---|
| `outputs_take_prefix` | running on the first `n` steps yields exactly the first `n` outputs of the full run (the future cannot change the past) |
| `outputs_causal` | input streams agreeing on their first `n` steps have identical first `n` outputs |

### `Decay.lean` — decay-ratio boundedness (handoff theorem #4)

Decay is carried in log space: with `aₜ = log αₜ ≤ 0`, the cumulative log-decay
`loggₙ = ∑_{t<n} aₜ` is non-increasing, so `γₙ = exp(loggₙ) = ∏_{t<n} αₜ` is too.

| Theorem | Statement |
|---|---|
| `logCumsum_antitone` | cumulative log-decay is non-increasing |
| `gamma_ratio_eq` | `γᵢ/γⱼ = exp(loggᵢ − loggⱼ)` (the unmasked lower-triangular ratio) |
| `gamma_ratio_le_one` | `γᵢ/γⱼ ≤ 1` for `j ≤ i` |
| `gamma_carry_le_one` | `γ_last/γⱼ ≤ 1` — the division-free log-space carry never overflows |

This justifies the kernel's numerical-safety fixes (mask-before-exp, log-space
carry): every realised ratio is `≤ 1`.

### `Streaming.lean` — streaming ⇔ full-sequence (handoff theorem #5)

`streamStep` is the exported single-step (`step`); `stream` folds it over the
sequence, threading state and accumulating outputs.

| Theorem | Statement |
|---|---|
| `stream_outputs` | streaming emits exactly the full-sequence outputs |
| `stream_state` | streaming ends in the full-sequence final state |
| `stream_eq_forward` | the ONNX contract: `step` looped == `forward` (outputs and final state) |

### `Scaling.lean` — scaling characterisation, precise part (handoff theorem #6)

The handoff flags scaling as partly outside Lean. The exactly-statable
recurrent-memory part is proved here; the expressivity side stays empirical
(documented in `docs/bs-deltagridnet-trainability.md` §4c).

| Theorem | Statement |
|---|---|
| `state_card` | one band's recurrent state holds `H · d_k · d_v` real scalars |
| `bandSplit_state_le` | per-band states (`K ≤ F`) reduce the budget `F·H·d_k·d_v → K·H·d_k·d_v` |
| `bandSplit_state_lt` | strict reduction when `K < F` and the per-band state is nonempty |

## Status

All six candidate theorems from the issue-#186 handoff are formalised
(`#1`–`#5` fully; `#6` for its precisely-statable recurrent-memory part, with the
expressivity side left as documented analysis). `lake build` is green and every
theorem type-checks with no `sorry`.
