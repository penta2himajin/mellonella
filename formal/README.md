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

## Roadmap

Remaining candidate theorems from the issue-#186 handoff (priority order):
causality, decay-ratio boundedness, streaming ⇔ full-sequence equivalence,
scaling characterisation.
