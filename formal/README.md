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

`Formal/BSDeltaGridNet/Stability.lean` — **non-expansiveness of the stable
gated-delta key-side operator** (handoff candidate theorem #1):

For `A = (I − β·k·kᵀ)·diag(α)` with `‖k‖₂ = 1`, `β ∈ [0, 2]`, `α ∈ (0, 1]`
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

## Roadmap

Further candidate theorems from the issue-#186 handoff (priority order):
chunkwise ⇔ recurrent equivalence, causality, decay-ratio boundedness,
streaming ⇔ full-sequence equivalence, scaling characterisation.
