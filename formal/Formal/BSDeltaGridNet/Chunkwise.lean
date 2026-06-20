/-
Copyright (c) 2026 penta2himajin. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: penta2himajin
-/
import Mathlib

/-!
# BS-DeltaGridNet — Chunkwise ⇔ recurrent equivalence

This formalises candidate theorem #2 from the issue-#186 handoff: the chunkwise
(WY / UT-transform) training path computes exactly the same thing as the per-step
recurrence.

The scalar-decay gated-delta recurrence (`recurrent_gated_delta` in
`training/tse/experiments/bs_deltagridnet_spike.py`) updates the state
`S ∈ ℝ^{d×dᵥ}` by

`Sₜ = αₜ Sₜ₋₁ + βₜ kₜ (zₜ − αₜ Sₜ₋₁ᵀ kₜ)ᵀ = Aₜ Sₜ₋₁ + Cₜ`,

with `Aₜ = αₜ (I − βₜ kₜ kₜᵀ)` and `Cₜ = βₜ kₜ zₜᵀ` (this is exactly the
`amap`/`cmap` of `forward_parallel`).  So the state evolution is a **first-order
affine matrix recurrence**, and the chunkwise kernel is a way of evaluating it.

Two facts make "chunkwise = recurrent" precise, and this file proves both with no
`sorry`:

* `recurrent_flatten` / `chunkwise_eq_recurrent` — **chunk-carry exactness**:
  splitting the timesteps into *arbitrary* chunks and folding chunk-by-chunk while
  carrying the boundary state reproduces the per-step result identically.  (This
  holds for any step function — it is the structural guarantee behind processing
  `T` steps as `T/chunk` chunk-steps.)
* `recurrent_linear` — **superposition / linearity in the initial state**:
  `recurrent S₀ ps = linPart ps * S₀ + recurrent 0 ps`, where `linPart` is the
  product of the per-step linear maps `Aₜ`.  This is what lets each chunk be
  computed as a single linear "carry" of the boundary state (`linPart * S₀`, the
  γ-decayed carry in the kernel) **plus** a chunk-local term evaluated from zero
  state (the WY matmul / triangular solve).

The per-step linear map `Aₜ = αₜ (I − βₜ kₜ kₜᵀ)` is exactly the operator whose
spectral norm is bounded by `1` in `Stability.lean` (theorem #1), so a bounded
`‖αₜ‖ ≤ 1` makes `linPart` non-expansive and the carry stable.
-/

namespace BSDeltaGridNet.Chunkwise

variable {d dv : ℕ}

/-- The recurrent state lives in `ℝ^{d × dᵥ}`. -/
abbrev St (d dv : ℕ) := Matrix (Fin d) (Fin dv) ℝ

/-- The per-step linear map `Aₜ` lives in `ℝ^{d × d}`. -/
abbrev Tr (d : ℕ) := Matrix (Fin d) (Fin d) ℝ

/-- One affine recurrence step `S ↦ A·S + C`, with the step data `p = (A, C)`.
The argument order (state first) matches `List.foldl`. -/
def affineStep (S : St d dv) (p : Tr d × St d dv) : St d dv := p.1 * S + p.2

/-- The recurrent state after folding the affine step over the whole sequence
`ps` of step data, starting from `S₀`.  This is the per-step recurrence. -/
def recurrent (S₀ : St d dv) (ps : List (Tr d × St d dv)) : St d dv :=
  ps.foldl affineStep S₀

@[simp] theorem recurrent_nil (S₀ : St d dv) : recurrent S₀ [] = S₀ := rfl

theorem recurrent_cons (S₀ : St d dv) (p : Tr d × St d dv) (ps : List (Tr d × St d dv)) :
    recurrent S₀ (p :: ps) = recurrent (affineStep S₀ p) ps := rfl

/-! ## 1. Chunk-carry exactness -/

/-- Processing two consecutive blocks in sequence, carrying the boundary state,
equals processing their concatenation. -/
theorem recurrent_append (S₀ : St d dv) (ps qs : List (Tr d × St d dv)) :
    recurrent S₀ (ps ++ qs) = recurrent (recurrent S₀ ps) qs := by
  simp [recurrent, List.foldl_append]

/-- **Chunk-carry exactness.** For *any* splitting of the timesteps into chunks,
folding chunk-by-chunk while carrying the boundary state reproduces the per-step
recurrence over the flattened sequence exactly. -/
theorem recurrent_flatten (S₀ : St d dv) (chunks : List (List (Tr d × St d dv))) :
    recurrent S₀ chunks.flatten = chunks.foldl recurrent S₀ := by
  induction chunks generalizing S₀ with
  | nil => simp [recurrent]
  | cons c cs ih => simp [List.flatten_cons, recurrent_append, ih]

/-- **Chunkwise ⇔ recurrent.** The chunkwise schedule (fold over chunks, carrying
state) computes exactly the per-step recurrence. -/
theorem chunkwise_eq_recurrent (S₀ : St d dv) (chunks : List (List (Tr d × St d dv))) :
    chunks.foldl recurrent S₀ = recurrent S₀ chunks.flatten :=
  (recurrent_flatten S₀ chunks).symm

/-! ## 2. Superposition: linearity in the initial state -/

/-- The linear part of the recurrence: the product `Aₙ ⋯ A₁` of the per-step
linear maps (newest on the left), i.e. the map the boundary state is carried
through. -/
def linPart : List (Tr d × St d dv) → Tr d
  | [] => 1
  | p :: ps => linPart ps * p.1

@[simp] theorem linPart_nil : linPart ([] : List (Tr d × St d dv)) = 1 := rfl

theorem linPart_cons (p : Tr d × St d dv) (ps : List (Tr d × St d dv)) :
    linPart (p :: ps) = linPart ps * p.1 := rfl

/-- **Superposition.** The affine recurrence is linear in its initial state:
the result splits into a linear carry of `S₀` through `linPart ps`, plus the
recurrence run from the zero state.  This is the identity that lets the chunkwise
kernel carry the boundary state with a single matmul (`linPart * S₀`) and compute
the rest (the WY / triangular-solve writes) from zero state. -/
theorem recurrent_linear (ps : List (Tr d × St d dv)) (S₀ : St d dv) :
    recurrent S₀ ps = linPart ps * S₀ + recurrent 0 ps := by
  induction ps generalizing S₀ with
  | nil => simp
  | cons p ps ih =>
    have h0 : recurrent (0 : St d dv) (p :: ps) = linPart ps * p.2 + recurrent 0 ps := by
      rw [recurrent_cons, ih (affineStep 0 p)]
      simp only [affineStep, Matrix.mul_zero, zero_add]
    rw [recurrent_cons, ih (affineStep S₀ p), h0, linPart_cons, affineStep,
      Matrix.mul_add, Matrix.mul_assoc]
    abel

/-! ## 3. The gated-delta recurrence as an instance of the affine recurrence -/

/-- The per-step linear map of the scalar-decay gated-delta core:
`Aₜ = αₜ (I − βₜ kₜ kₜᵀ)` (symmetric Householder erase × scalar decay). -/
def gatedDeltaA (α β : ℝ) (k : Fin d → ℝ) : Tr d :=
  α • (1 - β • Matrix.vecMulVec k k)

/-- The per-step affine offset of the gated-delta core: `Cₜ = βₜ kₜ zₜᵀ`. -/
def gatedDeltaC (β : ℝ) (k : Fin d → ℝ) (z : Fin dv → ℝ) : St d dv :=
  β • Matrix.vecMulVec k z

/-- The gated-delta state update is exactly the affine step with `(Aₜ, Cₜ)`,
so the whole spike recurrence `recurrent_gated_delta` is an instance of
`recurrent` and inherits both theorems above. -/
theorem gatedDelta_isAffineStep (α β : ℝ) (k : Fin d → ℝ) (z : Fin dv → ℝ) (S : St d dv) :
    gatedDeltaA α β k * S + gatedDeltaC β k z
      = affineStep S (gatedDeltaA α β k, gatedDeltaC β k z) := rfl

end BSDeltaGridNet.Chunkwise
