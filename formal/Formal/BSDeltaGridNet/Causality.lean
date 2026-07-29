/-
Copyright (c) 2026 penta2himajin. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: penta2himajin
-/
import Formal.BSDeltaGridNet.Chunkwise

/-!
# BS-DeltaGridNet — Causality of the gated-delta core

This formalises candidate theorem #3 from the issue-#186 handoff: the output `oₜ`
depends only on the inputs at times `≤ t`.

In the spike (`recurrent_gated_delta`), the output at each step is
`oₜ = Sₜᵀ qₜ`, where `Sₜ` is the recurrent state *after* absorbing step `t`, and
`Sₜ` is produced from `Sₜ₋₁` by the affine update of `Chunkwise.lean`.  We model
the per-step data as `(Aₜ, Cₜ, qₜ)`: `(Aₜ, Cₜ)` drives the affine state update and
`qₜ` reads the output `qₜ ᵥ* Sₜ` (`= Sₜᵀ qₜ`).

`outputs S₀ steps` is the resulting list of outputs.  Causality is the statement
that the future cannot affect the past:

* `outputs_take_prefix` — running the model on the first `n` steps yields exactly
  the first `n` outputs of the full run; steps after `n` do not change them.
* `outputs_causal` — if two input streams agree on their first `n` steps, their
  first `n` outputs agree.  In particular `oₜ` is a function of inputs `≤ t`.
-/

namespace BSDeltaGridNet.Chunkwise

variable {d dv : ℕ}

/-- Per-step output: `qₜ ᵥ* Sₜ = Sₜᵀ qₜ`, the read-out of the gated-delta core. -/
def readout (q : Fin d → ℝ) (S : St d dv) : Fin dv → ℝ := Matrix.vecMul q S

/-- The list of per-step outputs `oₜ = Sₜᵀ qₜ`.  Each step `(A, C, q)` first
updates the state by the affine rule `S ↦ A·S + C`, then reads out with `q`. -/
def outputs (S₀ : St d dv) :
    List (Tr d × St d dv × (Fin d → ℝ)) → List (Fin dv → ℝ)
  | [] => []
  | (A, C, q) :: rest =>
      let S := affineStep S₀ (A, C)
      readout q S :: outputs S rest

@[simp] theorem outputs_nil (S₀ : St d dv) : outputs S₀ [] = [] := rfl

theorem outputs_cons (S₀ : St d dv) (A : Tr d) (C : St d dv) (q : Fin d → ℝ)
    (rest : List (Tr d × St d dv × (Fin d → ℝ))) :
    outputs S₀ ((A, C, q) :: rest)
      = readout q (affineStep S₀ (A, C)) :: outputs (affineStep S₀ (A, C)) rest := rfl

/-- **Causality (truncation form).** Running the core on the first `n` steps
produces exactly the first `n` outputs of the full run: steps after time `n` have
no effect on the outputs at times `< n`. -/
theorem outputs_take_prefix (S₀ : St d dv)
    (steps : List (Tr d × St d dv × (Fin d → ℝ))) (n : ℕ) :
    outputs S₀ (steps.take n) = (outputs S₀ steps).take n := by
  induction steps generalizing S₀ n with
  | nil => simp
  | cons s rest ih =>
    cases n with
    | zero => simp
    | succ m =>
      obtain ⟨A, C, q⟩ := s
      simp [outputs_cons, ih]

/-- **Causality (agreement form).** If two input streams agree on their first `n`
steps, their first `n` outputs agree.  Hence `oₜ` depends only on inputs at times
`≤ t`. -/
theorem outputs_causal (S₀ : St d dv)
    (steps₁ steps₂ : List (Tr d × St d dv × (Fin d → ℝ))) (n : ℕ)
    (h : steps₁.take n = steps₂.take n) :
    (outputs S₀ steps₁).take n = (outputs S₀ steps₂).take n := by
  rw [← outputs_take_prefix, ← outputs_take_prefix, h]

end BSDeltaGridNet.Chunkwise
