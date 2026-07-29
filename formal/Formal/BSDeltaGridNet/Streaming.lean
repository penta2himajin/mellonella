/-
Copyright (c) 2026 penta2himajin. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: penta2himajin
-/
import Formal.BSDeltaGridNet.Causality

/-!
# BS-DeltaGridNet — Streaming ⇔ full-sequence equivalence

This formalises candidate theorem #5 from the issue-#186 handoff: folding the
single-step streaming function over a sequence equals the full-sequence forward
pass — the ONNX-export contract (`step` looped == `forward`).

`streamStep` is one streaming step (the exported `step`): it updates the state by
the affine rule and emits one output.  `stream` folds it over the sequence,
threading the state and accumulating outputs.  The theorem `stream_eq_forward`
says this produces exactly the full-sequence outputs (`outputs`) and the
full-sequence final state (`recurrent`).
-/

namespace BSDeltaGridNet.Chunkwise

variable {d dv : ℕ}

/-- One streaming step (the ONNX `step` export path): update the state by the
affine rule `S ↦ A·S + C`, then emit the output `q ᵥ* S'`. -/
def streamStep (S : St d dv) (s : Tr d × St d dv × (Fin d → ℝ)) :
    St d dv × (Fin dv → ℝ) :=
  let S' := affineStep S (s.1, s.2.1)
  (S', readout s.2.2 S')

/-- The full streaming pass: fold `streamStep` over the sequence, returning the
final state and the list of emitted outputs. -/
def stream (S₀ : St d dv) :
    List (Tr d × St d dv × (Fin d → ℝ)) → St d dv × List (Fin dv → ℝ)
  | [] => (S₀, [])
  | (A, C, q) :: rest =>
      let S' := affineStep S₀ (A, C)
      let r := stream S' rest
      (r.1, readout q S' :: r.2)

/-- Streaming emits exactly the full-sequence outputs. -/
theorem stream_outputs (S₀ : St d dv) (steps : List (Tr d × St d dv × (Fin d → ℝ))) :
    (stream S₀ steps).2 = outputs S₀ steps := by
  induction steps generalizing S₀ with
  | nil => rfl
  | cons s rest ih =>
    obtain ⟨A, C, q⟩ := s
    simp [stream, outputs_cons, ih]

/-- Streaming ends in the full-sequence final state. -/
theorem stream_state (S₀ : St d dv) (steps : List (Tr d × St d dv × (Fin d → ℝ))) :
    (stream S₀ steps).1 = recurrent S₀ (steps.map fun s => (s.1, s.2.1)) := by
  induction steps generalizing S₀ with
  | nil => rfl
  | cons s rest ih =>
    obtain ⟨A, C, q⟩ := s
    simp [stream, recurrent_cons, ih]

/-- **Streaming ⇔ full-sequence (ONNX contract).** Folding the single-step
streaming function over the sequence yields exactly the full-sequence outputs and
the full-sequence final state. -/
theorem stream_eq_forward (S₀ : St d dv) (steps : List (Tr d × St d dv × (Fin d → ℝ))) :
    stream S₀ steps
      = (recurrent S₀ (steps.map fun s => (s.1, s.2.1)), outputs S₀ steps) := by
  rw [Prod.ext_iff]
  exact ⟨stream_state S₀ steps, stream_outputs S₀ steps⟩

end BSDeltaGridNet.Chunkwise
