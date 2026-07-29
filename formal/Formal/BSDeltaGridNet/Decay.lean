/-
Copyright (c) 2026 penta2himajin. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: penta2himajin
-/
import Mathlib

/-!
# BS-DeltaGridNet — Decay-ratio boundedness

This formalises candidate theorem #4 from the issue-#186 handoff: the decay
ratios used by the chunkwise kernel satisfy `γᵢ / γⱼ ≤ 1` for `i ≥ j`, via
log-cumsum monotonicity.

In `chunkwise_gated_delta`, decay is carried in **log space**: with
`aₜ = log αₜ ≤ 0` (since `αₜ ∈ (0, 1]`), the cumulative log-decay
`loggₙ = ∑_{t<n} aₜ` is non-increasing, so `γₙ = exp(loggₙ) = ∏_{t<n} αₜ` is
non-increasing and every ratio `γᵢ / γⱼ = exp(loggᵢ − loggⱼ) ≤ 1` for `i ≥ j`.
This is exactly the property that makes the kernel's numerical-safety fixes valid:
the masked log-difference (`mask-before-exp`) and the division-free log-space
carry `γ_last / γⱼ` never overflow, because every realised ratio is `≤ 1`.
-/

namespace BSDeltaGridNet.Decay

/-- Cumulative log-decay `loggₙ = ∑_{t<n} aₜ`, where `aₜ = log αₜ ≤ 0`. -/
def logCumsum (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ t ∈ Finset.range n, a t

/-- Decay factor `γₙ = exp(loggₙ) = ∏_{t<n} αₜ ∈ (0, 1]`. -/
noncomputable def gamma (a : ℕ → ℝ) (n : ℕ) : ℝ := Real.exp (logCumsum a n)

variable (a : ℕ → ℝ)

/-- The cumulative log-decay is non-increasing when every step log-decay is
`≤ 0` (i.e. `αₜ ≤ 1`). -/
theorem logCumsum_antitone (ha : ∀ t, a t ≤ 0) : Antitone (logCumsum a) := by
  apply antitone_nat_of_succ_le
  intro n
  simp only [logCumsum, Finset.sum_range_succ]
  linarith [ha n]

/-- The decay factor is strictly positive. -/
theorem gamma_pos (n : ℕ) : 0 < gamma a n := Real.exp_pos _

/-- The decay factor is non-increasing in time. -/
theorem gamma_antitone (ha : ∀ t, a t ≤ 0) : Antitone (gamma a) := fun _ _ hij =>
  Real.exp_le_exp.mpr (logCumsum_antitone a ha hij)

/-- The ratio `γᵢ / γⱼ` equals `exp` of the log-difference — the *unmasked*
(`i ≥ j`, lower-triangular) entry the kernel keeps. -/
theorem gamma_ratio_eq (i j : ℕ) :
    gamma a i / gamma a j = Real.exp (logCumsum a i - logCumsum a j) := by
  rw [gamma, gamma, ← Real.exp_sub]

/-- **Decay-ratio boundedness.** For `j ≤ i`, `γᵢ / γⱼ ≤ 1`.  This underpins the
mask-before-exp ratio matrix and the division-free log-space state carry in the
chunkwise kernel. -/
theorem gamma_ratio_le_one (ha : ∀ t, a t ≤ 0) {i j : ℕ} (hji : j ≤ i) :
    gamma a i / gamma a j ≤ 1 := by
  rw [div_le_one (gamma_pos a j)]
  exact gamma_antitone a ha hji

/-- The chunkwise state carry ratio `γ_last / γⱼ ≤ 1` (for `j ≤ last`): the
log-space, division-free carry never overflows. -/
theorem gamma_carry_le_one (ha : ∀ t, a t ≤ 0) {j last : ℕ} (hj : j ≤ last) :
    gamma a last / gamma a j ≤ 1 :=
  gamma_ratio_le_one a ha hj

/-- Specialisation to actual gates: if `αₜ ∈ (0, 1]` then the log-decays are `≤ 0`,
so all the above applies with `aₜ = log αₜ`. -/
theorem log_decay_nonpos {α : ℕ → ℝ} (h0 : ∀ t, 0 ≤ α t) (h1 : ∀ t, α t ≤ 1) :
    ∀ t, Real.log (α t) ≤ 0 := fun t => Real.log_nonpos (h0 t) (h1 t)

end BSDeltaGridNet.Decay
