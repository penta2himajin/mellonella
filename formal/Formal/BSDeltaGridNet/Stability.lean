/-
Copyright (c) 2026 penta2himajin. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: penta2himajin
-/
import Mathlib

/-!
# BS-DeltaGridNet — Stability / non-expansiveness of the stable gated-delta core

This formalises the first candidate theorem from the issue-#186 handoff: the
**stable gated-delta key-side operator is non-expansive**.

The recurrent state of the stabilised Gated-DeltaNet core is updated by
`Sₜ = A · Sₜ₋₁ + (input)`, where the key-side operator is

`A = (I − β · k kᵀ) · diag(α)`, `‖k‖₂ = 1`, `β ∈ [0, 2]`, `α ∈ (0, 1]`.

`I − β k kᵀ` is the *symmetric Householder erase* (scalar `β` per head) and
`diag(α)` is the per-channel decay. The empirical finding (see
`docs/bs-deltagridnet-trainability.md` §3) is that `‖S‖ ≈ 1` across seeds — the
state neither blows up nor collapses. The precise property underlying that is
that `A` is *non-expansive*: `‖A x‖ ≤ ‖x‖` for every `x`, equivalently the
operator (spectral) norm satisfies `‖A‖₂ ≤ 1`. This file proves both forms with
no `sorry`.

Decomposition of the proof:
* `householder_norm_le` — `‖x − (β⟪k,x⟫)•k‖ ≤ ‖x‖` in any real inner product
  space (the genuinely interesting contraction; this is what fails for the naive
  asymmetric channel-wise erase).
* `diagLM_norm_le` — `diag(α)` is non-expansive when `|αᵢ| ≤ 1`.
* `gatedDelta_nonexpansive` — the composite `A` is non-expansive (pointwise).
* `gatedDeltaCLM_opNorm_le_one` — the bundled operator norm bound `‖A‖₂ ≤ 1`.
-/

open scoped RealInnerProductSpace

namespace BSDeltaGridNet

/-! ## 1. Symmetric Householder erase is non-expansive -/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The symmetric Householder erase `x ↦ x − (β·⟪k,x⟫)·k` is non-expansive when
`k` is a unit vector and `β ∈ [0, 2]`.

This is the key contraction property: with a *scalar* `β` the rank-one term is
symmetric, so the squared norm drops by `β(2−β)⟪k,x⟫² ≥ 0`.  (The naive
*channel-wise* erase `I − k eᵀ` is asymmetric, its operator norm can exceed `1`,
and the recurrent state diverges — see the handoff's "failed approaches".) -/
theorem householder_norm_le {β : ℝ} {k x : E} (hk : ‖k‖ = 1)
    (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) :
    ‖x - (β * inner ℝ k x) • k‖ ≤ ‖x‖ := by
  set t : ℝ := inner ℝ k x with ht
  set c : ℝ := β * t with hc
  -- squared-norm identity: ‖x − c•k‖² = ‖x‖² − β(2−β) t²
  have hck : ‖c • k‖ ^ 2 = c ^ 2 := by
    rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs, hk]; ring
  have hinner : inner ℝ x (c • k) = c * t := by
    rw [real_inner_smul_right, real_inner_comm]
  have hsq : ‖x - c • k‖ ^ 2 = ‖x‖ ^ 2 - β * (2 - β) * t ^ 2 := by
    rw [norm_sub_sq_real, hinner, hck, hc]; ring
  -- the subtracted term is non-negative
  have hterm : 0 ≤ β * (2 - β) * t ^ 2 :=
    mul_nonneg (mul_nonneg hβ0 (by linarith)) (sq_nonneg t)
  have key : ‖x - c • k‖ ^ 2 ≤ ‖x‖ ^ 2 := by rw [hsq]; linarith
  -- conclude on the norms themselves
  have := Real.sqrt_le_sqrt key
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _)] at this

/-! ## 2. Per-channel decay `diag(α)` is non-expansive -/

variable {n : ℕ}

/-- The diagonal (per-channel decay) operator `diag(α)` on Euclidean space. -/
def diagLM (α : Fin n → ℝ) :
    EuclideanSpace ℝ (Fin n) →ₗ[ℝ] EuclideanSpace ℝ (Fin n) where
  toFun x := WithLp.toLp 2 (fun i => α i * x i)
  map_add' x y := by
    apply PiLp.ext; intro i
    simp only [PiLp.add_apply]; ring
  map_smul' c x := by
    apply PiLp.ext; intro i
    simp only [PiLp.smul_apply, RingHom.id_apply, smul_eq_mul]; ring

@[simp] theorem diagLM_apply (α : Fin n → ℝ) (x : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    diagLM α x i = α i * x i := rfl

/-- `diag(α)` is non-expansive when every gain satisfies `|αᵢ| ≤ 1` (in
particular for the decay gates `αᵢ ∈ (0, 1]`). -/
theorem diagLM_norm_le {α : Fin n → ℝ} (hα : ∀ i, |α i| ≤ 1)
    (x : EuclideanSpace ℝ (Fin n)) : ‖diagLM α x‖ ≤ ‖x‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  apply Real.sqrt_le_sqrt
  apply Finset.sum_le_sum
  intro i _
  rw [diagLM_apply]
  simp only [Real.norm_eq_abs, abs_mul, mul_pow]
  have h1 : |α i| ^ 2 ≤ 1 := by nlinarith [abs_nonneg (α i), hα i]
  nlinarith [sq_nonneg (|x i|), h1]

/-! ## 3. The composite gated-delta operator `A = (I − β k kᵀ) diag(α)` -/

/-- Non-expansiveness of the stable gated-delta key-side operator, pointwise:
applying the symmetric Householder erase to the decayed state never increases the
norm.  This is exactly the empirical `‖S‖ ≤ ‖previous‖`-style stability. -/
theorem gatedDelta_nonexpansive {β : ℝ} {k : EuclideanSpace ℝ (Fin n)} {α : Fin n → ℝ}
    (hk : ‖k‖ = 1) (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) (hα : ∀ i, |α i| ≤ 1)
    (x : EuclideanSpace ℝ (Fin n)) :
    ‖(diagLM α x) - (β * inner ℝ k (diagLM α x)) • k‖ ≤ ‖x‖ :=
  (householder_norm_le hk hβ0 hβ2).trans (diagLM_norm_le hα x)

/-! ## 4. Bundled operator-norm statement `‖A‖₂ ≤ 1` -/

/-- The symmetric Householder erase as a continuous linear map
`x ↦ x − β·⟪k,x⟫·k`. -/
noncomputable def householderCLM (β : ℝ) (k : E) : E →L[ℝ] E :=
  ContinuousLinearMap.id ℝ E - β • (innerSL ℝ k).smulRight k

@[simp] theorem householderCLM_apply (β : ℝ) (k y : E) :
    householderCLM β k y = y - (β * inner ℝ k y) • k := by
  simp [householderCLM, ContinuousLinearMap.smulRight_apply, innerSL_apply_apply, smul_smul]

/-- The full stable gated-delta key-side operator `A = (I − β k kᵀ) · diag(α)` as
a continuous linear map on Euclidean space. -/
noncomputable def gatedDeltaCLM (β : ℝ) (k : EuclideanSpace ℝ (Fin n)) (α : Fin n → ℝ) :
    EuclideanSpace ℝ (Fin n) →L[ℝ] EuclideanSpace ℝ (Fin n) :=
  (householderCLM β k).comp (diagLM α).toContinuousLinearMap

/-- **Main theorem (operator-norm form).** The stable gated-delta operator
`A = (I − β k kᵀ) diag(α)` with unit key `‖k‖ = 1`, scalar erase `β ∈ [0, 2]`,
and per-channel decay `|αᵢ| ≤ 1` satisfies `‖A‖₂ ≤ 1`; hence the recurrent state
stays bounded. -/
theorem gatedDeltaCLM_opNorm_le_one {β : ℝ} {k : EuclideanSpace ℝ (Fin n)} {α : Fin n → ℝ}
    (hk : ‖k‖ = 1) (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) (hα : ∀ i, |α i| ≤ 1) :
    ‖gatedDeltaCLM β k α‖ ≤ 1 := by
  refine ContinuousLinearMap.opNorm_le_bound _ zero_le_one (fun x => ?_)
  rw [one_mul, gatedDeltaCLM, ContinuousLinearMap.comp_apply,
    LinearMap.coe_toContinuousLinearMap', householderCLM_apply]
  exact (householder_norm_le hk hβ0 hβ2).trans (diagLM_norm_le hα x)

end BSDeltaGridNet
