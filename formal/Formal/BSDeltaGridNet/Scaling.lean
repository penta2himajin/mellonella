/-
Copyright (c) 2026 penta2himajin. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: penta2himajin
-/
import Mathlib

/-!
# BS-DeltaGridNet — Scaling characterisation (precise facts)

This covers the precisely-statable part of candidate theorem #6 from the
issue-#186 handoff.  The handoff flags scaling as *partly outside Lean* — most of
it (expressivity vs depth/heads, capacity-vs-scale) is asymptotic/empirical and
is documented as analysis in `docs/bs-deltagridnet-trainability.md` §4c.  What
*can* be stated exactly is the **recurrent-state budget** and the effect of the
band-split, which this file proves:

* `state_card` — one band's recurrent state holds `H · d_k · d_v` real scalars
  (`H` heads, `d_k` key dim, `d_v` value dim).
* `bandSplit_state_le` — replacing per-bin recurrent states (`F` frequency bins)
  by per-band states (`K` bands, `K ≤ F`) reduces the total recurrent-state
  budget from `F · H · d_k · d_v` to `K · H · d_k · d_v`.

These are the recurrent-memory facts behind "band-split reduces per-bin recurrent
state count `F → K`"; the expressivity claims stay empirical.
-/

namespace BSDeltaGridNet.Scaling

/-- The index set of one band's recurrent state: a `(head, key, value)` triple. -/
abbrev StateIndex (H dk dv : ℕ) := Fin H × Fin dk × Fin dv

/-- **Recurrent-state budget.** One band's recurrent state (`heads · d_k · d_v`
gated-delta entries) holds exactly `H · d_k · d_v` real scalars. -/
theorem state_card (H dk dv : ℕ) :
    Fintype.card (StateIndex H dk dv) = H * dk * dv := by
  simp [StateIndex, Fintype.card_prod, mul_assoc]

/-- The total recurrent-state budget for `B` parallel recurrences (bins or bands),
each `H · d_k · d_v`. -/
def totalState (B H dk dv : ℕ) : ℕ := B * (H * dk * dv)

/-- **Band-split reduces the recurrent-state budget.** With `K ≤ F` (fewer bands
than frequency bins), carrying one recurrent state per band instead of per bin
reduces the total recurrent-state count. -/
theorem bandSplit_state_le {K F : ℕ} (H dk dv : ℕ) (hKF : K ≤ F) :
    totalState K H dk dv ≤ totalState F H dk dv :=
  Nat.mul_le_mul_right _ hKF

/-- The band-split is a strict reduction whenever there are strictly fewer bands
and the per-band state is nonempty. -/
theorem bandSplit_state_lt {K F : ℕ} {H dk dv : ℕ} (hKF : K < F)
    (hH : 0 < H) (hdk : 0 < dk) (hdv : 0 < dv) :
    totalState K H dk dv < totalState F H dk dv := by
  have hpos : 0 < H * dk * dv := mul_pos (mul_pos hH hdk) hdv
  exact Nat.mul_lt_mul_of_lt_of_le hKF (le_refl _) hpos

end BSDeltaGridNet.Scaling
