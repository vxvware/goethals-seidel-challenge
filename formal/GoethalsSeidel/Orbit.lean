/-
Row sums of sequences that are constant on the orbits of a multiplier subgroup.
-/
import GoethalsSeidel.Basic

namespace GoethalsSeidel

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- The row sum minus the entry at `0` is the sum over the units. -/
theorem rowSum_sub_zero (a : ZMod p → ℤ) : rowSum a - a 0 = ∑ u : (ZMod p)ˣ, a (u : ZMod p) := by
  unfold rowSum
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod p)), add_sub_cancel_left]
  symm
  refine Finset.sum_bij (fun u _ => (u : ZMod p)) ?_ ?_ ?_ ?_
  · intro u _
    simp
  · intro u _ v _ h
    exact Units.ext h
  · intro j hj
    exact ⟨Units.mk0 j (Finset.ne_of_mem_erase hj), Finset.mem_univ _, rfl⟩
  · intro u _
    rfl

/-- The number of units where an orbit-constant sign sequence is `-1` is a multiple of the
order of the subgroup. -/
theorem card_dvd_card_neg (H : Subgroup (ZMod p)ˣ) {a : ZMod p → ℤ}
    (hinv : ∀ h ∈ H, ∀ j : ZMod p, a ((h : ZMod p) * j) = a j) :
    Nat.card H ∣ Nat.card {u : (ZMod p)ˣ | a (u : ZMod p) = -1} := by
  set S : Set (ZMod p)ˣ := {u | a (u : ZMod p) = -1} with hS
  have hpre : (QuotientGroup.mk : (ZMod p)ˣ → (ZMod p)ˣ ⧸ H) ⁻¹' (QuotientGroup.mk '' S) = S := by
    ext v
    constructor
    · rintro ⟨u, hu, huv⟩
      have hmem : u⁻¹ * v ∈ H := QuotientGroup.eq.mp huv
      have hv : (v : ZMod p) = ((u⁻¹ * v : (ZMod p)ˣ) : ZMod p) * (u : ZMod p) := by
        push_cast
        field_simp
      change a (v : ZMod p) = -1
      rw [hv, hinv _ hmem]
      exact hu
    · intro hv
      exact ⟨v, hv, rfl⟩
  have h := QuotientGroup.card_preimage_mk H (QuotientGroup.mk '' S)
  rw [hpre] at h
  exact Dvd.intro _ h.symm

/-- Let `p` be an odd prime and `H` a subgroup of odd order of the units of `ZMod p`. If a
`±1` sequence is constant on the orbits of `H` under multiplication, then its row sum is
congruent to its entry at `0` modulo twice the order of `H`. -/
theorem two_mul_card_dvd_rowSum_sub (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    (hH : Odd (Nat.card H)) {a : ZMod p → ℤ} (ha : IsSign a)
    (hinv : ∀ h ∈ H, ∀ j : ZMod p, a ((h : ZMod p) * j) = a j) :
    (2 * (Nat.card H : ℤ)) ∣ rowSum a - a 0 := by
  classical
  rw [rowSum_sub_zero]
  have hterm : ∀ u : (ZMod p)ˣ,
      a (u : ZMod p) = 1 - 2 * (if a (u : ZMod p) = -1 then 1 else 0) := by
    intro u
    rcases ha (u : ZMod p) with h | h <;> simp [h]
  have hsum : ∑ u : (ZMod p)ˣ, a (u : ZMod p)
      = (Fintype.card (ZMod p)ˣ : ℤ) - 2 * (Nat.card {u : (ZMod p)ˣ | a (u : ZMod p) = -1} : ℤ) := by
    rw [Finset.sum_congr rfl (fun u _ => hterm u), Finset.sum_sub_distrib, ← Finset.mul_sum]
    simp [Finset.card_univ, Nat.card_eq_fintype_card, Fintype.card_subtype]
  obtain ⟨k, hk⟩ := card_dvd_card_neg H hinv
  have hcard : Nat.card H ∣ Fintype.card (ZMod p)ˣ := by
    rw [← Nat.card_eq_fintype_card]
    exact Subgroup.card_subgroup_dvd_card H
  obtain ⟨m, hm⟩ := hcard
  have hpodd : Odd p := (Fact.out : p.Prime).odd_of_ne_two hp
  have hunits : Fintype.card (ZMod p)ˣ = p - 1 := ZMod.card_units p
  have hmeven : Even m := by
    have hev : Even (Nat.card H * m) := by
      rw [← hm, hunits]
      exact Nat.Prime.even_sub_one Fact.out hp
    rcases Nat.even_mul.mp hev with h | h
    · exact absurd h (Nat.not_even_iff_odd.mpr hH)
    · exact h
  obtain ⟨r, hr⟩ := hmeven
  rw [hsum, hm, hk]
  refine ⟨(r : ℤ) - k, ?_⟩
  subst hr
  push_cast
  ring

end GoethalsSeidel
