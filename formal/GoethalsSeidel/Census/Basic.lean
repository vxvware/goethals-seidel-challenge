/-
Census foundations, step (a): the mod-4 structure of periodic autocorrelations.

For a `±1` row `x` the sign bit is `e_j = (1 - x_j)/2 ∈ {0, 1}` and
`u_x(k) = Σ_j e_j (1 - e_{j+k})`, the number of `j` with `x_j = -1` and `x_{j+k} = +1`.
Then `PAF_x(k) = n - 4 u_x(k)`, and `2 u_x(k)` is the number of `j` with `x_j ≠ x_{j+k}`,
which is the quantity `popcount(w ^ rot(w, k)) >> 1` that the census programs compute.
-/
import GoethalsSeidel.Types

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-- The sign bit of `x` at `j`: `0` when `x j = 1` and `1` when `x j = -1`. -/
def bit (x : ZMod n → ℤ) (j : ZMod n) : ℤ := if x j = 1 then 0 else 1

/-- `u_x(k)`: the number of `j` with `x_j = -1` and `x_{j+k} = 1`. -/
def uval (x : ZMod n → ℤ) (k : ZMod n) : ℤ := ∑ j, bit x j * (1 - bit x (j + k))

/-- The number of sign changes at lag `k`: `#{j | x_j ≠ x_{j+k}}`. -/
def changes (x : ZMod n → ℤ) (k : ZMod n) : ℤ := ∑ j, if x j = x (j + k) then 0 else 1

theorem bit_zero_or_one (x : ZMod n → ℤ) (j : ZMod n) : bit x j = 0 ∨ bit x j = 1 := by
  unfold bit; split_ifs <;> simp

theorem bit_mul_self (x : ZMod n → ℤ) (j : ZMod n) : bit x j * bit x j = bit x j := by
  rcases bit_zero_or_one x j with h | h <;> rw [h] <;> norm_num

theorem eq_one_sub_two_bit {x : ZMod n → ℤ} (hx : IsSign x) (j : ZMod n) :
    x j = 1 - 2 * bit x j := by
  unfold bit; rcases hx j with h | h <;> simp [h]

theorem sum_shift (f : ZMod n → ℤ) (k : ZMod n) : ∑ j, f (j + k) = ∑ j, f j :=
  Fintype.sum_equiv (Equiv.addRight k) _ _ (fun _ => rfl)

/-- Step (a): `PAF_x(k) = n - 4 u_x(k)` for every `±1` row and every lag. -/
theorem paf_eq (x : ZMod n → ℤ) (hx : IsSign x) (k : ZMod n) :
    paf x k = n - 4 * uval x k := by
  have h : ∀ j, x j * x (j + k) =
      1 - 2 * bit x j - 2 * bit x (j + k) + 4 * (bit x j * bit x (j + k)) := by
    intro j; rw [eq_one_sub_two_bit hx j, eq_one_sub_two_bit hx (j + k)]; ring
  unfold paf uval
  rw [Finset.sum_congr rfl (fun j _ => h j)]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, mul_sub, mul_one]
  rw [sum_shift (fun j => bit x j) k]
  simp [Finset.card_univ, ZMod.card]
  ring

theorem uval_nonneg (x : ZMod n → ℤ) (k : ZMod n) : 0 ≤ uval x k := by
  unfold uval
  refine Finset.sum_nonneg (fun j _ => ?_)
  rcases bit_zero_or_one x j with h | h <;> rcases bit_zero_or_one x (j + k) with h' | h' <;>
    rw [h, h'] <;> norm_num

/-- Step (a): `PAF_x(k) ≡ n (mod 4)`. -/
theorem paf_modEq_four (x : ZMod n → ℤ) (hx : IsSign x) (k : ZMod n) :
    paf x k ≡ n [ZMOD 4] := by
  rw [paf_eq x hx k, Int.modEq_iff_dvd]
  exact ⟨uval x k, by ring⟩

/-- Step (a): `u_x(k) = (n - PAF_x(k)) / 4`, the definition used by `uvals` in `best_hkt2.c`. -/
theorem uval_eq_div (x : ZMod n → ℤ) (hx : IsSign x) (k : ZMod n) :
    uval x k = (n - paf x k) / 4 := by
  rw [paf_eq x hx k]; omega

/-- `2 u_x(k)` is the number of sign changes at lag `k` (`popu(w ^ rotr(w, k)) >> 1`). -/
theorem changes_eq (x : ZMod n → ℤ) (hx : IsSign x) (k : ZMod n) :
    changes x k = 2 * uval x k := by
  have h : ∀ j, (if x j = x (j + k) then (0 : ℤ) else 1) =
      bit x j + bit x (j + k) - 2 * (bit x j * bit x (j + k)) := by
    intro j
    unfold bit
    rcases hx j with h1 | h1 <;> rcases hx (j + k) with h2 | h2 <;> simp [h1, h2]
  unfold changes uval
  rw [Finset.sum_congr rfl (fun j _ => h j)]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, mul_sub, mul_one]
  rw [sum_shift (fun j => bit x j) k]
  ring

/-- The PAF is even in the lag. -/
theorem paf_neg (x : ZMod n → ℤ) (k : ZMod n) : paf x (-k) = paf x k := by
  unfold paf
  rw [← sum_shift (fun j => x j * x (j + -k)) k]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [add_neg_cancel_right, mul_comm]

/-- Step (a): for `±1` rows the PAF-sum condition at a lag is the integer identity
`u_a + u_b + u_c + u_d = n`. -/
theorem paf_sum_zero_iff (a b c d : ZMod n → ℤ) (ha : IsSign a) (hb : IsSign b)
    (hc : IsSign c) (hd : IsSign d) (k : ZMod n) :
    paf a k + paf b k + paf c k + paf d k = 0 ↔
      uval a k + uval b k + uval c k + uval d k = n := by
  rw [paf_eq a ha, paf_eq b hb, paf_eq c hc, paf_eq d hd]
  constructor <;> intro h <;> linarith

/-- Step (a): a quadruple is the same as four `±1` rows with `Σ u = n` at every nonzero lag. -/
theorem isQuadruple_iff_uval (a b c d : ZMod n → ℤ) (ha : IsSign a) (hb : IsSign b)
    (hc : IsSign c) (hd : IsSign d) :
    IsQuadruple a b c d ↔
      ∀ k : ZMod n, k ≠ 0 → uval a k + uval b k + uval c k + uval d k = n := by
  constructor
  · intro h k hk
    exact (paf_sum_zero_iff a b c d ha hb hc hd k).mp (h.paf_zero k hk)
  · intro h
    exact ⟨ha, hb, hc, hd, fun k hk => (paf_sum_zero_iff a b c d ha hb hc hd k).mpr (h k hk)⟩

/-- For odd `n`, every nonzero residue is `k` or `-k` for a natural `k` with
`1 ≤ k ≤ (n-1)/2`. -/
theorem exists_half_rep (hn : Odd n) (s : ZMod n) (hs : s ≠ 0) :
    ∃ k : ℕ, 1 ≤ k ∧ k ≤ (n - 1) / 2 ∧ (s = (k : ZMod n) ∨ s = -(k : ZMod n)) := by
  have hv : s.val ≠ 0 := by rwa [ZMod.val_ne_zero]
  have hlt : s.val < n := ZMod.val_lt s
  obtain ⟨m, hm⟩ := hn
  by_cases hle : s.val ≤ (n - 1) / 2
  · exact ⟨s.val, by omega, hle, Or.inl (ZMod.natCast_zmod_val s).symm⟩
  · refine ⟨n - s.val, by omega, by omega, Or.inr ?_⟩
    have : ((n - s.val : ℕ) : ZMod n) = -s := by
      rw [Nat.cast_sub hlt.le, ZMod.natCast_self, ZMod.natCast_zmod_val, zero_sub]
    rw [this, neg_neg]

/-- Step (a): since the PAF is even in the lag, the condition at lags `1..(n-1)/2` (the
range `k = 1..H` of every census loop) is the whole condition. -/
theorem isQuadruple_iff_half (hn : Odd n) (a b c d : ZMod n → ℤ) (ha : IsSign a)
    (hb : IsSign b) (hc : IsSign c) (hd : IsSign d) :
    IsQuadruple a b c d ↔
      ∀ k : ℕ, 1 ≤ k → k ≤ (n - 1) / 2 →
        uval a k + uval b k + uval c k + uval d k = n := by
  rw [isQuadruple_iff_uval a b c d ha hb hc hd]
  constructor
  · intro h k hk1 hk2
    apply h
    intro h0
    have := (ZMod.natCast_eq_zero_iff k n).mp h0
    have := Nat.le_of_dvd (by omega) this
    obtain ⟨m, hm⟩ := hn
    omega
  · intro h s hs
    obtain ⟨k, hk1, hk2, hk⟩ := exists_half_rep hn s hs
    rcases hk with rfl | rfl
    · exact h k hk1 hk2
    · have e : ∀ x : ZMod n → ℤ, IsSign x → uval x (-(k : ZMod n)) = uval x k := by
        intro x hx
        rw [uval_eq_div x hx, uval_eq_div x hx, paf_neg]
      rw [e a ha, e b hb, e c hc, e d hd]
      exact h k hk1 hk2

end GoethalsSeidel.Census
