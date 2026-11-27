/-
Census steps (c) and (e): row sums, and the equivalence moves the censuses quotient by.

(c) a symmetric `±1` row with `x_0 = 1` and odd `n` has row sum `≡ n (mod 4)`; for best
matrices of order `n = r² + r + 1` the symmetric row sums to `±(2r + 1)`.  (The skew row sum `1`,
the sum-of-squares identity and `d² = 4n - 3` are `IsSkew.rowSum_eq_one`,
`IsQuadruple.rowSum_sq` and `IsQuadruple.best_rowSum` in the base library.)

(e) negation of a row, reversal of a row, any permutation of the rows, and one multiplier
`j ↦ u j` (`u` a unit) on all four rows map quadruples to quadruples, and preserve the skew and
symmetric types as stated, so each family (best, Williamson, G-matrices) is closed under the
moves its census uses.
-/
import GoethalsSeidel.Census.Stage
import GoethalsSeidel.Circulant

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-! ### (c) Row sums -/

theorem rowSum_eq_sub_bits {x : ZMod n → ℤ} (hx : IsSign x) :
    rowSum x = n - 2 * ∑ j, bit x j := by
  unfold rowSum
  rw [Finset.sum_congr rfl (fun j _ => eq_one_sub_two_bit hx j), Finset.sum_sub_distrib,
    ← Finset.mul_sum]
  simp [Finset.card_univ, ZMod.card]

/-- (c) A symmetric `±1` row of odd length with `x_0 = 1` has row sum `≡ n (mod 4)`
(`wil_hkt3.c`: "normalised to x_0 = +1, which fixes the sign of its row sum"). -/
theorem rowSum_symm_modEq (hn : Odd n) {x : ZMod n → ℤ} (hx : IsSign x) (hsym : IsSymmetric x)
    (h0 : x 0 = 1) : rowSum x ≡ n [ZMOD 4] := by
  obtain ⟨m, hm⟩ := sum_bit_symm hn hsym
  have hb0 : bit x 0 = 0 := by unfold bit; simp [h0]
  rw [rowSum_eq_sub_bits hx, hm, hb0, Int.modEq_iff_dvd]
  exact ⟨m, by ring⟩

/-- (c) Best matrices of order `n = r² + r + 1`: the symmetric row sums to `±(2r + 1)`. -/
theorem _root_.GoethalsSeidel.IsQuadruple.best_rowSum_d {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) (hb : IsSkew b) (hc : IsSkew c) {r : ℤ} (hr : (n : ℤ) = r ^ 2 + r + 1) :
    rowSum d = 2 * r + 1 ∨ rowSum d = -(2 * r + 1) := by
  have hs := h.best_rowSum ha hb hc
  have : rowSum d ^ 2 = (2 * r + 1) ^ 2 := by rw [hs, hr]; ring
  exact sq_eq_sq_iff_eq_or_eq_neg.mp this

/-! ### (e) Moves -/

theorem paf_neg_row (x : ZMod n → ℤ) (s : ZMod n) : paf (fun j => -x j) s = paf x s := by
  unfold paf; simp

theorem paf_rev (x : ZMod n → ℤ) (s : ZMod n) : paf (rev x) s = paf x s := by
  unfold paf
  refine Fintype.sum_equiv ((Equiv.neg (ZMod n)).trans (Equiv.subRight s)) _ _ (fun j => ?_)
  simp only [Equiv.trans_apply, Equiv.neg_apply, Equiv.subRight_apply, rev]
  rw [show -j - s + s = -j by ring, show -(j + s) = -j - s by ring, mul_comm]

/-- Multiplication by a unit. -/
def mulRow (u : (ZMod n)ˣ) (x : ZMod n → ℤ) : ZMod n → ℤ := fun j => x ((u : ZMod n) * j)

theorem paf_mulRow (u : (ZMod n)ˣ) (x : ZMod n → ℤ) (s : ZMod n) :
    paf (mulRow u x) s = paf x ((u : ZMod n) * s) := by
  unfold paf mulRow
  refine Fintype.sum_equiv (Units.mulLeft u) _ _ (fun j => ?_)
  simp [Units.mulLeft, mul_add]

theorem isSign_neg {x : ZMod n → ℤ} (hx : IsSign x) : IsSign (fun j => -x j) := by
  intro j; rcases hx j with h | h <;> simp [h]

theorem isSign_rev {x : ZMod n → ℤ} (hx : IsSign x) : IsSign (rev x) := fun j => hx (-j)

theorem isSign_mulRow (u : (ZMod n)ˣ) {x : ZMod n → ℤ} (hx : IsSign x) : IsSign (mulRow u x) :=
  fun j => hx _

theorem isSymmetric_neg {x : ZMod n → ℤ} (hx : IsSymmetric x) : IsSymmetric (fun j => -x j) := by
  intro j; simp [hx j]

theorem isSymmetric_rev {x : ZMod n → ℤ} (hx : IsSymmetric x) : rev x = x := by
  funext j; exact hx j

theorem isSkew_rev {x : ZMod n → ℤ} (hx : IsSkew x) : IsSkew (rev x) := by
  refine ⟨by simp [rev, hx.1], fun j hj => ?_⟩
  simp only [rev, neg_neg]
  have := hx.2 j hj
  rw [this, neg_neg]

theorem isSymmetric_mulRow (u : (ZMod n)ˣ) {x : ZMod n → ℤ} (hx : IsSymmetric x) :
    IsSymmetric (mulRow u x) := by
  intro j; simp only [mulRow, mul_neg]; exact hx _

theorem isSkew_mulRow (u : (ZMod n)ˣ) {x : ZMod n → ℤ} (hx : IsSkew x) :
    IsSkew (mulRow u x) := by
  refine ⟨by simp [mulRow, hx.1], fun j hj => ?_⟩
  simp only [mulRow, mul_neg]
  exact hx.2 _ (by simpa using hj)

/-- Reversing a skew row complements its sign bits off `0` (R5, R7, R8 in `best_hkt2.c`). -/
theorem bit_rev_skew {x : ZMod n → ℤ} (hs : IsSign x) (hx : IsSkew x) {j : ZMod n} (hj : j ≠ 0) :
    bit (rev x) j = 1 - bit x j := by
  unfold bit rev
  rw [hx.2 j hj]
  rcases hs j with h | h <;> simp [h]

section moves
variable {a b c d : ZMod n → ℤ}

theorem _root_.GoethalsSeidel.IsQuadruple.neg_a (h : IsQuadruple a b c d) : IsQuadruple (fun j => -a j) b c d :=
  ⟨isSign_neg h.signA, h.signB, h.signC, h.signD,
    fun s hs => by rw [paf_neg_row]; exact h.paf_zero s hs⟩

theorem _root_.GoethalsSeidel.IsQuadruple.rev_a (h : IsQuadruple a b c d) : IsQuadruple (rev a) b c d :=
  ⟨isSign_rev h.signA, h.signB, h.signC, h.signD,
    fun s hs => by rw [paf_rev]; exact h.paf_zero s hs⟩

theorem _root_.GoethalsSeidel.IsQuadruple.swap_ab (h : IsQuadruple a b c d) : IsQuadruple b a c d :=
  ⟨h.signB, h.signA, h.signC, h.signD,
    fun s hs => by have := h.paf_zero s hs; linarith⟩

theorem _root_.GoethalsSeidel.IsQuadruple.swap_bc (h : IsQuadruple a b c d) : IsQuadruple a c b d :=
  ⟨h.signA, h.signC, h.signB, h.signD,
    fun s hs => by have := h.paf_zero s hs; linarith⟩

theorem _root_.GoethalsSeidel.IsQuadruple.swap_cd (h : IsQuadruple a b c d) : IsQuadruple a b d c :=
  ⟨h.signA, h.signB, h.signD, h.signC,
    fun s hs => by have := h.paf_zero s hs; linarith⟩

/-- Rotating the rows `(a, b, c, d) ↦ (b, c, d, a)`; with the three transpositions above this
gives every permutation of the rows. -/
theorem _root_.GoethalsSeidel.IsQuadruple.cycle (h : IsQuadruple a b c d) : IsQuadruple b c d a :=
  ⟨h.signB, h.signC, h.signD, h.signA,
    fun s hs => by have := h.paf_zero s hs; linarith⟩

theorem _root_.GoethalsSeidel.IsQuadruple.mulRow (u : (ZMod n)ˣ) (h : IsQuadruple a b c d) :
    IsQuadruple (mulRow u a) (mulRow u b) (mulRow u c) (mulRow u d) :=
  ⟨isSign_mulRow u h.signA, isSign_mulRow u h.signB, isSign_mulRow u h.signC,
    isSign_mulRow u h.signD, fun s hs => by
      simp only [paf_mulRow]
      exact h.paf_zero _ (by simpa using hs)⟩

end moves

/-! ### The three families and their closure -/

/-- Best matrices: `a, b, c` skew, `d` symmetric. -/
def IsBest (a b c d : ZMod n → ℤ) : Prop :=
  IsQuadruple a b c d ∧ IsSkew a ∧ IsSkew b ∧ IsSkew c ∧ IsSymmetric d

/-- Williamson matrices: four symmetric rows. -/
def IsWilliamson (a b c d : ZMod n → ℤ) : Prop :=
  IsQuadruple a b c d ∧ IsSymmetric a ∧ IsSymmetric b ∧ IsSymmetric c ∧ IsSymmetric d

/-- G-matrices (`kkss` on the board): `a, b` skew, `c, d` symmetric. -/
def IsGMat (a b c d : ZMod n → ℤ) : Prop :=
  IsQuadruple a b c d ∧ IsSkew a ∧ IsSkew b ∧ IsSymmetric c ∧ IsSymmetric d

section closure
variable {a b c d : ZMod n → ℤ}

theorem IsBest.neg_d (h : IsBest a b c d) : IsBest a b c (fun j => -d j) := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.cycle.cycle.cycle.neg_a.cycle, ha, hb, hc, isSymmetric_neg hd⟩

theorem IsBest.rev_a (h : IsBest a b c d) : IsBest (rev a) b c d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.rev_a, isSkew_rev ha, hb, hc, hd⟩

theorem IsBest.swap_ab (h : IsBest a b c d) : IsBest b a c d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_ab, hb, ha, hc, hd⟩

theorem IsBest.swap_bc (h : IsBest a b c d) : IsBest a c b d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_bc, ha, hc, hb, hd⟩

theorem IsBest.mulRow (u : (ZMod n)ˣ) (h : IsBest a b c d) :
    IsBest (mulRow u a) (mulRow u b) (mulRow u c) (mulRow u d) := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.mulRow u, isSkew_mulRow u ha, isSkew_mulRow u hb, isSkew_mulRow u hc,
    isSymmetric_mulRow u hd⟩

theorem IsWilliamson.neg_a (h : IsWilliamson a b c d) : IsWilliamson (fun j => -a j) b c d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.neg_a, isSymmetric_neg ha, hb, hc, hd⟩

theorem IsWilliamson.swap_ab (h : IsWilliamson a b c d) : IsWilliamson b a c d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_ab, hb, ha, hc, hd⟩

theorem IsWilliamson.swap_bc (h : IsWilliamson a b c d) : IsWilliamson a c b d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_bc, ha, hc, hb, hd⟩

theorem IsWilliamson.swap_cd (h : IsWilliamson a b c d) : IsWilliamson a b d c := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_cd, ha, hb, hd, hc⟩

theorem IsWilliamson.cycle (h : IsWilliamson a b c d) : IsWilliamson b c d a := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.cycle, hb, hc, hd, ha⟩

theorem IsWilliamson.mulRow (u : (ZMod n)ˣ) (h : IsWilliamson a b c d) :
    IsWilliamson (mulRow u a) (mulRow u b) (mulRow u c) (mulRow u d) := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.mulRow u, isSymmetric_mulRow u ha, isSymmetric_mulRow u hb, isSymmetric_mulRow u hc,
    isSymmetric_mulRow u hd⟩

theorem IsGMat.neg_c (h : IsGMat a b c d) : IsGMat a b (fun j => -c j) d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.cycle.cycle.neg_a.cycle.cycle, ha, hb, isSymmetric_neg hc, hd⟩

theorem IsGMat.swap_ab (h : IsGMat a b c d) : IsGMat b a c d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_ab, hb, ha, hc, hd⟩

theorem IsGMat.swap_cd (h : IsGMat a b c d) : IsGMat a b d c := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.swap_cd, ha, hb, hd, hc⟩

theorem IsGMat.rev_a (h : IsGMat a b c d) : IsGMat (rev a) b c d := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.rev_a, isSkew_rev ha, hb, hc, hd⟩

theorem IsGMat.mulRow (u : (ZMod n)ˣ) (h : IsGMat a b c d) :
    IsGMat (mulRow u a) (mulRow u b) (mulRow u c) (mulRow u d) := by
  obtain ⟨q, ha, hb, hc, hd⟩ := h
  exact ⟨q.mulRow u, isSkew_mulRow u ha, isSkew_mulRow u hb, isSymmetric_mulRow u hc,
    isSymmetric_mulRow u hd⟩

end closure

end GoethalsSeidel.Census
