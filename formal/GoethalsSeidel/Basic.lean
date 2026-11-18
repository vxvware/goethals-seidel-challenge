/-
Goethals–Seidel quadruples: definitions and the row-sum identity.
-/
import Mathlib

namespace GoethalsSeidel

open Finset

variable {n : ℕ} [NeZero n]

/-- A sequence indexed by `ZMod n` with every entry `1` or `-1`. -/
def IsSign (a : ZMod n → ℤ) : Prop := ∀ j, a j = 1 ∨ a j = -1

/-- The periodic autocorrelation of `a` at shift `s`. -/
def paf (a : ZMod n → ℤ) (s : ZMod n) : ℤ := ∑ j, a j * a (j + s)

/-- The row sum of a sequence. -/
def rowSum (a : ZMod n → ℤ) : ℤ := ∑ j, a j

/-- Four `±1` sequences of length `n` whose periodic autocorrelations sum to zero at every
nonzero shift: the first rows of the four circulant blocks of a Goethals–Seidel array. -/
structure IsQuadruple (a b c d : ZMod n → ℤ) : Prop where
  signA : IsSign a
  signB : IsSign b
  signC : IsSign c
  signD : IsSign d
  paf_zero : ∀ s : ZMod n, s ≠ 0 → paf a s + paf b s + paf c s + paf d s = 0

/-- A skew sequence: `a 0 = 1` and `a (-j) = -a j` for `j ≠ 0`. -/
def IsSkew (a : ZMod n → ℤ) : Prop := a 0 = 1 ∧ ∀ j : ZMod n, j ≠ 0 → a (-j) = -a j

/-- A symmetric sequence: `a (-j) = a j`. -/
def IsSymmetric (a : ZMod n → ℤ) : Prop := ∀ j : ZMod n, a (-j) = a j

theorem paf_zero_shift {a : ZMod n → ℤ} (ha : IsSign a) : paf a 0 = n := by
  have h : ∀ j : ZMod n, a j * a (j + 0) = 1 := by
    intro j
    rcases ha j with h | h <;> simp [h]
  unfold paf
  rw [Finset.sum_congr rfl (fun j _ => h j)]
  simp [ZMod.card]

theorem sum_paf (a : ZMod n → ℤ) : ∑ s, paf a s = rowSum a ^ 2 := by
  unfold paf rowSum
  rw [Finset.sum_comm]
  have h : ∀ j : ZMod n, ∑ s, a j * a (j + s) = a j * ∑ k, a k := by
    intro j
    rw [← Finset.mul_sum]
    congr 1
    exact Fintype.sum_equiv (Equiv.addLeft j) _ _ (fun _ => rfl)
  rw [Finset.sum_congr rfl (fun j _ => h j), ← Finset.sum_mul, sq]

/-- The row sums of a quadruple satisfy `a² + b² + c² + d² = 4n`. -/
theorem IsQuadruple.rowSum_sq {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    rowSum a ^ 2 + rowSum b ^ 2 + rowSum c ^ 2 + rowSum d ^ 2 = 4 * n := by
  rw [← sum_paf a, ← sum_paf b, ← sum_paf c, ← sum_paf d, ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n))]
  rw [Finset.sum_eq_zero (fun s hs => h.paf_zero s (Finset.ne_of_mem_erase hs))]
  rw [paf_zero_shift h.signA, paf_zero_shift h.signB, paf_zero_shift h.signC,
    paf_zero_shift h.signD]
  ring

/-- A skew sequence has row sum `1`. -/
theorem IsSkew.rowSum_eq_one {a : ZMod n → ℤ} (h : IsSkew a) : rowSum a = 1 := by
  have key : ∑ j ∈ univ.erase (0 : ZMod n), a j = 0 := by
    have hneg : ∑ j ∈ univ.erase (0 : ZMod n), a j = ∑ j ∈ univ.erase (0 : ZMod n), a (-j) := by
      refine Finset.sum_nbij' (fun j => -j) (fun j => -j) ?_ ?_ ?_ ?_ ?_ <;> simp
    have hsk : ∑ j ∈ univ.erase (0 : ZMod n), a (-j) = -∑ j ∈ univ.erase (0 : ZMod n), a j := by
      rw [← Finset.sum_neg_distrib]
      exact Finset.sum_congr rfl (fun j hj => h.2 j (Finset.ne_of_mem_erase hj))
    omega
  unfold rowSum
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n)), key, h.1, add_zero]

end GoethalsSeidel
