/-
Circulant blocks over `ZMod n` and the back-diagonal permutation matrix.
-/
import GoethalsSeidel.Basic

namespace GoethalsSeidel

open Finset Matrix

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-- The circulant matrix with first row `a`: entry `(i, j)` is `a (j - i)`. -/
def circ (a : ZMod n → ℤ) : Matrix (ZMod n) (ZMod n) ℤ := Matrix.of fun i j => a (j - i)

/-- The back-diagonal permutation matrix: entry `(i, j)` is `1` when `i + j = n - 1`. -/
def backId (n : ℕ) : Matrix (ZMod n) (ZMod n) ℤ :=
  Matrix.of fun i j => if j = -1 - i then 1 else 0

/-- The sequence read backwards from `0`: `j ↦ a (-j)`. -/
def rev (a : ZMod n → ℤ) : ZMod n → ℤ := fun j => a (-j)

@[simp] theorem circ_apply (a : ZMod n → ℤ) (i j : ZMod n) : circ a i j = a (j - i) := rfl

theorem transpose_circ (a : ZMod n → ℤ) : (circ a)ᵀ = circ (rev a) := by
  ext i j
  simp [rev]

theorem rev_rev (a : ZMod n → ℤ) : rev (rev a) = a := by
  funext j
  simp [rev]

theorem transpose_backId : (backId n)ᵀ = backId n := by
  ext i j
  simp only [backId, transpose_apply, of_apply]
  congr 1
  apply propext
  constructor <;> intro h <;> rw [h] <;> ring

theorem backId_mul (M : Matrix (ZMod n) (ZMod n) ℤ) (i j : ZMod n) :
    (backId n * M) i j = M (-1 - i) j := by
  simp [backId, Matrix.mul_apply]

theorem mul_backId (M : Matrix (ZMod n) (ZMod n) ℤ) (i j : ZMod n) :
    (M * backId n) i j = M i (-1 - j) := by
  have h : ∀ k : ZMod n, (j = -1 - k) = (k = -1 - j) := by
    intro k
    apply propext
    constructor <;> intro h <;> rw [h] <;> ring
  simp [backId, Matrix.mul_apply, h]

theorem backId_mul_backId : backId n * backId n = 1 := by
  ext i j
  rw [backId_mul]
  simp only [backId, of_apply, Matrix.one_apply]
  congr 1
  apply propext
  constructor <;> intro h
  · rw [h]; ring
  · rw [h]; ring

theorem backId_mul_circ (a : ZMod n → ℤ) : backId n * circ a = circ (rev a) * backId n := by
  ext i j
  rw [backId_mul, mul_backId]
  simp only [circ_apply, rev]
  congr 1
  ring

theorem circ_add (a b : ZMod n → ℤ) : circ a + circ b = circ (a + b) := by
  ext i j
  simp

theorem circ_neg (a : ZMod n → ℤ) : -circ a = circ (-a) := by
  ext i j
  simp

theorem circ_zero : (0 : Matrix (ZMod n) (ZMod n) ℤ) = circ 0 := by
  ext i j
  simp

theorem circ_one : (1 : Matrix (ZMod n) (ZMod n) ℤ) = circ (fun s => if s = 0 then 1 else 0) := by
  ext i j
  simp only [Matrix.one_apply, circ_apply]
  congr 1
  apply propext
  constructor <;> intro h
  · rw [h]; ring
  · exact (sub_eq_zero.mp h).symm

/-- The product of two circulants is the circulant of the convolution. -/
theorem circ_mul (a b : ZMod n → ℤ) :
    circ a * circ b = circ (fun s => ∑ m, a m * b (s - m)) := by
  ext i j
  simp only [Matrix.mul_apply, circ_apply]
  refine Fintype.sum_equiv (Equiv.subRight i) _ _ (fun k => ?_)
  simp only [Equiv.subRight_apply]
  congr 2
  ring

theorem circ_mul_comm (a b : ZMod n → ℤ) : circ a * circ b = circ b * circ a := by
  rw [circ_mul, circ_mul]
  congr 1
  funext s
  refine Fintype.sum_equiv (Equiv.subLeft s) _ _ (fun m => ?_)
  simp only [Equiv.subLeft_apply]
  rw [mul_comm]
  congr 2
  ring

/-- `X Xᵀ` for a circulant `X` has entries the periodic autocorrelation. -/
theorem circ_mul_transpose (a : ZMod n → ℤ) (i j : ZMod n) :
    (circ a * (circ a)ᵀ) i j = paf a (i - j) := by
  simp only [Matrix.mul_apply, circ_apply, transpose_apply, paf]
  refine Fintype.sum_equiv (Equiv.subRight i) _ _ (fun k => ?_)
  simp only [Equiv.subRight_apply]
  congr 2
  ring

end GoethalsSeidel
