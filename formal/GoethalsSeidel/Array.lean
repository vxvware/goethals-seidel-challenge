/-
The Goethals–Seidel array of a quadruple is a Hadamard matrix of order `4n`.
-/
import GoethalsSeidel.Circulant
import GoethalsSeidel.ArrayAlgebra

namespace GoethalsSeidel

open Finset Matrix

variable {n : ℕ} [NeZero n]

/-- The circulant matrices over `ZMod n` as a subring of all matrices. -/
def circSubring (n : ℕ) [NeZero n] : Subring (Matrix (ZMod n) (ZMod n) ℤ) where
  carrier := Set.range circ
  mul_mem' := by
    rintro _ _ ⟨a, rfl⟩ ⟨b, rfl⟩
    exact ⟨_, (circ_mul a b).symm⟩
  one_mem' := ⟨_, circ_one.symm⟩
  add_mem' := by
    rintro _ _ ⟨a, rfl⟩ ⟨b, rfl⟩
    exact ⟨_, (circ_add a b).symm⟩
  zero_mem' := ⟨0, circ_zero.symm⟩
  neg_mem' := by
    rintro _ ⟨a, rfl⟩
    exact ⟨_, (circ_neg a).symm⟩

instance : CommRing (circSubring n) :=
  { (inferInstance : Ring (circSubring n)) with
    mul_comm := by
      rintro ⟨_, a, rfl⟩ ⟨_, b, rfl⟩
      exact Subtype.ext (circ_mul_comm a b) }

/-- A circulant as an element of the circulant subring. -/
def circK (a : ZMod n → ℤ) : circSubring n := ⟨circ a, a, rfl⟩

/-- The Goethals–Seidel array as a `4 × 4` matrix of `n × n` blocks, with `A, B, C, D` the
circulants of the four rows and `R` the back-diagonal permutation matrix:
```
  A     B R     C R     D R
 -B R   A      -Dᵀ R    Cᵀ R
 -C R   Dᵀ R    A      -Bᵀ R
 -D R  -Cᵀ R    Bᵀ R    A
```
-/
def gsBlockMatrix (a b c d : ZMod n → ℤ) :
    Matrix (Fin 4) (Fin 4) (Matrix (ZMod n) (ZMod n) ℤ) :=
  !![circ a, circ b * backId n, circ c * backId n, circ d * backId n;
     -(circ b * backId n), circ a, -((circ d)ᵀ * backId n), (circ c)ᵀ * backId n;
     -(circ c * backId n), (circ d)ᵀ * backId n, circ a, -((circ b)ᵀ * backId n);
     -(circ d * backId n), -((circ c)ᵀ * backId n), (circ b)ᵀ * backId n, circ a]

/-- The Goethals–Seidel array of four sequences: a square matrix of order `4n`. -/
def gsArray (a b c d : ZMod n → ℤ) : Matrix (Fin 4 × ZMod n) (Fin 4 × ZMod n) ℤ :=
  Matrix.comp (Fin 4) (Fin 4) (ZMod n) (ZMod n) ℤ (gsBlockMatrix a b c d)

theorem gsBlockMatrix_eq (a b c d : ZMod n → ℤ) :
    gsBlockMatrix a b c d = gsBlocks (circSubring n).subtype (backId n) (circK a)
      (circK b) (circK (rev b)) (circK c) (circK (rev c)) (circK d) (circK (rev d)) := by
  simp [gsBlockMatrix, gsBlocks, transpose_circ, circK]

theorem gsBlockMatrix_transpose (a b c d : ZMod n → ℤ) :
    (gsBlockMatrix a b c d)ᵀ.map (·ᵀ) = gsBlocksT (circSubring n).subtype (backId n)
      (circK (rev a)) (circK b) (circK (rev b)) (circK c) (circK (rev c)) (circK d)
      (circK (rev d)) := by
  ext1 i j
  fin_cases i <;> fin_cases j <;>
    simp [gsBlockMatrix, gsBlocksT, transpose_mul, transpose_circ, transpose_backId,
      backId_mul_circ, rev_rev, circK]

/-- The Gram matrices of the four circulants of a quadruple sum to `4n` times the identity. -/
theorem IsQuadruple.gram_sum {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    circ a * (circ a)ᵀ + circ b * (circ b)ᵀ + circ c * (circ c)ᵀ + circ d * (circ d)ᵀ
      = ((4 * n : ℕ) : ℤ) • (1 : Matrix (ZMod n) (ZMod n) ℤ) := by
  ext i j
  simp only [Matrix.add_apply, circ_mul_transpose, Matrix.smul_apply, Matrix.one_apply,
    smul_eq_mul]
  by_cases hij : i = j
  · subst hij
    simp only [sub_self, ite_true, mul_one]
    rw [paf_zero_shift h.signA, paf_zero_shift h.signB, paf_zero_shift h.signC,
      paf_zero_shift h.signD]
    push_cast
    ring
  · rw [if_neg hij, mul_zero]
    exact h.paf_zero (i - j) (sub_ne_zero.mpr hij)

/-- The Goethals–Seidel array `H` of a quadruple satisfies `H Hᵀ = 4n I`. -/
theorem IsQuadruple.gsArray_mul_transpose {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    gsArray a b c d * (gsArray a b c d)ᵀ
      = ((4 * n : ℕ) : ℤ) • (1 : Matrix (Fin 4 × ZMod n) (Fin 4 × ZMod n) ℤ) := by
  unfold gsArray
  rw [transpose_comp, gsBlockMatrix_transpose, gsBlockMatrix_eq]
  have hmul := (map_mul (Matrix.compRingEquiv (Fin 4) (ZMod n) ℤ)
    (gsBlocks (circSubring n).subtype (backId n) (circK a)
      (circK b) (circK (rev b)) (circK c) (circK (rev c)) (circK d) (circK (rev d)))
    (gsBlocksT (circSubring n).subtype (backId n)
      (circK (rev a)) (circK b) (circK (rev b)) (circK c) (circK (rev c)) (circK d)
      (circK (rev d)))).symm
  simp only [Matrix.compRingEquiv_apply] at hmul
  rw [hmul]
  have hR := backId_mul_backId (n := n)
  rw [gsBlocks_mul_gsBlocksT (circSubring n).subtype (backId n) (circK a) (circK (rev a))
    (circK b) (circK (rev b)) (circK c) (circK (rev c)) (circK d) (circK (rev d)) hR
    (by simpa [circK, rev_rev] using backId_mul_circ (rev a))
    (by simpa [circK] using backId_mul_circ b)
    (by simpa [circK, rev_rev] using backId_mul_circ (rev b))
    (by simpa [circK] using backId_mul_circ c)
    (by simpa [circK, rev_rev] using backId_mul_circ (rev c))
    (by simpa [circK] using backId_mul_circ d)
    (by simpa [circK, rev_rev] using backId_mul_circ (rev d))]
  have hgram : (circSubring n).subtype (circK a * circK (rev a) + circK b * circK (rev b)
      + circK c * circK (rev c) + circK d * circK (rev d))
      = ((4 * n : ℕ) : ℤ) • (1 : Matrix (ZMod n) (ZMod n) ℤ) := by
    rw [← h.gram_sum]
    simp [circK, transpose_circ]
  rw [hgram]
  ext ⟨i, k⟩ ⟨j, l⟩
  change (Matrix.diagonal (fun _ : Fin 4 =>
    ((4 * n : ℕ) : ℤ) • (1 : Matrix (ZMod n) (ZMod n) ℤ))) i j k l = _
  simp only [Matrix.diagonal_apply, Matrix.smul_apply, Matrix.one_apply, Prod.mk.injEq,
    smul_eq_mul]
  by_cases hij : i = j <;> by_cases hkl : k = l <;>
    simp [Matrix.smul_apply, Matrix.one_apply, hij, hkl]

/-- Every entry of the array of four sign sequences is `1` or `-1`. -/
theorem gsArray_entry {a b c d : ZMod n → ℤ} (ha : IsSign a) (hb : IsSign b) (hc : IsSign c)
    (hd : IsSign d) (x y : Fin 4 × ZMod n) :
    gsArray a b c d x y = 1 ∨ gsArray a b c d x y = -1 := by
  obtain ⟨i, k⟩ := x
  obtain ⟨j, l⟩ := y
  change gsBlockMatrix a b c d i j k l = 1 ∨ gsBlockMatrix a b c d i j k l = -1
  have neg_or : ∀ z : ℤ, (z = 1 ∨ z = -1) → (-z = 1 ∨ -z = -1) := by
    intro z hz
    rcases hz with h | h <;> simp [h]
  fin_cases i <;> fin_cases j <;>
    simp only [gsBlockMatrix, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.cons_val_one, Matrix.cons_val, Matrix.neg_apply, mul_backId, circ_apply,
      Matrix.transpose_apply, Fin.zero_eta, Fin.mk_one, Fin.reduceFinMk] <;>
    first
      | exact ha _ | exact hb _ | exact hc _ | exact hd _
      | exact neg_or _ (ha _) | exact neg_or _ (hb _) | exact neg_or _ (hc _)
      | exact neg_or _ (hd _)

/-- When the first block is skew, the array `H` satisfies `H + Hᵀ = 2 I`: it is of skew
type. -/
theorem IsSkew.gsArray_add_transpose {a : ZMod n → ℤ} (ha : IsSkew a) (b c d : ZMod n → ℤ) :
    gsArray a b c d + (gsArray a b c d)ᵀ
      = (2 : ℤ) • (1 : Matrix (Fin 4 × ZMod n) (Fin 4 × ZMod n) ℤ) := by
  have hA : circ a + circ (rev a) = (2 : ℤ) • (1 : Matrix (ZMod n) (ZMod n) ℤ) := by
    ext k l
    simp only [Matrix.add_apply, circ_apply, rev, Matrix.smul_apply, Matrix.one_apply,
      smul_eq_mul, neg_sub]
    by_cases hkl : k = l
    · subst hkl
      simp [ha.1]
    · have h := ha.2 (l - k) (sub_ne_zero.mpr (Ne.symm hkl))
      rw [neg_sub] at h
      simp [hkl, h]
  unfold gsArray
  rw [transpose_comp, gsBlockMatrix_transpose, gsBlockMatrix_eq]
  ext ⟨i, k⟩ ⟨j, l⟩
  have hblock : (gsBlocks (circSubring n).subtype (backId n) (circK a) (circK b)
      (circK (rev b)) (circK c) (circK (rev c)) (circK d) (circK (rev d))
      + gsBlocksT (circSubring n).subtype (backId n) (circK (rev a)) (circK b) (circK (rev b))
        (circK c) (circK (rev c)) (circK d) (circK (rev d))) i j
      = if i = j then (2 : ℤ) • (1 : Matrix (ZMod n) (ZMod n) ℤ) else 0 := by
    fin_cases i <;> fin_cases j <;> simp [gsBlocks, gsBlocksT, circK, hA]
  refine Eq.trans (show _ = (gsBlocks (circSubring n).subtype (backId n) (circK a) (circK b)
      (circK (rev b)) (circK c) (circK (rev c)) (circK d) (circK (rev d))
      + gsBlocksT (circSubring n).subtype (backId n) (circK (rev a)) (circK b) (circK (rev b))
        (circK c) (circK (rev c)) (circK d) (circK (rev d))) i j k l from rfl) ?_
  rw [hblock]
  by_cases hij : i = j <;> by_cases hkl : k = l <;>
    simp [hij, hkl, Matrix.one_apply, Matrix.smul_apply]

end GoethalsSeidel
