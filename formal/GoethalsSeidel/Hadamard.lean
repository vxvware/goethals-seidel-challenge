/-
The Goethals–Seidel array of a quadruple is a Hadamard matrix in Mathlib's sense, and the
order-119 quadruple on the board gives a skew-Hadamard matrix of order 476.
-/
import GoethalsSeidel.Array
import GoethalsSeidel.Order119

namespace GoethalsSeidel

open Matrix

variable {n : ℕ} [NeZero n]

/-- The Goethals–Seidel array of a quadruple of order `n` is a Hadamard matrix; its index type
`Fin 4 × ZMod n` has `4n` elements. -/
theorem IsQuadruple.gsArray_isHadamard {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    (gsArray a b c d).IsHadamard := by
  have hcard : (Fintype.card (Fin 4 × ZMod n) : ℤ) = ((4 * n : ℕ) : ℤ) := by
    simp [ZMod.card]
  refine IsHadamard.of_mul_conjTranspose ?_ ?_ ?_
  · intro x y
    rcases gsArray_entry h.signA h.signB h.signC h.signD x y with hxy | hxy <;>
      simp [hxy, Unitary.mem_iff]
  · rw [conjTranspose_eq_transpose_of_trivial, hcard]
    exact h.gsArray_mul_transpose
  · rw [hcard]
    exact IsRegular.of_ne_zero (by exact_mod_cast (Nat.mul_ne_zero (by norm_num) (NeZero.ne n)))

theorem card_index (n : ℕ) [NeZero n] : Fintype.card (Fin 4 × ZMod n) = 4 * n := by
  simp [ZMod.card]

/-- Moving the last sequence of a quadruple to the front gives a quadruple. -/
theorem IsQuadruple.rotate {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    IsQuadruple d a b c :=
  ⟨h.signD, h.signA, h.signB, h.signC, fun s hs => by have := h.paf_zero s hs; linarith⟩

/-- A skew-Hadamard matrix of order `476` exists: a Hadamard matrix `H` on an index type with
`476` elements with `H + Hᵀ = 2 I`. It is the Goethals–Seidel array of the board entry
`n119-f302cc45a2f83e60917b`. -/
theorem exists_skew_hadamard_476 :
    ∃ H : Matrix (Fin 4 × ZMod 119) (Fin 4 × ZMod 119) ℤ,
      Fintype.card (Fin 4 × ZMod 119) = 476 ∧ H.IsHadamard ∧
        H + Hᵀ = (2 : ℤ) • (1 : Matrix (Fin 4 × ZMod 119) (Fin 4 × ZMod 119) ℤ) :=
  ⟨gsArray Order119.d Order119.a Order119.b Order119.c, card_index 119,
    Order119.isQuadruple.rotate.gsArray_isHadamard,
    Order119.isSkew.gsArray_add_transpose _ _ _⟩

end GoethalsSeidel
