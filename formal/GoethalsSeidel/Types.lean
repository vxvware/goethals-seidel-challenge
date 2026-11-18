/-
Arithmetic conditions on the order for the symmetry types on the board.
-/
import GoethalsSeidel.Basic

namespace GoethalsSeidel

variable {n : ℕ} [NeZero n]

/-- Good matrices (one skew block): the other three row sums satisfy `b² + c² + d² = 4n - 1`. -/
theorem IsQuadruple.good_rowSum {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) : rowSum b ^ 2 + rowSum c ^ 2 + rowSum d ^ 2 = 4 * n - 1 := by
  have := h.rowSum_sq
  rw [ha.rowSum_eq_one] at this
  linarith

/-- G-matrices (two skew blocks): the other two row sums satisfy `c² + d² = 4n - 2`. -/
theorem IsQuadruple.kkss_rowSum {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) (hb : IsSkew b) : rowSum c ^ 2 + rowSum d ^ 2 = 4 * n - 2 := by
  have := h.rowSum_sq
  rw [ha.rowSum_eq_one, hb.rowSum_eq_one] at this
  linarith

/-- G-matrices of order `n` exist only when `2n - 1` is a sum of two squares. -/
theorem IsQuadruple.kkss_order {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) (hb : IsSkew b) : ∃ x y : ℤ, (2 * n - 1 : ℤ) = x ^ 2 + y ^ 2 := by
  have hs := h.kkss_rowSum ha hb
  set c' := rowSum c
  set d' := rowSum d
  have heven : Even (c' + d') := by
    have h2 : Even ((c' + d') ^ 2) := ⟨2 * (n : ℤ) - 1 + c' * d', by nlinarith⟩
    exact (Int.even_pow' (by norm_num)).mp h2
  obtain ⟨x, hx⟩ := heven
  refine ⟨x, x - d', ?_⟩
  have hc : c' = 2 * x - d' := by linarith
  rw [hc] at hs
  nlinarith

/-- Best matrices (three skew blocks): the last row sum satisfies `d² = 4n - 3`. -/
theorem IsQuadruple.best_rowSum {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) (hb : IsSkew b) (hc : IsSkew c) : rowSum d ^ 2 = 4 * n - 3 := by
  have := h.rowSum_sq
  rw [ha.rowSum_eq_one, hb.rowSum_eq_one, hc.rowSum_eq_one] at this
  linarith

/-- Best matrices of order `n` exist only when `n = r² + r + 1` for an integer `r`. -/
theorem IsQuadruple.best_order {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) (hb : IsSkew b) (hc : IsSkew c) : ∃ r : ℤ, (n : ℤ) = r ^ 2 + r + 1 := by
  have hs := h.best_rowSum ha hb hc
  have hodd : Odd (rowSum d) := by
    have h2 : Odd (rowSum d ^ 2) := ⟨2 * (n : ℤ) - 2, by linarith⟩
    exact (Int.odd_pow' (by norm_num)).mp h2
  obtain ⟨r, hr⟩ := hodd
  refine ⟨r, ?_⟩
  rw [hr] at hs
  nlinarith

end GoethalsSeidel
