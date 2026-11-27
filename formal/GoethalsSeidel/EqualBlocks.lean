/-
Order conditions for Goethals–Seidel quadruples with repeated sequences, from the row-sum
identity.
-/
import GoethalsSeidel.Basic

namespace GoethalsSeidel

variable {n : ℕ} [NeZero n]

/-- Three equal sequences force `n ≢ 2 (mod 3)`: the row sums give `3s² + t² = 4n`. -/
theorem IsQuadruple.not_three_equal {a d : ZMod n → ℤ} (hq : IsQuadruple a a a d)
    (h3 : n % 3 = 2) : False := by
  have hs := hq.rowSum_sq
  have h' : (n : ℤ) % 3 = 2 := by exact_mod_cast h3
  have : (rowSum d ^ 2) % 3 = 2 := by omega
  have hmod : ∀ t : ℤ, t ^ 2 % 3 ≠ 2 := by
    intro t
    have : t % 3 = 0 ∨ t % 3 = 1 ∨ t % 3 = 2 := by omega
    rcases this with h | h | h <;> simp [pow_two, Int.mul_emod, h]
  exact hmod _ this

/-- Two pairs of equal sequences force `n ≢ 3 (mod 4)` for odd `n`: the row sums are odd
and `2s² + 2t² = 4n`. -/
theorem IsQuadruple.not_two_pairs {a c : ZMod n → ℤ} (hq : IsQuadruple a a c c)
    (h4 : n % 4 = 3) : False := by
  have hs := hq.rowSum_sq
  have hodd : ∀ x : ZMod n → ℤ, IsSign x → Odd (rowSum x) := by
    intro x hx
    have h2 : ∀ j, x j % 2 = 1 := by
      intro j; rcases hx j with h | h <;> simp [h]
    have hn : Odd n := by
      exact Nat.odd_iff.mpr (by omega)
    unfold rowSum
    rw [Int.odd_iff, Finset.sum_int_mod, Finset.sum_congr rfl (fun j _ => h2 j)]
    simp only [Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul, mul_one]
    exact_mod_cast Nat.odd_iff.mp hn
  obtain ⟨u, hu⟩ := hodd a hq.signA
  obtain ⟨v, hv⟩ := hodd c hq.signC
  rw [hu, hv] at hs
  have h' : (n : ℤ) % 4 = 3 := by exact_mod_cast h4
  -- `(2u+1)² + (2v+1)² = 2n`, so `2n ≡ 2 (mod 8)` and `n ≡ 1 (mod 4)`
  have : (2 * ((2 * u + 1) ^ 2 + (2 * v + 1) ^ 2)) = 4 * (n : ℤ) := by linarith
  have e8 : ((2 * u + 1) ^ 2 + (2 * v + 1) ^ 2) % 8 = 2 := by
    have : (2 * u + 1) ^ 2 + (2 * v + 1) ^ 2 = 4 * (u * (u + 1) + v * (v + 1)) + 2 := by ring
    rw [this]
    have hu' : 2 ∣ u * (u + 1) := by
      rcases Int.even_or_odd u with ⟨r, rfl⟩ | ⟨r, rfl⟩
      · exact ⟨r * (2 * r + 1), by ring⟩
      · exact ⟨(2 * r + 1) * (r + 1), by ring⟩
    have hv' : 2 ∣ v * (v + 1) := by
      rcases Int.even_or_odd v with ⟨r, rfl⟩ | ⟨r, rfl⟩
      · exact ⟨r * (2 * r + 1), by ring⟩
      · exact ⟨(2 * r + 1) * (r + 1), by ring⟩
    obtain ⟨A, hA⟩ := hu'; obtain ⟨B, hB⟩ := hv'
    rw [hA, hB]; omega
  omega

end GoethalsSeidel
