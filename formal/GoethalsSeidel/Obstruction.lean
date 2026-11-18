/-
No quadruple with a skew block and three orbit-constant blocks at orders 89, 101 and 107.
-/
import GoethalsSeidel.Orbit

namespace GoethalsSeidel

/-- A sequence is constant on the orbits of the subgroup `H` acting by multiplication. -/
def OrbitConstant {p : ℕ} (H : Subgroup (ZMod p)ˣ) (a : ZMod p → ℤ) : Prop :=
  ∀ h ∈ H, ∀ j : ZMod p, a ((h : ZMod p) * j) = a j

instance : Fact (Nat.Prime 89) := ⟨by norm_num⟩
instance : Fact (Nat.Prime 101) := ⟨by norm_num⟩
instance : Fact (Nat.Prime 107) := ⟨by norm_num⟩

/-- If `m` divides `B - e` with `e = ±1` and `B² < (m - 1)²`, then `B² = 1`. -/
theorem sq_eq_one_of_dvd_sub {m B e : ℤ} (hm : 2 ≤ m) (he : e = 1 ∨ e = -1) (hd : m ∣ B - e)
    (hB : B ^ 2 < (m - 1) ^ 2) : B ^ 2 = 1 := by
  obtain ⟨k, hk⟩ := hd
  have hBlt : B < m - 1 := by nlinarith
  have hBgt : -(m - 1) < B := by nlinarith
  have hk0 : k = 0 := by
    rcases he with rfl | rfl
    · have h1 : m * k < m * 1 := by nlinarith
      have h2 : m * (-1) < m * k := by nlinarith
      have := lt_of_mul_lt_mul_left h1 (by omega)
      have := lt_of_mul_lt_mul_left h2 (by omega)
      omega
    · have h1 : m * k < m * 1 := by nlinarith
      have h2 : m * (-1) < m * k := by nlinarith
      have := lt_of_mul_lt_mul_left h1 (by omega)
      have := lt_of_mul_lt_mul_left h2 (by omega)
      omega
  subst hk0
  have : B = e := by linarith
  rcases he with rfl | rfl <;> simp [this]

/-- The common core: an odd prime `p`, a subgroup `H` of odd order with `(2|H| - 1)² > 4p - 1`,
a skew block and three blocks constant on the orbits of `H` force `4p = 4`. -/
theorem skew_orbit_core {p : ℕ} [Fact p.Prime] (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    (hH : Odd (Nat.card H)) (hbig : 4 * (p : ℤ) - 1 < (2 * (Nat.card H : ℤ) - 1) ^ 2)
    (hcard : 1 < Nat.card H)
    {a b c d : ZMod p → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) :
    (p : ℤ) = 1 := by
  have hsum := hq.rowSum_sq
  rw [ha.rowSum_eq_one] at hsum
  have hm : (2 : ℤ) ≤ 2 * (Nat.card H : ℤ) := by
    have : (1 : ℤ) < (Nat.card H : ℤ) := by exact_mod_cast hcard
    linarith
  have hB0 : 0 ≤ rowSum b ^ 2 := sq_nonneg _
  have hC0 : 0 ≤ rowSum c ^ 2 := sq_nonneg _
  have hD0 : 0 ≤ rowSum d ^ 2 := sq_nonneg _
  have hB := sq_eq_one_of_dvd_sub hm (hq.signB 0)
    (two_mul_card_dvd_rowSum_sub hp H hH hq.signB hb) (by linarith)
  have hC := sq_eq_one_of_dvd_sub hm (hq.signC 0)
    (two_mul_card_dvd_rowSum_sub hp H hH hq.signC hc) (by linarith)
  have hD := sq_eq_one_of_dvd_sub hm (hq.signD 0)
    (two_mul_card_dvd_rowSum_sub hp H hH hq.signD hd) (by linarith)
  linarith

/-- Order 89: no Goethals–Seidel quadruple has a skew block and three blocks constant on the
orbits of a nontrivial multiplier subgroup of odd order. -/
theorem no_skew_orbit_quadruple_89 (H : Subgroup (ZMod 89)ˣ) (hH : Odd (Nat.card H))
    (h1 : Nat.card H ≠ 1) {a b c d : ZMod 89 → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hdvd : Nat.card H ∣ 88 := by
    have hu : Nat.card (ZMod 89)ˣ = 88 := by rw [Nat.card_eq_fintype_card, ZMod.card_units]
    have h := Subgroup.card_subgroup_dvd_card H
    rwa [hu] at h
  have hle : Nat.card H ≤ 88 := Nat.le_of_dvd (by norm_num) hdvd
  have h11 : Nat.card H = 11 := by
    generalize Nat.card H = h at *
    interval_cases h <;> simp_all (config := { decide := true })
  have := skew_orbit_core (by norm_num) H hH (by rw [h11]; norm_num) (by omega) hq ha hb hc hd
  norm_num at this

/-- Order 107: the same statement. -/
theorem no_skew_orbit_quadruple_107 (H : Subgroup (ZMod 107)ˣ) (hH : Odd (Nat.card H))
    (h1 : Nat.card H ≠ 1) {a b c d : ZMod 107 → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hdvd : Nat.card H ∣ 106 := by
    have hu : Nat.card (ZMod 107)ˣ = 106 := by rw [Nat.card_eq_fintype_card, ZMod.card_units]
    have h := Subgroup.card_subgroup_dvd_card H
    rwa [hu] at h
  have hle : Nat.card H ≤ 106 := Nat.le_of_dvd (by norm_num) hdvd
  have h53 : Nat.card H = 53 := by
    generalize Nat.card H = h at *
    interval_cases h <;> simp_all (config := { decide := true })
  have := skew_orbit_core (by norm_num) H hH (by rw [h53]; norm_num) (by omega) hq ha hb hc hd
  norm_num at this

/-- If `10` divides `B - e` with `e = ±1` and `B² ≤ 403`, then `B²` is `1`, `81`, `121` or
`361`. -/
theorem sq_mem_of_ten_dvd_sub {B e : ℤ} (he : e = 1 ∨ e = -1) (hd : (10 : ℤ) ∣ B - e)
    (hB : B ^ 2 ≤ 403) : B ^ 2 = 1 ∨ B ^ 2 = 81 ∨ B ^ 2 = 121 ∨ B ^ 2 = 361 := by
  obtain ⟨k, hk⟩ := hd
  have hBlt : B ≤ 20 := by nlinarith
  have hBgt : -20 ≤ B := by nlinarith
  have hk1 : -2 ≤ k := by rcases he with rfl | rfl <;> omega
  have hk2 : k ≤ 2 := by rcases he with rfl | rfl <;> omega
  have hBe : B = e + 10 * k := by linarith
  subst hBe
  rcases he with rfl | rfl <;> interval_cases k <;> first | (exfalso; norm_num at hB; done) | norm_num

/-- Order 101: no Goethals–Seidel quadruple has a skew block and three blocks constant on the
orbits of a nontrivial multiplier subgroup of odd order. Here the subgroup has order 5 or 25,
each row sum is `±1` modulo `10`, and `403` is not a sum of three squares from
`1, 81, 121, 361`. -/
theorem no_skew_orbit_quadruple_101 (H : Subgroup (ZMod 101)ˣ) (hH : Odd (Nat.card H))
    (h1 : Nat.card H ≠ 1) {a b c d : ZMod 101 → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hdvd : Nat.card H ∣ 100 := by
    have hu : Nat.card (ZMod 101)ˣ = 100 := by rw [Nat.card_eq_fintype_card, ZMod.card_units]
    have h := Subgroup.card_subgroup_dvd_card H
    rwa [hu] at h
  have hle : Nat.card H ≤ 100 := Nat.le_of_dvd (by norm_num) hdvd
  have h5 : 5 ∣ Nat.card H := by
    generalize Nat.card H = h at *
    interval_cases h <;> simp_all (config := { decide := true })
  have h10 : (10 : ℤ) ∣ 2 * (Nat.card H : ℤ) := by
    obtain ⟨k, hk⟩ := h5
    exact ⟨k, by rw [hk]; push_cast; ring⟩
  have hsum := hq.rowSum_sq
  rw [ha.rowSum_eq_one] at hsum
  have hB0 : 0 ≤ rowSum b ^ 2 := sq_nonneg _
  have hC0 : 0 ≤ rowSum c ^ 2 := sq_nonneg _
  have hD0 : 0 ≤ rowSum d ^ 2 := sq_nonneg _
  have hsum' : rowSum b ^ 2 + rowSum c ^ 2 + rowSum d ^ 2 = 403 := by
    push_cast at hsum
    linarith
  have hB := sq_mem_of_ten_dvd_sub (hq.signB 0)
    (h10.trans (two_mul_card_dvd_rowSum_sub (by norm_num) H hH hq.signB hb)) (by linarith)
  have hC := sq_mem_of_ten_dvd_sub (hq.signC 0)
    (h10.trans (two_mul_card_dvd_rowSum_sub (by norm_num) H hH hq.signC hc)) (by linarith)
  have hD := sq_mem_of_ten_dvd_sub (hq.signD 0)
    (h10.trans (two_mul_card_dvd_rowSum_sub (by norm_num) H hH hq.signD hd)) (by linarith)
  rcases hB with hB | hB | hB | hB <;> rcases hC with hC | hC | hC | hC <;>
    rcases hD with hD | hD | hD | hD <;> omega

end GoethalsSeidel
