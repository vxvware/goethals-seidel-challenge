/-
Row-sum bounds for families of `±1` sequences with constant periodic autocorrelation sum whose
sequences are constant on the orbits of a multiplier subgroup.

* `IsPAFFamily k lam a`: `k` sign sequences on `ZMod n` whose periodic autocorrelations sum to
  `lam` at every nonzero shift. Goethals–Seidel quadruples are `k = 4`, `lam = 0`; the
  eight-block array uses `k = 8`, `lam = 0`; the bordered array uses `k = 4`, `lam = -4`.
* `IsPAFFamily.rowSum_sq`: `Σ sᵢ² = k n + lam (n - 1)`.
* `rowSum_orbitForm`: for any `n` and any subgroup `H` of `(ZMod n)ˣ`, if `g` divides the size
  of every orbit of `H` on the nonzero residues and `n = g E + 1`, a sign sequence constant on
  the orbits of `H` has row sum `a 0 + g m` with `m ≡ E (mod 2)` and `|m| ≤ E`.
* `rows_core`, `IsPAFFamily.orbitForm_bound`: the integer bound.
* Prime corollaries for each construction family.
-/
import GoethalsSeidel.Obstruction
import GoethalsSeidel.Types

namespace GoethalsSeidel

open Finset

/-! ### Families of sequences -/

section Family

variable {n : ℕ} [NeZero n]

/-- `k` sign sequences on `ZMod n` whose periodic autocorrelations sum to `lam` at every nonzero
shift. -/
structure IsPAFFamily (k : ℕ) (lam : ℤ) (a : Fin k → ZMod n → ℤ) : Prop where
  sign : ∀ i, IsSign (a i)
  paf_eq : ∀ s : ZMod n, s ≠ 0 → ∑ i, paf (a i) s = lam

/-- The row-sum identity `Σ sᵢ² = k n + lam (n - 1)`. -/
theorem IsPAFFamily.rowSum_sq {k : ℕ} {lam : ℤ} {a : Fin k → ZMod n → ℤ}
    (h : IsPAFFamily k lam a) :
    ∑ i, rowSum (a i) ^ 2 = k * n + lam * (n - 1) := by
  have h1 : ∑ i, rowSum (a i) ^ 2 = ∑ s, ∑ i, paf (a i) s := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl (fun i _ => (sum_paf (a i)).symm)
  rw [h1, ← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n))]
  rw [Finset.sum_congr rfl (fun s hs => h.paf_eq s (Finset.ne_of_mem_erase hs))]
  rw [Finset.sum_congr rfl (fun i _ => paf_zero_shift (h.sign i))]
  have hn : 1 ≤ n := Nat.pos_of_ne_zero (NeZero.ne n)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    Finset.card_erase_of_mem (Finset.mem_univ _), ZMod.card]
  push_cast [hn]
  ring

/-- A Goethals–Seidel quadruple is a family with `k = 4` and `lam = 0`. -/
theorem IsQuadruple.toFamily {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    IsPAFFamily 4 0 ![a, b, c, d] where
  sign i := by
    fin_cases i
    exacts [h.signA, h.signB, h.signC, h.signD]
  paf_eq s hs := by
    simp only [Fin.sum_univ_four, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_two, Matrix.cons_val_three]
    exact h.paf_zero s hs

end Family

/-! ### Row sums of orbit-constant sequences over `ZMod n` -/

section Orbit

variable {n : ℕ} [NeZero n] (H : Subgroup (ZMod n)ˣ)

open scoped Classical in
/-- The orbit of `j` under multiplication by `H`. -/
noncomputable def orbitFinset (j : ZMod n) : Finset (ZMod n) :=
  univ.image fun h : H => ((h : (ZMod n)ˣ) : ZMod n) * j

/-- `g` divides the size of every orbit of `H` on the nonzero residues. -/
def OrbitDivisor (g : ℕ) : Prop := ∀ j : ZMod n, j ≠ 0 → g ∣ (orbitFinset H j).card

/-- A finset closed under multiplication by `H`. -/
def MulStable (S : Finset (ZMod n)) : Prop :=
  ∀ h ∈ H, ∀ j ∈ S, ((h : (ZMod n)ˣ) : ZMod n) * j ∈ S

/-- The row sum is `x + g m` with `m ≡ E (mod 2)` and `-E ≤ m ≤ E`. -/
def OrbitForm (g E x s : ℤ) : Prop := ∃ m : ℤ, s = x + g * m ∧ Even (m - E) ∧ -E ≤ m ∧ m ≤ E

variable {H}

/-- If `g` divides every nonzero orbit size, it divides the size of every `H`-stable set of
nonzero residues. -/
theorem OrbitDivisor.dvd_card {g : ℕ} (hg : OrbitDivisor H g) :
    ∀ S : Finset (ZMod n), (0 : ZMod n) ∉ S → MulStable H S → g ∣ S.card := by
  intro S
  induction S using Finset.strongInduction with
  | H S ih =>
    intro h0 hst
    rcases S.eq_empty_or_nonempty with rfl | ⟨j, hj⟩
    · simp
    set O := orbitFinset H j with hO
    have hOS : O ⊆ S := by
      intro y hy
      simp only [hO, orbitFinset, Finset.mem_image, Finset.mem_univ, true_and, Subtype.exists, exists_prop] at hy
      obtain ⟨h, hh, rfl⟩ := hy
      exact hst h hh j hj
    have hjO : j ∈ O := by
      simp only [hO, orbitFinset, Finset.mem_image, Finset.mem_univ, true_and, Subtype.exists, exists_prop]
      exact ⟨1, H.one_mem, by simp⟩
    have hlt : S \ O ⊂ S := Finset.sdiff_ssubset hOS ⟨j, hjO⟩
    have hst' : MulStable H (S \ O) := by
      intro h hh y hy
      rw [Finset.mem_sdiff] at hy ⊢
      refine ⟨hst h hh y hy.1, fun hyO => hy.2 ?_⟩
      simp only [hO, orbitFinset, Finset.mem_image, Finset.mem_univ, true_and, Subtype.exists, exists_prop] at hyO ⊢
      obtain ⟨h', hh', hh''⟩ := hyO
      refine ⟨h⁻¹ * h', H.mul_mem (H.inv_mem hh) hh', ?_⟩
      have : ((h⁻¹ : (ZMod n)ˣ) : ZMod n) * ((h : ZMod n) * y) = y := by
        rw [← mul_assoc, ← Units.val_mul, inv_mul_cancel, Units.val_one, one_mul]
      rw [← this, ← hh'', Units.val_mul, mul_assoc]
    have h1 := ih _ hlt (fun h => h0 (Finset.sdiff_subset h)) hst'
    have h2 := hg j (fun h => h0 (h ▸ hj))
    rw [← Finset.card_sdiff_add_card_eq_card hOS]
    exact dvd_add h1 h2

omit [NeZero n] in
/-- A unit times a nonzero residue is nonzero. -/
theorem unit_mul_ne_zero (u : (ZMod n)ˣ) {j : ZMod n} (hj : j ≠ 0) : (u : ZMod n) * j ≠ 0 := by
  intro h
  apply hj
  have : ((u⁻¹ : (ZMod n)ˣ) : ZMod n) * ((u : ZMod n) * j) = j := by
    rw [← mul_assoc, ← Units.val_mul, inv_mul_cancel, Units.val_one, one_mul]
  rw [← this, h, mul_zero]

/-- A common divisor of the nonzero orbit sizes divides `n - 1`. -/
theorem OrbitDivisor.dvd_sub_one {g : ℕ} (hg : OrbitDivisor H g) : g ∣ n - 1 := by
  have h := hg.dvd_card (univ.erase 0) (by simp) (by
    intro h _ j hj
    simp only [Finset.mem_erase, Finset.mem_univ, and_true] at hj ⊢
    exact unit_mul_ne_zero _ hj)
  rwa [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, ZMod.card] at h

/-- The row sum as `a 0 + (n - 1) - 2N`, where `N` counts the nonzero `j` with `a j = -1`. -/
theorem rowSum_eq_neg_count {a : ZMod n → ℤ} (ha : IsSign a) :
    rowSum a = a 0 + ((n : ℤ) - 1) - 2 * (((univ.erase 0).filter fun j => a j = -1).card : ℤ) := by
  classical
  have hterm : ∀ j : ZMod n, a j = 1 - 2 * (if a j = -1 then 1 else 0) := by
    intro j
    rcases ha j with h | h <;> simp [h]
  unfold rowSum
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n))]
  rw [Finset.sum_congr rfl (fun j _ => hterm j), Finset.sum_sub_distrib, ← Finset.mul_sum,
    Finset.sum_boole]
  have hn : 1 ≤ n := Nat.pos_of_ne_zero (NeZero.ne n)
  simp only [Finset.sum_const, Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ,
    ZMod.card, nsmul_eq_mul, mul_one]
  push_cast [hn]
  ring

/-- **Orbit form of the row sum, any `n`.** Let `H` be a subgroup of the units of `ZMod n`, let
`g > 0` divide the size of every orbit of `H` on the nonzero residues, and write
`n = g E + 1`. A sign sequence constant on the orbits of `H` has row sum `a 0 + g m` with
`m ≡ E (mod 2)` and `-E ≤ m ≤ E`. -/
theorem rowSum_orbitForm {g : ℕ} {E : ℤ} (hg : OrbitDivisor H g) (hgpos : 0 < g)
    (hn : (n : ℤ) = g * E + 1) {a : ZMod n → ℤ} (ha : IsSign a) (hinv : OrbitConstant H a) :
    OrbitForm g E (a 0) (rowSum a) := by
  classical
  set S := (univ.erase (0 : ZMod n)).filter fun j => a j = -1 with hS
  have hdvd : g ∣ S.card := hg.dvd_card S (by simp [hS]) (by
    intro h hh j hj
    simp only [hS, Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true] at hj ⊢
    exact ⟨unit_mul_ne_zero _ hj.1, by rw [hinv h hh j]; exact hj.2⟩)
  obtain ⟨N, hN⟩ := hdvd
  have hle : S.card ≤ n - 1 := by
    have := Finset.card_filter_le (univ.erase (0 : ZMod n)) (fun j => a j = -1)
    rwa [Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, ZMod.card] at this
  have hn1 : 1 ≤ n := Nat.pos_of_ne_zero (NeZero.ne n)
  have hleZ : (g : ℤ) * N ≤ g * E := by
    have h1 : ((S.card : ℕ) : ℤ) ≤ ((n - 1 : ℕ) : ℤ) := by exact_mod_cast hle
    push_cast [hn1] at h1
    rw [hN] at h1
    push_cast at h1
    linarith
  have hNE : (N : ℤ) ≤ E := le_of_mul_le_mul_left hleZ (by exact_mod_cast hgpos)
  refine ⟨E - 2 * N, ?_, ⟨-(N : ℤ), by ring⟩, by linarith, by linarith [(N.cast_nonneg : (0 : ℤ) ≤ N)]⟩
  rw [rowSum_eq_neg_count ha, ← hS, hN]
  push_cast
  rw [hn]
  ring

/-- With `E` even, the orbit form gives `s ≡ x (mod 2g)`: the congruence behind the board's
obstruction at orders 89, 101 and 107 (`two_mul_card_dvd_rowSum_sub`). -/
theorem OrbitForm.two_mul_dvd {g E x s : ℤ} (h : OrbitForm g E x s) (hE : Even E) :
    2 * g ∣ s - x := by
  obtain ⟨m, rfl, hm, -, -⟩ := h
  have hme : Even m := by simpa using hm.add hE
  obtain ⟨r, hr⟩ := hme
  exact ⟨r, by rw [hr]; ring⟩

/-- A skew sequence has the orbit form with `m = 0` whenever `E` is even and nonnegative,
whether or not it is constant on any orbits. -/
theorem IsSkew.orbitForm {a : ZMod n → ℤ} (h : IsSkew a) {g E : ℤ} (hE : Even E)
    (hE0 : 0 ≤ E) : OrbitForm g E (a 0) (rowSum a) :=
  ⟨0, by rw [h.rowSum_eq_one, h.1]; ring, by simpa using hE, by linarith, hE0⟩

omit [NeZero n] in
/-- A skew sign sequence is constant on the orbits of `H` only if `-1 ∉ H`. -/
theorem IsSkew.neg_one_not_mem {a : ZMod n → ℤ} (h : IsSkew a) (ha : IsSign a)
    (h1 : (1 : ZMod n) ≠ 0) (hinv : OrbitConstant H a) : (-1 : (ZMod n)ˣ) ∉ H := by
  intro hm
  have e1 := hinv (-1) hm 1
  have e2 := h.2 1 h1
  simp only [Units.val_neg, Units.val_one, neg_mul, one_mul] at e1
  rw [e1] at e2
  rcases ha 1 with h' | h' <;> rw [h'] at e2 <;> norm_num at e2

omit [NeZero n] in
/-- A symmetric sequence constant on the orbits of `H` is constant on the orbits of the subgroup
generated by `H` and `-1`. -/
theorem IsSymmetric.orbitConstant_sup {a : ZMod n → ℤ} (hs : IsSymmetric a)
    (hinv : OrbitConstant H a) : OrbitConstant (H ⊔ Subgroup.zpowers (-1)) a := by
  let K : Subgroup (ZMod n)ˣ :=
    { carrier := {u | ∀ j, a ((u : ZMod n) * j) = a j}
      one_mem' := by intro j; simp
      mul_mem' := by
        intro u v hu hv j
        simp only [Set.mem_ofPred_eq] at hu hv
        rw [Units.val_mul, mul_assoc, hu, hv]
      inv_mem' := by
        intro u hu j
        simp only [Set.mem_ofPred_eq] at hu ⊢
        have := hu ((u⁻¹ : (ZMod n)ˣ) * j)
        rw [← mul_assoc, ← Units.val_mul, mul_inv_cancel, Units.val_one, one_mul] at this
        exact this.symm }
  have hHK : H ≤ K := fun u hu j => hinv u hu j
  have hm1 : (-1 : (ZMod n)ˣ) ∈ K := by
    intro j
    simp only [Units.val_neg, Units.val_one, neg_mul, one_mul]
    exact hs j
  have hle : H ⊔ Subgroup.zpowers (-1) ≤ K := sup_le hHK (Subgroup.zpowers_le.mpr hm1)
  intro u hu j
  exact hle hu j

end Orbit

/-! ### Integer core -/

/-- For `x = ±1` and even `m`, `g m² + 2 x m ≥ (g - 1) m²`. -/
theorem term_even {g x m : ℤ} (hx : x = 1 ∨ x = -1) (hm : Even m) :
    (g - 1) * m ^ 2 ≤ g * m ^ 2 + 2 * x * m := by
  obtain ⟨k, rfl⟩ := hm
  have : 0 ≤ k * (k + 1) := by rcases le_or_gt 0 k with h | h <;> nlinarith
  have : 0 ≤ k * (k - 1) := by rcases le_or_gt 1 k with h | h <;> nlinarith
  rcases hx with rfl | rfl <;> nlinarith

/-- For `x = ±1`, odd `m` and `g ≥ 1`, `g m² + 2 x m ≥ g - 2`. -/
theorem term_odd {g x m : ℤ} (hg : 1 ≤ g) (hx : x = 1 ∨ x = -1) (hm : Odd m) :
    g - 2 ≤ g * m ^ 2 + 2 * x * m := by
  obtain ⟨k, rfl⟩ := hm
  have hk : 0 ≤ k * (k + 1) := by rcases le_or_gt 0 k with h | h <;> nlinarith
  have h1 : 0 ≤ (g - 1) * ((2 * k + 1) ^ 2 - 1) := by
    apply mul_nonneg <;> nlinarith
  rcases hx with rfl | rfl <;> nlinarith [sq_nonneg (2 * k + 1 + 1), sq_nonneg (2 * k + 1 - 1)]

/-- **Integer core.** Let `g > 0`, and let `xᵢ = ±1` and `mᵢ ≡ E (mod 2)` for `i < r`. If
`Σ (xᵢ + g mᵢ)² = r + g T`, then `Σ (g mᵢ² + 2 xᵢ mᵢ) = T`; when `E` is even, either every
`mᵢ` is `0` or `4 (g - 1) ≤ T`; when `E` is odd, `r (g - 2) ≤ T`. -/
theorem rows_core {r : ℕ} {g E T : ℤ} (hg : 0 < g) {x m : Fin r → ℤ}
    (hx : ∀ i, x i = 1 ∨ x i = -1) (hpar : ∀ i, Even (m i - E))
    (h : ∑ i, (x i + g * m i) ^ 2 = r + g * T) :
    ∑ i, (g * m i ^ 2 + 2 * x i * m i) = T ∧
      (Even E → (∀ i, m i = 0) ∨ 4 * (g - 1) ≤ T) ∧
      (Odd E → r * (g - 2) ≤ T) := by
  have hexp : ∀ i, (x i + g * m i) ^ 2 = 1 + g * (g * m i ^ 2 + 2 * x i * m i) := by
    intro i
    rcases hx i with h | h <;> rw [h] <;> ring
  rw [Finset.sum_congr rfl (fun i _ => hexp i), Finset.sum_add_distrib, ← Finset.mul_sum] at h
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at h
  have hT : ∑ i, (g * m i ^ 2 + 2 * x i * m i) = T :=
    mul_left_cancel₀ hg.ne' (by linarith)
  refine ⟨hT, fun hE => ?_, fun hE => ?_⟩
  · have hme : ∀ i, Even (m i) := fun i => by
      have := (hpar i).add hE
      simpa using this
    by_cases hall : ∀ i, m i = 0
    · exact Or.inl hall
    right
    obtain ⟨j, hj⟩ := not_forall.mp hall
    have hnn : ∀ i ∈ univ, 0 ≤ g * m i ^ 2 + 2 * x i * m i := fun i _ => by
      have := term_even (g := g) (hx i) (hme i)
      nlinarith [sq_nonneg (m i)]
    have hsingle := Finset.single_le_sum hnn (Finset.mem_univ j)
    have hj4 : 4 ≤ m j ^ 2 := by
      obtain ⟨k, hk⟩ := hme j
      have hk0 : k ≠ 0 := by rintro rfl; simp at hk; exact hj hk
      have : 1 ≤ k ^ 2 := by
        rcases lt_or_gt_of_ne hk0 with h | h <;> nlinarith
      rw [hk]; nlinarith
    have := term_even (g := g) (hx j) (hme j)
    nlinarith
  · have hmo : ∀ i, Odd (m i) := fun i => by
      have := (hpar i).add_odd hE
      simpa using this
    have hle : ∑ _i : Fin r, (g - 2) ≤ ∑ i, (g * m i ^ 2 + 2 * x i * m i) :=
      Finset.sum_le_sum fun i _ => term_odd (by omega) (hx i) (hmo i)
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hle
    linarith

/-! ### Bounds for rows in orbit form -/

section Rows

variable {n : ℕ} [NeZero n]

/-- **Rows in orbit form.** If `r` sign sequences on `ZMod n` have row sums in orbit form for
`g > 0` and `E`, and `Σ sᵢ² = r + g T`, then for `E` even either every row sum equals the entry
at `0` or `4 (g - 1) ≤ T`, and for `E` odd `r (g - 2) ≤ T`. -/
theorem rows_orbitForm_bound {r : ℕ} {b : Fin r → ZMod n → ℤ} (hsign : ∀ i, IsSign (b i))
    {g E T : ℤ} (hg : 0 < g) (hform : ∀ i, OrbitForm g E (b i 0) (rowSum (b i)))
    (hsum : ∑ i, rowSum (b i) ^ 2 = r + g * T) :
    (Even E → (∀ i, rowSum (b i) = b i 0) ∨ 4 * (g - 1) ≤ T) ∧
      (Odd E → r * (g - 2) ≤ T) := by
  choose m hm using hform
  have hs : ∀ i, rowSum (b i) = b i 0 + g * m i := fun i => (hm i).1
  simp only [hs] at hsum
  obtain ⟨-, he, ho⟩ := rows_core hg (fun i => hsign i 0) (fun i => (hm i).2.1) hsum
  refine ⟨fun hE => ?_, ho⟩
  rcases he hE with h0 | h4
  · left
    intro i
    rw [hs i, h0 i, mul_zero, add_zero]
  · exact Or.inr h4

/-- **General family bound.** Let a family of `k` sign sequences on `ZMod n` have periodic
autocorrelation sum `lam` at every nonzero shift, let `n = g E + 1` with `g > 0`, and let every
row sum have the orbit form for `g` and `E`. Then, with `K = k + lam`: for `E` even, either every
row sum equals the entry at `0` or `4 (g - 1) ≤ K E`; for `E` odd, `k (g - 2) ≤ K E`. -/
theorem IsPAFFamily.orbitForm_bound {k : ℕ} {lam : ℤ} {a : Fin k → ZMod n → ℤ}
    (h : IsPAFFamily k lam a) {g E : ℤ} (hg : 0 < g) (hn : (n : ℤ) = g * E + 1)
    (hform : ∀ i, OrbitForm g E (a i 0) (rowSum (a i))) :
    (Even E → (∀ i, rowSum (a i) = a i 0) ∨ 4 * (g - 1) ≤ (k + lam) * E) ∧
      (Odd E → k * (g - 2) ≤ (k + lam) * E) := by
  have hsum := h.rowSum_sq
  have hsum' : ∑ i, rowSum (a i) ^ 2 = k + g * ((k + lam) * E) := by
    rw [hsum, hn]; ring
  exact rows_orbitForm_bound h.sign hg hform hsum'

/-- **General family bound for orbit-constant sequences, any `n`.** As
`IsPAFFamily.orbitForm_bound`, with the orbit form supplied by `rowSum_orbitForm`: every
sequence is constant on the orbits of `H`, and `g` divides every orbit size of `H` on the
nonzero residues. -/
theorem IsPAFFamily.orbit_bound {k : ℕ} {lam : ℤ} {a : Fin k → ZMod n → ℤ}
    (h : IsPAFFamily k lam a) (H : Subgroup (ZMod n)ˣ) {g : ℕ} {E : ℤ}
    (hg : OrbitDivisor H g) (hgpos : 0 < g) (hn : (n : ℤ) = g * E + 1)
    (hinv : ∀ i, OrbitConstant H (a i)) :
    (Even E → (∀ i, rowSum (a i) = a i 0) ∨ 4 * ((g : ℤ) - 1) ≤ (k + lam) * E) ∧
      (Odd E → k * ((g : ℤ) - 2) ≤ (k + lam) * E) :=
  h.orbitForm_bound (by exact_mod_cast hgpos) hn
    (fun i => rowSum_orbitForm hg hgpos hn (h.sign i) (hinv i))

end Rows

/-! ### Prime order -/

section Prime

variable {p : ℕ} [Fact p.Prime]

/-- At prime order every nonzero orbit of `H` has exactly `|H|` elements. -/
theorem orbitDivisor_prime (H : Subgroup (ZMod p)ˣ) : OrbitDivisor H (Nat.card H) := by
  classical
  intro j hj
  have hinj : Function.Injective fun h : H => ((h : (ZMod p)ˣ) : ZMod p) * j := by
    intro u v huv
    exact Subtype.ext (Units.ext (mul_right_cancel₀ hj huv))
  rw [orbitFinset, Finset.card_image_of_injective _ hinj, Finset.card_univ,
    Nat.card_eq_fintype_card]

theorem prime_eq_card_mul_index (H : Subgroup (ZMod p)ˣ) :
    (p : ℤ) = (Nat.card H : ℤ) * (H.index : ℤ) + 1 := by
  have hp : 2 ≤ p := (Fact.out : p.Prime).two_le
  have hidx : Nat.card H * H.index = p - 1 := by
    rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
  have h1 : p = Nat.card H * H.index + 1 := by omega
  exact_mod_cast h1

/-- The orbit form at prime order: `g = |H|`, `E` the index of `H`. -/
theorem rowSum_orbitForm_prime (H : Subgroup (ZMod p)ˣ) {a : ZMod p → ℤ} (ha : IsSign a)
    (hinv : OrbitConstant H a) : OrbitForm (Nat.card H) (H.index) (a 0) (rowSum a) :=
  rowSum_orbitForm (orbitDivisor_prime H) Nat.card_pos (prime_eq_card_mul_index H) ha hinv

/-- At prime order, a subgroup of even order contains `-1`. -/
theorem neg_one_mem_of_even_card (H : Subgroup (ZMod p)ˣ)
    (he : Even (Nat.card H)) : (-1 : (ZMod p)ˣ) ∈ H := by
  classical
  have : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩
  obtain ⟨y, hy⟩ := exists_prime_orderOf_dvd_card (G := H) 2
    (by rw [← Nat.card_eq_fintype_card]; exact even_iff_two_dvd.mp he)
  have hy2 : (y : (ZMod p)ˣ) ^ 2 = 1 := by
    rw [← hy]; exact_mod_cast pow_orderOf_eq_one y
  have hy1 : (y : (ZMod p)ˣ) ≠ 1 := by
    intro h
    have : y = 1 := Subtype.ext h
    rw [this, orderOf_one] at hy
    norm_num at hy
  have hval : ((y : (ZMod p)ˣ) : ZMod p) * ((y : (ZMod p)ˣ) : ZMod p) = 1 := by
    rw [← Units.val_mul, ← sq, hy2, Units.val_one]
  rcases mul_self_eq_one_iff.mp hval with h | h
  · exact absurd (Units.ext h) hy1
  · have : (y : (ZMod p)ˣ) = -1 := Units.ext (by rw [h]; simp)
    rw [← this]; exact y.2

/-- At prime order, a subgroup without `-1` has odd order. -/
theorem odd_card_of_neg_one_not_mem (H : Subgroup (ZMod p)ˣ)
    (h : (-1 : (ZMod p)ˣ) ∉ H) : Odd (Nat.card H) := by
  rcases Nat.even_or_odd (Nat.card H) with he | ho
  · exact absurd (neg_one_mem_of_even_card H he) h
  · exact ho

/-- A subgroup containing `-1` has even order at odd prime order. -/
theorem even_card_of_neg_one_mem (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    (h : (-1 : (ZMod p)ˣ) ∈ H) : Even (Nat.card H) := by
  have h2 : orderOf (-1 : (ZMod p)ˣ) = 2 := by
    apply orderOf_eq_prime
    · simp
    · intro h1
      have : ((-1 : (ZMod p)ˣ) : ZMod p) = 1 := by rw [h1]; rfl
      simp only [Units.val_neg, Units.val_one] at this
      have hp' : (2 : ZMod p) = 0 := by linear_combination -this
      have : ((2 : ℕ) : ZMod p) = 0 := by exact_mod_cast hp'
      rw [ZMod.natCast_eq_zero_iff] at this
      exact hp ((Nat.prime_dvd_prime_iff_eq (Fact.out) Nat.prime_two).mp this)
  have hdvd : orderOf (⟨-1, h⟩ : H) ∣ Nat.card H := orderOf_dvd_natCard _
  rw [Subgroup.orderOf_mk, h2] at hdvd
  exact even_iff_two_dvd.mpr hdvd

theorem odd_prime_of_ne_two (hp : p ≠ 2) : Odd p := (Fact.out : p.Prime).odd_of_ne_two hp

/-- `|H| · index = p - 1` is even at odd prime order. -/
theorem even_card_mul_index (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ) :
    Even (Nat.card H * H.index) := by
  rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
  exact Nat.Prime.even_sub_one Fact.out hp

/-- **Family bound at prime order.** A family of `k` sign sequences on `ZMod p` with periodic
autocorrelation sum `lam` at nonzero shifts, all constant on the orbits of `H` (order `f`,
index `e`): for `e` even, every row sum equals its entry at `0` or `4 (f - 1) ≤ (k + lam) e`; for
`e` odd, `k (f - 2) ≤ (k + lam) e`. -/
theorem IsPAFFamily.orbit_bound_prime {k : ℕ} {lam : ℤ} {a : Fin k → ZMod p → ℤ}
    (h : IsPAFFamily k lam a) (H : Subgroup (ZMod p)ˣ) (hinv : ∀ i, OrbitConstant H (a i)) :
    (Even H.index → (∀ i, rowSum (a i) = a i 0) ∨
        4 * ((Nat.card H : ℤ) - 1) ≤ (k + lam) * H.index) ∧
      (Odd H.index → k * ((Nat.card H : ℤ) - 2) ≤ (k + lam) * H.index) := by
  have := h.orbit_bound H (orbitDivisor_prime H) Nat.card_pos (prime_eq_card_mul_index H) hinv
  exact ⟨fun he => this.1 (by exact_mod_cast he), fun ho => this.2 (by exact_mod_cast ho)⟩

/-- The all-equal case `sᵢ = aᵢ 0` is impossible when `k + lam > 0`. -/
theorem IsPAFFamily.not_all_rowSum_eq_zero_entry {k : ℕ} {lam : ℤ} {a : Fin k → ZMod p → ℤ}
    (h : IsPAFFamily k lam a) (hK : 0 < k + lam) (hall : ∀ i, rowSum (a i) = a i 0) : False := by
  have hsum := h.rowSum_sq
  have hsq : ∀ i, rowSum (a i) ^ 2 = 1 := fun i => by
    rw [hall i]; rcases h.sign i 0 with h' | h' <;> rw [h'] <;> norm_num
  simp only [hsq, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    mul_one] at hsum
  have hp : (2 : ℤ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  nlinarith

/-- **Goethals–Seidel quadruples, prime order.** If all four sequences are constant on the orbits
of `H` (order `f`, index `e`), then `f ≤ e + 1`, so `p ≤ e² + e + 1`. -/
theorem IsQuadruple.card_le_index_add_one (H : Subgroup (ZMod p)ˣ) {a b c d : ZMod p → ℤ}
    (hq : IsQuadruple a b c d) (ha : OrbitConstant H a) (hb : OrbitConstant H b)
    (hc : OrbitConstant H c) (hd : OrbitConstant H d) : Nat.card H ≤ H.index + 1 := by
  have hinv : ∀ i, OrbitConstant H (![a, b, c, d] i) := by
    intro i; fin_cases i <;> assumption
  have hF := hq.toFamily
  obtain ⟨he, ho⟩ := hF.orbit_bound_prime H hinv
  rcases Nat.even_or_odd H.index with hev | hod
  · rcases he hev with hall | h4
    · exact (hF.not_all_rowSum_eq_zero_entry (by norm_num) hall).elim
    · push_cast at h4
      have : (Nat.card H : ℤ) ≤ H.index + 1 := by linarith
      exact_mod_cast this
  · have h4 := ho hod
    push_cast at h4
    have hle : Nat.card H ≤ H.index + 2 := by
      have : (Nat.card H : ℤ) ≤ H.index + 2 := by linarith
      exact_mod_cast this
    by_cases hp : p = 2
    · subst hp
      have : Nat.card H * H.index = 1 := by
        rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
      have h1 : Nat.card H = 1 := Nat.eq_one_of_mul_eq_one_right this
      omega
    · have hev := even_card_mul_index hp H
      have hfe : Even (Nat.card H) := by
        rcases Nat.even_mul.mp hev with h | h
        · exact h
        · exact absurd h (Nat.not_even_iff_odd.mpr hod)
      rcases Nat.lt_or_ge (Nat.card H) (H.index + 2) with hlt | hge
      · omega
      · have : Nat.card H = H.index + 2 := le_antisymm hle hge
        rw [this] at hfe
        have : Odd (H.index + 2) := hod.add_even even_two
        exact absurd hfe (Nat.not_even_iff_odd.mpr this)

/-- **Goethals–Seidel quadruples, all large primes.** No Goethals–Seidel quadruple of prime order
`p > e² + e + 1` has all four sequences constant on the orbits of a multiplier subgroup of index
`e`. This covers circulant Williamson, good, best, G-matrix and skew quadruples. -/
theorem no_orbit_quadruple_of_lt (H : Subgroup (ZMod p)ˣ)
    (hbig : H.index ^ 2 + H.index + 1 < p) {a b c d : ZMod p → ℤ}
    (hq : IsQuadruple a b c d) (ha : OrbitConstant H a) (hb : OrbitConstant H b)
    (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hidx : Nat.card H * H.index = p - 1 := by
    rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
  have h := hq.card_le_index_add_one H ha hb hc hd
  have h1 : Nat.card H * H.index ≤ (H.index + 1) * H.index := Nat.mul_le_mul_right _ h
  have h2 : (H.index + 1) * H.index = H.index ^ 2 + H.index := by ring
  generalize Nat.card H * H.index = X at hidx h1
  generalize H.index ^ 2 = Y at hbig h2
  omega

/-- **Eight-block array, prime order.** A family of eight sign sequences on `ZMod p` with
periodic autocorrelation sum `0` at nonzero shifts, all constant on the orbits of `H`, has
`f ≤ 2e + 1`, and `f ≤ e + 1` when `e` is odd. -/
theorem IsPAFFamily.eight_card_le (H : Subgroup (ZMod p)ˣ) {a : Fin 8 → ZMod p → ℤ}
    (h : IsPAFFamily 8 0 a) (hinv : ∀ i, OrbitConstant H (a i)) :
    Nat.card H ≤ 2 * H.index + 1 ∧ (Odd H.index → Nat.card H ≤ H.index + 1) := by
  obtain ⟨he, ho⟩ := h.orbit_bound_prime H hinv
  have hepos : 0 < H.index := Nat.pos_of_ne_zero Subgroup.index_ne_zero_of_finite
  rcases Nat.even_or_odd H.index with hev | hod
  · refine ⟨?_, fun hod => absurd hev (Nat.not_even_iff_odd.mpr hod)⟩
    rcases he hev with hall | h4
    · exact (h.not_all_rowSum_eq_zero_entry (by norm_num) hall).elim
    · push_cast at h4
      have : (Nat.card H : ℤ) ≤ 2 * H.index + 1 := by linarith
      exact_mod_cast this
  · have h4 := ho hod
    push_cast at h4
    have hle : Nat.card H ≤ H.index + 2 := by
      have : (Nat.card H : ℤ) ≤ H.index + 2 := by linarith
      exact_mod_cast this
    have hle' : Nat.card H ≤ H.index + 1 := by
      by_cases hp : p = 2
      · subst hp
        have : Nat.card H * H.index = 1 := by
          rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
        have h1 : Nat.card H = 1 := Nat.eq_one_of_mul_eq_one_right this
        omega
      · have hev := even_card_mul_index hp H
        have hfe : Even (Nat.card H) := by
          rcases Nat.even_mul.mp hev with h | h
          · exact h
          · exact absurd h (Nat.not_even_iff_odd.mpr hod)
        rcases Nat.lt_or_ge (Nat.card H) (H.index + 2) with hlt | hge
        · omega
        · have : Nat.card H = H.index + 2 := le_antisymm hle hge
          rw [this] at hfe
          exact absurd hfe (Nat.not_even_iff_odd.mpr (hod.add_even even_two))
    exact ⟨by omega, fun _ => hle'⟩

/-- **Bordered array, prime order.** A family of four sign sequences on `ZMod p` with periodic
autocorrelation sum `-4` at nonzero shifts, all constant on the orbits of `H`, with `|H| ≥ 3`:
the index of `H` is even and every row sum equals the sequence's entry at `0`. The row sums put
no upper bound on `|H|`. -/
theorem IsPAFFamily.bordered_orbit (H : Subgroup (ZMod p)ˣ) {a : Fin 4 → ZMod p → ℤ}
    (h : IsPAFFamily 4 (-4) a) (hinv : ∀ i, OrbitConstant H (a i)) (h3 : 3 ≤ Nat.card H) :
    Even H.index ∧ ∀ i, rowSum (a i) = a i 0 := by
  obtain ⟨he, ho⟩ := h.orbit_bound_prime H hinv
  have h3' : (3 : ℤ) ≤ Nat.card H := by exact_mod_cast h3
  rcases Nat.even_or_odd H.index with hev | hod
  · refine ⟨hev, ?_⟩
    rcases he hev with hall | h4
    · exact hall
    · push_cast at h4; linarith
  · have := ho hod
    push_cast at this
    linarith

/-! ### Families with skew and symmetric rows -/

/-- **One skew row, three orbit-constant rows.** Let `p` be an odd prime and `H` a subgroup of
odd order `f` and index `e`. If a Goethals–Seidel quadruple has a skew first sequence (otherwise
unrestricted) and its other three sequences constant on the orbits of `H`, then `f ≤ e + 1`.
This contains the board's obstructions at orders 89 and 107. -/
theorem IsQuadruple.skew_card_le (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ) (hH : Odd (Nat.card H))
    {a b c d : ZMod p → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) :
    Nat.card H ≤ H.index + 1 := by
  have hev : Even H.index := by
    rcases Nat.even_mul.mp (even_card_mul_index hp H) with h | h
    · exact absurd h (Nat.not_even_iff_odd.mpr hH)
    · exact h
  have hF := hq.toFamily
  have hform : ∀ i, OrbitForm (Nat.card H) H.index (![a, b, c, d] i 0)
      (rowSum (![a, b, c, d] i)) := by
    intro i
    fin_cases i
    · exact ha.orbitForm (by exact_mod_cast hev) (by positivity)
    · exact rowSum_orbitForm_prime H hq.signB hb
    · exact rowSum_orbitForm_prime H hq.signC hc
    · exact rowSum_orbitForm_prime H hq.signD hd
  have := (hF.orbitForm_bound (by exact_mod_cast Nat.card_pos) (prime_eq_card_mul_index H)
    hform).1 (by exact_mod_cast hev)
  rcases this with hall | h4
  · exact (hF.not_all_rowSum_eq_zero_entry (by norm_num) hall).elim
  · push_cast at h4
    have : (Nat.card H : ℤ) ≤ H.index + 1 := by
      have hepos : (0 : ℤ) < H.index := by
        exact_mod_cast Nat.pos_of_ne_zero Subgroup.index_ne_zero_of_finite
      linarith
    exact_mod_cast this

/-- If every sequence in a skew-type family is constant on the orbits of `H` and one is skew,
then `H` has odd order (`-1 ∉ H`). -/
theorem skew_forces_odd (H : Subgroup (ZMod p)ˣ) {a : ZMod p → ℤ} (hs : IsSkew a)
    (ha : IsSign a) (hinv : OrbitConstant H a) : Odd (Nat.card H) := by
  have h1 : (1 : ZMod p) ≠ 0 := one_ne_zero
  exact odd_card_of_neg_one_not_mem H (hs.neg_one_not_mem ha h1 hinv)

/-- Symmetric rows in a sum of squares. Let `-1 ∈ H` (order `f`, index `e`, `p` odd), and let `r`
sign sequences constant on the orbits of `H` have `Σ sᵢ² = 4p - (4 - r)`, as for the symmetric
rows of Williamson (`r = 4`), good (`r = 3`), G-matrix (`r = 2`) and best (`r = 1`) quadruples
once the skew rows (row sum `1`) are removed. Then `f ≤ e` when `e` is even, and
`r (f - 2) ≤ 4 e` when `e` is odd. -/
theorem symmetric_rows_bound (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ) (hneg : (-1 : (ZMod p)ˣ) ∈ H)
    {r : ℕ} {b : Fin r → ZMod p → ℤ} (hsign : ∀ i, IsSign (b i))
    (hinv : ∀ i, OrbitConstant H (b i))
    (hsum : ∑ i, rowSum (b i) ^ 2 = 4 * (p : ℤ) - (4 - r)) :
    (Even H.index → Nat.card H ≤ H.index) ∧
      (Odd H.index → r * ((Nat.card H : ℤ) - 2) ≤ 4 * H.index) := by
  have hpe := prime_eq_card_mul_index H
  have hsum' : ∑ i, rowSum (b i) ^ 2 = r + (Nat.card H : ℤ) * (4 * H.index) := by
    rw [hsum, hpe]; ring
  have hB := rows_orbitForm_bound hsign (by exact_mod_cast Nat.card_pos)
    (fun i => rowSum_orbitForm_prime H (hsign i) (hinv i)) hsum'
  have hfe := even_card_of_neg_one_mem hp H hneg
  refine ⟨fun hev => ?_, fun hod => hB.2 (by exact_mod_cast hod)⟩
  rcases hB.1 (by exact_mod_cast hev) with hall | h4
  · -- every row sum is `±1`, so `r = 4p - 4 + r`, impossible
    have hsq : ∀ i, rowSum (b i) ^ 2 = 1 := fun i => by
      rw [hall i]; rcases hsign i 0 with h' | h' <;> rw [h'] <;> norm_num
    simp only [hsq, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      mul_one] at hsum
    have : (2 : ℤ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
    linarith
  · have hle : (Nat.card H : ℤ) ≤ H.index + 1 := by linarith
    have hle' : Nat.card H ≤ H.index + 1 := by exact_mod_cast hle
    rcases Nat.lt_or_ge (Nat.card H) (H.index + 1) with hlt | hge
    · omega
    · have : Nat.card H = H.index + 1 := le_antisymm hle' hge
      rw [this] at hfe
      exact absurd hfe (Nat.not_even_iff_odd.mpr (hev.add_one))

/-- **Williamson quadruples, prime order.** Let `p` be an odd prime and `H` any subgroup of the
units. If the four sequences of a quadruple are symmetric and constant on the orbits of `H`, let
`H' = H ⊔ ⟨-1⟩` (they are constant on its orbits too), of order `f` and index `e`. Then `f ≤ e`
when `e` is even and `f ≤ e + 1` when `e` is odd. -/
theorem IsQuadruple.williamson_bound (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    {a b c d : ZMod p → ℤ} (hq : IsQuadruple a b c d) (sa : IsSymmetric a) (sb : IsSymmetric b)
    (sc : IsSymmetric c) (sd : IsSymmetric d) (ha : OrbitConstant H a) (hb : OrbitConstant H b)
    (hc : OrbitConstant H c) (hd : OrbitConstant H d) :
    let H' := H ⊔ Subgroup.zpowers (-1)
    (Even H'.index → Nat.card H' ≤ H'.index) ∧ (Odd H'.index → Nat.card H' ≤ H'.index + 1) := by
  intro H'
  have hneg : (-1 : (ZMod p)ˣ) ∈ H' := Subgroup.mem_sup_right (Subgroup.mem_zpowers _)
  have hinv : ∀ i, OrbitConstant H' (![a, b, c, d] i) := by
    intro i
    fin_cases i
    exacts [sa.orbitConstant_sup ha, sb.orbitConstant_sup hb, sc.orbitConstant_sup hc,
      sd.orbitConstant_sup hd]
  have hsum := hq.toFamily.rowSum_sq
  have := symmetric_rows_bound hp H' hneg hq.toFamily.sign hinv (by rw [hsum]; push_cast; ring)
  refine ⟨this.1, fun ho => ?_⟩
  have h2 := this.2 ho
  push_cast at h2
  have : (Nat.card H' : ℤ) ≤ H'.index + 2 := by linarith
  have hle : Nat.card H' ≤ H'.index + 2 := by exact_mod_cast this
  have hfe := even_card_of_neg_one_mem hp H' hneg
  rcases Nat.lt_or_ge (Nat.card H') (H'.index + 2) with hlt | hge
  · omega
  · have : Nat.card H' = H'.index + 2 := le_antisymm hle hge
    rw [this] at hfe
    exact absurd hfe (Nat.not_even_iff_odd.mpr (ho.add_even even_two))

/-- **Good matrices, prime order.** A quadruple whose first sequence is skew (otherwise
unrestricted) and whose other three are symmetric and constant on the orbits of `H`; with
`H' = H ⊔ ⟨-1⟩` of order `f` and index `e`: `f ≤ e` for `e` even, `3 (f - 2) ≤ 4 e` for `e`
odd. -/
theorem IsQuadruple.good_bound (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    {a b c d : ZMod p → ℤ} (hq : IsQuadruple a b c d) (ka : IsSkew a) (sb : IsSymmetric b)
    (sc : IsSymmetric c) (sd : IsSymmetric d) (hb : OrbitConstant H b)
    (hc : OrbitConstant H c) (hd : OrbitConstant H d) :
    let H' := H ⊔ Subgroup.zpowers (-1)
    (Even H'.index → Nat.card H' ≤ H'.index) ∧
      (Odd H'.index → 3 * ((Nat.card H' : ℤ) - 2) ≤ 4 * H'.index) := by
  intro H'
  have hneg : (-1 : (ZMod p)ˣ) ∈ H' := Subgroup.mem_sup_right (Subgroup.mem_zpowers _)
  have hinv : ∀ i, OrbitConstant H' (![b, c, d] i) := by
    intro i
    fin_cases i
    exacts [sb.orbitConstant_sup hb, sc.orbitConstant_sup hc, sd.orbitConstant_sup hd]
  have hsign : ∀ i, IsSign (![b, c, d] i) := by
    intro i; fin_cases i
    exacts [hq.signB, hq.signC, hq.signD]
  have hsum := hq.good_rowSum ka
  have := symmetric_rows_bound hp H' hneg hsign hinv (r := 3) (by
    rw [Fin.sum_univ_three]
    show rowSum b ^ 2 + rowSum c ^ 2 + rowSum d ^ 2 = _
    rw [hsum]; push_cast; ring)
  exact_mod_cast this

/-- **G-matrices, prime order.** Two skew sequences (otherwise unrestricted) and two symmetric
sequences constant on the orbits of `H`; with `H' = H ⊔ ⟨-1⟩` of order `f` and index `e`:
`f ≤ e` for `e` even, `f ≤ 2 e + 2` for `e` odd. -/
theorem IsQuadruple.kkss_bound (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    {a b c d : ZMod p → ℤ} (hq : IsQuadruple a b c d) (ka : IsSkew a) (kb : IsSkew b)
    (sc : IsSymmetric c) (sd : IsSymmetric d) (hc : OrbitConstant H c) (hd : OrbitConstant H d) :
    let H' := H ⊔ Subgroup.zpowers (-1)
    (Even H'.index → Nat.card H' ≤ H'.index) ∧
      (Odd H'.index → Nat.card H' ≤ 2 * H'.index + 2) := by
  intro H'
  have hneg : (-1 : (ZMod p)ˣ) ∈ H' := Subgroup.mem_sup_right (Subgroup.mem_zpowers _)
  have hinv : ∀ i, OrbitConstant H' (![c, d] i) := by
    intro i
    fin_cases i
    exacts [sc.orbitConstant_sup hc, sd.orbitConstant_sup hd]
  have hsign : ∀ i, IsSign (![c, d] i) := by
    intro i; fin_cases i
    exacts [hq.signC, hq.signD]
  have hsum := hq.kkss_rowSum ka kb
  have := symmetric_rows_bound hp H' hneg hsign hinv (r := 2) (by
    simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [hsum]; push_cast; ring)
  refine ⟨this.1, fun ho => ?_⟩
  have h2 := this.2 ho
  push_cast at h2
  have : (Nat.card H' : ℤ) ≤ 2 * H'.index + 2 := by linarith
  exact_mod_cast this

/-- **Best matrices, prime order.** Three skew sequences (otherwise unrestricted) and one
symmetric sequence `d` constant on the orbits of `H`; with `H' = H ⊔ ⟨-1⟩` of order `f` and
index `e`, the row sum of `d` is `d 0 + f m` with `m ≡ e (mod 2)`, `|m| ≤ e` and
`f m² + 2 (d 0) m = 4 e`. Hence `f ≤ e` for `e` even and `f ≤ 4 e + 2` for `e` odd. -/
theorem IsQuadruple.best_bound (hp : p ≠ 2) (H : Subgroup (ZMod p)ˣ)
    {a b c d : ZMod p → ℤ} (hq : IsQuadruple a b c d) (ka : IsSkew a) (kb : IsSkew b)
    (kc : IsSkew c) (sd : IsSymmetric d) (hd : OrbitConstant H d) :
    let H' := H ⊔ Subgroup.zpowers (-1)
    (∃ m : ℤ, rowSum d = d 0 + Nat.card H' * m ∧ Even (m - H'.index) ∧
        -(H'.index : ℤ) ≤ m ∧ m ≤ H'.index ∧
        (Nat.card H' : ℤ) * m ^ 2 + 2 * d 0 * m = 4 * H'.index) ∧
      (Even H'.index → Nat.card H' ≤ H'.index) ∧
      (Odd H'.index → Nat.card H' ≤ 4 * H'.index + 2) := by
  intro H'
  have hneg : (-1 : (ZMod p)ˣ) ∈ H' := Subgroup.mem_sup_right (Subgroup.mem_zpowers _)
  have hd' := sd.orbitConstant_sup hd
  have hsum := hq.best_rowSum ka kb kc
  have hB := symmetric_rows_bound hp H' hneg (b := ![d]) (fun i => by fin_cases i; exact hq.signD)
    (fun i => by fin_cases i; exact hd')
    (r := 1) (by simp only [Fin.sum_univ_one, Matrix.cons_val_zero]; rw [hsum]; push_cast; ring)
  obtain ⟨m, hm, hpar, hlo, hhi⟩ := rowSum_orbitForm_prime H' hq.signD hd'
  have hpe := prime_eq_card_mul_index H'
  have hfpos : (0 : ℤ) < Nat.card H' := by exact_mod_cast Nat.card_pos
  have heq : (Nat.card H' : ℤ) * m ^ 2 + 2 * d 0 * m = 4 * H'.index := by
    have h1 : (d 0 + (Nat.card H' : ℤ) * m) ^ 2 = 4 * p - 3 := by rw [← hm, hsum]
    have hx : d 0 ^ 2 = 1 := by rcases hq.signD 0 with h | h <;> rw [h] <;> norm_num
    rw [hpe] at h1
    have : (Nat.card H' : ℤ) * ((Nat.card H' : ℤ) * m ^ 2 + 2 * d 0 * m) =
        (Nat.card H' : ℤ) * (4 * H'.index) := by nlinarith
    exact mul_left_cancel₀ hfpos.ne' this
  refine ⟨⟨m, hm, hpar, hlo, hhi, heq⟩, hB.1, fun ho => ?_⟩
  have h2 := hB.2 ho
  push_cast at h2
  have : (Nat.card H' : ℤ) ≤ 4 * H'.index + 2 := by linarith
  exact_mod_cast this

end Prime

/-! ### Composite order -/

section Composite

variable {n : ℕ} [NeZero n]

/-- **Goethals–Seidel quadruples, any odd order.** Let `n > 1` be odd and `H` a subgroup of the
units of `ZMod n`, and let `g > 0` divide the size of every orbit of `H` on the nonzero residues
(for example their greatest common divisor), with `n = g E + 1`. If all four sequences of a
quadruple are constant on the orbits of `H`, then `g ≤ E + 1`, that is `g (g - 1) ≤ n - 1`. -/
theorem IsQuadruple.orbitDivisor_le (hn1 : 1 < n) (hodd : Odd n) (H : Subgroup (ZMod n)ˣ)
    {g : ℕ} {E : ℕ} (hg : OrbitDivisor H g) (hgpos : 0 < g) (hn : n = g * E + 1)
    {a b c d : ZMod n → ℤ} (hq : IsQuadruple a b c d) (ha : OrbitConstant H a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : g ≤ E + 1 := by
  have hinv : ∀ i, OrbitConstant H (![a, b, c, d] i) := by
    intro i; fin_cases i <;> assumption
  have hF := hq.toFamily
  have hnZ : (n : ℤ) = (g : ℤ) * (E : ℤ) + 1 := by exact_mod_cast hn
  obtain ⟨he, ho⟩ := hF.orbit_bound H hg hgpos hnZ hinv
  rcases Nat.even_or_odd E with hev | hod
  · rcases he (by exact_mod_cast hev) with hall | h4
    · have hsum := hF.rowSum_sq
      have hsq : ∀ i, rowSum (![a, b, c, d] i) ^ 2 = 1 := fun i => by
        rw [hall i]; rcases hF.sign i 0 with h' | h' <;> rw [h'] <;> norm_num
      simp only [hsq, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
        mul_one] at hsum
      have : (1 : ℤ) < n := by exact_mod_cast hn1
      push_cast at hsum
      linarith
    · push_cast at h4
      have : (g : ℤ) ≤ E + 1 := by linarith
      exact_mod_cast this
  · have h4 := ho (by exact_mod_cast hod)
    push_cast at h4
    have hle : g ≤ E + 2 := by
      have : (g : ℤ) ≤ E + 2 := by linarith
      exact_mod_cast this
    have hge : Even (g * E) := by
      obtain ⟨t, ht⟩ := hodd
      exact ⟨t, by omega⟩
    have hfe : Even g := by
      rcases Nat.even_mul.mp hge with h | h
      · exact h
      · exact absurd h (Nat.not_even_iff_odd.mpr hod)
    rcases Nat.lt_or_ge g (E + 2) with hlt | hge'
    · omega
    · have : g = E + 2 := le_antisymm hle hge'
      rw [this] at hfe
      exact absurd hfe (Nat.not_even_iff_odd.mpr (hod.add_even even_two))

end Composite

/-! ### The board's obstructions at 89 and 107 as special cases -/

/-- Order 89 from `IsQuadruple.skew_card_le`: the only nontrivial odd-order subgroup has order
`11` and index `8`, and `11 > 8 + 1`. -/
theorem no_skew_orbit_quadruple_89' (H : Subgroup (ZMod 89)ˣ) (hH : Odd (Nat.card H))
    (h1 : Nat.card H ≠ 1) {a b c d : ZMod 89 → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hle := hq.skew_card_le (by norm_num) H hH ha hb hc hd
  have hidx : Nat.card H * H.index = 88 := by
    rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
  have hdvd : Nat.card H ∣ 88 := ⟨_, hidx.symm⟩
  have hle88 : Nat.card H ≤ 88 := Nat.le_of_dvd (by norm_num) hdvd
  have hodd : Nat.card H % 2 = 1 := Nat.odd_iff.mp hH
  generalize Nat.card H = f at *
  generalize H.index = e at *
  interval_cases f <;> omega

/-- Order 107 from `IsQuadruple.skew_card_le`: the only nontrivial odd-order subgroup has order
`53` and index `2`. -/
theorem no_skew_orbit_quadruple_107' (H : Subgroup (ZMod 107)ˣ) (hH : Odd (Nat.card H))
    (h1 : Nat.card H ≠ 1) {a b c d : ZMod 107 → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hle := hq.skew_card_le (by norm_num) H hH ha hb hc hd
  have hidx : Nat.card H * H.index = 106 := by
    rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
  have hdvd : Nat.card H ∣ 106 := ⟨_, hidx.symm⟩
  have hle106 : Nat.card H ≤ 106 := Nat.le_of_dvd (by norm_num) hdvd
  have hodd : Nat.card H % 2 = 1 := Nat.odd_iff.mp hH
  generalize Nat.card H = f at *
  generalize H.index = e at *
  interval_cases f <;> omega

/-- Order 101 from the orbit form: with `|H|` odd the index is even, so each orbit-constant row
sum is `±1` modulo `2|H|` (`OrbitForm.two_mul_dvd`), and the board's three-squares check
`sq_mem_of_ten_dvd_sub` finishes as before. -/
theorem no_skew_orbit_quadruple_101' (H : Subgroup (ZMod 101)ˣ) (hH : Odd (Nat.card H))
    (h1 : Nat.card H ≠ 1) {a b c d : ZMod 101 → ℤ} (hq : IsQuadruple a b c d) (ha : IsSkew a)
    (hb : OrbitConstant H b) (hc : OrbitConstant H c) (hd : OrbitConstant H d) : False := by
  have hidx : Nat.card H * H.index = 100 := by
    rw [Subgroup.card_mul_index, Nat.card_eq_fintype_card, ZMod.card_units]
  have hev : Even (H.index : ℤ) := by
    have : Even H.index := by
      rcases Nat.even_mul.mp (⟨50, by omega⟩ : Even (Nat.card H * H.index)) with h | h
      · exact absurd h (Nat.not_even_iff_odd.mpr hH)
      · exact h
    exact_mod_cast this
  have h5 : 5 ∣ Nat.card H := by
    have hodd : Nat.card H % 2 = 1 := Nat.odd_iff.mp hH
    have hdvd : Nat.card H ∣ 100 := ⟨_, hidx.symm⟩
    have hle : Nat.card H ≤ 100 := Nat.le_of_dvd (by norm_num) hdvd
    generalize Nat.card H = f at *
    generalize H.index = e at *
    interval_cases f <;> omega
  have h10 : (10 : ℤ) ∣ 2 * (Nat.card H : ℤ) := by
    obtain ⟨k, hk⟩ := h5
    exact ⟨k, by rw [hk]; push_cast; ring⟩
  have hsum := hq.rowSum_sq
  rw [ha.rowSum_eq_one] at hsum
  have hsum' : rowSum b ^ 2 + rowSum c ^ 2 + rowSum d ^ 2 = 403 := by
    push_cast at hsum
    linarith
  have hB0 : 0 ≤ rowSum b ^ 2 := sq_nonneg _
  have hC0 : 0 ≤ rowSum c ^ 2 := sq_nonneg _
  have hD0 : 0 ≤ rowSum d ^ 2 := sq_nonneg _
  have hB := sq_mem_of_ten_dvd_sub (hq.signB 0)
    (h10.trans ((rowSum_orbitForm_prime H hq.signB hb).two_mul_dvd hev)) (by linarith)
  have hC := sq_mem_of_ten_dvd_sub (hq.signC 0)
    (h10.trans ((rowSum_orbitForm_prime H hq.signC hc).two_mul_dvd hev)) (by linarith)
  have hD := sq_mem_of_ten_dvd_sub (hq.signD 0)
    (h10.trans ((rowSum_orbitForm_prime H hq.signD hd).two_mul_dvd hev)) (by linarith)
  rcases hB with hB | hB | hB | hB <;> rcases hC with hC | hC | hC | hC <;>
    rcases hD with hD | hD | hD | hD <;> omega

end GoethalsSeidel
