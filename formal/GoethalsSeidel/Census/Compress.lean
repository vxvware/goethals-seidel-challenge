/-
Census step (f): compression.

For `m ∣ n` let `π : ZMod n → ZMod m` be reduction and let the `m`-compression of `x` be
`x'_r = Σ_{π j = r} x_j` (`wc_compress.c`: "entry j is the sum over the coset j + km").  Then
`PAF_{x'}(s) = Σ_{π t = s} PAF_x(t)`, so the compressions of a quadruple satisfy the length-`m`
equations `Σ PAF_{x'}(s) = 0` for `s ≠ 0` and `= 4n` at `s = 0`.  Compression maps symmetric rows
to symmetric rows, commutes with multipliers (`compress (x ∘ u) = compress x ∘ π(u)`), and every
unit of `ZMod m` lifts to a unit of `ZMod n`.  Each compressed entry is a sum of `n/m` signs, so
it has the parity of `n/m` and absolute value at most `n/m`.
-/
import GoethalsSeidel.Census.Symmetry

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n m : ℕ} [NeZero n] [NeZero m]

/-- Reduction `ZMod n → ZMod m` for `m ∣ n`. -/
def proj (hmn : m ∣ n) : ZMod n →+* ZMod m := ZMod.castHom hmn (ZMod m)

/-- The fibre of `π` over `r`. -/
def fib (hmn : m ∣ n) (r : ZMod m) : Finset (ZMod n) := univ.filter (fun j => proj hmn j = r)

/-- The `m`-compression of `x`. -/
def compress (hmn : m ∣ n) (x : ZMod n → ℤ) (r : ZMod m) : ℤ := ∑ j ∈ fib hmn r, x j

theorem mem_fib (hmn : m ∣ n) {r : ZMod m} {j : ZMod n} : j ∈ fib hmn r ↔ proj hmn j = r := by
  simp [fib]

/-- `Σ_{π l = π j + s} x_l = Σ_{π t = s} x_{j+t}`. -/
theorem sum_fib_shift (hmn : m ∣ n) (x : ZMod n → ℤ) (j : ZMod n) (s : ZMod m) :
    ∑ l ∈ fib hmn (proj hmn j + s), x l = ∑ t ∈ fib hmn s, x (j + t) := by
  symm
  refine Finset.sum_nbij' (fun t => j + t) (fun l => l - j) ?_ ?_ ?_ ?_ ?_
  · intro t ht
    simp only [mem_coe, mem_fib] at ht ⊢
    rw [map_add, ht]
  · intro l hl
    simp only [mem_coe, mem_fib] at hl ⊢
    rw [map_sub, hl]; ring
  · intro t _; simp
  · intro l _; simp
  · intro t _; rfl

/-- The compression lemma: `PAF_{x'}(s) = Σ_{π t = s} PAF_x(t)`. -/
theorem paf_compress (hmn : m ∣ n) (x : ZMod n → ℤ) (s : ZMod m) :
    paf (compress hmn x) s = ∑ t ∈ fib hmn s, paf x t := by
  unfold paf
  have h1 : ∑ r, compress hmn x r * compress hmn x (r + s) =
      ∑ j, x j * compress hmn x (proj hmn j + s) := by
    simp only [compress, Finset.sum_mul]
    rw [← Finset.sum_fiberwise (univ : Finset (ZMod n)) (proj hmn)
      (fun j => x j * ∑ l ∈ fib hmn (proj hmn j + s), x l)]
    refine Finset.sum_congr rfl (fun r _ => ?_)
    refine Finset.sum_congr rfl (fun j hj => ?_)
    have : proj hmn j = r := (mem_fib hmn).mp hj
    rw [this, Finset.mul_sum]
  rw [h1]
  simp only [compress, sum_fib_shift, Finset.mul_sum]
  rw [Finset.sum_comm]

theorem proj_ne_of_zero (hmn : m ∣ n) {s : ZMod m} (hs : s ≠ 0) {t : ZMod n}
    (ht : t ∈ fib hmn s) : t ≠ 0 := by
  rintro rfl
  rw [mem_fib, map_zero] at ht
  exact hs ht.symm

/-- The compressed equations at `s ≠ 0`. -/
theorem compress_paf_zero (hmn : m ∣ n) {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    {s : ZMod m} (hs : s ≠ 0) :
    paf (compress hmn a) s + paf (compress hmn b) s + paf (compress hmn c) s +
      paf (compress hmn d) s = 0 := by
  rw [paf_compress, paf_compress, paf_compress, paf_compress, ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_eq_zero (fun t ht => h.paf_zero t (proj_ne_of_zero hmn hs ht))

/-- The compressed equation at `s = 0`: the squared compressed entries add up to `4n`. -/
theorem compress_paf_zero_shift (hmn : m ∣ n) {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) :
    paf (compress hmn a) 0 + paf (compress hmn b) 0 + paf (compress hmn c) 0 +
      paf (compress hmn d) 0 = 4 * n := by
  rw [paf_compress, paf_compress, paf_compress, paf_compress, ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  have h0 : (0 : ZMod n) ∈ fib hmn 0 := by rw [mem_fib, map_zero]
  rw [← Finset.add_sum_erase _ _ h0]
  rw [Finset.sum_eq_zero (fun t ht => h.paf_zero t (Finset.ne_of_mem_erase ht))]
  rw [paf_zero_shift h.signA, paf_zero_shift h.signB, paf_zero_shift h.signC,
    paf_zero_shift h.signD]
  ring

/-- Compression maps symmetric rows to symmetric rows. -/
theorem compress_symm (hmn : m ∣ n) {x : ZMod n → ℤ} (hx : IsSymmetric x) :
    IsSymmetric (compress hmn x) := by
  intro r
  unfold compress
  refine Finset.sum_nbij' (fun j => -j) (fun j => -j) ?_ ?_ ?_ ?_ ?_
  · intro j hj
    simp only [mem_coe, mem_fib] at hj ⊢
    rw [map_neg, hj, neg_neg]
  · intro j hj
    simp only [mem_coe, mem_fib] at hj ⊢
    rw [map_neg, hj]
  · intro j _; simp
  · intro j _; simp
  · intro j _; exact (hx j).symm

/-- Compression commutes with multipliers: `compress (x ∘ u) r = compress x (π(u) r)`. -/
theorem compress_mulRow (hmn : m ∣ n) (u : (ZMod n)ˣ) (x : ZMod n → ℤ) (r : ZMod m) :
    compress hmn (mulRow u x) r = compress hmn x (proj hmn (u : ZMod n) * r) := by
  unfold compress mulRow
  refine Finset.sum_nbij' (fun j => (u : ZMod n) * j) (fun j => ((u⁻¹ : (ZMod n)ˣ) : ZMod n) * j)
    ?_ ?_ ?_ ?_ ?_
  · intro j hj
    simp only [mem_coe, mem_fib] at hj ⊢
    rw [map_mul, hj]
  · intro j hj
    simp only [mem_coe, mem_fib] at hj ⊢
    rw [map_mul, hj, ← mul_assoc, ← map_mul]
    simp
  · intro j _; simp
  · intro j _; simp
  · intro j _; rfl

/-- Every unit of `ZMod m` lifts to a unit of `ZMod n` (`wc_compress.c`). -/
theorem units_lift (hmn : m ∣ n) (v : (ZMod m)ˣ) :
    ∃ u : (ZMod n)ˣ, ((u : ZMod n).cast : ZMod m) = (v : ZMod m) := by
  obtain ⟨u, hu⟩ := ZMod.unitsMap_surjective hmn v
  exact ⟨u, by rw [← hu]; rfl⟩

/-- All fibres have the same size. -/
theorem card_fib (hmn : m ∣ n) (r : ZMod m) : (fib hmn r).card = (fib hmn 0).card := by
  obtain ⟨j0, hj0⟩ : ∃ j0 : ZMod n, proj hmn j0 = r := by
    refine ⟨(r.val : ZMod n), ?_⟩
    simp [proj, ZMod.castHom_apply, ZMod.cast_natCast hmn]
  refine Finset.card_nbij' (fun j => j - j0) (fun j => j + j0) ?_ ?_ ?_ ?_
  · intro j hj
    simp only [mem_coe, mem_fib] at hj ⊢
    rw [map_sub, hj, hj0, sub_self]
  · intro j hj
    simp only [mem_coe, mem_fib] at hj ⊢
    rw [map_add, hj, hj0, zero_add]
  · intro j _; simp
  · intro j _; simp

/-- Each fibre has `n / m` elements. -/
theorem card_fib_mul (hmn : m ∣ n) (r : ZMod m) : (fib hmn r).card * m = n := by
  have h := Finset.card_eq_sum_card_fiberwise (f := proj hmn) (s := univ) (t := univ)
    (fun _ _ => Finset.mem_univ _)
  simp only [Finset.card_univ, ZMod.card] at h
  have : ∀ r' ∈ (univ : Finset (ZMod m)), (univ.filter (fun j => proj hmn j = r')).card =
      (fib hmn r).card := by
    intro r' _
    have := card_fib hmn r'
    have h2 := card_fib hmn r
    simp only [fib] at this h2 ⊢
    rw [this, h2]
  rw [Finset.sum_congr rfl this] at h
  simp only [Finset.sum_const, Finset.card_univ, ZMod.card, smul_eq_mul] at h
  linarith

/-- A compressed entry of a `±1` row has the parity of the fibre size and absolute value at
most the fibre size (`wc_compress.c`: "odd in [-D, D]"). -/
theorem compress_bound (hmn : m ∣ n) {x : ZMod n → ℤ} (hx : IsSign x) (r : ZMod m) :
    |compress hmn x r| ≤ (fib hmn r).card ∧
      compress hmn x r ≡ (fib hmn r).card [ZMOD 2] := by
  unfold compress
  constructor
  · calc |∑ j ∈ fib hmn r, x j| ≤ ∑ j ∈ fib hmn r, |x j| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j ∈ fib hmn r, (1 : ℤ) := Finset.sum_congr rfl (fun j _ => by
          rcases hx j with h | h <;> simp [h])
      _ = (fib hmn r).card := by simp
  · rw [Finset.sum_congr rfl (fun j _ => eq_one_sub_two_bit hx j), Finset.sum_sub_distrib,
      ← Finset.mul_sum, Int.modEq_iff_dvd]
    simp only [Finset.sum_const, smul_eq_mul, mul_one]
    exact ⟨∑ i ∈ fib hmn r, bit x i, by ring⟩

end GoethalsSeidel.Census
