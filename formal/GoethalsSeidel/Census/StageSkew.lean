/-
Census step (b), best matrices: the two unknown skew rows `B`, `C` of `best_hkt2.c`.

With `p = b c` (symmetric, `p_0 = 1`), `σ_j = e^p_j = [b_j ≠ c_j]`, the excluded set
`E_k = {0, -k, -h}` (`E_[k]` in `stage_init` is its complement) and the kept set
`F_k = {j ∉ E_k | p_j = p_{j+k}}` (`F = ~(S ^ rotr(S, k)) & E_[k]` in `s2_rows_ref`):

* `u_b(k) + u_c(k) = 2 + u_p(k) - σ_k + Σ_{j ∈ F_k} [b_j ≠ b_{j+k}]`          (exact),
* `2 (u_b(k) + u_c(k)) = (n + 1) - |F_k| + 2 Σ_{j ∈ F_k} [b_j ≠ b_{j+k}]`       (exact),
* `|F_k|` and `Σ_{F_k} [b_j ≠ b_{j+k}]` are even,
* stage 1: `u_b(k) + u_c(k) ≡ σ_k + σ_h (mod 2)`,
* with the quadruple equation: `σ_k + σ_h ≡ 1 + u_a(k) + u_d(k) (mod 2)` (the right side `r` of
  `M1 t = r`) and `2 Σ_{F_k} [b_j ≠ b_{j+k}] = (n - 1) + |F_k| - 2 (u_a(k) + u_d(k))`, i.e.
  `v = K + H - (u_a + u_d)` with `K = |F_k|/2`, `H = (n-1)/2` as in the stage-2 self-check.
-/
import GoethalsSeidel.Census.Stage

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-- The excluded lags `{0, -k, -h}` of a skew row at lag `k` (complement of `E_[k]`). -/
def exclSkew (k h : ZMod n) : Finset (ZMod n) := insert 0 (insert (-k) {-h})

/-- The kept set `F_k` of the best-matrix stage 2. -/
def keptSkew (b c : ZMod n → ℤ) (k h : ZMod n) : Finset (ZMod n) :=
  ((univ : Finset (ZMod n)) \ exclSkew k h).filter (fun j => b j * c j = b (j + k) * c (j + k))

section
variable (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
include hn hk0 hk

theorem h_ne_zero : h ≠ 0 := by
  rintro rfl; apply hk0; rw [← hk]; ring

theorem zero_ne_neg_k : (0 : ZMod n) ≠ -k := by
  intro e; apply hk0; rw [← neg_neg k, ← e, neg_zero]

theorem zero_ne_neg_h : (0 : ZMod n) ≠ -h := by
  intro e; apply h_ne_zero hn hk0 hk; rw [← neg_neg h, ← e, neg_zero]

theorem neg_k_ne_neg_h : -k ≠ -h := by
  intro e
  apply h_ne_zero hn hk0 hk
  have : k = h := neg_injective e
  rw [← hk] at this
  linear_combination this

theorem sum_exclSkew (f : ZMod n → ℤ) : ∑ j ∈ exclSkew k h, f j = f 0 + f (-k) + f (-h) := by
  unfold exclSkew
  rw [Finset.sum_insert, Finset.sum_insert, Finset.sum_singleton, add_assoc]
  · simpa using neg_k_ne_neg_h hn hk0 hk
  · simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
    exact ⟨zero_ne_neg_k hn hk0 hk, zero_ne_neg_h hn hk0 hk⟩

theorem sum_split (f : ZMod n → ℤ) :
    ∑ j, f j = f 0 + f (-k) + f (-h) + ∑ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h, f j := by
  rw [← sum_exclSkew hn hk0 hk f, add_comm,
    Finset.sum_sdiff (Finset.subset_univ (exclSkew k h))]

/-- `ι` maps the complement of `E_k` to itself. -/
theorem iota_mem_rest {j : ZMod n} (hj : j ∈ (univ : Finset (ZMod n)) \ exclSkew k h) :
    -j - k ∈ (univ : Finset (ZMod n)) \ exclSkew k h := by
  simp only [exclSkew, Finset.mem_sdiff, Finset.mem_univ, Finset.mem_insert,
    Finset.mem_singleton, true_and, not_or] at hj ⊢
  obtain ⟨h1, h2, h3⟩ := hj
  refine ⟨?_, ?_, ?_⟩
  · intro e; apply h2; linear_combination -e
  · intro e; apply h1; linear_combination -e
  · intro e; apply h3; rw [← hk] at e; linear_combination -e

theorem iota_ne_rest {j : ZMod n} (hj : j ∈ (univ : Finset (ZMod n)) \ exclSkew k h) :
    -j - k ≠ j := by
  intro e
  simp only [exclSkew, Finset.mem_sdiff, Finset.mem_univ, Finset.mem_insert,
    Finset.mem_singleton, true_and, not_or] at hj
  exact hj.2.2 ((iota_eq_self_iff hn hk j).mp e)

end

theorem skew_neg {x : ZMod n → ℤ} (hx : IsSkew x) {j : ZMod n} (hj : j ≠ 0) : x (-j) = -x j :=
  hx.2 j hj

/-- The product of two skew rows is symmetric. -/
theorem isSymmetric_mul_skew {b c : ZMod n → ℤ} (hb : IsSkew b) (hc : IsSkew c) :
    IsSymmetric (fun i => b i * c i) := by
  intro j
  by_cases hj : j = 0
  · subst hj; simp
  · simp only; rw [hb.2 j hj, hc.2 j hj]; ring

/-- A skew row has exactly two sign changes on `E_k`. -/
theorem changes_skew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {x : ZMod n → ℤ} (hx : IsSign x) (hsk : IsSkew x) :
    changes x k = 2 + ∑ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h, chg x k j := by
  rw [changes_eq_sum_chg, sum_split hn hk0 hk]
  have hk' : -k ≠ 0 := fun e => hk0 (neg_eq_zero.mp e)
  have hh : h ≠ 0 := h_ne_zero hn hk0 hk
  have e1 : chg x k 0 + chg x k (-k) = 1 := by
    unfold chg
    rw [zero_add, neg_add_cancel, hsk.1, hsk.2 k hk0]
    rcases hx k with h1 | h1 <;> rw [h1] <;> norm_num
  have e2 : chg x k (-h) = 1 := by
    unfold chg
    rw [show -h + k = h by rw [← hk]; ring, hsk.2 h hh]
    rcases hx h with h1 | h1 <;> rw [h1] <;> norm_num
  rw [e1, e2]; ring

/-- The symmetric product row has `2 σ_k` sign changes on `E_k`. -/
theorem changes_prod_skew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hb : IsSkew b) (hc : IsSkew c) :
    changes (fun i => b i * c i) k = 2 * bit (fun i => b i * c i) k +
      ∑ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h, chg (fun i => b i * c i) k j := by
  rw [changes_eq_sum_chg, sum_split hn hk0 hk, chg_fix_symm (isSymmetric_mul_skew hb hc) hk]
  have hp0 : b 0 * c 0 = 1 := by rw [hb.1, hc.1]; norm_num
  have e1 : chg (fun i => b i * c i) k 0 = bit (fun i => b i * c i) k := by
    unfold chg bit; simp only [zero_add]; rw [hp0]
    by_cases h1 : b k * c k = 1 <;> simp [h1, eq_comm]
  have e2 : chg (fun i => b i * c i) k (-k) = bit (fun i => b i * c i) k := by
    have hs := isSymmetric_mul_skew hb hc k
    simp only at hs
    unfold chg bit; simp only [neg_add_cancel]; rw [hp0, hs]
  rw [e1, e2]; ring

/-- Stage 2 of `best_hkt2.c`, exact form. -/
theorem stage2_skew_exact (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hbs : IsSign b) (hcs : IsSign c) (hb : IsSkew b) (hc : IsSkew c) :
    uval b k + uval c k = 2 + uval (fun i => b i * c i) k - bit (fun i => b i * c i) k +
      ∑ j ∈ keptSkew b c k h, chg b k j := by
  have e2 : changes b k + changes c k = 4 + changes (fun i => b i * c i) k
      - 2 * bit (fun i => b i * c i) k + 2 * ∑ j ∈ keptSkew b c k h, chg b k j := by
    rw [changes_skew hn hk0 hk hbs hb, changes_skew hn hk0 hk hcs hc,
      changes_prod_skew hn hk0 hk hb hc]
    have : ∑ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h, (chg b k j + chg c k j) =
        ∑ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h, chg (fun i => b i * c i) k j +
          2 * ∑ j ∈ keptSkew b c k h, chg b k j := by
      rw [Finset.sum_congr rfl (fun j _ => chg_add_chg hbs hcs k j), Finset.sum_add_distrib,
        ← Finset.mul_sum, keptSkew, Finset.sum_filter]
    rw [Finset.sum_add_distrib] at this
    linarith
  rw [changes_eq b hbs, changes_eq c hcs, changes_eq _ (isSign_mul hbs hcs)] at e2
  linarith

/-- `|F_k| + 2 u_p(k) - 2 σ_k = n - 3`. -/
theorem card_keptSkew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hbs : IsSign b) (hcs : IsSign c) (hb : IsSkew b) (hc : IsSkew c) :
    ((keptSkew b c k h).card : ℤ) + 2 * uval (fun i => b i * c i) k
      - 2 * bit (fun i => b i * c i) k = n - 3 := by
  have hp := changes_prod_skew hn hk0 hk hb hc
  rw [changes_eq _ (isSign_mul hbs hcs)] at hp
  have hrest : ∑ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h,
      ((if b j * c j = b (j + k) * c (j + k) then (1 : ℤ) else 0) +
        chg (fun i => b i * c i) k j) = n - 3 := by
    have : ∀ j ∈ (univ : Finset (ZMod n)) \ exclSkew k h,
        (if b j * c j = b (j + k) * c (j + k) then (1 : ℤ) else 0) +
          chg (fun i => b i * c i) k j = 1 := by
      intro j _; unfold chg; split_ifs <;> simp_all
    rw [Finset.sum_congr rfl this]
    have hcard : (exclSkew k h).card = 3 := by
      unfold exclSkew
      rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem, Finset.card_singleton]
      · simpa using neg_k_ne_neg_h hn hk0 hk
      · simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
        exact ⟨zero_ne_neg_k hn hk0 hk, zero_ne_neg_h hn hk0 hk⟩
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
    rw [Finset.card_sdiff_of_subset (Finset.subset_univ _), hcard, Finset.card_univ, ZMod.card]
    have : 3 ≤ n := by
      obtain ⟨m, hm⟩ := hn
      rcases Nat.lt_or_ge n 3 with hlt | hge
      · exfalso
        have : n = 1 := by omega
        subst this
        exact hk0 (Subsingleton.elim _ _)
      · exact hge
    push_cast [this]
    ring
  rw [Finset.sum_add_distrib, ← Finset.sum_filter] at hrest
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one] at hrest
  rw [← keptSkew] at hrest
  linarith

theorem iota_mem_keptSkew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hb : IsSkew b) (hc : IsSkew c) {j : ZMod n}
    (hj : j ∈ keptSkew b c k h) : -j - k ∈ keptSkew b c k h := by
  have hp := isSymmetric_mul_skew hb hc
  simp only [keptSkew, Finset.mem_filter] at hj ⊢
  refine ⟨iota_mem_rest hn hk0 hk hj.1, ?_⟩
  have e1 : -j - k = -(j + k) := by ring
  have e2 : -j - k + k = -j := by ring
  rw [e2, e1]
  have := hp (j + k)
  have h' := hp j
  simp only at this h'
  rw [this, h']
  exact hj.2.symm

/-- `|F_k|` is even. -/
theorem even_card_keptSkew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hb : IsSkew b) (hc : IsSkew c) :
    (2 : ℤ) ∣ ((keptSkew b c k h).card : ℤ) := by
  have := two_mul_dvd_sum_invol (keptSkew b c k h) (fun j => -j - k) (fun _ => (1 : ℤ)) 1
    (fun j hj => iota_mem_keptSkew hn hk0 hk hb hc hj) (by intro j _; ring)
    (by
      intro j hj
      simp only [keptSkew, Finset.mem_filter] at hj
      exact iota_ne_rest hn hk0 hk hj.1)
    (by intro j _; rfl) (by intro j _; simp)
  simpa using this

/-- `Σ_{F_k} [b_j ≠ b_{j+k}]` is even (stage-2 integrality, `v & 1 = 0` in the self-check). -/
theorem even_sum_chg_keptSkew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hb : IsSkew b) (hc : IsSkew c) :
    (2 : ℤ) ∣ ∑ j ∈ keptSkew b c k h, chg b k j := by
  have := two_mul_dvd_sum_invol (keptSkew b c k h) (fun j => -j - k) (chg b k) 1
    (fun j hj => iota_mem_keptSkew hn hk0 hk hb hc hj) (by intro j _; ring)
    (by
      intro j hj
      simp only [keptSkew, Finset.mem_filter] at hj
      exact iota_ne_rest hn hk0 hk hj.1)
    (by
      intro j hj
      simp only [keptSkew, Finset.mem_filter, exclSkew, Finset.mem_sdiff, Finset.mem_univ,
        Finset.mem_insert, Finset.mem_singleton, true_and, not_or] at hj
      obtain ⟨⟨h0, hmk, _⟩, _⟩ := hj
      have hjk : j + k ≠ 0 := by intro e; apply hmk; linear_combination e
      unfold chg
      rw [show -j - k + k = -j by ring, show -j - k = -(j + k) by ring, hb.2 _ hjk, hb.2 _ h0]
      by_cases h1 : b (j + k) = b j
      · simp [h1]
      · have h2 : ¬ b j = b (j + k) := fun e => h1 e.symm
        simp [h1, h2])
    (by intro j _; simp)
  simpa using this

/-- Stage 1 of `best_hkt2.c` (`M1 t = r`): `u_b(k) + u_c(k) ≡ σ_k + σ_h (mod 2)` where
`σ = e^{bc}` is the xor pattern of `b` and `c` and `2 h = k`. -/
theorem stage1_skew (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
    {b c : ZMod n → ℤ} (hbs : IsSign b) (hcs : IsSign c) (hb : IsSkew b) (hc : IsSkew c) :
    uval b k + uval c k ≡ bit (fun i => b i * c i) k + bit (fun i => b i * c i) h [ZMOD 2] := by
  have hex := stage2_skew_exact hn hk0 hk hbs hcs hb hc
  obtain ⟨m, hm⟩ := even_sum_chg_keptSkew hn hk0 hk hb hc
  have hup := uval_symm_modEq hn (isSymmetric_mul_skew hb hc) hk
  have hp0 : bit (fun i => b i * c i) 0 = 0 := by
    unfold bit; simp [hb.1, hc.1]
  rw [hp0, zero_add] at hup
  obtain ⟨r, hr⟩ := (Int.modEq_iff_dvd.mp hup.symm)
  rw [hex, hm, Int.modEq_iff_dvd]
  exact ⟨-1 - m + bit (fun i => b i * c i) k - r, by linear_combination -hr⟩

/-- Stage 1 for a best quadruple: `σ_k + σ_h ≡ 1 + u_a(k) + u_d(k) (mod 2)`, the right side
`r_k = 1 + uA_k + uD_k` of `M1 t = r`. -/
theorem stage1_best (hn : Odd n) {a b c d : ZMod n → ℤ} (hq : IsQuadruple a b c d)
    (hb : IsSkew b) (hc : IsSkew c) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k) :
    bit (fun i => b i * c i) k + bit (fun i => b i * c i) h ≡ 1 + uval a k + uval d k
      [ZMOD 2] := by
  have hsum := (isQuadruple_iff_uval a b c d hq.signA hq.signB hq.signC hq.signD).mp hq k hk0
  have h1 := (stage1_skew hn hk0 hk hq.signB hq.signC hb hc).symm
  refine h1.trans ?_
  obtain ⟨m, hm⟩ := hn
  rw [Int.modEq_iff_dvd]
  refine ⟨-(m : ℤ) + uval a k + uval d k, ?_⟩
  have : (n : ℤ) = 2 * m + 1 := by exact_mod_cast hm
  linear_combination -hsum - this

/-- Stage 2 for a best quadruple (the stage-2 self-check of `stage_init`):
`2 Σ_{F_k} [b_j ≠ b_{j+k}] = (n - 1) + |F_k| - 2 (u_a(k) + u_d(k))`, i.e. the code's
`v = K + H - (uA + uD)` equals `Σ_{F_k} [b_j ≠ b_{j+k}]`, which is even. -/
theorem stage2_best (hn : Odd n) {a b c d : ZMod n → ℤ} (hq : IsQuadruple a b c d)
    (hb : IsSkew b) (hc : IsSkew c) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k) :
    2 * ∑ j ∈ keptSkew b c k h, chg b k j =
      ((n : ℤ) - 1) + (keptSkew b c k h).card - 2 * (uval a k + uval d k) := by
  have hsum := (isQuadruple_iff_uval a b c d hq.signA hq.signB hq.signC hq.signD).mp hq k hk0
  have hex := stage2_skew_exact hn hk0 hk hq.signB hq.signC hb hc
  have hcard := card_keptSkew hn hk0 hk hq.signB hq.signC hb hc
  linarith

end GoethalsSeidel.Census
