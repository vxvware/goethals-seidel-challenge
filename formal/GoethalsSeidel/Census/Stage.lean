/-
Census step (b): the two linear stages of the HKT solve, as exact integer identities.

Throughout `n` is odd, `k` is a lag and `h` is the half of `k` (`2 h = k`), which exists and is
unique because `2` is a unit modulo `n`.  The involution `ι j = -j - k` pairs the terms of
`PAF(k)`; its only fixed point is `-h`.

* Symmetric rows (`Williamson`, and the two symmetric rows `C`, `D` of the G-matrix and good-matrix
  solves).  `u_x(k) ≡ e_0 + e_h (mod 2)`.  For a pair `c, d` with product `q = c d` and
  `F = {j ≠ -h | q_j = q_{j+k}}`:  `u_c(k) + u_d(k) = u_q(k) + Σ_{j ∈ F} [c_j ≠ c_{j+k}]`.
* Skew rows (`best`: the two solved rows `B`, `C`).  With `p = b c`, `σ_j = [b_j ≠ c_j]`, the
  excluded set `{0, -k, -h}` and `F = {j ∉ {0,-k,-h} | p_j = p_{j+k}}`:
  `u_b(k) + u_c(k) = 2 + u_p(k) - σ_k + Σ_{j ∈ F} [b_j ≠ b_{j+k}]`,
  hence `u_b(k) + u_c(k) ≡ σ_k + σ_h (mod 2)` (stage 1) and the stage-2 identity.
-/
import GoethalsSeidel.Census.Basic

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-! ### Two general facts -/

/-- A sum over a fixed-point-free involution of an invariant function whose values are all
multiples of `m` is a multiple of `2 m`. -/
theorem two_mul_dvd_sum_invol {α : Type*} (s : Finset α) (ι : α → α) (g : α → ℤ) (m : ℕ)
    (hmem : ∀ x ∈ s, ι x ∈ s) (hinv : ∀ x ∈ s, ι (ι x) = x) (hne : ∀ x ∈ s, ι x ≠ x)
    (hg : ∀ x ∈ s, g (ι x) = g x) (hdvd : ∀ x ∈ s, (m : ℤ) ∣ g x) :
    (2 * m : ℤ) ∣ ∑ x ∈ s, g x := by
  have key : ((∑ x ∈ s, g x : ℤ) : ZMod (2 * m)) = 0 := by
    push_cast
    refine Finset.sum_involution (fun x _ => ι x) ?_ ?_ hmem ?_
    · intro x hx
      rw [hg x hx]
      obtain ⟨c, hc⟩ := hdvd x hx
      rw [hc]
      have : ((m * c : ℤ) : ZMod (2 * m)) + ((m * c : ℤ) : ZMod (2 * m))
          = ((2 * m : ℕ) : ZMod (2 * m)) * (c : ZMod (2 * m)) := by
        push_cast; ring
      rw [this, ZMod.natCast_self, zero_mul]
    · intro x hx _; exact hne x hx
    · intro x hx; exact hinv x hx
  have := (ZMod.intCast_zmod_eq_zero_iff_dvd _ (2 * m)).mp key
  exact_mod_cast this

/-- For odd `n`, `2` is cancellable in `ZMod n`. -/
theorem two_mul_cancel (hn : Odd n) {a b : ZMod n} (h : 2 * a = 2 * b) : a = b := by
  have hu : IsUnit (2 : ZMod n) := by
    have : Nat.Coprime 2 n := Nat.coprime_two_left.mpr hn
    have := (ZMod.unitOfCoprime 2 this).isUnit
    simpa using this
  exact hu.mul_left_cancel h

theorem neg_eq_self_iff (hn : Odd n) (j : ZMod n) : -j = j ↔ j = 0 := by
  constructor
  · intro h
    apply two_mul_cancel hn
    rw [mul_zero, two_mul]
    nth_rewrite 1 [← h]
    exact neg_add_cancel j
  · rintro rfl; simp

/-- The involution `ι j = -j - k` has the single fixed point `-h`. -/
theorem iota_eq_self_iff (hn : Odd n) {k h : ZMod n} (hk : 2 * h = k) (j : ZMod n) :
    -j - k = j ↔ j = -h := by
  constructor
  · intro e
    apply two_mul_cancel hn
    rw [← hk] at e
    linear_combination -e
  · rintro rfl
    rw [← hk]; ring

theorem bit_neg_of_symm {x : ZMod n → ℤ} (hsym : IsSymmetric x) (j : ZMod n) :
    bit x (-j) = bit x j := by
  unfold bit; rw [hsym j]

/-- The sign-change indicator at `j` for lag `k`. -/
def chg (x : ZMod n → ℤ) (k j : ZMod n) : ℤ := if x j = x (j + k) then 0 else 1

theorem changes_eq_sum_chg (x : ZMod n → ℤ) (k : ZMod n) : changes x k = ∑ j, chg x k j := rfl

theorem chg_zero_or_one (x : ZMod n → ℤ) (k j : ZMod n) : chg x k j = 0 ∨ chg x k j = 1 := by
  unfold chg; split_ifs <;> simp

/-! ### Symmetric rows -/

/-- For a symmetric row, `Σ_j e_j ≡ e_0 (mod 2)`. -/
theorem sum_bit_symm (hn : Odd n) {x : ZMod n → ℤ} (hsym : IsSymmetric x) :
    ∃ m : ℤ, ∑ j, bit x j = bit x 0 + 2 * m := by
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n))]
  have := two_mul_dvd_sum_invol ((univ : Finset (ZMod n)).erase 0) (fun j => -j) (bit x) 1
    (by intro j hj; simpa using hj) (by intro j _; simp)
    (by intro j hj e; exact (Finset.ne_of_mem_erase hj) ((neg_eq_self_iff hn j).mp e))
    (by intro j _; exact bit_neg_of_symm hsym j) (by intro j _; simp)
  obtain ⟨m, hm⟩ := this
  exact ⟨m, by rw [hm]; ring⟩

/-- For a symmetric row, `Σ_j e_j e_{j+k} ≡ e_h (mod 2)` where `2 h = k`. -/
theorem sum_bit_mul_symm (hn : Odd n) {x : ZMod n → ℤ} (hsym : IsSymmetric x)
    {k h : ZMod n} (hk : 2 * h = k) :
    ∃ m : ℤ, ∑ j, bit x j * bit x (j + k) = bit x h + 2 * m := by
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (-h))]
  have hfix : bit x (-h) * bit x (-h + k) = bit x h := by
    have : -h + k = h := by rw [← hk]; ring
    rw [this, bit_neg_of_symm hsym, bit_mul_self]
  have := two_mul_dvd_sum_invol ((univ : Finset (ZMod n)).erase (-h)) (fun j => -j - k)
    (fun j => bit x j * bit x (j + k)) 1
    (by
      intro j hj
      refine Finset.mem_erase.mpr ⟨?_, Finset.mem_univ _⟩
      intro e
      apply Finset.ne_of_mem_erase hj
      rw [← hk] at e
      linear_combination -e)
    (by intro j _; ring)
    (by intro j hj e; exact Finset.ne_of_mem_erase hj ((iota_eq_self_iff hn hk j).mp e))
    (by
      intro j _
      have e1 : -j - k = -(j + k) := by ring
      have e2 : -j - k + k = -j := by ring
      rw [e2, e1, bit_neg_of_symm hsym, bit_neg_of_symm hsym, mul_comm])
    (by intro j _; simp)
  obtain ⟨m, hm⟩ := this
  exact ⟨m, by rw [hfix, hm]; ring⟩

theorem uval_eq_sum_sub (x : ZMod n → ℤ) (k : ZMod n) :
    uval x k = ∑ j, bit x j - ∑ j, bit x j * bit x (j + k) := by
  unfold uval
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl (fun j _ => by ring)

/-- Stage 1, symmetric row: `u_x(k) ≡ e_0 + e_h (mod 2)` where `2 h = k`. -/
theorem uval_symm_modEq (hn : Odd n) {x : ZMod n → ℤ} (hsym : IsSymmetric x)
    {k h : ZMod n} (hk : 2 * h = k) :
    uval x k ≡ bit x 0 + bit x h [ZMOD 2] := by
  obtain ⟨m1, h1⟩ := sum_bit_symm hn hsym
  obtain ⟨m2, h2⟩ := sum_bit_mul_symm hn hsym hk
  rw [uval_eq_sum_sub, h1, h2, Int.modEq_iff_dvd]
  exact ⟨bit x h - m1 + m2, by ring⟩

/-- `bit (c d) ≡ bit c + bit d (mod 2)`: the sign bit of a product is the xor of the bits. -/
theorem bit_mul_modEq {c d : ZMod n → ℤ} (hc : IsSign c) (hd : IsSign d) (j : ZMod n) :
    bit (fun i => c i * d i) j ≡ bit c j + bit d j [ZMOD 2] := by
  unfold bit
  rcases hc j with h1 | h1 <;> rcases hd j with h2 | h2 <;> simp [h1, h2] <;> decide

/-- Stage 1 of the Williamson / G-matrix solve (`hkt_build`, `M1ROW`): for symmetric rows
`c, d`, `u_c(k) + u_d(k) ≡ t_0 + t_h (mod 2)` with `t_j = e^c_j + e^d_j` (the xor pattern
`c_j ≠ d_j`) and `2 h = k`.  With the normalization `c_0 = d_0 = 1` the code uses, `t_0 = 0`, so
the stage-1 matrix is the permutation `k ↦ h` (folded into `1..H`). -/
theorem stage1_symm (hn : Odd n) {c d : ZMod n → ℤ} (hcs : IsSymmetric c)
    (hds : IsSymmetric d) {k h : ZMod n} (hk : 2 * h = k) :
    uval c k + uval d k ≡ (bit c 0 + bit d 0) + (bit c h + bit d h) [ZMOD 2] := by
  have h1 := uval_symm_modEq hn hcs hk
  have h2 := uval_symm_modEq hn hds hk
  have := Int.ModEq.add h1 h2
  refine this.trans ?_
  rw [Int.modEq_iff_dvd]; exact ⟨0, by ring⟩

/-- Stage 1 for a Williamson quadruple (or a G-matrix / good-matrix quadruple with `c`, `d`
symmetric): the pattern `t` is determined by `a` and `b` through
`t_0 + t_h ≡ 1 + u_a(k) + u_b(k) (mod 2)` (here `n` odd, so `n ≡ 1`). -/
theorem stage1_symm_quadruple (hn : Odd n) {a b c d : ZMod n → ℤ} (hq : IsQuadruple a b c d)
    (hcs : IsSymmetric c) (hds : IsSymmetric d) {k h : ZMod n} (hk0 : k ≠ 0)
    (hk : 2 * h = k) :
    (bit c 0 + bit d 0) + (bit c h + bit d h) ≡ 1 + uval a k + uval b k [ZMOD 2] := by
  have hsum := (isQuadruple_iff_uval a b c d hq.signA hq.signB hq.signC hq.signD).mp hq k hk0
  have h1 := (stage1_symm hn hcs hds hk).symm
  refine h1.trans ?_
  obtain ⟨m, hm⟩ := hn
  rw [Int.modEq_iff_dvd]
  refine ⟨-(m : ℤ) + uval a k + uval b k, ?_⟩
  have : (n : ℤ) = 2 * m + 1 := by exact_mod_cast hm
  linear_combination -hsum - this

/-! ### The per-term identity behind stage 2 -/

/-- For `±1` rows `x, y` with product `p`, at every `j`:
`[x≠] + [y≠] = [p≠] + 2 [p_j = p_{j+k}] [x≠]`. -/
theorem chg_add_chg {x y : ZMod n → ℤ} (hx : IsSign x) (hy : IsSign y) (k j : ZMod n) :
    chg x k j + chg y k j = chg (fun i => x i * y i) k j +
      2 * (if x j * y j = x (j + k) * y (j + k) then chg x k j else 0) := by
  unfold chg
  rcases hx j with h1 | h1 <;> rcases hx (j + k) with h2 | h2 <;>
    rcases hy j with h3 | h3 <;> rcases hy (j + k) with h4 | h4 <;> simp [h1, h2, h3, h4]

/-! ### Symmetric pair: exact stage-2 identity -/

/-- The kept set of the symmetric solve at lag `k`: `j ≠ -h` (not the fixed point) with
`q_j = q_{j+k}` for the product `q = c d`.  In `hkt_prefix` this is `kept & NFIXM_[k]`. -/
def keptSymm (c d : ZMod n → ℤ) (k h : ZMod n) : Finset (ZMod n) :=
  ((univ : Finset (ZMod n)).erase (-h)).filter
    (fun j => c j * d j = c (j + k) * d (j + k))

theorem chg_fix_symm {x : ZMod n → ℤ} (hsym : IsSymmetric x) {k h : ZMod n} (hk : 2 * h = k) :
    chg x k (-h) = 0 := by
  have : -h + k = h := by rw [← hk]; ring
  unfold chg; rw [this, hsym h]; simp

theorem changes_symm (hn : Odd n) {x : ZMod n → ℤ} (hsym : IsSymmetric x)
    {k h : ZMod n} (hk : 2 * h = k) :
    changes x k = ∑ j ∈ (univ : Finset (ZMod n)).erase (-h), chg x k j := by
  rw [changes_eq_sum_chg, ← Finset.add_sum_erase _ _ (Finset.mem_univ (-h)),
    chg_fix_symm hsym hk, zero_add]

theorem isSymmetric_mul {c d : ZMod n → ℤ} (hc : IsSymmetric c) (hd : IsSymmetric d) :
    IsSymmetric (fun i => c i * d i) := by
  intro j; simp only; rw [hc j, hd j]

theorem isSign_mul {c d : ZMod n → ℤ} (hc : IsSign c) (hd : IsSign d) :
    IsSign (fun i => c i * d i) := by
  intro j
  rcases hc j with h1 | h1 <;> rcases hd j with h2 | h2 <;> simp [h1, h2]

/-- Stage 2 of the symmetric solve, exact form: for symmetric `±1` rows `c, d` with
product `q`, `u_c(k) + u_d(k) = u_q(k) + Σ_{j ∈ F} [c_j ≠ c_{j+k}]`. -/
theorem stage2_symm_exact (hn : Odd n) {c d : ZMod n → ℤ} (hc : IsSign c) (hd : IsSign d)
    (hcs : IsSymmetric c) (hds : IsSymmetric d) {k h : ZMod n} (hk : 2 * h = k) :
    uval c k + uval d k =
      uval (fun i => c i * d i) k + ∑ j ∈ keptSymm c d k h, chg c k j := by
  have e2 : changes c k + changes d k =
      changes (fun i => c i * d i) k + 2 * ∑ j ∈ keptSymm c d k h, chg c k j := by
    rw [changes_symm hn hcs hk, changes_symm hn hds hk,
      changes_symm hn (isSymmetric_mul hcs hds) hk, ← Finset.sum_add_distrib]
    rw [Finset.sum_congr rfl (fun j _ => chg_add_chg hc hd k j), Finset.sum_add_distrib,
      ← Finset.mul_sum, keptSymm, Finset.sum_filter]
  rw [changes_eq c hc, changes_eq d hd, changes_eq _ (isSign_mul hc hd)] at e2
  linarith

/-- `|F|` is even and `n - 1 = |F| + 2 u_q(k)` for the symmetric pair. -/
theorem card_keptSymm (hn : Odd n) {c d : ZMod n → ℤ} (hc : IsSign c) (hd : IsSign d)
    (hcs : IsSymmetric c) (hds : IsSymmetric d) {k h : ZMod n} (hk : 2 * h = k) :
    ((keptSymm c d k h).card : ℤ) + 2 * uval (fun i => c i * d i) k = n - 1 := by
  have hq := isSign_mul hc hd
  rw [← changes_eq _ hq, changes_symm hn (isSymmetric_mul hcs hds) hk, keptSymm,
    Finset.card_filter]
  push_cast
  rw [← Finset.sum_add_distrib]
  have : ∀ j ∈ (univ : Finset (ZMod n)).erase (-h),
      (if c j * d j = c (j + k) * d (j + k) then (1 : ℤ) else 0) +
        chg (fun i => c i * d i) k j = 1 := by
    intro j _
    unfold chg; split_ifs <;> simp_all
  rw [Finset.sum_congr rfl this]
  simp [Finset.card_erase_of_mem, ZMod.card]
  have : 1 ≤ n := NeZero.one_le
  omega

/-- The kept set is closed under the involution `ι`. -/
theorem iota_mem_keptSymm (hn : Odd n) {c d : ZMod n → ℤ} (hcs : IsSymmetric c)
    (hds : IsSymmetric d) {k h : ZMod n} (hk : 2 * h = k) {j : ZMod n}
    (hj : j ∈ keptSymm c d k h) : -j - k ∈ keptSymm c d k h := by
  simp only [keptSymm, Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true] at hj ⊢
  refine ⟨?_, ?_⟩
  · intro e
    apply hj.1
    rw [← hk] at e
    linear_combination -e
  · have e1 : -j - k = -(j + k) := by ring
    have e2 : -j - k + k = -j := by ring
    rw [e2, e1, hcs, hds, hcs, hds]
    exact hj.2.symm

/-- `Σ_{F} [c_j ≠ c_{j+k}]` is even (it is twice the sum over one representative per pair). -/
theorem even_sum_chg_keptSymm (hn : Odd n) {c d : ZMod n → ℤ} (hcs : IsSymmetric c)
    (hds : IsSymmetric d) {k h : ZMod n} (hk : 2 * h = k) :
    (2 : ℤ) ∣ ∑ j ∈ keptSymm c d k h, chg c k j := by
  have := two_mul_dvd_sum_invol (keptSymm c d k h) (fun j => -j - k) (chg c k) 1
    (fun j hj => iota_mem_keptSymm hn hcs hds hk hj) (by intro j _; ring)
    (by
      intro j hj e
      have hj' := hj
      simp only [keptSymm, Finset.mem_filter, Finset.mem_erase] at hj'
      exact hj'.1.1 ((iota_eq_self_iff hn hk j).mp e))
    (by
      intro j _
      have e1 : -j - k = -(j + k) := by ring
      have e2 : -j - k + k = -j := by ring
      unfold chg
      rw [e2, e1, hcs, hcs]
      by_cases h1 : c (j + k) = c j
      · simp [h1]
      · have h2 : ¬ c j = c (j + k) := fun e => h1 e.symm
        simp [h1, h2])
    (by intro j _; simp)
  simpa using this

end GoethalsSeidel.Census
