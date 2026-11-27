/-
The stage-2 equation of `best_hkt2.c` in the form the program builds it.

The unknowns are the code bits `y_u = e^b_u` for `u ∈ L = {1, ..., H}` (`H = (n-1)/2`), so
`e^b_{n-u} = 1 - y_u`.  For the kept set `F = F_k` the program's row `k` of `M2(t)` has
coefficient `[u ∈ F] + [u - k ∈ F]` at `y_u` (`row[k-1] = (F ^ rotl(F, k)) >> 1`), and the
constant `hx = |F ∩ HX| / 2` with `HX = {j | [j ∈ U] ≠ [j + k ∈ U]}`, `U = {H+1, ..., n-1}`
(`hx = popu(F & HX_[k]) >> 1`).  The theorem below is

  `Σ_{j ∈ F} [b_j ≠ b_{j+k}] ≡ 2 Σ_{u ∈ L} y_u ([u ∈ F] + [u - k ∈ F]) + |F ∩ HX|  (mod 4)`,

which together with `stage2_best` (`Σ_{F} [b_j ≠ b_{j+k}] = v = K + H - (u_a + u_d)`) is the
equation `parity(row_k · y) = (v/2 + hx) mod 2` checked by `stage_init`.
-/
import GoethalsSeidel.Census.StageSkew

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-- Lower half `L = {1..H}`. -/
def InL (j : ZMod n) : Prop := 1 ≤ j.val ∧ j.val ≤ (n - 1) / 2
/-- Upper half `U = {H+1..n-1}`. -/
def InU (j : ZMod n) : Prop := (n - 1) / 2 < j.val

instance (j : ZMod n) : Decidable (InL j) := by unfold InL; infer_instance
instance (j : ZMod n) : Decidable (InU j) := by unfold InU; infer_instance

/-- The `0/1` indicator of a proposition. -/
def ind (P : Prop) [Decidable P] : ℤ := if P then 1 else 0

theorem ind_congr {P Q : Prop} [Decidable P] [Decidable Q] (h : P ↔ Q) : ind P = ind Q := by
  unfold ind
  by_cases hp : P
  · simp [hp, h.mp hp]
  · have hq : ¬ Q := fun hq => hp (h.mpr hq)
    simp [hp, hq]

theorem card_filter_eq_sum_ind (S : Finset (ZMod n)) (P : ZMod n → Prop) [DecidablePred P] :
    ((S.filter P).card : ℤ) = ∑ j ∈ S, ind (P j) := by
  rw [Finset.card_filter]; push_cast; rfl

theorem inU_iff_not_inL (hn : Odd n) {j : ZMod n} (hj : j ≠ 0) : InU j ↔ ¬ InL j := by
  have : j.val ≠ 0 := by rwa [ne_eq, ZMod.val_eq_zero]
  unfold InU InL; omega

theorem inL_neg_iff (hn : Odd n) {j : ZMod n} (hj : j ≠ 0) : InL (-j) ↔ InU j := by
  haveI : NeZero j := ⟨hj⟩
  have hv : (-j).val = n - j.val := ZMod.val_neg_of_ne_zero j
  have : j.val ≠ 0 := by rwa [ne_eq, ZMod.val_eq_zero]
  have hlt : j.val < n := ZMod.val_lt j
  obtain ⟨m, hm⟩ := hn
  unfold InL InU; rw [hv]; omega

theorem inU_neg_iff (hn : Odd n) {j : ZMod n} (hj : j ≠ 0) : InU (-j) ↔ InL j := by
  have hj' : -j ≠ 0 := neg_ne_zero.mpr hj
  rw [inU_iff_not_inL hn hj', inL_neg_iff hn hj, inU_iff_not_inL hn hj, not_not]

/-- The code bit read at the folded position of `j`: `y_{|j|}`. -/
def yhat (b : ZMod n → ℤ) (j : ZMod n) : ℤ := if InL j then bit b j else bit b (-j)

theorem yhat_neg (hn : Odd n) (b : ZMod n → ℤ) (j : ZMod n) : yhat b (-j) = yhat b j := by
  by_cases hj : j = 0
  · subst hj; simp
  unfold yhat
  rw [neg_neg]
  by_cases h1 : InL j
  · have : ¬ InL (-j) := by
      rw [inL_neg_iff hn hj, inU_iff_not_inL hn hj, not_not]; exact h1
    simp [h1, this]
  · have : InL (-j) := by rw [inL_neg_iff hn hj, inU_iff_not_inL hn hj]; exact h1
    simp [h1, this]

theorem yhat_zero_or_one (b : ZMod n → ℤ) (j : ZMod n) : yhat b j = 0 ∨ yhat b j = 1 := by
  unfold yhat; split_ifs <;> exact bit_zero_or_one _ _

/-- For a skew row and `j ≠ 0`: `e_j = y_{|j|} xor [j ∈ U]`. -/
theorem bit_eq_xor (hn : Odd n) {b : ZMod n → ℤ} (hbs : IsSign b) (hb : IsSkew b) {j : ZMod n}
    (hj : j ≠ 0) :
    bit b j = yhat b j + ind (InU j) - 2 * yhat b j * ind (InU j) := by
  unfold yhat ind
  by_cases h1 : InL j
  · have : ¬ InU j := by rw [inU_iff_not_inL hn hj, not_not]; exact h1
    simp [h1, this]
  · have : InU j := by rw [inU_iff_not_inL hn hj]; exact h1
    simp only [h1, this, ite_false, ite_true]
    unfold bit
    rw [hb.2 j hj]
    rcases hbs j with h2 | h2 <;> simp [h2]

/-- The xor of two `0/1` integers. -/
def xr (a b : ℤ) : ℤ := a + b - 2 * a * b

theorem chg_eq_xor {b : ZMod n → ℤ} (hbs : IsSign b) (k j : ZMod n) :
    chg b k j = xr (bit b j) (bit b (j + k)) := by
  unfold chg xr bit
  rcases hbs j with h1 | h1 <;> rcases hbs (j + k) with h2 | h2 <;> simp [h1, h2]

theorem xr_xr (a b c d : ℤ) (ha : a = 0 ∨ a = 1) (hb : b = 0 ∨ b = 1) (hc : c = 0 ∨ c = 1)
    (hd : d = 0 ∨ d = 1) : xr (xr a c) (xr b d) = xr (xr a b) (xr c d) := by
  rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> rcases hc with rfl | rfl <;>
    rcases hd with rfl | rfl <;> simp [xr]

/-- The equivalence `j ↦ -j - k` of `ZMod n`. -/
def iotaEquiv (k : ZMod n) : ZMod n ≃ ZMod n where
  toFun j := -j - k
  invFun j := -j - k
  left_inv j := by simp
  right_inv j := by simp

/-- Reindexing a sum over an `ι`-invariant set. -/
theorem sum_ind_iota (S : Finset (ZMod n)) (k : ZMod n)
    (hS : ∀ j, j ∈ S ↔ -j - k ∈ S) (f : ZMod n → ℤ) :
    ∑ j, ind (j ∈ S) * f j = ∑ j, ind (j ∈ S) * f (-j - k) := by
  refine (Fintype.sum_equiv (iotaEquiv k) (fun j => ind (j ∈ S) * f (-j - k))
    (fun j => ind (j ∈ S) * f j) (fun j => ?_)).symm
  simp only [iotaEquiv, Equiv.coe_fn_mk]
  unfold ind
  by_cases h : j ∈ S
  · simp [h, (hS j).mp h]
  · have : ¬ (-j - k) ∈ S := fun h' => h ((hS j).mpr h')
    simp [h, this]

theorem sum_mem_eq_ind (S : Finset (ZMod n)) (f : ZMod n → ℤ) :
    ∑ j ∈ S, f j = ∑ j, ind (j ∈ S) * f j := by
  unfold ind
  simp only [ite_mul, one_mul, zero_mul]
  rw [Finset.sum_ite_mem, Finset.univ_inter]

theorem xr_ind (P Q : Prop) [Decidable P] [Decidable Q] :
    xr (ind P) (ind Q) = ind (¬ (P ↔ Q)) := by
  unfold xr ind
  by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]

theorem xr_comm (a b : ℤ) : xr a b = xr b a := by unfold xr; ring

theorem xr_one_sub (a b : ℤ) : xr (1 - a) (1 - b) = xr a b := by unfold xr; ring

section
variable (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
  {b c : ZMod n → ℤ} (hbs : IsSign b) (hcs : IsSign c) (hb : IsSkew b) (hc : IsSkew c)
include hn hk0 hk hb hc

theorem keptSkew_iota_iff (j : ZMod n) :
    j ∈ keptSkew b c k h ↔ -j - k ∈ keptSkew b c k h := by
  constructor
  · exact iota_mem_keptSkew hn hk0 hk hb hc
  · intro h'
    have := iota_mem_keptSkew hn hk0 hk hb hc h'
    simpa using this

theorem ne_zero_of_mem_keptSkew {j : ZMod n} (hj : j ∈ keptSkew b c k h) :
    j ≠ 0 ∧ j + k ≠ 0 := by
  simp only [keptSkew, Finset.mem_filter, exclSkew, Finset.mem_sdiff, Finset.mem_univ,
    Finset.mem_insert, Finset.mem_singleton, true_and, not_or] at hj
  obtain ⟨⟨h0, hmk, _⟩, _⟩ := hj
  exact ⟨h0, fun e => hmk (by linear_combination e)⟩

/-- Pairs of an invariant function over `F` give an even sum. -/
theorem even_sum_keptSkew (g : ZMod n → ℤ)
    (hg : ∀ j ∈ keptSkew b c k h, g (-j - k) = g j) :
    (2 : ℤ) ∣ ∑ j ∈ keptSkew b c k h, g j := by
  have := two_mul_dvd_sum_invol (keptSkew b c k h) (fun j => -j - k) g 1
    (fun j hj => iota_mem_keptSkew hn hk0 hk hb hc hj) (by intro j _; ring)
    (by
      intro j hj
      simp only [keptSkew, Finset.mem_filter] at hj
      exact iota_ne_rest hn hk0 hk hj.1)
    hg (by intro j _; simp)
  simpa using this

/-- The half-sum identity: `Σ_F (y_{|j|} + y_{|j+k|}) = 2 Σ_{u ∈ L} y_u ([u ∈ F] + [u-k ∈ F])`. -/
theorem sum_yhat_keptSkew :
    ∑ j ∈ keptSkew b c k h, (yhat b j + yhat b (j + k)) =
      2 * ∑ u, ind (InL u) * bit b u *
        (ind (u ∈ keptSkew b c k h) + ind (u - k ∈ keptSkew b c k h)) := by
  set F := keptSkew b c k h with hF
  have hS := keptSkew_iota_iff hn hk0 hk hb hc (c := c)
  have hne := fun {j} (hj : j ∈ F) => ne_zero_of_mem_keptSkew hn hk0 hk hb hc (j := j) hj
  -- split each `yhat` into its lower and upper parts
  have hy1 : ∀ j ∈ F, yhat b j = ind (InL j) * bit b j + ind (InU j) * bit b (-j) := by
    intro j hj
    unfold yhat ind
    by_cases h1 : InL j
    · have : ¬ InU j := by rw [inU_iff_not_inL hn (hne hj).1, not_not]; exact h1
      simp [h1, this]
    · have : InU j := by rw [inU_iff_not_inL hn (hne hj).1]; exact h1
      simp [h1, this]
  have hy2 : ∀ j ∈ F, yhat b (j + k) =
      ind (InL (j + k)) * bit b (j + k) + ind (InU (j + k)) * bit b (-(j + k)) := by
    intro j hj
    unfold yhat ind
    by_cases h1 : InL (j + k)
    · have : ¬ InU (j + k) := by rw [inU_iff_not_inL hn (hne hj).2, not_not]; exact h1
      simp [h1, this]
    · have : InU (j + k) := by rw [inU_iff_not_inL hn (hne hj).2]; exact h1
      simp [h1, this]
  -- the upper parts, reindexed by `ι`
  have hU1 : ∑ j, ind (j ∈ F) * (ind (InU j) * bit b (-j)) =
      ∑ j, ind (j ∈ F) * (ind (InL (j + k)) * bit b (j + k)) := by
    rw [sum_ind_iota F k hS]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hj : j ∈ F
    · have e1 : -j - k = -(j + k) := by ring
      rw [e1, neg_neg, ind_congr (inU_neg_iff hn (hne hj).2)]
    · simp [ind, hj]
  have hU2 : ∑ j, ind (j ∈ F) * (ind (InU (j + k)) * bit b (-(j + k))) =
      ∑ j, ind (j ∈ F) * (ind (InL j) * bit b j) := by
    rw [sum_ind_iota F k hS]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hj : j ∈ F
    · have e1 : -j - k + k = -j := by ring
      rw [e1, neg_neg, ind_congr (inU_neg_iff hn (hne hj).1)]
    · simp [ind, hj]
  have hshift : ∑ u, ind (InL u) * bit b u * ind (u - k ∈ F) =
      ∑ j, ind (j ∈ F) * (ind (InL (j + k)) * bit b (j + k)) := by
    rw [← sum_shift (fun u => ind (InL u) * bit b u * ind (u - k ∈ F)) k]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    simp only [add_sub_cancel_right]; ring
  rw [sum_mem_eq_ind]
  have hsplit : ∀ j, ind (j ∈ F) * (yhat b j + yhat b (j + k)) =
      ind (j ∈ F) * (ind (InL j) * bit b j) + ind (j ∈ F) * (ind (InU j) * bit b (-j)) +
      ind (j ∈ F) * (ind (InL (j + k)) * bit b (j + k)) +
      ind (j ∈ F) * (ind (InU (j + k)) * bit b (-(j + k))) := by
    intro j
    by_cases hj : j ∈ F
    · rw [hy1 j hj, hy2 j hj]; ring
    · simp [ind, hj]
  rw [Finset.sum_congr rfl (fun j _ => hsplit j)]
  simp only [Finset.sum_add_distrib]
  rw [hU1, hU2]
  have hsum2 : ∑ u, ind (InL u) * bit b u * (ind (u ∈ F) + ind (u - k ∈ F)) =
      ∑ j, ind (j ∈ F) * (ind (InL j) * bit b j) +
        ∑ j, ind (j ∈ F) * (ind (InL (j + k)) * bit b (j + k)) := by
    rw [← hshift, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun j _ => by ring)
  rw [hsum2]; ring

include hbs in
/-- The row form of stage 2 modulo 4. -/
theorem stage2_skew_rowform_mod4 :
    ∑ j ∈ keptSkew b c k h, chg b k j ≡
      2 * ∑ u, ind (InL u) * bit b u *
        (ind (u ∈ keptSkew b c k h) + ind (u - k ∈ keptSkew b c k h)) +
      ((keptSkew b c k h).filter (fun j => ¬ (InU j ↔ InU (j + k)))).card [ZMOD 4] := by
  set F := keptSkew b c k h with hF
  have hne := fun {j} (hj : j ∈ F) => ne_zero_of_mem_keptSkew hn hk0 hk hb hc (j := j) hj
  set A : ZMod n → ℤ := fun j => xr (yhat b j) (yhat b (j + k)) with hA
  set B : ZMod n → ℤ := fun j => xr (ind (InU j)) (ind (InU (j + k))) with hB
  have hchg : ∀ j ∈ F, chg b k j = A j + B j - 2 * (A j * B j) := by
    intro j hj
    rw [chg_eq_xor hbs, bit_eq_xor hn hbs hb (hne hj).1, bit_eq_xor hn hbs hb (hne hj).2]
    have := xr_xr (yhat b j) (yhat b (j + k)) (ind (InU j)) (ind (InU (j + k)))
      (yhat_zero_or_one b j) (yhat_zero_or_one b (j + k))
      (by unfold ind; split_ifs <;> simp) (by unfold ind; split_ifs <;> simp)
    unfold xr at this ⊢
    simp only [hA, hB, xr]
    linear_combination this
  have hAinv : ∀ j ∈ F, A (-j - k) = A j := by
    intro j _
    simp only [hA]
    rw [show -j - k + k = -j by ring, show -j - k = -(j + k) by ring, yhat_neg hn,
      yhat_neg hn, xr_comm]
  have hBinv : ∀ j ∈ F, B (-j - k) = B j := by
    intro j hj
    simp only [hB]
    rw [show -j - k + k = -j by ring, show -j - k = -(j + k) by ring]
    have e1 : ind (InU (-(j + k))) = 1 - ind (InU (j + k)) := by
      rw [ind_congr (inU_neg_iff hn (hne hj).2), ind_congr (inU_iff_not_inL hn (hne hj).2)]
      unfold ind
      by_cases h1 : InL (j + k) <;> simp [h1]
    have e2 : ind (InU (-j)) = 1 - ind (InU j) := by
      rw [ind_congr (inU_neg_iff hn (hne hj).1), ind_congr (inU_iff_not_inL hn (hne hj).1)]
      unfold ind
      by_cases h1 : InL j <;> simp [h1]
    rw [e1, e2, xr_one_sub, xr_comm]
  -- Σ_F A B is even
  obtain ⟨m1, hm1⟩ := even_sum_keptSkew hn hk0 hk hb hc (fun j => A j * B j)
    (fun j hj => by show A (-j - k) * B (-j - k) = A j * B j; rw [hAinv j hj, hBinv j hj])
  -- Σ_F y_j y_{j+k} is even
  obtain ⟨m2, hm2⟩ := even_sum_keptSkew hn hk0 hk hb hc (fun j => yhat b j * yhat b (j + k))
    (fun j _ => by
      show yhat b (-j - k) * yhat b (-j - k + k) = yhat b j * yhat b (j + k)
      rw [show -j - k + k = -j by ring, show -j - k = -(j + k) by ring, yhat_neg hn,
        yhat_neg hn, mul_comm])
  have hsumA : ∑ j ∈ F, A j = ∑ j ∈ F, (yhat b j + yhat b (j + k)) -
      2 * ∑ j ∈ F, yhat b j * yhat b (j + k) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    simp only [hA, xr]; ring
  have hsumB : ∑ j ∈ F, B j = ((F.filter (fun j => ¬ (InU j ↔ InU (j + k)))).card : ℤ) := by
    simp only [hB, xr_ind]
    rw [card_filter_eq_sum_ind]
  rw [Finset.sum_congr rfl hchg, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, hm1, hsumA, hm2, hsumB, sum_yhat_keptSkew hn hk0 hk hb hc]
  rw [Int.modEq_iff_dvd]
  exact ⟨m1 + m2, by ring⟩

/-- `|F ∩ HX|` is even (so `hx = popu(F & HX_[k]) >> 1` is exact). -/
theorem even_card_hx :
    (2 : ℤ) ∣ ((keptSkew b c k h).filter (fun j => ¬ (InU j ↔ InU (j + k)))).card := by
  have hne := fun {j} (hj : j ∈ keptSkew b c k h) =>
    ne_zero_of_mem_keptSkew hn hk0 hk hb hc (j := j) hj
  have := even_sum_keptSkew hn hk0 hk hb hc
    (fun j => xr (ind (InU j)) (ind (InU (j + k))))
    (by
      intro j hj
      show xr (ind (InU (-j - k))) (ind (InU (-j - k + k))) = xr (ind (InU j)) (ind (InU (j + k)))
      rw [show -j - k + k = -j by ring, show -j - k = -(j + k) by ring]
      have e1 : ind (InU (-(j + k))) = 1 - ind (InU (j + k)) := by
        rw [ind_congr (inU_neg_iff hn (hne hj).2), ind_congr (inU_iff_not_inL hn (hne hj).2)]
        unfold ind
        by_cases h1 : InL (j + k) <;> simp [h1]
      have e2 : ind (InU (-j)) = 1 - ind (InU j) := by
        rw [ind_congr (inU_neg_iff hn (hne hj).1), ind_congr (inU_iff_not_inL hn (hne hj).1)]
        unfold ind
        by_cases h1 : InL j <;> simp [h1]
      rw [e1, e2, xr_one_sub, xr_comm])
  simp only [xr_ind] at this
  rw [card_filter_eq_sum_ind]
  exact this

end

/-- The stage-2 equation of `best_hkt2.c` exactly as the program forms it.  With
`H = (n-1)/2`, `K = |F|/2`, `v = K + H - (u_a(k) + u_d(k))` and `hx = |F ∩ HX|/2`, every best
quadruple satisfies `v/2 ≡ Σ_{u=1}^{H} y_u ([u ∈ F] + [u-k ∈ F]) + hx (mod 2)` where `y_u` is the
code bit of `b` at `u`.  This is the identity `parc(row[k-1] & yb) == ((v >> 1) + hx) & 1`
that `stage_init` samples, so every solution of the integer problem solves the program's
stage-2 system for its pattern `t`. -/
theorem stage2_best_code (hn : Odd n) {a b c d : ZMod n → ℤ} (hq : IsQuadruple a b c d)
    (hb : IsSkew b) (hc : IsSkew c) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k) :
    (((n : ℤ) - 1) / 2 + ((keptSkew b c k h).card : ℤ) / 2 - (uval a k + uval d k)) / 2 ≡
      ∑ u, ind (InL u) * bit b u *
          (ind (u ∈ keptSkew b c k h) + ind (u - k ∈ keptSkew b c k h)) +
        (((keptSkew b c k h).filter (fun j => ¬ (InU j ↔ InU (j + k)))).card : ℤ) / 2
      [ZMOD 2] := by
  have h2 := stage2_best hn hq hb hc hk0 hk
  have h4 := stage2_skew_rowform_mod4 hn hk0 hk hq.signB hb hc
  obtain ⟨mK, hmK⟩ := even_card_keptSkew hn hk0 hk hb hc
  obtain ⟨mS, hmS⟩ := even_sum_chg_keptSkew hn hk0 hk hb hc
  obtain ⟨mX, hmX⟩ := even_card_hx hn hk0 hk hb hc
  obtain ⟨mn, hmn⟩ := hn
  have hn' : (n : ℤ) = 2 * mn + 1 := by exact_mod_cast hmn
  rw [hmK, hmX, hn']
  rw [hmK, hmS, hn'] at h2
  rw [hmS, hmX] at h4
  have e1 : (2 * (mn : ℤ) + 1 - 1) / 2 = mn := by omega
  have e2 : (2 * mK) / 2 = mK := by omega
  have e3 : (2 * mX) / 2 = mX := by omega
  rw [e1, e2, e3]
  have hv : (mn : ℤ) + mK - (uval a k + uval d k) = 2 * mS := by linarith
  rw [hv, show (2 * mS) / 2 = mS by omega]
  obtain ⟨r, hr⟩ := (Int.modEq_iff_dvd.mp h4)
  rw [Int.modEq_iff_dvd]
  refine ⟨r, ?_⟩
  linarith

end GoethalsSeidel.Census
