/-
No circulant Williamson quadruples of order 35, checked end to end in Lean. This is the board's
census `williamson-n35`, which reproduces the classification `holzmann2008williamson` in
`baselines/references.bib`.

The search is `CensusCore.searchAll P35 decs35` (plain Lean, no Mathlib), evaluated by
`native_decide`.  Everything else is proved here: every Williamson quadruple of order 35 can be
normalized (negation, `x_0 = 1`) and reordered (a permutation of the rows) so that its row sums
are one of the three tuples of `decs35`; each normalized symmetric row is the decoded row of some
half-code in `allVecs 17`; the program's keys are linear forms of the PAF, so the four keys of
a quadruple add up to the targets; and `join_complete` shows the join then returns a pair.
-/
import GoethalsSeidel.Census.Join
import GoethalsSeidel.Census.Symmetry

namespace GoethalsSeidel.Census

open CensusCore Finset

set_option linter.unusedSectionVars false

/-! ### Bridge between the program and the definitions -/

/-- The row decoded from a half-code. -/
def xv (v : List Bool) : ZMod 35 → ℤ := fun j => entry 35 v j.val

theorem sumTo_eq (f : ℕ → ℤ) (k : ℕ) : sumTo f k = ∑ i ∈ range k, f i := by
  induction k with
  | zero => rfl
  | succ k ih => rw [sumTo, ih, Finset.sum_range_succ]

theorem sum_zmod35 (f : ℕ → ℤ) : ∑ j : ZMod 35, f j.val = ∑ i ∈ range 35, f i :=
  Fin.sum_univ_eq_sum_range f 35

theorem rowArr_get (v : List Bool) {j : ℕ} (hj : j < 35) :
    (rowArr 35 v)[j]! = entry 35 v j := by
  simp [rowArr, hj]

theorem pafAt_eq (v : List Bool) {k : ℕ} (hk : k < 35) :
    pafAt 35 (rowArr 35 v) k = paf (xv v) (k : ZMod 35) := by
  unfold pafAt paf xv
  rw [sumTo_eq, ← sum_zmod35]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [rowArr_get v (ZMod.val_lt j), rowArr_get v (Nat.mod_lt _ (by norm_num)), ZMod.val_add,
    ZMod.val_natCast, Nat.mod_eq_of_lt hk]

theorem pafArr_get (x : Array ℤ) {k : ℕ} (hk : k < 35) : (pafArr 35 x)[k]! = pafAt 35 x k := by
  simp [pafArr, hk]

theorem lin_eq (w : ℕ → ℤ) (v : List Bool) :
    lin 35 w (pafArr 35 (rowArr 35 v)) = ∑ s : ZMod 35, w s.val * paf (xv v) s := by
  unfold lin
  rw [sumTo_eq, ← sum_zmod35]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [pafArr_get _ (ZMod.val_lt s), pafAt_eq v (ZMod.val_lt s), ZMod.natCast_zmod_val]

theorem mkRow_v (v : List Bool) : (mkRow P35 v).v = v := rfl

theorem mkRow_sum (v : List Bool) : (mkRow P35 v).sum = rowSum (xv v) := by
  show sumTo (fun j => (rowArr 35 v)[j]!) 35 = _
  rw [sumTo_eq, rowSum, ← sum_zmod35]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [rowArr_get v (ZMod.val_lt j)]; rfl

theorem mkRow_ck (v : List Bool) : (mkRow P35 v).ck =
    (∑ s : ZMod 35, P35.wc1 s.val * paf (xv v) s, ∑ s : ZMod 35, P35.wc2 s.val * paf (xv v) s) := by
  show (lin 35 P35.wc1 (pafArr 35 (rowArr 35 v)), lin 35 P35.wc2 (pafArr 35 (rowArr 35 v))) = _
  rw [lin_eq, lin_eq]

theorem mkRow_fk (v : List Bool) : (mkRow P35 v).fk =
    (∑ s : ZMod 35, P35.wf1 s.val * paf (xv v) s, ∑ s : ZMod 35, P35.wf2 s.val * paf (xv v) s,
      ∑ s : ZMod 35, P35.wf3 s.val * paf (xv v) s) := by
  show (lin 35 P35.wf1 (pafArr 35 (rowArr 35 v)), lin 35 P35.wf2 (pafArr 35 (rowArr 35 v)),
    lin 35 P35.wf3 (pafArr 35 (rowArr 35 v))) = _
  rw [lin_eq, lin_eq, lin_eq]

/-- Any linear form of the PAF: the four values of a quadruple add up to `4n w(0)`. -/
theorem quad_lin (w : ℕ → ℤ) {a b c d : ZMod 35 → ℤ} (h : IsQuadruple a b c d) :
    ∑ s : ZMod 35, w s.val * paf a s + ∑ s : ZMod 35, w s.val * paf b s +
      (∑ s : ZMod 35, w s.val * paf c s + ∑ s : ZMod 35, w s.val * paf d s) = 4 * 35 * w 0 := by
  have : ∀ s : ZMod 35, w s.val * paf a s + w s.val * paf b s +
      (w s.val * paf c s + w s.val * paf d s) =
      w s.val * (paf a s + paf b s + paf c s + paf d s) := fun s => by ring
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib,
    Finset.sum_congr rfl (fun s _ => this s),
    ← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod 35)),
    Finset.sum_eq_zero (fun s hs => by rw [h.paf_zero s (Finset.ne_of_mem_erase hs), mul_zero]),
    paf_zero_shift h.signA, paf_zero_shift h.signB, paf_zero_shift h.signC,
    paf_zero_shift h.signD]
  simp; ring

/-! ### Decoding -/

theorem mem_allVecs (v : List Bool) : v ∈ allVecs v.length := by
  induction v with
  | nil => simp [allVecs]
  | cons b v ih =>
    simp only [List.length_cons, allVecs, List.mem_flatMap]
    exact ⟨v, ih, by cases b <;> simp⟩

/-- Every symmetric `±1` row of length 35 with `x_0 = 1` is a decoded row. -/
theorem exists_code {x : ZMod 35 → ℤ} (hx : IsSign x) (hs : IsSymmetric x) (h0 : x 0 = 1) :
    ∃ v ∈ allVecs 17, xv v = x := by
  set v := (List.range 17).map (fun i => decide (x ((i + 1 : ℕ) : ZMod 35) = -1)) with hv
  have hlen : v.length = 17 := by simp [hv]
  refine ⟨v, hlen ▸ mem_allVecs v, ?_⟩
  funext j
  have hj := ZMod.val_lt j
  simp only [xv, entry]
  by_cases hle : j.val ≤ 35 - j.val
  · rw [if_pos hle]
    by_cases hz : j.val = 0
    · rw [if_pos hz, ← h0]
      congr 1
      exact ((ZMod.val_eq_zero j).mp hz).symm
    · rw [if_neg hz]
      have hidx : j.val - 1 < 17 := by omega
      have hget : v.getD (j.val - 1) false = decide (x j = -1) := by
        rw [List.getD_eq_getElem _ _ (by rw [hlen]; exact hidx)]
        simp only [hv, List.getElem_map, List.getElem_range]
        rw [show j.val - 1 + 1 = j.val by omega, ZMod.natCast_zmod_val]
      rw [hget]
      rcases hx j with h | h <;> simp [h]
  · rw [if_neg hle]
    have hz : ¬ (35 - j.val = 0) := by omega
    rw [if_neg hz]
    have hidx : 35 - j.val - 1 < 17 := by omega
    have hneg : (((35 - j.val - 1 + 1 : ℕ)) : ZMod 35) = -j := by
      rw [show 35 - j.val - 1 + 1 = 35 - j.val by omega, Nat.cast_sub hj.le, ZMod.natCast_self,
        ZMod.natCast_zmod_val, zero_sub]
    have hget : v.getD (35 - j.val - 1) false = decide (x j = -1) := by
      rw [List.getD_eq_getElem _ _ (by rw [hlen]; exact hidx)]
      simp only [hv, List.getElem_map, List.getElem_range]
      rw [hneg, hs j]
    rw [hget]
    rcases hx j with h | h <;> simp [h]

/-! ### Soundness of the search for one row-sum tuple -/

theorem mem_rows {x : ZMod 35 → ℤ} (hx : IsSign x) (hs : IsSymmetric x) (h0 : x 0 = 1)
    (sum : ℤ) (hsum : rowSum x = sum) :
    ∃ r ∈ (allRows P35).filter (fun r => r.sum == sum), xv r.v = x ∧
      r.ck = (∑ s : ZMod 35, P35.wc1 s.val * paf x s, ∑ s : ZMod 35, P35.wc2 s.val * paf x s) ∧
      r.fk = (∑ s : ZMod 35, P35.wf1 s.val * paf x s, ∑ s : ZMod 35, P35.wf2 s.val * paf x s,
        ∑ s : ZMod 35, P35.wf3 s.val * paf x s) := by
  obtain ⟨v, hv, hxv⟩ := exists_code hx hs h0
  refine ⟨mkRow P35 v, ?_, by rw [mkRow_v, hxv], by rw [mkRow_ck, hxv], by rw [mkRow_fk, hxv]⟩
  rw [List.mem_filter]
  refine ⟨List.mem_map.mpr ⟨v, hv, rfl⟩, ?_⟩
  rw [mkRow_sum, hxv, hsum]; simp

/-- If a normalized Williamson quadruple has row sums `dec`, the search for `dec` returns a
pair. -/
theorem search_ne_nil {a b c d : ZMod 35 → ℤ} (h : IsWilliamson a b c d)
    (ha0 : a 0 = 1) (hb0 : b 0 = 1) (hc0 : c 0 = 1) (hd0 : d 0 = 1)
    (dec : ℤ × ℤ × ℤ × ℤ) (hsa : rowSum a = dec.1) (hsb : rowSum b = dec.2.1)
    (hsc : rowSum c = dec.2.2.1) (hsd : rowSum d = dec.2.2.2) :
    search P35 (allRows P35) dec ≠ [] := by
  obtain ⟨q, hA, hB, hC, hD⟩ := h
  obtain ⟨ra, hra, -, hcka, hfka⟩ := mem_rows q.signA hA ha0 _ hsa
  obtain ⟨rb, hrb, -, hckb, hfkb⟩ := mem_rows q.signB hB hb0 _ hsb
  obtain ⟨rc, hrc, -, hckc, hfkc⟩ := mem_rows q.signC hC hc0 _ hsc
  obtain ⟨rd, hrd, -, hckd, hfkd⟩ := mem_rows q.signD hD hd0 _ hsd
  have hmem := join_complete (T := ctarget P35) (T' := ftarget P35) hra hrb hrc hrd
    (by
      rw [hcka, hckb, hckc, hckd]
      simp only [add2, ctarget, Prod.mk.injEq]
      have h1 := quad_lin P35.wc1 q
      have h2 := quad_lin P35.wc2 q
      have hn : ((P35.n : ℕ) : ℤ) = 35 := rfl
      rw [hn]
      exact ⟨by linarith, by linarith⟩)
    (by
      rw [hfka, hfkb, hfkc, hfkd]
      simp only [add3, ftarget, Prod.mk.injEq]
      have h1 := quad_lin P35.wf1 q
      have h2 := quad_lin P35.wf2 q
      have h3 := quad_lin P35.wf3 q
      have hn : ((P35.n : ℕ) : ℤ) = 35 := rfl
      rw [hn]
      exact ⟨by linarith, by linarith, by linarith⟩)
  intro hnil
  unfold search at hnil
  rw [hnil] at hmem
  simp at hmem

/-! ### Normalization and reordering -/

theorem normalize {x : ZMod 35 → ℤ} (hx : IsSign x) (hs : IsSymmetric x) :
    IsSign (fun j => x 0 * x j) ∧ IsSymmetric (fun j => x 0 * x j) ∧ (x 0 * x 0 = 1) ∧
      ∀ s, paf (fun j => x 0 * x j) s = paf x s := by
  have h00 : x 0 * x 0 = 1 := by rcases hx 0 with h | h <;> rw [h] <;> norm_num
  refine ⟨fun j => ?_, fun j => by simp only; rw [hs j], h00, fun s => ?_⟩
  · rcases hx 0 with h | h <;> rcases hx j with h' | h' <;> simp [h, h']
  · unfold paf
    refine Finset.sum_congr rfl (fun j _ => ?_)
    calc x 0 * x j * (x 0 * x (j + s)) = (x 0 * x 0) * (x j * x (j + s)) := by ring
      _ = x j * x (j + s) := by rw [h00, one_mul]

def vals : Fin 6 → ℤ := ![-9, -5, -1, 3, 7, 11]

set_option maxRecDepth 100000 in
theorem perm_exists : ∀ f : Fin 4 → Fin 6,
    (vals (f 0)) ^ 2 + (vals (f 1)) ^ 2 + (vals (f 2)) ^ 2 + (vals (f 3)) ^ 2 = 140 →
    ∃ σ : Equiv.Perm (Fin 4),
      (vals (f (σ 0)), vals (f (σ 1)), vals (f (σ 2)), vals (f (σ 3))) ∈ decs35 := by
  decide

theorem rowSum_mem_vals {x : ZMod 35 → ℤ} (hx : IsSign x) (hs : IsSymmetric x) (h0 : x 0 = 1)
    (hsq : rowSum x ^ 2 ≤ 140) : ∃ i : Fin 6, vals i = rowSum x := by
  have hmod := rowSum_symm_modEq (by decide) hx hs h0
  have hb : -12 < rowSum x ∧ rowSum x < 12 := by constructor <;> nlinarith
  have hm : rowSum x % 4 = 3 := by
    have := Int.ModEq.eq hmod; push_cast at this; omega
  obtain ⟨h1, h2⟩ := hb
  generalize rowSum x = s at h1 h2 hm ⊢
  interval_cases s <;> first | (exfalso; omega) | decide

/-- The rows of a quadruple given as a function of `Fin 4` may be permuted. -/
theorem williamson_perm (q : Fin 4 → ZMod 35 → ℤ)
    (h : IsWilliamson (q 0) (q 1) (q 2) (q 3)) (σ : Equiv.Perm (Fin 4)) :
    IsWilliamson (q (σ 0)) (q (σ 1)) (q (σ 2)) (q (σ 3)) := by
  obtain ⟨hq, h0, h1, h2, h3⟩ := h
  have hsign : ∀ i, IsSign (q i) := by
    intro i; fin_cases i
    exacts [hq.signA, hq.signB, hq.signC, hq.signD]
  have hsym : ∀ i, IsSymmetric (q i) := by
    intro i; fin_cases i
    exacts [h0, h1, h2, h3]
  refine ⟨⟨hsign _, hsign _, hsign _, hsign _, fun s hs => ?_⟩, hsym _, hsym _, hsym _, hsym _⟩
  have e1 := Equiv.sum_comp σ (fun i => paf (q i) s)
  simp only [Fin.sum_univ_four] at e1
  rw [e1]
  exact hq.paf_zero s hs

/-- The search result, evaluated by the compiled program. -/
theorem searchAll_P35 : searchAll P35 decs35 = true := by native_decide

/-- **No circulant (symmetric) Williamson matrices of order 35.** -/
theorem no_williamson_35 : ¬ ∃ a b c d : ZMod 35 → ℤ, IsWilliamson a b c d := by
  rintro ⟨a, b, c, d, hW⟩
  obtain ⟨hq, hA, hB, hC, hD⟩ := hW
  obtain ⟨sa, ya, a00, pa⟩ := normalize hq.signA hA
  obtain ⟨sb, yb, b00, pb⟩ := normalize hq.signB hB
  obtain ⟨sc, yc, c00, pc⟩ := normalize hq.signC hC
  obtain ⟨sd, yd, d00, pd⟩ := normalize hq.signD hD
  set q : Fin 4 → ZMod 35 → ℤ :=
    ![fun j => a 0 * a j, fun j => b 0 * b j, fun j => c 0 * c j, fun j => d 0 * d j] with hqdef
  have hW' : IsWilliamson (q 0) (q 1) (q 2) (q 3) :=
    ⟨⟨sa, sb, sc, sd, fun s hs => by
      simp only [hqdef, Matrix.cons_val]
      rw [pa, pb, pc, pd]; exact hq.paf_zero s hs⟩, ya, yb, yc, yd⟩
  have hnorm : ∀ i, q i 0 = 1 := by
    intro i; fin_cases i
    exacts [a00, b00, c00, d00]
  have hsign : ∀ i, IsSign (q i) := by
    intro i; fin_cases i
    exacts [sa, sb, sc, sd]
  have hsym : ∀ i, IsSymmetric (q i) := by
    intro i; fin_cases i
    exacts [ya, yb, yc, yd]
  have hsq := hW'.1.rowSum_sq
  have hle : ∀ i, rowSum (q i) ^ 2 ≤ 140 := by
    have := sq_nonneg (rowSum (q 0)); have := sq_nonneg (rowSum (q 1))
    have := sq_nonneg (rowSum (q 2)); have := sq_nonneg (rowSum (q 3))
    intro i; fin_cases i <;> simp <;> push_cast at hsq <;> linarith
  choose f hf using fun i => rowSum_mem_vals (hsign i) (hsym i) (hnorm i) (hle i)
  obtain ⟨σ, hσ⟩ := perm_exists f (by rw [hf, hf, hf, hf]; push_cast at hsq; linarith)
  have hWσ := williamson_perm q hW' σ
  have hall := searchAll_P35
  unfold searchAll at hall
  rw [List.all_eq_true] at hall
  have hempty := hall _ hσ
  rw [List.isEmpty_iff] at hempty
  exact search_ne_nil hWσ (hnorm _) (hnorm _) (hnorm _) (hnorm _) _
    (hf _).symm (hf _).symm (hf _).symm (hf _).symm hempty

end GoethalsSeidel.Census
