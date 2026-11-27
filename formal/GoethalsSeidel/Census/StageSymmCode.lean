/-
The stage-2 equation of the symmetric HKT solve (`hkt_solve64.h`, used by `wil_hkt3.c` for
Williamson matrices and by `gmat_c*.c` for G-matrices) in the form the program builds it.

The unknowns are the bits `γ_u = e^c_u` for `u ∈ L = {1..H}`, with `c_0 = 1` fixed.  For the
kept set `F = {j ≠ -h | q_j = q_{j+k}}` (`kept & NFIXM_[k]`, `q = c d`) the row `k` of the stage-2
matrix has coefficient `[u ∈ F] + [u - k ∈ F]` at `γ_u` (`oddu = G ^ rot(G, k)` read at bits
`1..H`).  Every quadruple satisfies

  `v / 2 ≡ Σ_{u ∈ L} γ_u ([u ∈ F] + [u - k ∈ F])  (mod 2)`,   `v = H + 1 + |F|/2 - (u_a + u_b)`.

The program's own variable is the complemented bit `1 - γ_u` (`hkt_symword`: code bit `1` means
`+1`); that substitution only moves a constant to the right-hand side.
-/
import GoethalsSeidel.Census.StageSkewCode

namespace GoethalsSeidel.Census

open Finset

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

theorem inU_neg_iff' (hn : Odd n) (j : ZMod n) : InU (-j) ↔ InL j := by
  by_cases hj : j = 0
  · subst hj; simp [InU, InL]
  · exact inU_neg_iff hn hj

section
variable (hn : Odd n) {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k)
  {c d : ZMod n → ℤ} (hcs : IsSign c) (hcsym : IsSymmetric c) (hdsym : IsSymmetric d)
include hn hk hcsym hdsym

theorem keptSymm_iota_iff (j : ZMod n) :
    j ∈ keptSymm c d k h ↔ -j - k ∈ keptSymm c d k h := by
  constructor
  · exact iota_mem_keptSymm hn hcsym hdsym hk
  · intro h'
    have := iota_mem_keptSymm hn hcsym hdsym hk h'
    simpa using this

theorem even_sum_keptSymm (g : ZMod n → ℤ)
    (hg : ∀ j ∈ keptSymm c d k h, g (-j - k) = g j) :
    (2 : ℤ) ∣ ∑ j ∈ keptSymm c d k h, g j := by
  have := two_mul_dvd_sum_invol (keptSymm c d k h) (fun j => -j - k) g 1
    (fun j hj => iota_mem_keptSymm hn hcsym hdsym hk hj) (by intro j _; ring)
    (by
      intro j hj e
      simp only [keptSymm, Finset.mem_filter, Finset.mem_erase] at hj
      exact hj.1.1 ((iota_eq_self_iff hn hk j).mp e))
    hg (by intro j _; simp)
  simpa using this

include hcs in
/-- The row form of the symmetric stage 2 modulo 4 (`c_0 = 1`). -/
theorem stage2_symm_rowform_mod4 (hc0 : c 0 = 1) :
    ∑ j ∈ keptSymm c d k h, chg c k j ≡
      2 * ∑ u, ind (InL u) * bit c u *
        (ind (u ∈ keptSymm c d k h) + ind (u - k ∈ keptSymm c d k h)) [ZMOD 4] := by
  set F := keptSymm c d k h with hF
  have hS := keptSymm_iota_iff hn hk hcsym hdsym (c := c) (d := d)
  have hg0 : bit c 0 = 0 := by unfold bit; simp [hc0]
  -- chg = γ_j + γ_{j+k} - 2 γ_j γ_{j+k}
  have hchg : ∀ j, chg c k j = bit c j + bit c (j + k) - 2 * (bit c j * bit c (j + k)) := by
    intro j; rw [chg_eq_xor hcs]; unfold xr; ring
  obtain ⟨m, hm⟩ := even_sum_keptSymm hn hk hcsym hdsym (fun j => bit c j * bit c (j + k))
    (fun j _ => by
      show bit c (-j - k) * bit c (-j - k + k) = bit c j * bit c (j + k)
      rw [show -j - k + k = -j by ring, show -j - k = -(j + k) by ring,
        bit_neg_of_symm hcsym, bit_neg_of_symm hcsym, mul_comm])
  -- Σ_F γ_{j+k} = Σ_F γ_j
  have hshiftF : ∑ j ∈ F, bit c (j + k) = ∑ j ∈ F, bit c j := by
    rw [sum_mem_eq_ind F, sum_mem_eq_ind F, sum_ind_iota F k hS (fun j => bit c (j + k))]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [show -j - k + k = -j by ring, bit_neg_of_symm hcsym]
  -- Σ_F γ_j as a sum over the lower half
  have hsplit : ∀ j, bit c j = ind (InL j) * bit c j + ind (InU j) * bit c j := by
    intro j
    by_cases hj : j = 0
    · subst hj; rw [hg0]; ring
    · unfold ind
      by_cases h1 : InL j
      · have : ¬ InU j := by rw [inU_iff_not_inL hn hj, not_not]; exact h1
        simp [h1, this]
      · have : InU j := by rw [inU_iff_not_inL hn hj]; exact h1
        simp [h1, this]
  have hU : ∑ j, ind (j ∈ F) * (ind (InU j) * bit c j) =
      ∑ j, ind (j ∈ F) * (ind (InL (j + k)) * bit c (j + k)) := by
    rw [sum_ind_iota F k hS]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    rw [show -j - k = -(j + k) by ring, ind_congr (inU_neg_iff' hn (j + k)),
      bit_neg_of_symm hcsym]
  have hshift : ∑ u, ind (InL u) * bit c u * ind (u - k ∈ F) =
      ∑ j, ind (j ∈ F) * (ind (InL (j + k)) * bit c (j + k)) := by
    rw [← sum_shift (fun u => ind (InL u) * bit c u * ind (u - k ∈ F)) k]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    simp only [add_sub_cancel_right]; ring
  have hF1 : ∑ j ∈ F, bit c j = ∑ u, ind (InL u) * bit c u * (ind (u ∈ F) + ind (u - k ∈ F)) := by
    rw [sum_mem_eq_ind F]
    rw [Finset.sum_congr rfl (fun j _ => by rw [hsplit j])]
    simp only [mul_add, Finset.sum_add_distrib]
    rw [hU, ← hshift, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  rw [Finset.sum_congr rfl (fun j _ => hchg j), Finset.sum_sub_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, hm, hshiftF, hF1, Int.modEq_iff_dvd]
  exact ⟨m, by ring⟩

end

/-- `|F|` is even for the symmetric kept set. -/
theorem even_card_keptSymm (hn : Odd n) {k h : ZMod n} (hk : 2 * h = k) {c d : ZMod n → ℤ}
    (hcsym : IsSymmetric c) (hdsym : IsSymmetric d) :
    (2 : ℤ) ∣ ((keptSymm c d k h).card : ℤ) := by
  have := even_sum_keptSymm hn hk hcsym hdsym (fun _ => (1 : ℤ)) (fun _ _ => rfl)
  simpa using this

/-- The stage-2 equation of the symmetric HKT solve for a quadruple `a, b, c, d` whose rows
`c, d` are symmetric and `c_0 = 1` (Williamson: all four rows; G-matrices: `a, b` skew).  With
`H = (n-1)/2`, `K = |F|/2` and `v = H + 1 + K - (u_a(k) + u_b(k))`:
`v = Σ_F [c_j ≠ c_{j+k}]` is even and `v/2 ≡ Σ_{u ∈ L} γ_u ([u ∈ F] + [u - k ∈ F]) (mod 2)`. -/
theorem stage2_symm_code (hn : Odd n) {a b c d : ZMod n → ℤ} (hq : IsQuadruple a b c d)
    (hcsym : IsSymmetric c) (hdsym : IsSymmetric d) (hc0 : c 0 = 1)
    {k h : ZMod n} (hk0 : k ≠ 0) (hk : 2 * h = k) :
    ((n : ℤ) - 1) / 2 + 1 + ((keptSymm c d k h).card : ℤ) / 2 - (uval a k + uval b k) =
        ∑ j ∈ keptSymm c d k h, chg c k j ∧
      ((((n : ℤ) - 1) / 2 + 1 + ((keptSymm c d k h).card : ℤ) / 2 - (uval a k + uval b k)) / 2
        ≡ ∑ u, ind (InL u) * bit c u *
          (ind (u ∈ keptSymm c d k h) + ind (u - k ∈ keptSymm c d k h)) [ZMOD 2]) := by
  have hsum := (isQuadruple_iff_uval a b c d hq.signA hq.signB hq.signC hq.signD).mp hq k hk0
  have hex := stage2_symm_exact hn hq.signC hq.signD hcsym hdsym hk
  have hcard := card_keptSymm hn hq.signC hq.signD hcsym hdsym hk
  have h4 := stage2_symm_rowform_mod4 hn hk hq.signC hcsym hdsym hc0
  obtain ⟨mK, hmK⟩ := even_card_keptSymm hn hk hcsym hdsym
  obtain ⟨mS, hmS⟩ := even_sum_chg_keptSymm hn hcsym hdsym hk
  obtain ⟨mn, hmn⟩ := hn
  have hn' : (n : ℤ) = 2 * mn + 1 := by exact_mod_cast hmn
  have e1 : ((n : ℤ) - 1) / 2 = mn := by rw [hn']; omega
  have e2 : ((keptSymm c d k h).card : ℤ) / 2 = mK := by rw [hmK]; omega
  rw [e1, e2]
  have hv : (mn : ℤ) + 1 + mK - (uval a k + uval b k) = ∑ j ∈ keptSymm c d k h, chg c k j := by
    rw [hmK] at hcard
    linarith
  refine ⟨hv, ?_⟩
  rw [hv, hmS, show (2 * mS) / 2 = mS by omega]
  rw [hmS] at h4
  obtain ⟨r, hr⟩ := Int.modEq_iff_dvd.mp h4
  rw [Int.modEq_iff_dvd]
  exact ⟨r, by linarith⟩

end GoethalsSeidel.Census
