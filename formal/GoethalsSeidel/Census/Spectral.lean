/-
Census step (d): the spectral bounds behind the PSD filters.

For an additive character `ψ` of `ZMod n` (every `j ↦ ω^{kj}` with `ω = e^{2πi/n}` is one) the
discrete Fourier coefficient is `x̂(ψ) = Σ_j x_j ψ(j)` and `PSD_x(ψ) = |x̂(ψ)|²`.

* `|x̂(ψ)|² = Σ_s PAF_x(s) ψ(s)`,
* for a quadruple `PSD_a + PSD_b + PSD_c + PSD_d = 4n` at every `ψ`,
* for a skew row `Re x̂(ψ) = 1`, so `PSD_x(ψ) ≥ 1`,
* hence the bounds of `best_hkt2.c` (R3, R4): `PSD_d ≤ 4n - 3`, `PSD_a ≤ 4n - 2 - PSD_d`, and
  the largest of the three skew spectra is at least `(4n - PSD_d)/3`; and the bounds of
  `wil_hkt3.c` / `gmat_c*.c`: every `PSD ≤ 4n`, `PSD_a + PSD_b ≤ 4n`.
The programs test these in floating point with a relaxation of `1e-6` in the accepting direction;
that numerical step is not covered here.
-/
import GoethalsSeidel.Census.Basic

namespace GoethalsSeidel.Census

open Finset Complex

set_option linter.unusedSectionVars false

variable {n : ℕ} [NeZero n]

/-- The Fourier coefficient of `x` at the character `ψ`. -/
noncomputable def dft (ψ : AddChar (ZMod n) ℂ) (x : ZMod n → ℤ) : ℂ := ∑ j, (x j : ℂ) * ψ j

/-- The power spectral density `|x̂(ψ)|²`. -/
noncomputable def psd (ψ : AddChar (ZMod n) ℂ) (x : ZMod n → ℤ) : ℝ := Complex.normSq (dft ψ x)

theorem psd_nonneg (ψ : AddChar (ZMod n) ℂ) (x : ZMod n → ℤ) : 0 ≤ psd ψ x :=
  Complex.normSq_nonneg _

/-- Wiener–Khinchin for `ZMod n`: `|x̂(ψ)|² = Σ_s PAF_x(s) ψ(-s)`. -/
theorem psd_eq_sum_paf (ψ : AddChar (ZMod n) ℂ) (x : ZMod n → ℤ) :
    (psd ψ x : ℂ) = ∑ s, (paf x s : ℂ) * ψ (-s) := by
  unfold psd
  rw [← Complex.mul_conj, dft, map_sum, Finset.sum_mul_sum]
  have h1 : ∀ j l : ZMod n, (x j : ℂ) * ψ j * (starRingEnd ℂ) ((x l : ℂ) * ψ l) =
      (x j : ℂ) * x l * ψ (j - l) := by
    intro j l
    rw [map_mul, map_intCast, ← AddChar.map_neg_eq_conj, sub_eq_add_neg,
      AddChar.map_add_eq_mul]
    ring
  simp_rw [h1]
  -- substitute `l = j + s`
  have h2 : ∀ j : ZMod n, ∑ l, (x j : ℂ) * x l * ψ (j - l) =
      ∑ s, (x j : ℂ) * x (j + s) * ψ (-s) := by
    intro j
    refine (Fintype.sum_equiv (Equiv.addLeft j) _ _ (fun s => ?_)).symm
    simp only [Equiv.coe_addLeft]
    rw [show j - (j + s) = -s by ring]
  simp_rw [h2]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  unfold paf
  push_cast
  rw [Finset.sum_mul]

/-- For a quadruple, the four spectra add up to `4n` at every character. -/
theorem _root_.GoethalsSeidel.IsQuadruple.psd_sum {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ψ : AddChar (ZMod n) ℂ) : psd ψ a + psd ψ b + psd ψ c + psd ψ d = 4 * n := by
  have key : ((psd ψ a + psd ψ b + psd ψ c + psd ψ d : ℝ) : ℂ) = ((4 * n : ℝ) : ℂ) := by
    push_cast
    rw [psd_eq_sum_paf, psd_eq_sum_paf, psd_eq_sum_paf, psd_eq_sum_paf,
      ← Finset.sum_add_distrib, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n))]
    rw [Finset.sum_eq_zero (fun s hs => by
      have := h.paf_zero s (Finset.ne_of_mem_erase hs)
      rw [← add_mul, ← add_mul, ← add_mul]
      have : ((paf a s : ℂ) + paf b s + paf c s + paf d s) = 0 := by exact_mod_cast this
      rw [this, zero_mul])]
    rw [paf_zero_shift h.signA, paf_zero_shift h.signB, paf_zero_shift h.signC,
      paf_zero_shift h.signD]
    simp; ring
  exact_mod_cast key

/-- A skew row has `Re x̂(ψ) = 1`. -/
theorem re_dft_skew {x : ZMod n → ℤ} (hs : IsSign x) (hx : IsSkew x) (ψ : AddChar (ZMod n) ℂ) :
    (dft ψ x).re = 1 := by
  unfold dft
  rw [Complex.re_sum, ← Finset.add_sum_erase _ _ (Finset.mem_univ (0 : ZMod n))]
  rw [Finset.sum_involution (fun j _ => -j)]
  · simp [hx.1]
  · intro j hj
    have hj0 : j ≠ 0 := Finset.ne_of_mem_erase hj
    rw [hx.2 j hj0, AddChar.map_neg_eq_conj]
    push_cast
    simp [Complex.mul_re]
  · intro j hj hne e
    apply hne
    have hj0 : j ≠ 0 := Finset.ne_of_mem_erase hj
    have hxj : x (-j) = -x j := hx.2 j hj0
    rw [e] at hxj
    have : x j = 0 := by linarith
    rcases hs j with h | h <;> rw [h] at this <;> norm_num at this
  · intro j hj
    simpa using Finset.ne_of_mem_erase hj
  · intro j _; simp

/-- A skew row has `PSD ≥ 1` at every character. -/
theorem psd_skew_ge_one {x : ZMod n → ℤ} (hs : IsSign x) (hx : IsSkew x)
    (ψ : AddChar (ZMod n) ℂ) : 1 ≤ psd ψ x := by
  unfold psd
  rw [Complex.normSq_apply, re_dft_skew hs hx ψ]
  nlinarith [mul_self_nonneg (dft ψ x).im]

section best
variable {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d) (ha : IsSkew a) (hb : IsSkew b)
  (hc : IsSkew c) (ψ : AddChar (ZMod n) ℂ)
include h ha hb hc

/-- R3 of `best_hkt2.c`: `PSD_d ≤ 4n - 3`. -/
theorem best_psd_d_le : psd ψ d ≤ 4 * n - 3 := by
  have := h.psd_sum ψ
  have := psd_skew_ge_one h.signA ha ψ
  have := psd_skew_ge_one h.signB hb ψ
  have := psd_skew_ge_one h.signC hc ψ
  linarith

/-- R3 of `best_hkt2.c`: `PSD_a ≤ 4n - 2 - PSD_d`. -/
theorem best_psd_a_le : psd ψ a ≤ 4 * n - 2 - psd ψ d := by
  have := h.psd_sum ψ
  have := psd_skew_ge_one h.signB hb ψ
  have := psd_skew_ge_one h.signC hc ψ
  linarith

/-- R4 of `best_hkt2.c`: the largest skew spectrum at a character is at least
`(4n - PSD_d)/3`, so permuting the skew rows puts a row with `PSD_a ≥ (4n - PSD_d)/3` first. -/
theorem best_psd_max_ge :
    (4 * n - psd ψ d) / 3 ≤ max (psd ψ a) (max (psd ψ b) (psd ψ c)) := by
  have := h.psd_sum ψ
  have h1 := le_max_left (psd ψ a) (max (psd ψ b) (psd ψ c))
  have h2 := le_trans (le_max_left (psd ψ b) (psd ψ c))
    (le_max_right (psd ψ a) (max (psd ψ b) (psd ψ c)))
  have h3 := le_trans (le_max_right (psd ψ b) (psd ψ c))
    (le_max_right (psd ψ a) (max (psd ψ b) (psd ψ c)))
  linarith

end best

/-- The ceiling used by `wil_hkt3.c` and `gmat_c*.c`: in any quadruple, `PSD_a + PSD_b ≤ 4n`
(and so each spectrum is at most `4n`). -/
theorem _root_.GoethalsSeidel.IsQuadruple.psd_pair_le {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ψ : AddChar (ZMod n) ℂ) : psd ψ a + psd ψ b ≤ 4 * n := by
  have := h.psd_sum ψ
  have := psd_nonneg ψ c
  have := psd_nonneg ψ d
  linarith

/-- G-matrices: `PSD_c + PSD_d ≤ 4n - 2` (two skew rows each contribute at least `1`). -/
theorem _root_.GoethalsSeidel.IsQuadruple.gmat_psd_cd_le {a b c d : ZMod n → ℤ} (h : IsQuadruple a b c d)
    (ha : IsSkew a) (hb : IsSkew b) (ψ : AddChar (ZMod n) ℂ) :
    psd ψ c + psd ψ d ≤ 4 * n - 2 := by
  have := h.psd_sum ψ
  have := psd_skew_ge_one h.signA ha ψ
  have := psd_skew_ge_one h.signB hb ψ
  linarith

end GoethalsSeidel.Census
