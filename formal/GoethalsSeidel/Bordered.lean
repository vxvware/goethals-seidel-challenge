/-
Bordered Goethals–Seidel arrays and skew-Hadamard matrices.

The bordered arrays are those of Egan, Kotsireas, London and Ó Catháin, "Recent progress in the
construction of Hadamard matrices" (arXiv:2610.27175), which describe the 2026 matrices of
Alpöge, Voinov and Reynolds-Haertle at the orders 668, 716, 1388, 1436, 1676, 1772 and 1916.
A matrix of order `b + 4v` is written in block form

```
  H = [ K      E Πᵀ ]
      [ Π F    M    ]
```

where `G` is a finite abelian group of order `v`, `S` is a subgroup of index `k`, `Π = I₄ ⊗ Π_S`
with `Π_S` the incidence matrix of elements and cosets of `S`, `M` is the Goethals–Seidel array of
four group-developed `±1` matrices over `G`, and `K`, `E`, `F` are integer matrices of sizes
`b × b`, `b × 4k` and `4k × b`.

* `card_odd_of_skew`: if `H + Hᵀ = 2I` and the first generator is a `±1` function, then `v` and
  `k` are odd.  Nothing about `K`, `E`, `F` or the other generators is used.
* `even_index_of_border`: the border condition `F Fᵀ = 4k I − 4 (I₄ ⊗ J_k)` of the width
  `4(k − 1)` construction forces `k` to be even when `b > 0`.
* `not_skew_of_border`: hence no matrix of the width `4(k − 1)` construction is skew.
* `eight_dvd_of_skew_width_four_mul_index`: for border width `4k`, a skew matrix has order
  divisible by 8.
-/
import Mathlib

namespace GoethalsSeidel.Bordered

open Matrix

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]

/-- The group-developed matrix of `f`: entry `(x, y)` is `f (y - x)`. -/
def dev (f : G → ℤ) : Matrix G G ℤ := Matrix.of fun x y => f (y - x)

/-- The permutation matrix of `x ↦ -x`. -/
def negPerm : Matrix G G ℤ := Matrix.of fun x y => if y = -x then 1 else 0

/-- The sixteen blocks of the Goethals–Seidel array in the generators `X, Y, Z, W`, as printed
in the survey: rows `A, BR, CR, DR`; `-BR, A, DᵀR, -CᵀR`; `-CR, -DᵀR, A, BᵀR`;
`-DR, CᵀR, -BᵀR, A`. -/
def gsBlocks (X Y Z W : G → ℤ) : Fin 4 → Fin 4 → Matrix G G ℤ :=
  let A := dev X
  let B := dev Y
  let C := dev Z
  let D := dev W
  let R : Matrix G G ℤ := negPerm
  ![![A, B * R, C * R, D * R],
    ![-(B * R), A, Dᵀ * R, -(Cᵀ * R)],
    ![-(C * R), -(Dᵀ * R), A, Bᵀ * R],
    ![-(D * R), Cᵀ * R, -(Bᵀ * R), A]]

/-- The Goethals–Seidel array over `G`, indexed by `Fin 4 × G`. -/
def gsArray (X Y Z W : G → ℤ) : Matrix (Fin 4 × G) (Fin 4 × G) ℤ :=
  Matrix.of fun p q => gsBlocks X Y Z W p.1 q.1 p.2 q.2

/-- Every diagonal block of the array is the group-developed matrix of the first generator. -/
theorem gsArray_diag (X Y Z W : G → ℤ) (i : Fin 4) (x y : G) :
    gsArray X Y Z W (i, x) (i, y) = X (y - x) := by
  fin_cases i <;> simp [gsArray, gsBlocks, dev]

variable (S : AddSubgroup G) [Fintype (G ⧸ S)] [DecidableEq (G ⧸ S)]

/-- `Π = I₄ ⊗ Π_S`: entry `((i, x), (j, c))` is `1` when `i = j` and `x` lies in the coset `c`. -/
def incidence : Matrix (Fin 4 × G) (Fin 4 × (G ⧸ S)) ℤ :=
  Matrix.of fun p c => if p.1 = c.1 ∧ ((p.2 : G ⧸ S) = c.2) then 1 else 0

/-- The bordered array `[[K, E Πᵀ], [Π F, M]]` with `M` the Goethals–Seidel array. -/
def bordered {b : ℕ} (K : Matrix (Fin b) (Fin b) ℤ) (E : Matrix (Fin b) (Fin 4 × (G ⧸ S)) ℤ)
    (F : Matrix (Fin 4 × (G ⧸ S)) (Fin b) ℤ) (X Y Z W : G → ℤ) :
    Matrix (Fin b ⊕ (Fin 4 × G)) (Fin b ⊕ (Fin 4 × G)) ℤ :=
  Matrix.fromBlocks K (E * (incidence S)ᵀ) (incidence S * F) (gsArray X Y Z W)

/-- `I₄ ⊗ J_k`: entry `((i, c), (j, d))` is `1` when `i = j`. -/
def blockOnes : Matrix (Fin 4 × (G ⧸ S)) (Fin 4 × (G ⧸ S)) ℤ :=
  Matrix.of fun p q => if p.1 = q.1 then 1 else 0

omit [DecidableEq G] in
/-- A `±1` function `f` on a finite abelian group with `f x + f (-x) = 2` at `0` and `0`
elsewhere exists only on a group of odd order: an element of order 2 would give `2 f x = 0`. -/
theorem card_odd_of_skewType {f : G → ℤ} (hsign : ∀ x, f x = 1 ∨ f x = -1)
    (hskew : ∀ x : G, x ≠ 0 → f x + f (-x) = 0) : Odd (Fintype.card G) := by
  rcases Nat.even_or_odd (Fintype.card G) with he | ho
  · exfalso
    have : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩
    obtain ⟨x, hx⟩ := exists_prime_addOrderOf_dvd_card 2 (even_iff_two_dvd.mp he)
    have hx0 : x ≠ 0 := by
      intro h
      rw [h, addOrderOf_zero] at hx
      norm_num at hx
    have h2 : x + x = 0 := by
      rw [← two_nsmul, ← hx]
      exact addOrderOf_nsmul_eq_zero x
    have hneg : -x = x := by
      rw [neg_eq_iff_add_eq_zero]
      exact h2
    have := hskew x hx0
    rw [hneg] at this
    rcases hsign x with h | h <;> rw [h] at this <;> norm_num at this
  · exact ho

/-- A skew bordered array (`H + Hᵀ = 2I`) whose first generator is a `±1` function has `v` odd and
`k` odd.  The border blocks `K`, `E`, `F` and the generators `Y, Z, W` are arbitrary. -/
theorem card_odd_of_skew {b : ℕ} (K : Matrix (Fin b) (Fin b) ℤ)
    (E : Matrix (Fin b) (Fin 4 × (G ⧸ S)) ℤ) (F : Matrix (Fin 4 × (G ⧸ S)) (Fin b) ℤ)
    (X Y Z W : G → ℤ) (hX : ∀ x, X x = 1 ∨ X x = -1)
    (hskew : bordered S K E F X Y Z W + (bordered S K E F X Y Z W)ᵀ = 2 • (1 : Matrix _ _ ℤ)) :
    Odd (Fintype.card G) ∧ Odd (Fintype.card (G ⧸ S)) := by
  have hX' : ∀ x : G, x ≠ 0 → X x + X (-x) = 0 := by
    intro x hx
    have h := congrFun (congrFun hskew (Sum.inr ((0 : Fin 4), 0))) (Sum.inr ((0 : Fin 4), x))
    simp only [bordered, Matrix.add_apply, Matrix.transpose_apply, Matrix.fromBlocks_apply₂₂,
      gsArray_diag, Matrix.smul_apply, Matrix.one_apply] at h
    have hne : (Sum.inr ((0 : Fin 4), (0 : G)) : Fin b ⊕ (Fin 4 × G)) ≠
        Sum.inr ((0 : Fin 4), x) := by
      intro h'
      apply hx
      have := Sum.inr_injective h'
      exact (congrArg Prod.snd this).symm
    simp only [hne, ↓reduceIte] at h
    simpa using h
  have hG := card_odd_of_skewType hX hX'
  refine ⟨hG, ?_⟩
  have hcard : Nat.card G = Nat.card (G ⧸ S) * Nat.card S :=
    AddSubgroup.card_eq_card_quotient_mul_card_addSubgroup S
  rw [Nat.card_eq_fintype_card, Nat.card_eq_fintype_card] at hcard
  rw [hcard] at hG
  exact (Nat.odd_mul.mp hG).1

omit [DecidableEq G] [Fintype G] [DecidableEq (G ⧸ S)] in
/-- A sum of `±1` terms that vanishes has an even number of terms. -/
theorem even_card_of_sum_sign_eq_zero {g : G ⧸ S → ℤ} (hg : ∀ c, g c = 1 ∨ g c = -1)
    (h0 : ∑ c, g c = 0) : Even (Fintype.card (G ⧸ S)) := by
  have hmod : ∀ c, (g c : ZMod 2) = 1 := by
    intro c
    rcases hg c with h | h <;> rw [h] <;> decide
  have : ((∑ c, g c : ℤ) : ZMod 2) = (Fintype.card (G ⧸ S) : ZMod 2) := by
    push_cast
    simp [hmod]
  rw [h0] at this
  simp only [Int.cast_zero] at this
  exact (ZMod.natCast_eq_zero_iff_even).mp this.symm

omit [Fintype G] [DecidableEq G] in
/-- In the width `4(k − 1)` construction the border satisfies `F Fᵀ = 4k I − 4 (I₄ ⊗ J_k)`.
When the border is nonempty this forces the index `k` to be even: the rows of the first block row
of `F` add up to a vector of squared length `4k² − 4k² = 0`, so every column of that block row has
as many `1` as `-1` entries. -/
theorem even_index_of_border {b : ℕ} (hb : 0 < b) (F : Matrix (Fin 4 × (G ⧸ S)) (Fin b) ℤ)
    (hF : ∀ p j, F p j = 1 ∨ F p j = -1)
    (hFF : F * Fᵀ = (4 * (Fintype.card (G ⧸ S) : ℤ)) • (1 : Matrix _ _ ℤ) - 4 • blockOnes S) :
    Even (Fintype.card (G ⧸ S)) := by
  set k : ℤ := (Fintype.card (G ⧸ S) : ℤ) with hk
  -- `w j` is the sum of column `j` over the first block row.
  set w : Fin b → ℤ := fun j => ∑ c, F ((0 : Fin 4), c) j with hw
  have hentry : ∀ c d : G ⧸ S, ∑ j, F ((0 : Fin 4), c) j * F ((0 : Fin 4), d) j =
      4 * k * (if c = d then 1 else 0) - 4 := by
    intro c d
    have h := congrFun (congrFun hFF ((0 : Fin 4), c)) ((0 : Fin 4), d)
    simp only [Matrix.mul_apply, Matrix.transpose_apply, Matrix.sub_apply, Matrix.smul_apply,
      Matrix.one_apply, blockOnes, Matrix.of_apply, ite_true, smul_eq_mul, Prod.mk.injEq,
      true_and] at h
    rw [h]
    split_ifs <;> ring
  have hsq : ∑ j, w j ^ 2 = 0 := by
    have : ∑ j, w j ^ 2 = ∑ c, ∑ d, ∑ j, F ((0 : Fin 4), c) j * F ((0 : Fin 4), d) j := by
      simp only [hw, sq, Finset.sum_mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun c _ => ?_
      rw [Finset.sum_comm]
    rw [this]
    simp only [hentry]
    simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      mul_ite, mul_one, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    rw [← hk]
    ring
  have hw0 : ∀ j, w j = 0 := by
    intro j
    have hnn : ∀ i ∈ Finset.univ, 0 ≤ w i ^ 2 := fun i _ => sq_nonneg _
    have := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hsq j (Finset.mem_univ j)
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
  have j0 : Fin b := ⟨0, hb⟩
  exact even_card_of_sum_sign_eq_zero S (g := fun c => F ((0 : Fin 4), c) j0)
    (fun c => hF _ _) (hw0 j0)

/-- No matrix of the width `4(k − 1)` bordered construction is skew.  The hypotheses are the
first generator being `±1`, the border `F` being `±1` with `F Fᵀ = 4k I − 4 (I₄ ⊗ J_k)`, a
nonempty border (`b = 4(k − 1) > 0` for a proper subgroup `S`), and `H + Hᵀ = 2I`.  The
Goethals–Seidel condition on the generators and the conditions on `K` and `E` are not needed. -/
theorem not_skew_of_border {b : ℕ} (hb : 0 < b) (K : Matrix (Fin b) (Fin b) ℤ)
    (E : Matrix (Fin b) (Fin 4 × (G ⧸ S)) ℤ) (F : Matrix (Fin 4 × (G ⧸ S)) (Fin b) ℤ)
    (X Y Z W : G → ℤ) (hX : ∀ x, X x = 1 ∨ X x = -1)
    (hF : ∀ p j, F p j = 1 ∨ F p j = -1)
    (hFF : F * Fᵀ = (4 * (Fintype.card (G ⧸ S) : ℤ)) • (1 : Matrix _ _ ℤ) - 4 • blockOnes S)
    (hskew : bordered S K E F X Y Z W + (bordered S K E F X Y Z W)ᵀ = 2 • (1 : Matrix _ _ ℤ)) :
    False := by
  have hodd := (card_odd_of_skew S K E F X Y Z W hX hskew).2
  have heven := even_index_of_border S hb F hF hFF
  exact (Nat.not_even_iff_odd.mpr hodd) heven

/-- For border width `b = 4k`, a skew bordered array has order `b + 4v` divisible by 8. -/
theorem eight_dvd_of_skew_width_four_mul_index {b : ℕ} (K : Matrix (Fin b) (Fin b) ℤ)
    (E : Matrix (Fin b) (Fin 4 × (G ⧸ S)) ℤ) (F : Matrix (Fin 4 × (G ⧸ S)) (Fin b) ℤ)
    (X Y Z W : G → ℤ) (hX : ∀ x, X x = 1 ∨ X x = -1)
    (hb : b = 4 * Fintype.card (G ⧸ S))
    (hskew : bordered S K E F X Y Z W + (bordered S K E F X Y Z W)ᵀ = 2 • (1 : Matrix _ _ ℤ)) :
    8 ∣ b + 4 * Fintype.card G := by
  obtain ⟨⟨v, hv⟩, ⟨k, hk⟩⟩ := card_odd_of_skew S K E F X Y Z W hX hskew
  rw [hb, hk, hv]
  exact ⟨k + v + 1, by ring⟩

end GoethalsSeidel.Bordered
