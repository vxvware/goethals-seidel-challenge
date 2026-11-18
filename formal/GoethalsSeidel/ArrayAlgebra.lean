/-
The algebra behind the Goethals–Seidel array, stated abstractly.

`K` is a commutative ring (the circulant matrices), `ι` embeds it in a ring `S` (all matrices),
and `R` is an element of `S` with `R * R = 1` that conjugates each of the eight elements
`a, a', b, b', c, c', d, d'` to its partner (`R * ι a = ι a' * R` and so on); in the
application `x'` is the transpose of `x` and `R` is the back-diagonal permutation matrix.
-/
import Mathlib

namespace GoethalsSeidel

variable {K S : Type*} [CommRing K] [Ring S]

section helpers

variable (ι : K →+* S) (R : S) (hR : R * R = 1)
include hR

theorem mulR_mulR {y y' : K} (hy : R * ι y = ι y' * R) (x : K) :
    ι x * R * (ι y * R) = ι (x * y') := by
  rw [mul_assoc, ← mul_assoc R, hy, mul_assoc, hR, mul_one, map_mul]

omit hR in
theorem mulR_mul {y y' : K} (hy : R * ι y = ι y' * R) (x : K) :
    ι x * R * ι y = ι (x * y') * R := by
  rw [mul_assoc, hy, ← mul_assoc, map_mul]

omit hR in
theorem mul_mulR (x y : K) : ι x * (ι y * R) = ι (x * y) * R := by
  rw [← mul_assoc, map_mul]

end helpers

variable (ι : K →+* S) (R : S) (a a' b b' c c' d d' : K)

/-- The Goethals–Seidel array as a `4 × 4` matrix of blocks. -/
def gsBlocks : Matrix (Fin 4) (Fin 4) S :=
  !![ι a, ι b * R, ι c * R, ι d * R;
     -(ι b * R), ι a, -(ι d' * R), ι c' * R;
     -(ι c * R), ι d' * R, ι a, -(ι b' * R);
     -(ι d * R), -(ι c' * R), ι b' * R, ι a]

/-- The blockwise transpose of the array: block `(i, j)` is the transpose of block `(j, i)`
of `gsBlocks`, using that every block of the form `ι x * R` is symmetric. -/
def gsBlocksT : Matrix (Fin 4) (Fin 4) S :=
  !![ι a', -(ι b * R), -(ι c * R), -(ι d * R);
     ι b * R, ι a', ι d' * R, -(ι c' * R);
     ι c * R, -(ι d' * R), ι a', ι b' * R;
     ι d * R, ι c' * R, -(ι b' * R), ι a']

set_option linter.unusedSimpArgs false in
theorem gsBlocks_mul_gsBlocksT (hR : R * R = 1)
    (ha' : R * ι a' = ι a * R)
    (hb : R * ι b = ι b' * R) (hb' : R * ι b' = ι b * R)
    (hc : R * ι c = ι c' * R) (hc' : R * ι c' = ι c * R)
    (hd : R * ι d = ι d' * R) (hd' : R * ι d' = ι d * R) :
    gsBlocks ι R a b b' c c' d d' * gsBlocksT ι R a' b b' c c' d d'
      = Matrix.diagonal fun _ => ι (a * a' + b * b' + c * c' + d * d') := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [gsBlocks, gsBlocksT, Matrix.mul_apply, Fin.sum_univ_four,
      mulR_mulR ι R hR ha', mulR_mulR ι R hR hb, mulR_mulR ι R hR hb',
      mulR_mulR ι R hR hc, mulR_mulR ι R hR hc', mulR_mulR ι R hR hd, mulR_mulR ι R hR hd',
      mulR_mul ι R ha', mulR_mul ι R hb, mulR_mul ι R hb',
      mulR_mul ι R hc, mulR_mul ι R hc', mulR_mul ι R hd, mulR_mul ι R hd',
      mul_mulR, ← map_mul, mul_comm, mul_left_comm] <;>
    abel

end GoethalsSeidel
