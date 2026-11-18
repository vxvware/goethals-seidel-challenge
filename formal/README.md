# Lean formalisation

Machine-checked statements about Goethals–Seidel quadruples, in Lean 4 with Mathlib. The toolchain is pinned in `lean-toolchain` and Mathlib in `lake-manifest.json`. Every theorem depends only on the axioms `propext`, `Classical.choice` and `Quot.sound`.

Build with `lake exe cache get` and then `lake build`. The board's CI does not build this directory, and nothing in the verifier depends on it.

## What is proved

`GoethalsSeidel/Basic.lean` defines a quadruple as four sequences `ZMod n → ℤ` with entries `1` or `-1` whose periodic autocorrelations sum to zero at every nonzero shift, and the skew and symmetric conditions on a sequence.

- `IsQuadruple.rowSum_sq`: the row sums `a, b, c, d` of a quadruple satisfy `a² + b² + c² + d² = 4n`.
- `IsSkew.rowSum_eq_one`: a skew sequence has row sum `1`.

`GoethalsSeidel/Types.lean`: conditions on the order for the symmetry types on the board, each from the row-sum identity.

- `IsQuadruple.best_order`: best matrices (three skew blocks) of order `n` exist only when `n = r² + r + 1`.
- `IsQuadruple.kkss_order`: G-matrices (two skew blocks) of order `n` exist only when `2n - 1` is a sum of two squares.
- `IsQuadruple.good_rowSum`: for good matrices (one skew block) the other row sums satisfy `b² + c² + d² = 4n - 1`.

`GoethalsSeidel/Orbit.lean`:

- `two_mul_card_dvd_rowSum_sub`: for an odd prime `p` and a subgroup `H` of odd order of the units of `ZMod p`, a `±1` sequence that is constant on the orbits of `H` under multiplication has row sum congruent to its entry at `0` modulo `2|H|`.

`GoethalsSeidel/Obstruction.lean`:

- `no_skew_orbit_quadruple_89`, `no_skew_orbit_quadruple_101`, `no_skew_orbit_quadruple_107`: at each of these orders there is no quadruple with one skew block and three blocks constant on the orbits of a nontrivial multiplier subgroup of odd order. The skew block itself is unrestricted. These are the three smallest orders `4n` with no known skew-Hadamard matrix, and the theorems say only that this one construction cannot reach them.

`GoethalsSeidel/Order119.lean`:

- `Order119.exists_skew_quadruple`: the board entry `n119-f302cc45a2f83e60917b` is a quadruple with a skew block, checked by kernel evaluation.

`GoethalsSeidel/Circulant.lean`, `GoethalsSeidel/ArrayAlgebra.lean`, `GoethalsSeidel/Array.lean`: the Goethals–Seidel array. `gsArray a b c d` is the matrix of order `4n`, indexed by `Fin 4 × ZMod n`, with blocks

```
  A     B R     C R     D R
 -B R   A      -Dᵀ R    Cᵀ R
 -C R   Dᵀ R    A      -Bᵀ R
 -D R  -Cᵀ R    Bᵀ R    A
```

where `A, B, C, D` are the circulants with first rows `a, b, c, d` (entry `(i, j)` is `a (j - i)`) and `R` is the back-diagonal permutation matrix. This is the array the site draws.

- `IsQuadruple.gsArray_mul_transpose`: the array `H` of a quadruple satisfies `H Hᵀ = 4n I`.
- `gsArray_entry`: every entry of the array is `1` or `-1`.
- `IsSkew.gsArray_add_transpose`: when the first block is skew, `H + Hᵀ = 2 I`.

`GoethalsSeidel/Hadamard.lean`:

- `IsQuadruple.gsArray_isHadamard`: the array of a quadruple is a Hadamard matrix in the sense of Mathlib's `Matrix.IsHadamard`.
- `exists_skew_hadamard_476`: a skew-Hadamard matrix of order 476 exists, namely the array of the order-119 entry above.

## What is not proved

- Any census or nonexistence result that rests on an exhaustive search.
- Anything about equivalence of quadruples or the board's canonical form.
