# Lean formalisation

Machine-checked statements about Goethals–Seidel quadruples, in Lean 4 with Mathlib. The toolchain is pinned in `lean-toolchain` and Mathlib in `lake-manifest.json`. Every theorem depends only on the axioms `propext`, `Classical.choice` and `Quot.sound`, except `Census.searchAll_P35` and `Census.no_williamson_35`, which also depend on the axiom `searchAll_P35._native.native_decide.ax_1_1` that `native_decide` adds for its one evaluation. `AxiomCheck.lean` prints the axioms of the main theorems (`lake env lean AxiomCheck.lean`).

Build with `lake exe cache get` and then `lake build`. The `native_decide` step of the order-35 proof runs the search in `CensusCore` as compiled code, which takes about three minutes on a laptop; `CensusCore` is built with `precompileModules`. The board's CI does not build this directory, and nothing in the verifier depends on it.

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

`GoethalsSeidel/Order49.lean`:

- `Order49.exists_g_quadruple`: the board entry `n49-703082c3c1ffa487f376` is a quadruple with two skew and two symmetric blocks, so G-matrices of order 49 exist. The board records a published exhaustive search that reports none at this order.

`GoethalsSeidel/Bordered.lean`:

- `card_odd_of_skew`: in a bordered Goethals–Seidel array of order `b + 4v` over an abelian group of order `v` with a subgroup of index `k`, the kind behind the 2026 Hadamard matrices of orders 668 to 1916, a skew matrix forces `v` and `k` to be odd.
- `not_skew_of_border`: the width `4(k - 1)` border construction never gives a skew matrix, because its border condition forces `k` to be even.
- `eight_dvd_of_skew_width_four_mul_index`: with border width `4k`, a skew matrix has order divisible by 8. So neither border width reaches a skew-Hadamard matrix of order congruent to 4 modulo 8, which includes 404 and 428.

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

`GoethalsSeidel/EqualBlocks.lean`:

- `IsQuadruple.not_three_equal`: a quadruple with three equal sequences has `n ≢ 2 (mod 3)`.
- `IsQuadruple.not_two_pairs`: a quadruple of the form `(a, a, c, c)` has `n ≢ 3 (mod 4)`.

`GoethalsSeidel/OrbitFamily.lean`: row-sum bounds for sequences constant on the orbits of a multiplier subgroup `H` of the units of `ZMod n`. `IsPAFFamily k lam a` says that `k` sign sequences have periodic autocorrelations summing to `lam` at every nonzero shift. A quadruple is the case `k = 4`, `lam = 0`; the eight-block array is `k = 8`, `lam = 0`; the bordered array is `k = 4`, `lam = -4`. At prime order `p`, `f` is the order of `H` and `e` its index, so `p = ef + 1`.

- `IsPAFFamily.rowSum_sq`: the row sums satisfy `Σ sᵢ² = kn + lam (n - 1)`.
- `rowSum_orbitForm`: if `g > 0` divides the size of every orbit of `H` on the nonzero residues and `n = gE + 1`, a sign sequence constant on these orbits has row sum `a 0 + gm` with `m ≡ E (mod 2)` and `|m| ≤ E`.
- `IsPAFFamily.orbitForm_bound`, `IsPAFFamily.orbit_bound`: for a family whose row sums have this form, with `K = k + lam`: if `E` is even, every row sum equals the entry at `0` or `4(g - 1) ≤ KE`; if `E` is odd, `k(g - 2) ≤ KE`.
- `IsPAFFamily.orbit_bound_prime`: the same at prime order with `g = f` and `E = e`.
- `IsQuadruple.card_le_index_add_one`: a quadruple of prime order with all four sequences constant on the orbits of `H` has `f ≤ e + 1`.
- `no_orbit_quadruple_of_lt`: hence no such quadruple exists when `p > e² + e + 1`. No symmetry is assumed, so this covers Williamson, good, best, G-matrix and skew quadruples.
- `IsPAFFamily.eight_card_le`: an eight-block family with all sequences orbit-constant has `f ≤ 2e + 1`, and `f ≤ e + 1` when `e` is odd.
- `IsPAFFamily.bordered_orbit`: a bordered-array family (`lam = -4`) with all sequences orbit-constant and `f ≥ 3` has `e` even and every row sum equal to the entry at `0`. The row sums give no upper bound on `f`.
- `IsQuadruple.skew_card_le`: for an odd prime `p` and `f` odd, a quadruple with a skew first sequence (otherwise unrestricted) and the other three orbit-constant has `f ≤ e + 1`.
- `skew_forces_odd`: a skew sequence constant on the orbits of `H` forces `f` odd.
- `symmetric_rows_bound`: if `-1 ∈ H`, `p` is odd and `r` orbit-constant sign sequences have `Σ sᵢ² = 4p - (4 - r)`, then `f ≤ e` for `e` even and `r(f - 2) ≤ 4e` for `e` odd.
- `IsQuadruple.williamson_bound`, `IsQuadruple.good_bound`, `IsQuadruple.kkss_bound`, `IsQuadruple.best_bound`: for odd prime `p`, symmetric sequences constant on the orbits of `H` are constant on the orbits of `H' = H ⊔ ⟨-1⟩`, and with `f`, `e` the order and index of `H'`: Williamson (four symmetric sequences) has `f ≤ e` for `e` even and `f ≤ e + 1` for `e` odd; good matrices (one skew, three symmetric orbit-constant) have `f ≤ e` for `e` even and `3(f - 2) ≤ 4e` for `e` odd; G-matrices (two skew, two symmetric orbit-constant) have `f ≤ e` for `e` even and `f ≤ 2e + 2` for `e` odd; best matrices (three skew, one symmetric orbit-constant `d`) have `f ≤ e` for `e` even and `f ≤ 4e + 2` for `e` odd, and the row sum of `d` is pinned down by `f m² + 2 d(0) m = 4e`. The skew sequences are unrestricted.
- `IsQuadruple.orbitDivisor_le`: for odd `n > 1`, if `g > 0` divides every orbit size of `H` on the nonzero residues and `n = gE + 1`, a quadruple with all four sequences orbit-constant has `g ≤ E + 1`.
- `no_skew_orbit_quadruple_89'`, `no_skew_orbit_quadruple_101'`, `no_skew_orbit_quadruple_107'`: the three theorems of `Obstruction.lean` derived again from `IsQuadruple.skew_card_le` (89 and 107) and from the orbit form (101).

`GoethalsSeidel/Census/`: soundness of the steps that the board's census programs in `research/starter/` (`wil_hkt3.c`, `best_hkt2.c`, `gmat_c*.c`, `hkt_solve64.h`, `wc_compress.c`) use to prune or reduce the search. Each file states the program identity it covers.

- `Basic.lean`: `PAF_x(k) = n - 4 u_x(k)`, where `u_x(k)` counts the `j` with `x_j = -1` and `x_{j+k} = 1`; a quadruple is the same as four sign sequences with `Σ u = n` at every nonzero shift, and for odd `n` the shifts `1, ..., (n-1)/2` suffice.
- `Stage.lean`, `StageSkew.lean`: the stage-1 parity equations and the exact stage-2 identities of the two-stage linear solve, for pairs of symmetric sequences (Williamson, G-matrices) and for the two unknown skew sequences of best matrices.
- `StageSymmCode.lean`, `StageSkewCode.lean`: the stage-2 equations in the bit-coded form the programs build, so every quadruple solves the program's linear system.
- `Symmetry.lean`: a symmetric sequence of odd length with `x_0 = 1` has row sum `≡ n (mod 4)`; for best matrices of order `r² + r + 1` the symmetric sequence has row sum `±(2r + 1)`; negating or reversing a sequence, permuting the four sequences and applying a multiplier to all four preserve quadruples and the best, Williamson and G-matrix types as the censuses use them.
- `Spectral.lean`: the power spectral densities of a quadruple add up to `4n` at every additive character, and a skew sequence has `PSD ≥ 1`, which gives the PSD bounds of the programs. The programs test these bounds in floating point; that step is not covered.
- `Compress.lean`: the `m`-compression of a quadruple for `m ∣ n` satisfies the compressed PAF equations, maps symmetric sequences to symmetric sequences and commutes with multipliers; every unit of `ZMod m` lifts to `ZMod n`; each compressed entry has the parity of `n/m` and absolute value at most `n/m`.
- `Join.lean`, `Order35.lean`, `CensusCore/Search.lean`: `Census.no_williamson_35`: there is no Williamson quadruple (four symmetric sequences) of order 35. This is the board's census `williamson-n35`. The proof normalizes and reorders any quadruple to one of three row-sum tuples, shows that the hash join in `CensusCore` returns a pair whenever a quadruple exists, and evaluates the search on all three tuples with `native_decide`.

## What is not proved

- Any census or nonexistence result that rests on an exhaustive search, other than the Williamson census at order 35.
- That the census programs are implemented as the `Census` files describe them; those files prove the mathematical identities the programs rely on, not the C code.
- Anything about equivalence of quadruples or the board's canonical form.
