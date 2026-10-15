# Mathematical contract: equiv-v1

The accepted v1 group is cyclic of odd order 1 ≤ n ≤ 511. An entry supplies
four length-n rows of integers in {-1,1}, or four distinct-index sets encoding
negative entries. The invariant factors must be [n]. Noncyclic abelian groups
are reserved in the schema but rejected by this verifier version.

For every shift s, PAF_X(s) = sum_j X[j] X[(j+s) mod n]. The verifier requires
the four PAFs to sum to 4n at zero and zero at all other shifts. Python integer
bit counts implement this identity without FFT rounding or fixed-width overflow.

Normalize each row's first entry to +1. Symmetric means X[-j]=X[j]; skew means
X[-j]=-X[j] for j≠0. Membership is computed from these predicates, invariant
under equiv-v1. At n=1 both predicates hold, so all six tracks apply.

## Equivalence decision

Generators: independent whole-row sign changes, independent reversal j→-j,
permutation of blocks, and simultaneous multiplication of all indices by a
unit modulo n. The canonical representative is the lexicographically least
sign-normalized, independently reversal-normalized, sorted tuple of rows over
all simultaneous multipliers. Fingerprints include n, the cyclic group, and
the literal version equiv-v1. Equal fingerprints are backed by equal canonical
rows; the hash is an index, not a substitute for the exact comparison.

Independent cyclic shifts are excluded. For example, (1,1,-1) is skew at n=3,
whereas its rotation (-1,1,1), normalized to (1,-1,-1), is symmetric. Including
shifts could merge named strata and alter published counts. This resolves the
plan's open shift question in favor of its primary requirement: compatibility
with the published best-matrix convention. A future shift quotient requires a
new version and separate statistics. No claim of Hadamard-matrix isomorphism
is made; the decision is exact only under the stated generators.

The secondary branch-v1 convention on best quadruples permits simultaneous
multipliers, permutation of the three skew rows, and D-negation, with skew
origins fixed to +1. At n=43 the full normalized catalogue gives 20 branch-v1
orbits and five equiv-v1 classes. Independent reversals preserve every PAF.

Source: Bright et al., https://arxiv.org/html/1907.04987v2, §§5.2 and 6.

## Evidence and status

Existence is reproduced on every validation. Only implemented, exactly checked
proof rules get proven. Receipt-backed means the actual archived coverage
records pass the receipt checks; spot-checked additionally requires a trusted
recheck report bound to the entry and engine. Reproduced nonexistence requires
full re-execution by an implemented engine. Citations alone have no computed
verification tier: their status is literature, with the citation displayed.
Likewise, historical records with missing execution identity remain archived
claims with explicit gaps. Null is not a fifth verification tier.

Partial coverage and subgroup-restricted nulls never mark a whole cell empty.
Sourced existence propagates to containing strata; nonexistence propagates only
to contained strata. Mathematical status and artifact availability are separate.
Census representatives are lower bounds unless an exhaustive computation is
actually replayed. Published exact counts remain separately attributed claims.

An order gap is typed skew-Hadamard and belongs to a pinned dated source. Only
an exactly verified quadruple with a skew row can fill it. Novelty is a human
attribution decision. Submitted-by is never inferred to mean discovered-by.

## Bounds

Each entry is at most 256 KiB; a board has at most 4096 entries and 64 MiB of
entry JSON. Arbitrary-order, constant-time verification is not promised.
External programs, paths, and commands supplied in submissions are never run.
