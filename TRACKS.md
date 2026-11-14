# Tracks, status, and evidence

All rows satisfy the same exact periodic-autocorrelation identity. After
normalizing each origin to +1, `s` means symmetric and `k` means skew away from
zero. Membership is computed, including membership in containing strata.

| Track | Predicate | Construction |
|---|---|---|
| williamson | ssss | Hadamard order 4n |
| good | ksss | skew-Hadamard order 4n |
| best | kkks | skew-Hadamard order 4n; 4n−3 must be an odd square |
| kkss | kkss (G-matrices) | skew-Hadamard order 4n |
| skew-sds | at least one k | skew-Hadamard order 4n |
| general | any verified quadruple | Hadamard order 4n |

At n=1 the predicates overlap; all tracks contain the trivial quadruple.
Cyclic groups are supported now. Noncyclic abelian invariant factors are
reserved in the schema and fail validation until a future version implements
the required operations.

## Verification levels

Origin is a separate dimension from these levels. **Participant work** covers
constructions, searches, and claims produced by challenge participants.
**Independent reproduction** credits work on an already published result.
**Literature / catalogue** identifies imported reference results. Multiple
origins can coexist in one cell: n=73 skew-sds includes participant-generated
families and a separately attributed published construction.

“Unpublished” means no paper is recorded for the contribution as of its registry
date. It is not a proof of global priority, nor does it upgrade a claim's evidence.
The curated [contribution registry](baselines/contributions.json) preserves that
distinction even when the mathematical status remains unresolved.

| Computed label | What actually happened |
|---|---|
| proven | An implemented exact mathematical obstruction was checked. |
| reproduced | The rows were exactly verified, or the whole supported census/search was replayed. |
| spot-checked | Validated receipts plus a registered pinned-engine sample replay. |
| receipt-backed | The archived machine receipts pass identity, interval, seam, and pending-work checks. |
| no tier | Literature attribution, an unreplayed participant census, an incomplete historical claim, or a census lower bound. |

The last row is absence of a verification tier, not a fifth tier. The evidence
page lists the checks, gaps, and limitations. A successful file import is not a
computation replay. Old good-engine digests contain 16 hexadecimal characters:
they are truncated SHA-256, and are labeled accordingly.

Uniformly sampling m distinct classes from N covered classes detects a single
bad class with probability m/N. The same engine can reproduce its own bugs.
A failed replay fails validation and cannot silently retain a stronger label.
Partial and restricted results remain evidence attached to their cells; they
never imply full-stratum emptiness.

The supported independent proof rules are the best-matrix square obstruction
and the exact sum-of-squares obstruction for same-subgroup coset unions over a
prime cyclic group. Source citations alone are not machine-checked proofs.

## Census

The small good-matrix enumerator exhausts the normalized search space at odd
n≤15 using exact PAF joins, and returns the one quadruple at n=1 for every
stratum. The bundled C good census replays odd orders 3 to 43, the bundled
Williamson census replays odd orders 3 to 37, and the bundled best census
replays orders 3,7,13,21,31,43; every returned solution is independently
checked. Its spectral pruning is
floating point with conservative margins; the label is reproduced computation,
not formally checked proof. The full-replay output set must equal the submitted
representative set, rather than merely agree in cardinality.

For larger orders the board reports `≥ N` from verified inequivalent
representatives and displays any published exact count separately.

A census beyond the inline range becomes exact through a registered offline
replay: a maintainer runs a pinned engine to completion, records its raw rows
with the engine source hashes, host, date and wall time under
`evidence/census-replays/`, and registers the report in
`baselines/census_replays.json` (tools/register_census_replay.py does both).
On every load the verifier rechecks each recorded row exactly, re-derives the
equivalence classes with the board's canonical form, and requires them to equal
the submitted representatives and the claimed count; only then is the count
exact and the tier reproduced. The registered engines are the bundled
Williamson census (williamson-census-c-v1, used offline above order 37) and the
Williamson compression pipeline (williamson-compress-v1, orders n = d m): it
enumerates every symmetric sequence passing the PSD ceiling, keeps the realized
d-compressions, matches them into compressed quadruples by the length-m PAF
condition, reduces those by the multipliers of Z_m (every unit of Z_m lifts to
a unit of Z_n, so one instance per orbit lifts to every class), and lifts each
instance coset by coset with the same exact matching; and the G-matrix census
(kkss-census-c-v1, the vxvware gmat_c engine that produced every kkss census
on the board, replayed at odd orders up to 63). Registrations bind the
report to the exact entry bytes and engine sources; entry PRs cannot register
their own replays. A census with no representatives and a claimed count of
zero is a nonexistence claim; once replayed, inline or by registration, it
makes the cell empty with tier reproduced, and the cell shows 0. Primary
counts use `equiv-v1`; legacy statistics explicitly use `branch-v1`.

## Replay a campaign slice

Use a Python environment with NumPy and CuPy for the installed CUDA version:

```sh
./gs recheck certificates/good-n41-campaign.json --samples 3 --python /path/to/cuda-env/bin/python
./gs recheck certificates/good-n41-campaign.json --full --python /path/to/cuda-env/bin/python
```

The default sampling seed comes from OS randomness after the entry is fixed.
`--seed` is for reproducible investigation. Each execution uses a fresh temporary
directory, regenerates the class ordering, checks its count/digest against the
archive, and requires the requested interval to be completely settled. Timeouts,
missing receipts, solutions, or pending work fail the check. Large-order class
pool construction can require substantial RAM in addition to a GPU.

Reports are written under ignored `.gs/rechecks/`. To promote a replay to the
board, a maintainer copies a passed report into `evidence/rechecks/`, registers
its path/hash in `baselines/rechecks.json`, and updates the trusted manifest in
a separate maintenance PR. Registrations bind the report to the exact entry
bytes and bundled engine hash. Entry PRs cannot register their own verdicts.
