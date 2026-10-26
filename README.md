# Goethals–Seidel Challenge

![Verified quadruples](https://vxvware.github.io/goethals-seidel-challenge/badges/quadruples.svg)
![Census records](https://vxvware.github.io/goethals-seidel-challenge/badges/censuses.svg)

An exactly verified catalogue of four circulant ±1 rows satisfying
`AAᵀ + BBᵀ + CCᵀ + DDᵀ = 4nI`, with research targets, equivalence-class
censuses, and scoped computational evidence.

[Browse the board](https://vxvware.github.io/goethals-seidel-challenge/) ·
[Track definitions](TRACKS.md) · [Contribute](CONTRIBUTING.md) ·
[Mathematical contract](notes/mathematical-contract.md)

The flagship historical gap is a skew-Hadamard matrix of order **356**, obtained
from a skew GS quadruple at n=89. The baseline is
[Cati–Pasechnik, arXiv:2411.18897v2](https://arxiv.org/html/2411.18897v2), dated
2025-08-30. A baseline-gap flag never automatically establishes current novelty.

## Run locally

Python 3.11+, [uv](https://docs.astral.sh/uv/), and GCC with OpenMP are required
for the full census checks. No GPU is needed for the board or its ordinary CI.

```sh
uv sync --locked --group dev
./gs verify                       # exact board checks + supported census replays
./gs targets --stratum good        # sourced unresolved cells, JSON
./gs import-tasks                  # known constructions awaiting imported rows
./gs census                       # class counts and verification bounds
./gs recent --limit 5
./gs build                        # validated, deterministic output in docs/
python3 -m http.server 8769 --directory docs
```

`./gs verify --quick` and `./gs build --quick` skip census replay and report those
counts as lower bounds. Never publish a quick build as a fully replayed census.
The committed site is generated with the full build.

## What is checked

Every existence entry is validated from its rows with Python integer bit
operations. Family membership, row sums, equivalence fingerprints, and typed
baseline-gap flags are computed. The v1 group is cyclic; odd orders 1 through
511 are accepted, and the map covers orders through 131.

The primary `equiv-v1` convention includes independent reversals, signs, block
permutations, and simultaneous cyclic-group automorphisms. It excludes
independent shifts. At n=43, 10,080 best quadruples yield five primary classes
and 20 legacy `branch-v1` orbits. This is equivalence of quadruples under the
specified operations, not an assertion of full Hadamard-matrix isomorphism.

Mathematical status (`exists`, `empty`, `unresolved`) and artifact availability
(`verified-on-board`, `published-not-imported`, `none`) are separate. A known
construction without imported rows is an import task. Its importer is never
credited as the original discoverer.

The board also distinguishes **participant work**, **independent reproductions**,
and **literature/catalogue imports**. Participant work is credited to the
vxvware organisation. The participant section highlights the G-matrix census at
n=49, which finds a class where the published classification reports none, the
good matrices at n=53, 55, and 61, the G-matrix censuses at n=51 through 59,
and the n=73 campaigns. Purple map marks and the Origin filter identify participant work; entry pages
show the credited people separately from the importer. Publication status and
verification remain independent, so an unpublished claim can be credited while
its evidence gaps remain visible.

## Seed data and evidence

The inventories are [the original seeds](baselines/seed-manifest.json) and
[later participant constructions](baselines/participant-seed-manifest.json).
All 36,655 raw imported quadruples remain in compact archives under `evidence/`, with
source commits, hashes, counts, and representative mappings. The explicit plan
inventory is imported, including best/good catalogues, the campaign's n=15 and
n=21 examples, the n=13 CPU/GPU fixtures, 15,360 p=73 families, and published
Williamson and good rows. The 1,228 rows imported from
[vxvware/saw-best-matrix](https://github.com/vxvware/saw-best-matrix) supply
one verified good-matrix class at each of n=53, 55, and 61 and the G-matrix
(kkss) classes at every admissible odd order through 61, plus an
orbit-restricted family at 63. The source reports complete censuses; the board
records lower bounds because those searches have not been replayed here. The
empty censuses at good n=59 and kkss n=57 are attributed claims without a tier.

```sh
uv run python -m verify.audit_seeds
uv run pytest -q
uv run ruff check verify tools site tests
uv run python -m verify.check_manifest
```

The board does not inflate historical receipts into proofs. See
[the artifact audit](notes/artifact-audit.md) and
[implementation status](notes/implementation-status.md):

- Best-73 retains its documented 104,500-class legacy identity gap.
- Good-65 retains a newly detected missing prefix in the available machine
  receipt chain. The reported campaign conclusion is preserved as an archived
  claim; the board does not present it as fully verified emptiness.
- Good-67 is partial coverage. Good-41 has a full archived coverage receipt,
  but has not been replayed on this CPU-only implementation machine.
- The p=89 order-11-subgroup null has an independently checked, exact row-sum
  obstruction. It does not close the full n=89 skew-sds cell.
- Literature nonexistence is sourced per order, separately from local proof
  or computation tiers. Published census totals remain attributed claims
  unless the full enumerator is replayed.

GPU slice replay is available through `./gs recheck`; it uses a bundled pinned
engine and requires an appropriate CUDA environment. No arbitrary submission
command or downloaded engine is executed. See [TRACKS.md](TRACKS.md).

To repeat the original import, check out the campaign repositories at the
commits recorded in the seed manifest. Download the Sage table from the pinned
source in [the import instructions](notes/reimport.md), then use `./gs import`.
Normal board checks use the bundled archives and require no sibling checkouts.

## Repository

`quadruples/`, `certificates/`, and `census/` hold one JSON entry per file.
`verify/` and `schema/` define the trusted validator. `baselines/` holds sourced
status, seed inventories, and reviewed replay registrations. `site/` builds the
static GitHub Pages board. `research/` supplies agent instructions and starters.

Entry PRs are append-only and cannot change the verifier, schemas, baseline,
site, or workflows. A trusted-base CI job binds `submitted_by` to the PR author,
checks scope, and validates candidate data with the base verifier. Changes to
the trusted stack require a separate maintainer PR and a reviewed manifest
update.
