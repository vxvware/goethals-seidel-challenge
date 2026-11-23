# Review of the Goethals–Seidel Challenge plan

Saved 2026-10-14 from the review in this conversation. Reviewed document:
[plan.md](plan.md), drafted 2026-09-24. Line references below refer to that
version of the plan.

The core idea is sound, but these six issues should be resolved before
implementation. The review checked local campaign artifacts and the cited
literature, and recomputed catalogue equivalence counts. It did not rerun the
large campaign searches.

## 1. [P1] The nonexistence seed list contains a false claim

**Location:** `plan.md:165` (also `plan.md:80`).

The plan lists “good 47–63,” but good matrices **exist at n=57**; the published
paper supplies their rows. The local n=65 writeup instead lists the previous
nonexistence orders as `{41,47,49,51,59,63,69}`.

Replace ranges with individually sourced orders, and make conflicting
existence/nonexistence entries fail validation.

Sources:

- [Published n=57 construction](https://arxiv.org/html/1811.05094), Table 1 and Figure 1.
- [Local n=65 writeup](../good-matrix-campaign/docs/n65_nonexistence.md).

## 2. [P1] The “full equivalence” promise conflicts with the mandatory census gate

**Locations:** `plan.md:108`, `plan.md:197`.

The `saw-best-matrix` repository's `origin/exact-census` branch counts **20
orbits at n=43** under simultaneous decimation, skew-block permutations, and
the sign of D. Its orbit-counting code omits independent reversals of the
skew rows, which the published classification includes.

The distinction was reproduced locally:

**10,080 quadruples → 20 branch-convention orbits → 5 published-convention classes.**

The local `best-matrix-campaign/data/catalogue_solutions.npz` contains 2,520
n=43 rows. Expanding them under independent reversals gives 10,080 distinct
quadruples. Those form 20 orbits under the branch convention and five classes
when independent reversals are included, matching the published count.

Define and version the equivalence generators before building fingerprints
or census acceptance gates. Either retain the narrower convention and label
it explicitly, or use the published convention and change the expected count
to five. General GS equivalence also needs an explicit decision about
independent cyclic shifts, which preserve PAF but can change the displayed
symmetry pattern.

Sources:

- [Published equivalence operations and counts](https://arxiv.org/html/1907.04987v2#S5.SS2), Sections 5.2 and 6.
- `../saw-best-matrix`, branch `origin/exact-census`: `test_census.py`, `check_c.py`, and `census.py`.
- [Local best-matrix catalogue](../best-matrix-campaign/data/catalogue_solutions.npz).

## 3. [P1] Spot checks cannot support the proposed certification language

**Locations:** `plan.md:70`, `plan.md:102`.

The plan calls the process “spot-check trustlessness,” while the board uses
`empty-certified`. Coverage receipts establish what an engine reports;
checking a few slices cannot establish that every slice is empty. If only
one of N classes contains a missed solution, uniformly sampling m distinct
classes detects it with probability m/N. Reusing the same engine also
preserves systematic bugs.

Use distinct statuses for:

- Receipt-backed claims.
- Spot-checked computations.
- Fully reproduced searches.
- Independently checked proofs.

Apply the same distinction to census completeness: verified inequivalent
representatives alone establish a lower bound. The UI and schema should
record which checks actually ran and avoid promoting a sampled claim to a
stronger verification tier.

## 4. [P2] Several launch artifacts do not satisfy the planned receipt schema

**Location:** `plan.md:160` and the surrounding certificate inventory.

The best-73 audit explicitly records **104,500 classes needing provenance
repair or reruns** for a fully identified coverage union. Meanwhile,
`skew356-campaign/data/sweep_p89.json` contains only `[]`, and the n=67 progress
JSON lacks the engine commit required by the proposed schema.

Add an artifact-import milestone that checks every required field and
records unresolved gaps. These observations identify evidence-format gaps;
they do not establish that the underlying mathematical results are wrong.

For p=89, encode the precise scope—blocks invariant under the same order-11
subgroup—rather than the broader phrase “cyclotomic route.” The empty array
alone does not identify the search scope or establish exhaustive completion.

Sources:

- [Documented best-73 receipt gaps](../best-matrix-campaign/docs/enum_audit_2026-09-21.md), “Remaining (lower-priority) audit items from the review.”
- [p=89 sweep output](../skew356-campaign/data/sweep_p89.json).
- [n=67 progress receipt](../good-matrix-campaign/data/box_192.222.50.112/progress_complete_n67_sB-13.json).

## 5. [P2] An unfilled board cell is not necessarily an open mathematical problem

**Locations:** `plan.md:99`, `plan.md:181`.

The proposed states conflate missing submissions with unresolved existence.
A published construction whose rows have not been imported would appear
open, and its importer could receive “first-finder” credit.

Separate mathematical status from artifact availability, and separate
original discovery attribution from first accepted submission. Generate
research targets from sourced unresolved cases; generate import tasks from
known cases missing verified artifacts.

## 6. [P2] The breakthrough flag needs a typed, dated baseline

**Location:** `plan.md:24`.

The listed gaps concern **skew-Hadamard** matrices; ordinary Hadamard
matrices at those orders are already known. A general GS submission must
not receive that flag merely because its order matches. Matching a pinned
historical gap list also cannot automatically establish present-day novelty.

Require the skew predicate and record the baseline version and date.
Distinguish “fills a baseline gap” from confirmed discovery credit. The
cited database is by **Cati–Pasechnik**, with Đoković contributing the
order-292 construction.

Source: [A database of constructions of Hadamard matrices](https://arxiv.org/html/2411.18897v2).

## Milestone sequencing

Add a mathematical-contract and artifact-audit milestone before M0. It
should settle the equivalence convention, source the order-status baseline,
define verification-tier semantics, and identify which seed artifacts meet
the submission contract.

Move certificate support before public launch. Publishing at M2 currently
conflicts with the requirement to include every null result, because
certificate support arrives at M3.

## Verification scope

The review inspected the plan, local campaign documentation and artifacts,
the `origin/exact-census` branch, and the cited papers. It independently
recomputed the equivalence-count distinction described above. It did not
rerun the best-73 or good-65 searches or fully reproduce their coverage
audits. No plan or campaign files were changed as part of the review.
