# Goethals–Seidel Challenge — Plan

*Drafted 2026-09-24; revised 2026-10-14 after [plan_review.md](plan_review.md)
(all six review findings accepted — the revision notes below mark what changed).
Target repo: `vxvware/goethals-seidel-challenge`. Modeled on `qldpc-challenge`
(see `~/Codes/vxvware/qldpc-challenge-work/qldpc-challenge/`), adapted to a
domain where verification is exact, integer, and millisecond-cheap.*

## 1. What the board is

A public, automatically verified leaderboard for **Goethals–Seidel quadruples**:
four circulant ±1 matrices (A,B,C,D) of odd order n with
`AAᵀ+BBᵀ+CCᵀ+DDᵀ = 4nI`. Every named family in this corner of design theory is
a symmetry stratum of this one object:

| pattern | family | yields |
|---|---|---|
| ssss | Williamson | Hadamard 4n |
| ksss | good matrices | skew-Hadamard 4n |
| kkks | best matrices | skew-Hadamard 4n (forces n = r²+r+1) |
| kkss | (unnamed, barely studied) | skew-Hadamard 4n |
| k··· | skew GS difference family | skew-Hadamard 4n (this cracked order 292) |
| ···· | general GS family / 4-SDS | Hadamard 4n |

Submit a quadruple; the verifier recomputes everything from the four rows; if
it holds up it goes on the board. **The flagship bounty cell:** any verified
k···-or-stronger quadruple at n=89 constructs the first skew-Hadamard matrix of
order 356 — the smallest open order. The verifier flags
"fills-a-baseline-gap" automatically; the flag is **typed and dated** (review
§6): it requires the derived **skew predicate** (≥1 skew block — a general ····
quadruple at a gap order gives only an ordinary Hadamard matrix, which is
already known at every gap order), and it names the pinned baseline (the
**Cati–Pasechnik** database, arXiv:2411.18897, snapshot version + date; the
292 construction itself is Đoković's). "Fills a baseline gap" is a board flag;
confirmed present-day novelty/discovery credit is a human review step layered
on top, never auto-granted from a historical list.

Why this domain fits a trustless leaderboard even better than qLDPC: there is
no witness asymmetry. The object *is* the certificate — exact integer PAF
verification of a claimed quadruple takes milliseconds at any order, so CI can
re-verify the entire board on every PR.

## 2. Prior art to absorb (all in-house)

- **Vincent's `exact-census` branch of `saw-best-matrix`** — the direct
  ancestor of this idea: `census.py`/`census.c` (exact equivalence-class census
  of best quadruples with the full reduction set: row-sum sign, decimation
  orbits, S₃ block ordering, complement pairing), `verify.py` (exact quadruple
  verifier), `check_c.py` (expand-and-verify under the symmetry group), plus
  the SAW heuristic kernels (`saw_fast.py`, record-energy ladder) as a
  research starter kit. Counts to seed the census track: e.g. 10,080
  quadruples / 20 orbits at n=43.
- **best-matrix-campaign** (vxvware GitHub): n=73 nonexistence with the
  enumeration audit — the template for receipt-backed nonexistence entries.
- **good-matrix-campaign** (vxvware GitHub): n=65 nonexistence, the coverage
  receipt discipline (per-pass cursors, ordered-class-list sha256 digests,
  fail-closed pending), catalogues (all 10,080 good matrices n≤15, Russo
  n=15/21), engines whose `--pass-sB/--start/--dmax` make any slice of a
  nonexistence claim independently recomputable.
- **skew356-campaign**: cyclotomic sweep tool + tier ladder; its tier-0 empty
  receipt at p=89 is a seed entry for the evidence track.
- **qldpc-challenge**: the whole operational design — schema'd one-file-per-entry
  submissions, computed (not self-declared) track membership, scope-separation
  CI, pinned validator manifest, authorship binding, duplicate fingerprints,
  stats.json + GitHub Pages board, `./qldpc` CLI, AGENTS.md / AUTORESEARCH.md
  agent workflows.

## 3. Entry types and their verification tiers

Three entry types. **Verification level is a four-tier ladder (review §3) and
the board always displays which checks actually ran** — a sampled claim is
never promoted to a stronger tier:

- `proven` — independently checked mathematical proof (e.g. best matrices at
  n ≠ r²+r+1, by the row-sum theorem).
- `reproduced` — the full computation re-run end-to-end by CI or a second
  independent engine (cheap orders: good-41 re-exhausts in minutes).
- `spot-checked` — receipts validated AND K random slices re-settled by the
  pinned engine. Honest limits stated on the board: uniform sampling of m of
  N classes catches a single bad class with probability m/N, and re-running
  the same engine preserves its systematic bugs. This is confidence, not
  certification.
- `receipt-backed` — machine receipts validated (digests, tiling intervals,
  pending == []) but no recomputation.

Entry types:

1. **Existence** (`quadruples/`): the four first rows (or index sets), order,
   group. Verifier recomputes: the Gram/PAF identity (exact integers), the
   symmetry pattern (derived, not declared — you cannot claim "best" without
   being kkks), row sums, canonical equivalence fingerprint (dedup), and the
   typed baseline-gap flag (§1). **Fully trustless — every existence entry is
   tier `reproduced` by construction; CI re-verifies the whole board per PR.**
2. **Nonexistence certificates** (`certificates/`): a claim "(n, stratum) is
   empty" — or, with a mandatory scope field, "empty within family F" (stated
   precisely, e.g. "blocks = coset unions of the order-11 subgroup of Z₈₉*",
   not "cyclotomic route") or "empty on interval [a,b) of the pinned class
   order" (partial coverage, campaign-in-progress) — backed by machine
   receipts: engine commit hash, per-pass coverage intervals tiling
   [0, #classes), ordered-class-list digest, pending == [], and a pinned
   re-derivation of #classes. CI recomputes the class count + digest and
   re-settles K random slices (`--start i --dmax i+k`): that yields tier
   `spot-checked`, no higher. Seeds and their honest tiers: best-73 and
   good-65 (`spot-checked`, with best-73's documented legacy-provenance gap
   of 104,500 classes recorded in the entry), good-41 (`reproduced`),
   literature nonexistence as `provenance: literature` with individually
   sourced orders — **good {41, 47, 49, 51, 59, 63, 69}** (NOT a range:
   good matrices exist at 43, 45, 53, 55, 57, 61 — review §1), Williamson
   {35, 47, 53, 59, …}, best at non-r²+r+1 orders (`proven`).
3. **Census** (`census/`): count of equivalence classes at (n, stratum) under
   the **versioned equivalence convention** (§4). Tier `reproduced` where CI
   re-runs the pinned census (best ≤ ~45, good ≤ ~21); above that, verified
   inequivalent representatives establish a **lower bound** only, and the
   board says "≥ N (lower bound)" rather than "N". Seed: Vincent's counts +
   our catalogue counts, re-expressed in the primary convention.

A cross-consistency validator runs over the whole board in CI: an existence
entry and a nonexistence certificate for the same (n, stratum, scope) is a
hard failure (this check alone would have caught the "good 47–63" error).

## 4. Board structure (computed, like qLDPC's tracks)

Primary grid: **stratum × group class**, membership derived by the verifier.

- Stratum axis: `williamson (ssss)`, `good (ksss)`, `best (kkks)`, `kkss`,
  `skew-sds (k···)`, `general (····)`. These nest downward (a best quadruple
  is also a skew-sds and a general entry), exactly like qLDPC's nesting weight
  classes — a ksss entry auto-competes in k··· and ····.
- Group axis: `cyclic` (v1) with the schema reserving `abelian` (group given
  as invariant factors) for v2. At prime n they coincide; the schema should
  not need migration when v2 lands.

Within a cell the board is not a Pareto frontier (one scalar dominates): it is
a **completeness map over odd n**, with **mathematical status and artifact
availability as separate axes** (review §5):

- Mathematical status (from sourced baseline + verified entries + certificates):
  `exists` / `empty` (with tier) / `unresolved`.
- Artifact status: `verified-on-board` / `published-not-imported` / `none`.

So a published construction whose rows haven't been imported shows as
`exists / published-not-imported` — an **import task**, never a research
target, and its importer gets *importer* credit, never discovery credit.
Attribution is two separate fields: `discovered_by` (original source, person or
paper) and `submitted_by` (GitHub handle on the accepted PR). Research targets
are generated only from `unresolved` cells; import tasks from
`published-not-imported` ones. Headline sortables: largest verified n per
cell, unresolved cells settled, typed baseline-gap flags.

**Equivalence convention — decided and versioned before any fingerprint exists
(review §2).** Two conventions differ materially: Vincent's census branch
(simultaneous decimation + skew-block permutation + D-negation) gives **20
orbits at n=43**; the published classification (adds independent reversals of
the skew rows) gives **5 classes** (10,080 raw quadruples both ways; verified
against our catalogue's 2,520 stored n=43 rows × 4 reversal-classes). The
board's **primary convention is the published one** (field credibility and
literature comparability win), with the branch-convention orbit count kept as
a secondary census statistic (Vincent's reductions are what make large-order
censuses computable). The generator set gets an explicit version tag
(`equiv-v1`) in the schema; changing it is a schema migration, not a code
tweak. Open sub-decision to settle in M-1 with a worked note: whether
independent cyclic shifts (PAF-preserving but symmetry-pattern-changing) are
in the group — proposal: normalize shifts away in the canonical form but
derive the displayed stratum from the shift-normalized representative.

Duplicate detection: canonical form under the versioned equivalence group —
exact and decidable by finite search at every order, so unlike qLDPC's
WL-signature heuristic, equivalence here is a hard verdict, not a review flag.

## 5. Repo layout (mirroring qldpc-challenge where it earned its keep)

```
goethals-seidel-challenge/
  README.md               what/why/how; badges off stats.json
  TRACKS.md               the grid, entry types, tier semantics, reference bars
  CONTRIBUTING.md         ./gs submit walkthrough; LLM contribution prompt
  AGENTS.md               agent operating rules (stage-only vs authorized-PR)
  schema/
    entry.schema.json     quadruple | certificate | census (one JSON per entry)
    SCHEMA.md
  verify/
    gs_verify.py          exact PAF + derived stratum + fingerprint + flags
    equivalence.py        canonical form under the full symmetry group
    check_submission_scope.py, check_authorship.py, validator_manifest.json
    recheck_slices.py     certificate spot-checker (pinned engine, K random slices)
  quadruples/  certificates/  census/        one JSON file per entry
  site/build.py -> docs/  static board + stats.json (badges, open-order banner)
  research/
    AUTORESEARCH.md       the search loop manual for agents
    starter/              SAW kernels (from exact-census branch), cyclotomic_sweep,
                          pointers to the GPU engines in the campaign repos
  tools/gs                CLI: verify | submit | targets | census | recent
```

## 6. Seeding plan (launch with a full, honest board)

**Hard rule: every matrix and every null result we have already collected goes
on the board at launch.** The concrete manifest of what we hold and where each
artifact lands:

### Existence entries (quadruples/, dedup via equivalence.py)

| artifact | location | board entries |
|---|---|---|
| Full best-matrix catalogue: **9,848 solutions, orders 3–57** | `best-matrix-campaign/data/catalogue_solutions.npz` | kkks cells 3–57 (orbit representatives; the rest feed census) |
| All good matrices n ≤ 15: **10,080 solutions** | `good-matrix-campaign/benchmark/good_matrices_n{3..15}.npz` | ksss cells 3–15 |
| Russo good matrices n=15, 21 | `good_solution_n{15,21}.npy` | ksss cells (provenance: Russo) |
| n=13 full verified solution set (52) from the `--all` gates | `good_all_n13.npy`, `good_all_cpu_n13.npy` | ksss 13 census cross-check |
| **Our p=73 cyclotomic sweep output: 15,360 raw skew families** (incl. the published 292 family) | `skew356-campaign/data/sweep_p73_d9.json` | k··· cell 73 — dedup to inequivalent orbits; ours independently re-derived, provenance both |
| Literature quadruples (Williamson known orders, order-57 best triples, 292 family as published, …) | citations + published rows | provenance-tagged entries per cell |

### Nonexistence certificates (certificates/)

| artifact | location | entry |
|---|---|---|
| **best kkks n=73 empty** — full receipt chain incl. the float64 enumeration audit | `best-matrix-campaign` (docs/enum_audit, campaign_state, writeup) | worked example #1, tier `spot-checked`; entry must record the documented **104,500-class legacy-provenance gap** (audit's open item) rather than hide it |
| **good ksss n=65 empty** — two-pass coverage receipts, digest-matched seam | `good-matrix-campaign/docs/n65_nonexistence.md` + `data/box_*/coverage_n65_sB*.json` | worked example #2, tier `spot-checked` (defines the two-box union format) |
| good ksss n=41 empty — reproduced in minutes by every engine gate | `coverage_n41_sB9.json` etc. | tier `reproduced` exemplar (CI re-runs it ENTIRELY) |
| **good ksss n=67, pass σ_B=−13 prefix [0, 927,600) of 2,062,646** — in progress | `data/box_192.222.50.112/progress_complete_n67_sB-13.json` | partial-coverage entry; **import audit: receipt lacks the engine commit hash the schema requires — backfill from the git log at import** |
| skew356 tier-0 null at p=89 | `skew356-campaign/data/sweep_p89.json` | scoped null; **import audit: the file is a bare `[]` — the entry must carry the precise scope ("all four blocks = coset unions of the order-11 subgroup of Z₈₉*, optionally +{0}, X0 skew") and the sweep parameters, reconstructed from `cyclotomic_sweep.py` + its log** |
| Literature nonexistence — **individually sourced orders, never ranges** (review §1): good {47, 49, 51, 59, 63, 69} (41 is ours/`reproduced`); Williamson {35, 47, 53, 59, …}; best at non-r²+r+1 orders (`proven`, by the row-sum theorem) | citations per order | `provenance: literature` / tier `proven` entries; the CI cross-consistency validator guards against existence/nonexistence conflicts |

### Census entries (census/)

| artifact | counts |
|---|---|
| Vincent's exact-census branch (best quadruples ≤ 43) | e.g. 10,080 quadruples / 20 orbits at n=43; per-order table from the branch |
| Our good-matrix catalogue counts (≤ 15) | 48/96/528/288/1440/3456/4224 per order, orbit counts via equivalence.py |
| Our n=13 `--all` cross-validated count (52 per B-organization) | census cross-check fixture |

Two schema consequences this inventory forces (now reflected in §3): the
certificate type needs a **partial-coverage state** (n=67 today) and a
**family-restricted scope field** (the p=89 cyclotomic null is "empty within
cyclotomic family", not "empty") — without these, honest seeding of what we
hold is impossible.

- Targets page auto-computed: every open (n, stratum) cell with n ≤ 131, the
  open-Hadamard flags on 89/101/107 …, and "what it takes" notes (e.g. best
  requires n = r²+r+1 → next candidate 91).

## 7. Trust model (inherit qldpc-challenge wholesale, simplified)

Scope separation (entry PRs cannot touch verifier/schema/site), pinned
validator manifest with drift-fails-CI, authorship binding by GitHub handle,
prose gate for evidence paths. Simplifications earned by the domain: no
witness tier split (existence verification is total), equivalence is exact
(no flag-and-review), no circuit tier. The certificate spot-checker is the one
genuinely new verification component to design carefully (sampling policy,
pinned engine provenance, how a failed spot-check demotes an entry).

## 8. Milestones (resequenced per the review: contract first, certificates
before any publish — the §6 "every null result at launch" rule demands it)

- **M-1 — mathematical contract + artifact audit** (1 session, no code):
  (a) settle and write down the versioned equivalence convention (`equiv-v1`,
  published convention primary; the cyclic-shift sub-decision with a worked
  note); (b) build the sourced order-status baseline per stratum (existence
  AND nonexistence, individually cited — no ranges); (c) define the
  verification-tier semantics exactly as §3; (d) audit every §6 seed artifact
  against the submission contract and list each gap with its repair
  (best-73 provenance note, n=67 commit backfill, p=89 scope reconstruction).
- **M0 — skeleton + core verifier** (1 session): schema (incl. tier, scope,
  partial-coverage, discovered_by/submitted_by fields), `gs_verify.py` +
  `equivalence.py`. Gates: 10k+ known quadruples verify, mutated ones fail;
  n=43 collapses to **5 classes under equiv-v1 (published)** AND 20 orbits
  under the secondary branch convention — both counts exact.
- **M1 — certificate tier + artifact import** (1 session): receipt schema,
  `recheck_slices.py` against the pinned good-matrix engine, the M-1 audit
  repairs executed, best-73/good-65/good-41/n=67-partial/p=89-scoped wired in
  at their honest tiers.
- **M2 — seeds + site** (1 session): all existence/census seeds imported and
  deduped under equiv-v1, `site/build.py`, stats.json, completeness map with
  the status/artifact axes, typed baseline-gap banner.
- **M3 — CLI + CI + publish** (1 session): `./gs submit|targets|verify|import`,
  GitHub Actions (verify-all + cross-consistency + scope + authorship +
  manifest), publish to `vxvware/goethals-seidel-challenge` (private first,
  public once the board renders the full seeded state).
- **M4 — agent workflow** (1 session): AGENTS.md + AUTORESEARCH.md + starter
  kit (SAW + cyclotomic sweep + GPU engine pointers), ready-to-paste LLM prompt.

## 9. Open questions for Trevor

1. Public from day one, or private until M3 seeding is done (recommended)?
2. Bounty/recognition for the 356 cell (and the other open-Hadamard cells)?
3. v1 scope: cyclic-only (recommended) or abelian groups immediately?
4. Confirm the equivalence call in §4: published convention primary,
   Vincent's branch convention as a secondary census statistic. (Review §2
   left the choice open; the plan now takes a position.)
5. CLI name: `./gs` (clean) vs `./gsq` vs `./goethals`.

(Resolved since the first draft: n=67 DOES land at launch — as a
partial-coverage certificate entry, which the schema now supports.)
