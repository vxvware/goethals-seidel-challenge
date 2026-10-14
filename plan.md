# Goethals–Seidel Challenge — Plan

*Drafted 2026-09-24. Target repo: `vxvware/goethals-seidel-challenge`. Modeled on
`qldpc-challenge` (see `~/Codes/vxvware/qldpc-challenge-work/qldpc-challenge/`),
adapted to a domain where verification is exact, integer, and millisecond-cheap.*

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
order 356 — the smallest open order. The verifier detects and flags
"settles-open-Hadamard-order" automatically against a pinned gap list
(356, 404, 428, … from the Cati–Đoković database, arXiv:2411.18897).

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

Three entry types, with honest and *different* trust levels:

1. **Existence** (`quadruples/`): the four first rows (or index sets), order,
   group. Verifier recomputes: the Gram/PAF identity (exact integers), the
   symmetry pattern (derived, not declared — you cannot claim "best" without
   being kkks), row sums, canonical equivalence fingerprint (dedup), and
   whether the entry settles an open Hadamard order. **Fully trustless; CI
   re-verifies everything.** This is the core board.
2. **Nonexistence certificates** (`certificates/`): a claim "(n, stratum) is
   empty", backed by machine receipts: engine commit hash, per-pass coverage
   intervals tiling [0, #classes), ordered-class-list digest, pending == [],
   and a pinned re-derivation of #classes. CI cannot re-run a trillion-system
   search, but it CAN (a) recompute the class count and digest, (b) re-settle
   K random slices with the pinned engine (`--start i --dmax i+k`) and check
   they report 0 solutions — spot-check trustlessness, clearly labeled as
   such on the board. Seed: best-73, good-65 (ours), good-41 (reproduced),
   literature entries marked `provenance: literature` (good 47–63, Williamson
   35/47/53/59, …).
3. **Census** (`census/`): exact count of equivalence classes at (n, stratum).
   Verified by re-running the pinned census code when cheap (n ≤ ~45 for best;
   n ≤ ~21 for good), spot-check tier above that. Ranking: deepest exactly
   censused order per stratum. Seed: Vincent's counts + our catalogue counts.

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
a **completeness map over odd n** — each order colored
`known` (verified quadruple, with first-finder credit) / `open` /
`empty-certified` (receipt tier) / `empty-literature`. Headline sortables:
largest verified n per cell, number of open cells settled, and the
open-Hadamard-order flags. First verified entry at an (n, stratum) gets the
cell; later inequivalent quadruples at the same cell are counted (toward
census) but the cell credit is first-come, like qLDPC records.

Duplicate detection: canonical form under the full equivalence group
(decimations t ∈ Z_n*, per-block negation where the stratum allows it, block
permutations preserving the pattern, the C↔D-style swaps) — exact and
decidable by finite search at every order, so unlike qLDPC's WL-signature
heuristic, equivalence here is a hard verdict, not a review flag.

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

- Existence: Williamson/good/best known-order lists from the literature
  (provenance-tagged, each with an explicit quadruple where published — Russo
  n=15/21, the three order-57 best triples, the 292 family at n=73 k···, …)
  plus our catalogues (n ≤ 15 good: census counts + one representative per
  class as entries).
- Certificates: best-73 and good-65 with full receipt chains (the worked
  examples that define the format); good-41 as the reproduced-literature
  exemplar; literature-only entries for the rest.
- Census: Vincent's best-quadruple counts (≤ 43), our good counts (≤ 15).
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

## 8. Milestones

- **M0 — skeleton + core verifier** (1 session): schema, `gs_verify.py` +
  `equivalence.py` (port/extend campaign `verify_good.py`/`verify.py` +
  census reductions), tests against all catalogues (10k+ known quadruples
  must verify; mutated ones must fail; equivalence must collapse the n=43
  10,080 → 20 orbits exactly — Vincent's numbers as the gate).
- **M1 — seeds + site** (1 session): seed entries, `site/build.py`, stats.json,
  completeness-map rendering, open-order banner.
- **M2 — CLI + CI** (1 session): `./gs submit|targets|verify`, GitHub Actions
  (verify-all + scope + authorship + manifest), publish to
  `vxvware/goethals-seidel-challenge` (private first, public when seeded).
- **M3 — certificate tier** (1 session): receipt schema, `recheck_slices.py`
  against the pinned good-matrix engine, wire best-73/good-65 in.
- **M4 — agent workflow** (1 session): AGENTS.md + AUTORESEARCH.md + starter
  kit (SAW + cyclotomic sweep + GPU engine pointers), ready-to-paste LLM prompt.

## 9. Open questions for Trevor

1. Public from day one, or private until M2 seeding is done (recommended)?
2. Bounty/recognition for the 356 cell (and the other open-Hadamard cells)?
3. v1 scope: cyclic-only (recommended) or abelian groups immediately?
4. Should n=67 (our campaign, in progress at 45% of pass 1) land on the board
   as the first *live* certificate-in-progress entry — a nice launch story?
5. CLI name: `./gs` (clean) vs `./gsq` vs `./goethals`.
