# Implementation status

Implemented: strict schema and exact verifier; versioned canonicalization;
full small-order census replay; raw seed archives and deduplication; scoped
row-sum proofs; receipt validation with explicit gaps; whole-board consistency;
static board and JSON data; submission/import/research CLI; trusted-base PR
validation; manifest pinning; GitHub Pages deployment; agent workflow and starters.

## Validation on 2026-10-15

- 50 pytest checks pass, including exact PAF comparisons, symmetry generators,
  census replay, receipt failures, replay registration, and submission boundaries.
- All 36,655 raw quadruples in 89 source archives verify and map to 216 distinct
  representatives. There are 72 certificate records and 39 census entries.
- Full census replay reproduces n=43 best: five equiv-v1 classes, 20 branch-v1
  orbits, and 10,080 expanded quadruples.
- Ruff and the trusted manifest pass. The full static build succeeds.
- Chromium checks pass on desktop and mobile: filters, evidence navigation,
  deep links, no JavaScript exceptions, and no page-width overflow.
- A fresh inequivalent candidate stages successfully; an equivalent candidate
  is rejected. An engine-start failure leaves a durable failed replay report.
- A temporary Git checkout passes the trusted-base submission gate with a valid
  addition. Wrong authorship and an accompanying verifier edit both fail.

## Participant attribution

A curated registry distinguishes participant results, independent reproductions,
and published catalogue imports. Participant work is credited to vxvware. The
G-matrix census at n=49 that contradicts the published nonexistence, the
good-53/55/61 constructions, the G-matrix censuses at n=51 through 59, the
best-73 campaign, and the skew-73 families are featured on the board.
Map badges, an Origin filter, named credits, and JSON output preserve that
distinction independently of publication and verification status.

## Evidence that remains incomplete

The implementation does not invent runs or provenance to satisfy a planned tier.

- No CUDA GPU is available on the implementation machine. Good-41 is currently
  receipt-backed, not reproduced. No campaign is labeled spot-checked merely
  because the replay adapter exists.
- The archived good-65 prefix receipt is actually `[454800,517400)`. Combined
  with the later receipt it leaves `[0,454800)` uncovered by the retained
  modern JSON records. Older logs record earlier work but lack matching
  execution/digest identities; the original empty conclusion is an archived
  claim. Repair requires stronger retained records or replay.
- Best-73 keeps its documented 104,500-class legacy identity gap and requires
  an independent heterogeneous-deck audit for a stronger tier.
- Good-67 remains a partial result. Git history pins the archive snapshot; it
  cannot retroactively create a missing execution-commit field.
- Good n=53, 55, and 61 and the G-matrix orders through 63 are verified from
  the vxvware/saw-best-matrix import. Their census counts remain lower bounds
  because the source's exhaustive searches have not been replayed here. The
  empty censuses at good n=59 and G-matrix n=57 are unreplayed attributed
  claims with no tier. At G-matrix n=59 the source counts two classes and
  equiv-v1 one; the discrepancy is recorded in the census notes.
- The published nonexistence of G-matrices at n=49 (arXiv:1802.00556) is
  contradicted by eight exactly verified rows. The sourced empty record was
  removed from the baseline and the contradiction is recorded in the baseline
  notes and the contribution registry.

The scoped n=89 null is stronger than the old empty-array artifact: a small
exact row-sum proof is rechecked directly. It applies only to the order-11
same-subgroup family and does not close the general skew-Hadamard target.
