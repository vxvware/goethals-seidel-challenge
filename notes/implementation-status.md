# Implementation status

Implemented: strict schema and exact verifier; versioned canonicalization;
full small-order census replay; raw seed archives and deduplication; scoped
row-sum proofs; receipt validation with explicit gaps; whole-board consistency;
static board and JSON data; submission/import/research CLI; trusted-base PR
validation; manifest pinning; GitHub Pages deployment; agent workflow and starters.

## Validation on 2026-10-14

- 43 pytest checks pass, including exact PAF comparisons, symmetry generators,
  census replay, receipt failures, replay registration, and submission boundaries.
- All 35,439 raw quadruples in 64 source archives verify and map to 93 distinct
  representatives. There are 71 certificate records and 14 census entries.
- Full census replay reproduces n=43 best: five equiv-v1 classes, 20 branch-v1
  orbits, and 10,080 expanded quadruples.
- Ruff and the trusted manifest pass. The full static build succeeds.
- Chromium checks pass on desktop and mobile: filters, evidence navigation,
  deep links, no JavaScript exceptions, and no page-width overflow.
- A fresh inequivalent candidate stages successfully; an equivalent candidate
  is rejected. An engine-start failure leaves a durable failed replay report.
- A temporary Git checkout passes the trusted-base submission gate with a valid
  addition. Wrong authorship and an accompanying verifier edit both fail.

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
- The plan's claims of known good matrices at 53,55,61 are not backed by the
  primary tables inspected for the baseline. They remain unresolved here
  pending a construction or an individually supporting source. Published
  general skew constructions at 61 do not establish good matrices there.

The scoped n=89 null is stronger than the old empty-array artifact: a small
exact row-sum proof is rechecked directly. It applies only to the order-11
same-subgroup family and does not close the general skew-Hadamard target.
