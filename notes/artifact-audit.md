# Seed import audit

The importer records SHA-256, source repository commit, source path, actual
row counts, and the resulting representative IDs for every listed matrix
artifact. Raw matrices remain available in compact archives; representatives
alone populate the board. Every raw matrix is checked on import and during
the full seed-audit gate. Raw counts and equivalence-class counts are separate.

- Best catalogue: 9,848 stored rows. n=31 has 720 and n=43 has 2,520; these
  are not the branch's expanded raw census counts. Expansion is tested
  separately, including the 10,080 / 20 / 5 n=43 gate.
- Good catalogue: 10,080 stored rows at n=3,5,7,9,11,13,15. Import the
  campaign's n=15 and n=21 example rows and both n=13 CPU/GPU all-solution
  files as additional source records.
- Supplemental participant import (2026-10-15): the vxvware/saw-best-matrix
  census files, each pinned to the commit that last changed it. Good matrices
  at n=53, 55, and 61 (104, 40, and 120 raw rows) give one class per order.
  G-matrices (kkss) at every admissible odd order through 59 and the
  orbit-restricted family at 63 give 952 raw rows; the equiv-v1 class counts
  agree with the published counts at 33, 37, 41, 43, and 45 and with the
  source at every order except 59, where the source counts two classes and
  equiv-v1 one. The source claims complete censuses, but the board records
  lower bounds because those searches have not been replayed here. The empty
  census files at good n=59 and kkss n=57 are archived as unreplayed-census
  certificates. Original sign rows, gzip mappings, and hashes are retained in
  the supplemental participant inventory under
  `evidence/participants/saw-best-matrix/`. Three pinned commits (kkss 49 and
  59, good 61) are not yet on the source repository's public default branch,
  so their source URLs resolve only after that branch is pushed; the bytes
  are bundled here regardless.
- p=73: import and exactly verify all 15,360 recorded coset families. They
  yield eight equiv-v1 representatives. The published order-292 construction
  verifies but yields a ninth representative under this convention; the plan's
  asserted inclusion in the saved sweep is not reproduced. Both are retained.
- Best-73: preserve the fleet receipts, omission closure, and audit note.
  Keep the 104,500-class legacy identity gap visible. Do not invent a digest
  or call the entire result spot-checked without an actual replay.
- Good-65 / good-41: import the archived coverage and engine snapshots. Their
  old order digests are truncated SHA-256 (16 hex characters), not full hashes.
  An archived engine snapshot is not proof of the exact binary used in a run.
  The good-65 prefix JSON actually starts at 454,800, not zero. Older v1/v3/v4
  logs and the campaign writeup are preserved as supporting sources, separately
  from the modern interval union. They cannot repair its identity gap.
- Good-67: git history identifies the archived engine and receipt together at
  018ef2d7754017eca1f9f22c8ddd84b9b6cf5581. Record that binding as an archive
  snapshot; do not fabricate an execution commit. Its interval is partial.
- p=89: [] is not a receipt. Reconstruct the scope explicitly and independently
  settle it using exact row sums: subgroup size 11 forces free block row sums
  89-22u-2e, u=0..8, e∈{0,1}; none of three such squared sums adds to 355.
  The checked row-sum obstruction proves this precisely restricted null.

The revised plan calls good n=61 known, but the local source table leaves it
open. The import must use individually verified primary sources, not promote
that assertion without a source. Any unresolved citation remains visible in
this audit and never silently changes the mathematical baseline.

The implementation machine has no CUDA device. GPU replay is an explicit,
pinned-engine operation; no spot-check or full-reproduction label is awarded
on the basis of an old filename, a successful import, or a stated intention.
