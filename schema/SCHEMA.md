# Submission schema v1

The canonical machine-readable contract is [entry.schema.json](entry.schema.json).
Unknown properties are rejected. Common fields are `schema_version`, `id`,
`type`, `n`, `group`, `submitted_by`, `submitted_at`, `discovered_by`, and `sources`.
The filename must be `<id>.json` in the directory matching its type. JSON
booleans and floats are never accepted as row entries; duplicate object keys
and duplicate block indices are rejected.

`group` currently must equal `{"kind":"cyclic","invariant_factors":[n]}`.
The schema reserves `abelian`, but the v1 semantic validator rejects it.

## Quadruple

`type: quadruple`, `equivalence: equiv-v1`, and exactly one of:

- `rows`: four arrays of n integer signs.
- `blocks`: four arrays of distinct indices in `[0,n)`, specifying negative entries.

Derived outputs are not submission fields. The verifier computes track
membership, exact canonical rows and fingerprint, normalized row sums, and
baseline-gap flags.

## Certificate

Fields: `stratum`, `claim: empty|partial`, `scope`, `method`, `records`, `gaps`,
and an `engine` snapshot for campaign records. Scope is either `full` or
`same-subgroup-coset-unions` with explicit subgroup order, zero-inclusion
policy, and one required skew block.

Each record has a repository-local `path` under `evidence/` and a full `sha256`.
The good-coverage adapter reads order, row-sum pass, class count, truncated
order digest, interval, pending work, and unsettled count from the original
receipt. It requires every expected pass and an exact interval tiling for a
strict emptiness claim. It rejects overlaps and mismatched seam identities.

`archived-good-coverage-v1` preserves a historical full claim whose surviving
records are incomplete. It receives no computed tier. `best-coverage-v1`
preserves the heterogeneous fleet archive and its documented provenance gaps;
it also receives no full-coverage tier. The result records the checks actually
performed. A pinned archive snapshot is explicitly different from a recorded
execution commit.

## Census

Fields: `stratum`, `equivalence`, `representatives` (entry IDs), `claimed_count`,
and `claim_kind`. An optional `enumerator` selects a built-in bounded replay;
submissions cannot provide executable code. `secondary` stores clearly labeled
legacy/stored-row statistics, which are not a replacement for primary counts.

`computed-exact` is accepted as exact only after the chosen full enumerator
runs and its complete canonical set matches the submitted representatives.
Quick mode intentionally downgrades this to a lower bound. `published-exact`
retains the attributed count but does not promote it to locally reproduced.
