# Contribute a quadruple

1. Install the environment with `uv sync --locked --group dev`.
2. Copy an existing entry as a format example. Supply four rows or four sets of
   negative indices. Choose a new ID, your GitHub handle as `submitted_by`, an
   ISO submission date, and accurate original discovery attribution and sources.
   Do not add a fingerprint, family, verdict, or tier: those are computed.
3. Run `./gs verify candidate.json`. A malformed row, failed PAF identity, or
   unsupported group is an error.
4. Run `./gs submit candidate.json`. It checks board consistency and duplicate
   equivalence and stages the entry under ignored `research/candidates/`.
5. To contribute, run `./gs submit candidate.json --write`, or use
   `./gs submit candidate.json --open-pr` from a clean working tree. The latter
   creates a branch, commits the entry, pushes to `origin`, and opens a draft PR.

A source is an HTTPS citation with a title. Local evidence references must be
repository-relative files with SHA-256 hashes. Do not use private checkout
paths or promise artifacts that are not in the submitted tree. An importer
must retain original authorship; a verified baseline-gap flag is not a novelty
claim.

## Evidence and census submissions

Follow [schema/SCHEMA.md](schema/SCHEMA.md). Add the entry and all its referenced
evidence files, then run `./gs verify` over the whole board. A literature result
needs an individual citation, and an interval claim needs its actual half-open
coverage and class-order identity. A missing field or unknown execution commit
must be reported as a gap, not invented. An incomplete archive cannot use the
strict full-coverage method.

Census representatives must already be accepted quadruples or accompany the
same PR. Counts are lower bounds until an implemented full enumerator reproduces
the complete representative set. Never infer completeness from a collection
size or from matching a published count alone.

## Contribution boundaries

An entry PR adds files under `quadruples/`, `certificates/`, `census/`,
`evidence/`, and `notes/`. It cannot edit existing entries or trusted files.
Corrections, baseline updates, equivalence migrations, and recheck registrations
are separate maintainer changes. This keeps an entry author from changing the
rules used to accept their entry. The PR author must match `submitted_by`.

Before submitting:

```sh
./gs verify
uv run pytest -q
uv run python -m verify.check_manifest
```

Do not regenerate `docs/` in an entry PR. The trusted publishing build generates
it from the accepted data after merge.

## Agent prompt

> Read AGENTS.md and research/AUTORESEARCH.md. Choose a sourced unresolved
> cyclic GS target. Search within an explicit budget. Verify every candidate
> with the trusted ./gs verifier and keep its provenance. Stage successful
> candidates; only commit, push, or open a PR when the user authorizes that
> publication. A failed search is partial evidence unless exhaustive coverage
> and its precise scope can be demonstrated.
