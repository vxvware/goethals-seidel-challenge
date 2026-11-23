# Agent guide

Implement and maintain the challenge using the mathematical contract in
notes/mathematical-contract.md. Read the relevant schema and track definitions
before changing verification or evidence semantics.

## Contributions

For unattended research, stage candidates under research/candidates/. Use the
trusted ./gs verifier. Do not commit, push, or open PRs without user authorization.
An explicit request to implement, submit, commit, push, or publish authorizes
that requested workflow; do not add a redundant approval step.

For entry contributions, keep verifier/schema/baseline/site changes separate.
Authorship is the submitted_by GitHub handle; discovery credit belongs to the
original source. Never label receipt validation or a sample as a full proof.

## Citations

Every paper, dataset or software source the board names has one entry in
baselines/references.bib, with metadata checked against the publisher, arXiv or
Zenodo record. Prose in baselines/contributions.json cites it as [@key]; a
source URL in an entry resolves to its entry through the doi, eprint, url and
urlalias fields. Do not name a work by author and year, or by a bare arXiv
number, without a key. tests/test_references.py fails on a literature link or
marker that has no entry.

## Maintenance

Use uv. Run ruff, pytest, the full seed audit when seed import logic changes,
and a full site build. Changes to trusted files require regenerating
verify/validator_manifest.json with `uv run python -m verify.check_manifest --update`
after review, then checking it. docs/ is build output and is not tracked.
Do not silently overwrite existing research results or source archives.

The tracked campaign engines are reference sources, not permission to launch
large searches. Ordinary verification needs no GPU. Only explicitly requested
campaign replay should invoke the pinned GPU adapter.
