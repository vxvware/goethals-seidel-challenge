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

## Maintenance

Use uv. Run ruff, pytest, the full seed audit when seed import logic changes,
and a full site build. Changes to trusted files require regenerating
verify/validator_manifest.json with `uv run python -m verify.check_manifest --update`
after review, then checking it. Regenerate docs/ for maintenance changes.
Do not silently overwrite existing research results or source archives.

The tracked campaign engines are reference sources, not permission to launch
large searches. Ordinary verification needs no GPU. Only explicitly requested
campaign replay should invoke the pinned GPU adapter.
