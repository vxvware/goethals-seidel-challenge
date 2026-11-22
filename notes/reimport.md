# Repeat the initial import

The import is a maintainer operation that regenerates seed entries and archives.
Use a clean working tree and retain any later contribution files. Supply a
parent directory containing `best-matrix-campaign`, `good-matrix-campaign`, and
`skew356-campaign`, each at the source commit in
[the manifest](../baselines/seed-manifest.json).

The first two repositories are private and `vxvware/skew356-campaign` no longer exists, so only a maintainer who holds all three checkouts can repeat this step. Nothing else needs them. Every file of the private repositories that an entry or manifest record cites is archived in this repository, and [the index](../evidence/participants/private-sources.json) gives its original repository, commit, path and SHA-256 next to the path of the copy: the catalogues and fixtures under `evidence/participants/best-matrix-campaign/` and `evidence/participants/good-matrix-campaign/`, the campaign receipts and write-ups under `evidence/seeds/`. Source records link those copies at the commit pinned as `ARCHIVE` in `tools/import_seeds.py` and `tools/import_dashboard.py`, and keep the original commit and path in their `commit` and `source_path` fields. No copy of `data/sweep_p73_d9.json` from `skew356-campaign` is known; the 15,360 rows imported from it are in `evidence/seeds/skew-p73-d9.jsonl.gz`, which the audit checks row by row. To move the pinned commit, run `uv run python tools/repin_archive.py <commit>`: it rewrites the links, rebinds the registered census replays whose entries changed by the pin alone, and updates the trusted manifest.

The literal Sage tables were extracted from
[this pinned source](https://raw.githubusercontent.com/sagemath/sage/0835358cba62dc19fccdff54f32e52fded9f22b0/src/sage/combinat/matrices/hadamard_matrix.py).
Download it as `hadamard_matrix.py`. Create a text file `sage-source.txt` with
exactly these two lines:

```text
0835358cba62dc19fccdff54f32e52fded9f22b0
https://raw.githubusercontent.com/sagemath/sage/0835358cba62dc19fccdff54f32e52fded9f22b0/src/sage/combinat/matrices/hadamard_matrix.py
```

```sh
./gs import --source-root /path/to/checkouts --sage-file /path/to/hadamard_matrix.py --sage-source /path/to/sage-source.txt
uv run python -m verify.audit_seeds
uv run python -m verify.check_manifest --update
./gs build
```

The parser extracts literal numerical tables with Python's AST utilities; it
does not execute Sage code. Every extracted quadruple is independently checked.
Review all changed source identities before updating the trusted manifest.

## The vxvware/saw-best-matrix census files

The supplemental import reads the good-matrix and G-matrix sign-row files from
`vxvware/saw-best-matrix`, each pinned in `tools/import_dashboard.py` to the
commit that last changed it and to its SHA-256. Their original bytes are
bundled under `evidence/participants/saw-best-matrix/`; the source repository
is private and access to it is not needed to repeat the import. The importer checks the pinned
hashes, exactly verifies all 1,228 quadruples, writes empty census files as
unreplayed-census certificates, and refuses to replace different records.

```sh
uv run python tools/import_dashboard.py --source-dir evidence/participants/saw-best-matrix
uv run python -m verify.audit_seeds
```

This writes the separate participant seed manifest. The audit and site combine
both inventories, deduplicating representatives across them. Update the trusted
manifest and run a full build after reviewing any intentional import changes.
