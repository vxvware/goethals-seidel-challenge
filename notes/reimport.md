# Repeat the initial import

The import is a maintainer operation that regenerates seed entries and archives.
Use a clean working tree and retain any later contribution files. Supply a
parent directory containing `best-matrix-campaign`, `good-matrix-campaign`, and
`skew356-campaign`, each at the source commit in
[the manifest](../baselines/seed-manifest.json).

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
bundled under `evidence/participants/saw-best-matrix/`; access to the source
repository is not needed to repeat the import. The importer checks the pinned
hashes, exactly verifies all 1,216 quadruples, writes empty census files as
unreplayed-census certificates, and refuses to replace different records.

```sh
uv run python tools/import_dashboard.py --source-dir evidence/participants/saw-best-matrix
uv run python -m verify.audit_seeds
```

This writes the separate participant seed manifest. The audit and site combine
both inventories, deduplicating representatives across them. Update the trusted
manifest and run a full build after reviewing any intentional import changes.
