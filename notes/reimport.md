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
