# Running and maintaining the board

Detail that used to live in the README: the full command list, what the verifier checks, the baseline of open skew orders, the seed imports and the evidence that is still incomplete. The sections on seed data and incomplete evidence are carried over from the README as it stood on 2026-11-23; the cell pages of the site are the current statement for any one cell. Track definitions are in [TRACKS.md](../TRACKS.md) and the definitions the verifier implements are in [the mathematical contract](mathematical-contract.md).

## Commands

Python 3.11+, [uv](https://docs.astral.sh/uv/), and GCC with OpenMP are required for the full census checks. No GPU is needed for the board or its ordinary CI.

```sh
uv sync --locked --group dev
./gs verify                       # exact board checks + supported census replays
./gs targets --stratum good        # sourced unresolved cells, JSON
./gs import-tasks                  # known constructions awaiting imported rows
./gs census                       # class counts and verification bounds
./gs recent --limit 5
./gs build                        # validated, deterministic output in docs/ (not tracked)
python3 -m http.server 8769 --directory docs
```

`./gs verify --quick` and `./gs build --quick` skip census replay and report those counts as lower bounds. Never publish a quick build as a fully replayed census. The committed site is generated with the full build.

The maintenance checks are the ones CI runs:

```sh
uv run python -m verify.audit_seeds
uv run pytest -q
uv run ruff check verify tools site tests
uv run python -m verify.check_manifest
```

GPU slice replay is available through `./gs recheck`; it uses a bundled pinned engine and requires an appropriate CUDA environment. No arbitrary submission command or downloaded engine is executed. See [TRACKS.md](../TRACKS.md).

The picture at the top of the README is `assets/skew-hadamard-356.png`. Regenerate it with `uv run python assets/make_readme_image.py`; the script checks that the matrix it draws is skew-Hadamard before it writes the file, and the grid strip under the matrix changes when cell statuses do.

## What is checked

Every existence entry is validated from its rows with Python integer bit operations. Family membership, row sums, equivalence fingerprints, and typed baseline-gap flags are computed. The v1 group is cyclic; odd orders 1 through 511 are accepted, and the map covers orders through 131.

The primary `equiv-v1` convention includes independent reversals, signs, block permutations, and simultaneous cyclic-group automorphisms. It excludes independent shifts. At n=43, 10,080 best quadruples yield five primary classes and 20 legacy `branch-v1` orbits. This is equivalence of quadruples under the specified operations, not an assertion of full Hadamard-matrix isomorphism. The general and skew censuses are counted under `equiv-v2`, which also identifies independent cyclic shifts of each row; without it every class at those orders splits into thousands (514 against 4 at n=7). Entries and every other census keep `equiv-v1`.

Mathematical status (`exists`, `empty`, `unresolved`) and artifact availability (`verified-on-board`, `published-not-imported`, `none`) are separate. A known construction without imported rows is an import task. Its importer is never credited as the original discoverer.

The board also distinguishes participant work, independent reproductions, and literature/catalogue imports. Participant work is credited to the vxvware organisation. The participant section highlights the G-matrix census at n=49, which finds a class where the published classification reports none, the good matrices at n=53, 55, and 61, the G-matrix censuses at n=51 through 59, and the n=73 campaigns. Purple map marks and the Origin filter identify participant work; entry pages show the credited people separately from the importer. Publication status and verification remain independent, so an unpublished claim can be credited while its evidence gaps remain visible.

## Baseline of open skew orders

The board flags a skew quadruple whose order is in a dated baseline list of open skew-Hadamard orders, [baselines/skew-gaps.json](../baselines/skew-gaps.json). The list is 356, 404, 428 and 476, taken from version 2 of the database paper with key `cati2024database` in [baselines/references.bib](../baselines/references.bib) ([arXiv:2411.18897v2](https://arxiv.org/html/2411.18897v2), dated 2025-08-30). A baseline-gap flag never automatically establishes current novelty.

Two of the four orders have board entries. Eleven skew quadruples at n=89 are dated 2026-11-22 (the entry [n89-eec2e43c042090883c04](../quadruples/n89-eec2e43c042090883c04.json) and ten more listed on the [n=89 cell page](https://hadamards.com/cells/n89-skew-sds.html)); the Goethals–Seidel array of each is a skew-Hadamard matrix of order 356. A search of the literature and public repositories on 2026-11-22 found no earlier skew-Hadamard matrix of order 356. That is a report of the search, not a claim of priority, and the result has not been reviewed. A skew quadruple at n=119, giving order 476, was imported on 2026-11-17 from a public repository whose author found it ([n119-f302cc45a2f83e60917b](../quadruples/n119-f302cc45a2f83e60917b.json)). Orders 404 and 428 have no board entry.

[baselines/hadamard-orders.json](../baselines/hadamard-orders.json) records, for the orders 4 to 1200, which have no Hadamard or skew-Hadamard matrix in the sources it cites and what has been reported since; its `structure` field describes the layout. The front-page grid of the front page and the first two columns of the board grid are built from it, and `tests/test_orders.py` checks it against the board.

## Seed data and evidence

The inventories are [the original seeds](../baselines/seed-manifest.json) and [later participant constructions](../baselines/participant-seed-manifest.json). All 36,931 raw imported quadruples remain in compact archives under `evidence/`, with source commits, hashes, counts, and representative mappings. The explicit plan inventory is imported, including best/good catalogues, the campaign's n=15 and n=21 examples, the n=13 CPU/GPU fixtures, 15,360 p=73 families, and published Williamson and good rows. The 1,228 rows imported from vxvware/saw-best-matrix, a private repository whose imported files are [archived here](../evidence/participants/saw-best-matrix), supply one verified good-matrix class at each of n=53, 55, and 61 and the G-matrix (kkss) classes at every admissible odd order through 61, plus an orbit-restricted family at 63. The source reports complete censuses; the board records lower bounds where those searches have not been replayed here. The empty censuses at good n=59 and kkss n=57 are attributed claims without a tier.

Normal board checks use the bundled archives and need no other checkouts.

### Repeating the seed import

The import is a maintainer operation that regenerates seed entries and archives. Use a clean working tree and retain any later contribution files. Supply a parent directory containing `best-matrix-campaign`, `good-matrix-campaign`, and `skew356-campaign`, each at the source commit in [the manifest](../baselines/seed-manifest.json).

The first two repositories are private and `vxvware/skew356-campaign` no longer exists, so only a maintainer who holds all three checkouts can repeat this step. Nothing else needs them. Every file of the private repositories that an entry or manifest record cites is archived in this repository, and [the index](../evidence/participants/private-sources.json) gives its original repository, commit, path and SHA-256 next to the path of the copy: the catalogues and fixtures under `evidence/participants/best-matrix-campaign/` and `evidence/participants/good-matrix-campaign/`, the campaign receipts and write-ups under `evidence/seeds/`. Source records link those copies at the commit pinned as `ARCHIVE` in `tools/import_seeds.py` and `tools/import_dashboard.py`, and keep the original commit and path in their `commit` and `source_path` fields. No copy of `data/sweep_p73_d9.json` from `skew356-campaign` is known; the 15,360 rows imported from it are in `evidence/seeds/skew-p73-d9.jsonl.gz`, which the audit checks row by row. To move the pinned commit, run `uv run python tools/repin_archive.py <commit>`: it rewrites the links, rebinds the registered census replays whose entries changed by the pin alone, and updates the trusted manifest.

The literal Sage tables were extracted from [this pinned source](https://raw.githubusercontent.com/sagemath/sage/0835358cba62dc19fccdff54f32e52fded9f22b0/src/sage/combinat/matrices/hadamard_matrix.py). Download it as `hadamard_matrix.py`. Create a text file `sage-source.txt` with exactly these two lines:

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

The parser extracts literal numerical tables with Python's AST utilities; it does not execute Sage code. Every extracted quadruple is independently checked. Review all changed source identities before updating the trusted manifest.

### The vxvware/saw-best-matrix census files

The supplemental import reads the good-matrix and G-matrix sign-row files from `vxvware/saw-best-matrix`, each pinned in `tools/import_dashboard.py` to the commit that last changed it and to its SHA-256. Their original bytes are bundled under `evidence/participants/saw-best-matrix/`; the source repository is private and access to it is not needed to repeat the import. The importer checks the pinned hashes, exactly verifies all 1,228 quadruples, writes empty census files as unreplayed-census certificates, and refuses to replace different records.

```sh
uv run python tools/import_dashboard.py --source-dir evidence/participants/saw-best-matrix
uv run python -m verify.audit_seeds
```

This writes the separate participant seed manifest. The audit and site combine both inventories, deduplicating representatives across them. Update the trusted manifest and run a full build after reviewing any intentional import changes.

## Evidence that is incomplete

The board does not inflate historical receipts into proofs:

- Best-73 retains its documented 104,500-class legacy identity gap.
- Good-65 retains a newly detected missing prefix in the available machine receipt chain. The reported campaign conclusion is preserved as an archived claim; the board does not present it as fully verified emptiness.
- The first good-67 campaign is partial coverage; the later complete census, which is empty, is recorded separately as the contribution `good-67-empty`. Good-41 has a full archived coverage receipt, but has not been replayed on this CPU-only implementation machine.
- The order-11-subgroup certificate at n=89 is an exact row-sum obstruction for one restricted family: quadruples with a skew block whose four blocks are all unions of cosets of the order-11 multiplier subgroup. It says nothing about other quadruples at n=89. The n=89 skew-sds cell holds eleven verified quadruples, each with three blocks outside that family.
- Literature nonexistence is sourced per order, separately from local proof or computation tiers. Published census totals remain attributed claims unless the full enumerator is replayed.

## Trusted stack and entry PRs

`quadruples/`, `certificates/`, and `census/` hold one JSON entry per file. `verify/` and `schema/` define the trusted validator. `baselines/` holds sourced status, seed inventories, and reviewed replay registrations. `site/` builds the static GitHub Pages board. `research/` supplies agent instructions and starters.

`formal/` is a Lean 4 project with machine-checked statements about quadruples, among them that the Goethals–Seidel array of a quadruple is a Hadamard matrix; see [formal/README.md](../formal/README.md). CI does not build it and the verifier does not depend on it.

Entry PRs are append-only and cannot change the verifier, schemas, baseline, site, or workflows. A trusted-base CI job binds `submitted_by` to the PR author, checks scope, and validates candidate data with the base verifier. Changes to the trusted stack require a separate maintainer PR and a reviewed manifest update.
