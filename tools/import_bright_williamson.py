#!/usr/bin/env python3
"""Import the published enumeration of Williamson quadruples at even n as catalogue files.

The source is the dataset the authors of bright2020williamson published with
the paper (key bright2018enumeration, Zenodo, CC BY 4.0): one file per order,
one quadruple per line, one line per class of the paper's equivalence. The
files are read as data from a directory given on the command line; nothing in
them is executed, and a file with any byte other than +, -, space and newline
is rejected.

For every even n the importer

1. checks each quadruple exactly with the board's verifier and requires
   symmetric rows;
2. requires the number of lines to equal the class count of the paper's
   Table 1 as recorded in baselines/order-status.json, and the lines to be
   pairwise inequivalent under the paper's relation;
3. where the board already holds a census of the cell (n up to 20), requires
   every line to be a class of that census and the two lists to have the same
   classes under the paper's relation, and writes nothing;
4. otherwise splits every published class into its classes under the board's
   equivalence, by taking the canonical forms of the images under a half
   shift of each row and the alternating negation, and writes a catalogue of
   all of them when there are at most SPLIT_LIMIT, or else a catalogue with
   one quadruple per published class.

A catalogue is written with the census entry that names it. The split list is
complete exactly when the published classification is; it is a lower bound on
the board and no search is rerun.

    uv run python tools/import_bright_williamson.py DATASET_DIR --date YYYY-MM-DD [--dry-run]
"""

import argparse
import gzip
import hashlib
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store
from tools.import_replayed_censuses import board_representatives
from verify.catalogue import format_line, parse_line
from verify.common import ROOT, JSONObject, Rows, digest, read_json
from verify.equivalence import canonical, canonical_half_shifts
from verify.gs_verify import memberships, validate_rows
from verify.model import validate

SUBMITTER = "vprusso"
# An order's catalogue holds every class under the board's equivalence up to this many, and one
# quadruple per published class beyond it.
SPLIT_LIMIT = 5000
ORDERS = range(2, 71, 2)
PAPER = {
    "title": "Bright, Kotsireas, Ganesh, Applying computer algebra systems with SAT solvers to the Williamson "
    "conjecture, J. Symbolic Comput. 100 (2020), Section 2.2 and Table 1",
    "url": "https://doi.org/10.1016/j.jsc.2019.07.024",
}
DATASET_URL = "https://doi.org/10.5281/zenodo.1195693"
DISCOVERED = [
    "Bright, Kotsireas, Ganesh 2020 exhaustive enumeration of the even orders up to 70, and the earlier "
    "constructions credited there; rows from the authors' dataset (Zenodo 10.5281/zenodo.1195693, CC BY 4.0), "
    "put in canonical form by tools/import_bright_williamson.py"
]


def read_dataset(path: Path, n: int) -> list[Rows]:
    """The quadruples of one dataset file, each checked exactly and symmetric."""
    data = path.read_bytes()
    if len(data) > 64 * 1024 * 1024 or set(data) - set(b"+- \n"):
        raise ValueError(f"{path.name}: oversized, or a byte other than +, -, space and newline")
    found = []
    for line in data.decode("ascii").split("\n"):
        if not line:
            continue
        rows = validate_rows(parse_line(line, n), n)
        if "williamson" not in memberships(rows):
            raise ValueError(f"{path.name}: a quadruple with a row that is not symmetric")
        found.append(rows)
    return found


def split(rows: Rows) -> set[Rows]:
    """The classes under equiv-v1 inside the published class of a quadruple of even length."""
    h = len(rows[0]) // 2
    found = set()
    for alternate in (False, True):
        base = [tuple(-x if alternate and j % 2 else x for j, x in enumerate(row)) for row in rows]
        for shifted in product((False, True), repeat=4):
            found.add(canonical([row[h:] + row[:h] if s else row for row, s in zip(base, shifted, strict=True)]))
    return found


def published_count(n: int) -> int:
    records = read_json(ROOT / "baselines/order-status.json", max_bytes=4 * 1024 * 1024)["records"]
    record = next(r for r in records if r["n"] == n and r["stratum"] == "williamson" and "published_classes" in r)
    return record["published_classes"]["count"]


def write_catalogue(path: Path, classes: list[Rows], dry_run: bool) -> JSONObject:
    """Write the catalogue file, byte for byte reproducible, and return its reference."""
    body = "".join(format_line(rows) + "\n" for rows in classes).encode("ascii")
    packed = gzip.compress(body, compresslevel=9, mtime=0)
    if path.exists() and path.read_bytes() != packed:
        raise ValueError(f"refusing to overwrite {path.relative_to(ROOT)} with different contents")
    if not dry_run and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packed)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(packed).hexdigest()}


def import_order(n: int, source: Path, date: str, dry_run: bool) -> JSONObject:
    file = source / f"williamson-{n:02d}.txt"
    reps = read_dataset(file, n)
    published = published_count(n)
    coarse = {canonical_half_shifts(q) for q in reps}
    if not len(reps) == len(coarse) == published:
        raise ValueError(f"n={n}: {len(reps)} lines in {len(coarse)} published classes, the paper lists {published}")
    report: JSONObject = {"n": n, "published": published, "dataset_sha256": digest(file)}
    existing = board_representatives(n, "williamson")
    if n <= 20:
        ours = set(existing)
        if not {canonical(q) for q in reps} <= ours or {canonical_half_shifts(q) for q in ours} != coarse:
            raise ValueError(f"n={n}: the dataset and the board's census disagree")
        return report | {"board_classes": len(ours), "written": "nothing, the board holds every class"}
    # More published classes than the limit means more board classes than the limit, and the split of the
    # largest orders has millions of quadruples: it is not computed.
    whole = set().union(*(split(q) for q in reps)) if published <= SPLIT_LIMIT else None
    full = whole is not None and len(whole) <= SPLIT_LIMIT
    if whole is not None:
        report["board_classes"] = len(whole)
        if len({canonical_half_shifts(q) for q in whole}) != published:
            raise ValueError(f"n={n}: the split classes do not fall into the published ones")
    if whole is not None and full and not set(existing) <= whole:
        raise ValueError(f"n={n}: a board quadruple is in no published class")
    classes = sorted(whole) if whole is not None and full else sorted(canonical(q) for q in reps)
    ref = write_catalogue(ROOT / f"evidence/catalogues/williamson-n{n}.txt.gz", classes, dry_run)
    if full:
        notes = (
            f"{len(classes)} quadruples, no two equivalent: every class under the board's equivalence inside "
            f"the {published} published classes of the cited enumeration. They were obtained from the one "
            "quadruple per class in the authors' dataset by applying the two operations the published "
            "equivalence has and the board's does not, the cyclic shift of a single block by n/2 and the "
            "negation of every second entry of all four blocks, and are stored in canonical form. The list is "
            "complete exactly when the published classification is. No search was rerun for this entry, so the "
            "count is a lower bound here."
        )
    else:
        into = (
            f"these classes split into {len(whole)} classes under the board's equivalence"
            if whole is not None
            else "each of these classes splits into as many as 32 classes under the board's equivalence"
        )
        notes = (
            f"{len(classes)} quadruples, one from each of the {published} classes of the cited enumeration, "
            "from the authors' dataset and stored in canonical form. The published equivalence also identifies "
            "the cyclic shift of a single block by n/2 and the negation of every second entry of all four "
            f"blocks, which the board's does not: {into}, more than the {SPLIT_LIMIT} up to which a catalogue "
            "lists them all. The count here is a lower bound on the board's classes. No search was rerun for "
            "this entry."
        )
    census = validate(
        {
            "schema_version": 1,
            "n": n,
            "group": {"kind": "cyclic", "invariant_factors": [n]},
            "submitted_by": SUBMITTER,
            "submitted_at": date,
            "discovered_by": DISCOVERED,
            "sources": [
                PAPER,
                {
                    "title": f"Bright, Kotsireas, Ganesh, Enumeration of Williamson sequences divisible by 2 or 3, "
                    f"Zenodo (2018), file williamson-{n:02d}.txt, sha256 {report['dataset_sha256']}",
                    "url": DATASET_URL,
                },
            ],
            "equivalence": "equiv-v1",
            "id": f"williamson-n{n}",
            "type": "census",
            "stratum": "williamson",
            "representatives": [],
            "catalogue": ref,
            "published_classes": {"count": published, "split": full},
            "claimed_count": len(classes),
            "claim_kind": "lower-bound",
            "notes": notes,
        }
    )
    if not dry_run:
        store(ROOT / f"census/williamson-n{n}.json", census)
    return report | {"written": f"{len(classes)} rows, {'every class' if full else 'one per published class'}"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("--date", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    for order in ORDERS:
        print(import_order(order, a.source, a.date, a.dry_run), flush=True)
