#!/usr/bin/env python3
"""Import the published Williamson classes as quadruple entries.

Sources are the archived text files under evidence/literature/: the 73 class
representatives transcribed from Tables 2 and 3 of Holzmann, Kharaghani and
Tayfeh-Rezaie (2008), and the MathCheck enumeration files for orders 63 and
69 (Bright, Kotsireas, Ganesh). Every row set is verified before it is
written, and a set equivalent to an entry already on the board is skipped.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify import gs_verify
from verify.common import ROOT, digest, read_json

LITERATURE = ROOT / "evidence/literature"
HKT = {
    "file": "hkt2008-williamson-tables.txt",
    "title": "Holzmann, Kharaghani, Tayfeh-Rezaie, Williamson matrices up to order 59, "
    "Des. Codes Cryptogr. 46 (2008), Tables 2 and 3",
    "url": "https://doi.org/10.1007/s10623-007-9163-5",
    "discoverers": [
        "Holzmann, Kharaghani, Tayfeh-Rezaie 2008 classification; earlier constructions credited there"
    ],
}
MATHCHECK = {
    63: {
        "file": "mathcheck-williamson-63.txt",
        "title": "MathCheck Williamson sequences of order 63 (Bright, Kotsireas, Ganesh), Zenodo record 1195693",
        "url": "https://zenodo.org/record/1195693/files/williamson-63.txt",
        "discoverers": ["Bright, Kotsireas, Ganesh 2019 (new set); Turyn 1972 (the other)"],
    },
    69: {
        "file": "mathcheck-williamson-69.txt",
        "title": "MathCheck Williamson sequences of order 69 (Bright, Kotsireas, Ganesh), Zenodo record 1195693",
        "url": "https://zenodo.org/record/1195693/files/williamson-69.txt",
        "discoverers": ["Turyn 1972 construction; enumerated by Bright, Kotsireas, Ganesh 2019"],
    },
}


def hkt_rows():
    for line in (LITERATURE / HKT["file"]).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        n, *words = line.split()
        n = int(n)
        if len(words) != 4 or any(len(w) != (n - 1) // 2 for w in words):
            raise ValueError(f"malformed HKT line: {line}")
        rows = []
        for w in words:
            half = [1 if c == "+" else -1 for c in w]
            rows.append([1] + half + half[::-1])
        yield n, rows


def mathcheck_rows(n, spec):
    for line in (LITERATURE / spec["file"]).read_text().splitlines():
        words = re.findall(r"[+-]+", line)
        if not words:
            continue
        if len(words) != 4 or any(len(w) != n for w in words):
            raise ValueError(f"malformed MathCheck line for n={n}")
        yield n, [[1 if c == "+" else -1 for c in w] for w in words]


def entry(n, rows, spec, author, date):
    path = LITERATURE / spec["file"]
    return {
        "schema_version": 1,
        "n": n,
        "group": {"kind": "cyclic", "invariant_factors": [n]},
        "submitted_by": author,
        "submitted_at": date,
        "discovered_by": spec["discoverers"],
        "sources": [
            {
                "title": spec["title"],
                "url": spec["url"],
                "artifact": {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)},
            }
        ],
        "equivalence": "equiv-v1",
        "id": None,
        "type": "quadruple",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    baseline = read_json(ROOT / "baselines/skew-gaps.json")
    held = {}
    for path in glob.glob(str(ROOT / "quadruples/*.json")):
        doc = read_json(Path(path))
        held[gs_verify.verify(doc, baseline)["fingerprint"]] = doc["id"]
    candidates = [(n, rows, HKT) for n, rows in hkt_rows()]
    for n, spec in MATHCHECK.items():
        candidates += [(n, rows, spec) for _, rows in mathcheck_rows(n, spec)]
    written = skipped = 0
    for n, rows, spec in candidates:
        doc = entry(n, rows, spec, args.author, args.date)
        doc["id"] = "pending"
        result = gs_verify.verify(doc, baseline)
        if "williamson" not in result["tracks"]:
            raise ValueError(f"not a Williamson quadruple at n={n}")
        fp = result["fingerprint"]
        if fp in held:
            skipped += 1
            continue
        doc["id"] = f"n{n}-{fp[3:23]}"
        held[fp] = doc["id"]
        written += 1
        if args.write:
            (ROOT / f"quadruples/{doc['id']}.json").write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
            )
    print(
        f"{len(candidates)} published sets: {written} new entries, {skipped} already on the board"
    )


if __name__ == "__main__":
    main()
