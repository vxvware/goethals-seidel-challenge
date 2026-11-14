"""Import published cyclic supplementary difference sets as quadruple entries.

The board records several orders as "published, not imported": a paper proves
existence but no verified rows are on the board. This tool holds a table of
such published sets, transcribed as index lists, converts each to four ±1 rows
with Djokovic's convention (x_i = -1 for i in the set, +1 otherwise, so a skew
set not containing 0 gives a skew row with x_0 = +1), verifies the rows with the
board's exact verifier, and writes one entry per new equiv-v1 class.

Sets in the table:

  n = 47   Djokovic, Skew-Hadamard matrices of orders 188 and 388 exist,
           Int. Math. Forum 3 (2008) 1063-1068 (arXiv:0704.0640), Propositions
           2.1 and 2.2: four 3-(47; 30, 22, 22; 39) and 3-(47; 21, 19, 19; 24)
           difference families completed by the Paley skew (47; 23; 11) set.
  n = 47   Djokovic, Supplementary difference sets with symmetry for Hadamard
           matrices, Oper. Matrices 3 (2009) 557-569 (arXiv:0903.5053),
           Section 5.4: the (47; 23, 21, 19, 19; 35) SDS of type (ks**).
  n = 127  The same paper, Section 5.7: the (127; 63, 57, 57, 57; 107) SDS of
           type (ks**) built from the Paley set and unions of cosets of the
           subgroup H = <2> of order 7 in Z_127^*.
  n = 127  Good matrices (type ksss). The existence source is Djokovic, Good
           matrices of orders 33, 35 and 127, JCMCC 14 (1993) 145-152, which is
           not available online. The rows here were not transcribed from it:
           they are the unique equiv-v1 class of (ksss) quadruples whose blocks
           are unions of cosets of H (plus 0 in the symmetric blocks), the same
           ansatz as Section 5.7 of the 2009 paper, found by an exhaustive
           search over that ansatz (512 skew by 126 symmetric choices). The
           entry notes say so.

    uv run python tools/import_literature_rows.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store  # noqa: E402
from verify.common import ROOT  # noqa: E402
from verify.equivalence import canonical, fingerprint  # noqa: E402
from verify.gs_verify import memberships, validate_rows  # noqa: E402
from verify.model import validate  # noqa: E402

SUBMITTER = "vprusso"
DATE = "2026-11-14"

DJOKOVIC_2008 = {
    "title": "Djokovic, Skew-Hadamard matrices of orders 188 and 388 exist, "
    "Int. Math. Forum 3 (2008) 1063-1068",
    "url": "https://arxiv.org/abs/0704.0640",
}
DJOKOVIC_2009 = {
    "title": "Djokovic, Supplementary difference sets with symmetry for Hadamard matrices, "
    "Oper. Matrices 3 (2009) 557-569",
    "url": "https://arxiv.org/abs/0903.5053",
}
DJOKOVIC_1993 = {
    "title": "Djokovic, Good matrices of orders 33, 35 and 127, JCMCC 14 (1993) 145-152",
    "url": "https://arxiv.org/abs/1811.05094",
}


def paley(n):
    return sorted({(x * x) % n for x in range(1, n)})


# Cosets of H = <2> (order 7) in Z_127^*, indexed as in Djokovic 2009, Section 5.7:
# alpha_{2i} = r_i H and alpha_{2i+1} = -r_i H for r_i = 1, 3, 5, 7, 9, 11, 13, 19, 21.
COSET_REPS = [1, 3, 5, 7, 9, 11, 13, 19, 21]


def coset(index, n=127):
    r = COSET_REPS[index // 2] * (-1 if index % 2 else 1)
    return {(r * pow(2, i, n)) % n for i in range(7)}


def coset_union(indices, with_zero, n=127):
    out = {0} if with_zero else set()
    for k in indices:
        out |= coset(k, n)
    return sorted(out)


# fmt: off
TABLE = [
    {
        "n": 47,
        "label": "Djokovic 2008, Proposition 2.1, family {X1, X2, X3}",
        "parameters": "(47; 23, 30, 22, 22; 50), type (k***)",
        "source": DJOKOVIC_2008,
        "discovered_by": ["Djokovic 2008, Proposition 2.1 (X), with the Paley skew set"],
        "sets": [
            paley(47),
            [2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 25, 27, 30,
             31, 33, 35, 37, 38, 39, 40, 42, 43, 44],
            [1, 3, 6, 7, 8, 11, 13, 14, 15, 19, 20, 21, 24, 27, 30, 33, 39, 41, 43, 44, 45, 46],
            [3, 6, 8, 10, 11, 12, 14, 20, 21, 23, 24, 25, 26, 27, 30, 31, 32, 34, 35, 41, 42, 45],
        ],
        "expect": "skew-sds",
    },
    {
        "n": 47,
        "label": "Djokovic 2008, Proposition 2.1, family {Y1, Y2, Y3}",
        "parameters": "(47; 23, 30, 22, 22; 50), type (k***)",
        "source": DJOKOVIC_2008,
        "discovered_by": ["Djokovic 2008, Proposition 2.1 (Y), with the Paley skew set"],
        "sets": [
            paley(47),
            [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21, 23, 24, 25, 27,
             28, 29, 30, 31, 35, 38, 41, 43, 44, 46],
            [3, 6, 7, 8, 10, 11, 12, 16, 22, 25, 26, 31, 32, 33, 34, 37, 39, 41, 42, 43, 44, 46],
            [3, 7, 12, 13, 15, 16, 18, 20, 21, 23, 25, 26, 27, 28, 32, 35, 38, 39, 42, 44, 45, 46],
        ],
        "expect": "skew-sds",
    },
    {
        "n": 47,
        "label": "Djokovic 2008, Proposition 2.2, family {P1, P2, P3}",
        "parameters": "(47; 23, 21, 19, 19; 35), type (k***)",
        "source": DJOKOVIC_2008,
        "discovered_by": ["Djokovic 2008, Proposition 2.2 (P), with the Paley skew set"],
        "sets": [
            paley(47),
            [0, 2, 4, 5, 9, 10, 12, 16, 17, 19, 21, 22, 23, 25, 27, 28, 35, 36, 37, 43, 46],
            [0, 1, 2, 6, 8, 9, 11, 15, 16, 19, 25, 32, 33, 35, 36, 37, 38, 40, 44],
            [1, 2, 3, 4, 5, 6, 7, 10, 11, 16, 18, 22, 24, 28, 31, 35, 38, 40, 43],
        ],
        "expect": "skew-sds",
    },
    {
        "n": 47,
        "label": "Djokovic 2008, Proposition 2.2, family {Q1, Q2, Q3}",
        "parameters": "(47; 23, 21, 19, 19; 35), type (k***)",
        "source": DJOKOVIC_2008,
        "discovered_by": ["Djokovic 2008, Proposition 2.2 (Q), with the Paley skew set"],
        "sets": [
            paley(47),
            [4, 5, 6, 8, 11, 12, 15, 20, 21, 23, 25, 26, 28, 29, 30, 31, 32, 36, 39, 41, 43],
            [1, 2, 5, 7, 13, 14, 21, 22, 24, 26, 31, 32, 35, 36, 37, 39, 40, 42, 46],
            [1, 2, 3, 4, 5, 9, 12, 18, 20, 21, 24, 25, 32, 34, 38, 39, 43, 44, 46],
        ],
        "expect": "skew-sds",
    },
    {
        "n": 47,
        "label": "Djokovic 2009, Section 5.4",
        "parameters": "(47; 23, 21, 19, 19; 35), type (ks**)",
        "source": DJOKOVIC_2009,
        "discovered_by": ["Djokovic 2009, Section 5.4"],
        "sets": [
            [1, 2, 3, 4, 6, 7, 8, 9, 12, 14, 16, 17, 18, 21, 24, 25, 27, 28, 32, 34, 36, 37, 42],
            [0, 6, 8, 10, 11, 14, 17, 18, 19, 21, 23, 24, 26, 28, 29, 30, 33, 36, 37, 39, 41],
            [0, 1, 2, 5, 6, 8, 9, 15, 16, 19, 21, 23, 27, 28, 33, 36, 38, 39, 40],
            [0, 2, 3, 4, 7, 8, 9, 10, 12, 18, 21, 23, 24, 25, 26, 30, 34, 35, 44],
        ],
        "expect": "skew-sds",
    },
    {
        "n": 127,
        "label": "Djokovic 2009, Section 5.7",
        "parameters": "(127; 63, 57, 57, 57; 107), type (ks**)",
        "source": DJOKOVIC_2009,
        "discovered_by": ["Djokovic 2009, Section 5.7, with the Paley skew set"],
        "sets": [
            paley(127),
            coset_union([0, 1, 2, 3, 6, 7, 16, 17], True),
            coset_union([4, 6, 7, 11, 13, 14, 15, 16], True),
            coset_union([0, 4, 5, 7, 11, 12, 15, 16], True),
        ],
        "expect": "skew-sds",
        "notes": (
            "Blocks 2 to 4 are {0} together with the cosets alpha_k, k in J_i, of the "
            "subgroup H = <2> of order 7 in Z_127^*, with J_1 = {0, 1, 2, 3, 6, 7, 16, 17}, "
            "J_2 = {4, 6, 7, 11, 13, 14, 15, 16}, J_3 = {0, 4, 5, 7, 11, 12, 15, 16} and the "
            "coset numbering of the paper. The rows are stored in canonical form under equiv-v1."
        ),
    },
    {
        "n": 127,
        "label": "good matrices over the cosets of <2> (Djokovic 1993 ansatz)",
        "parameters": "(127; 63, 57, 57, 57; 107), type (ksss)",
        "source": DJOKOVIC_1993,
        "discovered_by": [
            "Djokovic 1993 proved existence; these rows are the unique equiv-v1 class with "
            "blocks that are unions of cosets of <2> in Z_127^*, found by tools/import_literature_rows.py"
        ],
        "sets": [
            coset_union([0, 2, 5, 6, 8, 10, 13, 14, 16], False),
            coset_union([2, 3, 6, 7, 8, 9, 10, 11], True),
            coset_union([2, 3, 10, 11, 12, 13, 16, 17], True),
            coset_union([6, 7, 8, 9, 12, 13, 16, 17], True),
        ],
        "expect": "good",
        "notes": (
            "Not transcribed from the 1993 paper, which is not available online; the URL "
            "points to a survey that cites it. The skew block is a union of nine cosets of "
            "H = <2> (one from each pair {alpha_{2i}, alpha_{2i+1}}) and each symmetric block "
            "is {0} with four pairs, the ansatz of Djokovic 2009, Section 5.7. An exhaustive "
            "search over this ansatz finds 18 solutions in a single equiv-v1 class. The rows "
            "are stored in canonical form under equiv-v1."
        ),
    },
]
# fmt: on


def rows_from_sets(n, sets):
    return validate_rows([[-1 if j in set(s) else 1 for j in range(n)] for s in sets], n)


def board_classes(n):
    found = {}
    for path in sorted((ROOT / "quadruples").glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("type") == "quadruple" and doc.get("n") == n:
            found[canonical(validate_rows([list(r) for r in doc["rows"]], n))] = doc["id"]
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DATE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seen = {}
    for item in TABLE:
        n = item["n"]
        rows = rows_from_sets(n, item["sets"])
        tracks = memberships(rows)
        if item["expect"] not in tracks:
            raise ValueError(f"{item['label']}: expected {item['expect']}, got {tracks}")
        rep = canonical(rows)
        existing = board_classes(n)
        existing.update(seen.get(n, {}))
        sizes = [len(s) for s in item["sets"]]
        if rep in existing:
            print(f"n={n} {item['label']}: already on the board as {existing[rep]}")
            continue
        quad_id = f"n{n}-{fingerprint(rep)[3:23]}"
        seen.setdefault(n, {})[rep] = quad_id
        print(f"n={n} {item['label']}: sizes {sizes}, memberships {tracks}, new entry {quad_id}")
        if a.dry_run:
            continue
        doc = validate(
            {
                "schema_version": 1,
                "n": n,
                "group": {"kind": "cyclic", "invariant_factors": [n]},
                "submitted_by": SUBMITTER,
                "submitted_at": a.date,
                "discovered_by": item["discovered_by"],
                "sources": [item["source"]],
                "equivalence": "equiv-v1",
                "id": quad_id,
                "type": "quadruple",
                "rows": [list(r) for r in rep],
                "notes": item.get(
                    "notes",
                    f"Parameters {item['parameters']}. Rows follow x_i = -1 for i in the "
                    "published set and +1 otherwise, then canonical form under equiv-v1.",
                ),
            }
        )
        store(ROOT / f"quadruples/{quad_id}.json", doc, rows_may_match=True)


if __name__ == "__main__":
    main()
