#!/usr/bin/env python3
"""Write the census entries behind the pinned enumerators.

Good matrices at odd orders 17 to 43 come from the pinned C good census and
Williamson quadruples at odd orders 3 to 37 from the pinned C Williamson
census; quadruple entries for classes not yet on the board are written
alongside, and a class already present under an earlier entry keeps that
entry's id. Williamson orders 41 to 57 record the published classification of
Holzmann, Kharaghani and Tayfeh-Rezaie as a published-exact claim over the
representatives already imported from it. The six strata at n=1 share the one
quadruple of order 1.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.import_dashboard import store
from verify.census_engine import C_ENGINES, c_census
from verify.common import ROOT, TRACKS
from verify.equivalence import canonical, fingerprint
from verify.gs_verify import memberships, validate_rows
import json
from verify.model import validate

SUBMITTER = "vprusso"
BOARD = "https://github.com/vxvware/goethals-seidel-challenge"
HKT = {
    "title": "Holzmann, Kharaghani, Tayfeh-Rezaie, Williamson matrices up to order 59, "
    "Des. Codes Cryptogr. 46 (2008), Tables 2 and 3",
    "url": "https://doi.org/10.1007/s10623-007-9163-5",
}
HKT_COUNTS = {39: 1, 41: 1, 43: 2, 45: 1, 47: 0, 49: 1, 51: 2, 53: 0, 55: 1, 57: 1, 59: 0}
PAPER = {
    "title": "Bright, Đoković, Kotsireas, Ganesh, A SAT+CAS approach to finding good "
    "matrices: new examples and counterexamples (AAAI 2019), Table 1 and its survey of "
    "earlier searches",
    "url": "https://arxiv.org/abs/1811.05094",
}
DISCOVERED = [
    "Published searches surveyed in arXiv:1811.05094 (Szekeres 1988; Koukouvinos 1999; "
    "Georgiou, Koukouvinos, Stylianou 2002; Đoković and Kotsireas 2018; Bright, Đoković, "
    "Kotsireas, Ganesh 2019); enumerated here by the vxvware good census"
]
ORDER_ONE = "n1-0a982ea1a0df30b3b8d9"


def board_representatives(n, stratum):
    """Canonical rows of every quadruple entry at order n in the stratum, by entry id."""
    found = {}
    for path in sorted((ROOT / "quadruples").glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("type") == "quadruple" and doc.get("n") == n:
            q = validate_rows([list(r) for r in doc["rows"]], n)
            if stratum in memberships(q):
                found[canonical(q)] = doc["id"]
    return found


def common(n, date):
    return {
        "schema_version": 1,
        "n": n,
        "group": {"kind": "cyclic", "invariant_factors": [n]},
        "submitted_by": SUBMITTER,
        "submitted_at": date,
    }


def engine_source(engine, commit, n):
    path = C_ENGINES[engine]["source"]
    return {
        "title": f"This board's pinned census {engine}, replayed at n={n}",
        "url": f"{BOARD}/blob/{commit}/{path}",
        "commit": commit,
        "source_path": path,
    }


def import_replayed(engine, n, commit, date):
    stratum = C_ENGINES[engine]["stratum"]
    reps = c_census(engine, n)
    paper = PAPER if stratum == "good" else HKT
    discovered = (
        DISCOVERED
        if stratum == "good"
        else [
            "Holzmann, Kharaghani, Tayfeh-Rezaie 2008 classification; earlier constructions "
            "credited there"
        ]
    )
    sources = [engine_source(engine, commit, n), paper]
    existing = board_representatives(n, stratum)
    ids = set()
    new = 0
    for rep in sorted(reps):
        if rep in existing:
            ids.add(existing[rep])
            continue
        quad_id = f"n{n}-{fingerprint(rep)[3:23]}"
        doc = validate(
            {
                **common(n, date),
                "discovered_by": discovered,
                "sources": sources,
                "equivalence": "equiv-v1",
                "id": quad_id,
                "type": "quadruple",
                "rows": [list(r) for r in rep],
            }
        )
        store(ROOT / f"quadruples/{quad_id}.json", doc, rows_may_match=True)
        ids.add(quad_id)
        new += 1
    if stratum == "good":
        note = (
            "Full census replayed on this board by good-census-c-v1 on every verification. "
            "Before the engine was pinned its class sets were checked against the Python "
            "enumerator at orders 3 to 15, the stored saw-best-matrix censuses at 17 to 45, "
            "and the published counts of arXiv:1811.05094 at orders divisible by 3."
        )
    else:
        note = (
            "Full census replayed on this board by williamson-census-c-v1 on every "
            "verification. It reproduces the classification of Holzmann, Kharaghani and "
            "Tayfeh-Rezaie at this order."
        )
    census = validate(
        {
            **common(n, date),
            "discovered_by": discovered,
            "sources": sources,
            "equivalence": "equiv-v1",
            "id": f"{stratum}-n{n}",
            "type": "census",
            "stratum": stratum,
            "representatives": sorted(ids),
            "claimed_count": len(ids),
            "claim_kind": "computed-exact",
            "enumerator": engine,
            "notes": note,
        }
    )
    store(ROOT / f"census/{stratum}-n{n}.json", census)
    print(f"{stratum} n={n}: {len(ids)} classes, {new} new quadruples")


def import_published_williamson(n, date):
    """The published classification over the representatives already imported from it."""
    reps = board_representatives(n, "williamson")
    ids = sorted(set(reps.values()))
    if len(ids) != HKT_COUNTS[n]:
        raise ValueError(
            f"n={n}: board holds {len(ids)} Williamson classes, paper lists {HKT_COUNTS[n]}"
        )
    census = validate(
        {
            **common(n, date),
            "discovered_by": [
                "Holzmann, Kharaghani, Tayfeh-Rezaie 2008 classification; earlier "
                "constructions credited there"
            ],
            "sources": [HKT],
            "equivalence": "equiv-v1",
            "id": f"williamson-n{n}",
            "type": "census",
            "stratum": "williamson",
            "representatives": ids,
            "claimed_count": len(ids),
            "claim_kind": "published-exact",
            "notes": (
                "The paper's exhaustive classification; its representatives are verified "
                "here and pairwise inequivalent, but the search has not been replayed on "
                "this board, so the count is recorded as a lower bound."
            ),
        }
    )
    store(ROOT / f"census/williamson-n{n}.json", census)
    print(f"williamson n={n}: published count {len(ids)}")


def import_order_one(date):
    for stratum in TRACKS:
        census = validate(
            {
                **common(1, date),
                "discovered_by": ["Trivial: the single quadruple of order 1"],
                "sources": [
                    {
                        "title": "Mathematical contract, equiv-v1: at n=1 all six tracks apply",
                        "url": f"{BOARD}/blob/main/notes/mathematical-contract.md",
                    }
                ],
                "equivalence": "equiv-v1",
                "id": f"order-one-{stratum}",
                "type": "census",
                "stratum": stratum,
                "representatives": [ORDER_ONE],
                "claimed_count": 1,
                "claim_kind": "computed-exact",
                "enumerator": "small-exact-v1",
                "notes": "The row (1) is both symmetric and skew, so the one quadruple of order 1 is the whole census of every stratum.",
            }
        )
        store(ROOT / f"census/order-one-{stratum}.json", census)
    print("n=1: six censuses")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-commit", required=True, help="commit of the pinned good census")
    p.add_argument("--date", required=True)
    a = p.parse_args()
    import_order_one(a.date)
    for n in (17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 43):
        import_replayed("good-census-c-v1", n, a.engine_commit, a.date)
    williamson = C_ENGINES["williamson-census-c-v1"]["orders"]
    for n in williamson:
        if n != 35:
            import_replayed("williamson-census-c-v1", n, a.engine_commit, a.date)
    for n in sorted(HKT_COUNTS):
        if n not in williamson and HKT_COUNTS[n]:
            import_published_williamson(n, a.date)
