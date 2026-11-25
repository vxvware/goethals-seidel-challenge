#!/usr/bin/env python3
"""Write the Williamson census entries at even orders from the inline enumerator.

Every equiv-v1 class that williamson-even-exact-v1 returns at an even order
becomes a quadruple entry, and one census entry per order lists them as a
computed-exact census. Before anything is written, the classes are merged
under the two further operations of the published equivalence and the result
is compared with the published count in baselines/order-status.json; a
difference stops the import.
"""

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store
from tools.import_replayed_censuses import BOARD, board_representatives, common
from verify.census_engine import WILLIAMSON_EVEN_ORDERS, williamson_even
from verify.common import ROOT, Rows, read_json
from verify.equivalence import canonical, fingerprint
from verify.model import validate

ENGINE = "williamson-even-exact-v1"
PAPER = {
    "title": "Bright, Kotsireas, Ganesh, Applying computer algebra systems with SAT solvers to the "
    "Williamson conjecture, J. Symbolic Comput. 100 (2020), Section 2.2 and Table 1",
    "url": "https://doi.org/10.1016/j.jsc.2019.07.024",
}
DISCOVERED = [
    "Bright, Kotsireas, Ganesh 2020 enumeration of the even orders up to 70; earlier constructions credited "
    "there. Enumerated here under equiv-v1 by this board's williamson-even-exact-v1"
]


def coarse(q: Rows) -> Rows:
    """The canonical form of a quadruple of even order under the published equivalence.

    The published relation adds two operations to the equiv-v1 generators: the
    cyclic shift of a single row by n/2, and the negation of every second
    entry of all four rows. Both commute with multipliers and with sign
    changes, and block permutations permute the shifted rows, so the least
    equiv-v1 canonical form over the 32 combinations is a complete invariant.
    """
    h = len(q[0]) // 2
    images = []
    for alternate in (False, True):
        base = [tuple(-x if alternate and j % 2 else x for j, x in enumerate(row)) for row in q]
        for shifted in product((False, True), repeat=4):
            images.append(canonical([row[h:] + row[:h] if s else row for row, s in zip(base, shifted, strict=True)]))
    return min(images)


def published_count(n: int) -> int:
    records = read_json(ROOT / "baselines/order-status.json", max_bytes=4 * 1024 * 1024)["records"]
    record = next(r for r in records if r["n"] == n and r["stratum"] == "williamson" and "published_classes" in r)
    return record["published_classes"]["count"]


def import_order(n: int, date: str) -> None:
    reps = williamson_even(n)
    merged = len({coarse(q) for q in reps})
    published = published_count(n)
    if merged != published:
        raise ValueError(f"n={n}: {merged} classes under the published equivalence, the paper lists {published}")
    sources = [
        {
            "title": f"This board's inline enumerator {ENGINE}, replayed at n={n}",
            "url": f"{BOARD}/blob/main/verify/census_engine.py",
            "source_path": "verify/census_engine.py",
        },
        PAPER,
    ]
    existing = board_representatives(n, "williamson")
    ids: set[str] = set()
    new = 0
    for rep in sorted(reps):
        if rep in existing:
            ids.add(existing[rep])
            continue
        quad_id = f"n{n}-{fingerprint(rep)[3:23]}"
        doc = validate(
            {
                **common(n, date),
                "discovered_by": DISCOVERED,
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
    noun = "class" if published == 1 else "classes"
    census = validate(
        {
            **common(n, date),
            "discovered_by": DISCOVERED,
            "sources": sources,
            "equivalence": "equiv-v1",
            "id": f"williamson-n{n}",
            "type": "census",
            "stratum": "williamson",
            "representatives": sorted(ids),
            "claimed_count": len(ids),
            "claim_kind": "computed-exact",
            "enumerator": ENGINE,
            "notes": (
                f"Full census replayed on this board by {ENGINE} on every verification: {len(ids)} "
                f"{'class' if len(ids) == 1 else 'classes'} under equiv-v1. The published enumeration counts "
                f"{published} {noun} at this order under a coarser equivalence, which also identifies the cyclic "
                "shift of a single block by n/2 and the negation of every second entry of all four blocks. With "
                f"those two operations added, the classes here fall into {merged}, the published count."
            ),
        }
    )
    store(ROOT / f"census/williamson-n{n}.json", census)
    print(f"williamson n={n}: {len(ids)} equiv-v1 classes, {merged} published classes, {new} new quadruples")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True)
    a = p.parse_args()
    for order in WILLIAMSON_EVEN_ORDERS:
        import_order(order, a.date)
