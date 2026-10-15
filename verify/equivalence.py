"""Exact canonical representatives under documented, versioned generators."""

from functools import lru_cache
from hashlib import sha256
from math import gcd
import json

VERSION = "equiv-v1"


@lru_cache(maxsize=512)
def multipliers(n):
    return tuple(
        tuple((t * j) % n for j in range(n)) for t in range(1, max(n, 2)) if gcd(t, n) == 1
    )


def normalize(row):
    return tuple(x * row[0] for x in row)


def reverse(row):
    return row[:1] + row[:0:-1]


@lru_cache(maxsize=20000)
def row_images(row, reversal=True):
    row = normalize(row)
    images = []
    for indices in multipliers(len(row)):
        image = tuple(row[j] for j in indices)
        images.append(min(image, reverse(image)) if reversal else image)
    return tuple(images)


def canonical(rows):
    images = [row_images(tuple(r)) for r in rows]
    return min(tuple(sorted(column)) for column in zip(*images))


def branch_canonical(rows):
    """Best-only legacy quotient: A/B/C permutations, simultaneous units, ±D."""
    from verify.gs_verify import row_types

    normalized = [normalize(tuple(r)) for r in rows]
    skew = [r for r in normalized if "k" in row_types(r)]
    sym = [r for r in normalized if "s" in row_types(r)]
    if len(rows[0]) != 1 and (len(skew) != 3 or len(sym) != 1):
        raise ValueError("branch-v1 is defined only for best quadruples")
    if len(rows[0]) == 1:
        return ((1,),) * 4
    images = [row_images(r, False) for r in skew + sym]
    return min(tuple(sorted(c[:3])) + (c[3],) for c in zip(*images))


def fingerprint(rows):
    rep = canonical(rows)
    payload = {
        "equivalence": VERSION,
        "group": {"kind": "cyclic", "invariant_factors": [len(rep[0])]},
        "rows": rep,
    }
    return "gs-" + sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
