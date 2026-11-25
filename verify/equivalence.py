"""Exact canonical representatives under documented, versioned generators."""

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from hashlib import sha256
from math import gcd

from verify.common import Row, Rows, RowsLike

VERSION = "equiv-v1"
VERSION_SHIFTS = "equiv-v2"


@lru_cache(maxsize=512)
def multipliers(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple((t * j) % n for j in range(n)) for t in range(1, max(n, 2)) if gcd(t, n) == 1)


def normalize(row: Sequence[int]) -> Row:
    return tuple(x * row[0] for x in row)


def reverse(row: Row) -> Row:
    return row[:1] + row[:0:-1]


@lru_cache(maxsize=20000)
def row_images(row: Row, reversal: bool = True) -> Rows:
    row = normalize(row)
    images = []
    for indices in multipliers(len(row)):
        image = tuple(row[j] for j in indices)
        images.append(min(image, reverse(image)) if reversal else image)
    return tuple(images)


def canonical(rows: RowsLike) -> Rows:
    images = [row_images(tuple(r)) for r in rows]
    return min(tuple(sorted(column)) for column in zip(*images, strict=False))


@lru_cache(maxsize=20000)
def dihedral_min(row: Row) -> Row:
    """Least sign-normalized image of a row under cyclic shifts and reversal."""
    n = len(row)
    shifted = (normalize(row[s:] + row[:s]) for s in range(n))
    return min(min(image, reverse(image)) for image in shifted)


@lru_cache(maxsize=20000)
def row_images_shifts(row: Row) -> Rows:
    row = normalize(row)
    return tuple(dihedral_min(tuple(row[j] for j in indices)) for indices in multipliers(len(row)))


def canonical_shifts(rows: RowsLike) -> Rows:
    """equiv-v2: equiv-v1 plus independent cyclic shifts of each row.

    Multipliers normalize the dihedral group of Z_n, so the least sorted tuple
    of per-row dihedral minima over all multipliers is a complete invariant.
    Strata are not invariant under shifts: a census under this version counts
    the classes that contain a member of its stratum.
    """
    images = [row_images_shifts(tuple(r)) for r in rows]
    return min(tuple(sorted(column)) for column in zip(*images, strict=False))


def canonical_half_shifts(rows: RowsLike) -> Rows:
    """The published relation on quadruples of even length n: equiv-v1 with two further operations.

    The operations are the cyclic shift of a single row by n/2 and the
    negation of every second entry of all four rows at once. Both commute with
    multipliers, and a row can be shifted or not independently of the others,
    so for each multiplier and each of the two alternations the least sorted
    tuple takes the smaller of every row's two images. This is a complete
    invariant of the relation. It is not a board equivalence version: no entry
    or census is declared under it, and it is used only to say how catalogue
    rows sit in a published classification.
    """
    n = len(rows[0])
    if n % 2:
        raise ValueError("the half shift needs even length")
    h = n // 2
    best: Rows | None = None
    for alternate in (False, True):
        columns = []
        for row in rows:
            base = tuple(-x if alternate and j % 2 else x for j, x in enumerate(row))
            pairs = zip(row_images(base), row_images(base[h:] + base[:h]), strict=True)
            columns.append([min(a, b) for a, b in pairs])
        least = min(tuple(sorted(column)) for column in zip(*columns, strict=True))
        best = least if best is None or least < best else best
    assert best is not None
    return best


CANONICAL: dict[str, Callable[[RowsLike], Rows]] = {VERSION: canonical, VERSION_SHIFTS: canonical_shifts}


def branch_canonical(rows: RowsLike) -> Rows:
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
    return min((*sorted(c[:3]), c[3]) for c in zip(*images, strict=False))


def fingerprint(rows: RowsLike) -> str:
    rep = canonical(rows)
    payload = {
        "equivalence": VERSION,
        "group": {"kind": "cyclic", "invariant_factors": [len(rep[0])]},
        "rows": rep,
    }
    return "gs-" + sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
