#!/usr/bin/env python3
"""Build one Williamson quadruple at every even n that two published constructions reach from board entries.

The constructions are those of Bright, Kotsireas and Ganesh, read in the arXiv
versions of the two papers (keys bright2020williamson and bright2020new in
baselines/references.bib). Sequences are indexed from 0 and written as tuples
of +1 and -1.

Doubling (bright2020williamson, Section 2.6). For Williamson sequences A, B,
C, D of odd length m, the four sequences A x B', (-A) x B', C x D', (-C) x D'
are Williamson sequences of length 2m, where x interleaves two sequences and
X' is the cyclic shift of X that is palindromic. The paper prints the shift as
[x_(m-1)/2, ..., x_(m-1), x_0, ..., x_(m-3)/2], which is not palindromic for a
symmetric X; its proof uses y_k = y_(m-k-1), which holds for the shift by
(m-1)/2 places to the right, [x_(m+1)/2, ..., x_(m-1), x_0, ..., x_(m-1)/2],
and that is the one used here.

Powers of two (bright2020new, the proof that symmetric perfect sequences over
the quaternion group exist at every length 2^t). Williamson sequences of
length 4n come from Williamson sequences W of even length n and
antipalindromic nega Williamson sequences V of length n as d(W_i) x n(V_i),
with d(X) = [X; X] and n(X) = [X; -X]. The sequences V come from a Golay pair
(A, B) of length n/4 as [A; B; B~; A~], [B~; A~; A; B], [-B~; A~; A; -B],
[-A; B; B~; -A~], each followed by n/2 negacyclic shifts, where X~ is X
reversed. The Golay pairs are (+), (+) and its images under
(A, B) -> (A x B, A x -B).

Product (bright2020new, the periodic product of perfect sequences with the
correspondence of Barrera Acevedo and Dietrich that the paper tabulates). A
quadruple of sequences corresponds entry by entry to a sequence over sixteen
quaternions. If X is a Williamson quadruple of even length e whose entries
satisfy a_r b_r c_r d_r = 1 for every r, Y is a Williamson quadruple of odd
length m, and e and m are coprime, then the sequence with entries
x_(r mod e) y_(r mod m) corresponds to a Williamson quadruple of length em.

Nothing is taken on trust from these statements: every quadruple is checked
exactly by the board's verifier and must be symmetric before it is written,
and an order whose output fails is reported and skipped.

    uv run python tools/construct_even_williamson.py --date YYYY-MM-DD [--max 300] [--dry-run]
"""

import argparse
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store
from verify import catalogue
from verify.common import ROOT, JSONObject, Row, Rows, read_json
from verify.equivalence import canonical, fingerprint
from verify.gs_verify import memberships, validate_rows
from verify.model import validate

SUBMITTER = "vprusso"
DOUBLING = {
    "title": "Bright, Kotsireas, Ganesh, Applying computer algebra systems with SAT solvers to the Williamson "
    "conjecture, J. Symbolic Comput. 100 (2020), Section 2.6: the doubling construction",
    "url": "https://doi.org/10.1016/j.jsc.2019.07.024",
}
PRODUCT = {
    "title": "Bright, Kotsireas, Ganesh, New infinite families of perfect quaternion sequences and Williamson "
    "sequences, IEEE Trans. Inform. Theory 66 (2020): Williamson sequences of length 2^t, and the periodic "
    "product of a perfect sequence over the quaternion group with a Williamson quadruple of coprime odd length",
    "url": "https://doi.org/10.1109/TIT.2020.3016510",
}
type Quaternion = tuple[int, int, int, int]
"""Twice a quaternion, as its integer coordinates on 1, i, j, k."""


def interleave(x: Row, y: Row) -> Row:
    return tuple(v for pair in zip(x, y, strict=True) for v in pair)


def negate(x: Row) -> Row:
    return tuple(-v for v in x)


def double(x: Row) -> Row:
    return x + x


def negadouble(x: Row) -> Row:
    return x + negate(x)


def negacyclic_shift(x: Row, times: int) -> Row:
    """[x_0, ..., x_(n-1)] -> [-x_(n-1), x_0, ..., x_(n-2)], applied the given number of times."""
    for _ in range(times):
        x = (-x[-1], *x[:-1])
    return x


def has_unit_products(q: Rows) -> bool:
    """Whether a_r b_r c_r d_r = 1 at every position, the hypothesis of the product on its even factor."""
    return all(a * b * c * d == 1 for a, b, c, d in zip(*q, strict=True))


def with_unit_products(q: Rows) -> Rows | None:
    """An equivalent quadruple with a_r b_r c_r d_r = 1 everywhere, or None when the products are not constant."""
    if has_unit_products(q):
        return q
    flipped = (negate(q[0]), *q[1:])
    return flipped if has_unit_products(flipped) else None


def doubling(q: Rows) -> Rows:
    """Williamson sequences of length 2m from Williamson sequences of odd length m."""
    a, b, c, d = q
    m = len(a)
    if m % 2 == 0:
        raise ValueError("the doubling construction needs odd length")
    k = (m + 1) // 2
    bp, dp = b[k:] + b[:k], d[k:] + d[:k]
    return (interleave(a, bp), interleave(negate(a), bp), interleave(c, dp), interleave(negate(c), dp))


def golay_pair(t: int) -> tuple[Row, Row]:
    """A Golay pair of length 2^t by Golay's interleaving."""
    a: Row = (1,)
    b: Row = (1,)
    for _ in range(t):
        a, b = interleave(a, b), interleave(a, negate(b))
    return a, b


def antipalindromic_nega(t: int) -> Rows:
    """Antipalindromic nega Williamson sequences of length 2^t, t >= 1."""
    if t == 1:
        return ((1, -1),) * 4
    a, b = golay_pair(t - 2)
    ar, br = a[::-1], b[::-1]
    palindromic = (a + b + br + ar, br + ar + a + b, negate(br) + ar + a + negate(b), negate(a) + b + br + negate(ar))
    return tuple(negacyclic_shift(x, len(x) // 2) for x in palindromic)


def power_of_two(t: int) -> Rows:
    """Williamson sequences of length 2^t with unit products, by the induction from lengths 1, 2 and 4."""
    if t == 0:
        return ((1,),) * 4
    if t == 1:
        return ((1, -1), (1, -1), (1, 1), (1, 1))
    if t == 2:
        return ((1, 1, -1, 1),) * 4
    shorter, nega = power_of_two(t - 2), antipalindromic_nega(t - 2)
    return tuple(interleave(double(x), negadouble(y)) for x, y in zip(shorter, nega, strict=True))


def qmul(x: Quaternion, y: Quaternion) -> Quaternion:
    """Twice the product of two quaternions given as twice their coordinates."""
    a1, b1, c1, d1 = x
    a2, b2, c2, d2 = y
    doubled = (
        a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
        a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
        a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
        a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
    )
    if any(v % 2 for v in doubled):
        raise ValueError("quaternion product left the lattice of the correspondence")
    return (doubled[0] // 2, doubled[1] // 2, doubled[2] // 2, doubled[3] // 2)


def correspondence() -> dict[tuple[int, ...], Quaternion]:
    """Entries (a_r, b_r, c_r, d_r) to twice the quaternion s_r, as tabulated in bright2020new.

    The table gives 1, i, j, k, q, qi, qj, qk with q = (1 + i + j + k)/2, and
    the negated entries map to the negated quaternion.
    """
    q: Quaternion = (1, 1, 1, 1)
    table: dict[tuple[int, ...], Quaternion] = {
        (-1, -1, -1, -1): (2, 0, 0, 0),
        (1, -1, -1, 1): (0, 2, 0, 0),
        (1, 1, -1, -1): (0, 0, 2, 0),
        (1, -1, 1, -1): (0, 0, 0, 2),
        (1, -1, -1, -1): q,
        (1, 1, -1, 1): qmul(q, (0, 2, 0, 0)),
        (1, 1, 1, -1): qmul(q, (0, 0, 2, 0)),
        (1, -1, 1, 1): qmul(q, (0, 0, 0, 2)),
    }
    for entries, value in list(table.items()):
        table[tuple(-v for v in entries)] = (-value[0], -value[1], -value[2], -value[3])
    if len(table) != 16 or len(set(table.values())) != 16:
        raise ValueError("the correspondence is not a bijection")
    return table


def product(x: Rows, y: Rows) -> Rows:
    """The quadruple of length em that corresponds to the entrywise quaternion product of x and y."""
    e, m = len(x[0]), len(y[0])
    if gcd(e, m) != 1 or not has_unit_products(x):
        raise ValueError("the product needs coprime lengths and unit products in its first factor")
    table = correspondence()
    back = {value: entries for entries, value in table.items()}
    xs = [table[entries] for entries in zip(*x, strict=True)]
    ys = [table[entries] for entries in zip(*y, strict=True)]
    columns = [back[qmul(xs[r % e], ys[r % m])] for r in range(e * m)]
    return tuple(tuple(column[i] for column in columns) for i in range(4))


def board_williamson() -> dict[int, list[tuple[str, Rows]]]:
    """The Williamson quadruples on the board by length, each list in id order, with rows as stored."""
    found: dict[int, list[tuple[str, Rows]]] = {}
    for path in sorted((ROOT / "quadruples").glob("*.json")):
        doc = read_json(path)
        if "rows" not in doc:
            continue
        rows = validate_rows(doc["rows"], doc["n"])
        if "williamson" in memberships(rows):
            found.setdefault(doc["n"], []).append((doc["id"], canonical(rows)))
    return found


def catalogue_seed(e: int) -> Rows | None:
    """A quadruple with unit products from the board's catalogue of Williamson quadruples of length e.

    It is the first line of the file that has unit products after shifting some of its rows by half their
    length, with those rows shifted.
    """
    path = ROOT / f"evidence/catalogues/williamson-n{e}.txt.gz"
    if not path.is_file():
        return None
    h = e // 2
    for line in catalogue.lines(path):
        rows = validate_rows(catalogue.parse_line(line, e), e)
        # A catalogue may hold one quadruple per published class, and the products change when a single row
        # is shifted by half its length, which stays inside that class: try the sixteen choices of rows.
        for mask in range(16):
            image = tuple(row[h:] + row[:h] if mask >> i & 1 else row for i, row in enumerate(rows))
            if unit := with_unit_products(image):
                return unit
    return None


def recipe(n: int, board: dict[int, list[tuple[str, Rows]]]) -> tuple[Rows, JSONObject] | None:
    """A Williamson quadruple of even length n and how it was built, or None when no construction here reaches n.

    The order of preference is: a power of two; the doubling of an odd board
    entry; the product of the power of two dividing n with an odd board entry;
    the product of an even quadruple with unit products and an odd board entry
    of coprime length, taking the least even length that works. The even
    factor is a board entry of that length, the doubling of an odd one, or
    failing both a line of the board's catalogue of that length.
    """
    t = (n & -n).bit_length() - 1
    odd = n >> t
    if odd == 1:
        return power_of_two(t), {"kind": "power-of-two", "t": t}
    if t == 1 and odd in board:
        ident, rows = board[odd][0]
        return doubling(rows), {"kind": "doubling", "m": odd, "entry": ident}
    if odd in board:
        ident, rows = board[odd][0]
        return product(power_of_two(t), rows), {"kind": "product", "e": 1 << t, "m": odd, "entry": ident, "t": t}
    for e in range(2, n, 2):
        m = n // e
        if n % e or m % 2 == 0 or m == 1 or gcd(e, m) != 1 or m not in board:
            continue
        seed: tuple[str, Rows, str] | None = None
        for ident, rows in board.get(e, []):
            if unit := with_unit_products(rows):
                seed = (ident, unit, "entry")
                break
        if seed is None and e % 4 == 2 and e // 2 in board:
            ident, rows = board[e // 2][0]
            seed = (ident, doubling(rows), "doubling")
        if seed is None and (found := catalogue_seed(e)):
            seed = (f"williamson-n{e}", found, "catalogue")
        if seed is None or not has_unit_products(seed[1]):
            continue
        other, rows = board[m][0]
        return product(seed[1], rows), {
            "kind": "product",
            "e": e,
            "m": m,
            "entry": other,
            "even": seed[0],
            "via": seed[2],
        }
    return None


def describe(n: int, how: JSONObject) -> tuple[str, str, list[JSONObject]]:
    """The discovery credit, the notes and the sources of a constructed entry."""
    made = "rows generated by tools/construct_even_williamson.py and stored in canonical form"
    if how["kind"] == "power-of-two":
        credit = (
            "Bright, Kotsireas, Ganesh 2020, Williamson sequences of every length 2^t from Golay pairs and nega "
            f"Williamson sequences; {made}"
        )
        notes = (
            f"n = 2^{how['t']}. Built by the induction of the cited paper from lengths 1, 2 and 4: a quadruple of "
            "length 4k is d(W) x n(V) for a Williamson quadruple W of length k and antipalindromic nega Williamson "
            "sequences V of length k, which come from a Golay pair. Every entry of the four rows has product 1 "
            "at each position."
        )
        return credit, notes, [PRODUCT]
    if how["kind"] == "doubling":
        credit = (
            "Bright, Kotsireas, Ganesh 2020, the doubling construction, applied to the Williamson quadruple "
            f"{how['entry']} of length {how['m']} on this board, whose discovery is credited on its own entry; {made}"
        )
        notes = (
            f"n = 2 x {how['m']}. With A, B, C, D the rows of {how['entry']}, the rows are A x B', (-A) x B', "
            "C x D', (-C) x D', where x interleaves two sequences and X' is X shifted cyclically to the "
            "palindromic position."
        )
        return credit, notes, [DOUBLING]
    even = (
        f"the Williamson quadruple of length 2^{how['t']} of the cited paper"
        if "t" in how
        else f"the doubling of the board entry {how['even']}"
        if how["via"] == "doubling"
        else f"a quadruple of the catalogue of the census {how['even']}, the first that has product 1 at every "
        "position after some of its rows are shifted by half their length"
        if how["via"] == "catalogue"
        else f"the board entry {how['even']}, with one row negated where needed so that every position has product 1"
    )
    credit = (
        "Bright, Kotsireas, Ganesh 2020, the periodic product of perfect quaternion sequences with the "
        "correspondence of Barrera Acevedo and Dietrich cited there, applied to the Williamson quadruple "
        f"{how['entry']} of length {how['m']} on this board, whose discovery is credited on its own entry; {made}"
    )
    notes = (
        f"n = {how['e']} x {how['m']}, coprime. The even factor is {even}; the odd factor is {how['entry']}. Each "
        "quadruple is read position by position as a quaternion through the correspondence tabulated in the cited "
        "paper, the two sequences are multiplied entrywise with indices taken modulo their lengths, and the "
        "product is read back as four rows."
    )
    sources = [PRODUCT, DOUBLING] if how.get("via") == "doubling" else [PRODUCT]
    return credit, notes, sources


def construct(max_n: int, date: str, dry_run: bool) -> None:
    board = board_williamson()
    written = skipped = 0
    missing = []
    for n in range(2, max_n + 1, 2):
        if n in board:
            continue
        built = recipe(n, board)
        if built is None:
            missing.append(n)
            continue
        rows, how = built
        try:
            rep = canonical(validate_rows([list(r) for r in rows], n))
            if "williamson" not in memberships(rep):
                raise ValueError("the rows are not all symmetric")
        except ValueError as error:
            print(f"n={n}: {how['kind']} failed the exact check: {error}")
            missing.append(n)
            continue
        credit, notes, sources = describe(n, how)
        ident = f"n{n}-{fingerprint(rep)[3:23]}"
        doc = validate(
            {
                "schema_version": 1,
                "n": n,
                "group": {"kind": "cyclic", "invariant_factors": [n]},
                "submitted_by": SUBMITTER,
                "submitted_at": date,
                "discovered_by": [credit],
                "sources": sources,
                "equivalence": "equiv-v1",
                "id": ident,
                "type": "quadruple",
                "rows": [list(r) for r in rep],
                "notes": notes,
            }
        )
        detail = ", ".join(f"{k}={v}" for k, v in how.items())
        if (ROOT / f"quadruples/{ident}.json").exists():
            skipped += 1
        elif not dry_run:
            store(ROOT / f"quadruples/{ident}.json", doc)
            written += 1
        print(f"n={n}: {ident} ({detail})")
    print(f"{written} written, {skipped} already present; no construction here reaches n = {missing}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True)
    p.add_argument("--max", type=int, default=300)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    construct(a.max, a.date, a.dry_run)
