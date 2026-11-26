"""A symmetric array for a quadruple whose rows are tied by a multiplier s with s² = 1.

The rows are functions on the integers modulo n. One row is unchanged by
k ↦ −sk, one is unchanged by k ↦ sk, and the other two are carried to each
other by k ↦ sk. For s = 1 these are the conditions of the propus array
(site/propus.py): a symmetric row and two equal rows. For s different from 1
and −1 the two middle rows are different rows, and the array below takes the
place of the propus array.

The array is not from a published source. It was worked out for this board on
2026-11-25 as a variant of the propus array of seberry2015propus and
balonin2017construction, and the entry page says so, gives an argument that is
marked as ours and unreviewed, and rests the claim on the exact check of the
build, which this module runs before a matrix file is written.
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from propus import check_symmetric_hadamard
from skew import matrix_text

if TYPE_CHECKING:
    from references import Bibliography

type Matrix = list[list[int]]

# The references for the propus array, of which this array is a variant.
ARRAY_SOURCES = ("seberry2015propus", "balonin2017construction")
# The array: a sign and a letter for each block. T, U and V are the three
# matrices of a row defined in array(); w, y, c and z are the row unchanged by
# k ↦ −sk, one of the two tied rows, the other one, and the row unchanged by
# k ↦ sk.
BANDS = (
    ((-1, "T", "w"), (1, "U", "y"), (1, "U", "c"), (1, "U", "z")),
    ((1, "U", "c"), (1, "V", "z"), (1, "T", "w"), (-1, "V", "y")),
    ((1, "U", "y"), (1, "T", "w"), (-1, "V", "z"), (1, "V", "c")),
    ((1, "U", "z"), (-1, "V", "c"), (1, "V", "y"), (1, "T", "w")),
)


def involutions(n: int) -> list[int]:
    """The s with 1 < s < n − 1 and s² = 1 modulo n, in increasing order."""
    return [s for s in range(2, n - 1) if s * s % n == 1]


def carried(row: Sequence[int], t: int) -> list[int]:
    """The row k ↦ row[tk mod n]."""
    n = len(row)
    return [row[t * k % n] for k in range(n)]


def shape(rows: Sequence[Sequence[int]]) -> tuple[int, int, int, int, int] | None:
    """(s, w, y, c, z) for a quadruple tied by a multiplier s, or None.

    s is the smallest s with 1 < s < n − 1 and s² = 1 modulo n for which
    positions exist with rows[w] unchanged by k ↦ −sk, rows[z] unchanged by
    k ↦ sk and rows[c] equal to rows[y] carried by k ↦ sk, where y < c; w and z
    are the smallest such positions.
    """
    n = len(rows[0])
    for s in involutions(n):
        for w in range(4):
            if carried(rows[w], -s) != list(rows[w]):
                continue
            for z in range(4):
                if z == w or carried(rows[z], s) != list(rows[z]):
                    continue
                y, c = (i for i in range(4) if i not in (w, z))
                if carried(rows[y], s) == list(rows[c]):
                    return s, w, y, c, z
    return None


def array(w: Sequence[int], y: Sequence[int], z: Sequence[int], s: int) -> Matrix:
    """The array of the rows w, y, c, z on the integers modulo n, where c[k] = y[sk].

    For a row x, the matrices T(x), U(x) and V(x) of order n have the entries
    x[sj − i], x[−sj − i] and x[sj + i] in row i and column j, with the indices
    read modulo n. The block rows are those of BANDS: (−T(w), U(y), U(c), U(z)),
    (U(c), V(z), T(w), −V(y)), (U(y), T(w), −V(z), V(c)) and
    (U(z), −V(c), V(y), T(w)). The function builds the matrix from this
    definition and checks nothing.
    """
    n = len(w)
    if len(y) != n or len(z) != n:
        raise ValueError("the rows have one length")
    rows = {"w": w, "y": y, "c": carried(y, s), "z": z}
    index = {
        "T": [[(s * j - i) % n for j in range(n)] for i in range(n)],
        "U": [[(-s * j - i) % n for j in range(n)] for i in range(n)],
        "V": [[(s * j + i) % n for j in range(n)] for i in range(n)],
    }
    return [
        [sign * rows[name][k] for sign, kind, name in band for k in index[kind][i]] for band in BANDS for i in range(n)
    ]


def multipliers(rows: Sequence[Sequence[int]]) -> list[int]:
    """The units t modulo n with x[tk] = x[k] for every k and every one of the rows x, in increasing order."""
    n = len(rows[0])
    return [t for t in range(1, n) if math.gcd(t, n) == 1 and all(carried(x, t) == list(x) for x in rows)]


def file_name(ident: str) -> str:
    """The matrix file of an entry; named by the entry, since several entries can share an order."""
    return f"matrices/symmetric-{ident}.txt"


def matrix_file(rows: Sequence[Sequence[int]]) -> str:
    """The text of the array of a tied quadruple, after the exact check. Raises ValueError otherwise."""
    found = shape(rows)
    if found is None:
        raise ValueError("the rows are not tied by a multiplier s with s² = 1 other than 1 and −1")
    s, w, y, _, z = found
    h = array(rows[w], rows[y], rows[z], s)
    check_symmetric_hadamard(h)
    return matrix_text(h)


def array_text(names: dict[str, str], forms: dict[str, str]) -> str:
    """The array as printed on the page: names gives the page's letter for each row, forms the text of each kind
    of block with {} for the letter."""
    cells = [
        [("−" if sign < 0 else " ") + forms[kind].format(names[name]) for sign, kind, name in band] for band in BANDS
    ]
    width = max(len(cell) for line in cells for cell in line)
    return "\n".join(" ".join(cell.ljust(width) for cell in line).rstrip() for line in cells)


def block(ident: str, rows: Sequence[Sequence[int]], bib: Bibliography, prefix: str = "../") -> tuple[str, set[str]]:
    """The entry-page block about the symmetric Hadamard matrix of a tied quadruple, and the keys it cites.

    rows are the ones the page lists as A, B, C, D. The block states only what
    shape and multipliers compute from them.
    """
    found = shape(rows)
    if found is None:
        raise ValueError("the rows are not tied by a multiplier s with s² = 1 other than 1 and −1")
    s, *positions = found
    w, y, c, z = ("ABCD"[i] for i in positions)
    n = len(rows[0])
    order, last = 4 * n, n - 1
    # s is 1 modulo m1 and −1 modulo m2, and n = m1 m2 with m1 and m2 coprime, since n is odd.
    m1, m2 = math.gcd(s - 1, n), math.gcd(s + 1, n)
    if m1 * m2 != n or math.gcd(m1, m2) != 1:
        raise ValueError("the page describes a multiplier that is 1 modulo one factor of n and −1 modulo a coprime one")
    group = multipliers(rows)

    def times(factor: int) -> str:
        """A factor before a variable; 1 is not written."""
        return "" if factor == 1 else str(factor)

    h = len(group)
    generator = next((t for t in group if len({pow(t, k, n) for k in range(h)}) == h), None)
    pairs = ""
    if h == 1:
        symmetry = ""
    elif generator is None:
        symmetry = f" All four rows are also unchanged by a group of {h} multipliers: x(tk) = x(k) for each of them."
    else:
        symmetry = (
            f" All four rows are also unchanged by multiplication by {generator}, which has order {h} modulo {n}: "
            f"x({generator}k) = x(k) for every k."
        )
        pairs = (
            f" Multiplication by {generator} is the map (x<sub>1</sub>, x<sub>2</sub>) ↦ "
            f"({times(generator % m1)}x<sub>1</sub>, {times(generator % m2)}x<sub>2</sub>)."
        )
    cites = " ".join(bib.mark(key, prefix) for key in ARRAY_SOURCES)
    t = "<sup>T</sup>"
    names = {"w": w, "y": y, "c": c, "z": z}
    pattern = array_text(names, {"T": "T({})", "U": "U({})", "V": "V({})"})
    plain = array_text(names, {"T": "{}", "U": "{}N", "V": "{}ᵀN"})
    return (
        f'<section class="rows-text"><h2>Symmetric Hadamard matrix of order {order}</h2>'
        f"<p>Read a row x as the function on the integers modulo {n} whose value x(k) is the entry in position k, "
        f"counted from 0, and let s = {s}, for which s² = 1 modulo {n}. Row {w} is unchanged by k ↦ −sk, that is, "
        f"x(−sk) = x(k) for every k; row {z} is unchanged by k ↦ sk; and rows {y} and {c} are carried to each other "
        f"by it: {c}(k) = {y}(sk) for every k.{symmetry}</p>"
        f"<p>Since {n} = {m1} × {m2} and {m1} and {m2} have no common factor, an integer k modulo {n} is the same "
        f"thing as the pair (x<sub>1</sub>, x<sub>2</sub>) of its remainders modulo {m1} and modulo {m2}, and the "
        f"integers modulo {n} are the group Z<sub>{m1}</sub> × Z<sub>{m2}</sub> written with one number in place of "
        f"two. In terms of the pairs, k ↦ sk is the map (x<sub>1</sub>, x<sub>2</sub>) ↦ (x<sub>1</sub>, "
        f"−x<sub>2</sub>).{pairs} The search that found the rows worked on such pairs. The rows here are on the "
        f"integers modulo {n}, in the board's normal form of the quadruple, which applies one multiplier to the "
        "positions of all four rows and may negate, reverse and reorder rows.</p>"
        f"<p>For a row x let T(x), U(x) and V(x) be the matrices of order {n} whose entries in row i and column j, "
        f"for i and j from 0 to {last}, are x(sj − i), x(−sj − i) and x(sj + i), with the arguments read modulo "
        f"{n}. The matrix H of order {order} is the block matrix</p><pre>{pattern}</pre>"
        f"<p>The three matrices are a circulant and two back-circulant matrices with their columns reordered: "
        f"column j of T(x) is column sj of the circulant matrix of x, whose entry is x(j − i), column j of V(x) is "
        f"column sj of the back-circulant matrix with entry x(i + j), and U(x) is V of the reversed row "
        f"k ↦ x(−k).</p>"
        f"<p>This array is not taken from a published source. It is a variant of the propus array {cites}, which is "
        f"the case s = 1: the conditions then say that one row is symmetric and that two rows are equal, where here "
        f"those two are different rows. It was worked out for this board on 2026-11-25. The "
        "statement that H is a symmetric Hadamard matrix rests on the exact check of the build.</p>"
        f"<p>The argument that follows is ours and has not been reviewed. Let S be the permutation matrix of order "
        f"{n} with a 1 in row sj and column j, and N the one with a 1 in row −j and column j, for j = 0, …, {last}, "
        f"both read modulo {n}. Each of S and N is symmetric and is its own inverse, and SN = NS. For a row x let X "
        f"be its circulant matrix, with entry x(j − i), and write a superscript T for the transpose. Then "
        f"T(x) = XS, U(x) = XNS and V(x) = X{t}NS, so every block of H is a block of the matrix Q below multiplied "
        f"on the right by S, where {w}, {y}, {c} and {z} now stand for the circulant matrices of the four rows:</p>"
        f"<pre>{plain}</pre>"
        f"<p>Circulant matrices commute with one another and with one another's transposes, and NX = X{t}N for a "
        f"circulant matrix X. A block row of Q times its own transpose is therefore the sum of XX{t} over the four "
        "rows, and XX" + t + " is the circulant matrix of the periodic autocorrelation of x, the sums of x(k) x(k + r) "
        f"over all k; these four autocorrelations add to {order} at r = 0 and to 0 at every other r, which is what "
        f"the verifier checks, so the sum is {order}I. A block row of Q times the transpose of a different one is a "
        f"sum of four products that cancel in two pairs by the same rules: for the first two rows it is "
        f"−{w}{c}N + {y}{z} + {c}{w}N − {z}{y} = 0. So QQ{t} = {order}I, for any four rows with these "
        f"autocorrelations, and HH{t} = {order}I because multiplying every block on the right by S multiplies Q on "
        f"the right by a permutation matrix. For H = H{t}: the block of H in position (i, j) is the block "
        f"Q<sub>ij</sub> of Q times S, so H equals its transpose exactly when SQ<sub>ij</sub>S is the transpose of "
        f"Q<sub>ji</sub> for all i and j. SXS is the circulant matrix of the row k ↦ x(sk), and S commutes with N. "
        f"Where both positions hold ±{w} the condition is that {w} carried by s is the reversed row, x(sk) = x(−k), "
        f"which is the condition on row {w}. Where both hold {z}N or {z}{t}N it is that row {z} is unchanged by s. "
        f"Where one holds {y}N and the other {c}N, or one {y}{t}N and the other {c}{t}N, it is that row {y} carried "
        f"by s is row {c}. These are the ten pairs of positions.</p>"
        f"<p>The Goethals–Seidel array of the same rows, drawn below, is a Hadamard matrix that is not symmetric.</p>"
        f'<p><a href="{prefix}{file_name(ident)}" download>Download the matrix</a>: one row per line, + for 1 and - '
        "for −1. The build rebuilds H from the four rows by the definitions above and checks H = H<sup>T</sup> and "
        "H H<sup>T</sup> = nI exactly, with n the order, before it writes the file.</p></section>",
        set(ARRAY_SOURCES),
    )
