"""The propus array of a quadruple with a symmetric block and two equal blocks.

The Goethals–Seidel array of such a quadruple is a Hadamard matrix that is not
symmetric in general. The symmetric Hadamard matrix comes from the propus
array, which this module builds, checks exactly and describes on the entry
page. The array is the one for commuting blocks with one symmetric block in
seberry2015propus; balonin2017construction gives the same construction as the
Goethals–Seidel array with its first column negated and two rows exchanged.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from skew import matrix_text

if TYPE_CHECKING:
    from references import Bibliography

type Matrix = list[list[int]]

# The references for the propus array.
ARRAY_SOURCES = ("seberry2015propus", "balonin2017construction")
ARRAY_TEXT = " X    YR    YR    ZR\n YR   ZᵀR  −X    −YᵀR\n YR  −X    −ZᵀR   YᵀR\n ZR  −YᵀR   YᵀR  −X"


def shape(rows: Sequence[Sequence[int]], pattern: str) -> tuple[int, int, int, int] | None:
    """Positions (symmetric row, the two equal rows, the remaining row) of a propus quadruple, or None."""
    for i in range(4):
        if pattern[i] != "s":
            continue
        j, k, m = (x for x in range(4) if x != i)
        for pair, other in (((j, k), m), ((j, m), k), ((k, m), j)):
            if list(rows[pair[0]]) == list(rows[pair[1]]):
                return i, pair[0], pair[1], other
    return None


def array(x: Sequence[int], y: Sequence[int], z: Sequence[int]) -> Matrix:
    """The propus array of the rows x, y, y, z, where x is symmetric.

    X, Y, Z are the circulants with entry (i, j) equal to the row's entry
    (j - i) mod n, and R is the back-diagonal permutation matrix: M R reverses
    the columns of M. The block rows are (X, YR, YR, ZR), (YR, Z^T R, -X, -Y^T R),
    (YR, -X, -Z^T R, Y^T R) and (ZR, -Y^T R, Y^T R, -X).
    """
    n = len(x)
    if len(y) != n or len(z) != n:
        raise ValueError("the rows of a propus family have one length")

    def circ(row: Sequence[int]) -> Matrix:
        return [[row[(j - i) % n] for j in range(n)] for i in range(n)]

    def r(m: Matrix) -> Matrix:
        return [line[::-1] for line in m]

    def t(m: Matrix) -> Matrix:
        return [list(column) for column in zip(*m, strict=True)]

    def neg(m: Matrix) -> Matrix:
        return [[-v for v in line] for line in m]

    cx, cy, cz = circ(x), circ(y), circ(z)
    yr, zr, ytr, ztr = r(cy), r(cz), r(t(cy)), r(t(cz))
    blocks = [
        [cx, yr, yr, zr],
        [yr, ztr, neg(cx), neg(ytr)],
        [yr, neg(cx), neg(ztr), ytr],
        [zr, neg(ytr), ytr, neg(cx)],
    ]
    return [[v for block in band for v in block[i]] for band in blocks for i in range(n)]


def check_symmetric_hadamard(h: Matrix) -> None:
    """Raise ValueError unless H has entries 1 and -1, H = H^T and H H^T = mI, for H of order m.

    The arithmetic is exact. Each row is packed into an integer with a bit set
    where the entry is -1; the inner product of two rows is m minus twice the
    number of positions in which they differ, and every pair of rows is compared.
    """
    m = len(h)
    if m == 0 or any(len(row) != m or any(v not in (1, -1) for v in row) for row in h):
        raise ValueError("not a square matrix with entries 1 and -1")
    if any(h[i][j] != h[j][i] for i in range(m) for j in range(i)):
        raise ValueError("H differs from its transpose")
    rows = [int("".join("1" if v < 0 else "0" for v in row), 2) for row in h]
    half, odd = divmod(m, 2)
    for i in range(1, m):
        for j in range(i):
            if odd or (rows[i] ^ rows[j]).bit_count() != half:
                raise ValueError(f"rows {j} and {i} of H are not orthogonal")


def file_name(ident: str) -> str:
    """The matrix file of an entry; named by the entry, since several entries can share an order."""
    return f"matrices/propus-{ident}.txt"


def matrix_file(rows: Sequence[Sequence[int]], pattern: str) -> str:
    """The text of the propus array of a quadruple, after the exact check. Raises ValueError otherwise."""
    found = shape(rows, pattern)
    if found is None:
        raise ValueError("not a propus quadruple: it needs a symmetric row and two equal rows")
    s, e, _, o = found
    h = array(rows[s], rows[e], rows[o])
    check_symmetric_hadamard(h)
    return matrix_text(h)


def block(
    ident: str, rows: Sequence[Sequence[int]], pattern: str, bib: Bibliography, prefix: str = "../"
) -> tuple[str, set[str]]:
    """The entry-page block about the symmetric Hadamard matrix of a propus quadruple, and the keys it cites.

    rows and pattern are the ones the page lists as A, B, C, D.
    """
    found = shape(rows, pattern)
    if found is None:
        raise ValueError("not a propus quadruple: it needs a symmetric row and two equal rows")
    s, e1, e2, o = ("ABCD"[i] for i in found)
    order = 4 * len(rows[0])
    cites = " ".join(bib.mark(key, prefix) for key in ARRAY_SOURCES)
    return (
        f'<section class="rows-text"><h2>Symmetric Hadamard matrix of order {order}</h2>'
        f"<p>The rows form a propus family: row {s} is symmetric and rows {e1} and {e2} are equal. "
        f"Let X, Y and Z be the circulant matrices whose first rows are the rows {s}, {e1} and {o}, "
        "let M<sup>T</sup> be the transpose of a matrix M, and let R be the back-diagonal permutation matrix. "
        f"The propus array {cites}</p><pre>{ARRAY_TEXT}</pre>"
        f"<p>is a symmetric Hadamard matrix H of order {order}. The Goethals–Seidel array of the same rows, "
        "drawn below, is a Hadamard matrix that is not symmetric.</p>"
        f'<p><a href="{prefix}{file_name(ident)}" download>Download the matrix</a>: one row per line, + for 1 and - '
        "for −1. The build checks H = H<sup>T</sup> and H H<sup>T</sup> = nI exactly, with n the order, before it "
        "writes the file.</p></section>",
        set(ARRAY_SOURCES),
    )
