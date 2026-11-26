"""Skew-Hadamard matrix files and the block that describes the matrix of one order.

Every order that is not open has a matrix file in the build output. For an order
the board holds, the file is the Goethals–Seidel array of one quadruple. For
every other order it is a file stored in matrices/skew and listed in
baselines/skew-matrices.json. Each is written only after an exact check that it
is a skew-Hadamard matrix. A stored matrix of origin found also carries the
rows it is built from, and is rebuilt from them whenever it is read. A found
matrix of an order above the range of the board has no file: its record holds
the rows and the sha256 of the matrix text, and the build rebuilds it.
"""

import hashlib
import itertools
import lzma
import math
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from orders import STATES, esc, listing, marks

if TYPE_CHECKING:
    from orders import OrderStates
    from references import Bibliography
    from verify.common import JSONObject

type Matrix = list[list[int]]

NAME = "skew-hadamard"
ROOT = Path(__file__).resolve().parents[1]
# The reference for the Goethals–Seidel array.
ARRAY_SOURCE = "goethals1970skew"
# How a parameter of a stored construction is worded; a parameter not named
# here is shown by its value alone.
PARAMETER_WORDS = {
    "q": "q = {}",
    "m": "m = {}",
    "v": "v = {}",
    "group_order": "group of order {}",
    "amicable_order": "amicable matrices of order {}",
    "from_order": "of a matrix of order {}",
}
# The 8-Williamson array as kotsireas2008new prints it: entry k stands for the
# k-th of the blocks A, ..., H and -k for its negative.
WILLIAMSON8 = (
    (1, -2, -3, -4, -5, -6, -7, -8),
    (2, 1, -4, 3, -6, 5, 8, -7),
    (3, 4, 1, -2, -7, -8, 5, 6),
    (4, -3, 2, 1, -8, 7, -6, 5),
    (5, 6, 7, 8, 1, -2, -3, -4),
    (6, -5, 8, -7, 2, 1, 4, -3),
    (7, -8, -5, 6, 3, -4, 1, 2),
    (8, 7, -6, -5, 4, 3, -2, 1),
)
BLOCK_NAMES = "ABCDEFGH"
# The bordered array of wallis1972some for four blocks, in the form momihara2019constructions states it,
# which is -1 times the matrix the first prints, and written for the matrices A, B, C, D of four rows of
# 1 and -1, which are the negatives of the (1, -1)-matrices of the four subsets in both sources. The
# matrix has four rows and columns of border around four bands of four blocks. CORNER is the 4 x 4
# corner. Row i of EDGE gives the sign of the row of ones over each block in row i, and the columns
# left of band i repeat row i of CORNER down the band. An entry of BANDS is a block, - for its
# negative, T for its transpose.
BORDER_CORNER = ((1, -1, -1, -1), (1, 1, 1, -1), (1, -1, 1, 1), (1, 1, -1, 1))
BORDER_EDGE = ((-1, -1, -1, -1), (1, -1, 1, -1), (1, -1, -1, 1), (1, 1, -1, -1))
BORDERED_BANDS = (
    ("A", "B", "C", "D"),
    ("-BT", "AT", "-D", "C"),
    ("-C", "DT", "A", "-BT"),
    ("-DT", "-C", "B", "AT"),
)
SWAP = str.maketrans("+-", "-+")
# The pattern of the eight blocks in the bordered eight-block array, which was not found in the
# sources read: entry k stands for the k-th block and -k for its negative. With commuting symbols its rows
# are orthogonal, block 1 fills the diagonal, the same block stands in positions (i, j) and (j, i),
# and its signs form a skew-Hadamard matrix of order 8; tests/test_orders.py checks each of these.
BORDERED8 = (
    (1, 2, 3, 4, 5, 6, 7, 8),
    (-2, 1, 4, -3, 6, -5, -8, 7),
    (-3, -4, 1, 2, 7, 8, -5, -6),
    (-4, 3, -2, 1, 8, -7, 6, -5),
    (-5, -6, -7, -8, 1, 2, 3, 4),
    (-6, 5, -8, 7, -2, 1, -4, 3),
    (-7, 8, 5, -6, -3, 4, 1, -2),
    (-8, -7, 6, 5, -4, -3, 2, 1),
)
# The site offers a matrix file as a download while its text is at most this many bytes; above that
# it offers the rows. GitHub Pages refuses a file above 100 MB and a site above about 1 GB.
DOWNLOAD_LIMIT = 50_000_000
# The build rebuilds a found matrix without a file, compares its sha256 and runs the exact check up to
# this order. Above it that takes a minute or more for each matrix: the build checks in integers what
# the array asks of the rows, and the tests marked slow rebuild the matrix and run the exact check.
BUILD_REBUILD_MAX = 8_000
CHECKED = "The build checks H H<sup>T</sup> = nI and H + H<sup>T</sup> = 2I exactly, with n the order, before it writes the file."  # noqa: E501


def gs_array(rows: Sequence[Sequence[int]]) -> Matrix:
    """The Goethals–Seidel array of four rows, in the block layout site/picture.js draws.

    Block X is the circulant with entry (i, j) equal to x[(j - i) mod n]; X R
    reverses its columns and R X its rows.
    """
    n = len(rows[0])
    if len(rows) != 4 or any(len(row) != n for row in rows):
        raise ValueError("a quadruple is four rows of one length")

    def circ(x: Sequence[int]) -> Matrix:
        return [[x[(j - i) % n] for j in range(n)] for i in range(n)]

    def xr(m: Matrix) -> Matrix:
        return [row[::-1] for row in m]

    def rx(m: Matrix) -> Matrix:
        return m[::-1]

    def neg(m: Matrix) -> Matrix:
        return [[-v for v in row] for row in m]

    a, b, c, d = (circ(row) for row in rows)
    blocks = [
        [a, xr(b), xr(c), xr(d)],
        [neg(xr(b)), a, neg(rx(d)), rx(c)],
        [neg(xr(c)), rx(d), a, neg(rx(b))],
        [neg(xr(d)), neg(rx(c)), rx(b), a],
    ]
    return [[v for block in band for v in block[i]] for band in blocks for i in range(n)]


def williamson8_array(rows: Sequence[Sequence[int]], signs: Sequence[int]) -> Matrix:
    """The 8-Williamson array W of kotsireas2008new for eight rows of one length v and a sign for each block.

    Let T be the permutation matrix of order v with a 1 in position
    (i, i + 1 mod v) and R the one with ones on the back diagonal. A row x gives
    the circulant x[0] I + x[1] T + ... + x[v-1] T^(v-1), whose entry (i, j) is
    x[(j - i) mod v], and the back-circulant that is this matrix times R, whose
    entry (i, j) is x[(v - 1 - i - j) mod v]. Block A is the circulant of
    rows[0] and blocks B, ..., H are the back-circulants of rows[1], ...,
    rows[7], each multiplied by its sign. The source has every sign equal to 1.
    """
    v = len(rows[0])
    if len(rows) != 8 or len(signs) != 8 or any(len(row) != v for row in rows):
        raise ValueError("the 8-Williamson array takes eight rows of one length and eight signs")
    blocks = [[[signs[0] * rows[0][(j - i) % v] for j in range(v)] for i in range(v)]]
    blocks += [
        [[s * x[(v - 1 - i - j) % v] for j in range(v)] for i in range(v)]
        for s, x in zip(signs[1:], rows[1:], strict=True)
    ]
    negated = [[[-t for t in row] for row in block] for block in blocks]
    return [
        [t for e in band for t in (blocks if e > 0 else negated)[abs(e) - 1][i]]
        for band in WILLIAMSON8
        for i in range(v)
    ]


def label(record: JSONObject) -> str:
    """How an error names a stored matrix: its file, or its order when it has none."""
    return record.get("file") or f"the matrix of order {record['order']}"


def group_elements(moduli: Sequence[int]) -> list[tuple[int, ...]]:
    """The elements of Z_m1 x ... x Z_mk as tuples, in lexicographic order with the last component fastest."""
    return list(itertools.product(*(range(m) for m in moduli)))


@cache
def group_table(moduli: tuple[int, ...]) -> tuple[list[list[int]], list[int]]:
    """The addition table and the negatives of Z_m1 x ... x Z_mk, by position in group_elements.

    With z_0, z_1, ... the elements in that order, table[i][j] is the position
    of z_i + z_j and negative[i] the position of -z_i, both componentwise.
    """
    elements = group_elements(moduli)
    position = {z: i for i, z in enumerate(elements)}
    table = [
        [position[tuple((x + y) % m for x, y, m in zip(z, t, moduli, strict=True))] for t in elements] for z in elements
    ]
    negative = [position[tuple(-x % m for x, m in zip(z, moduli, strict=True))] for z in elements]
    return table, negative


def type_one(row: str, moduli: Sequence[int]) -> list[str]:
    """The type 1 matrix of a function on the group, as lines: entry (i, j) is row[z_j - z_i].

    Over one cyclic group line i is the row moved i places to the right, cyclically.
    """
    if len(moduli) == 1:
        return [row[len(row) - i :] + row[: len(row) - i] for i in range(len(row))]
    table, negative = group_table(tuple(moduli))
    return ["".join(map(row.__getitem__, table[negative[i]])) for i in range(len(row))]


def type_two(row: str, moduli: Sequence[int]) -> list[str]:
    """The type 2 matrix of a function on the group, as lines: entry (i, j) is row[z_j + z_i].

    Over one cyclic group line i is the row moved i places to the left, cyclically.
    """
    if len(moduli) == 1:
        return [row[i:] + row[:i] for i in range(len(row))]
    table, _ = group_table(tuple(moduli))
    return ["".join(map(row.__getitem__, table[i])) for i in range(len(row))]


def bordered_text(rows: Sequence[str], moduli: Sequence[int]) -> str:
    """The bordered array of wallis1972some for four rows of + and - on Z_m1 x ... x Z_mk, as a matrix file.

    The rows are functions on the group, in the order of group_elements. A, B
    and D are the type 1 matrices of rows[0], rows[1] and rows[3], and C is the
    type 2 matrix of rows[2]. The matrix has order 4(w + 1) with w the order of
    the group: BORDER_CORNER in the corner, a row of w equal entries over each
    block with the signs of BORDER_EDGE, the rows of BORDER_CORNER repeated down
    the left of the four bands, and the blocks of BORDERED_BANDS.
    """
    w = math.prod(moduli)
    if len(rows) != 4 or any(len(row) != w or row.strip("+-") for row in rows):
        raise ValueError("the bordered array takes four rows of + and - indexed by the group")
    blocks = {"A": type_one(rows[0], moduli), "B": type_one(rows[1], moduli), "D": type_one(rows[3], moduli)}
    blocks["C"] = type_two(rows[2], moduli)
    for name in "ABD":
        blocks[name + "T"] = ["".join(column) for column in zip(*blocks[name], strict=True)]
    for name in list(blocks):
        blocks["-" + name] = [line.translate(SWAP) for line in blocks[name]]
    sign = {1: "+", -1: "-"}
    lines = [
        "".join(sign[s] for s in corner) + "".join(sign[s] * w for s in edge)
        for corner, edge in zip(BORDER_CORNER, BORDER_EDGE, strict=True)
    ]
    for corner, band in zip(BORDER_CORNER, BORDERED_BANDS, strict=True):
        left = "".join(sign[s] for s in corner)
        lines += [left + "".join(blocks[name][i] for name in band) for i in range(w)]
    return "".join(line + "\n" for line in lines)


def multiplier_orbits(moduli: Sequence[int], generator: Sequence[int]) -> list[list[int]]:
    """The orbits of z -> (g1 z1, ..., gk zk) on the group, as lists of positions in group_elements."""
    elements = group_elements(moduli)
    position = {z: i for i, z in enumerate(elements)}
    seen: set[int] = set()
    orbits = []
    for i, start in enumerate(elements):
        if i in seen:
            continue
        orbit, z = [i], start
        while True:
            z = tuple(g * x % m for g, x, m in zip(generator, z, moduli, strict=True))
            if z == start:
                break
            orbit.append(position[z])
        seen.update(orbit)
        orbits.append(orbit)
    return orbits


def bordered_rows(record: JSONObject) -> list[str]:
    """The four rows of a found matrix from the bordered array, after an exact check of what its record says.

    The record names the group Z_m1 x ... x Z_mk by its moduli, all odd, with
    w their product and 4(w + 1) the order, and gives four rows of + and - of
    length w, read as functions x on the group in the order of group_elements.
    The first is of skew type: x(0) = 1 and x(-z) = -x(z) for z != 0. Every row
    has sum 1, as the record states. Each row is constant on the orbits of the
    multiplier z -> (g1 z1, ..., gk zk) of the record, which has the stated
    order. The periodic autocorrelations of the four rows, the sums of
    x(z) x(z + s) over z, add to -4 at every nonzero s. Raises ValueError otherwise.
    """
    name, order, moduli = label(record), record["order"], record["group"]
    w = math.prod(moduli)
    rows = record["rows"]
    if not moduli or any(m < 3 or m % 2 == 0 for m in moduli) or 4 * (w + 1) != order:
        raise ValueError(f"{name}: the group is not a product of odd cyclic groups of order {order // 4 - 1}")
    if record["parameters"].get("group_order") != w:
        raise ValueError(f"{name}: the parameters do not give the group order {w}")
    if len(rows) != 4 or any(len(row) != w or row.strip("+-") for row in rows):
        raise ValueError(f"{name}: the record does not hold four rows of + and - of length {w}")
    # Over one cyclic group the element at place s + j is z_s + z_j, and no table is needed.
    cyclic = len(moduli) == 1
    table, negative = ([], [-i % w for i in range(w)]) if cyclic else group_table(tuple(moduli))
    first = rows[0]
    if first[0] != "+" or any(first[negative[i]] == first[i] for i in range(1, w)):
        raise ValueError(f"{name}: the first row is not of skew type")
    sums = [row.count("+") - row.count("-") for row in rows]
    if sums != record["row_sums"] or sums != [1, 1, 1, 1]:
        raise ValueError(f"{name}: the row sums are not 1, 1, 1, 1 as the array needs and the record states")
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    if len(g) != len(moduli) or any(math.gcd(x, m) != 1 for x, m in zip(g, moduli, strict=True)):
        raise ValueError(f"{name}: the multiplier {g} is not a unit in each component")
    orbits = multiplier_orbits(moduli, g)
    if math.lcm(*(len(orbit) for orbit in orbits)) != h:
        raise ValueError(f"{name}: the multiplier {g} does not have order {h}")
    if any(len({row[i] for i in orbit}) != 1 for row in rows for orbit in orbits):
        raise ValueError(f"{name}: a row is not fixed by the multiplier {g}")
    bits = str.maketrans("+-", "01")
    packed = [int(row.translate(bits), 2) for row in rows]
    for s in range(1, w):
        # A row and its translate by z_s differ in d places, and their inner product is w - 2d.
        if cyclic:
            shifted = (int((row[s:] + row[:s]).translate(bits), 2) for row in rows)
        else:
            shifted = (int("".join(map(row.__getitem__, table[s])).translate(bits), 2) for row in rows)
        differ = sum((x ^ y).bit_count() for x, y in zip(packed, shifted, strict=True))
        if 4 * w - 2 * differ != -4:
            raise ValueError(f"{name}: the periodic autocorrelations of the rows do not add to -4")
    if "tie" in record:
        tie_classes(record)
    return rows


def tie_classes(record: JSONObject) -> int:
    """How many classes of nonzero shifts the tie of a bordered record leaves, after an exact check of the tie.

    A tie names a multiplier t = (t1, ..., tk), a unit in each component, with
    the map f: z -> (t1 z1, ..., tk zk) of the group, and sorts the rows 1 to 4
    into chains and fixed rows, each row once. A row x carried by f is the row
    y with y(f(z)) = x(z). Along a chain each row carried by f is the next, and
    the last row carried by f is the first row or the first row read backwards,
    x(-z); a fixed row carried by f is itself or itself read backwards. For a
    chain of n rows, f applied n times is z -> hz or z -> -hz with h in the
    group generated by the multiplier of the rows. Then f permutes the periodic
    autocorrelations of the rows up to the sign of the shift, so their sum is
    the same at s and at f(s). The classes counted are those of the nonzero
    elements under f, the multiplier of the rows and z -> -z. Raises ValueError
    if the tie does not hold.
    """
    name, moduli, rows = label(record), record["group"], record["rows"]
    t, chains, fixed = record["tie"]["multiplier"], record["tie"]["chains"], record["tie"].get("fixed", [])
    if len(t) != len(moduli) or any(math.gcd(x, m) != 1 for x, m in zip(t, moduli, strict=True)):
        raise ValueError(f"{name}: the tie multiplier {t} is not a unit in each component")
    if sorted([*fixed, *(k for chain in chains for k in chain)]) != [1, 2, 3, 4] or any(len(c) < 2 for c in chains):
        raise ValueError(f"{name}: the tie does not sort the rows 1 to 4 into chains of two or more and fixed rows")
    elements = group_elements(moduli)
    position = {z: i for i, z in enumerate(elements)}
    negative = [position[tuple(-x % m for x, m in zip(z, moduli, strict=True))] for z in elements]

    def scaled(factor: Sequence[int], i: int) -> int:
        return position[tuple(x * c % m for x, c, m in zip(factor, elements[i], moduli, strict=True))]

    image = [scaled(t, i) for i in range(len(elements))]
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    group = {tuple(pow(x, k, m) for x, m in zip(g, moduli, strict=True)) for k in range(h)}

    def carried(row: str) -> str:
        out = [""] * len(row)
        for i, c in enumerate(row):
            out[image[i]] = c
        return "".join(out)

    def backwards(row: str) -> str:
        return "".join(row[negative[i]] for i in range(len(row)))

    for chain in chains:
        for a, b in itertools.pairwise(chain):
            if carried(rows[a - 1]) != rows[b - 1]:
                raise ValueError(f"{name}: row {b} is not row {a} carried by the tie multiplier {t}")
        first, last = rows[chain[0] - 1], carried(rows[chain[-1] - 1])
        if last not in (first, backwards(first)):
            raise ValueError(f"{name}: the chain of rows {chain} does not close under the tie multiplier {t}")
        power = tuple(pow(x, len(chain), m) for x, m in zip(t, moduli, strict=True))
        if power not in group and tuple(-x % m for x, m in zip(power, moduli, strict=True)) not in group:
            raise ValueError(
                f"{name}: the tie multiplier {t} to the power {len(chain)} is not a multiplier of the rows"
            )
    for k in fixed:
        if carried(rows[k - 1]) not in (rows[k - 1], backwards(rows[k - 1])):
            raise ValueError(f"{name}: row {k} is not fixed by the tie multiplier {t}")
    seen: set[int] = set()
    classes = 0
    for start in range(1, len(elements)):
        if start in seen:
            continue
        classes += 1
        todo = [start]
        while todo:
            i = todo.pop()
            if i not in seen:
                seen.add(i)
                todo += [image[i], scaled(g, i), negative[i]]
    return classes


def bordered8_text(rows: Sequence[str]) -> str:
    """The bordered eight-block array of eight rows of + and - of one length w, as a matrix file.

    Block 1 is the circulant matrix of rows[0], with entry rows[0][(j - i) mod w]
    in row i and column j, and blocks 2, ..., 8 are the back-circulant matrices
    of rows[1], ..., rows[7], the circulant matrix times the back diagonal, with
    entry x[(w - 1 - i - j) mod w]. P is the block matrix of BORDERED8 and S the
    8 x 8 matrix of its signs. With e the column of w ones, the matrix of order
    8(w + 1) has the transpose of S in the corner, a row of w entries s over
    each block for each entry s of S, a column of w entries -s beside each band
    for each entry s of the transpose of S, and P.
    """
    w = len(rows[0])
    if len(rows) != 8 or any(len(row) != w or row.strip("+-") for row in rows):
        raise ValueError("the bordered eight-block array takes eight rows of + and - of one length")
    blocks = [["".join(rows[0][(j - i) % w] for j in range(w)) for i in range(w)]]
    blocks += [["".join(x[(w - 1 - i - j) % w] for j in range(w)) for i in range(w)] for x in rows[1:]]
    negated = [[line.translate(SWAP) for line in block] for block in blocks]
    sign = {True: "+", False: "-"}
    lines = [
        "".join(sign[BORDERED8[j][i] > 0] for j in range(8)) + "".join(sign[s > 0] * w for s in BORDERED8[i])
        for i in range(8)
    ]
    for i, band in enumerate(BORDERED8):
        left = "".join(sign[BORDERED8[j][i] < 0] for j in range(8))
        lines += [left + "".join((blocks if e > 0 else negated)[abs(e) - 1][r] for e in band) for r in range(w)]
    return "".join(line + "\n" for line in lines)


def bordered8_rows(record: JSONObject) -> list[str]:
    """The eight rows of a found matrix from the bordered eight-block array, after an exact check of its record.

    The record gives eight rows of + and - of length w, with 8(w + 1) the
    order. The first is of skew type (x[0] = 1 and x[w - j] = -x[j] for j != 0)
    and the others are symmetric (x[w - j] = x[j]). Every row has sum 1, as the
    record states. Each is fixed by the multiplier g of the record,
    x[g j mod w] = x[j], and g has the stated order modulo w. The periodic
    autocorrelations of the eight rows add to -8 at every nonzero shift.
    Raises ValueError otherwise.
    """
    name, order, rows = label(record), record["order"], record["rows"]
    w = order // 8 - 1
    if 8 * (w + 1) != order or w % 2 == 0 or record["parameters"].get("group_order") != w:
        raise ValueError(f"{name}: the order is not 8(w + 1) with w odd and the group order of the parameters")
    if len(rows) != 8 or any(len(row) != w or row.strip("+-") for row in rows):
        raise ValueError(f"{name}: the record does not hold eight rows of + and - of length {w}")
    if rows[0][0] != "+" or any(rows[0][w - j] == rows[0][j] for j in range(1, w)):
        raise ValueError(f"{name}: the first row is not of skew type")
    if any(x[w - j] != x[j] for x in rows[1:] for j in range(1, w)):
        raise ValueError(f"{name}: a row after the first is not symmetric")
    sums = [x.count("+") - x.count("-") for x in rows]
    if sums != record["row_sums"] or sums != [1] * 8:
        raise ValueError(f"{name}: the row sums are not all 1 as the array needs and the record states")
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    if math.gcd(g, w) != 1 or min(k for k in range(1, w) if pow(g, k, w) == 1) != h:
        raise ValueError(f"{name}: {g} does not have order {h} modulo {w}")
    if any(x[g * j % w] != x[j] for x in rows for j in range(w)):
        raise ValueError(f"{name}: a row is not fixed by the multiplier {g}")
    bits = str.maketrans("+-", "01")
    packed = [int(x.translate(bits), 2) for x in rows]
    for s in range(1, w):
        # A row and its cyclic shift by s differ in d places, and their inner product is w - 2d.
        differ = sum(
            (p ^ int((x[s:] + x[:s]).translate(bits), 2)).bit_count() for p, x in zip(packed, rows, strict=True)
        )
        if 8 * w - 2 * differ != -8:
            raise ValueError(f"{name}: the periodic autocorrelations of the rows do not add to -8")
    return rows


def found_rows(record: JSONObject) -> list[list[int]]:
    """The eight rows of a found matrix from the 8-Williamson array, after an exact check of what its record says.

    The record gives eight rows of + and - of length v with 8v the order. The
    first is of skew type (x[0] = 1 and x[v - j] = -x[j] for j != 0) and the
    others are symmetric (x[v - j] = x[j]). Each is fixed by the multiplier g of
    the record, x[g j mod v] = x[j], and g has the stated order modulo v. The row
    sums are the ones stated, and the periodic autocorrelations of the eight
    rows add to zero at every nonzero shift. Raises ValueError otherwise.
    """
    name, order, v = label(record), record["order"], record["parameters"]["v"]
    lines = record["rows"]
    if 8 * v != order or len(lines) != 8 or any(len(line) != v or line.strip("+-") for line in lines):
        raise ValueError(f"{name}: the record does not hold eight rows of + and - of length {v}")
    rows = [[1 if c == "+" else -1 for c in line] for line in lines]
    if rows[0][0] != 1 or any(rows[0][v - j] != -rows[0][j] for j in range(1, v)):
        raise ValueError(f"{name}: the first row is not of skew type")
    if any(x[v - j] != x[j] for x in rows[1:] for j in range(1, v)):
        raise ValueError(f"{name}: a row after the first is not symmetric")
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    if math.gcd(g, v) != 1 or min(k for k in range(1, v) if pow(g, k, v) == 1) != h:
        raise ValueError(f"{name}: {g} does not have order {h} modulo {v}")
    if any(x[g * j % v] != x[j] for x in rows for j in range(v)):
        raise ValueError(f"{name}: a row is not fixed by the multiplier {g}")
    if [sum(x) for x in rows] != record["row_sums"]:
        raise ValueError(f"{name}: the row sums are not {record['row_sums']}")
    if any(sum(x[j] * x[(j + s) % v] for x in rows for j in range(v)) for s in range(1, v)):
        raise ValueError(f"{name}: the periodic autocorrelations of the rows do not add to zero")
    return rows


def found_text(record: JSONObject) -> str:
    """The matrix file of a stored matrix of origin found, rebuilt from the rows of its record."""
    if record["array"] == "bordered":
        return bordered_text(bordered_rows(record), record["group"])
    if record["array"] == "bordered8":
        return bordered8_text(bordered8_rows(record))
    if record["array"] != "williamson8":
        raise ValueError(f"{label(record)}: unknown array {record['array']}")
    signs = record["block_signs"]
    if len(signs) != 8 or signs[0] != 1 or any(s not in (1, -1) for s in signs):
        raise ValueError(f"{label(record)}: the block signs are not 1 followed by seven of 1 and -1")
    return matrix_text(williamson8_array(found_rows(record), signs))


def matrix_text(h: Matrix) -> str:
    """A matrix with entries 1 and -1 as lines of + and -, one row per line."""
    return "".join("".join("+" if v > 0 else "-" for v in row) + "\n" for row in h)


def check_skew_text(text: str) -> int:
    """Raise ValueError unless the text is a skew-Hadamard matrix H; return its order m.

    The text is m lines of m characters, + for 1 and - for -1. The conditions
    are H H^T = mI and H + H^T = 2I, and the arithmetic is exact. Each row and
    each column is packed into an integer with a bit set where the entry is -1.
    H + H^T = 2I says that the diagonal is 1 and that row i and column i differ
    in every position but the diagonal one, which is one comparison of two
    integers. The inner product of two rows is m minus twice the number of
    positions in which they differ, and every pair of rows is compared.
    """
    lines = text.split("\n")
    if lines.pop() != "" or not lines:
        raise ValueError("not lines of + and -, each ended by a newline")
    m = len(lines)
    if any(len(line) != m or line.strip("+-") for line in lines):
        raise ValueError("not a square matrix with entries 1 and -1")
    bits = str.maketrans("+-", "01")
    rows = [int(line.translate(bits), 2) for line in lines]
    columns = [int("".join(column).translate(bits), 2) for column in zip(*lines, strict=True)]
    full = (1 << m) - 1
    for i in range(m):
        diagonal = 1 << (m - 1 - i)
        if rows[i] & diagonal or rows[i] ^ columns[i] != full ^ diagonal:
            raise ValueError(f"H + H^T differs from 2I in row {i}")
    half, odd = divmod(m, 2)
    for i in range(1, m):
        row = rows[i]
        for j in range(i):
            if odd or (row ^ rows[j]).bit_count() != half:
                raise ValueError(f"rows {j} and {i} of H are not orthogonal")
    return m


def check_skew_hadamard(h: Matrix) -> None:
    """Raise ValueError unless H has entries 1 and -1, H H^T = mI and H + H^T = 2I, for H of order m."""
    m = len(h)
    if m == 0 or any(len(row) != m or any(v not in (1, -1) for v in row) for row in h):
        raise ValueError("not a square matrix with entries 1 and -1")
    check_skew_text(matrix_text(h))


def file_name(order: int) -> str:
    return f"matrices/skew-{order}.txt"


def rows_file_name(order: int) -> str:
    return f"matrices/skew-{order}-rows.txt"


def has_download(order: int) -> bool:
    """Whether the site offers the matrix file of an order: its text is at most DOWNLOAD_LIMIT bytes."""
    return order * (order + 1) <= DOWNLOAD_LIMIT


def rebuilt_in_build(order: int) -> bool:
    """Whether the build rebuilds and checks a found matrix of this order, or its rows only; see BUILD_REBUILD_MAX."""
    return order <= BUILD_REBUILD_MAX


def checked_rows(record: JSONObject) -> Sequence[Sequence[int] | str]:
    """The rows of a found matrix, after the exact check of what its array asks of them."""
    if record["array"] == "bordered":
        return bordered_rows(record)
    if record["array"] == "bordered8":
        return bordered8_rows(record)
    return found_rows(record)


def kept_words(record: JSONObject, count: str) -> str:
    """What the repository holds of a found matrix without a file, and what the build does with it."""
    order = record["order"]
    megabytes = round(order * (order + 1) / 1e6, 1)
    held = (
        f"The repository holds the {count} rows and the sha256 of the matrix file, not the file, which is "
        f"{megabytes} MB of text. The build checks in integers that the rows have the properties stated here"
    )
    if rebuilt_in_build(order):
        return (
            f"{held}, rebuilds the matrix from them by the definitions above, compares the sha256 and checks the two "
            "defining conditions exactly."
        )
    return (
        f"{held}. It does not rebuild the matrix: that the array of rows with these properties is a skew-Hadamard "
        "matrix is the theorem of the sources cited for the array. A test of the repository that is not part of "
        "every build rebuilds the matrix from the rows by the definitions above, compares the sha256 and checks the "
        "two defining conditions exactly."
    )


def chosen_entries(states: OrderStates) -> dict[int, str]:
    """The entry whose array is written for each order the board holds: the smallest id."""
    return {order: min(s[NAME]["entries"]) for order, s in states.items() if s[NAME]["state"] == "board"}


def stored_text(record: JSONObject, check: bool = True) -> str:
    """The text of a stored matrix, after comparing its sha256 with the record.

    A matrix of origin found must equal the array rebuilt from the rows of its
    record, whose stated properties are checked on the way; one without a file
    is that array, and its sha256 must be the one recorded. With check set, the
    text must also pass the exact skew-Hadamard check and have the order of its
    record. Raises ValueError otherwise.
    """
    order, name = record["order"], label(record)
    if "file" in record:
        text = lzma.decompress((ROOT / record["file"]).read_bytes())
        if hashlib.sha256(text).hexdigest() != record["sha256"]:
            raise ValueError(f"{name} does not match its sha256")
        decoded = text.decode("ascii")
        if record["origin"] == "found" and found_text(record) != decoded:
            raise ValueError(f"{name} is not the array of the rows of its record")
    else:
        decoded = found_text(record)
        if hashlib.sha256(decoded.encode("ascii")).hexdigest() != record["sha256"]:
            raise ValueError(f"{name}, rebuilt from the rows of its record, does not match its sha256")
    if check:
        try:
            found = check_skew_text(decoded)
        except ValueError as error:
            raise ValueError(f"{name} is not a skew-Hadamard matrix: {error}") from error
        if found != order:
            raise ValueError(f"{name} does not have order {order}")
    return decoded


def construction(record: JSONObject) -> str:
    """The construction of a stored matrix and its parameters, in a few words."""
    p = record["parameters"]
    words = [PARAMETER_WORDS.get(key, "{}").format(value) for key, value in p.items() if key != "from_construction"]
    text = record["construction"].replace("skew SDS", "skew supplementary difference sets")
    if words:
        text += (" " if "from_order" in p else ", ") + ", ".join(words)
    if "from_construction" in p:
        text += f" ({p['from_construction'].replace('skew SDS', 'skew supplementary difference sets')})"
    return text


# The sources read for a bordered skew array of eight blocks before the one used here was worked out.
BORDERED8_SOURCES_READ = ("wallis1972some", "momihara2019constructions", "kotsireas2008new")
# The odd v below 200 that djokovic2023skew lists with a skew-Hadamard matrix of order 4v undecided.
DJOKOVIC_UNDECIDED = (89, 101, 107, 119, 149, 153, 167, 177, 179, 191, 193)


def listed_statements(order: int, t: JSONObject, bib: Bibliography, prefix: str) -> str:
    """What each source behind the list of open orders says about one order of that list.

    The sources do not all list the order: one names it, one gives a table from which it follows,
    one lists odd v for order 4v, and one names only the smallest order. Each is credited with
    what it says, and a source that says nothing about the order is left out.
    """
    odd = order
    while odd % 2 == 0:
        odd //= 2
    says = {
        "sage2026unknown": "The SageMath list {mark} names order {order} among the orders up to 1200 with no known "
        "skew-Hadamard matrix.",
        "cati2024database": "By the table in {mark} of the least t for which a skew-Hadamard matrix of order "
        "2<sup>t</sup>m is known, m odd, none of order {order} is known.",
        "djokovic2023skew": "{mark} lists v = {odd} among the odd v below 200 for which a skew-Hadamard matrix of "
        "order 4v is undecided."
        if order == 4 * odd and odd in DJOKOVIC_UNDECIDED
        else "",
        "egan2026recent": "The survey {mark} gives {order} as the smallest order with no skew-Hadamard matrix known."
        if order == min(t["listed_open"])
        else "",
    }
    if set(says) != set(t["listed_open_sources"]):
        raise ValueError("skew-hadamard: a source of the open list has no statement of what it lists")
    return " ".join(
        says[key].format(mark=bib.mark(key, prefix), order=order, odd=odd)
        for key in sorted(says, key=bib.number.__getitem__)
        if says[key]
    )


# Where the parts of the method of a found matrix were used before, by bibliography key. Rows constant
# on the orbits of a group of multipliers: djokovic1992construction builds its sets from the cosets of
# such a group, and djokovic2014some calls the method well known. Three rows carried round by a
# multiplier: the spin structure of djokovic2024two, for the array without a border. Two pairs of
# rows carried to each other by a multiplier: the pair-spin structure of kishan2026pairspin, likewise.
# The bordered array with rows constant on classes of powers in a prime field: kline2026skew.
ORBIT_SOURCES = ("djokovic1992construction", "djokovic2014some")
TIE_SOURCES = {3: "djokovic2024two", 2: "kishan2026pairspin"}
BORDERED_BEFORE = "kline2026skew"


def tie_source(record: JSONObject) -> str | None:
    """The key of the earlier work that uses the relation between rows of a tie, if there is one for its shape."""
    if "tie" not in record:
        return None
    lengths = sorted(len(chain) for chain in record["tie"]["chains"])
    return TIE_SOURCES.get(lengths[0]) if lengths in ([3], [2, 2]) else None


def orbit_credit(bib: Bibliography, prefix: str) -> str:
    """One sentence, with its citations, saying that rows constant on the orbits of multipliers are a known method."""
    return (
        " Taking rows that are constant on the orbits of a group of multipliers is a standard way to search for "
        f"such rows {marks(bib, ORBIT_SOURCES, prefix)}."
    )


def method_credit(record: JSONObject, bib: Bibliography, prefix: str) -> str:
    """The sentences of a record that credit earlier uses of its search method, with their citations."""
    return " " + bib.cite(record["credits"], prefix) if "credits" in record else ""


def stored_keys(record: JSONObject, bib: Bibliography) -> set[str]:
    """Every bibliography key the block of a stored matrix cites."""
    keys = {*record["sources"], *bib.keys_in(record.get("note", "")), *bib.keys_in(record.get("credits", ""))}
    if record["origin"] == "found":
        keys.update(ORBIT_SOURCES)
        if record["array"] == "bordered":
            keys.add(BORDERED_BEFORE)
        if key := tie_source(record):
            keys.add(key)
    if record.get("array") == "bordered8":
        keys.update(BORDERED8_SOURCES_READ)
    if "software" in record:
        keys.add(record["software"])
    return keys


def origin_sentence(record: JSONObject, bib: Bibliography, prefix: str) -> str:
    """Where a stored matrix file comes from."""
    if record["origin"] == "sagemath":
        return (
            f"The file was generated with SageMath {bib.mark(record['software'], prefix)}: {esc(record['generated'])}."
        )
    if record["origin"] == "repository":
        return (
            f"The file comes from {bib.mark(record['software'], prefix)}: {esc(record['generated'])}. "
            "The source is a software repository, not peer reviewed."
        )
    if record["origin"] == "found":
        people = ", ".join(
            f'<a href="https://github.com/{esc(person["github"])}">{esc(person["name"])}</a>'
            if person.get("github")
            else esc(person["name"])
            for person in record["contributors"]
        )
        if record["array"] == "bordered":
            what = (
                f"four rows on a group of order {math.prod(record['group'])} placed in a bordered array, not a "
                "Goethals–Seidel quadruple"
            )
            if "file" not in record:
                what += ", and the order is above the board's range"
        elif record["array"] == "bordered8":
            what = (
                f"eight rows of length {record['parameters']['group_order']} placed in a bordered array, not a "
                "Goethals–Seidel quadruple, and the order is above the board's range"
            )
        else:
            what = f"eight rows of length {record['parameters']['v']}, not a Goethals–Seidel quadruple"
            if "file" not in record:
                what += ", and the order is above the board's range"
        return (
            f"The matrix was found on {esc(record['date'])} by {people} with a computer search. It is not a "
            f"board entry: the object is {what}."
        )
    half = record["order"] // 2
    return (
        f"The file is the doubling of the board's matrix of order {half}: with H the array of entry "
        f'<a href="{prefix}entries/{esc(record["entry"])}.html">{esc(record["entry"])}</a>, it is the block matrix '
        "with rows (H, H) and (−H<sup>T</sup>, H<sup>T</sup>)."
    )


def found_details(record: JSONObject, bib: Bibliography, prefix: str) -> str:
    """The array, the rows, the search and the checks of a stored matrix of origin found.

    The rows pass found_rows, bordered_rows or bordered8_rows first, so the page states only what was checked.
    """
    if record["array"] == "bordered":
        return bordered_details(record, bib, prefix)
    if record["array"] == "bordered8":
        return bordered8_details(record, bib, prefix)
    rows = found_rows(record)
    v, last = len(rows[0]), len(rows[0]) - 1
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    group = ", ".join(str(x) for x in sorted({pow(g, k, v) for k in range(h)}))
    array = "\n".join(" ".join(("−" if e < 0 else " ") + BLOCK_NAMES[abs(e) - 1] for e in band) for band in WILLIAMSON8)
    negated = [BLOCK_NAMES[k] for k, s in enumerate(record["block_signs"]) if s < 0]
    if len(negated) == 7:
        signs = ", each multiplied by −1"
    else:
        signs = f", with {listing(negated)} multiplied by −1" if negated else ""
    minus = [k + 1 for k, x in enumerate(rows) if x[0] != 1]
    start = f"; here row {listing(minus)} begins with −1" if len(minus) == 1 else ""
    if len(minus) > 1:
        start = f"; here rows {listing(minus)} begin with −1"
    sums = ", ".join(str(x) for x in record["row_sums"])
    if "file" in record:
        kept = "The build rebuilds the matrix from the eight rows and compares it with the stored file."
    else:
        megabytes = round(record["order"] * (record["order"] + 1) / 1e6, 1)
        kept = (
            "The repository holds the eight rows and the sha256 of the matrix file, not the file, which is "
            f"{megabytes} MB of text. The build rebuilds the matrix from the rows and compares the sha256."
        )
    return (
        f"<p>The array of {bib.mark(record['sources'][0], prefix)} is the block matrix W below, in eight blocks of "
        f"order {v}. The eighth block is written H as in the source; elsewhere on this page H is the whole matrix.</p>"
        f'<div class="rows-text"><pre>{array}</pre></div>'
        f"<p>Let T be the permutation matrix of order {v} with a 1 in position (i, i + 1 mod {v}) for "
        f"i = 0, …, {last}, and let R be the permutation matrix with ones on the back diagonal. A row "
        f"x = (x<sub>0</sub>, …, x<sub>{last}</sub>) gives the circulant matrix x<sub>0</sub>I + x<sub>1</sub>T + … + "
        f"x<sub>{last}</sub>T<sup>{last}</sup>, and the back-circulant matrix that is this one times R. A is the "
        "circulant matrix of the first row below, and B, …, H are the back-circulant matrices of the other seven "
        f"rows, in order{signs}. The first row is of skew type: x<sub>0</sub> = 1 and x<sub>{v}−j</sub> = "
        f"−x<sub>j</sub> for j = 1, …, {last}. The other seven are symmetric: x<sub>{v}−j</sub> = x<sub>j</sub>.</p>"
        f"<p>The source takes every row to begin with 1 and multiplies no block by −1{start}. It gives W for "
        "commuting symbols and states the conditions on the rows by analogy with the Williamson array, without a "
        "theorem for this array. The statement that this matrix is skew-Hadamard therefore rests on the exact check "
        "of the build.</p>"
        f'<div class="rows-text"><pre>{"\n".join(esc(line) for line in record["rows"])}</pre></div>'
        f"<p>Each row is constant on the orbits of {{{group}}}, the subgroup of order {h} generated by {g} in the "
        f"multiplicative group of the integers modulo {v}, which acts on the positions 0, …, {last} by "
        f"multiplication. The row sums are {sums}, and their squares add to {record['order']}. The periodic "
        f"autocorrelations of the eight rows add to zero at every nonzero shift.{orbit_credit(bib, prefix)}</p>"
        f"<p>{esc(record['search'])}{method_credit(record, bib, prefix)}</p>"
        f"<p>{kept} It checks in "
        "integers that the rows have the symmetry types, the multiplier and the row sums stated here and that their "
        f"periodic autocorrelations add to zero. {esc(record['checked'])}</p>"
    )


def prime_powers(n: int) -> list[tuple[int, int]]:
    """The primes that divide n, each with its exponent."""
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            k = 0
            while n % p == 0:
                n, k = n // p, k + 1
            out.append((p, k))
        p += 1
    return [*out, (n, 1)] if n > 1 else out


def bordered_picture() -> str:
    """The bordered array as text: 1 and −1 in the corner, e for a row of ones, ᵀ for a transpose."""

    def word(sign: int, name: str) -> str:
        return ("−" if sign < 0 else "") + name

    def block(name: str) -> str:
        return name.replace("-", "−").replace("T", "ᵀ")

    lines = [
        [word(s, "1") for s in corner] + [word(s, "e") for s in edge]
        for corner, edge in zip(BORDER_CORNER, BORDER_EDGE, strict=True)
    ]
    lines += [
        [word(s, "eᵀ") for s in corner] + [block(name) for name in band]
        for corner, band in zip(BORDER_CORNER, BORDERED_BANDS, strict=True)
    ]
    return "\n".join(" ".join(cell.rjust(3) for cell in line) for line in lines)


def bordered_details(record: JSONObject, bib: Bibliography, prefix: str) -> str:
    """The group, the array, the rows, the search and the checks of a found matrix from the bordered array.

    The rows pass bordered_rows first, so the page states only what was checked.
    """
    rows = bordered_rows(record)
    moduli, order = record["group"], record["order"]
    w = len(rows[0])
    first, second = (bib.mark(key, prefix) for key in record["sources"][:2])
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    orbits = len(multiplier_orbits(moduli, g)) - 1
    if len(moduli) == 1:
        group = (
            f"Here G is the group of the integers modulo {w} under addition, with its elements in the order "
            f"0, 1, …, {w - 1}."
        )
        shape = "They are the circulant and the back-circulant matrix of x."
        factors = [p**k for p, k in prime_powers(w)]
        if len(factors) > 1:
            product = " × ".join(f"Z<sub>{q}</sub>" for q in factors)
            group += f" Since {w} = {' × '.join(map(str, factors))}, G is also the product {product}."
        multiplier = f"the map z ↦ {g[0]}z of G, which has order {h}"
    else:
        coprime = all(math.gcd(a, b) == 1 for a, b in itertools.combinations(moduli, 2))
        if len(moduli) != 2 or not coprime:
            raise ValueError(f"{label(record)}: the page describes one cyclic group or two of coprime orders")
        m1, m2 = moduli
        name = f"Z<sub>{m1}</sub> × Z<sub>{m2}</sub>"
        # The multiplier on the cyclic form: the t modulo w that is g1 modulo m1 and g2 modulo m2.
        t = next(t for t in range(w) if (t % m1, t % m2) == (g[0], g[1]))
        group = (
            f"Here G is the group {name}: its {w} elements are the pairs (x<sub>1</sub>, x<sub>2</sub>) with "
            f"x<sub>1</sub> an integer modulo {m1} and x<sub>2</sub> an integer modulo {m2}, added componentwise, "
            f"in the order (0, 0), (0, 1), …, (0, {m2 - 1}), (1, 0), …, ({m1 - 1}, {m2 - 1}). Since {m1} and {m2} "
            f"have no common factor, G is a cyclic group of order {w}, generated by (1, 1); the rows are kept on the "
            "pairs, as the search produced them."
        )
        shape = (
            f"In this order of the elements they are not circulant matrices: the type 1 matrix consists of "
            f"{m1} × {m1} blocks of order {m2}, each of them circulant, placed in a circulant pattern. In the order "
            f"0, (1, 1), 2(1, 1), …, {w - 1}(1, 1) of the elements they become the circulant and the back-circulant "
            "matrix of x, so this matrix differs from one with circulant and back-circulant blocks by a "
            "reordering of the rows and columns inside the blocks."
        )
        multiplier = (
            f"the map (x<sub>1</sub>, x<sub>2</sub>) ↦ ({g[0]}x<sub>1</sub>, {g[1]}x<sub>2</sub>) of G, which has "
            f"order {h} and takes k(1, 1) to {t}k(1, 1)"
        )
    if "file" in record:
        kept = (
            "The build rebuilds the matrix from the four rows by the definitions above and compares it with the "
            "stored file. It checks in integers that the rows have the properties stated here."
        )
    else:
        kept = kept_words(record, "four")
    before = ""
    if BORDERED_BEFORE not in bib.keys_in(record.get("note", "")):
        before = (
            " The same bordered array, with rows constant on classes of powers in a field of a prime number of "
            f"elements, is used in the release {bib.mark(BORDERED_BEFORE, prefix)} of 2026-11-19 for the orders 4040, "
            "5192, 5960 and 6216."
        )
    pairing = ""
    if "tie" in record:
        t, chains, fixed = record["tie"]["multiplier"], record["tie"]["chains"], record["tie"].get("fixed", [])
        carried = (
            f"z ↦ {t[0]}z"
            if len(t) == 1
            else f"(x<sub>1</sub>, x<sub>2</sub>) ↦ ({t[0]}x<sub>1</sub>, {t[1]}x<sub>2</sub>)"
        )
        nth = ("first", "second", "third", "fourth")
        links = [
            f"the {nth[b - 1]} row is the {nth[a - 1]} carried by σ" for c in chains for a, b in itertools.pairwise(c)
        ]
        links += [f"the {nth[k - 1]} row carried by σ is itself or itself read backwards" for k in fixed]
        free = listing(nth[k - 1] for k in sorted([*fixed, *(c[0] for c in chains)]))
        earlier = ""
        if (key := tie_source(record)) == "djokovic2024two":
            earlier = (
                " Rows related in this way, three of them carried round by a multiplier, were used before in the "
                f"Goethals–Seidel array, which has no border, where {bib.mark(key, prefix)} calls it a spin structure; "
                "here the relation is used in the bordered array."
            )
        elif key:
            earlier = (
                " Rows related in this way, two pairs each carried by a multiplier, are described for the "
                f"Goethals–Seidel array, which has no border, in the notes of the repository {bib.mark(key, prefix)}, "
                "which call it a pair-spin structure; here the relation is used in the bordered array."
            )
        pairing = (
            f"<p>The rows are tied by the map σ: {carried} of G. Say that a row x carried by σ is the row y with "
            f"y(σ(z)) = x(z) for every z in G. Then {listing(links)}. Carrying the last row of such a chain once more "
            "gives back the row the chain began with, or that row read backwards, which is x(−z) in place of x(z). "
            "The periodic autocorrelation of a row carried by σ, taken at σ(s), is that of the row at s, and a row "
            "read backwards has the same autocorrelations as the row; so the sum of the four autocorrelations is the "
            f"same at s and at σ(s). A search with the rows tied in this way has the {free} rows to choose, and one "
            "condition for each set "
            "of nonzero shifts that σ, the multipliers of the rows and z ↦ −z carry to one another: there are "
            f"{tie_classes(record)} such sets.{earlier}</p>"
        )
    return (
        f"<p>The array of {first}, in the form in which {second} states it, takes four rows of 1 and −1 whose "
        f"entries are indexed by the elements of an abelian group G of odd order. {group}</p>"
        f"<p>Write z<sub>1</sub>, …, z<sub>{w}</sub> for the elements of G in this order, and read a row x as the "
        "function on G whose value x(z<sub>j</sub>) is its j-th entry. The type 1 matrix of x has the entry "
        "x(z<sub>j</sub> − z<sub>i</sub>) in row i and column j, and the type 2 matrix of x has the entry "
        f"x(z<sub>j</sub> + z<sub>i</sub>). {shape}</p>"
        "<p>Let A, B and D be the type 1 matrices of the first, second and fourth rows below, and let C be the type "
        f"2 matrix of the third. Let e be the row of {w} ones, and write a superscript T for the transpose. The "
        f"matrix H of order {order} = 4({w} + 1) is the block matrix</p>"
        f'<div class="rows-text"><pre>{bordered_picture()}</pre></div>'
        "<p>The sources state the array for four subsets of G: the matrix of a subset has the entry 1 where "
        "z<sub>j</sub> − z<sub>i</sub> (type 1) or z<sub>j</sub> + z<sub>i</sub> (type 2) lies in the subset and −1 "
        "elsewhere. The subset of a row is the set of elements at which the row is −1, so A, B, C and D are the "
        f"negatives of the matrices of the four subsets. With them the matrix above is the one {second} gives, and "
        f"it is −1 times the matrix printed in {first}, whose diagonal is −1. Both sources ask that each subset "
        f"have {(w - 1) // 2} elements, that the first contain exactly one of z and −z for every nonzero z, and that "
        f"every nonzero element of G be a difference of two elements of one subset in {w - 3} ways, counted over the "
        "four subsets, and state as a theorem that the matrix is then skew-Hadamard. For the rows the conditions "
        "say that the first row is of skew type, x(0) = 1 and x(−z) = −x(z) for every nonzero z, that every row "
        "has sum 1, and that the periodic autocorrelations of the four rows, the sums of x(z) x(z + s) over all z "
        "in G, add to −4 at every nonzero s.</p>"
        f'<div class="rows-text"><pre>{"\n".join(esc(line) for line in record["rows"])}</pre></div>'
        f"<p>Each row is constant on the orbits of {multiplier}. It has {orbits} orbits besides {{0}}."
        f"{orbit_credit(bib, prefix)}{before}</p>"
        f"{pairing}"
        f"<p>{esc(record['search'])}{method_credit(record, bib, prefix)}</p>"
        f"<p>{kept} {esc(record['checked'])}</p>"
    )


def bordered8_details(record: JSONObject, bib: Bibliography, prefix: str) -> str:
    """The array, an argument for it, the rows, the search and the checks of a found matrix from the bordered
    eight-block array.

    The rows pass bordered8_rows first, so the page states only what was checked.
    The array was not found in the sources read, and the page names them.
    """
    rows = bordered8_rows(record)
    order, w = record["order"], len(rows[0])
    last = w - 1
    border, plain = (bib.mark(key, prefix) for key in record["sources"][:2])
    g, h = record["multiplier"]["generator"], record["multiplier"]["order"]
    orbits = len({min(j * pow(g, k, w) % w for k in range(h)) for j in range(1, w)})
    pattern = "\n".join(" ".join(("−" if e < 0 else " ") + f"X{abs(e)}" for e in band) for band in BORDERED8)
    signs = "\n".join(" ".join("−1" if e < 0 else " 1" for e in band) for band in BORDERED8)
    t = "<sup>T</sup>"
    return (
        "<p>We did not find an eight-block bordered skew array in the sources we read "
        f"{marks(bib, BORDERED8_SOURCES_READ, prefix)}, which is not a proof that none is published. The array "
        "used here is an analogue for eight blocks of the "
        f"bordered array of {border} for four, built on the eight-block array of {plain}. "
        f"{esc(record['derivation'])} The statement that this matrix is skew-Hadamard rests on the exact check of "
        "the build.</p>"
        f"<p>Let T be the permutation matrix of order {w} with a 1 in position (i, i + 1 mod {w}) for "
        f"i = 0, …, {last}, and let R be the permutation matrix with ones on the back diagonal. A row "
        f"x = (x<sub>0</sub>, …, x<sub>{last}</sub>) gives the circulant matrix x<sub>0</sub>I + x<sub>1</sub>T + … + "
        f"x<sub>{last}</sub>T<sup>{last}</sup>, and the back-circulant matrix that is this one times R. X1 is the "
        "circulant matrix of the first of the eight rows of this page, and X2, …, X8 are the back-circulant matrices "
        f"of the other seven, in order. The first row is of skew type: x<sub>0</sub> = 1 and x<sub>{w}−j</sub> = "
        f"−x<sub>j</sub> for j = 1, …, {last}. The other seven are symmetric: x<sub>{w}−j</sub> = x<sub>j</sub>. "
        "Every row has sum 1, and the periodic autocorrelations of the eight rows add to −8 at every nonzero "
        "shift.</p>"
        "<p>Let P be the block matrix</p>"
        f'<div class="rows-text"><pre>{pattern}</pre></div>'
        "<p>and let S be the matrix of order 8 of its signs:</p>"
        f'<div class="rows-text"><pre>{signs}</pre></div>'
        f"<p>Let e be the column of {w} ones, and write a superscript T for the transpose. The matrix H of order "
        f"{order} = 8({w} + 1) has four parts. Its first 8 rows are S{t} followed by S ⊗ e{t}, which is S with "
        f"each entry s replaced by a row of {w} entries s. Its other rows are −S{t} ⊗ e, which is S{t} with each "
        f"entry s replaced by a column of {w} entries −s, followed by P.</p>"
        f'<div class="rows-text"><pre>  Sᵀ       S ⊗ eᵀ\n −Sᵀ ⊗ e   P</pre></div>'
        "<p>The argument that follows is ours and has not been reviewed. A back-circulant matrix is symmetric, "
        f"circulant matrices commute, and RX{t} = XR for a circulant matrix X, so that XY{t} = YX{t} for any two of "
        f"X1, …, X8, the rows after the first being symmetric. The product of X1, …, X8 each with its own transpose "
        "is the circulant matrix of the periodic autocorrelations of its row, so these eight products add to "
        f"{order}I − 8J, with J the matrix of ones. In P every block stands once in each row and each column of "
        "blocks, and for two different rows of blocks the columns fall into pairs in which the same two blocks meet "
        f"in exchanged places and with opposite products of signs; hence PP{t} has {order}I − 8J in each diagonal "
        "position and zero elsewhere. S is a skew-Hadamard matrix of order 8, and each of X1, …, X8 has every row "
        f"sum equal to 1. Multiplying H by its transpose part by part then gives 8I + 8 × {w}I = {order}I for the "
        f"first 8 rows, −8I ⊗ e{t} + 8I ⊗ e{t} = 0 between the first 8 rows and the others, and "
        f"8I ⊗ J + I ⊗ ({order}I − 8J) = {order}I for the others, so HH{t} = {order}I. For H + H{t} = 2I: "
        f"S{t} + S = 2I, the two border parts are the negatives of each other's transposes, the diagonal of P holds "
        f"X1 with X1 + X1{t} = 2I, and positions (i, j) and (j, i) of P hold the same symmetric block with opposite "
        "signs.</p>"
        f'<div class="rows-text"><pre>{"\n".join(esc(line) for line in record["rows"])}</pre></div>'
        f"<p>Each row is constant on the orbits of the subgroup of order {h} generated by {g} in the multiplicative "
        f"group of the integers modulo {w}, which acts on the positions 0, …, {last} by multiplication. It has "
        f"{orbits} orbits besides {{0}}.{orbit_credit(bib, prefix)}</p>"
        f"<p>{esc(record['search'])}{method_credit(record, bib, prefix)}</p>"
        f"<p>{kept_words(record, 'eight')} {esc(record['checked'])}</p>"
    )


def matrix_block(
    data: JSONObject,
    order: int,
    info: JSONObject,
    stored: dict[int, JSONObject],
    chosen: dict[int, str],
    bib: Bibliography,
    prefix: str = "../",
) -> tuple[str, set[str]]:
    """The block "Matrix of order 4n" of one skew-Hadamard order, and the keys it cites.

    It states the order's state, the construction with its citations and the
    matrix file. An open order has no file: the block names the sources that
    list it.
    """
    t = data["types"][NAME]
    state = info["state"]
    parts = [f'<p><span class="tag">{esc(STATES[state][0])}</span>']
    if info["record"] and state != "open":
        parts.append(
            f"</p><p>{listed_statements(order, t, bib, prefix)} This one dates from {esc(info['record']['date'])}.</p>"
        )
    else:
        parts.append("</p>")
    download = f'<p><a href="{prefix}{file_name(order)}" download>Download the matrix</a>: one row per line, + for 1 and - for −1. {CHECKED}</p>'  # noqa: E501
    if state not in ("open", "board") and not has_download(order):
        megabytes = round(order * (order + 1) / 1e6, 1)
        if rebuilt_in_build(order):
            how = (
                "the build rebuilds it, compares the sha256 and checks H H<sup>T</sup> = nI and "
                "H + H<sup>T</sup> = 2I exactly, with n the order"
            )
        else:
            how = (
                "a test of the repository that is not part of every build rebuilds it, compares the sha256 and "
                "checks H H<sup>T</sup> = nI and H + H<sup>T</sup> = 2I exactly, with n the order, and the build "
                "checks the rows"
            )
        download = (
            f"<p>The matrix file would be {megabytes} MB of text, so the site does not offer it. "
            f'<a href="{prefix}{rows_file_name(order)}" download>Download the rows</a>: one row per line, + for 1 '
            "and - for −1. The rows and the definitions on this page determine the matrix, and the build script of "
            "the repository rebuilds it from them as lines of + and -, one line for each row of the "
            f"matrix. The sha256 of that text is {esc(stored[order]['sha256'])}; {how}.</p>"
        )
    if state == "open":
        keys = set(t["listed_open_sources"])
        parts.append(
            f"<p>None of the sources read gives or reports a skew-Hadamard matrix of order {order} "
            f"{marks(bib, keys, prefix)}. The order is called open only relative to these sources, and nothing "
            "here is a statement about priority.</p>"
        )
    elif state == "board":
        ident = chosen[order]
        keys = {ARRAY_SOURCE}
        parts.append(
            f"<p>Construction: Goethals–Seidel array {bib.mark(ARRAY_SOURCE, prefix)} of the board entry "
            f'<a href="{prefix}entries/{esc(ident)}.html">{esc(ident)}</a>, with a skew block placed first.</p>'
            f"{download}"
        )
    else:
        record = stored[order]
        keys = stored_keys(record, bib)
        parts.append(
            f"<p>Construction: {esc(construction(record))} {marks(bib, record['sources'], prefix)}. "
            f"{origin_sentence(record, bib, prefix)}</p>"
        )
        if record["origin"] == "found":
            parts.append(found_details(record, bib, prefix))
        if "note" in record:
            parts.append(f"<p>{bib.cite(record['note'], prefix)}</p>")
        parts.append(download)
    if info["record"] and state != "open":
        keys |= set(t["listed_open_sources"])
    return f"<h2>Matrix of order {order}</h2>{''.join(parts)}", keys


def order_page(
    data: JSONObject,
    order: int,
    info: JSONObject,
    stored: dict[int, JSONObject],
    chosen: dict[int, str],
    bib: Bibliography,
) -> str:
    """The page of a skew-Hadamard order that has no board cell page."""
    block, keys = matrix_block(data, order, info, stored, chosen, bib)
    if order > data["range"]["max"]:
        where = (
            f"Order {order} is above the range of the board, which has a row for every order up to "
            f"{data['range']['max']}, so it has no cell there."
        )
    else:
        where = (
            f"The board has no cell for quadruples with a skew block at n = {order // 4}: "
            "a skew circulant block needs odd n."
        )
    return (
        f'<article class="entry"><div class="eyebrow">order {order} · skew-hadamard</div>'
        f"<h1>Skew-Hadamard matrix of order {order}</h1>"
        "<p>A skew-Hadamard matrix of order n is an n × n matrix H with entries 1 and −1 such that "
        "H H<sup>T</sup> = nI and H + H<sup>T</sup> = 2I, where H<sup>T</sup> is the transpose of H and I is the "
        f"identity matrix of order n. {where}</p>"
        f"{block}<h2>References</h2>{bib.listing(keys, '../')}"
        '<p><a href="../#board">Back to the board</a></p></article>'
    )
