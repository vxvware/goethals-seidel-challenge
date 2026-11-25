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
    """The type 1 matrix of a function on the group, as lines: entry (i, j) is row[z_j - z_i]."""
    table, negative = group_table(tuple(moduli))
    return ["".join(map(row.__getitem__, table[negative[i]])) for i in range(len(row))]


def type_two(row: str, moduli: Sequence[int]) -> list[str]:
    """The type 2 matrix of a function on the group, as lines: entry (i, j) is row[z_j + z_i]."""
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
    table, negative = group_table(tuple(moduli))
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
        shifted = (int("".join(map(row.__getitem__, table[s])).translate(bits), 2) for row in rows)
        differ = sum((x ^ y).bit_count() for x, y in zip(packed, shifted, strict=True))
        if 4 * w - 2 * differ != -4:
            raise ValueError(f"{name}: the periodic autocorrelations of the rows do not add to -4")
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


def stored_keys(record: JSONObject, bib: Bibliography) -> set[str]:
    """Every bibliography key the block of a stored matrix cites."""
    keys = {*record["sources"], *bib.keys_in(record.get("note", ""))}
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
        return (
            f"The matrix was found on {esc(record['date'])} by {people} with a computer search. It is not a board "
            f"entry: the object is eight rows of length {record['parameters']['v']}, not a Goethals–Seidel quadruple."
        )
    half = record["order"] // 2
    return (
        f"The file is the doubling of the board's matrix of order {half}: with H the array of entry "
        f'<a href="{prefix}entries/{esc(record["entry"])}.html">{esc(record["entry"])}</a>, it is the block matrix '
        "with rows (H, H) and (−H<sup>T</sup>, H<sup>T</sup>)."
    )


def found_details(record: JSONObject, bib: Bibliography, prefix: str) -> str:
    """The array, the rows, the search and the checks of a stored matrix of origin found.

    The rows pass found_rows first, so the page states only what was checked.
    """
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
        "autocorrelations of the eight rows add to zero at every nonzero shift.</p>"
        f"<p>{esc(record['search'])}</p>"
        "<p>The build rebuilds the matrix from the eight rows and compares it with the stored file. It checks in "
        "integers that the rows have the symmetry types, the multiplier and the row sums stated here and that their "
        f"periodic autocorrelations add to zero. {esc(record['checked'])}</p>"
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
            f"</p><p>The sources {marks(bib, t['listed_open_sources'], prefix)} list order {order} without a matrix; "
            f"this one dates from {esc(info['record']['date'])}.</p>"
        )
    else:
        parts.append("</p>")
    download = f'<p><a href="{prefix}{file_name(order)}" download>Download the matrix</a>: one row per line, + for 1 and - for −1. {CHECKED}</p>'  # noqa: E501
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
            f"<p>Construction: Goethals–Seidel array {bib.mark(ARRAY_SOURCE, prefix)} of a board entry, "
            f'<a href="{prefix}entries/{esc(ident)}.html">{esc(ident)}</a>, the one with the smallest id, with a skew '
            f"block placed first.</p>{download}"
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
    return (
        f'<article class="entry"><div class="eyebrow">order {order} · skew-hadamard</div>'
        f"<h1>Skew-Hadamard matrix of order {order}</h1>"
        "<p>A skew-Hadamard matrix of order n is an n × n matrix H with entries 1 and −1 such that "
        "H H<sup>T</sup> = nI and H + H<sup>T</sup> = 2I, where H<sup>T</sup> is the transpose of H and I is the "
        f"identity matrix of order n. The board has no cell for quadruples with a skew block at n = {order // 4}: "
        "a skew circulant block needs odd n.</p>"
        f"{block}<h2>References</h2>{bib.listing(keys, '../')}"
        '<p><a href="../#board">Back to the board</a></p></article>'
    )
