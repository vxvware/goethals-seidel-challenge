"""Skew-Hadamard matrix files and the block that describes the matrix of one order.

Every order that is not open has a matrix file in the build output. For an order
the board holds, the file is the Goethals–Seidel array of one quadruple. For
every other order it is a file stored in matrices/skew and listed in
baselines/skew-matrices.json. Each is written only after an exact check that it
is a skew-Hadamard matrix.
"""

import hashlib
import lzma
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from orders import BOARD_MAX_N, STATES, esc, marks

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

    With check set, the text must also pass the exact skew-Hadamard check and
    have the order of its record. Raises ValueError otherwise.
    """
    order = record["order"]
    text = lzma.decompress((ROOT / record["file"]).read_bytes())
    if hashlib.sha256(text).hexdigest() != record["sha256"]:
        raise ValueError(f"{record['file']} does not match its sha256")
    decoded = text.decode("ascii")
    if check:
        try:
            found = check_skew_text(decoded)
        except ValueError as error:
            raise ValueError(f"{record['file']} is not a skew-Hadamard matrix: {error}") from error
        if found != order:
            raise ValueError(f"{record['file']} does not have order {order}")
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
    half = record["order"] // 2
    return (
        f"The file is the doubling of the board's matrix of order {half}: with H the array of entry "
        f'<a href="{prefix}entries/{esc(record["entry"])}.html">{esc(record["entry"])}</a>, it is the block matrix '
        "with rows (H, H) and (−H<sup>T</sup>, H<sup>T</sup>)."
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
        f"identity matrix of order n. The board has no cell for n = {order // 4}: its cells cover odd n up to "
        f"{BOARD_MAX_N}.</p>"
        f"{block}<h2>References</h2>{bib.listing(keys, '../')}"
        '<p><a href="../#board">Back to the board</a></p></article>'
    )
