"""The page of skew-Hadamard matrices by order, and the matrix files it links.

The states come from orders.table(), which joins baselines/hadamard-orders.json
with the verified entries of the board. For every order the board holds, the
Goethals–Seidel array of one quadruple is written as a text file after an exact
check that it is a skew-Hadamard matrix.
"""

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from orders import STATES, esc, in_range, listing, marks

if TYPE_CHECKING:
    from orders import OrderStates
    from references import Bibliography
    from verify.common import JSONObject

type Matrix = list[list[int]]

NAME = "skew-hadamard"
# The reference for the Goethals–Seidel array.
ARRAY_SOURCE = "goethals1970skew"
# The states in the order the filter and the counts list them.
STATE_ORDER = ("open", "board", "reported", "known")
# A row lists every entry of its order up to this many; beyond it, the entry of
# the matrix file and a link to the board cell, which lists them all.
MAX_ENTRY_LINKS = 12


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


def check_skew_hadamard(h: Matrix) -> None:
    """Raise ValueError unless H has entries 1 and -1, H H^T = mI and H + H^T = 2I, for H of order m.

    The arithmetic is exact. Each row is packed into an integer with a bit set
    where the entry is -1, so the inner product of two rows is m minus twice the
    number of positions in which they differ.
    """
    m = len(h)
    if any(len(row) != m or any(v not in (1, -1) for v in row) for row in h):
        raise ValueError("not a square matrix with entries 1 and -1")
    for i in range(m):
        for j in range(i, m):
            if h[i][j] + h[j][i] != (2 if i == j else 0):
                raise ValueError(f"H + H^T differs from 2I at ({i}, {j})")
    bits = [sum(1 << j for j, v in enumerate(row) if v < 0) for row in h]
    for i in range(m):
        for j in range(i + 1, m):
            if 2 * (bits[i] ^ bits[j]).bit_count() != m:
                raise ValueError(f"rows {i} and {j} of H are not orthogonal")


def matrix_text(h: Matrix) -> str:
    """A matrix with entries 1 and -1 as lines of + and -, one row per line."""
    return "".join("".join("+" if v > 0 else "-" for v in row) + "\n" for row in h)


def file_name(order: int) -> str:
    return f"matrices/skew-{order}.txt"


def chosen_entries(states: OrderStates) -> dict[int, str]:
    """The entry whose array is written for each order the board holds: the smallest id."""
    return {order: min(s[NAME]["entries"]) for order, s in states.items() if s[NAME]["state"] == "board"}


def counts(states: OrderStates) -> dict[str, int]:
    found = Counter(s[NAME]["state"] for s in states.values())
    return {state: found[state] for state in STATE_ORDER}


def row_sources(data: JSONObject, info: JSONObject, bib: Bibliography) -> str:
    """The citation marks of one order, with the note of its later record when it has one."""
    t = data["types"][NAME]
    record = info["record"]
    if record:
        # A source the note already cites is not marked a second time.
        cited = marks(bib, set(record["sources"]) - set(bib.keys_in(record["note"])))
        return " ".join(x for x in (cited, bib.cite(record["note"])) if x)
    return marks(bib, t["listed_open_sources"] if info["state"] == "open" else t["known_sources"])


def row_board(order: int, info: JSONObject, chosen: dict[int, str]) -> str:
    """The entry links, board cell link and matrix file link of an order the board holds."""
    if info["state"] != "board":
        return ""
    entries, ident = info["entries"], chosen[order]
    shown = entries if len(entries) <= MAX_ENTRY_LINKS else [ident]
    links = ", ".join(f'<a href="entries/{esc(x)}.html">{esc(x)}</a>' for x in shown)
    count = f"all {len(entries)} entries" if len(entries) > 1 else "board cell"
    cell = f'<a href="cells/n{order // 4}-skew-sds.html">{count}</a>'
    file = f'<a href="{file_name(order)}" download>matrix of {esc(ident)}</a>'
    return f"{links} · {cell} · {file}"


def table_rows(data: JSONObject, states: OrderStates, bib: Bibliography, chosen: dict[int, str]) -> str:
    rows = []
    for order in in_range(data):
        info = states[order][NAME]
        label, classes = STATES[info["state"]]
        rows.append(
            f'<tr data-order="{order}" data-state="{info["state"]}"><td>{order}</td>'
            f'<td class="{classes}"><span class="cell">{esc(label)}</span></td>'
            f"<td>{row_sources(data, info, bib)}</td><td>{row_board(order, info, chosen)}</td></tr>"
        )
    return "".join(rows)


def cited_keys(data: JSONObject, bib: Bibliography) -> set[str]:
    """Every bibliography key the page cites."""
    t = data["types"][NAME]
    keys = {ARRAY_SOURCE, *t["known_sources"], *t["listed_open_sources"]}
    for record in t["since"]:
        keys.update(record["sources"])
        keys.update(bib.keys_in(record["note"]))
    return keys


def page(data: JSONObject, states: OrderStates, bib: Bibliography, chosen: dict[int, str]) -> str:
    """The body of skew.html."""
    r = data["range"]
    found = counts(states)
    total = sum(found.values())
    tally = listing(f"{found[state]} {STATES[state][0]}" for state in STATE_ORDER)
    open_orders = [order for order, s in states.items() if s[NAME]["state"] == "open"]
    open_sentence = (
        f"The open orders are {listing(open_orders)}."
        if len(open_orders) > 1
        else f"The open order is {open_orders[0]}."
        if open_orders
        else "No order in the range is open."
    )
    options = "".join(f'<option value="{state}">{esc(STATES[state][0])}</option>' for state in STATE_ORDER)
    references = bib.listing(cited_keys(data, bib), "")
    return (
        '<article class="entry skew-orders"><div class="eyebrow">orders '
        f"{r['min']} to {r['max']}</div><h1>Skew-Hadamard matrices by order</h1>"
        "<p>A skew-Hadamard matrix of order n is an n × n matrix H with entries 1 and −1 such that "
        "H H<sup>T</sup> = nI and H + H<sup>T</sup> = 2I, where H<sup>T</sup> is the transpose of H "
        "and I is the identity matrix of order n. "
        f"This table has one row for each multiple of 4 from {r['min']} to {r['max']}.</p>"
        f'<p id="open-list">{open_sentence}</p>'
        f'<p id="state-counts">The data were compiled on {esc(data["compiled"])}. Of the {total} orders, '
        f"{tally}.</p>"
        "<p>On board means that this board holds an exactly verified matrix of the order. Known means "
        "that the cited sources give a matrix and the board holds none. Reported means that a matrix "
        "is reported in a source that is not peer reviewed, or follows from a matrix of this board by a "
        "cited construction, and the board holds none. An order is called open only relative to the "
        "sources cited in its row: none of the sources read gives or reports a matrix. Nothing here is "
        "a statement about priority.</p>"
        "<p>For each order on the board, the matrix file is the Goethals–Seidel array "
        f"{bib.mark(ARRAY_SOURCE)} of the entry with the smallest id, with a skew block placed "
        "first: one row per line, + for 1 and - for −1. The build checks H H<sup>T</sup> = nI and "
        "H + H<sup>T</sup> = 2I in exact integer arithmetic before it writes a file.</p>"
        '<div class="controls"><label>Show <select id="state"><option value="">all</option>'
        f"{options}</select></label></div>"
        '<div class="scroll"><table id="skew-orders"><thead><tr><th scope="col">order</th>'
        '<th scope="col">state</th><th scope="col">sources</th>'
        '<th scope="col">on this board</th></tr></thead>'
        f"<tbody>{table_rows(data, states, bib, chosen)}</tbody></table></div>"
        f"<h2>References</h2>{references}"
        '<p><a href="./#board">Back to the board</a></p></article>'
        "<noscript><p>JavaScript enables the filter. The table is complete without it.</p></noscript>"
        '<script src="skew.js" defer></script>'
    )
