"""Existence of Hadamard and skew-Hadamard matrices by order, from cited sources.

baselines/hadamard-orders.json records, for each matrix type, the orders that
the cited sources list without a matrix and the later statements about them.
This module checks that file, joins it with the verified entries of the board
and with the stored matrices of baselines/skew-matrices.json, and renders the
two order columns of the front-page grid and the sentence that describes it.
"""

import html
import json
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotations are evaluated lazily, so this module still imports only the standard library.
    from references import Bibliography
    from verify.common import JSON, JSONObject, PathLike

type OrderStates = dict[int, dict[str, JSONObject]]
"""The state of each order for each matrix type, as table() returns it."""

DATA = Path(__file__).resolve().parents[1] / "baselines/hadamard-orders.json"
MATRICES = Path(__file__).resolve().parents[1] / "baselines/skew-matrices.json"
TYPES = ("hadamard", "skew-hadamard")
TYPE_NAMES = {"hadamard": "Hadamard", "skew-hadamard": "skew-Hadamard"}
# The state of an order for one type, as displayed, with the grid's cell class.
STATES = {
    "board": ("on board", "exists verified-on-board"),
    "known": ("known", "exists"),
    "verified": ("verified", "exists"),
    "reported": ("reported", "reported"),
    "open": ("open", "unresolved"),
}
STATE_WORDS = {
    "board": "a verified matrix is on this board",
    "known": "known according to the cited sources, not held on this board",
    "verified": "the site holds a matrix file that the build checks exactly",
    "reported": "reported, not yet in a reviewed source",
    "open": "open according to the cited sources",
}
# The board cell a grid existence cell links to, by type.
CELL_TRACKS = {"hadamard": "general", "skew-hadamard": "skew-sds"}
# The board covers odd n up to this value; summary() uses the same bound.
BOARD_MAX_N = 131
ORIGINS = ("sagemath", "repository", "board")


def covered(n: int) -> bool:
    """Whether the board has cells for n: the odd n up to BOARD_MAX_N."""
    return n % 2 == 1 and n <= BOARD_MAX_N


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_data(path: PathLike = DATA) -> JSON:
    return json.loads(Path(path).read_text())


def in_range(data: JSONObject) -> range:
    r = data["range"]
    return range(r["min"], r["max"] + 1, r["step"])


def validate(data: JSONObject, bib: Bibliography) -> JSONObject:
    """Raise ValueError unless the data file is consistent and every source has a reference."""
    orders = in_range(data)
    if (data["range"]["min"], data["range"]["step"]) != (4, 4) or data["range"]["max"] % 4:
        raise ValueError("hadamard-orders: the range must be the multiples of 4 from 4")
    cited = set()
    if set(data["types"]) != set(TYPES):
        raise ValueError("hadamard-orders: the types must be hadamard and skew-hadamard")
    for name, t in data["types"].items():
        listed, current = t["listed_open"], t["open"]
        for label, values in (("listed_open", listed), ("open", current)):
            if values != sorted(set(values)) or any(o not in orders for o in values):
                raise ValueError(f"hadamard-orders: {name} {label} is not a sorted list in range")
        settled = []
        for record in t["since"]:
            if record["status"] not in ("board", "reported"):
                raise ValueError(f"hadamard-orders: unknown status {record['status']}")
            if record["status"] == "board" and not record.get("entry_ids"):
                raise ValueError("hadamard-orders: a board record names its entries")
            if record["status"] == "reported" and not record["sources"]:
                raise ValueError("hadamard-orders: a reported order names a source")
            settled.extend(record["orders"])
            cited.update(record["sources"])
            cited.update(bib.keys_in(record["note"]))
        if len(settled) != len(set(settled)) or not set(settled) <= set(listed):
            raise ValueError(f"hadamard-orders: {name} since must partition part of listed_open")
        if sorted(set(listed) - set(settled)) != current:
            raise ValueError(f"hadamard-orders: {name} open is not listed_open without since")
        if not t["known_sources"] or not t["listed_open_sources"]:
            raise ValueError(f"hadamard-orders: {name} lacks sources")
        cited.update(t["known_sources"])
        cited.update(t["listed_open_sources"])
    missing = sorted(key for key in cited | set(data["sources"]) if key not in bib.entries)
    if missing:
        raise ValueError(f"hadamard-orders: sources without a reference: {missing}")
    undated = sorted(cited - set(data["sources"]))
    if undated:
        raise ValueError(f"hadamard-orders: cited sources without a date: {undated}")
    return data


def load_matrices(path: PathLike = MATRICES) -> JSON:
    return json.loads(Path(path).read_text())


def validate_matrices(doc: JSONObject, data: JSONObject, bib: Bibliography) -> dict[int, JSONObject]:
    """The stored skew-Hadamard matrices by order. Raises ValueError unless the file is consistent.

    The orders are distinct, in range and not open, and every source, software
    key and note marker has a reference. The files themselves are checked when
    the build reads them.
    """
    stored: dict[int, JSONObject] = {}
    is_open = set(data["types"]["skew-hadamard"]["open"])
    for record in doc["matrices"]:
        order = record["order"]
        if order in stored or order not in in_range(data) or order in is_open:
            raise ValueError(f"skew-matrices: order {order} is repeated, out of range or open")
        if record["file"] != f"matrices/skew/skew-{order}.txt.xz":
            raise ValueError(f"skew-matrices: unexpected file name for order {order}")
        if record["origin"] not in ORIGINS or not record["sources"] or not record["generated"]:
            raise ValueError(f"skew-matrices: order {order} lacks an origin, a source or how it was generated")
        if (record["origin"] == "board") != ("entry" in record) or (record["origin"] != "board") != (
            "software" in record
        ):
            raise ValueError(f"skew-matrices: order {order} names an entry or a software key that its origin lacks")
        keys = [*record["sources"], *bib.keys_in(record.get("note", "")), record.get("software", record["sources"][0])]
        missing = sorted(key for key in keys if key not in bib.entries)
        if missing:
            raise ValueError(f"skew-matrices: sources without a reference: {missing}")
        stored[order] = record
    return stored


def board_holdings(results: dict[str, JSONObject]) -> dict[int, dict[str, list[str]]]:
    """The verified quadruples of the board by order 4n and matrix type.

    Every quadruple gives a Hadamard matrix of order 4n through the
    Goethals-Seidel array, and one with a skew block gives a skew-Hadamard matrix.
    """
    held: dict[int, dict[str, list[str]]] = {}
    for ident, x in results.items():
        if x["type"] != "quadruple":
            continue
        slot = held.setdefault(4 * x["n"], {name: [] for name in TYPES})
        slot["hadamard"].append(ident)
        if "skew-sds" in x["tracks"]:
            slot["skew-hadamard"].append(ident)
    return held


def table(data: JSONObject, results: dict[str, JSONObject], stored: Collection[int] | None = None) -> OrderStates:
    """The state of every order in range: {order: {type: {...}}}.

    For Hadamard matrices a state is board when the board holds a verified
    matrix of the order, otherwise reported or open as the data file says,
    otherwise known. For skew-Hadamard matrices it is board, open, or verified:
    every order that is neither open nor held by the board has a stored matrix
    in baselines/skew-matrices.json, which is what stored lists, and the build
    checks that matrix exactly. Raises ValueError if such an order has none.
    """
    if stored is None:
        stored = [x["order"] for x in load_matrices()["matrices"]]
    held = board_holdings(results)
    out: OrderStates = {}
    for order in in_range(data):
        out[order] = {}
        for name in TYPES:
            t = data["types"][name]
            record = next((x for x in t["since"] if order in x["orders"]), None)
            entries = sorted(held.get(order, {}).get(name, []))
            if record and record["status"] == "board" and not set(record["entry_ids"]) <= set(entries):
                raise ValueError(
                    f"hadamard-orders: the board does not hold {record['entry_ids']} "
                    f"as {name} matrices of order {order}"
                )
            if entries:
                state = "board"
            elif order in t["open"]:
                state = "open"
            elif name == "skew-hadamard":
                if order not in stored:
                    raise ValueError(f"skew-matrices: no stored matrix of order {order}, which is not open")
                state = "verified"
            elif record:
                state = "reported"
            else:
                state = "known"
            out[order][name] = {
                "state": state,
                "listed_open": order in t["listed_open"],
                "record": record,
                "entries": entries,
            }
    return out


def marks(bib: Bibliography, keys: Iterable[str], prefix: str = "") -> str:
    return " ".join(bib.mark(key, prefix) for key in sorted(set(keys), key=bib.number.__getitem__))


def state_cell(
    order: int,
    name: str,
    info: JSONObject,
    href: str | None,
    text: str | int | None = None,
    new: bool = False,
    board: JSONObject | None = None,
) -> str:
    """One table cell for an order and a type, in the grid's cell style.

    Without href the cell is not a link. With new set, the cell carries the grid's new-result marker. board, when
    given, describes the board cell the grid cell links to: its class count
    (named in the tooltip when the board holds a matrix of this type; the cell
    shows only the order), whether it contradicts
    a published result, the words appended to the tooltip, and the attributes
    the grid filters read.
    """
    label, classes = STATES[info["state"]]
    spoken = f"{TYPE_NAMES[name]} matrix of order {order}: {STATE_WORDS[info['state']]}"
    record = info.get("record")
    filled = info["listed_open"] and info["state"] in ("board", "verified")
    if filled:
        spoken += "; the cited sources list this order without a matrix" + (
            f", and this one dates from {record['date']}" if record else ""
        )
    board = board or {}
    data = dict(board.get("data", {}))
    # The origin filter asks who settled the order, not who supplied the board cell's rows:
    # an order the sources listed as open and the board now holds takes the board cell's
    # origins, an order known from the sources is a literature result, and an open order
    # has no origin.
    if info["state"] == "board" and info["listed_open"]:
        pass
    elif info["state"] in ("board", "known") or (info["state"] == "verified" and not info["listed_open"]):
        data["origins"] = "published-import"
    else:
        data["origins"] = ""
    attrs = "".join(f' data-{key}="{esc(value)}"' for key, value in data.items())
    count = board.get("count", "") if info["state"] == "board" else ""
    if board.get("contradicted") and info["state"] == "board":
        classes += " contradicted"
    if board.get("words"):
        spoken += f"; {board['words']}"
    new = new or board.get("new", False)
    marker = '<span class="participant" title="New result"></span>' if new else ""
    if new:
        spoken += "; new result of this board"
    if count and count not in spoken:
        spoken += f"; {count}"
    tag, link = ("a", f' href="{esc(href)}"') if href else ("span", "")
    return (
        f'<td class="{classes}" data-type="{name}" data-order="{order}" '
        f'data-state="{info["state"]}"{attrs}><{tag} class="cell"{link} '
        f'aria-label="{esc(spoken)}" title="{esc(spoken)}"><span>{esc(text or label)}'
        f"</span>{marker}</{tag}></td>"
    )


def order_href(n: int, name: str) -> str | None:
    """The page an order cell of row n links to.

    Where the board covers n it is the board cell of the type's track. Elsewhere
    a skew-Hadamard order has a page of its own and a Hadamard order has none.
    """
    if covered(n):
        return f"cells/n{n}-{CELL_TRACKS[name]}.html"
    return f"orders/skew-{4 * n}.html" if name == "skew-hadamard" else None


def grid_cells(
    n: int, states: OrderStates, new_skew: Collection[int] = (), board: dict[str, JSONObject] | None = None
) -> str:
    """The two order cells of the front-page grid row of n, showing the order 4n.

    Where the board covers n, each links to the board cell of n for its type:
    any quadruple for a Hadamard matrix, a quadruple with a skew block for a
    skew-Hadamard matrix. Its colour is the state of the order, so a
    skew-Hadamard order with a matrix from another construction is not shown as
    open while the board's cell is unresolved. Where the board does not cover n,
    the cells carry the attributes the grid filters read: the track of the
    column, and whether the order is settled.

    new_skew holds the n whose skew-Hadamard order is a featured existence
    result of this board; that cell gets the new-result marker. board maps a
    type to the description of its board cell that state_cell takes.
    """
    order = 4 * n
    board = board or {}
    if not covered(n):
        status = {"open": "unresolved", "reported": ""}
        board = {
            name: {"data": {"track": CELL_TRACKS[name], "status": status.get(states[order][name]["state"], "exists")}}
            for name in TYPES
        }
    return "".join(
        state_cell(
            order,
            name,
            states[order][name],
            order_href(n, name),
            text=order,
            new=name == "skew-hadamard" and n in new_skew,
            board=board.get(name),
        )
        for name in TYPES
    )


def listing(values: Iterable[object]) -> str:
    """Orders joined as prose: "1, 2 and 3"."""
    words = [str(v) for v in values]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def intro(data: JSONObject) -> str:
    """The two-sentence description of the front-page grid, with the smallest open skew orders."""
    op = data["types"]["skew-hadamard"]["open"]
    first = f"{op[0]} and {op[1]}" if len(op) > 1 else (str(op[0]) if op else "")
    tail = f"; the smallest open skew-Hadamard orders are {first}" if op else ""
    return (
        "Each row is an order 4n. Green means a matrix of that order is known and amber "
        f"that it is open{tail}. The family columns count cyclic Goethals–Seidel quadruples "
        f"of each type, for odd n up to {BOARD_MAX_N}; the rows for even n and for larger n have the two order "
        f"columns only, and choosing {BOARD_MAX_N} under Through n hides them."
    )
