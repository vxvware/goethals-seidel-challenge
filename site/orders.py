"""Existence of Hadamard and skew-Hadamard matrices by order, from cited sources.

baselines/hadamard-orders.json records, for each matrix type, the orders that
the cited sources list without a matrix and the later statements about them.
This module checks that file, joins it with the verified entries of the board,
and renders the orders page and the two existence columns of the front-page grid.
"""

import html
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "baselines/hadamard-orders.json"
TYPES = ("hadamard", "skew-hadamard")
TYPE_NAMES = {"hadamard": "Hadamard", "skew-hadamard": "skew-Hadamard"}
# The state of an order for one type, as displayed, with the grid's cell class.
STATES = {
    "board": ("on board", "exists verified-on-board"),
    "known": ("known", "exists published-not-imported"),
    "reported": ("reported", "reported"),
    "open": ("open", "unresolved"),
}
STATE_WORDS = {
    "board": "a verified matrix is on this board",
    "known": "known according to the cited sources, not held on this board",
    "reported": "reported, not yet in a reviewed source",
    "open": "open according to the cited sources",
}
# The board cell a grid existence cell links to, by type.
CELL_TRACKS = {"hadamard": "general", "skew-hadamard": "skew-sds"}
# The board covers odd n up to this value; summary() uses the same bound.
BOARD_MAX_N = 131


def esc(value):
    return html.escape(str(value), quote=True)


def load_data(path=DATA):
    return json.loads(Path(path).read_text())


def in_range(data):
    r = data["range"]
    return range(r["min"], r["max"] + 1, r["step"])


def validate(data, bib):
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


def board_holdings(results):
    """The verified quadruples of the board by order 4n and matrix type.

    Every quadruple gives a Hadamard matrix of order 4n through the
    Goethals-Seidel array, and one with a skew block gives a skew-Hadamard matrix.
    """
    held = {}
    for ident, x in results.items():
        if x["type"] != "quadruple":
            continue
        slot = held.setdefault(4 * x["n"], {name: [] for name in TYPES})
        slot["hadamard"].append(ident)
        if "skew-sds" in x["tracks"]:
            slot["skew-hadamard"].append(ident)
    return held


def table(data, results):
    """The state of every order in range: {order: {type: {...}}}.

    A state is board when the board holds a verified matrix of the order,
    otherwise reported or open as the data file says, otherwise known.
    """
    held = board_holdings(results)
    out = {}
    for order in in_range(data):
        out[order] = {}
        for name in TYPES:
            t = data["types"][name]
            record = next((x for x in t["since"] if order in x["orders"]), None)
            entries = sorted(held.get(order, {}).get(name, []))
            if record and record["status"] == "board":
                if not set(record["entry_ids"]) <= set(entries):
                    raise ValueError(
                        f"hadamard-orders: the board does not hold {record['entry_ids']} "
                        f"as {name} matrices of order {order}"
                    )
            if entries:
                state = "board"
            elif record:
                state = "reported"
            elif order in t["open"]:
                state = "open"
            else:
                state = "known"
            out[order][name] = {
                "state": state,
                "listed_open": order in t["listed_open"],
                "record": record,
                "entries": entries,
            }
    return out


def marks(bib, keys, prefix=""):
    return " ".join(bib.mark(key, prefix) for key in sorted(set(keys), key=bib.number.get))


def state_cell(order, name, info, href, text=None, new=False):
    """One table cell for an order and a type, in the grid's cell style.

    With new set, the cell carries the grid's new-result marker.
    """
    label, classes = STATES[info["state"]]
    spoken = f"{TYPE_NAMES[name]} matrix of order {order}: {STATE_WORDS[info['state']]}"
    if info["listed_open"] and info["state"] == "board":
        spoken += "; listed as open in the older sources"
    marker = '<span class="participant" title="New result"></span>' if new else ""
    if new:
        spoken += "; new result of this board"
    return (
        f'<td class="{classes}" data-type="{name}" data-order="{order}" '
        f'data-state="{info["state"]}"><a class="cell" href="{esc(href)}" '
        f'aria-label="{esc(spoken)}" title="{esc(spoken)}">{esc(text or label)}{marker}</a></td>'
    )


def grid_cells(n, states, new_skew=()):
    """The two existence cells of the front-page grid row of n, showing the order 4n.

    Each links to the board cell of n for its type: any quadruple for a
    Hadamard matrix, a quadruple with a skew block for a skew-Hadamard matrix.

    new_skew holds the n whose skew-Hadamard order is a featured existence
    result of this board; that cell gets the new-result marker.
    """
    order = 4 * n
    return "".join(
        state_cell(
            order,
            name,
            states[order][name],
            f"cells/n{n}-{CELL_TRACKS[name]}.html",
            text=order,
            new=name == "skew-hadamard" and n in new_skew,
        )
        for name in TYPES
    )


def page(data, results, bib):
    """The body of orders.html: the open, reported and newly settled orders of each type."""
    states = table(data, results)
    top = data["range"]["max"]
    sections = []
    for name in TYPES:
        t = data["types"][name]
        current = t["open"]
        items = [
            f'<li id="open-{name}" data-orders="{" ".join(map(str, current))}">Open: '
            + (", ".join(map(str, current)) if current else "none")
            + ".</li>"
        ]
        board = []
        for record in t["since"]:
            listed = ", ".join(map(str, record["orders"]))
            cite = f" {marks(bib, record['sources'])}" if record["sources"] else ""
            if record["status"] == "board":
                links = ", ".join(
                    f'<a href="cells/n{o // 4}-{CELL_TRACKS[name]}.html">{o}</a>'
                    for o in record["orders"]
                    if states[o][name]["state"] == "board"
                )
                found = (
                    cite
                    if record["sources"]
                    else " (found here; a literature search found no earlier one)"
                )
                board.append(f"{links}{found}")
            elif "doubles" in record:
                twice = " and ".join(map(str, record["doubles"]))
                items.append(f"<li>By doubling {twice}: {listed}{cite}.</li>")
            else:
                items.append(f"<li>Reported, not reviewed: {listed}{cite}.</li>")
        if board:
            items.insert(1, f"<li>On this board: {', '.join(board)}.</li>")
        title = TYPE_NAMES[name][0].upper() + TYPE_NAMES[name][1:]
        sections.append(f"<h2>{title}</h2><ul>{''.join(items)}</ul>")
    sources = marks(bib, data["sources"])
    return (
        '<article class="entry orders"><div class="eyebrow">open orders</div>'
        f"<h1>Open orders up to {top}</h1>"
        "<p>Hadamard and skew-Hadamard matrices are conjectured to exist in every order "
        f"divisible by 4. An order is open here when none of the sources read gives a matrix "
        f"of it. Sources, as read on {esc(data['compiled'])}: {sources}.</p>"
        + "".join(sections)
        + '<p><a href="index.html#board">Back to the board</a></p></article>'
    )
