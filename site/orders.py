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


def state_cell(order, name, info, href, text=None):
    """One table cell for an order and a type, in the grid's cell style."""
    label, classes = STATES[info["state"]]
    spoken = f"{TYPE_NAMES[name]} matrix of order {order}: {STATE_WORDS[info['state']]}"
    if info["listed_open"] and info["state"] == "board":
        spoken += "; listed as open in the older sources"
    return (
        f'<td class="{classes}" data-type="{name}" data-order="{order}" '
        f'data-state="{info["state"]}"><a class="cell" href="{esc(href)}" '
        f'aria-label="{esc(spoken)}" title="{esc(spoken)}">{esc(text or label)}</a></td>'
    )


def grid_cells(n, states):
    """The two existence cells of the front-page grid row of n, showing the order 4n."""
    order = 4 * n
    return "".join(
        state_cell(order, name, states[order][name], f"orders.html#o{order}", text=order)
        for name in TYPES
    )


def order_links(orders):
    return ", ".join(f'<a href="#o{o}">{o}</a>' for o in orders)


def board_links(order, held):
    """Links to the board cells of n = order / 4, for odd n in the board's range."""
    n, odd = order // 4, order % 8 == 4
    if not odd or n > BOARD_MAX_N:
        return ""
    slot = held.get(order, {name: [] for name in TYPES})
    if not slot["hadamard"]:
        return f'<a href="cells/n{n}-general.html">none</a>'
    parts = []
    for name, track, word in (
        ("hadamard", "general", "any"),
        ("skew-hadamard", "skew-sds", "skew"),
    ):
        text = word if slot[name] else f"no {word}"
        parts.append(f'<a href="cells/n{n}-{track}.html">{text}</a>')
    return " · ".join(parts)


def summary_lines(data, states, bib):
    """One paragraph per type: what the sources list as open and what has been said since."""
    out = []
    for name in TYPES:
        t = data["types"][name]
        title = TYPE_NAMES[name]
        listed, current = t["listed_open"], t["open"]
        text = (
            f"<strong>{title}.</strong> The smallest order listed without a matrix in "
            f"{marks(bib, t['listed_open_sources'])} is {listed[0]}; they list {len(listed)} "
            f"such orders up to {data['range']['max']}. "
        )
        board = [o for o in listed if states[o][name]["state"] == "board"]
        later = [x for x in t["since"] if x["status"] == "reported"]
        reported = sum(len(x["orders"]) for x in later if "doubles" not in x)
        doubled = sum(len(x["orders"]) for x in later if "doubles" in x)
        if board:
            text += f"This board holds verified matrices of {len(board)} of them. "
        if reported:
            text += f"Matrices are reported for {reported} in sources that are not peer reviewed. "
        if doubled:
            text += f"A further {doubled} follow by doubling a matrix of this board. "
        if current:
            text += (
                f"That leaves {len(current)} orders with no matrix in any source read; the "
                f'smallest is <a href="#o{current[0]}">{current[0]}</a>.'
            )
        else:
            text += "That leaves no order in this range without a matrix or a report of one."
        out.append(f"<p>{text}</p>")
    return "".join(out)


def status_lists(data, states, bib):
    """The open and reported orders, listed before the table, with their sources."""
    out = []
    for name in TYPES:
        t = data["types"][name]
        title = TYPE_NAMES[name]
        current = t["open"]
        out.append(
            f'<h3>{title}: open</h3><p id="open-{name}" data-orders="'
            f'{" ".join(map(str, current))}">'
            + (
                f"{order_links(current)}. No matrix in "
                f"{marks(bib, t['listed_open_sources'])}, and none reported in the other "
                "sources read."
                if current
                else "None in this range."
            )
            + "</p>"
        )
        if not t["since"]:
            continue
        items = []
        for record in t["since"]:
            head = (
                "On this board"
                if record["status"] == "board"
                else "By doubling"
                if "doubles" in record
                else "Reported"
            )
            items.append(
                f'<li data-status="{record["status"]}">{head}: {order_links(record["orders"])}. '
                f"{bib.cite(record['note'])}</li>"
            )
        out.append(
            f"<h3>{title}: listed open in {marks(bib, t['listed_open_sources'])}, with a "
            f"matrix since</h3><ul>{''.join(items)}</ul>"
        )
    return "".join(out)


def sources_block(data, bib):
    items = []
    for key in sorted(data["sources"], key=bib.number.get):
        s = data["sources"][key]
        text = (
            f"{bib.mark(key)} {esc(bib.short(key))}. As of {esc(s['as_of'])}: "
            f"{esc(s['version'])}. {esc(s['read'])}"
        )
        if "pending" in s:
            text += f" {esc(s['pending'])}"
        items.append(f"<li>{text}</li>")
    return f'<h2 id="sources">Sources and how they were read</h2><ul>{"".join(items)}</ul>'


def cited_keys(data, bib):
    keys = set(data["sources"])
    for t in data["types"].values():
        keys.update(t["known_sources"])
        keys.update(t["listed_open_sources"])
        for record in t["since"]:
            keys.update(record["sources"])
            keys.update(bib.keys_in(record["note"]))
    return keys


def page(data, results, bib):
    """The body of orders.html."""
    states = table(data, results)
    held = board_holdings(results)
    top = data["range"]["max"]
    known = {name: marks(bib, data["types"][name]["known_sources"]) for name in TYPES}
    rows = []
    for order, by_type in states.items():
        cells = "".join(
            state_cell(
                order,
                name,
                by_type[name],
                "#sources" if by_type[name]["state"] == "known" else "#status",
            )
            for name in TYPES
        )
        flagged = " ".join(sorted({by_type[name]["state"] for name in TYPES} - {"known", "board"}))
        rows.append(
            f'<tr id="o{order}" data-flag="{flagged}"><td>{order}</td><td>{order // 4}</td>{cells}'
            f'<td class="held">{board_links(order, held)}</td></tr>'
        )
    same = known["hadamard"] == known["skew-hadamard"]
    known_text = (
        '<span class="published">known: a construction is recorded in '
        + (
            known["hadamard"]
            if same
            else f"{known['hadamard']} (Hadamard), {known['skew-hadamard']} (skew-Hadamard)"
        )
        + "</span>"
    )
    return (
        '<article class="entry orders"><div class="eyebrow">orders</div>'
        f"<h1>Hadamard and skew-Hadamard matrices, orders 4 to {top}</h1>"
        r"<p>A Hadamard matrix of order \(m\) is an \(m \times m\) matrix \(H\) with entries "
        r"\(1\) and \(-1\) such that \(HH^{\mathsf T} = mI_m\), where \(H^{\mathsf T}\) is the "
        r"transpose of \(H\) and \(I_m\) is the \(m \times m\) identity matrix. It is "
        r"skew-Hadamard if in addition \(H + H^{\mathsf T} = 2I_m\). For \(m > 2\) a Hadamard "
        "matrix can exist only when 4 divides \\(m\\). Two conjectures, both open, say that "
        "every multiple of 4 is the order of a Hadamard matrix and of a skew-Hadamard matrix; "
        f"{marks(bib, ['egan2026recent'])} attributes the first to Paley and the second to "
        "Seberry.</p>"
        f"<p>This page records, for each order from 4 to {top}, whether a matrix of each kind "
        "is known by any construction. It is compiled from the sources listed at the foot of "
        f"the page, as read on {esc(data['compiled'])}, and says nothing beyond them: an order "
        "is called open when none of those sources gives or reports a matrix, which is not a "
        "claim that none exists elsewhere.</p>"
        f'<section id="status"><h2>Summary</h2>{summary_lines(data, states, bib)}'
        f"{status_lists(data, states, bib)}</section>"
        '<section><h2>Every order</h2><p class="key">The last column links the cells of this '
        "board for n = order / 4. The board tracks cyclic Goethals–Seidel quadruples for odd n "
        f"up to {BOARD_MAX_N}, that is orders 4 to {4 * BOARD_MAX_N} with n odd, so most orders "
        "have no entry there. A quadruple gives a Hadamard matrix of order 4n, and one with a "
        "skew block gives a skew-Hadamard matrix.</p>"
        '<div class="legend"><span class="known">on board: a verified matrix of this order is '
        f"held here</span>{known_text}"
        '<span class="reported">reported: a matrix is reported in a source that is not peer '
        'reviewed, or follows by doubling a matrix of this board; see the summary</span><span class="unresolved">open: no matrix in the sources '
        "read</span></div>"
        '<p class="key"><label><input type="checkbox" id="only-flagged"> show only open and '
        "reported orders</label></p>"
        '<div class="grid-wrap"><table id="orders"><thead><tr><th scope="col">order</th>'
        '<th scope="col">n</th><th scope="col">hadamard</th><th scope="col">skew-hadamard</th>'
        f'<th scope="col">on this board</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        "</div></section>"
        f"{sources_block(data, bib)}"
        f"<h2>References</h2>{bib.listing(cited_keys(data, bib), '')}"
        '<p><a href="index.html#board">Back to the board</a></p></article>'
        "<script>document.getElementById('only-flagged').addEventListener('change',function(e){"
        "for(const tr of document.querySelectorAll('#orders tbody tr'))"
        "tr.classList.toggle('hidden',e.target.checked&&!tr.dataset.flag);});</script>"
    )
