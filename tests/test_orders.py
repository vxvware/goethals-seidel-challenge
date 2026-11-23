"""The table of Hadamard and skew-Hadamard orders: its data file, grid columns and front-page box."""

import copy
import re
import sys

import pytest
from verify.board import load, summary
from verify.common import ROOT, TRACKS, read_json

sys.path.insert(0, str(ROOT / "site"))
import orders  # noqa: E402
from references import Bibliography  # noqa: E402

from test_site import gs_array, site_build  # noqa: E402


@pytest.fixture(scope="module")
def board():
    return load(reproduce=False)


@pytest.fixture(scope="module")
def data():
    return orders.load_data()


@pytest.fixture(scope="module")
def states(board, data):
    return orders.table(data, board[1])


def cells(markup):
    """(class, type, order, state, text) of every order cell in a piece of HTML."""
    return re.findall(
        r'<td class="([^"]*)" data-type="([a-z-]+)" data-order="(\d+)" data-state="([a-z]+)"'
        r"[^>]*><a [^>]*><span>([^<]*)</span>",
        markup,
    )


def test_data_file_is_consistent(data):
    bib = Bibliography()
    orders.validate(data, bib)
    in_range = set(orders.in_range(data))
    assert min(in_range) == 4 and max(in_range) >= 1000
    assert all(o % 4 == 0 for o in in_range)
    for t in data["types"].values():
        since = [o for record in t["since"] for o in record["orders"]]
        # No order is both open and settled, and every listed order is one or the other.
        assert not set(t["open"]) & set(since)
        assert sorted(t["open"] + since) == t["listed_open"]
        assert set(t["listed_open"]) <= in_range
        for record in t["since"]:
            assert set(record["sources"]) <= set(bib.entries)
            assert re.fullmatch(r"\d{4}-\d\d-\d\d", record["date"])
    for key, source in data["sources"].items():
        assert key in bib.entries
        assert re.fullmatch(r"\d{4}-\d\d-\d\d", source["as_of"])
    # A skew-Hadamard matrix is a Hadamard matrix, so an order cannot be open
    # for Hadamard matrices alone.
    assert set(data["types"]["hadamard"]["open"]) <= set(data["types"]["skew-hadamard"]["open"])


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["types"]["hadamard"]["open"].append(668),
        lambda d: d["types"]["skew-hadamard"]["open"].remove(404),
        lambda d: d["types"]["skew-hadamard"]["listed_open"].append(1202),
        lambda d: d["types"]["skew-hadamard"]["known_sources"].append("nobody1900nothing"),
        lambda d: d["types"]["skew-hadamard"]["since"][2].update(sources=[]),
        lambda d: d["sources"].pop("kishan2026skew"),
    ],
)
def test_inconsistent_data_is_rejected(data, change):
    broken = copy.deepcopy(data)
    change(broken)
    with pytest.raises(ValueError):
        orders.validate(broken, Bibliography())


def test_board_records_match_the_board_and_the_gap_baseline(board, data, states):
    gaps = read_json(ROOT / "baselines/skew-gaps.json")
    filled = {x["order"]: sorted(x["entry_ids"]) for x in gaps["filled_on_board"]}
    recorded = {
        order: sorted(record["entry_ids"])
        for record in data["types"]["skew-hadamard"]["since"]
        if record["status"] == "board"
        for order in record["orders"]
    }
    assert recorded == filled
    assert set(gaps["orders"]) <= set(data["types"]["skew-hadamard"]["listed_open"])
    assert states[356]["skew-hadamard"]["state"] == "board"
    assert states[356]["skew-hadamard"]["listed_open"]
    # The board may hold further quadruples of this order before the baselines list them.
    assert {"n89-b2a223b0fb7500feb95f", "n89-eec2e43c042090883c04"} <= set(
        states[356]["skew-hadamard"]["entries"]
    )
    assert states[476]["skew-hadamard"]["state"] == "board"
    missing = copy.deepcopy(data)
    missing["types"]["skew-hadamard"]["since"][0]["entry_ids"] = ["n89-0000"]
    with pytest.raises(ValueError, match="does not hold"):
        orders.table(missing, board[1])


def test_every_board_entry_appears_as_held(board, states):
    seen = {name: set() for name in orders.TYPES}
    for x in board[1].values():
        if x["type"] != "quadruple":
            continue
        order = 4 * x["n"]
        assert states[order]["hadamard"]["state"] == "board", x["id"]
        seen["hadamard"].add(order)
        if "skew-sds" in x["tracks"]:
            assert states[order]["skew-hadamard"]["state"] == "board", x["id"]
            seen["skew-hadamard"].add(order)
    for name in orders.TYPES:
        held = {o for o, s in states.items() if s[name]["state"] == "board"}
        assert held == seen[name]
        # The board covers odd n only.
        assert all(o % 8 == 4 and o <= 4 * orders.BOARD_MAX_N for o in held)


def test_states_follow_the_data_file(data, states):
    for name in orders.TYPES:
        t = data["types"][name]
        is_open = [o for o, s in states.items() if s[name]["state"] == "open"]
        assert is_open == t["open"]
        reported = {o for o, s in states.items() if s[name]["state"] == "reported"}
        assert reported == {
            o for record in t["since"] if record["status"] == "reported" for o in record["orders"]
        }
    # Orders the sources give as known and the board does not hold. Use orders beyond the
    # board's range, n > 131, so that a new board entry cannot turn one into "board".
    assert states[532]["skew-hadamard"]["state"] == "known"
    assert states[540]["skew-hadamard"]["state"] == "known"
    assert states[532]["hadamard"]["state"] == "known"
    assert states[404]["skew-hadamard"]["state"] == "open"
    assert states[668]["hadamard"]["state"] == "reported"
    assert states[8]["hadamard"]["state"] == "known"


def test_doubled_board_matrices_are_skew_hadamard(board, data):
    """The doubling construction applied to the board's matrices of orders 356 and 476."""
    import numpy as np

    display_rows = site_build().display_rows
    (record,) = [x for x in data["types"]["skew-hadamard"]["since"] if "doubles" in x]
    assert record["orders"] == [2 * o for o in record["doubles"]]
    held = orders.board_holdings(board[1])
    for order in record["doubles"]:
        rows, _ = display_rows(board[1][held[order]["skew-hadamard"][0]])
        h = gs_array(rows)
        s = h - np.eye(order, dtype=int)
        one = np.eye(order, dtype=int)
        double = np.block([[s + one, s + one], [s - one, -s + one]])
        m = 2 * order
        assert np.array_equal(double + double.T, 2 * np.eye(m, dtype=int))
        assert np.array_equal(double @ double.T, m * np.eye(m, dtype=int))


def test_box_lists_exactly_the_open_orders_of_the_data_file(board, data, states):
    bib = Bibliography()
    box = orders.box(data, board[1], bib)
    assert box.startswith('<aside class="open-orders" id="open-orders"><h3>Open orders</h3>')
    for name in orders.TYPES:
        listed = re.search(rf'<li id="open-{name}" data-orders="([^"]*)">([^<]*)', box)
        assert [int(o) for o in listed.group(1).split()] == data["types"][name]["open"]
        assert listed.group(1).split() == [
            str(o) for o, s in states.items() if s[name]["state"] == "open"
        ]
        # The visible text lists the same orders, in order, and no others.
        shown = [int(o) for o in re.findall(r"\b\d+\b", listed.group(2).split(":", 1)[-1])]
        shown = [o for o in shown if o != data["range"]["max"]]
        assert shown == data["types"][name]["open"]
        # Every order settled or reported since the sources is in the box.
        for record in data["types"][name]["since"]:
            for order in record["orders"]:
                assert re.search(rf"\b{order}\b", box), order
    assert box.count("<li") <= 4
    assert "first" not in box.lower()
    assert 'href="cells/n89-skew-sds.html">356</a> (found here)' in box
    assert 'href="cells/n119-skew-sds.html">476</a>' in box
    # Every source of the data file is cited by number and resolves to the references page.
    for key in data["sources"]:
        assert f'href="references.html#ref-{key}">[{bib.number[key]}]' in box


def test_front_page_carries_the_box_and_no_orders_page():
    module = site_build()
    source = (ROOT / "site/build.py").read_text()
    assert "orders.box(ORDERS, r, BIB)" in source
    assert "orders.html" not in source and not hasattr(orders, "page")
    assert '<a href="#open-orders">open orders</a>' in source
    assert "orders.html" not in module.shell("x", "")
    for name in ("README.md", "TRACKS.md", "notes/board-operations.md"):
        assert "orders.html" not in (ROOT / name).read_text(), name


def test_grid_columns_agree_with_the_data_file(data, states):
    for n in range(1, orders.BOARD_MAX_N + 1, 2):
        found = cells(orders.grid_cells(n, states))
        assert [name for _, name, *_ in found] == list(orders.TYPES)
        for classes, name, order, state, text in found:
            assert int(order) == 4 * n and text == str(4 * n)
            t = data["types"][name]
            if state == "open":
                assert 4 * n in t["open"]
            elif state == "reported":
                assert 4 * n in t["listed_open"] and 4 * n not in t["open"]
            elif state == "known":
                assert 4 * n not in t["listed_open"]
            else:
                assert state == "board" and states[4 * n][name]["entries"]
            assert classes == orders.STATES[state][1]
        # Each existence cell links to the board cell of n for its type.
        hrefs = re.findall(r'href="([^"]+)"', orders.grid_cells(n, states))
        assert hrefs == [f"cells/n{n}-general.html", f"cells/n{n}-skew-sds.html"]
    row = dict(
        (name, (state, text)) for _, name, _, state, text in cells(orders.grid_cells(89, states))
    )
    assert row == {"hadamard": ("board", "356"), "skew-hadamard": ("board", "356")}
    # Where the board holds no skew-Hadamard matrix of order 4n and the sources do
    # not list the order as open, the grid shows the order as known.
    skew = data["types"]["skew-hadamard"]
    for n in range(1, orders.BOARD_MAX_N + 1, 2):
        if not states[4 * n]["skew-hadamard"]["entries"] and 4 * n not in skew["listed_open"]:
            assert ("skew-hadamard", "known") in [
                (name, state) for _, name, _, state, _ in cells(orders.grid_cells(n, states))
            ]


def test_unresolved_skew_cells_with_a_known_order(board, states):
    d, r, b = board
    by_cell = {(c["n"], c["stratum"]): c for c in summary(d, r, b)["cells"]}
    # An unresolved cell for quadruples with a skew block does not make its order
    # open: the order is known whenever the sources do not list it as open.
    for n in range(1, orders.BOARD_MAX_N + 1, 2):
        if (
            by_cell[n, "skew-sds"]["status"] == "unresolved"
            and not states[4 * n]["skew-hadamard"]["listed_open"]
        ):
            assert states[4 * n]["skew-hadamard"]["state"] == "known"


def test_track_labels_and_groups():
    module = site_build()
    assert module.TRACK_LABELS["skew-sds"] == "With a skew block"
    assert module.TRACK_LABELS["general"] == "Any"
    assert tuple(module.TRACK_LABELS) == TRACKS
    assert module.EXISTENCE_GROUP == "matrix of order 4n"
    assert module.TRACK_GROUP == "cyclic Goethals–Seidel quadruples at n, by family"


def test_front_page_grid_merges_the_order_and_track_columns(board, states):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    head, rows, options = module.grid(stats, r)
    # n, the two order columns, then the four named families; the tracks for
    # any quadruple and for a skew block have no columns of their own.
    columns = re.findall(r'<th scope="col"[^>]*>([^<]*)</th>', head)
    assert columns == ["n", "hadamard", "skew-hadamard"] + [
        module.TRACK_LABELS[x] for x in module.PRIMARY
    ]
    assert "With a skew block<" not in head and ">Any<" not in head
    # The column filter offers every column, the two tracks through their order columns.
    values = re.findall(r'<option value="([a-z-]+)">', options)
    assert values == ["general", "skew-sds", *module.PRIMARY]
    by_cell = {(c["n"], c["stratum"]): c for c in stats["cells"]}
    marked = module.new_cells(stats, r)
    for n, row in zip(range(1, orders.BOARD_MAX_N + 1, 2), rows.split("</tr>")):
        found = {name: (classes, state) for classes, name, _, state, _ in cells(row)}
        assert {name: state for name, (_, state) in found.items()} == {
            name: states[4 * n][name]["state"] for name in orders.TYPES
        }
        hrefs = re.findall(r'href="([^"]+)"', row)
        assert hrefs == [f"cells/n{n}-{t}.html" for t in ("general", "skew-sds", *module.PRIMARY)]
        tracks = re.findall(r'data-track="([a-z-]+)"', row)
        assert tracks == ["general", "skew-sds", *module.PRIMARY]
        # Markers of the merged tracks sit on the order cells.
        cells_html = row.split("<td")[2:]
        for track, html in zip(("general", "skew-sds", *module.PRIMARY), cells_html):
            assert ('class="participant"' in html) == ((track, n) in marked), (n, track)
            c = by_cell[n, track]
            assert f'data-status="{c["status"]}"' in html
            if c["contradicted"]:
                assert "contradicted" in html.split('"')[1]
    row = dict(zip(range(1, orders.BOARD_MAX_N + 1, 2), rows.split("</tr>")))
    # At 85 the board's cyclic skew cell is unresolved and the order is known elsewhere.
    skew85 = dict((name, (state, text)) for _, name, _, state, text in cells(row[85]))
    assert skew85["skew-hadamard"] == ("known", "340")
    assert 'data-status="unresolved"' in row[85].split("<td")[3]
    assert row[89].split("<td")[3].count('class="participant"') == 1


def test_order_cell_shows_only_the_order():
    info = {"state": "board", "listed_open": False}
    board = {"hadamard": {"count": "588 classes"}, "skew-hadamard": {"count": "247 classes"}}
    held = orders.grid_cells(13, {52: {name: info for name in orders.TYPES}}, board=board)
    assert [text for *_, text in cells(held)] == ["52", "52"]
    assert "<span>52</span>" in held and 'class="classes"' not in held
    # The class count stays in the tooltip while the board holds the matrix.
    assert "; 588 classes" in held and "; 247 classes" in held
    known = {"state": "known", "listed_open": False}
    other = orders.grid_cells(13, {52: {name: known for name in orders.TYPES}}, board=board)
    assert "classes" not in other
    source = (ROOT / "site/build.py").read_text() + (ROOT / "site/style.css").read_text()
    assert "count-key" not in source and ".classes" not in source


def test_cell_page_lists_the_quadruples_before_a_restricted_obstruction(board):
    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (89, "skew-sds"))
    page = module.cell_page(cell, d, r)
    assert "n = 89 · With a skew block" in page
    found = page.index("Eleven skew quadruples at n = 89")
    obstruction = page.index("Rules out one restricted family at n = 89 only")
    assert found < obstruction


def test_new_skew_order_carries_the_new_result_marker():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "orders_for_marker", pathlib.Path(__file__).parents[1] / "site" / "orders.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = {"state": "board", "listed_open": True}
    states = {356: {name: info for name in module.TYPES}}
    marked = module.grid_cells(89, states, {89})
    plain = module.grid_cells(89, states)
    assert marked.count('class="participant"') == 1 and 'class="participant"' not in plain
    assert marked.index('data-type="skew-hadamard"') < marked.index('class="participant"')
