"""The table of Hadamard and skew-Hadamard orders: its data file, page and grid columns."""

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
    """(type, order, state, class, text) of every existence cell in a piece of HTML."""
    return re.findall(
        r'<td class="([^"]*)" data-type="([a-z-]+)" data-order="(\d+)" data-state="([a-z]+)">'
        r"<a [^>]*>([^<]*)</a></td>",
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
    assert states[356]["skew-hadamard"]["entries"] == [
        "n89-b2a223b0fb7500feb95f",
        "n89-eec2e43c042090883c04",
    ]
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
    # Orders the sources give as known and the board does not hold.
    assert states[340]["skew-hadamard"]["state"] == "known"
    assert states[516]["skew-hadamard"]["state"] == "known"
    assert states[404]["hadamard"]["state"] == "known"
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


def test_page_lists_and_rows_equal_the_data_file(board, data, states):
    bib = Bibliography()
    page = orders.page(data, board[1], bib)
    for name in orders.TYPES:
        listed = re.search(rf'<p id="open-{name}" data-orders="([^"]*)">', page).group(1)
        assert [int(o) for o in listed.split()] == data["types"][name]["open"]
    found = cells(page)
    assert len(found) == 2 * len(orders.in_range(data))
    shown_open = {name: [] for name in orders.TYPES}
    for classes, name, order, state, text in found:
        order = int(order)
        assert state == states[order][name]["state"]
        assert (text, classes) == orders.STATES[state]
        if state == "open":
            shown_open[name].append(order)
    for name in orders.TYPES:
        assert shown_open[name] == data["types"][name]["open"]
    assert [int(o) for o in re.findall(r'<tr id="o(\d+)"', page)] == list(orders.in_range(data))
    # Open and reported rows are marked for the filter and styled apart.
    assert '<tr id="o404" data-flag="open">' in page
    assert '<tr id="o668" data-flag="open reported">' in page
    assert '<tr id="o356" data-flag="">' in page
    assert "found no earlier skew-Hadamard matrix of order 356" in page
    assert "first" not in page.split('<section id="status">')[1].split("</section>")[0].lower()
    # Every source of the data file is cited by number and listed in full.
    for key in data["sources"]:
        assert f'href="references.html#ref-{key}">[{bib.number[key]}]' in page
    assert 'href="cells/n89-skew-sds.html">skew</a>' in page
    assert 'href="cells/n129-skew-sds.html">no skew</a>' in page
    assert '<td class="held"><a href="cells/n101-general.html">none</a></td>' in page


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
    row = dict(
        (name, (state, text)) for _, name, _, state, text in cells(orders.grid_cells(89, states))
    )
    assert row == {"hadamard": ("board", "356"), "skew-hadamard": ("board", "356")}
    # The cell for quadruples with a skew block is unresolved at 85 and 129,
    # while skew-Hadamard matrices of orders 340 and 516 are known.
    for n in (85, 129):
        assert ("skew-hadamard", "known") in [
            (name, state) for _, name, _, state, _ in cells(orders.grid_cells(n, states))
        ]


def test_unresolved_skew_cells_with_a_known_order(board, states):
    d, r, b = board
    by_cell = {(c["n"], c["stratum"]): c for c in summary(d, r, b)["cells"]}
    for n in (85, 129):
        assert by_cell[n, "skew-sds"]["status"] == "unresolved"
        assert states[4 * n]["skew-hadamard"]["state"] == "known"


def test_track_labels_and_groups():
    module = site_build()
    assert module.TRACK_LABELS["skew-sds"] == "With a skew block"
    assert module.TRACK_LABELS["general"] == "Any"
    assert tuple(module.TRACK_LABELS) == TRACKS
    assert module.EXISTENCE_GROUP == "matrix of order 4n, any construction"
    assert module.TRACK_GROUP == "cyclic Goethals–Seidel quadruples at n"
    assert 'href="orders.html">orders</a>' in module.shell("x", "")


def test_cell_page_lists_the_quadruples_before_a_restricted_obstruction(board):
    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (89, "skew-sds"))
    page = module.cell_page(cell, d, r)
    assert "n = 89 · With a skew block" in page
    found = page.index("Two quadruples with a skew block at n=89")
    obstruction = page.index("No quadruple at n=89 has one skew block and all four blocks")
    assert found < obstruction
