"""The table of Hadamard and skew-Hadamard orders: its data file, grid columns and front-page box."""

import copy
import re
import sys

import pytest

from verify.board import load, summary
from verify.common import ROOT, TRACKS, read_json

sys.path.insert(0, str(ROOT / "site"))
from test_site import gs_array, site_build

import orders
from references import Bibliography


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
        r"[^>]*><(?:a|span) [^>]*><span>([^<]*)</span>",
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
    assert {"n89-b2a223b0fb7500feb95f", "n89-eec2e43c042090883c04"} <= set(states[356]["skew-hadamard"]["entries"])
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
        since = {o for record in t["since"] if record["status"] == "reported" for o in record["orders"]}
        # No order is shown as reported: an announced Hadamard order is known, and a
        # skew-Hadamard order has a checked matrix file.
        assert not [o for o, s in states.items() if s[name]["state"] == "reported"]
        if name == "hadamard":
            assert since <= {o for o, s in states.items() if s[name]["state"] == "known"}
        if name == "skew-hadamard":
            assert {s[name]["state"] for s in states.values()} == {"board", "verified", "open"}
            assert since <= {o for o, s in states.items() if s[name]["state"] == "verified"}
    # Orders the board does not hold. Use orders beyond the board's range, n > 131, so
    # that a new board entry cannot turn one into "board".
    assert states[532]["skew-hadamard"]["state"] == "verified"
    assert states[540]["skew-hadamard"]["state"] == "verified"
    assert states[712]["skew-hadamard"]["state"] == states[820]["skew-hadamard"]["state"] == "verified"
    assert states[532]["hadamard"]["state"] == "known"
    assert states[404]["skew-hadamard"]["state"] == "open"
    assert states[668]["hadamard"]["state"] == "known"
    assert states[8]["hadamard"]["state"] == "known"
    # An order that is neither open nor held needs a stored matrix.
    with pytest.raises(ValueError, match="no stored matrix of order 4,"):
        orders.table(data, {}, stored=[])


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


def test_stored_doublings_are_rebuilt_from_their_board_entries(board):
    """The stored matrices of origin board are the doubling [[H, H], [-H^T, H^T]] of the entry's array."""
    import hashlib
    import lzma

    import skew

    display_rows = site_build().display_rows
    records = [x for x in read_json(ROOT / "baselines/skew-matrices.json")["matrices"] if x["origin"] == "board"]
    assert [x["order"] for x in records] == [712, 952]
    for x in records:
        assert x["generated"] == f"doubling of board entry {x['entry']}"
        rows, _ = display_rows(board[1][x["entry"]])
        h = skew.gs_array(rows)
        ht = [list(column) for column in zip(*h, strict=True)]
        double = [row + row for row in h] + [[-v for v in row] + row for row in ht]
        text = skew.matrix_text(double).encode()
        assert len(double) == x["order"] and hashlib.sha256(text).hexdigest() == x["sha256"]
        assert lzma.decompress((ROOT / x["file"]).read_bytes()) == text


def test_intro_names_the_smallest_open_skew_orders(data):
    text = orders.intro(data)
    first, second = data["types"]["skew-hadamard"]["open"][:2]
    assert f"the smallest open skew-Hadamard orders are {first} and {second}" in text
    assert "first" not in text.lower().replace("first,", "")
    assert "for odd n up to 131; the rows for even n and for larger n have the two order columns only" in text
    assert "choosing 131 under Through n hides them" in text


def test_front_page_has_no_orders_page_or_box():
    source = (ROOT / "site/build.py").read_text()
    assert "orders.intro(ORDERS)" in source and "orders.box" not in source
    assert "orders.html" not in source and not hasattr(orders, "page")
    for name in ("README.md", "TRACKS.md", "notes/board-operations.md"):
        text = (ROOT / name).read_text()
        assert "orders.html" not in text and "#open-orders" not in text, name


def test_grid_columns_agree_with_the_data_file(data, states):
    for n in range(1, 301):
        found = cells(orders.grid_cells(n, states))
        assert [name for _, name, *_ in found] == list(orders.TYPES)
        for classes, name, order, state, text in found:
            assert int(order) == 4 * n and text == str(4 * n)
            t = data["types"][name]
            if state == "open":
                assert 4 * n in t["open"]
            elif state == "known":
                assert name == "hadamard" and 4 * n not in t["open"]
            elif state == "verified":
                assert name == "skew-hadamard" and 4 * n not in t["open"] and not states[4 * n][name]["entries"]
            else:
                assert state == "board" and states[4 * n][name]["entries"]
            assert classes == orders.STATES[state][1]
        hrefs = re.findall(r'href="([^"]+)"', orders.grid_cells(n, states))
        if orders.covered(n):
            # Each order cell links to the board cell of n for its type.
            assert hrefs == [f"cells/n{n}-general.html", f"cells/n{n}-skew-sds.html"]
        else:
            # The board has no cell: the skew-Hadamard order has its own page and the Hadamard cell no link.
            assert hrefs == [f"orders/skew-{4 * n}.html"]
    row = {name: (state, text) for _, name, _, state, text in cells(orders.grid_cells(89, states))}
    assert row == {"hadamard": ("board", "356"), "skew-hadamard": ("board", "356")}
    assert [orders.covered(n) for n in (1, 2, 131, 132, 133)] == [True, False, True, False, False]


def test_orders_listed_without_a_matrix_say_so_in_the_tooltip(data, states):
    """An order the cited lists give without a matrix and that now has one says so, with the date."""
    t = data["types"]["skew-hadamard"]
    filled = {o for record in t["since"] for o in record["orders"]}
    assert {356, 476, 712, 952, 820, 996} <= filled
    for n in range(1, 301):
        skew_cell = orders.grid_cells(n, states).split("<td")[2]
        assert 'class="since"' not in skew_cell and "filled since" not in skew_cell
        if 4 * n in filled:
            date = next(record["date"] for record in t["since"] if 4 * n in record["orders"])
            assert f"the cited sources list this order without a matrix, and this one dates from {date}" in skew_cell
    assert states[668]["hadamard"]["state"] == "known"


def test_unresolved_skew_cells_with_a_known_order(board, states):
    d, r, b = board
    by_cell = {(c["n"], c["stratum"]): c for c in summary(d, r, b)["cells"]}
    # An unresolved cell for quadruples with a skew block does not make its order
    # open: the order has a checked matrix whenever the sources do not list it as open.
    for n in range(1, orders.BOARD_MAX_N + 1, 2):
        if by_cell[n, "skew-sds"]["status"] == "unresolved" and not states[4 * n]["skew-hadamard"]["listed_open"]:
            assert states[4 * n]["skew-hadamard"]["state"] == "verified"


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
    assert columns == ["n", "hadamard", "skew-hadamard"] + [module.TRACK_LABELS[x] for x in module.PRIMARY]
    assert "With a skew block<" not in head and ">Any<" not in head
    # The column filter offers every column, the two tracks through their order columns.
    values = re.findall(r'<option value="([a-z-]+)">', options)
    assert values == ["general", "skew-sds", *module.PRIMARY]
    by_cell = {(c["n"], c["stratum"]): c for c in stats["cells"]}
    marked = module.new_cells(stats, r)
    by_n = {int(re.match(r'<tr data-n="(\d+)"', row).group(1)): row for row in rows.split("</tr>") if row}
    assert list(by_n) == list(range(1, 301))
    for n in range(1, orders.BOARD_MAX_N + 1, 2):
        row = by_n[n]
        assert "data-extra" not in row
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
        for track, html in zip(("general", "skew-sds", *module.PRIMARY), cells_html, strict=False):
            assert ('class="participant"' in html) == ((track, n) in marked), (n, track)
            c = by_cell[n, track]
            assert f'data-status="{c["status"]}"' in html
            if c["contradicted"]:
                assert "contradicted" in html.split('"')[1]
    row = by_n
    # At 85 the board's cyclic skew cell is unresolved and the order has a matrix by another construction.
    skew85 = {name: (state, text) for _, name, _, state, text in cells(row[85])}
    assert skew85["skew-hadamard"] == ("verified", "340")
    assert 'data-status="unresolved"' in row[85].split("<td")[3]
    assert row[89].split("<td")[3].count('class="participant"') == 1
    # A row the board has no cells for: the two order cells, then a blank cell per family.
    for n in (2, 132, 133, 300):
        extra = by_n[n]
        assert extra.startswith(f'<tr data-n="{n}" data-extra>')
        assert {name: state for _, name, _, state, _ in cells(extra)} == {
            name: states[4 * n][name]["state"] for name in orders.TYPES
        }
        assert re.findall(r'data-track="([a-z-]+)"', extra) == ["general", "skew-sds", *module.PRIMARY]
        assert extra.count('<td class="blank') == len(module.PRIMARY) and "cells/" not in extra
        assert re.findall(r'href="([^"]+)"', extra) == [f"orders/skew-{4 * n}.html"]
        # At even n no skew circulant exists, so the three families with a skew block are marked impossible.
        assert extra.count('class="blank impossible"') == (0 if n % 2 else 3)
    # An open order off the board reads as a research target in the status filter.
    assert 'data-state="open" data-track="skew-sds" data-status="unresolved"' in by_n[101]
    assert 'data-state="verified" data-track="skew-sds" data-status="exists"' in by_n[2]


def test_default_view_shows_every_row_and_a_smaller_range_hides_the_uncovered_ones():
    source = (ROOT / "site/build.py").read_text()
    assert "<option>{orders.BOARD_MAX_N}</option><option selected value=" in source
    assert ", every n</option></select>" in source
    script = (ROOT / "site/app.js").read_text()
    assert "!every&&row.hasAttribute('data-extra')" in script and "\nfilter();\n" in script
    assert "if(td.classList.contains('blank'))continue;" in script


def test_order_cell_shows_only_the_order():
    info = {"state": "board", "listed_open": False}
    board = {"hadamard": {"count": "588 classes"}, "skew-hadamard": {"count": "247 classes"}}
    held = orders.grid_cells(13, {52: dict.fromkeys(orders.TYPES, info)}, board=board)
    assert [text for *_, text in cells(held)] == ["52", "52"]
    assert "<span>52</span>" in held and 'class="classes"' not in held
    # The class count stays in the tooltip while the board holds the matrix.
    assert "; 588 classes" in held and "; 247 classes" in held
    known = {"state": "known", "listed_open": False}
    other = orders.grid_cells(13, {52: dict.fromkeys(orders.TYPES, known)}, board=board)
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
    states = {356: dict.fromkeys(module.TYPES, info)}
    marked = module.grid_cells(89, states, {89})
    plain = module.grid_cells(89, states)
    assert marked.count('class="participant"') == 1 and 'class="participant"' not in plain
    assert marked.index('data-type="skew-hadamard"') < marked.index('class="participant"')


def test_no_page_lists_the_skew_orders():
    """The grid carries the skew-Hadamard orders; the separate page and its script are gone."""
    import skew

    source = (ROOT / "site/build.py").read_text()
    assert 'skew.html">' not in source and "skew orders" not in source
    # The build copies no skew.js and removes the two files an older build left in the output.
    assert source.count('"skew.js"') == source.count('"skew.html"') == 1 and "unlink(missing_ok=True)" in source
    assert not (ROOT / "site/skew.js").exists() and not hasattr(skew, "page")


@pytest.fixture(scope="module")
def blocks(data, states):
    """The matrix block of every skew-Hadamard order, with the keys it cites."""
    import skew

    bib = Bibliography()
    stored = orders.validate_matrices(orders.load_matrices(), data, bib)
    chosen = skew.chosen_entries(states)
    return {
        order: skew.matrix_block(data, order, s["skew-hadamard"], stored, chosen, bib) for order, s in states.items()
    }


def test_matrix_block_states_the_construction_and_links_the_file(data, states, blocks):
    import skew

    bib = Bibliography()
    t = data["types"]["skew-hadamard"]
    chosen = skew.chosen_entries(states)
    assert len(blocks) == 300
    for order, (block, keys) in blocks.items():
        state = states[order]["skew-hadamard"]["state"]
        assert block.startswith(f"<h2>Matrix of order {order}</h2>")
        assert f'<span class="tag">{orders.STATES[state][0]}</span>' in block
        assert "[@" not in block and "—" not in block and "reported" not in block
        assert keys <= set(bib.entries) and 'class="cite"' in block
        assert (f'href="../matrices/skew-{order}.txt" download' in block) == (state != "open"), order
        if state == "open":
            assert order in t["open"] and keys == set(t["listed_open_sources"])
            assert "only relative to these sources" in block and "priority" in block
        elif state == "board":
            assert chosen[order] == min(states[order]["skew-hadamard"]["entries"])
            assert f'href="../entries/{chosen[order]}.html"' in block and "Goethals–Seidel array" in block
        else:
            assert "Construction: " in block
    filled = {o for record in t["since"] for o in record["orders"]}
    for order, (block, _) in blocks.items():
        assert "filled since" not in block
        assert ("without a matrix; this one dates from 20" in block) == (order in filled), order
    assert "Construction: Paley I, q = 7 " in blocks[8][0]
    assert "generated with SageMath" in blocks[8][0] and "hadamard_matrix_paleyI" in blocks[8][0]
    assert "Doubling of a matrix of order 92 (Good matrices)" in blocks[184][0] and "is not cited" in blocks[184][0]
    assert "doubling of the board's matrix of order 356" in blocks[712][0]
    assert "doubling of the board's matrix of order 476" in blocks[952][0]
    assert "No source read" not in blocks[712][0]
    for order in (820, 996, 1148):
        assert "The source is a software repository, not peer reviewed." in blocks[order][0]
        assert "kishan2026skew" in blocks[order][1]
    assert "passagemath2026combinat" in blocks[8][1] and "passagemath2026combinat" not in blocks[820][1]


def test_inconsistent_matrix_records_are_rejected(data):
    bib = Bibliography()
    doc = orders.load_matrices()
    assert len(orders.validate_matrices(doc, data, bib)) == len(doc["matrices"])
    for change in (
        lambda m: m.append(dict(m[0])),
        lambda m: m[0].update(order=404, file="matrices/skew/skew-404.txt.xz"),
        lambda m: m[0].update(sources=["nobody1900nothing"]),
        lambda m: m[0].update(origin="memory"),
        lambda m: m[0].update(file="matrices/skew/other.txt.xz"),
    ):
        broken = copy.deepcopy(doc)
        change(broken["matrices"])
        with pytest.raises(ValueError):
            orders.validate_matrices(broken, data, bib)


def test_stored_matrix_with_a_wrong_digest_or_a_changed_entry_is_rejected(data, tmp_path, monkeypatch):
    import lzma

    import skew

    record = dict(orders.load_matrices()["matrices"][0])
    text = skew.stored_text(record)
    assert skew.check_skew_text(text) == record["order"]
    with pytest.raises(ValueError, match="sha256"):
        skew.stored_text(record | {"sha256": "0" * 64}, check=False)
    # A changed matrix whose record carries the digest of the changed text fails the exact check.
    import hashlib

    changed = ("-" if text[1] == "+" else "+").join((text[:1], text[2:])).encode()
    monkeypatch.setattr(skew, "ROOT", tmp_path)
    (tmp_path / "bad.txt.xz").write_bytes(lzma.compress(changed))
    bad = record | {"file": "bad.txt.xz", "sha256": hashlib.sha256(changed).hexdigest()}
    assert skew.stored_text(bad, check=False) == changed.decode()
    with pytest.raises(ValueError, match="not a skew-Hadamard matrix"):
        skew.stored_text(bad)


def test_skew_array_matches_the_pictured_layout(board, states):
    import numpy as np

    import skew

    display_rows = site_build().display_rows
    for order in (12, 52, 356):
        rows, _ = display_rows(board[1][skew.chosen_entries(states)[order]])
        assert np.array_equal(np.array(skew.gs_array(rows)), gs_array(rows))


def test_skew_check_rejects_a_matrix_that_is_not_skew_hadamard():
    import skew

    good = skew.gs_array([[1], [1], [1], [1]])
    skew.check_skew_hadamard(good)
    assert skew.matrix_text(good) == "++++\n-+-+\n-++-\n--++\n"
    symmetric = [[1, 1], [1, -1]]
    flipped = [list(row) for row in good]
    flipped[1][2], flipped[2][1] = 1, -1
    for bad in (symmetric, flipped, [[1, 0], [0, 1]], [[1, 1, 1]], []):
        with pytest.raises(ValueError):
            skew.check_skew_hadamard(bad)
    assert skew.check_skew_text("++++\n-+-+\n-++-\n--++\n") == 4
    for bad in ("++++\n-+-+\n-++-\n--++", "++\n+-\n", "+0\n0+\n", "+++\n", "", "\n"):
        with pytest.raises(ValueError):
            skew.check_skew_text(bad)


def test_every_written_matrix_is_skew_hadamard(board, data, states, tmp_path):
    """Every order that is not open has a file that is a skew-Hadamard matrix; an open order has none."""
    import numpy as np

    import skew

    module = site_build()
    chosen = module.write_skew_matrices(tmp_path, states, board[1])
    assert chosen == skew.chosen_entries(states)
    is_open = set(data["types"]["skew-hadamard"]["open"])
    written = {int(p.stem.removeprefix("skew-")) for p in (tmp_path / "matrices").iterdir()}
    assert written == set(orders.in_range(data)) - is_open and len(written) == 300 - len(is_open)
    assert {356, 476, 712, 952, 996, 1200} <= written and not {404, 428, 668} & written
    for order in sorted(written):
        lines = (tmp_path / skew.file_name(order)).read_text().splitlines()
        assert len(lines) == order and all(len(x) == order and set(x) <= {"+", "-"} for x in lines)
        # Entries are 1 and -1 and the order is at most 1200, so the float64 products are exact integers.
        h = np.array([[1.0 if c == "+" else -1.0 for c in line] for line in lines])
        assert np.array_equal(h + h.T, 2 * np.eye(order)), order
        assert np.array_equal(h @ h.T, order * np.eye(order)), order
        if order in chosen:
            # The file is the array of the chosen entry's rows with a skew block first.
            rows, pattern = module.display_rows(board[1][chosen[order]])
            assert pattern[0] == "k" and np.array_equal(h, gs_array(rows)), order
        else:
            assert states[order]["skew-hadamard"]["state"] == "verified"


def test_order_pages_cover_the_rows_without_a_board_cell(board, data, states, tmp_path):
    import skew

    module = site_build()
    module.write_order_pages(tmp_path, states, skew.chosen_entries(states))
    written = {int(p.stem.removeprefix("skew-")) for p in (tmp_path / "orders").iterdir()}
    assert written == {4 * n for n in range(1, 301) if not orders.covered(n)} and len(written) == 234
    is_open = set(data["types"]["skew-hadamard"]["open"])
    for order in sorted(written):
        page = (tmp_path / f"orders/skew-{order}.html").read_text()
        assert f"<h1>Skew-Hadamard matrix of order {order}</h1>" in page and "<h2>References</h2>" in page
        assert (f'href="../matrices/skew-{order}.txt" download' in page) == (order not in is_open), order
        assert 'href="../style.css"' in page and "—" not in page
    # The pages of the covered rows are the board cells, which carry the same block.
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (85, "skew-sds"))
    block, _ = skew.matrix_block(data, 340, states[340]["skew-hadamard"], module.STORED, {}, Bibliography())
    assert block in module.cell_page(cell, d, r, block) and "Matrix of order 340" in block
    assert "Matrix of order" not in module.cell_page(cell, d, r)
