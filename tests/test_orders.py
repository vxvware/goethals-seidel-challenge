"""The table of Hadamard and skew-Hadamard orders: its data file, grid columns and front-page box."""

import copy
import re
import sys

import pytest

from verify import census_engine
from verify.board import load, summary
from verify.common import BOARD_MAX_N, EVEN_TRACKS, ROOT, TRACKS, read_json

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
        # A quadruple with a skew block has odd n.
        assert all(o <= 4 * BOARD_MAX_N and (name == "hadamard" or o % 8 == 4) for o in held)


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
    # Orders the board does not hold. Use even n, where no quadruple has a skew block, so
    # that a new board entry cannot turn one into "board".
    assert states[536]["skew-hadamard"]["state"] == "verified"
    assert states[544]["skew-hadamard"]["state"] == "verified"
    assert states[712]["skew-hadamard"]["state"] == states[820]["skew-hadamard"]["state"] == "verified"
    assert states[532]["hadamard"]["state"] == "known"
    assert states[404]["skew-hadamard"]["state"] == "open"
    assert states[668]["hadamard"]["state"] == "known"
    # n = 94 is reached by none of the constructions of Williamson quadruples at even n.
    assert states[376]["hadamard"]["state"] == "known"
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


# The 8-Williamson array, transcribed from the paper cited as kotsireas2008new.
WILLIAMSON8 = """
A -B -C -D -E -F -G -H
B  A -D  C -F  E  H -G
C  D  A -B -G -H  E  F
D -C  B  A -H  G -F  E
E  F  G  H  A -B -C -D
F -E  H -G  B  A  D -C
G -H -E  F  C -D  A  B
H  G -F -E  D  C -B  A
"""


def found_record(data):
    stored = orders.validate_matrices(orders.load_matrices(), data, Bibliography())
    found = [x for x in stored.values() if x["origin"] == "found" and x["array"] == "williamson8"]
    assert [x["order"] for x in found] == [808]
    return found[0]


def test_found_matrix_is_the_array_of_its_rows(data):
    """The stored matrix of order 808 is the 8-Williamson array of the eight rows of its record."""
    import lzma

    import numpy as np

    import skew

    x = found_record(data)
    since = [s for s in data["types"]["skew-hadamard"]["since"] if s["status"] == "found"]
    assert [(s["orders"], s["date"], s["sources"]) for s in since] == [([808], x["date"], x["sources"])]
    text = lzma.decompress((ROOT / x["file"]).read_bytes()).decode()
    assert skew.found_text(x) == text == skew.stored_text(x, check=False)
    # The same array from the matrices of the paper: T is the cyclic shift, R the back
    # diagonal, a block is a polynomial in T, and the symmetric blocks are multiplied by R.
    v = x["parameters"]["v"]
    one = np.eye(v, dtype=int)
    shift, back = np.roll(one, 1, axis=1), one[::-1]
    assert shift[0, 1] == shift[v - 1, 0] == 1 and back[0, v - 1] == 1
    powers = [np.linalg.matrix_power(shift, k) for k in range(v)]
    rows = [[1 if c == "+" else -1 for c in line] for line in x["rows"]]
    poly = [sum(c * power for c, power in zip(row, powers, strict=True)) for row in rows]
    blocks = [poly[0]] + [sign * (m @ back) for sign, m in zip(x["block_signs"][1:], poly[1:], strict=True)]
    names = dict(zip("ABCDEFGH", blocks, strict=True))
    pattern = [line.split() for line in WILLIAMSON8.strip().splitlines()]
    numbered = [[("ABCDEFGH".index(e[-1]) + 1) * (-1 if e[0] == "-" else 1) for e in band] for band in pattern]
    assert numbered == [list(band) for band in skew.WILLIAMSON8]
    w = np.block([[-names[e[1]] if e[0] == "-" else names[e] for e in band] for band in pattern])
    assert skew.matrix_text(w.tolist()) == text
    # What the paper asks of the blocks, and what the record says of the rows.
    assert np.array_equal(blocks[0] + blocks[0].T, 2 * one)
    assert all(np.array_equal(m, m.T) for m in blocks[1:])
    assert np.array_equal(sum(m @ m.T for m in blocks), 8 * v * one)
    assert [sum(row) for row in rows] == x["row_sums"] == [1, 1, 1, 19, 21, 1, 1, 1]
    assert x["multiplier"] == {"generator": 36, "order": 5} and pow(36, 5, v) == 1
    assert all(row[36 * j % v] == row[j] for row in rows for j in range(v))


def test_found_matrix_that_disagrees_with_its_record_is_rejected(data):
    import skew

    x = found_record(data)
    rows = x["rows"]
    for change, message in (
        ({"rows": [rows[0], rows[2], rows[1], *rows[3:]]}, "not the array of the rows"),
        ({"block_signs": [1] * 8}, "not the array of the rows"),
        ({"block_signs": [-1] * 8}, "block signs"),
        ({"rows": [rows[1], *rows[1:]]}, "skew type"),
        ({"rows": [*rows[:7], rows[0]]}, "not symmetric"),
        ({"rows": rows[:7]}, "eight rows"),
        ({"multiplier": {"generator": 2, "order": 5}}, "order 5 modulo 101"),
        ({"multiplier": {"generator": 10, "order": 4}}, "not fixed by the multiplier"),
        ({"row_sums": [1, 1, 1, 21, 19, 1, 1, 1]}, "row sums"),
        ({"rows": [*rows[:7], rows[1]]}, "do not add to zero"),
    ):
        with pytest.raises(ValueError, match=message):
            skew.stored_text(x | change, check=False)


# The bordered array as wallis1972some prints it, for the (1, -1)-matrices A, B, D (type 1) and C (type 2)
# of four subsets of the group, with e the row of ones; its diagonal is -1.
WALLIS_WHITEMAN = """
-1  +1  +1  +1    e    e    e    e
-1  -1  -1  +1   -e    e   -e    e
-1  +1  -1  -1   -e    e    e   -e
-1  -1  +1  -1   -e   -e    e    e
-eT  eT  eT  eT   A    B    C    D
-eT -eT -eT  eT  -BT   AT  -D    C
-eT  eT -eT -eT  -C    DT   A   -BT
-eT -eT  eT -eT  -DT  -C    B    AT
"""
# The same array as momihara2019constructions states it, with M1, M2, M4 of type 1 and M3 of type 2.
MOMIHARA_XIANG = """
 1  -1  -1  -1   -e   -e   -e   -e
 1   1   1  -1    e   -e    e   -e
 1  -1   1   1    e   -e   -e    e
 1   1  -1   1    e    e   -e   -e
 eT -eT -eT -eT  -M1  -M2  -M3  -M4
 eT  eT  eT -eT   M2T -M1T  M4  -M3
 eT -eT  eT  eT   M3  -M4T -M1   M2T
 eT  eT -eT  eT   M4T  M3  -M2  -M1T
"""
# Orders above the range of the board, with the group, the multiplier, its order and its orbits outside 0.
BORDERED = {
    1416: ([353], [231], 11, 32),
    1712: ([7, 61], [4, 12], 15, 30),
    2360: ([19, 31], [16, 20], 45, 16),
    5336: ([31, 43], [18, 17], 105, 16),
}


def block_matrix(pattern, names):
    """The numpy block matrix of a pattern of signed names: a leading - negates, a trailing T transposes."""
    import numpy as np

    def block(word):
        sign = -1 if word[0] == "-" else 1
        word = word.lstrip("+-")
        if word == "1":
            return np.full((1, 1), sign, dtype=np.int8)
        m = names[word.removesuffix("T")]
        return sign * (m.T if word.endswith("T") else m)

    return np.block([[block(word) for word in line.split()] for line in pattern.strip().splitlines()])


def as_text(h):
    """A numpy matrix of 1 and -1 as the bytes of a matrix file."""
    import numpy as np

    out = np.full((len(h), len(h) + 1), ord("\n"), dtype=np.uint8)
    out[:, :-1] = np.where(h > 0, ord("+"), ord("-"))
    return out.tobytes()


def bordered_records(data):
    stored = orders.validate_matrices(orders.load_matrices(), data, Bibliography())
    found = {order: x for order, x in stored.items() if orders.beyond_range(data, order)}
    assert sorted(found) == sorted(BORDERED) and all(x["array"] == "bordered" for x in found.values())
    return found


@pytest.mark.parametrize("order", sorted(BORDERED))
def test_bordered_matrix_is_the_array_of_the_sources(data, order):
    """A found matrix above the range is the bordered array of four subsets of a cyclic group, as both sources give it.

    The build places the rows on Z_m1 x Z_m2 with pairs in lexicographic order. Here the group is taken as
    the integers modulo w, the blocks are the circulant and back-circulant (1, -1)-matrices of the four
    subsets where a row is -1, and the two printed arrays are transcribed above.
    """
    import hashlib
    import math

    import numpy as np

    import skew

    x = bordered_records(data)[order]
    moduli, generator, size, orbits = BORDERED[order]
    assert (x["group"], x["multiplier"]) == (moduli, {"generator": generator, "order": size})
    assert "file" not in x and x["origin"] == "found" and x["date"] == "2026-11-25"
    assert order not in orders.in_range(data) and not (ROOT / f"matrices/skew/skew-{order}.txt.xz").exists()
    w = math.prod(moduli)
    assert order == 4 * (w + 1) and x["parameters"]["group_order"] == w
    text = skew.stored_text(x, check=False)
    assert text == skew.found_text(x) and hashlib.sha256(text.encode()).hexdigest() == x["sha256"]
    # The moduli are coprime, so r -> (r mod m1, r mod m2) is a bijection from the integers modulo w,
    # and place[r] is the position of the image of r in the record's order.
    place = np.zeros(w, dtype=int)
    for m in moduli:
        place = place * m + np.arange(w) % m
    assert sorted(place.tolist()) == list(range(w))
    rows = np.array([[1 if c == "+" else -1 for c in line] for line in x["rows"]], dtype=np.int8)
    cyclic = rows[:, place]
    # The subsets of the sources: where a row is -1. Each has (w - 1) / 2 elements, and the first holds
    # exactly one of z and -z for every nonzero z.
    subsets = [set(np.flatnonzero(row == -1).tolist()) for row in cyclic]
    assert all(len(subset) == (w - 1) // 2 for subset in subsets)
    assert 0 not in subsets[0] and all((z in subsets[0]) != ((w - z) in subsets[0]) for z in range(1, w))
    r = np.arange(w)
    difference, total = (r[None, :] - r[:, None]) % w, (r[None, :] + r[:, None]) % w
    back = np.empty(w, dtype=int)
    back[place] = r

    def incidence(subset, index):
        """The (1, -1)-matrix with 1 where the index lies in the subset, in the record's order of the group."""
        member = np.full(w, -1, dtype=np.int8)
        member[sorted(subset)] = 1
        return member[index][np.ix_(back, back)]

    m1, m2, m4 = (incidence(subsets[k], difference) for k in (0, 1, 3))
    m3 = incidence(subsets[2], total)
    # The hypothesis of both sources on the four subsets, as the first states it for their type 1 matrices.
    gram = sum(m.astype(np.float32) @ m.astype(np.float32).T for m in (m1, m2, incidence(subsets[2], difference), m4))
    assert np.array_equal(gram, 4 * (w + 1) * np.eye(w, dtype=np.float32) - 4)
    del gram
    e = np.ones((1, w), dtype=np.int8)
    printed = block_matrix(WALLIS_WHITEMAN, {"e": e, "A": m1, "B": m2, "C": m3, "D": m4})
    stated = block_matrix(MOMIHARA_XIANG, {"e": e, "M1": m1, "M2": m2, "M3": m3, "M4": m4})
    assert printed.shape == (order, order) and np.array_equal(stated, -printed)
    assert as_text(stated) == text.encode()
    del printed
    # Entries are 1 and -1 and the order is below 2^24, so the float32 products are exact integers.
    h = stated.astype(np.float32)
    identity = np.eye(order, dtype=np.float32)
    assert np.array_equal(h + h.T, 2 * identity)
    assert np.array_equal(h @ h.T, order * identity)
    # What the record says of the rows.
    assert x["row_sums"] == [1, 1, 1, 1] == rows.sum(axis=1).tolist()
    found = skew.multiplier_orbits(moduli, generator)
    assert len(found) == orbits + 1 and found[0] == [0] and math.lcm(*map(len, found)) == size
    assert all(len({line[i] for i in orbit}) == 1 for line in x["rows"] for orbit in found)


def test_bordered_array_over_a_group_that_is_not_cyclic():
    """Four rows on Z_3 x Z_3 give a skew-Hadamard matrix of order 40; the blocks follow the definition."""
    import itertools

    import numpy as np

    import skew

    moduli, rows = [3, 3], ["++-+++---", "--+-++-++", "--+-++++-", "--++-+++-"]
    text = skew.bordered_text(rows, moduli)
    assert skew.check_skew_text(text) == 40
    elements = list(itertools.product(range(3), range(3)))
    assert skew.group_elements(moduli) == elements and elements[:4] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    values = [dict(zip(elements, (1 if c == "+" else -1 for c in row), strict=True)) for row in rows]
    one = lambda f: np.array([[f[(c - a) % 3, (d - b) % 3] for c, d in elements] for a, b in elements])  # noqa: E731
    two = lambda f: np.array([[f[(c + a) % 3, (d + b) % 3] for c, d in elements] for a, b in elements])  # noqa: E731
    a, b, c, d = one(values[0]), one(values[1]), two(values[2]), one(values[3])
    assert skew.type_one(rows[0], moduli) == skew.matrix_text(a.tolist()).split()
    assert skew.type_two(rows[2], moduli) == skew.matrix_text(c.tolist()).split()
    # The type 1 matrix is not circulant here, and no order of the elements makes it so: none has order 9.
    assert a.tolist() != [[a[0][(j - i) % 9] for j in range(9)] for i in range(9)]
    names = {"e": np.ones((1, 9), dtype=int), "M1": -a, "M2": -b, "M3": -c, "M4": -d}
    assert as_text(block_matrix(MOMIHARA_XIANG, names)) == text.encode()
    # Over a cyclic group the two types are the circulant and the back-circulant matrix of the row.
    row = "++-+---"
    assert skew.type_one(row, [7]) == ["".join(row[(j - i) % 7] for j in range(7)) for i in range(7)]
    assert skew.type_two(row, [7]) == ["".join(row[(j + i) % 7] for j in range(7)) for i in range(7)]
    with pytest.raises(ValueError, match="four rows"):
        skew.bordered_text(rows[:3], moduli)
    with pytest.raises(ValueError, match="four rows"):
        skew.bordered_text(rows, [3, 5])


def test_bordered_record_that_disagrees_with_its_rows_is_rejected(data):
    import skew

    x = bordered_records(data)[1712]
    rows = x["rows"]
    for change, message in (
        ({"rows": [rows[0], rows[2], rows[1], rows[3]]}, "does not match its sha256"),
        ({"sha256": "0" * 64}, "does not match its sha256"),
        ({"rows": [rows[1], *rows[1:]]}, "skew type"),
        ({"rows": rows[:3]}, "four rows"),
        ({"rows": [rows[0], rows[1], rows[2], rows[3].translate(str.maketrans("+-", "-+"))]}, "row sums"),
        ({"row_sums": [1, 1, 1, 3]}, "row sums"),
        ({"rows": [*rows[:3], rows[1]]}, "do not add to -4"),
        ({"multiplier": {"generator": [4, 12], "order": 5}}, "does not have order 5"),
        ({"multiplier": {"generator": [4, 13], "order": 3}}, "not fixed by the multiplier"),
        ({"multiplier": {"generator": [7, 12], "order": 15}}, "not a unit"),
        ({"group": [3, 7, 61]}, "not a product of odd cyclic groups of order 427"),
        ({"group": [427]}, "skew type"),
        ({"parameters": {"group_order": 428}}, "group order"),
        ({"array": "other"}, "unknown array"),
    ):
        with pytest.raises(ValueError, match=message):
            skew.stored_text(x | change, check=False)
    # A record whose rows give a matrix that is not skew-Hadamard fails the exact check: here the
    # digest is the one of the rebuilt text and the rows are checked by a function that accepts them.
    import hashlib

    wrong = skew.bordered_text([rows[0], rows[1], rows[2], rows[1]], x["group"])
    bad = x | {"rows": [*rows[:3], rows[1]], "sha256": hashlib.sha256(wrong.encode()).hexdigest()}
    with pytest.MonkeyPatch.context() as patch, pytest.raises(ValueError, match="not a skew-Hadamard matrix"):
        patch.setattr(skew, "bordered_rows", lambda record: record["rows"])
        skew.stored_text(bad)


def test_found_order_is_marked_as_a_new_result_of_the_board(data, states, blocks):
    info = states[808]["skew-hadamard"]
    assert info["state"] == "verified" and info["listed_open"] and orders.found(info)
    assert 808 not in data["types"]["skew-hadamard"]["open"]
    cell = orders.grid_cells(202, states).split("<td")[2]
    assert 'data-state="verified"' in cell and 'href="orders/skew-808.html"' in cell
    assert '<span class="participant" title="New result"></span>' in cell
    assert "new result of this board" in cell and 'data-origins="participant-result"' in cell
    assert "this one dates from 2026-11-25" in cell
    # Order 712 is marked through the featured result at 356, which the build passes in.
    assert 'class="participant"' not in orders.grid_cells(178, states).split("<td")[2]
    block, keys = blocks[808]
    assert "Construction: 8-Williamson array, v = 101, Z_101 " in block
    assert " A −B −C −D −E −F −G −H\n B  A −D  C −F  E  H −G\n" in block
    assert "in order, each multiplied by −1" in block and "here row 4 begins with −1" in block
    assert "{1, 36, 84, 87, 95}, the subgroup of order 5 generated by 36" in block
    assert "The row sums are 1, 1, 1, 19, 21, 1, 1, 1, and their squares add to 808." in block
    assert "The build rebuilds the matrix from the eight rows" in block
    assert "The result has not been reviewed." in block and "does not claim priority" in block
    assert "on 2026-11-25 found no earlier skew-Hadamard matrix of order 808" in block
    assert all(line in block for line in found_record(data)["rows"])
    assert {"kotsireas2008new", "sage2026unknown", "cati2024database", "kishan2026skew"} <= keys
    assert "not a Goethals–Seidel quadruple" in block and "to our knowledge" not in block


def test_intro_is_one_sentence(data):
    text = orders.intro(data)
    assert text.startswith("Each row is an order 4n") and text.count(". ") == 0 and text.endswith(".")


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
        if n % 2:
            # Each order cell links to the board cell of n for its type.
            assert hrefs == [f"cells/n{n}-general.html", f"cells/n{n}-skew-sds.html"]
        else:
            # No quadruple has a skew block at even n: the skew-Hadamard order has its own page.
            assert hrefs == [f"cells/n{n}-general.html", f"orders/skew-{4 * n}.html"]
    row = {name: (state, text) for _, name, _, state, text in cells(orders.grid_cells(89, states))}
    assert row == {"hadamard": ("board", "356"), "skew-hadamard": ("board", "356")}
    assert all(orders.has_cell(n, track) for n in (1, 131, 133, 299) for track in TRACKS)
    assert [t for t in TRACKS if orders.has_cell(300, t)] == list(orders.EVEN_TRACKS) == list(EVEN_TRACKS)
    assert not orders.has_cell(301, "williamson") and orders.BOARD_MAX_N == BOARD_MAX_N == 300


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
    # An even row: the two order cells, a Williamson board cell, and the three families with a skew block
    # marked impossible. No row has a cell without a state.
    assert "data-extra" not in rows and ">·<" not in rows and "Not tracked" not in rows
    for n in (2, 132, 300):
        even = by_n[n]
        assert {name: state for _, name, _, state, _ in cells(even)} == {
            name: states[4 * n][name]["state"] for name in orders.TYPES
        }
        assert re.findall(r'data-track="([a-z-]+)"', even) == ["general", "skew-sds", *module.PRIMARY]
        assert re.findall(r'href="([^"]+)"', even) == [
            f"cells/n{n}-general.html",
            f"orders/skew-{4 * n}.html",
            f"cells/n{n}-williamson.html",
        ]
        assert even.count('class="blank impossible"') == 3 and even.count(">×</td>") == 3
    for n in range(1, 301):
        assert by_n[n].count('class="blank impossible"') == (0 if n % 2 else 3)
        assert by_n[n].count("<td") == 7

    def family(n, track, by_n=by_n):
        """The text of a family cell and its tooltip."""
        html = by_n[n].split("<td")[4 + module.PRIMARY.index(track)]
        return re.search(r">([^<]*)(?:<span[^>]*></span>)?</a>", html).group(1), re.search(
            r'title="([^"]*)"', html
        ).group(1)

    # Even n: the board's own equiv-v1 count where it enumerated the order, with the published count in the
    # tooltip. The module's board skips the replays, so the ten censuses are replayed here.
    quads = {ident: x for ident, x in r.items() if x["type"] == "quadruple"}
    replayed = dict(r)
    for n in range(2, 21, 2):
        ident = f"williamson-n{n}"
        assert family(n, "williamson")[0].endswith("+")
        replayed[ident] = r[ident] | census_engine.verify(d[ident], quads)
    full = module.grid(summary(d, replayed, b), replayed)[1]
    full = {int(re.match(r'<tr data-n="(\d+)"', row).group(1)): row for row in full.split("</tr>") if row}
    counts = [int(family(n, "williamson", full)[0]) for n in range(2, 21, 2)]
    assert counts == [1, 5, 4, 5, 12, 36, 76, 27, 540, 412]
    assert "23 classes published, under an equivalence that merges more" in family(18, "williamson", full)[1]
    # Published and not on the board: the count stays in the tooltip, since it is not an equiv-v1 count.
    text, tip = family(64, "williamson")
    assert text == "69,960+" and "69,960 classes published" in tip
    # The same cell without quadruples on the board says so, whatever the board holds today.
    bare = by_cell[64, "williamson"] | {
        "quadruples": [],
        "censuses": [],
        "catalogues": [],
        "artifact_status": "published-not-imported",
    }
    assert "the board holds no quadruple of this type yet" in module.grid_cell(bare, set(), r)["accessible"]
    assert family(128, "williamson")[0] == "1+" and "classes published" not in family(128, "williamson")[1]
    # Open: no source the board cites gives one, which is all the tooltip says.
    for n, track in ((94, "williamson"), (106, "williamson"), (133, "williamson"), (133, "best"), (299, "good")):
        text, tip = family(n, track)
        assert text == "?" and tip.endswith("no quadruple of this type is known to us in the cited sources"), (n, track)
    # Past 131: the constructed families are verified cells, and best cells follow the square rule.
    assert family(135, "williamson")[0] == family(297, "williamson")[0] == family(135, "kkss")[0] == "1+"
    assert family(299, "best")[0] == "×" and family(137, "kkss")[0] in ("×", "?")
    assert by_n[178].count('class="participant"') == 1
    # A best cell where 4n - 3 is not a square is empty by a theorem and shows the cross, not a zero.
    assert ">×</a>" in row[5] and ">×</a>" not in row[7]
    # An open order off the board reads as a research target in the status filter.
    assert 'data-state="open" data-track="skew-sds" data-status="unresolved"' in by_n[101]
    assert 'data-state="verified" data-track="skew-sds" data-status="exists"' in by_n[2]


def test_default_view_shows_every_row_and_through_n_shortens_it():
    source = (ROOT / "site/build.py").read_text()
    assert '<option>131</option><option selected value="{orders.BOARD_MAX_N}">' in source
    assert ", every n</option></select>" in source
    # Every row has cells, so no row is hidden for lack of them, and the legend has no item for an untracked cell.
    assert "untracked" not in source and "untracked" not in (ROOT / "site/style.css").read_text()
    script = (ROOT / "site/app.js").read_text()
    assert "data-extra" not in script and "\nfilter();\n" in script
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
        lambda m: next(x for x in m if x["origin"] == "found").pop("rows"),
        lambda m: next(x for x in m if x["origin"] == "found").update(date="2026-11-24"),
        lambda m: next(x for x in m if x["origin"] == "found").update(software="kishan2026skew"),
        lambda m: m.remove(next(x for x in m if x["origin"] == "found")),
        lambda m: m[0].update(origin="found"),
        # An order above the range is a found matrix without a file, with the fields of its array.
        lambda m: m[-1].update(file="matrices/skew/skew-5336.txt.xz"),
        lambda m: m[-1].update(origin="repository", software="kishan2026skew"),
        lambda m: m[-1].pop("group"),
        lambda m: m[-1].pop("array"),
        lambda m: m[-1].update(array="other"),
        lambda m: m[-1].update(order=5338),
        lambda m: m[-1].update(order=1416),
        lambda m: m[0].pop("file"),
        lambda m: next(x for x in m if x["origin"] == "found").pop("block_signs"),
    ):
        broken = copy.deepcopy(doc)
        change(broken["matrices"])
        with pytest.raises(ValueError):
            orders.validate_matrices(broken, data, bib)
    # A found matrix needs its record of status found in the orders file, and that record a source.
    unlisted = copy.deepcopy(data)
    since = unlisted["types"]["skew-hadamard"]["since"]
    record = next(x for x in since if x["status"] == "found")
    record["status"] = "reported"
    with pytest.raises(ValueError, match="found in one of the two baseline files only"):
        orders.validate_matrices(doc, unlisted, bib)
    record.update(status="found", sources=[])
    with pytest.raises(ValueError, match="names a source"):
        orders.validate(unlisted, bib)


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
    assert written == {4 * n for n in range(2, 301, 2)} and len(written) == 150
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
