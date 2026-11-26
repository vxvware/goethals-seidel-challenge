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


@pytest.fixture(scope="module")
def pages(board, data, states, tmp_path_factory):
    """The order pages, written once: each found matrix has its rows checked when its page is written."""
    import skew

    out = tmp_path_factory.mktemp("pages")
    site_build().write_order_pages(out, states, skew.chosen_entries(states))
    return out


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
    records = [x for x in orders.load_matrices()["matrices"] if x["origin"] == "board"]
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
    assert [x["order"] for x in found] == [808, 3352]
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


def test_order_3352_is_the_eight_block_array_of_its_rows(board, data, states, tmp_path, pages):
    """The found matrix of order 3352 has no stored file: it is the 8-Williamson array of the rows of its record."""
    import hashlib

    import numpy as np
    from test_site import SLUGS, visible

    import skew

    x = above_records(data)[3352]
    rows, v = x["rows"], 419
    assert "file" not in x and x["array"] == "williamson8" and x["sources"] == ["kotsireas2008new"]
    assert not (ROOT / "matrices/skew/skew-3352.txt.xz").exists() and 3352 not in orders.in_range(data)
    text = skew.stored_text(x, check=False)
    assert text == skew.found_text(x) and hashlib.sha256(text.encode()).hexdigest() == x["sha256"]
    # The same array from the matrices of the paper: a block is a polynomial in the cyclic shift T,
    # whose entry (i, j) is the coefficient of T^(j - i), and the symmetric blocks are multiplied by R.
    values = np.array([[1 if c == "+" else -1 for c in line] for line in rows], dtype=np.float32)
    one = np.eye(v, dtype=np.float32)
    back = one[::-1]
    j, i = np.meshgrid(np.arange(v), np.arange(v))
    poly = [row[(j - i) % v] for row in values]
    shift = np.roll(one, 1, axis=1)
    assert all(
        np.array_equal(np.linalg.matrix_power(shift, k), ((j - i) % v == k).astype(np.float32)) for k in (1, 5, 418)
    )
    blocks = [poly[0]] + [sign * (m @ back) for sign, m in zip(x["block_signs"][1:], poly[1:], strict=True)]
    names = dict(zip("ABCDEFGH", (m.astype(np.int8) for m in blocks), strict=True))
    w = block_matrix(WILLIAMSON8, names)
    assert w.shape == (3352, 3352) and as_text(w) == text.encode()
    # Entries are 1 and -1 and the order is below 2^24, so the float32 products are exact integers.
    h = w.astype(np.float32)
    identity = np.eye(3352, dtype=np.float32)
    assert np.array_equal(h + h.T, 2 * identity) and np.array_equal(h @ h.T, 3352 * identity)
    del h, identity
    # What the paper asks of the blocks, and what the record says of the rows.
    assert np.array_equal(blocks[0] + blocks[0].T, 2 * one) and all(np.array_equal(m, m.T) for m in blocks[1:])
    assert np.array_equal(sum(m @ m.T for m in blocks), 8 * v * one)
    sums = [int(row.sum()) for row in values]
    assert sums == x["row_sums"] == [1, 21, 21, 21, 21, 23, 23, 23] and sum(t * t for t in sums) == 3352
    assert x["block_signs"] == [1, -1, -1, -1, -1, -1, -1, -1]
    assert x["multiplier"] == {"generator": 13, "order": 11}
    assert pow(13, 11, v) == 1 and 13 % v != 1 and all(v % p for p in range(2, 21))
    assert all(line[13 * k % v] == line[k] for line in rows for k in range(v))
    assert [line[0] for line in rows] == list("+----+++")
    # The arithmetic of the note.
    assert (3351, 1675, 837, 3352) == (3 * 1117, 5**2 * 67, 3**3 * 31, 2**3 * 419)
    tmp_path = pages
    page = (tmp_path / "orders/skew-3352.html").read_text()
    seen = visible(page)
    assert "Order 3352 is above the range of the board, which has a row for every order up to 1200" in seen
    assert "Construction: 8-Williamson array, v = 419, Z_419" in seen
    assert "eight rows of length 419, not a Goethals–Seidel quadruple, and the order is above the board's range" in seen
    assert "in order, each multiplied by −1" in seen and "here rows 2, 3, 4 and 5 begin with −1" in seen
    assert "the subgroup of order 11 generated by 13" in seen
    assert "The row sums are 1, 21, 21, 21, 21, 23, 23, 23, and their squares add to 3352." in seen
    assert "524,288 rows of skew type, and 92,378 symmetric rows for each of the row sums 21 and 23" in seen
    assert "a sample and not a census" in seen and "share the seven symmetric rows" in seen
    assert (
        "The repository holds the eight rows and the sha256 of the matrix file, not the file, which is 11.2 MB" in seen
    )
    assert "compares it with the stored file" not in seen
    assert "rests on the exact check of the build" in seen
    assert "which gives t = 4 for m = 419, while 3352 = 2³ × 419" in seen
    assert "3351 = 3 × 1117, 1675 = 5² × 67 and 837 = 3³ × 31 are not prime powers" in seen
    assert (
        "The result has not been reviewed." in seen and "is not a proof that no published construction applies" in seen
    )
    assert 'href="../matrices/skew-3352.txt" download' in page and all(line in page for line in rows)
    assert not re.search(SLUGS, seen) and "—" not in seen and "[@" not in seen and "to our knowledge" not in seen
    # The page of order 808, whose matrix is a stored file, keeps its wording.
    old = visible((tmp_path / "orders/skew-808.html").read_text())
    assert "compares it with the stored file" in old and "above the board's range" not in old


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
    2696: ([673], [600], 21, 32),
    7496: ([1873], [491], 39, 48),
    9672: ([2417], [1131], 151, 16),
    10376: ([2593], [483], 81, 32),
    10440: ([2609], [2525], 163, 16),
    12360: ([3089], [1677], 193, 16),
    5336: ([31, 43], [18, 17], 105, 16),
}
# Orders found with the tie of order 7496: the prime w = n/4 - 1, the multiplier, the map of the tie,
# the factors of n - 1 and of n/2 - 1, and how many rows of sum 1 the search went through.
TIED = {
    14792: (3697, 27, 1430, (7, 2113), (3, 5, 17, 29), "83,646,370,560"),
    16712: (4177, 2, 479, (17, 983), (3, 5, 557), "20,437,026,510"),
    17096: (4273, 22, 685, (5, 13, 263), (3, 7, 11, 37), "136,804,567,065"),
    18248: (4561, 25, 2589, (71, 257), (3, 3041), "49,559,549,805"),
    18632: (4657, 16, 1100, (31, 601), (3, 3, 3, 3, 5, 23), "2,090,801,775"),
    20936: (5233, 49, 4107, (5, 53, 79), (3, 3, 1163), "121,734,443,700"),
    22088: (5521, 34, 4654, (13, 1699), (3, 3, 3, 409), "30,503,715,615"),
}
# Each joins the orders of the bordered array, with 48 orbits of a multiplier of order (w - 1)/48, so that
# the tests of that array, the slow ones among them, cover it. A further order of this kind is one line above.
BORDERED |= {order: ([w], [g], (w - 1) // 48, 48) for order, (w, g, *_) in TIED.items()}


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


def text_sha256(h):
    """The sha256 of the matrix file of a numpy matrix of 1 and -1, built in place to spare memory."""
    import hashlib

    import numpy as np

    out = np.full((len(h), len(h) + 1), ord("-"), dtype=np.uint8)
    out[:, -1] = ord("\n")
    out[:, :-1][h > 0] = ord("+")
    return hashlib.sha256(out).hexdigest()


def check_products(h, step=1024):
    """Assert H H^T = nI and H + H^T = 2I for an int8 matrix H of order n, in blocks of rows.

    Entries are 1 and -1 and n is below 2^24, so the float32 products are exact integers. Only one
    block of rows and one of columns are held as float32 at a time.
    """
    import numpy as np

    n = len(h)
    assert h.shape == (n, n) and n < 2**24 and np.array_equal(np.abs(h), np.ones_like(h))
    for i in range(0, n, step):
        rows = h[i : i + step]
        twice = rows.astype(np.int16) + h[:, i : i + step].T
        assert np.array_equal(twice, 2 * np.eye(n, dtype=np.int16)[i : i + step]), i
        left = rows.astype(np.float32)
        for j in range(i, n, step):
            product = left @ h[j : j + step].T.astype(np.float32)
            expected = np.zeros_like(product)
            if i == j:
                np.fill_diagonal(expected, n)
            assert np.array_equal(product, expected), (i, j)


def as_text(h):
    """A numpy matrix of 1 and -1 as the bytes of a matrix file."""
    import numpy as np

    out = np.full((len(h), len(h) + 1), ord("\n"), dtype=np.uint8)
    out[:, :-1] = np.where(h > 0, ord("+"), ord("-"))
    return out.tobytes()


def bordered_records(data):
    found = {order: x for order, x in above_records(data).items() if x["array"] == "bordered"}
    assert sorted(found) == sorted(BORDERED)
    return found


def above_records(data):
    """The found matrices of orders above the range of the board, which have no stored file."""
    stored = orders.validate_matrices(orders.load_matrices(), data, Bibliography())
    found = {order: x for order, x in stored.items() if orders.beyond_range(data, order)}
    assert sorted(found) == sorted([*BORDERED, 3352, 5392]) and found[5392]["array"] == "bordered8"
    return found


@pytest.mark.parametrize(
    "order", [pytest.param(o, marks=[pytest.mark.slow] if o > 8000 else []) for o in sorted(BORDERED)]
)
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
    assert "file" not in x and x["origin"] == "found"
    assert x["date"] == ("2026-11-26" if order > 14000 else "2026-11-25")
    assert order not in orders.in_range(data) and not (ROOT / f"matrices/skew/skew-{order}.txt.xz").exists()
    w = math.prod(moduli)
    assert order == 4 * (w + 1) and x["parameters"]["group_order"] == w
    text = skew.stored_text(x, check=False)
    assert hashlib.sha256(text.encode()).hexdigest() == x["sha256"] and text.count("\n") == order
    if not skew.rebuilt_in_build(order):
        # The build checks the rows only at this order; this is its exact check of the matrix.
        assert skew.check_skew_text(text) == order
    # From here on the text is needed only through its sha256, and the largest is 438 MB.
    del text
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
    stated = block_matrix(MOMIHARA_XIANG, {"e": e, "M1": m1, "M2": m2, "M3": m3, "M4": m4})
    assert stated.shape == (order, order) and stated.dtype == np.int8
    if order < 6000:
        printed = block_matrix(WALLIS_WHITEMAN, {"e": e, "A": m1, "B": m2, "C": m3, "D": m4})
        assert np.array_equal(stated, -printed)
        del printed
    assert text_sha256(stated) == x["sha256"]
    check_products(stated)
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


# The pattern of the bordered eight-block array, transcribed apart from site/skew.py.
EIGHT_BLOCKS = """
 X1  X2  X3  X4  X5  X6  X7  X8
-X2  X1  X4 -X3  X6 -X5 -X8  X7
-X3 -X4  X1  X2  X7  X8 -X5 -X6
-X4  X3 -X2  X1  X8 -X7  X6 -X5
-X5 -X6 -X7 -X8  X1  X2  X3  X4
-X6  X5 -X8  X7 -X2  X1 -X4  X3
-X7  X8  X5 -X6 -X3  X4  X1 -X2
-X8 -X7  X6  X5 -X4 -X3  X2  X1
"""


def eight_block_rows(w):
    """Every row of + and - of odd length w of skew type, and every symmetric one of sum 1."""
    import itertools

    half = (w - 1) // 2
    swap = {"+": "-", "-": "+"}
    of_skew_type = [
        "+" + "".join(x) + "".join(swap[c] for c in reversed(x)) for x in itertools.product("+-", repeat=half)
    ]
    symmetric = [c + "".join(x) + "".join(reversed(x)) for c in "+-" for x in itertools.product("+-", repeat=half)]
    return of_skew_type, [x for x in symmetric if x.count("+") - x.count("-") == 1]


def autocorrelations(rows):
    """The sum over the rows of their periodic autocorrelations, at each nonzero shift."""
    w = len(rows[0])
    values = [[1 if c == "+" else -1 for c in x] for x in rows]
    return [sum(v[j] * v[(j + s) % w] for v in values for j in range(w)) for s in range(1, w)]


def eight_block_matrix(rows):
    """The bordered eight-block array from the matrices T and R, Kronecker products and the pattern above."""
    import numpy as np

    w = len(rows[0])
    one = np.eye(w, dtype=np.int8)
    shift, back = np.roll(one, 1, axis=1), one[::-1]
    j, i = np.meshgrid(np.arange(w), np.arange(w))
    values = np.array([[1 if c == "+" else -1 for c in x] for x in rows], dtype=np.int8)
    # x_0 I + x_1 T + ... + x_(w-1) T^(w-1) has x[(j - i) mod w] in position (i, j): T^k has its ones at j - i = k.
    circulant = [v[(j - i) % w] for v in values]
    if w < 20:
        powers = [np.linalg.matrix_power(shift.astype(int), k) for k in range(w)]
        assert all(
            np.array_equal(c, sum(a * m for a, m in zip(v, powers, strict=True)))
            for c, v in zip(circulant, values, strict=True)
        )
    # Entries are 0, 1 and -1, so the float32 products are exact.
    times_back = [(c.astype(np.float32) @ back.astype(np.float32)).astype(np.int8) for c in circulant[1:]]
    names = {"X1": circulant[0]} | {f"X{k + 2}": m for k, m in enumerate(times_back)}
    p = block_matrix(EIGHT_BLOCKS, names)
    s = block_matrix(EIGHT_BLOCKS, {f"X{k}": np.ones((1, 1), dtype=np.int8) for k in range(1, 9)})
    e = np.ones((w, 1), dtype=np.int8)
    return np.block([[s.T, np.kron(s, e.T)], [-np.kron(s.T, e), p]]), names


def test_eight_block_pattern_has_the_properties_the_argument_uses():
    """What the page's argument for the bordered eight-block array asks of the pattern, checked exactly."""
    import numpy as np

    import skew

    words = [line.split() for line in EIGHT_BLOCKS.strip().splitlines()]
    pattern = [[int(word.lstrip("-")[1:]) * (-1 if word[0] == "-" else 1) for word in line] for line in words]
    assert pattern == [list(band) for band in skew.BORDERED8]
    sign = np.sign(np.array(pattern))
    block = np.abs(np.array(pattern))
    # S is a skew-Hadamard matrix of order 8.
    assert np.array_equal(sign @ sign.T, 8 * np.eye(8, dtype=int)) and np.array_equal(
        sign + sign.T, 2 * np.eye(8, dtype=int)
    )
    # Block 1 fills the diagonal, the same block stands in (i, j) and (j, i), and every block stands once
    # in each row and each column.
    assert np.array_equal(block, block.T) and set(np.diag(block)) == {1}
    assert all(sorted(line) == list(range(1, 9)) for line in [*block.tolist(), *block.T.tolist()])
    # For two different rows the columns fall into pairs: the same two blocks in exchanged places,
    # with opposite products of signs.
    for a in range(8):
        for b in range(a):
            where = {(block[a][m], block[b][m]): sign[a][m] * sign[b][m] for m in range(8)}
            assert len(where) == 8 and all(where[x, y] == -where[y, x] for x, y in where)


@pytest.mark.parametrize(("w", "count"), [(3, 2), (5, 4), (7, 0), (9, 30)])
def test_bordered_eight_block_array_at_small_orders(w, count):
    """Every choice of rows that meets the conditions of the array gives a skew-Hadamard matrix of order 8(w + 1).

    The rows are found by exhaustive search: one of skew type and seven symmetric of sum 1, up to the
    order of the seven, with periodic autocorrelations adding to -8.
    """
    import itertools

    import numpy as np

    import skew

    of_skew_type, symmetric = eight_block_rows(w)
    found = [
        [first, *others]
        for first in of_skew_type
        for others in itertools.combinations_with_replacement(symmetric, 7)
        if set(autocorrelations([first, *others])) == {-8}
    ]
    assert len(found) == count
    order = 8 * (w + 1)
    for rows in found:
        text = skew.bordered8_text(rows)
        assert skew.check_skew_text(text) == order
        h, names = eight_block_matrix(rows)
        assert as_text(h) == text.encode()
        h = h.astype(int)
        assert np.array_equal(h @ h.T, order * np.eye(order, dtype=int))
        assert np.array_equal(h + h.T, 2 * np.eye(order, dtype=int))
        # The steps of the page's argument: the blocks are amicable, their Gram matrices add to
        # 8(w + 1)I - 8J, and every row sum is 1.
        blocks = [names[f"X{k}"].astype(int) for k in range(1, 9)]
        assert all(np.array_equal(x @ y.T, y @ x.T) for x in blocks for y in blocks)
        assert np.array_equal(sum(x @ x.T for x in blocks), order * np.eye(w, dtype=int) - 8)
        assert all((x.sum(axis=1) == 1).all() for x in blocks)
        assert np.array_equal(blocks[0] + blocks[0].T, 2 * np.eye(w, dtype=int))
        assert all(np.array_equal(x, x.T) for x in blocks[1:])
    # Rows that miss one condition do not give a skew-Hadamard matrix: the sum -8, a row sum, the skew type.
    if w == 9:
        good = found[0]
        other = next(x for x in symmetric if set(autocorrelations([*good[:7], x])) != {-8})
        minus = good[7].translate(str.maketrans("+-", "-+"))
        assert set(autocorrelations([*good[:7], minus])) == {-8}
        for rows in ([*good[:7], other], [*good[:7], minus], [good[1], *good[1:]]):
            with pytest.raises(ValueError):
                skew.check_skew_text(skew.bordered8_text(rows))


def eight_block_record(rows, **changes):
    """A record for the bordered eight-block array of rows of length 9, as the baseline file would hold it."""
    import hashlib

    import skew

    return {
        "order": 80,
        "sha256": hashlib.sha256(skew.bordered8_text(rows).encode()).hexdigest(),
        "construction": "Bordered eight-block array",
        "parameters": {"group_order": 9, "group": "Z_9"},
        "sources": ["wallis1972some", "kotsireas2008new"],
        "origin": "found",
        "array": "bordered8",
        "date": "2026-11-25",
        "contributors": [{"name": "vxvware", "github": "vxvware"}],
        "derivation": "It was derived for a test.",
        "rows": rows,
        "multiplier": {"generator": 1, "order": 1},
        "row_sums": [1] * 8,
        "search": "An exhaustive search.",
        "checked": "Nothing else was checked.",
        "generated": "a test",
    } | changes


def test_bordered_eight_block_record_is_checked_and_described(data):
    from test_site import SLUGS, visible

    import skew

    rows = ["+++++----", "+++----++", "++-+--+-+", "++--++--+", "+-++--++-", "+-++--++-", "+-++--++-", "+-+-++-+-"]
    x = eight_block_record(rows)
    text = skew.stored_text(x)
    assert text == skew.bordered8_text(rows) == skew.found_text(x) and text.count("\n") == 80
    swap = str.maketrans("+-", "-+")
    for change, message in (
        ({"rows": [rows[0], rows[2], rows[1], *rows[3:]]}, "does not match its sha256"),
        ({"sha256": "0" * 64}, "does not match its sha256"),
        ({"rows": [rows[1], *rows[1:]]}, "skew type"),
        ({"rows": [*rows[:7], rows[0]]}, "not symmetric"),
        ({"rows": rows[:7]}, "eight rows"),
        ({"rows": [*rows[:7], rows[7].translate(swap)]}, "row sums"),
        ({"row_sums": [1] * 7 + [3]}, "row sums"),
        ({"rows": [*rows[:7], rows[1]]}, "do not add to -8"),
        ({"multiplier": {"generator": 1, "order": 3}}, "does not have order 3"),
        ({"multiplier": {"generator": 2, "order": 6}}, "not fixed by the multiplier"),
        ({"multiplier": {"generator": 3, "order": 1}}, "does not have order"),
        ({"order": 84}, "8\\(w \\+ 1\\)"),
        ({"parameters": {"group_order": 11}}, "group order"),
    ):
        with pytest.raises(ValueError, match=message):
            skew.stored_text(x | change, check=False)
    # The record needs its fields in the baseline file.
    assert orders.ARRAY_FIELDS["bordered8"] == ("derivation",)
    doc = {"matrices": [*orders.load_matrices()["matrices"], x | {"order": 2000}]}
    assert 2000 in orders.validate_matrices(doc, data, Bibliography())
    doc["matrices"][-1] = {k: v for k, v in x.items() if k != "derivation"} | {"order": 2000}
    with pytest.raises(ValueError, match="lacks a field of a found matrix"):
        orders.validate_matrices(doc, data, Bibliography())
    # The page says that the array is not from a published source and gives the argument as ours.
    page = skew.order_page(
        data | {"range": data["range"] | {"max": 76}}, 80, orders.beyond_state(), {80: x}, {}, Bibliography()
    )
    seen = visible(page)
    assert (
        "We did not find this array in the sources we read, which is not a proof that none is published: "
        "it was worked out for this board from the two cited." in seen
    )
    assert "We did not find an eight-block bordered skew array in the sources we read" in seen
    assert "which is not a proof that none is published" in seen and "It was derived for a test." in seen
    assert "The statement that this matrix is skew-Hadamard rests on the exact check of the build." in seen
    assert "The argument that follows is ours and has not been reviewed." in seen
    assert "X1 is the circulant matrix of the first of the eight rows" in seen
    assert "X2, …, X8 are the back-circulant matrices of the other seven" in seen
    assert "add to −8 at every nonzero shift" in seen and "these eight products add to 80I − 8J" in seen
    assert "The matrix H of order 80 = 8(9 + 1) has four parts." in seen
    assert "gives 8I + 8 × 9I = 80I for the first 8 rows" in seen
    assert "−X2  X1  X4 −X3  X6 −X5 −X8  X7\n" in page and "−1  1  1 −1  1 −1 −1  1\n" in page
    assert "  Sᵀ       S ⊗ eᵀ\n −Sᵀ ⊗ e   P" in page
    assert "the subgroup of order 1 generated by 1" in seen and "It has 8 orbits besides {0}." in seen
    assert "eight rows of length 9 placed in a bordered array, not a Goethals–Seidel quadruple" in seen
    assert all(line in page for line in rows)
    assert not re.search(SLUGS, seen) and "—" not in seen and "[@" not in seen and "to our knowledge" not in seen


def carried_by(row, t, w):
    """The row y with y(tz) = x(z) for every z modulo w."""
    out = [""] * w
    for z, c in enumerate(row):
        out[t * z % w] = c
    return "".join(out)


def autocorrelation_sum(rows, shifts, w):
    values = [[1 if c == "+" else -1 for c in row] for row in rows]
    return [sum(v[z] * v[(z + s) % w] for v in values for z in range(w)) for s in shifts]


def test_order_2696_ties_the_third_and_fourth_rows_to_the_first_two(board, data, states, tmp_path, pages):
    """The record of order 2696 names the map z -> 58z, which carries the first two rows to the other two."""
    from test_site import visible

    import skew

    records = bordered_records(data)
    x, w = records[2696], 673
    assert x["tie"] == {"multiplier": [58], "chains": [[1, 3], [2, 4]]}
    assert [o for o, r in records.items() if "tie" in r] == [2696, 7496, 10376, *TIED]
    a, b, c, d = x["rows"]
    assert 58 * 58 % w == w - 1 and skew.tie_classes(x) == 8
    assert all(c[58 * z % w] == a[z] and d[58 * z % w] == b[z] for z in range(w))
    assert carried_by(a, 58, w) == c and carried_by(b, 58, w) == d
    # Carried once more, the third row is the first read backwards, and the fourth the second read backwards.
    assert carried_by(c, 58, w) == a[0] + a[:0:-1] != a and carried_by(d, 58, w) == b[0] + b[:0:-1]
    # 58 is not among the multipliers that fix the rows, nor the negative of one.
    fixing = {pow(600, k, w) for k in range(21)}
    assert len(fixing) == 21 and 58 not in fixing and w - 58 not in fixing and c != a
    # The sum of the four autocorrelations is the same at s and at 58s, from the tie alone:
    # it holds for any two rows in place of a and b.
    one, two = "+" * 300 + "-" * 373, ("+-+" * 225)[:673]
    rows = [one, two, carried_by(one, 58, w), carried_by(two, 58, w)]
    total = autocorrelation_sum(rows, range(1, 40), w)
    assert total == autocorrelation_sum(rows, [58 * s % w for s in range(1, 40)], w) and len(set(total)) > 1
    for change, message in (
        ({"tie": {"multiplier": [600], "chains": [[1, 3], [2, 4]]}}, "row 3 is not row 1 carried"),
        ({"tie": {"multiplier": [672], "chains": [[1, 3], [2, 4]]}}, "row 3 is not row 1 carried"),
        ({"tie": {"multiplier": [615], "chains": [[1, 3], [2, 4]]}}, "row 3 is not row 1 carried"),
        ({"tie": {"multiplier": [58], "chains": [[1, 3], [4, 2]]}}, "row 2 is not row 4 carried"),
        ({"tie": {"multiplier": [58], "chains": [[1, 3]]}}, "does not sort the rows"),
        ({"tie": {"multiplier": [58], "chains": [[1, 3], [2], [4]]}}, "does not sort the rows"),
        ({"tie": {"multiplier": [58], "chains": [[1, 3]], "fixed": [2, 4]}}, "row 2 is not fixed"),
        ({"tie": {"multiplier": [0], "chains": [[1, 3], [2, 4]]}}, "not a unit"),
        ({"tie": {"multiplier": [58, 1], "chains": [[1, 3], [2, 4]]}}, "not a unit"),
    ):
        with pytest.raises(ValueError, match=message):
            skew.stored_text(x | change, check=False)
    assert w - 58 == 615
    tmp_path = pages
    seen = visible((tmp_path / "orders/skew-2696.html").read_text())
    assert "The rows are tied by the map σ: z ↦ 58z of G." in seen
    assert "a row x carried by σ is the row y with y(σ(z)) = x(z) for every z in G" in seen
    assert "Then the third row is the first carried by σ and the fourth row is the second carried by σ." in seen
    assert "is the same at s and at σ(s)" in seen and "has the first and second rows to choose" in seen
    assert "there are 8 such sets" in seen
    assert (
        "all 65,536 rows of skew type and all 601,080,390 rows of sum 1" in seen and "It found 1,024 solutions" in seen
    )
    assert "the map z ↦ 600z of G, which has order 21. It has 32 orbits besides {0}." in seen
    assert "which gives t = 18 for m = 337, while 2696 = 2³ × 337" in seen
    assert "673 is a prime congruent to 33 modulo 64, which is neither 5 modulo 8 nor 9 modulo 16" in seen
    assert (w % 64, w % 8, w % 16, 2695, 1347) == (33, 1, 1, 5 * 7**2 * 11, 3 * 449)
    assert "tied by the map" not in visible((tmp_path / "orders/skew-1416.html").read_text())
    other = visible((tmp_path / "orders/skew-5392.html").read_text())
    assert "Order 5392 is also twice 2696: doubling the matrix of order 2696" in other


def test_order_7496_ties_the_third_and_fourth_rows_to_the_second(board, data, states, tmp_path, pages):
    """The record of order 7496 names the map z -> 1823z, which fixes the first row and carries the second on."""
    from test_site import visible

    import skew

    x, w, t = bordered_records(data)[7496], 1873, 1823
    assert x["tie"] == {"multiplier": [t], "chains": [[2, 3, 4]], "fixed": [1]}
    a, b, c, d = x["rows"]
    fixing = {pow(491, k, w) for k in range(39)}
    assert len(fixing) == 39 and pow(t, 3, w) in fixing and t not in fixing and w - t not in fixing
    assert all(w % p for p in range(2, 44))
    # c(x) = b(t^-1 x) and d(x) = b(t^-2 x): the second row carried once and twice.
    inverse = pow(t, -1, w)
    assert all(c[z] == b[inverse * z % w] and d[z] == b[inverse * inverse * z % w] for z in range(w))
    assert carried_by(a, t, w) == a and carried_by(b, t, w) == c and carried_by(c, t, w) == d
    assert carried_by(d, t, w) == b != c and b != d
    # The first row is constant on the classes of sixteenth powers, which the multipliers and t generate;
    # the second is not.
    sixteenth = {pow(z, 16, w) for z in range(1, w)}
    assert sixteenth == {h * pow(t, k, w) % w for h in fixing for k in range(3)} and len(sixteenth) == 117
    assert all(a[h * z % w] == a[z] for h in sixteenth for z in range(1, w, 97))
    assert any(b[h * z % w] != b[z] for h in sixteenth for z in range(1, 40))
    # 16 classes of nonzero shifts under those maps, and 8 with z -> -z.
    assert skew.tie_classes(x) == 8 and (w - 1) // 117 == 16 and w - 1 not in sixteenth
    # The sum of the four autocorrelations is the same at s and at ts.
    shifts = [1, 2, 3, 5, 7, 11]
    assert autocorrelation_sum(x["rows"], [t * s % w for s in shifts], w) == autocorrelation_sum(x["rows"], shifts, w)
    for change, message in (
        ({"tie": {"multiplier": [t], "chains": [[2, 4, 3]], "fixed": [1]}}, "row 4 is not row 2 carried"),
        ({"tie": {"multiplier": [t], "chains": [[1, 2, 3, 4]]}}, "row 2 is not row 1 carried"),
        ({"tie": {"multiplier": [t], "chains": [[3, 4]], "fixed": [1, 2]}}, "does not close"),
        ({"tie": {"multiplier": [t], "chains": [[2, 3, 4]], "fixed": []}}, "does not sort the rows"),
        ({"tie": {"multiplier": [w - t], "chains": [[2, 3, 4]], "fixed": [1]}}, "row 3 is not row 2 carried"),
        ({"rows": [a, b, c, b]}, "do not add to -4"),
    ):
        with pytest.raises(ValueError, match=message):
            skew.stored_text(x | change, check=False)
    # A chain that closes on the rows needs its power of the multiplier among the multipliers of the rows.
    with pytest.raises(ValueError, match="to the power 3 is not a multiplier of the rows"):
        skew.tie_classes(x | {"multiplier": {"generator": [1], "order": 1}})
    tmp_path = pages
    seen = visible((tmp_path / "orders/skew-7496.html").read_text())
    assert "The rows are tied by the map σ: z ↦ 1823z of G." in seen
    assert (
        "Then the third row is the second carried by σ, the fourth row is the third carried by σ and the first row "
        "carried by σ is itself or itself read backwards."
    ) in seen
    assert "has the first and second rows to choose" in seen and "there are 8 such sets" in seen
    assert "the map z ↦ 491z of G, which has order 39. It has 48 orbits besides {0}." in seen
    assert "all 256 rows of skew type that the map leaves unchanged" in seen
    assert "It is a sample and not a census." in seen and "16,080,018,104 steps" in seen and "1,227 seconds" in seen
    assert "found 4 matches: 2 solutions, each together with the one that has its first row read backwards" in seen
    assert "which gives t = 5 for m = 937, while 7496 = 2³ × 937" in seen
    assert "7495 = 5 × 1499 and 3747 = 3 × 1249 are not prime powers" in seen
    assert "1873 is a prime congruent to 1 modulo 16" in seen and "congruent to 17 modulo 32" in seen
    assert (
        "order 7496 is not among its orders" in seen
        and 'ref-kline2026skew"' in (tmp_path / "orders/skew-7496.html").read_text()
    )
    assert (w % 16, w % 32, 7495, 3747, 7496) == (1, 17, 5 * 1499, 3 * 1249, 2**3 * 937)
    assert all(p % 32 == 17 for p in (1009, 1297, 1489, 1553, w))
    # Its first row read backwards gives the other solution of the pair, with the same other rows.
    twin = [a[0] + a[:0:-1], b, c, d]
    assert twin[0] != a and set(autocorrelation_sum(twin, shifts, w)) == {-4}


def test_tied_orders_above_8000_are_checked_by_their_rows(board, data, states, tmp_path, pages, monkeypatch):
    """Six orders whose matrix the build does not rebuild: the rows, the tie, the notes and the page wording."""
    import math

    from test_site import visible

    import skew

    records = bordered_records(data)
    for order, (w, g, t, first, second, examined) in TIED.items():
        x = records[order]
        a, b, c, d = x["rows"]
        odd = order // 8
        assert x["group"] == [w] and order == 4 * (w + 1) == 8 * odd and odd > 999 and odd % 2 == 1
        assert all(w % p for p in range(2, 73)) and (w % 16, w % 32, w % 48) == (1, 17, 1)
        assert math.prod(first) == order - 1 and math.prod(second) == order // 2 - 1 and len(first) > 1 < len(second)
        fixing = {pow(g, k, w) for k in range((w - 1) // 48)}
        assert x["multiplier"] == {"generator": [g], "order": (w - 1) // 48} and len(fixing) == (w - 1) // 48
        assert x["tie"] == {"multiplier": [t], "chains": [[2, 3, 4]], "fixed": [1]} and skew.tie_classes(x) == 8
        assert pow(t, 3, w) in fixing and t not in fixing and w - t not in fixing
        # c(x) = b(t^-1 x) and d(x) = b(t^-2 x); the first row is unchanged, the second is not.
        inverse = pow(t, -1, w)
        assert all(c[z] == b[inverse * z % w] and d[z] == b[inverse * inverse * z % w] for z in range(w))
        assert carried_by(a, t, w) == a and carried_by(b, t, w) == c != b and carried_by(c, t, w) == d
        # The multipliers and t generate the sixteenth powers, on whose classes the first row is constant.
        assert {h * pow(t, k, w) % w for h in fixing for k in range(3)} == {pow(z, 16, w) for z in range(1, w)}
        assert a[0] == "+" and all(a[w - z] != a[z] for z in range(1, w))
        assert [row.count("+") - row.count("-") for row in x["rows"]] == [1, 1, 1, 1]
        shifts = [1, 2, 3, 5, 7, 11, w - 1, w // 2]
        assert set(autocorrelation_sum(x["rows"], shifts, w)) == {-4}
        # Not a prime of the release cited for the method: neither one of its four nor one its test selects.
        assert w not in {1009, 1297, 1489, 1553, 17, 113, 881, 2129, 2801, 3217, 4721, 6577, 9521}
        seen = visible((pages / f"orders/skew-{order}.html").read_text())
        assert f"The rows are tied by the map σ: z ↦ {t}z of G." in seen and "there are 8 such sets" in seen
        assert f"the map z ↦ {g}z of G, which has order {(w - 1) // 48}. It has 48 orbits besides {{0}}." in seen
        assert f"it had gone through {examined} second rows" in seen and "is not exhaustive" in seen
        assert "It does not rebuild the matrix" in seen and "is the theorem of the sources cited for the array" in seen
        assert "one program rebuilt it from the rows and checked it exactly" in seen
        assert "The check is weaker than for the smaller orders on this board" in seen
        assert f"{order} = 2³ × {odd}" in seen and "stops at m = 999" in seen
        if odd < 2500:
            assert f"lists v = {odd} as undecided" in seen
        else:
            assert f"covers odd v below 2500 only and says nothing about v = {odd}" in seen
        assert f"{w} is a prime congruent to 1 modulo 16, to 17 modulo 32 and to 1 modulo 48" in seen
        assert f"so that release does not give order {order}" in seen
        assert "written from memory of the literature and not from the sources" in seen
        assert "This check is not a proof that no published construction applies." in seen
        assert f"The sha256 of that text is {x['sha256']}" in seen and "Download the matrix" not in seen
    assert [o // 8 for o in TIED][:7] == [1849, 2089, 2137, 2281, 2329, 2617, 2761]
    # The build checks the rows of such a record and writes them; it does not rebuild the matrix.
    monkeypatch.setattr(skew, "bordered_text", lambda *args: pytest.fail("the build rebuilt a matrix above order 8000"))
    (tmp_path / "matrices").mkdir()
    for order in (9672, 20936):
        site_build().write_above_matrix(tmp_path, records[order])
        assert (tmp_path / skew.rows_file_name(order)).read_text().split() == records[order]["rows"]
        assert not (tmp_path / skew.file_name(order)).exists()
    with pytest.raises(ValueError, match="do not add to -4"):
        rows = records[20936]["rows"]
        site_build().write_above_matrix(tmp_path, records[20936] | {"rows": [*rows[:3], rows[1]]})
    assert not skew.rebuilt_in_build(8001) and skew.rebuilt_in_build(7496)


# Orders above the range of the table of least known orders, with the prime q = n/4 - 1, the odd part,
# the factors of n - 1 and of n/2 - 1, and the number of solutions of the complete search.
ABOVE_TABLE = {
    9672: (2417, 1209, (19, 509), (5, 967), 768),
    10376: (2593, 1297, (5, 5, 5, 83), (3, 7, 13, 19), 128),
    10440: (2609, 1305, (11, 13, 73), (17, 307), 768),
    12360: (3089, 1545, (17, 727), (37, 167), 384),
}


def test_orders_above_the_table_say_how_weak_the_literature_check_is(board, data, states, tmp_path, pages):
    """The four orders whose odd part is above 999: the rows, the arithmetic of their notes and the page wording."""
    import math

    from test_site import visible

    import skew

    records = bordered_records(data)
    tmp_path = pages
    for order, (q, odd, first, second, solutions) in ABOVE_TABLE.items():
        x = records[order]
        assert x["group"] == [q] and order == 4 * (q + 1) == 8 * odd and odd > 999 and odd % 2 == 1
        assert all(q % p for p in range(2, 60)) and q % 16 == 1
        assert math.prod(first) == order - 1 and math.prod(second) == order // 2 - 1
        assert len(set(first)) > 1 and len(set(second)) > 1
        g, h = x["multiplier"]["generator"][0], x["multiplier"]["order"]
        fixing = {pow(g, k, q) for k in range(h)}
        assert len(fixing) == h and all(row[t * z % q] == row[z] for row in x["rows"] for t in (g,) for z in range(q))
        page = (tmp_path / f"orders/skew-{order}.html").read_text()
        seen = visible(page)
        if order == 10376:
            # 32 classes, and the first two rows carried to the other two by 918, whose square is -1.
            assert (q - 1) // h == 32 and q % 32 == 1 and 918 * 918 % q == q - 1
            assert x["tie"] == {"multiplier": [918], "chains": [[1, 3], [2, 4]]} and skew.tie_classes(x) == 8
            assert carried_by(x["rows"][0], 918, q) == x["rows"][2] and carried_by(x["rows"][1], 918, q) == x["rows"][3]
            assert "The rows are tied by the map σ: z ↦ 918z of G." in seen and "there are 8 such sets" in seen
            assert "all 65,536 rows of skew type and all 601,080,390 rows of sum 1" in seen
            assert "2593 is congruent to 1 modulo 32" in seen
        else:
            # The orbits are the 16 classes of sixteenth powers, and q is congruent to 17 modulo 32.
            assert fixing == {pow(z, 16, q) for z in range(1, q)} and (q - 1) // h == 16 and q % 32 == 17
            assert "tie" not in x and "tied by the map" not in seen
            assert f"the 16 classes of sixteenth powers in the field of {q} elements" in seen
            assert "256 rows, and every such row of sum 1, 12,870 rows" in seen
            assert f"{q} is such a prime; it is not one of those four" in seen
            assert "the method used here is the one it describes" in seen
        assert f"gives {solutions} solutions" in seen or f"It found {solutions} solutions" in seen
        assert "The check is weaker than for the smaller orders on this board" in seen
        assert f"{order} = 2³ × {odd}" in seen and "stops at m = 999" in seen
        assert "no table of known skew-Hadamard orders that we read covers this odd part at this order" in seen
        assert f"lists v = {odd} as undecided" in seen and f"nor its half {order // 2}" in seen
        assert f"{q} is a prime congruent to 1 modulo 16" in seen
        assert "A web search for the order found nothing." in seen
        assert "This check is not a proof that no published construction applies." in seen
        assert "The result has not been reviewed." in seen and "does not claim priority" in seen
        for key in ("djokovic2025census", "kline2026skew", "kishan2026skew", "cati2024database"):
            assert f'href="../references.html#ref-{key}"' in page, key
        assert not skew.has_download(order) and not skew.rebuilt_in_build(order)
        assert "It does not rebuild the matrix" in seen and "the build checks the rows" in seen
    # None of the primes is one that the release cited for the method treats or selects.
    assert not {q for q, *_ in ABOVE_TABLE.values()} & {1009, 1297, 1489, 1553, 17, 113, 881, 2129, 2801, 3217}
    assert "lists v = 937 as undecided" in visible((tmp_path / "orders/skew-7496.html").read_text())


def test_order_5392_is_the_bordered_eight_block_array_of_its_rows(data):
    """The found matrix of order 5392, rebuilt a second way from shift matrices, Kronecker products and the pattern."""
    import hashlib

    import numpy as np

    import skew

    x = above_records(data)[5392]
    rows, w = x["rows"], 673
    assert "file" not in x and x["origin"] == "found" and x["date"] == "2026-11-25"
    assert x["parameters"] == {"group_order": w, "group": "Z_673"} and x["sources"][:2] == [
        "wallis1972some",
        "kotsireas2008new",
    ]
    text = skew.stored_text(x, check=False)
    assert text == skew.found_text(x) and hashlib.sha256(text.encode()).hexdigest() == x["sha256"]
    h, names = eight_block_matrix(rows)
    assert h.shape == (5392, 5392) and as_text(h) == text.encode()
    # Entries are 1 and -1 and the order is below 2^24, so the float32 products are exact integers.
    h = h.astype(np.float32)
    identity = np.eye(5392, dtype=np.float32)
    assert np.array_equal(h + h.T, 2 * identity)
    assert np.array_equal(h @ h.T, 5392 * identity)
    del h, identity
    # What the record says of the rows, and the steps of the page's argument for this matrix.
    assert x["row_sums"] == [1] * 8 and set(autocorrelations(rows)) == {-8}
    assert rows[0][0] == "+" and all(rows[0][w - j] != rows[0][j] for j in range(1, w))
    assert all(r[w - j] == r[j] for r in rows[1:] for j in range(1, w))
    assert x["multiplier"] == {"generator": 208, "order": 21}
    assert pow(208, 21, w) == 1 and all(pow(208, k, w) != 1 for k in range(1, 21))
    assert all(r[208 * j % w] == r[j] for r in rows for j in range(w))
    blocks = [names[f"X{k}"].astype(np.float32) for k in range(1, 9)]
    assert np.array_equal(sum(m @ m.T for m in blocks), 5392 * np.eye(w, dtype=np.float32) - 8)
    assert all(np.array_equal(m @ n.T, n @ m.T) for m in blocks for n in blocks)
    assert all((m.sum(axis=1) == 1).all() for m in blocks)
    # 673 is prime, and the arithmetic of the note.
    assert all(w % p for p in range(2, 26)) and w % 16 == 1 and w == 2 * 337 - 1
    assert (5391, 2695, 1347, 5392) == (9 * 599, 5 * 7**2 * 11, 3 * 449, 2**4 * 337)


def test_page_of_order_5392_says_the_array_is_not_from_a_published_source(board, data, states, tmp_path, pages):
    from test_site import SLUGS, visible

    tmp_path = pages
    page = (tmp_path / "orders/skew-5392.html").read_text()
    seen = visible(page)
    assert "Order 5392 is above the range of the board, which has a row for every order up to 1200" in seen
    assert "Construction: Bordered eight-block array, group of order 673, Z_673" in seen
    assert (
        "We did not find this array in the sources we read, which is not a proof that none is published: "
        "it was worked out for this board from the two cited." in seen
    )
    assert "We did not find an eight-block bordered skew array in the sources we read" in seen
    assert "checked exactly at orders 48, 80, 96, 112, 496, 592 and 784" in seen
    assert "The statement that this matrix is skew-Hadamard rests on the exact check of the build." in seen
    assert "The argument that follows is ours and has not been reviewed." in seen
    assert "The matrix H of order 5392 = 8(673 + 1) has four parts." in seen
    assert "these eight products add to 5392I − 8J" in seen and "gives 8I + 8 × 673I = 5392I" in seen
    assert "the subgroup of order 21 generated by 208" in seen and "It has 32 orbits besides {0}." in seen
    assert "−X8 −X7  X6  X5 −X4 −X3  X2  X1</pre>" in page and "  Sᵀ       S ⊗ eᵀ\n −Sᵀ ⊗ e   P" in page
    assert all(line in page for line in above_records(data)[5392]["rows"])
    assert "The repository holds the eight rows and the sha256 of the matrix file, not the file" in seen
    assert "The result has not been reviewed." in seen and "does not claim priority" in seen
    assert "which gives t = 18 for m = 337, while 5392 = 2⁴ × 337" in seen
    assert "5391 = 9 × 599, 2695 = 5 × 7² × 11 and 1347 = 3 × 449 are not prime powers" in seen
    assert "is not a proof that no published construction applies" in seen
    assert 'href="../matrices/skew-5392.txt" download' in page
    for key in (
        "wallis1972some",
        "kotsireas2008new",
        "cati2024database",
        "kishan2026skew",
        "momihara2019constructions",
    ):
        assert f'href="../references.html#ref-{key}"' in page, key
    assert not re.search(SLUGS, seen) and "—" not in seen and "[@" not in seen and "to our knowledge" not in seen


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
    """An order none of the cited sources gives a matrix of and that now has one says so, with the date."""
    t = data["types"]["skew-hadamard"]
    filled = {o for record in t["since"] for o in record["orders"]}
    assert {356, 476, 712, 952, 820, 996} <= filled
    for n in range(1, 301):
        skew_cell = orders.grid_cells(n, states).split("<td")[2]
        assert 'class="since"' not in skew_cell and "filled since" not in skew_cell
        if 4 * n in filled:
            date = next(record["date"] for record in t["since"] if 4 * n in record["orders"])
            assert (
                f"none of the cited sources gives a matrix of this order, and this one dates from {date}" in skew_cell
            )
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
        assert ("This one dates from 20" in block) == (order in filled), order
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
    of = lambda m, order: next(x for x in m if x["order"] == order)  # noqa: E731
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
        lambda m: of(m, 5336).update(file="matrices/skew/skew-5336.txt.xz"),
        lambda m: of(m, 5336).update(origin="repository", software="kishan2026skew"),
        lambda m: of(m, 5336).pop("group"),
        lambda m: of(m, 5336).pop("array"),
        lambda m: of(m, 5336).update(array="other"),
        lambda m: of(m, 5336).update(order=5338),
        lambda m: of(m, 5336).update(order=1416),
        lambda m: of(m, 5392).pop("derivation"),
        lambda m: of(m, 5392).update(file="matrices/skew/skew-5392.txt.xz"),
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
    import hashlib

    import numpy as np

    import skew

    module = site_build()
    # The found matrices above the range are left out here: the build rebuilds and checks each, the tests
    # of their arrays rebuild them a second way, and one of them is written below.
    chosen = module.write_skew_matrices(tmp_path, states, board[1], above=False)
    assert chosen == skew.chosen_entries(states)
    is_open = set(data["types"]["skew-hadamard"]["open"])
    written = {int(p.stem.removeprefix("skew-")) for p in (tmp_path / "matrices").iterdir()}
    above = above_records(data)
    assert not written & set(above)
    # A matrix above the range is written from its record: the file while its text is at most 50 MB,
    # and above that the rows of the record.
    module.write_above_matrix(tmp_path, above[1416])
    text = (tmp_path / skew.file_name(1416)).read_bytes()
    assert hashlib.sha256(text).hexdigest() == above[1416]["sha256"] and text.count(b"\n") == 1416
    (tmp_path / skew.file_name(1416)).unlink()
    module.write_above_matrix(tmp_path, above[7496], quick=True)
    assert not (tmp_path / skew.file_name(7496)).exists()
    assert (tmp_path / skew.rows_file_name(7496)).read_text().split() == above[7496]["rows"]
    (tmp_path / skew.rows_file_name(7496)).unlink()
    assert {o for o in above if skew.has_download(o)} == {1416, 1712, 2360, 2696, 3352, 5336, 5392}
    assert skew.has_download(7070) and not skew.has_download(7072) and skew.rebuilt_in_build(7496)
    with pytest.raises(ValueError, match="does not match its sha256"):
        module.write_above_matrix(tmp_path, above[1416] | {"sha256": "0" * 64}, quick=True)
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


def test_order_pages_cover_the_rows_without_a_board_cell(board, data, states, tmp_path, pages):
    import skew

    tmp_path = pages
    written = {int(p.stem.removeprefix("skew-")) for p in (tmp_path / "orders").iterdir()}
    assert written == {4 * n for n in range(2, 301, 2)} | set(BORDERED) | {3352, 5392} and len(written) == 152 + len(
        BORDERED
    )
    is_open = set(data["types"]["skew-hadamard"]["open"])
    for order in sorted(written):
        page = (tmp_path / f"orders/skew-{order}.html").read_text()
        assert f"<h1>Skew-Hadamard matrix of order {order}</h1>" in page and "<h2>References</h2>" in page
        offered = order not in is_open and skew.has_download(order)
        assert (f'href="../matrices/skew-{order}.txt" download' in page) == offered, order
        assert (f'href="../matrices/skew-{order}-rows.txt" download' in page) == (not skew.has_download(order)), order
        assert 'href="../style.css"' in page and "—" not in page
    # The pages of the covered rows are the board cells, which carry the same block.
    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (85, "skew-sds"))
    block, _ = skew.matrix_block(data, 340, states[340]["skew-hadamard"], module.STORED, {}, Bibliography())
    assert block in module.cell_page(cell, d, r, block) and "Matrix of order 340" in block
    assert "Matrix of order" not in module.cell_page(cell, d, r)


def test_pages_of_orders_above_the_range_describe_the_bordered_array(board, data, states, tmp_path, pages):
    from test_site import SLUGS, visible

    import skew

    tmp_path = pages
    records = bordered_records(data)
    for order, (moduli, _, size, orbits) in BORDERED.items():
        page = (tmp_path / f"orders/skew-{order}.html").read_text()
        text = visible(page)
        w = order // 4 - 1
        assert f"<h1>Skew-Hadamard matrix of order {order}</h1>" in page
        assert f"Order {order} is above the range of the board, which has a row for every order up to 1200" in text
        assert "needs odd n" not in text and '<span class="tag">verified</span>' in page
        assert f"Construction: Wallis–Whiteman bordered array, group of order {w}, Z_" in text
        assert f"found on {records[order]['date']}" in text and "with a computer search" in text
        assert "not a Goethals–Seidel quadruple, and the order is above the board's range" in text
        assert f"The matrix H of order {order} = 4({w} + 1) is the block matrix" in text
        assert "  1  −1  −1  −1  −e  −e  −e  −e\n" in page and " eᵀ  eᵀ  eᵀ −eᵀ −Bᵀ  Aᵀ  −D   C\n" in page
        assert "A, B and D be the type 1 matrices of the first, second and fourth rows" in text
        assert "C be the type 2 matrix of the third" in text
        assert f"each subset have {(w - 1) // 2} elements" in text and f"in {w - 3} ways" in text
        assert "it is −1 times the matrix printed in" in text and "state as a theorem" in text
        assert "every row has sum 1" in text and "add to −4 at every nonzero s" in text
        assert f"which has order {size}" in text and f"It has {orbits} orbits besides {{0}}." in text
        assert all(line in page for line in records[order]["rows"])
        assert "The repository holds the four rows and the sha256 of the matrix file, not the file" in text
        assert "The result has not been reviewed." in text and "does not claim priority" in text
        day = "2026-11-26" if order > 9000 else "2026-11-25"
        assert f"on {day} found no earlier skew-Hadamard matrix of order {order}" in text
        assert (f'href="../matrices/skew-{order}.txt" download' in page) == skew.has_download(order)
        assert "<h2>References</h2>" in page
        if not skew.has_download(order):
            megabytes = round(order * (order + 1) / 1e6, 1)
            assert f"The matrix file would be {megabytes} MB of text, so the site does not offer it." in text
            assert f'href="../matrices/skew-{order}-rows.txt" download>Download the rows</a>' in page
            assert f"The sha256 of that text is {records[order]['sha256']}" in text
            assert "Download the matrix" not in text
        for key in ("wallis1972some", "momihara2019constructions", "cati2024database", "kishan2026skew"):
            assert f'href="../references.html#ref-{key}"' in page, key
        # The survey was among the sources read for the first four orders.
        assert ('href="../references.html#ref-egan2026recent"' in page) == (order < 2696 or order == 5336)
        # Plain language: no schema value, no marker left unresolved, no priority wording.
        assert not re.search(SLUGS, text), re.search(SLUGS, text).group(0)
        assert "—" not in text and "[@" not in text and "to our knowledge" not in text
        assert "first skew" not in text.lower() and "new record" not in text.lower()
        if len(moduli) == 1:
            assert f"G is the group of the integers modulo {w} under addition" in text
            assert "They are the circulant and the back-circulant matrix of x." in text
            assert "not circulant" not in text
        else:
            m1, m2 = moduli
            assert f"Here G is the group Z {m1} × Z {m2} :" in text
            assert f"Since {m1} and {m2} have no common factor, G is a cyclic group of order {w}" in text
            assert "they are not circulant matrices" in text
            assert f"consists of {m1} × {m1} blocks of order {m2}, each of them circulant" in text
    page = visible((tmp_path / "orders/skew-1712.html").read_text())
    assert "(4x 1 , 12x 2 ) of G, which has order 15 and takes k(1, 1) to 256k(1, 1)" in page
    assert 256 % 7 == 4 and 256 % 61 == 12
    assert "which gives t = 10 for m = 107, while 1712 = 2⁴ × 107" in page
    page = visible((tmp_path / "orders/skew-1416.html").read_text())
    assert "the map z ↦ 231z of G, which has order 11. It has 32 orbits besides {0}." in page
    # The page of a grid order keeps its own sentence.
    assert "a skew circulant block needs odd n" in (tmp_path / "orders/skew-808.html").read_text()
