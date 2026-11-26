"""What the site generator draws and words, checked against the verified board."""

import pytest

from verify.board import load, summary
from verify.common import ROOT


@pytest.fixture(scope="module")
def board():
    return load(reproduce=False)


def gs_array(rows):
    """The block layout drawn by site/picture.js, as in tests/test_board.py."""
    import numpy as np

    n = len(rows[0])
    circ = lambda x: np.array([[x[(j - i) % n] for j in range(n)] for i in range(n)])  # noqa: E731
    a, b, c, d = (circ(r) for r in rows)
    xr = lambda m: m[:, ::-1]  # noqa: E731
    rx = lambda m: m[::-1, :]  # noqa: E731
    return np.block(
        [
            [a, xr(b), xr(c), xr(d)],
            [-xr(b), a, -rx(d), rx(c)],
            [-xr(c), rx(d), a, -rx(b)],
            [-xr(d), -rx(c), rx(b), a],
        ]
    )


def site_build():
    import importlib.util

    spec = importlib.util.spec_from_file_location("site_build", ROOT / "site/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_displayed_array_of_every_quadruple_with_a_skew_block_is_skew_hadamard(board):
    import numpy as np

    display_rows = site_build().display_rows
    checked = moved = 0
    for ident, x in board[1].items():
        if x["type"] != "quadruple" or "skew-sds" not in x["tracks"]:
            continue
        rows, pattern = display_rows(x)
        assert pattern[0] == "k", ident
        assert sorted(rows) == sorted(x["canonical_rows"]), ident
        assert sorted(pattern) == sorted(x["pattern"]), ident
        h = gs_array(rows)
        m = len(h)
        assert np.array_equal(h + h.T, 2 * np.eye(m, dtype=int)), ident
        assert np.array_equal(h @ h.T, m * np.eye(m, dtype=int)), ident
        checked += 1
        moved += pattern != x["pattern"]
    # The canonical order puts a row without the skew symmetry first in many
    # entries, among them n89-eec2e43c042090883c04.
    assert checked > 500 and moved > 100
    rows, pattern = display_rows(board[1]["n89-eec2e43c042090883c04"])
    assert board[1]["n89-eec2e43c042090883c04"]["pattern"][0] != "k" and pattern[0] == "k"


def test_display_order_leaves_quadruples_without_a_skew_block_alone(board):
    display_rows = site_build().display_rows
    for x in board[1].values():
        if x["type"] == "quadruple" and "k" not in x["pattern"]:
            assert display_rows(x) == (list(x["canonical_rows"]), x["pattern"])


def test_existence_results_appear_only_when_featured(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    entries = ["n89-b2a223b0fb7500feb95f", "n89-eec2e43c042090883c04"]
    stats["contributions"] = [g for g in stats["contributions"] if g["id"] != "skew-pool-join-89"]
    group = {"id": "skew-pool-join-89", "featured": False, "entry_ids": entries}
    stats["contributions"].append(group)
    assert not [x for x in module.existence_results(stats, r) if x["kind"] == "skew"]
    assert ("skew-sds", 89) not in module.new_cells(stats, r)
    group["featured"] = True
    (item,) = (x for x in dict(module.new_results(stats, r))["Existence"] if x["kind"] == "skew")
    assert item["text"] == "A skew-Hadamard matrix of order 356."
    assert item["double"] == 712
    assert item["qualifier"] is None
    assert ("skew-sds", 89) in module.new_cells(stats, r)
    html = module.new_results_section(stats, r)
    assert "cells/n89-skew-sds.html" in html
    assert "first" not in html.lower()


def test_found_matrix_is_listed_among_the_existence_results(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    found = [x for x in module.existence_results(stats, r) if x["kind"] == "matrix"]
    assert [x["text"] for x in found] == [
        f"A skew-Hadamard matrix of order {order}."
        for order in sorted(o for o, x in module.STORED.items() if x["origin"] == "found")
    ]
    listed = [x["href"] for x in found]
    assert listed[:14] == [
        f"orders/skew-{order}.html"
        for order in (808, 856, 1416, 1712, 2360, 2696, 3352, 5336, 5392, 7496, 9672, 10376, 10440, 12360)
    ]
    assert {f"orders/skew-{order}.html" for order in (14792, 16712, 17096, 18248, 18632, 20936, 22088)} <= set(listed)
    item = found[0]
    assert item["qualifier"] is None
    assert (item["n"], item["href"]) == (202, "orders/skew-808.html")
    html = module.new_results_section(stats, r)
    assert '<a href="orders/skew-808.html">A skew-Hadamard matrix of order 808.</a>' in html
    assert "first" not in html.lower() and "1616" not in html
    # Every found order has a line of its own, and the skew-Hadamard lines stand together before the others.
    assert '<a href="orders/skew-5392.html">A skew-Hadamard matrix of order 5392.</a>' in html
    kinds = [x["kind"] for x in module.existence_results(stats, r)]
    symmetric = [i for i, kind in enumerate(kinds) if kind in module.SYMMETRIC_KINDS]
    assert symmetric == list(range(len(kinds) - len(symmetric), len(kinds)))


def test_propus_result_is_worded_and_linked_without_marking_the_grid(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    (group,) = (g for g in stats["contributions"] if g["id"] == "propus-pool-join-65")
    (ident,) = group["entry_ids"]
    (item,) = (x for x in module.existence_results(stats, r) if x["kind"] == "propus")
    assert item["text"] == "A symmetric Hadamard matrix of order 260."
    html = module.new_results_section(stats, r)
    assert f'<a href="entries/{ident}.html">{item["text"]}</a>' in html
    assert "first" not in html.lower()
    # Neither the Hadamard nor the skew-Hadamard order 260 is new, so no cell is marked.
    assert not any(n == 65 for _, n in module.new_cells(stats, r))
    assert module.propus_entries(stats) == {ident}
    stats["contributions"] = [g | {"featured": False} for g in stats["contributions"]]
    # A stored matrix found for this board belongs to no contribution, so it has no featured switch.
    assert {x["kind"] for x in module.existence_results(stats, r)} == {"matrix"}
    assert module.propus_entries(stats) == {ident}


def test_propus_array_is_symmetric_hadamard_and_the_goethals_seidel_array_is_not(board):
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / "site"))
    import propus
    from references import Bibliography

    module = site_build()
    rows, pattern = module.display_rows(board[1]["n65-9bf801f0de9e2d01c459"])
    s, e1, e2, o = propus.shape(rows, pattern)
    assert pattern[s] == "s" and rows[e1] == rows[e2] and rows[e1] != rows[o]
    n = len(rows[0])
    circ = lambda x: np.array([[x[(j - i) % n] for j in range(n)] for i in range(n)])  # noqa: E731
    x, y, z = circ(rows[s]), circ(rows[e1]), circ(rows[o])
    back = np.fliplr(np.eye(n, dtype=int))
    yr, zr, ytr, ztr = y @ back, z @ back, y.T @ back, z.T @ back
    # The array for commuting blocks with X symmetric, as printed on the entry page.
    h = np.block([[x, yr, yr, zr], [yr, ztr, -x, -ytr], [yr, -x, -ztr, ytr], [zr, -ytr, ytr, -x]])
    assert np.array_equal(h, np.array(propus.array(rows[s], rows[e1], rows[o])))
    assert np.array_equal(h, h.T) and np.array_equal(h @ h.T, 4 * n * np.eye(4 * n, dtype=int))
    # The other published form: the Goethals–Seidel array with its first column
    # negated and its second and third block rows exchanged.
    g = gs_array([rows[s], rows[e1], rows[e2], rows[o]])
    assert np.array_equal(g @ g.T, 4 * n * np.eye(4 * n, dtype=int)) and not np.array_equal(g, g.T)
    p = np.vstack([g[:n], g[2 * n : 3 * n], g[n : 2 * n], g[3 * n :]])
    p[:, :n] *= -1
    assert np.array_equal(p, p.T) and np.array_equal(p @ p.T, 4 * n * np.eye(4 * n, dtype=int))
    text = propus.matrix_file(rows, pattern)
    assert text.count("\n") == 4 * n and text.splitlines()[0] == "".join("+" if v > 0 else "-" for v in h[0])
    with pytest.raises(ValueError, match="transpose"):
        propus.check_symmetric_hadamard(g.tolist())
    with pytest.raises(ValueError, match="orthogonal"):
        propus.check_symmetric_hadamard([[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="not a propus quadruple"):
        propus.matrix_file(rows, "····")
    page, keys = propus.block("n65-9bf801f0de9e2d01c459", rows, pattern, Bibliography())
    assert "row A is symmetric and rows B and C are equal" in page and "not symmetric" in page
    assert "matrices/propus-n65-9bf801f0de9e2d01c459.txt" in page and keys == set(propus.ARRAY_SOURCES)


# The quadruples whose rows are tied by a multiplier s with s² = 1, by matrix order: the entry, s, the two
# coprime factors of n on which s is 1 and −1, and the number of multipliers that fix all four rows.
TWISTED = {
    988: ("n247-308b784a56a0a484fe03", 170, (13, 19), 9),
    1036: ("n259-89a912c3ec591210d016", 36, (7, 37), 9),
    1068: ("n267-53011cacf3f3fb5b6b64", 179, (89, 3), 11),
}


def test_twisted_results_are_worded_linked_and_placed_after_the_skew_lines(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    items = module.existence_results(stats, r)
    kinds = [x["kind"] for x in items]
    # The skew-Hadamard lines stand together first; the symmetric lines follow in order of the matrix order.
    first = min(i for i, kind in enumerate(kinds) if kind in module.SYMMETRIC_KINDS)
    assert all(kind in module.SYMMETRIC_KINDS for kind in kinds[first:])
    assert [x["text"] for x in items[first:]] == [
        "A symmetric Hadamard matrix of order 260.",
        "A symmetric Hadamard matrix of order 988.",
        "A symmetric Hadamard matrix of order 1036.",
        "A symmetric Hadamard matrix of order 1068.",
    ]
    html = module.new_results_section(stats, r)
    for order, (ident, *_) in TWISTED.items():
        assert f'<a href="entries/{ident}.html">A symmetric Hadamard matrix of order {order}.</a>' in html
        # Neither the Hadamard nor the skew-Hadamard order is new, so no cell is marked.
        assert not any(n == order // 4 for _, n in module.new_cells(stats, r))
    assert "first" not in html.lower()
    assert module.symmetric_entries(stats, "twisted") == {ident for ident, *_ in TWISTED.values()}
    assert module.propus_entries(stats) == {"n65-9bf801f0de9e2d01c459"}


def tied_solutions(n, s):
    """Every (w, y, z) of rows of 1 and −1 of length n with w unchanged by k ↦ −sk, z unchanged by k ↦ sk and
    the periodic autocorrelations of w, y, k ↦ y(sk) and z adding to zero at every nonzero shift.

    The rows y are found by a table of the autocorrelation of every row, not by the site code.
    """
    import itertools

    import numpy as np

    lags = range(1, n // 2 + 1)

    def paf(m):
        return np.stack([(m * np.roll(m, -r, axis=1)).sum(axis=1) for r in range(n)], axis=1)

    def code(q):
        # An autocorrelation sum lies between −4n and 4n: one digit in base 8n + 1 for each shift.
        return sum((q[:, r] + 4 * n) * (8 * n + 1) ** i for i, r in enumerate(lags))

    every = np.array(list(itertools.product((1, -1), repeat=n)), dtype=np.int64)
    fixed = lambda t: every[(every[:, [t * k % n for k in range(n)]] == every).all(axis=1)]  # noqa: E731
    p = paf(every)
    # The autocorrelation of k ↦ y(sk) at r is that of y at sr.
    pair = code(p + p[:, [s * r % n for r in range(n)]])
    order = np.argsort(pair, kind="stable")
    keys = pair[order]
    ws, zs = fixed(-s % n), fixed(s % n)
    pw, pz = paf(ws), paf(zs)
    for i, w in enumerate(ws):
        want = code(-(pw[i] + pz))
        lo, hi = np.searchsorted(keys, want, "left"), np.searchsorted(keys, want, "right")
        for j in np.nonzero(hi > lo)[0]:
            for k in order[lo[j] : hi[j]]:
                yield w.tolist(), every[k].tolist(), zs[j].tolist()


def test_twisted_array_is_symmetric_hadamard_for_every_choice_of_rows_at_small_lengths():
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / "site"))
    import propus
    import twisted

    # Lengths 3 and 5 have only s = 1 and s = −1, which the page never uses and the array still covers:
    # s = 1 is the propus case. Every choice of rows is checked.
    for n, s, count in ((3, 1, 96), (3, 2, 96), (5, 1, 400), (5, 4, 400)):
        found = list(tied_solutions(n, s))
        assert len(found) == count
        for w, y, z in found:
            propus.check_symmetric_hadamard(twisted.array(w, y, z, s))
    # Length 15 is the smallest with an s other than 1 and −1: s = 4, which is 1 modulo 3 and −1 modulo 5.
    # The search that found the board's matrices counted 2,649,600 choices of rows there, and 691,200 for
    # s = 1; this enumeration shares no code with it. Every 5000th choice is checked.
    assert twisted.involutions(15) == [4, 11] and twisted.involutions(13) == []
    for s, count in ((4, 2649600), (1, 691200)):
        total = 0
        for w, y, z in tied_solutions(15, s):
            if total % 5000 == 0:
                h = twisted.array(w, y, z, s)
                propus.check_symmetric_hadamard(h)
                if s == 4:
                    c = twisted.carried(y, s)
                    assert twisted.shape([w, y, c, z]) is not None
                    # A shift of w keeps its autocorrelation and loses the symmetry of the matrix, unless
                    # the shifted row happens to be unchanged by k ↦ −sk as well.
                    shifted = w[1:] + w[:1]
                    g = twisted.array(shifted, y, z, s)
                    assert np.array_equal(np.array(g) @ np.array(g).T, 60 * np.eye(60, dtype=int))
                    if twisted.carried(shifted, -s) != shifted:
                        with pytest.raises(ValueError, match="transpose"):
                            propus.check_symmetric_hadamard(g)
            total += 1
        assert total == count


def test_twisted_array_of_the_board_entries_rebuilt_a_second_way(board):
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / "site"))
    import twisted
    from references import Bibliography

    module = site_build()
    for order, (ident, s, (m1, m2), multipliers) in TWISTED.items():
        rows, _ = module.display_rows(board[1][ident])
        n = len(rows[0])
        assert 4 * n == order and n == m1 * m2
        found = twisted.shape(rows)
        assert found is not None and found[0] == s and sorted(found[1:]) == [0, 1, 2, 3]
        assert s * s % n == 1 and s % m1 == 1 and s % m2 == m2 - 1
        w, y, c, z = (np.array(rows[i]) for i in found[1:])
        k = np.arange(n)
        assert np.array_equal(w[-s * k % n], w) and np.array_equal(z[s * k % n], z) and np.array_equal(y[s * k % n], c)
        assert not np.array_equal(y, c)
        group = twisted.multipliers(rows)
        assert len(group) == multipliers and all(np.array_equal(x[t * k % n], x) for t in group for x in (w, y, c, z))
        # The second way: circulant matrices and the two permutation matrices of the page's argument.
        eye = np.eye(n, dtype=np.int64)
        big_s, big_n = eye[:, s * k % n], eye[:, -k % n]
        assert big_s[s * 5 % n, 5] == 1 and big_n[-5 % n, 5] == 1
        for m in (big_s, big_n):
            assert np.array_equal(m, m.T) and np.array_equal(m @ m, eye)
        assert np.array_equal(big_s @ big_n, big_n @ big_s)
        shift = (k[None, :] - k[:, None]) % n
        cw, cy, cc, cz = w[shift], y[shift], c[shift], z[shift]
        for x in (cw, cy, cc, cz):
            assert np.array_equal(big_n @ x, x.T @ big_n)
        assert np.array_equal(big_s @ cy @ big_s, cc) and np.array_equal(big_s @ cz @ big_s, cz)
        assert np.array_equal(big_s @ cw @ big_s, cw.T)
        nn = big_n
        q = np.block(
            [
                [-cw, cy @ nn, cc @ nn, cz @ nn],
                [cc @ nn, cz.T @ nn, cw, -cy.T @ nn],
                [cy @ nn, cw, -cz.T @ nn, cc.T @ nn],
                [cz @ nn, -cc.T @ nn, cy.T @ nn, cw],
            ]
        )
        scale = order * np.eye(order, dtype=np.int64)
        assert np.array_equal(q @ q.T, scale)
        h = q @ np.kron(np.eye(4, dtype=np.int64), big_s)
        assert np.array_equal(h, np.array(twisted.array(rows[found[1]], rows[found[2]], rows[found[4]], s)))
        assert np.array_equal(h, h.T) and np.array_equal(h @ h.T, scale) and set(np.unique(h)) == {-1, 1}
        # The entries of the blocks as the page defines them.
        i, j = 3, 7
        assert h[i, j] == -w[(s * j - i) % n] and h[i, n + j] == y[(-s * j - i) % n]
        assert h[n + i, n + j] == z[(s * j + i) % n]
        text = twisted.matrix_file(rows)
        assert text.count("\n") == order and text.splitlines()[0] == "".join("+" if v > 0 else "-" for v in h[0])
        g = gs_array(rows)
        assert np.array_equal(g @ g.T, scale) and not np.array_equal(g, g.T)
        page, keys = twisted.block(ident, rows, Bibliography())
        names = ["ABCD"[p] for p in found[1:]]
        assert f"Symmetric Hadamard matrix of order {order}</h2>" in page and f"let s = {s}," in page
        assert f"Row {names[0]} is unchanged by k ↦ −sk" in page and f"row {names[3]} is unchanged by k ↦ sk" in page
        assert f"{names[2]}(k) = {names[1]}(sk)" in page
        assert f"the group Z<sub>{m1}</sub> × Z<sub>{m2}</sub> written with one number in place of two" in page
        assert f"which has order {multipliers} modulo {n}" in page
        assert "is not taken from a published source" in page and "ours and has not been reviewed" in page
        assert "rests on the exact check of the build" in page and "first" not in page.replace("first two rows", "")
        assert f"matrices/symmetric-{ident}.txt" in page and keys == {*twisted.ARRAY_SOURCES, twisted.TWIST_BEFORE}
        assert "The nearest earlier work found in the sources read is in the notes of a repository" in page
        assert "conclude that they give nothing beyond the propus array" in page
    with pytest.raises(ValueError, match="not tied by a multiplier"):
        twisted.matrix_file(module.display_rows(board[1]["n65-9bf801f0de9e2d01c459"])[0])


def test_restricted_certificate_is_not_labelled_as_a_statement_about_the_cell(board):
    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (89, "skew-sds"))
    page = module.cell_page(cell, d, r)
    assert "proved empty within a restricted family, not for the whole cell" in page
    assert "skew-Hadamard matrix of order 356" in page
    assert "rows verified" in page and "reproduced" not in page.split("<h2>")[0]
    assert "skew-Hadamard matrix of order 356" in module.cell_description(cell)


def test_cell_with_a_registered_replay_never_says_it_was_not_replayed(board):
    import re

    from verify.common import read_json

    module = site_build()
    d, r, b = board
    replays = read_json(ROOT / "baselines/census_replays.json")["reports"]
    replayed = {read_json(ROOT / x["path"])["entry_id"] for x in replays}
    pages = 0
    for cell in summary(d, r, b)["cells"]:
        if not replayed & set(cell["censuses"]):
            continue
        pages += 1
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", module.cell_page(cell, d, r)))
        assert not re.search(
            r"not (been )?(replayed|rerun)|awaiting a full replay",
            text.split("Search and evidence records")[0].split(' class="credit"')[0],
        ), (cell["n"], cell["stratum"])
    assert pages > 0


def test_cell_page_states_a_published_count_beside_the_board_count(board):
    module = site_build()
    d, r, b = board
    by_cell = {(c["n"], c["stratum"]): c for c in summary(d, r, b)["cells"]}
    page = module.cell_page(by_cell[18, "williamson"], d, r)
    assert "Published count: <strong>23 classes</strong>" in page and "half-period shift of a block" in page
    assert 'href="../references.html#ref-bright2020williamson"' in page
    assert "<h2>540 quadruples</h2>" in page and 'href="n18-williamson.txt"' in page
    # A cited construction gives existence and no count.
    assert "Published count" not in module.cell_page(by_cell[128, "williamson"], d, r)
    assert "classes are published" not in module.cell_page(by_cell[128, "williamson"], d, r)
    # An open cell says what open means here.
    open_page = module.cell_page(by_cell[94, "williamson"], d, r)
    assert "is known to us in the sources the board cites" in open_page and "Published count" not in open_page
    assert "is known to us" not in page


def visible(page):
    """The text a reader sees or hears: text nodes, titles and labels, without scripts and attribute values."""
    import html
    import re

    page = re.sub(r"<script.*?</script>", " ", page, flags=re.S)
    spoken = re.findall(r'(?:title|aria-label)="([^"]*)"', page)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page) + " | " + " | ".join(spoken)))


# Schema values and internal names that must not reach a reader. Entry ids are link text and may contain a
# track name, and (kkss) is the symmetry pattern in a column name, so those are matched as whole words only
# where they cannot be either.
SLUGS = (
    r"published-not-imported|verified-on-board|equiv-v\d|branch-v1|E1 to E5|receipt-backed|spot-checked|"
    r"unreplayed|computed-exact|published-exact|participant-result|published-import|stratum|strata|"
    r"\breproduced\b|\bunresolved\b|\bproven\b"
)


def test_cell_pages_and_tooltips_use_no_schema_slug(board):
    import re

    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    for cell in stats["cells"]:
        page = module.cell_page(cell, d, r)
        text = visible(page)
        assert not re.search(SLUGS, text), (cell["n"], cell["stratum"], re.search(SLUGS, text).group(0))
        assert "—" not in text
        # The slugs stay available to scripts and styles as data attributes.
        assert f'data-track="{cell["stratum"]}"' in page and f'data-status="{cell["status"]}"' in page
    _, rows, _ = module.grid(stats, r)
    tips = " | ".join(re.findall(r'title="([^"]*)"', rows))
    assert not re.search(SLUGS, tips), re.search(SLUGS, tips).group(0)


def test_cell_page_opens_with_one_plain_sentence(board):
    import re

    module = site_build()
    d, r, b = board
    by_cell = {(c["n"], c["stratum"]): c for c in summary(d, r, b)["cells"]}

    def bare(n, track):
        """The cell as it reads while the board holds none of its quadruples, whatever it holds today."""
        return by_cell[n, track] | {
            "quadruples": [],
            "censuses": [],
            "catalogues": [],
            "artifact_status": "published-not-imported",
        }

    def lead(n, track, cell=None):
        return re.sub(r"<[^>]+>", "", module.cell_lead(cell or by_cell[n, track], r))

    assert lead(28, "williamson", bare(28, "williamson")).startswith(
        "Williamson quadruples of length 28 exist: 83 classes are published ["
    )
    assert lead(128, "williamson", bare(128, "williamson")).startswith(
        "Williamson quadruples of length 128 exist by a published result ["
    )
    assert lead(83, "good").startswith("Open: no quadruple of good matrices of length 83 is known to us")
    assert lead(47, "best") == (
        "No quadruple of best matrices of length 47 exists: best matrices need 4n − 3 to be a square, "
        "and 185 is not one."
    )
    # The module's board skips the inline replays, and the sentence says so instead of claiming a rerun.
    assert lead(18, "williamson") == (
        "The board holds 540 Williamson quadruples of length 18, one per class of a complete search that this "
        "build did not rerun. Each gives a Hadamard matrix of order 72."
    )
    # A registered replay holds in every build.
    assert lead(61, "williamson") == (
        "The board holds 1 Williamson quadruple of length 61, the only class. A complete search rerun here "
        "found no others. It gives a Hadamard matrix of order 244."
    )
    assert lead(89, "skew-sds").startswith(
        "The board holds 11 verified Goethals–Seidel quadruples of length 89 with a skew block, no two in the "
        "same class. No complete search is recorded here."
    )
    assert module.cell_tag(bare(28, "williamson"), r) == "exists · published, none here yet"
    assert module.cell_tag(by_cell[83, "good"], r) == "open"
    assert module.cell_tag(by_cell[47, "best"], r) == "none exists · by arithmetic"
    # A published count of Williamson quadruples is not a count of the cell for any quadruple.
    assert "83 classes" not in lead(28, "general", bare(28, "general"))


def test_cell_text_file_lists_the_rows_of_every_quadruple(board):
    from verify.gs_verify import validate_rows

    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (18, "williamson"))
    lines = module.cell_rows_text(cell, r).splitlines()
    assert len(lines) == 540
    for line, ident in zip(lines, cell["quadruples"], strict=True):
        rows = [[1 if x == "+" else -1 for x in word] for word in line.split()]
        assert validate_rows(rows, 18) == tuple(map(tuple, module.display_rows(r[ident])[0]))


def catalogue_cells(module, tmp_path, records, split):
    """Board cells at n = 10 whose Williamson quadruples are in a catalogue file under tmp_path."""
    import gzip
    import hashlib

    from verify import catalogue
    from verify.board import cells
    from verify.census_engine import williamson_even
    from verify.equivalence import canonical_half_shifts

    classes = sorted(williamson_even(10))
    if not split:
        classes = sorted({canonical_half_shifts(q): q for q in classes}.values())
    packed = gzip.compress("".join(catalogue.format_line(q) + "\n" for q in classes).encode(), mtime=0)
    (tmp_path / "evidence/catalogues").mkdir(parents=True, exist_ok=True)
    (tmp_path / "evidence/catalogues/williamson-n10.txt.gz").write_bytes(packed)
    entry = {
        "id": "williamson-n10",
        "type": "census",
        "n": 10,
        "stratum": "williamson",
        "equivalence": "equiv-v1",
        "representatives": [],
        "claimed_count": len(classes),
        "claim_kind": "lower-bound",
        "catalogue": {
            "path": "evidence/catalogues/williamson-n10.txt.gz",
            "sha256": hashlib.sha256(packed).hexdigest(),
        },
        "published_classes": {"count": 2, "split": split},
        "sources": [{"title": "The dataset of the enumeration", "url": "https://doi.org/10.5281/zenodo.1195693"}],
    }
    d = {entry["id"]: entry}
    r = {entry["id"]: catalogue.verify(entry, {}, tmp_path, True) | {"contributions": []}}
    module.ROOT = tmp_path
    return d, r, {(c["n"], c["stratum"]): c for c in cells(d, r, records, max_n=10)}


def test_catalogue_cell_page_says_what_the_file_holds(board, tmp_path):
    import re

    module = site_build()
    d, r, grid = catalogue_cells(module, tmp_path, board[2], split=True)
    cell = grid[10, "williamson"]
    page = module.cell_page(cell, d, r)
    text = visible(page)
    assert "exists · 12 quadruples here, rows verified" in text
    assert (
        "The board holds 12 Williamson quadruples of length 10, no two in the same class: the 2 classes of the "
        "published enumeration"
    ) in text
    assert "The list is complete if the published one is; no search was rerun here." in text
    assert "Published count: 2 classes" in text and "half-period shift of a block" in text
    assert (
        'href="../catalogues/williamson-n10.txt" download' in page
        and "evidence/catalogues/williamson-n10.txt.gz" in page
    )
    # Twelve lines are drawn and each carries its rows for picture.js.
    assert page.count('<figure class="quad">') == 12 and "line 12" in text and "picture.js" in page
    assert "catalogue file · 12 quadruples, every line checked here · 2 published classes, every class inside" in text
    # The dataset is a source of the page, with its reference number.
    assert "The dataset of the enumeration" in text and "ref-bright2018enumeration" in page
    assert not re.search(SLUGS, text)
    tip = module.grid_cell(cell, set(), r)
    assert tip["text"] in ("12", "12+") and "12 quadruples in a catalogue file" in tip["accessible"]
    assert "exists, verified here" in tip["accessible"] and "2 classes published" in tip["accessible"]
    assert module.cell_description(cell).startswith("12 exactly verified Williamson quadruples of length 10")
    wider = re.sub(r"<[^>]+>", "", module.cell_lead(grid[10, "general"], r))
    assert wider.startswith("The board holds 12 verified Williamson quadruples of length 10; every one is a Goethals")
    assert "listed on the page of the Williamson (ssss) cell" in wider


def test_catalogue_with_one_quadruple_per_published_class_says_so(board, tmp_path):
    import re

    module = site_build()
    d, r, grid = catalogue_cells(module, tmp_path, board[2], split=False)
    cell = grid[10, "williamson"]
    text = visible(module.cell_page(cell, d, r))
    assert (
        "The board holds 2 Williamson quadruples of length 10, one from each of the 2 classes of the published "
        "enumeration"
    ) in text
    assert "a published class splits into as many as 32 classes; those are not listed for this length" in text
    assert "2 published classes, one quadruple each" in text and "The first 2 are drawn below" in text
    assert not re.search(SLUGS, text)
    # A quadruple entry outside the file adds to the count, and the page says it is not in the file.
    from verify.census_engine import williamson_even
    from verify.gs_verify import verify as verify_rows

    lines = set(module.catalogue_lines(cell["catalogues"][0]))
    other = next(q for q in sorted(williamson_even(10)) if module.catalogue.format_line(q) not in lines)
    doc = {"id": "n10-other", "n": 10, "group": {"kind": "cyclic", "invariant_factors": [10]}, "rows": other}
    doc["sources"] = []
    with_entry = cell | {"quadruples": ["n10-other"]}
    results = r | {"n10-other": verify_rows(doc)}
    assert module.cell_tag(with_entry, results) == "exists · 3 quadruples here, rows verified"
    page = visible(module.cell_page(with_entry, d | {"n10-other": doc}, results))
    assert "The board holds 1 more Williamson quadruple of length 10 with a page of its own, listed below." in page
    assert "1 quadruple with a page of its own, not in the file" in page
