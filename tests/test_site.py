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
    assert item["text"] == "A skew-Hadamard matrix of order 356, and of order 712 by doubling."
    assert item["double"] == 712
    assert item["qualifier"] is None
    assert ("skew-sds", 89) in module.new_cells(stats, r)
    html = module.new_results_section(stats, r)
    assert "cells/n89-skew-sds.html" in html
    # The section states its search qualifier once, for every item in it.
    assert "did not find in the sources it read" in html and "does not claim priority" in html
    assert "first" not in html.lower()


def test_found_matrix_is_listed_among_the_existence_results(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    (item,) = (x for x in module.existence_results(stats, r) if x["kind"] == "matrix")
    assert item["text"] == "A skew-Hadamard matrix of order 808." and item["qualifier"] is None
    assert (item["n"], item["href"]) == (202, "orders/skew-808.html")
    html = module.new_results_section(stats, r)
    assert '<a href="orders/skew-808.html">A skew-Hadamard matrix of order 808.</a>' in html
    assert "did not find in the sources it read" in html and "does not claim priority" in html
    assert "first" not in html.lower() and "1616" not in html
    # The board has no cell for n = 202, so no family cell is marked; the order cell carries the marker.
    assert not any(n == 202 for _, n in module.new_cells(stats, r))
    _, rows, _ = module.grid(stats, r)
    row = rows[rows.index('<tr data-n="202"') :]
    assert row[: row.index("</tr>")].count('class="participant"') == 1


def test_propus_result_is_worded_and_linked_without_marking_the_grid(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    (group,) = (g for g in stats["contributions"] if g["id"] == "propus-pool-join-65")
    (ident,) = group["entry_ids"]
    (item,) = (x for x in module.existence_results(stats, r) if x["kind"] == "propus")
    assert item["text"] == "A propus family over Z_65: a symmetric Hadamard matrix of order 260 from circulant blocks."
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


def test_restricted_certificate_is_not_labelled_as_a_statement_about_the_cell(board):
    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (89, "skew-sds"))
    page = module.cell_page(cell, d, r)
    assert "proven empty within a restricted family, not for the whole cell" in page
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
        assert not re.search(r"not (been )?replayed|awaiting a full replay", text), (cell["n"], cell["stratum"])
    assert pages > 0
