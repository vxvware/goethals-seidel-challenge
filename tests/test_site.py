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
    assert "Existence" not in dict(module.new_results(stats, r))
    assert ("skew-sds", 89) not in module.new_cells(stats, r)
    group["featured"] = True
    (item,) = dict(module.new_results(stats, r))["Existence"]
    assert item["text"] == ("A skew-Hadamard matrix of order 356.")
    assert item["qualifier"] is None
    assert ("skew-sds", 89) in module.new_cells(stats, r)
    html = module.new_results_section(stats, r)
    assert "cells/n89-skew-sds.html" in html
    # The section states its search qualifier once, for every item in it.
    assert "did not find in the sources it read" in html and "does not claim priority" in html
    assert "first" not in html.lower()


def test_restricted_certificate_is_not_labelled_as_a_statement_about_the_cell(board):
    module = site_build()
    d, r, b = board
    cell = next(c for c in summary(d, r, b)["cells"] if (c["n"], c["stratum"]) == (89, "skew-sds"))
    page = module.cell_page(cell, d, r)
    assert "proven empty within a restricted family, not for the whole cell" in page
    assert "skew-Hadamard matrix of order 356" in page
    assert "rows verified" in page and "reproduced" not in page.split("<h2>")[0]
    assert "skew-Hadamard matrix of order 356" in module.cell_description(cell)
