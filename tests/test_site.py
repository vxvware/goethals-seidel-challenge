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
    found = [x for x in module.existence_results(stats, r) if x["kind"] == "matrix"]
    assert [x["text"] for x in found] == [
        f"A skew-Hadamard matrix of order {order}." for order in (808, 1416, 1712, 2360, 2696, 5336, 5392)
    ]
    item = found[0]
    assert item["qualifier"] is None
    assert (item["n"], item["href"]) == (202, "orders/skew-808.html")
    html = module.new_results_section(stats, r)
    assert '<a href="orders/skew-808.html">A skew-Hadamard matrix of order 808.</a>' in html
    assert "did not find in the sources it read" in html and "does not claim priority" in html
    assert "first" not in html.lower() and "1616" not in html
    # Every found order has a line of its own, and the skew-Hadamard lines stand together before the others.
    assert '<a href="orders/skew-5392.html">A skew-Hadamard matrix of order 5392.</a>' in html
    kinds = [x["kind"] for x in module.existence_results(stats, r)]
    assert "propus" not in kinds or kinds.index("propus") == len(kinds) - 1


def test_propus_result_is_worded_and_linked_without_marking_the_grid(board):
    module = site_build()
    d, r, b = board
    stats = summary(d, r, b)
    (group,) = (g for g in stats["contributions"] if g["id"] == "propus-pool-join-65")
    (ident,) = group["entry_ids"]
    (item,) = (x for x in module.existence_results(stats, r) if x["kind"] == "propus")
    assert item["text"] == "A symmetric Hadamard matrix of order 260 from circulant blocks."
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
