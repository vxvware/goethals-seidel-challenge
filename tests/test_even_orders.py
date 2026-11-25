"""Even orders: accepted by the verifier, with cells for the two strata that need no skew row."""

from itertools import product

import pytest

from tools.import_even_williamson import coarse
from verify import census_engine
from verify.board import load, summary
from verify.census_engine import WILLIAMSON_EVEN_ORDERS, williamson_even
from verify.common import EVEN_TRACKS, ROOT, TRACKS, read_json, tracks_at
from verify.equivalence import canonical
from verify.gs_verify import memberships, row_types, validate_rows, verify
from verify.model import validate

# n: (classes under equiv-v1, classes in Table 1 of the published enumeration).
COUNTS = {
    2: (1, 1),
    4: (5, 1),
    6: (4, 1),
    8: (5, 1),
    10: (12, 2),
    12: (36, 3),
    14: (76, 5),
    16: (27, 6),
    18: (540, 23),
    20: (412, 17),
}


def entry(rows):
    n = len(rows[0])
    return {
        "schema_version": 1,
        "id": f"even-{n}",
        "n": n,
        "group": {"kind": "cyclic", "invariant_factors": [n]},
        "submitted_by": "vprusso",
        "submitted_at": "2026-11-25",
        "discovered_by": ["test"],
        "sources": [{"title": "test", "url": "https://example.org/"}],
        "type": "quadruple",
        "equivalence": "equiv-v1",
        "rows": rows,
    }


def test_verifier_accepts_even_orders():
    symmetric = verify(validate(entry([[1, 1], [1, 1], [1, -1], [1, -1]])))
    assert symmetric["tracks"] == ["williamson", "general"] and symmetric["pattern"] == "ssss"
    mixed = verify(validate(entry([[1, 1, -1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [1, 1, 1, 1]])))
    assert mixed["tracks"] == ["general"] and mixed["flags"] == []
    with pytest.raises(ValueError, match="PAF identity fails at shift 1"):
        verify(validate(entry([[1, 1], [1, 1], [1, 1], [1, -1]])))


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_no_row_is_skew_at_even_n(n):
    assert all("k" not in row_types((1, *tail)) for tail in product((1, -1), repeat=n - 1))
    assert tracks_at(n) == EVEN_TRACKS == ("williamson", "general") and tracks_at(n + 1) == TRACKS


@pytest.mark.parametrize("n", COUNTS)
def test_even_williamson_counts_agree_with_the_published_enumeration(n):
    classes = williamson_even(n)
    for q in classes:
        validate_rows(q, n)
        assert memberships(q) == ["williamson", "general"]
    records = read_json(ROOT / "baselines/order-status.json", max_bytes=4 * 1024 * 1024)["records"]
    published = next(r["published_classes"]["count"] for r in records if r["n"] == n)
    # The published relation also identifies a shift of one row by n/2 and alternating negation of all four.
    assert (len(classes), len({coarse(q) for q in classes}), published) == (*COUNTS[n], COUNTS[n][1])


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_even_enumerator_agrees_with_the_definition(n):
    """Every 4-tuple of symmetric rows, each checked by the verifier, with no pair table."""
    symmetric = [r for tail in product((1, -1), repeat=n - 1) if "s" in row_types(r := (1, *tail))]
    assert len(symmetric) == 2 ** (n // 2)
    found = set()
    for q in product(symmetric, repeat=4):
        try:
            found.add(canonical(validate_rows(q, n)))
        except ValueError:
            continue
    assert found == williamson_even(n)


def test_even_enumerator_has_a_fixed_range():
    assert list(WILLIAMSON_EVEN_ORDERS) == list(COUNTS)
    for n in (1, 9, 22):
        with pytest.raises(ValueError, match="even orders 2 to 20"):
            williamson_even(n)


def test_even_census_is_exact_only_when_it_equals_the_enumeration():
    n = 8
    quads = {
        f"q{i}": {"n": n, "tracks": ["williamson", "general"], "canonical_rows": q}
        for i, q in enumerate(sorted(williamson_even(n)))
    }
    census = {
        "id": "williamson-n8",
        "n": n,
        "stratum": "williamson",
        "equivalence": "equiv-v1",
        "representatives": sorted(quads),
        "claimed_count": len(quads),
        "claim_kind": "computed-exact",
        "enumerator": "williamson-even-exact-v1",
    }
    result = census_engine.verify(census, quads)
    assert (result["bound"], result["tier"], result["count"]) == ("exact", "reproduced", 5)
    assert census_engine.verify(census, quads, reproduce=False)["bound"] == "lower"
    with pytest.raises(ValueError, match="disagrees"):
        census_engine.verify({**census, "representatives": sorted(quads)[1:], "claimed_count": 4}, quads)
    with pytest.raises(ValueError, match="no trusted enumerator"):
        census_engine.verify({**census, "stratum": "general"}, quads)


def test_cells_at_even_n_are_the_strata_without_a_skew_row():
    cells = summary(*load(reproduce=False))["cells"]
    by_n: dict[int, list[str]] = {}
    for c in cells:
        by_n.setdefault(c["n"], []).append(c["stratum"])
    assert sorted(by_n) == list(range(1, 301))
    assert all(tuple(by_n[n]) == tracks_at(n) for n in by_n)
    lookup = {(c["n"], c["stratum"]): c for c in cells}
    # A published existence record is an import task until rows are on the board, and reaches the general cell.
    for n in (22, 70, 128):
        assert lookup[n, "williamson"]["status"] == lookup[n, "general"]["status"] == "exists"
        assert lookup[n, "williamson"]["origins"] == ["published-import"]
    for n in (94, 106, 118, 130):
        assert lookup[n, "williamson"]["status"] == "unresolved" and lookup[n, "williamson"]["sources"] == []
