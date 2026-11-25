"""Catalogue files: many quadruples of one cell in one file, each line checked in full."""

import gzip
import hashlib

import pytest

from tools.import_bright_williamson import split
from tools.import_even_williamson import coarse
from verify import catalogue
from verify.board import cells
from verify.census_engine import verify, williamson_even
from verify.equivalence import canonical_half_shifts
from verify.model import validate

N = 10


def write(root, classes, name="williamson-n10"):
    body = "".join(catalogue.format_line(q) + "\n" for q in classes).encode()
    return write_bytes(root, gzip.compress(body, mtime=0), name)


def write_bytes(root, packed, name="williamson-n10"):
    folder = root / "evidence/catalogues"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.txt.gz").write_bytes(packed)
    return {"path": f"evidence/catalogues/{name}.txt.gz", "sha256": hashlib.sha256(packed).hexdigest()}


def census(ref, count, published=None, **changes):
    entry = {
        "schema_version": 1,
        "id": "williamson-n10",
        "type": "census",
        "n": N,
        "group": {"kind": "cyclic", "invariant_factors": [N]},
        "submitted_by": "tester",
        "submitted_at": "2026-11-25",
        "discovered_by": ["test"],
        "sources": [{"title": "test", "url": "https://example.org/"}],
        "stratum": "williamson",
        "equivalence": "equiv-v1",
        "representatives": [],
        "catalogue": ref,
        "claimed_count": count,
        "claim_kind": "lower-bound",
    }
    if published:
        entry["published_classes"] = published
    return validate(entry | changes)


@pytest.fixture(scope="module")
def classes():
    return sorted(williamson_even(N))


def test_half_shift_canonical_form_is_the_published_relation():
    # The exhaustive form over the 32 combinations of shifts and alternation is the importer's reference.
    for n in (2, 4, 6, 8, 10, 12, 14):
        found = williamson_even(n)
        assert all(canonical_half_shifts(q) == coarse(q) for q in found)
    with pytest.raises(ValueError, match="even length"):
        canonical_half_shifts(((1, 1, 1),) * 4)


def test_split_of_one_quadruple_per_published_class_is_the_whole_enumeration():
    for n in (4, 8, 10, 12, 14, 16):
        found = williamson_even(n)
        one_each = {canonical_half_shifts(q): q for q in sorted(found)}.values()
        assert set().union(*(split(q) for q in one_each)) == found


def test_catalogue_of_every_class_is_checked_line_by_line(tmp_path, classes):
    ref = write(tmp_path, classes)
    entry = census(ref, 12, {"count": 2, "split": True})
    result = verify(entry, {}, True, tmp_path)
    assert (result["count"], result["bound"], result["tier"]) == (12, "lower", None)
    assert result["catalogue"] == {
        "path": ref["path"],
        "sha256": ref["sha256"],
        "rows": 12,
        "published_classes": 2,
        "split": True,
        "published_checked": True,
    }
    assert "published-classes-recomputed" in result["checks"]
    quick = verify(entry, {}, False, tmp_path)
    assert quick["catalogue"]["published_checked"] is False and quick["count"] == 12
    with pytest.raises(ValueError, match="disagrees with its stated published classes"):
        verify(census(ref, 12, {"count": 3, "split": True}), {}, True, tmp_path)
    with pytest.raises(ValueError, match="differs from the claimed count"):
        verify(census(ref, 13), {}, True, tmp_path)


def test_catalogue_with_one_quadruple_per_published_class(tmp_path, classes):
    one_each = sorted({canonical_half_shifts(q): q for q in classes}.values())
    ref = write(tmp_path, one_each)
    result = verify(census(ref, 2, {"count": 2, "split": False}), {}, True, tmp_path)
    assert result["catalogue"]["rows"] == 2 and result["catalogue"]["split"] is False
    # It is not closed under the two further operations, so it cannot claim to hold every class.
    with pytest.raises(ValueError, match="disagrees with its stated published classes"):
        verify(census(ref, 2, {"count": 2, "split": True}), {}, True, tmp_path)
    with pytest.raises(ValueError, match="as many lines as classes"):
        verify(census(write(tmp_path, classes[:3], "three"), 3, {"count": 2, "split": False}), {}, True, tmp_path)


def test_catalogue_lines_must_be_exact_canonical_and_increasing(tmp_path, classes):
    def fails(lines, match, name):
        ref = write_bytes(tmp_path, gzip.compress("".join(line + "\n" for line in lines).encode(), mtime=0), name)
        with pytest.raises(ValueError, match=match):
            verify(census(ref, len(lines)), {}, False, tmp_path)

    good = [catalogue.format_line(q) for q in classes]
    fails([good[1], good[0]], "line 2: the lines are not strictly increasing", "order")
    fails([good[0], good[0]], "line 2: the lines are not strictly increasing", "repeat")
    flipped = ("-" if good[0][1] == "+" else "+").join((good[0][:1], good[0][2:]))
    fails([flipped], "line 1: PAF identity fails", "paf")
    a, b, c, d = good[3].split(" ")
    assert b != a
    fails([" ".join((b, a, c, d))], "line 1: the rows are not the canonical form", "form")
    fails([good[0].replace(" ", "  ", 1)], "line 1: a catalogue line is four sign strings", "spaces")
    fails([good[0][:-1]], "line 1: a catalogue line is four sign strings", "short")
    with pytest.raises(ValueError, match="oversized or not one complete gzip stream"):
        packed = gzip.compress((good[0] + "\n").encode(), mtime=0)
        verify(census(write_bytes(tmp_path, packed + b"x", "tail"), 1), {}, False, tmp_path)
    with pytest.raises(ValueError, match="does not end with a newline"):
        verify(census(write_bytes(tmp_path, gzip.compress(good[0].encode(), mtime=0), "nl"), 1), {}, False, tmp_path)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        ref = write(tmp_path, classes, "hash")
        verify(census(ref | {"sha256": "0" * 64}, 12), {}, False, tmp_path)


def test_catalogue_census_is_a_lower_bound_without_representatives(tmp_path, classes):
    ref = write(tmp_path, classes)
    with pytest.raises(ValueError, match="lower bound under equiv-v1"):
        verify(census(ref, 12, claim_kind="computed-exact", enumerator="williamson-even-exact-v1"), {}, True, tmp_path)
    with pytest.raises(ValueError, match="lower bound under equiv-v1"):
        verify(census(ref, 12, representatives=["n10-x"]), {}, True, tmp_path)
    with pytest.raises(ValueError, match="published_classes describes a catalogue"):
        entry = census(ref, 12, {"count": 2, "split": True})
        del entry["catalogue"]
        verify(entry, {}, True, tmp_path)
    with pytest.raises(ValueError, match="schema"):
        census(ref | {"path": "evidence/other/williamson-n10.txt.gz"}, 12)


def test_board_quadruple_must_be_in_a_catalogue_that_holds_whole_published_classes(tmp_path, classes):
    ref = write(tmp_path, classes)
    entry = census(ref, 12, {"count": 2, "split": True})
    inside = {"n10-a": {"id": "n10-a", "n": N, "tracks": ["williamson", "general"], "canonical_rows": classes[5]}}
    assert verify(entry, inside, True, tmp_path)["count"] == 12
    # Remove one published class: the rest is still closed, and a board quadruple of the missing class is flagged.
    kept = [q for q in classes if canonical_half_shifts(q) != canonical_half_shifts(classes[5])]
    assert 0 < len(kept) < 12
    partial = census(write(tmp_path, kept, "partial"), len(kept), {"count": 1, "split": True})
    with pytest.raises(ValueError, match="n10-a is missing from the catalogue"):
        verify(partial, inside, True, tmp_path)


def test_cell_with_a_catalogue_holds_verified_quadruples(tmp_path, classes):
    ref = write(tmp_path, classes)
    entry = census(ref, 12, {"count": 2, "split": True})
    result = verify(entry, {}, True, tmp_path) | {"contributions": []}
    grid = {(c["n"], c["stratum"]): c for c in cells({entry["id"]: entry}, {entry["id"]: result}, [], max_n=N)}
    cell = grid[N, "williamson"]
    assert (cell["status"], cell["artifact_status"], cell["tier"]) == ("exists", "verified-on-board", "reproduced")
    assert cell["catalogues"] == [{"census": "williamson-n10", "stratum": "williamson"} | result["catalogue"]]
    assert cell["censuses"] == ["williamson-n10"] and cell["quadruples"] == []
    # A Williamson quadruple is a quadruple of the cell for any quadruple, which has no census of its own here.
    wider = grid[N, "general"]
    assert wider["status"] == "exists" and wider["catalogues"] == cell["catalogues"] and wider["censuses"] == []
    assert grid[N - 2, "williamson"]["catalogues"] == []
