from copy import deepcopy
import pytest
from verify.board import load, summary, check_consistency
from verify.common import ROOT, read_json
from verify.census_engine import small_exact, best_census


@pytest.fixture(scope="module")
def board():
    return load(reproduce=False)


def test_no_false_empty_inherited_to_parent(board):
    d, r, b = board
    s = summary(d, r, b)
    lookup = {(c["n"], c["stratum"]): c for c in s["cells"]}
    assert lookup[89, "skew-sds"]["status"] == "unresolved"
    assert lookup[67, "good"]["status"] == "unresolved"
    assert lookup[65, "good"]["status"] == "unresolved"
    assert lookup[41, "good"]["status"] == "empty"
    assert lookup[41, "general"]["status"] == "exists"
    assert lookup[57, "good"]["status"] == "exists"
    assert lookup[57, "good"]["artifact_status"] == "published-not-imported"
    assert lookup[61, "good"]["status"] == "unresolved"


def test_targets_and_import_tasks_separate(board):
    s = summary(*board)
    assert all(c["status"] == "unresolved" for c in s["research_targets"])
    assert all(
        c["status"] == "exists" and c["artifact_status"] == "published-not-imported"
        for c in s["import_tasks"]
    )


def test_known_larger_constructions_and_impossible_g_orders_are_not_targets(board):
    stats = summary(*board)
    cells = {(c["n"], c["stratum"]): c for c in stats["cells"]}
    for n in (69, 85, 91, 129):
        assert cells[n, "williamson"]["artifact_status"] == "published-not-imported"
        assert cells[n, "general"]["status"] == "exists"
    for n in (11, 17, 35, 39, 47, 49):
        assert cells[n, "kkss"]["status"] == "empty"


def test_conflicting_claim_fails(board):
    d, r, b = board
    r = deepcopy(r)
    cert = deepcopy(r["good-n41-campaign"])
    cert["n"] = 13
    cert["id"] = "bad"
    r["bad"] = cert
    with pytest.raises(ValueError, match="conflict"):
        check_consistency(d, r, b)


@pytest.mark.parametrize("n,count", [(3, 1), (5, 1), (7, 3), (9, 1), (11, 3), (13, 6), (15, 11)])
def test_good_full_censuses(n, count):
    assert len(small_exact(n, "good")) == count


def test_full_best43_census_gate():
    reps, secondary = best_census(43)
    assert len(reps) == 5
    assert secondary == {"branch_orbits": 20, "expanded_quadruples": 10080}


def test_published_292_is_verified_and_attributed(board):
    manifest = read_json(ROOT / "baselines/seed-manifest.json", max_bytes=4 * 1024 * 1024)
    artifacts = {r["id"]: r for r in manifest["artifacts"]}
    d, results, _ = board
    for ident in artifacts["published-skew-n73"]["representatives"]:
        assert "skew-sds" in results[ident]["tracks"]
        assert results[ident]["n"] == 73
        assert any("Đoković" in credit for credit in d[ident]["discovered_by"])
