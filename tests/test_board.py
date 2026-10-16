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
    assert lookup[61, "good"]["status"] == "exists"
    assert lookup[59, "good"]["status"] == "empty"
    assert lookup[59, "good"]["tier"] is None
    assert lookup[59, "good"]["empty_basis"] == ["unreplayed-census"]
    assert lookup[59, "skew-sds"]["status"] == "exists"  # kkss rows, not the empty good cell


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
    for n in (11, 17, 35, 39, 47, 53):
        assert cells[n, "kkss"]["status"] == "empty"
    assert cells[49, "kkss"]["status"] == "exists"
    assert cells[49, "kkss"]["sources"] == []
    assert [f["kind"] for f in cells[49, "kkss"]["flags"]] == ["contradicts-published-nonexistence"]
    assert "1802.00556" in cells[49, "kkss"]["contradicted"][0]["source"]["url"]
    assert cells[57, "kkss"]["status"] == "empty"
    assert cells[57, "kkss"]["empty_basis"] == ["unreplayed-census"]
    assert cells[57, "kkss"]["origins"] == ["participant-result"]


def test_contradicted_claim_requires_verified_rows(board):
    d, r, b = board
    stale = [{**rec, "n": 57} if rec["status"] == "contradicted" else rec for rec in deepcopy(b)]
    with pytest.raises(ValueError, match="contradicted claim without verified rows"):
        check_consistency(d, r, stale)


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
        assert {g["relation"] for g in results[ident]["contributions"]} == {"published-import"}


def test_participant_credit_does_not_change_evidence_status(board):
    d, r, b = board
    cells = {(c["n"], c["stratum"]): c for c in summary(d, r, b)["cells"]}
    for n in (53, 55, 61):
        cell = cells[n, "good"]
        assert cell["status"] == "exists"
        assert cell["origins"] == ["participant-result"]
        credit = cell["contributions"][0]
        assert credit["contributors"] == [{"name": "vxvware", "github": "vxvware"}]
        assert credit["publication_status"] == "unpublished"
        assert r[f"vxvware-good-n{n}"]["bound"] == "lower"
    correction = cells[49, "kkss"]
    assert correction["status"] == "exists" and correction["tier"] == "reproduced"
    assert correction["origins"] == ["participant-result"]
    assert correction["contributions"][0]["id"] == "kkss-49-correction"
    assert cells[33, "kkss"]["origins"] == ["independent-reproduction", "published-import"]
    assert cells[33, "kkss"]["artifact_status"] == "verified-on-board"
    best73 = cells[73, "best"]
    assert best73["status"] == "unresolved" and best73["tier"] is None
    assert best73["origins"] == ["participant-result"]
    assert best73["contributions"][0]["contributors"][0]["name"] == "vxvware"
    assert cells[73, "skew-sds"]["origins"] == ["participant-result", "published-import"]
    assert cells[41, "good"]["origins"] == ["independent-reproduction", "published-import"]
    assert cells[89, "skew-sds"]["origins"] == ["participant-result"]
    assert cells[89, "best"]["origins"] == ["published-import"]
    assert "participant-result" not in cells[89, "good"]["origins"]
    assert all(c["id"] != "best-73-campaign" for c in cells[73, "general"]["contributions"])


def test_submitter_is_not_automatically_discoverer(board):
    from verify.contributions import apply

    d, r, _ = deepcopy(board)
    for doc in d.values():
        doc["submitted_by"] = "unrelated-importer"
    apply(d, r)
    assert r["best-n73-campaign"]["contributions"][0]["contributors"][0]["github"] == "vxvware"
    assert all(
        p.get("github") != "unrelated-importer"
        for result in r.values()
        for g in result["contributions"]
        for p in g["contributors"]
    )


@pytest.mark.parametrize("bad_reference", ["seed", "entry", "stratum", "duplicate", "schema"])
def test_invalid_contribution_registry_rejected(board, tmp_path, bad_reference):
    import shutil
    from verify.contributions import apply
    from verify.common import write_json

    shutil.copytree(ROOT / "baselines", tmp_path / "baselines")
    registry = read_json(tmp_path / "baselines/contributions.json")
    first = registry["contributions"][0]
    if bad_reference == "seed":
        first["seed_artifacts"] = ["missing-seed"]
    elif bad_reference == "entry":
        first["entry_ids"] = ["missing-entry"]
    elif bad_reference == "stratum":
        first["stratum"] = "williamson"
    elif bad_reference == "duplicate":
        registry["contributions"].append(deepcopy(first))
    else:
        first["relation"] = "self-certified-discovery"
    write_json(tmp_path / "baselines/contributions.json", registry)
    d, r, _ = deepcopy(board)
    with pytest.raises(ValueError):
        apply(d, r, tmp_path)


def gs_array(rows):
    """Mirror of the block layout drawn by site/picture.js."""
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


@pytest.mark.parametrize("ident", ["n49-703082c3c1ffa487f376", "n13-02eb7cd20dc9fc9999b6"])
def test_pictured_array_is_hadamard(board, ident):
    import numpy as np

    rows = board[1][ident]["canonical_rows"]
    h = gs_array(rows)
    assert np.array_equal(h @ h.T, 4 * len(rows[0]) * np.eye(len(h), dtype=int))
