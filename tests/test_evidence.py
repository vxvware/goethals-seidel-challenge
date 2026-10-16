import json
import pytest
from verify.common import ROOT, read_json, write_json, digest
from verify.evidence import intervals_union, verify, proof
from verify.recheck_slices import sample_classes, validate_replayed


def test_tiling_gaps_overlaps_and_bounds():
    assert intervals_union([[5, 10], [0, 5]], 10) == (10, [])
    assert intervals_union([[3, 5]], 10) == (2, [[0, 3], [5, 10]])
    for intervals in ([[0, 6], [5, 10]], [[0, 11]], [[-1, 2]], [[2, 2]]):
        with pytest.raises(ValueError):
            intervals_union(intervals, 10)


def test_good65_retains_actual_missing_prefix():
    entry = read_json(ROOT / "certificates/good-n65-campaign.json")
    result = verify(entry, ROOT)
    assert result["tier"] is None
    first = next(p for p in result["coverage"] if p["pass"] == -15)
    assert first["uncovered"] == [[0, 454800]]
    entry["method"] = "good-coverage-v1"
    with pytest.raises(ValueError, match="gaps"):
        verify(entry, ROOT)


def test_unreplayed_census_is_an_attributed_claim():
    entry = read_json(ROOT / "certificates/vxvware-kkss-n57-census.json")
    result = verify(entry, ROOT)
    assert result["tier"] is None
    assert result["provenance"] == "unreplayed-census"
    assert any("not been replayed" in gap for gap in result["gaps"])
    narrowed = {**entry, "scope": {"kind": "same-subgroup-coset-unions"}}
    with pytest.raises(ValueError, match="full stratum"):
        verify(narrowed, ROOT)
    unarchived = {**entry, "sources": [{"title": "t", "url": "https://example.org"}]}
    with pytest.raises(ValueError, match="archived"):
        verify(unarchived, ROOT)


def test_partial_is_not_full_emptiness():
    entry = read_json(ROOT / "certificates/good-n67-campaign.json")
    result = verify(entry, ROOT)
    assert result["claim"] == "partial"
    assert result["coverage"][0]["covered"] == 927600
    entry["claim"] = "empty"
    with pytest.raises(ValueError):
        verify(entry, ROOT)


def test_receipt_hash_and_pending_fail_closed(tmp_path):
    entry = read_json(ROOT / "certificates/good-n41-campaign.json")
    import shutil

    for ref in [*entry["records"], entry["engine"]["artifact"]]:
        p = tmp_path / ref["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / ref["path"], p)
    ref = entry["records"][0]
    p = tmp_path / ref["path"]
    data = json.loads(p.read_text())
    data["pending"] = ["x"]
    write_json(p, data)
    with pytest.raises(ValueError, match="hash"):
        verify(entry, tmp_path)
    ref["sha256"] = digest(p)
    with pytest.raises(ValueError, match="unsettled"):
        verify(entry, tmp_path)
    data["pending"] = []
    del data["unsettled"]
    write_json(p, data)
    ref["sha256"] = digest(p)
    with pytest.raises(ValueError, match="missing"):
        verify(entry, tmp_path)


def test_exact_scoped_obstruction():
    entry = read_json(ROOT / "certificates/skew-n89-subgroup11.json")
    result = verify(entry, ROOT)
    assert result["tier"] == "proven"
    entry["scope"]["subgroup_order"] = 1
    with pytest.raises(ValueError, match="does not rule out"):
        proof(entry)


def test_best_arithmetic_obstruction():
    entry = read_json(ROOT / "certificates/best-n5-row-sums.json")
    assert verify(entry, ROOT)["tier"] == "proven"
    entry["n"] = 7
    with pytest.raises(ValueError):
        proof(entry)


def test_random_slices_bind_actual_intervals():
    coverage = [{"pass": -3, "intervals": [[10, 20], [30, 40]]}, {"pass": 5, "intervals": [[0, 5]]}]
    sample = sample_classes(coverage, 25, 123)
    assert len(set(sample)) == 25
    assert sample == sorted(
        [(-3, i) for i in range(10, 20)]
        + [(-3, i) for i in range(30, 40)]
        + [(5, i) for i in range(5)]
    )
    with pytest.raises(ValueError):
        sample_classes(coverage, 26, 1)


def test_replay_receipt_rejects_changed_deck_and_cursor():
    r = {
        "n": 41,
        "sB": 9,
        "start": 7,
        "end": 8,
        "classes": 1870,
        "order_digest": "a" * 16,
        "unsettled": 0,
        "pending": [],
    }
    validate_replayed(r, 41, 9, 7, 8, r)
    with pytest.raises(ValueError):
        validate_replayed(r, 41, 9, 7, 9, r)
    with pytest.raises(ValueError):
        validate_replayed(r, 41, 9, 7, 8, {**r, "order_digest": "b" * 16})


def test_registered_replay_promotion_and_failure():
    from copy import deepcopy
    from verify.replay_registry import apply_report
    from verify.recheck_slices import ENGINE

    entry = read_json(ROOT / "certificates/good-n41-campaign.json")
    result = verify(entry, ROOT)
    source = read_json(ROOT / entry["records"][0]["path"])
    s, index = sample_classes(result["coverage"], 1, 76)[0]
    report = {
        "entry_id": entry["id"],
        "entry_sha256": "test-entry-hash",
        "engine_id": "good-gpu-v4",
        "engine_sha256": digest(ENGINE),
        "status": "passed",
        "mode": "sample",
        "seed": 76,
        "sample_count": 1,
        "samples": [{**source, "sB": s, "start": index, "end": index + 1}],
    }
    apply_report(entry, result, report, "test-entry-hash")
    assert result["tier"] == "spot-checked"
    for patch in (
        {"status": "failed"},
        {"entry_sha256": "stale"},
        {"engine_sha256": "wrong"},
        {"sample_count": 2},
    ):
        with pytest.raises(ValueError):
            apply_report(entry, deepcopy(result), {**report, **patch}, "test-entry-hash")
    report.update(mode="full", samples=[source])
    apply_report(entry, result, report, "test-entry-hash")
    assert result["tier"] == "reproduced"


def test_sampling_does_not_repair_archived_coverage():
    from verify.replay_registry import apply_report
    from verify.recheck_slices import ENGINE

    entry = read_json(ROOT / "certificates/good-n65-campaign.json")
    result = verify(entry, ROOT)
    s, index = sample_classes(result["coverage"], 1, 50)[0]
    p = next(x for x in result["coverage"] if x["pass"] == s)
    report = {
        "entry_id": entry["id"],
        "entry_sha256": "test",
        "engine_id": "good-gpu-v4",
        "engine_sha256": digest(ENGINE),
        "status": "passed",
        "mode": "sample",
        "seed": 50,
        "sample_count": 1,
        "samples": [
            {
                "n": 65,
                "sB": s,
                "start": index,
                "end": index + 1,
                "classes": p["classes"],
                "order_digest": p["order_digest"],
                "pending": [],
                "unsettled": 0,
            }
        ],
    }
    apply_report(entry, result, report, "test")
    assert result["tier"] is None
