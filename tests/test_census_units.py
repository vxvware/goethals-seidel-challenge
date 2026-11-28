"""The per-class records of the best-matrix census at n=73 and the two programs that check them.

The registered manifest is validated as the board validates it, then altered in
the ways a stale or forged record would be. best_cover.c and best_plant.c run at
n=43 against the engine itself: the D and A counts of the coverage program must
equal the engine's log, its class forms must equal the engine's, and every
plant in every one of the 512 classes must come back.
"""

import copy
import os
import re
import subprocess

import pytest

from verify import census_units
from verify.common import ROOT, read_json

BEST = ROOT / "research/starter/best"


def manifest_and_summary():
    ((manifest, summary),) = census_units.registered()
    return manifest, summary


def test_registered_manifest_is_consistent():
    manifest, summary = manifest_and_summary()
    assert manifest["census"] == "best-n73" and summary["classes"] == 512
    assert summary["done"] and not set(summary["done"]) & set(summary["missing"])
    assert len(summary["done"]) + len(summary["missing"]) == 512
    header = manifest["header"]
    assert header["d_minima"] == 11178678 and header["d_signature"] == 1400391
    assert header["a_codes"] == 1 << 35
    assert summary["coverage"]["total_pairs"] == header["d_signature"] * header["a_ceiling"]
    for unit in manifest["units"]:
        assert unit["counters"]["capped"] == 0
        assert unit["counters"]["pairs_index"] <= summary["coverage"]["pairs"][unit["class"]]


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["units"][0]["counters"].update(patterns=m["units"][0]["counters"]["patterns"] + 1),
        lambda m: m["units"][0].update(log_sha256="0" * 64),
        lambda m: m["units"].append(copy.deepcopy(m["units"][0])),
        lambda m: m["units"][0].update(**{"class": (m["units"][0]["class"] + 1) % 512}),
        lambda m: m["header"].update(a_ceiling=m["header"]["a_ceiling"] - 1),
        lambda m: m["engine_sha256"].update({next(iter(m["engine_sha256"])): "0" * 64}),
        lambda m: m["units"].pop(),
    ],
)
def test_altered_manifest_is_rejected(change):
    manifest, _ = manifest_and_summary()
    altered = copy.deepcopy(manifest)
    change(altered)
    with pytest.raises(ValueError):
        census_units.check_manifest(altered)


def test_report_needs_a_complete_manifest():
    manifest, summary = manifest_and_summary()
    ref = next(r for r in read_json(ROOT / "baselines/census_replays.json")["units"])
    report = {"engine_id": manifest["engine_id"], "units": ref, "rows": []}
    entry = {"id": "best-n73", "n": 73}
    if summary["complete"]:
        assert census_units.apply_units(report, entry, 0)["complete"]
    else:
        with pytest.raises(ValueError, match="incomplete"):
            census_units.apply_units(report, entry, 0)
    with pytest.raises(ValueError, match="another census"):
        census_units.apply_units(report, {"id": "best-n57", "n": 57}, 0)


def test_formal_theorems_exist():
    for t in read_json(ROOT / "baselines/formal.json")["theorems"]:
        text = (ROOT / t["file"]).read_text()
        name = t["theorem"].rsplit(".", 1)[-1]
        owner = t["theorem"].rsplit(".", 2)[-2]
        assert re.search(rf"^theorem (?:_root_\.\S*)?(?:{owner}\.)?{re.escape(name)}\b", text, re.M), t["theorem"]


def run(*args, env=None):
    full = {k: v for k, v in os.environ.items() if not k.startswith("BEST_")}
    full.update(env or {})
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True, env=full, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:] + p.stdout[-2000:]
    return p.stdout


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    folder = tmp_path_factory.mktemp("best")
    for name in ("best_hkt2", "best_cover", "best_plant"):
        run("gcc", "-O3", "-march=native", "-fopenmp", "-I", BEST, BEST / f"{name}.c", "-lm", "-o", folder / name)
    return folder


def test_coverage_and_plants_at_43(built):
    log = run(built / "best_hkt2", 43, 2, built / "out.txt", env={"BEST_QB": "9", "BEST_CLASS": "0"})
    cover = census_units.parse_coverage(run(built / "best_cover", 43, 9, 2))
    d = re.search(r"D rows: (\d+) with the row sum, (\d+) under the ceiling, (\d+) kept", log)
    sig = re.search(r"\(R6\): (\d+) of", log)
    a = re.search(r"A rows under the ceiling: (\d+) of (\d+)", log)
    assert d and sig and a
    assert cover["D with the row sum"] == int(d[1]) and cover["D under the ceiling"] == int(d[2])
    assert cover["D orbit minima"] == int(d[3]) and cover["D with the signature"] == int(sig[1])
    assert cover["A under the ceiling"] == int(a[1]) and cover["A codes"] == int(a[2])
    assert sum(cover["dlabel"].values()) == int(sig[1]) and sum(cover["alabel"].values()) == int(a[1])
    plant = census_units.parse_plant(run(built / "best_plant", 43, 9, 2, 0, 511, 2))
    assert plant["forms"] == {tuple(cover["forms"])}
    assert sorted(plant["classes"]) == list(range(512)) and not plant["misses"]
    assert all(c["recovered"] == c["planted"] == 2 and c["capped"] == 0 for c in plant["classes"].values())
