"""The two-divisor compression census engine (research/starter/compress2) at small orders.

Each case writes the instances at length n/d1 with pc2_inst.c, sorts them by
signature, lifts them through the length n/d2 with pc2_join.c and compares the
equiv-v1 classes of the rows with the board's census of the cell, then checks
that the rows pass the registered-replay check of the engine for the stratum.
pc2_inst.c must write the same instances as the pinned pc_compress.c.
"""

import os
import subprocess

import pytest

from verify import census_replays
from verify.common import ROOT, digest, read_json
from verify.equivalence import canonical
from verify.gs_verify import memberships, validate_rows

SRC = ROOT / "research/starter/compress2"
OLD = ROOT / "research/starter/compress"
# (pattern, n, d1, d2, census, threads): both orders of the divisors, fibres
# with g = 1 and g = 3, and two threads on one case.
CASES = (
    ("ksss", 15, 5, 3, "good-n15", 1),
    ("ksss", 21, 3, 7, "good-n21", 1),
    ("ksss", 33, 11, 3, "good-n33", 1),
    ("ksss", 35, 5, 7, "good-n35", 1),
    ("ksss", 39, 3, 13, "good-n39", 1),
    ("ksss", 45, 5, 3, "vxvware-good-n45", 2),
    ("ssss", 15, 3, 5, "williamson-n15", 1),
    ("ssss", 21, 7, 3, "williamson-n21", 1),
    ("ssss", 33, 3, 11, "williamson-n33", 1),
    ("ssss", 35, 7, 5, "williamson-n35", 1),
    ("ssss", 39, 13, 3, "williamson-n39", 1),
    ("ssss", 45, 5, 3, "williamson-n45", 1),
    ("kkss", 15, 5, 3, "vxvware-kkss-n15", 1),
    ("kkss", 21, 3, 7, "vxvware-kkss-n21", 1),
    ("kkss", 33, 3, 11, "vxvware-kkss-n33", 1),
    ("kkss", 35, 5, 7, "vxvware-kkss-n35", 1),
    ("kkss", 39, 3, 13, "vxvware-kkss-n39", 1),
    ("kkss", 45, 5, 3, "vxvware-kkss-n45", 1),
    ("kkks", 21, 3, 7, "best-n21", 1),
    ("kkks", 21, 7, 3, "best-n21", 1),
)
STRATUM_ENGINE = {
    "good": "good-compress2-v1",
    "kkss": "kkss-compress2-v1",
    "best": "best-compress2-v1",
    "williamson": "williamson-compress2-v1",
}


def run(*args, env=None):
    full = {k: v for k, v in os.environ.items() if not k.startswith("PC_")}
    full.update(env or {})
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True, env=full, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:] + p.stdout[-2000:]
    return p


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    d = tmp_path_factory.mktemp("compress2")
    out = {}
    for name, source in (
        ("inst", SRC / "pc2_inst.c"),
        ("join", SRC / "pc2_join.c"),
        ("pc_compress", OLD / "pc_compress.c"),
    ):
        exe = d / name
        run("gcc", "-O3", "-march=native", "-fopenmp", source, "-lm", "-o", exe)
        out[name] = exe
    return out


def board_classes(census_id):
    entry = read_json(ROOT / f"census/{census_id}.json")
    reps = [read_json(ROOT / f"quadruples/{q}.json")["rows"] for q in entry["representatives"]]
    return entry, [canonical(r) for r in reps]


@pytest.mark.parametrize("pattern,n,d1,d2,census_id,threads", CASES)
def test_classes_equal_board_census(engines, tmp_path, pattern, n, d1, d2, census_id, threads):
    raw, inst, out = tmp_path / "inst.raw", tmp_path / "inst.txt", tmp_path / "rows.txt"
    run(engines["inst"], pattern, n, d1, 2, raw)
    run(engines["join"], "sort", pattern, n, d1, d2, raw, inst)
    assert sorted(inst.read_text().splitlines()) == sorted(raw.read_text().splitlines())
    run(engines["join"], pattern, n, d1, d2, inst, threads, out)
    lines = sorted(set(out.read_text().splitlines()))
    entry, reps = board_classes(census_id)
    classes = set()
    for line in lines:
        q = validate_rows([[1 if c == "+" else -1 for c in w] for w in line.split()], n)
        assert entry["stratum"] in memberships(q)
        classes.add(canonical(q))
    assert classes == set(reps) and len(classes) == entry["claimed_count"]

    engine = STRATUM_ENGINE[entry["stratum"]]
    report = {
        "version": 1,
        "entry_id": census_id,
        "entry_sha256": digest(ROOT / f"census/{census_id}.json"),
        "engine_id": engine,
        "engine_sha256": census_replays.engine_digests(engine),
        "mode": "full",
        "status": "passed",
        "rows": lines,
        "compression": [d1, d2],
    }
    result = {"checks": []}
    census_replays.apply_report(entry, result, report, report["entry_sha256"], reps)
    assert (result["bound"], result["tier"]) == ("exact", "reproduced")
    for bad in (d1, [d1, d1], [d1 * d2, d2], [d1, d2, 1]):
        report["compression"] = bad
        with pytest.raises(ValueError):
            census_replays.apply_report(entry, {"checks": []}, report, report["entry_sha256"], reps)


def test_line_ranges_budgets_and_stop_file(engines, tmp_path):
    """Line ranges tile the run, tiny memory budgets (match passes, pair shards, cache
    clearing) and two threads write the same quadruples, and a STOP file stops before
    the next line with exit 3."""
    pattern, n, d1, d2 = "ksss", 45, 5, 3
    raw, inst = tmp_path / "inst.raw", tmp_path / "inst.txt"
    run(engines["inst"], pattern, n, d1, 2, raw)
    run(engines["join"], "sort", pattern, n, d1, d2, raw, inst)
    total = len(inst.read_text().splitlines())
    whole = tmp_path / "whole.txt"
    run(engines["join"], pattern, n, d1, d2, inst, 1, whole)
    parts = tmp_path / "parts.txt"
    cut = total // 3
    run(engines["join"], pattern, n, d1, d2, inst, 1, parts, 0, cut - 1)
    run(engines["join"], pattern, n, d1, d2, inst, 1, parts, cut, total - 1)
    assert sorted(set(parts.read_text().splitlines())) == sorted(set(whole.read_text().splitlines()))
    small = tmp_path / "small.txt"
    tight = {"PC_MEM_MB": "0.05", "PC_PAIR_MB": "0.07", "PC_CACHE_MB": "0.5", "PC_ORBIT_MB": "0.1", "PC_ORBIT_CAP": "4"}
    run(engines["join"], pattern, n, d1, d2, inst, 2, small, env=tight)
    assert sorted(set(small.read_text().splitlines())) == sorted(set(whole.read_text().splitlines()))
    stop = tmp_path / "STOP"
    stop.touch()
    p = subprocess.run(
        [str(engines["join"]), pattern, str(n), str(d1), str(d2), str(inst), "1", str(tmp_path / "x.txt")],
        capture_output=True,
        text=True,
        env={**os.environ, "PC_STOP": str(stop)},
        timeout=120,
    )
    assert p.returncode == 3


PINNED_CASES = (("ksss", 45, 3), ("ksss", 45, 5), ("ssss", 45, 3), ("kkss", 45, 5), ("kkks", 21, 7), ("ssss", 35, 7))


@pytest.mark.parametrize("pattern,n,d", PINNED_CASES)
def test_instances_equal_pinned_compression(engines, tmp_path, pattern, n, d):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    run(engines["inst"], pattern, n, d, 2, a)
    run(engines["pc_compress"], "all", pattern, n, d, 2, b)
    assert sorted(a.read_text().splitlines()) == sorted(b.read_text().splitlines())
