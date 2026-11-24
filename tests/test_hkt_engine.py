"""The shared HKT census engine (research/starter/hkt) at small orders.

Each case runs hkt_census.c for one family and compares the equiv-v1 classes
of its rows with the board's census of the cell, then passes the rows through
the registered-replay check of the engine for that stratum. Runs split into A
parts, B pieces and slice ranges, by hand and through run_census.sh under a
memory and pair budget, must write the same rows and count the same pairs as
the whole run.
"""

import itertools
import os
import re
import subprocess

import pytest

from verify import census_replays
from verify.common import ROOT, digest, read_json
from verify.equivalence import canonical
from verify.gs_verify import memberships, validate_rows

SRC = ROOT / "research/starter/hkt"
CASES = (
    ("wil", 21, "williamson-n21"),
    ("wil", 25, "williamson-n25"),
    ("wil", 29, "williamson-n29"),
    ("wil", 33, "williamson-n33"),
    ("wil", 35, "williamson-n35"),
    ("wil", 37, "williamson-n37"),
    ("wil", 39, "williamson-n39"),
    ("good", 21, "good-n21"),
    ("good", 25, "good-n25"),
    ("good", 29, "good-n29"),
    ("good", 33, "good-n33"),
    ("good", 35, "good-n35"),
    ("good", 37, "good-n37"),
    ("good", 39, "good-n39"),
)
FAMILY_ENGINE = {"wil": "williamson-hkt-v4", "good": "good-hkt-v2"}


def run(*args, env=None):
    full = {k: v for k, v in os.environ.items() if not k.startswith("HKT_")}
    full.update(env or {})
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True, env=full, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:] + p.stdout[-2000:]
    return p


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    exe = tmp_path_factory.mktemp("hkt") / "hkt_census"
    run(
        "gcc",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-DHKT_WIDE",
        "-I",
        ROOT / "research/starter/kkss",
        SRC / "hkt_census.c",
        "-lm",
        "-o",
        exe,
    )
    return exe


def lines(path):
    return sorted(path.read_text().splitlines())


def unit(stdout):
    m = re.search(r"^UNIT .* pairs=(\d+) distinct=(\d+) capped=(\d+)", stdout, re.M)
    assert m, stdout[-2000:]
    return int(m[1]), int(m[3])


@pytest.mark.parametrize("family,n,census_id", CASES)
def test_classes_equal_board_census(engine, tmp_path, family, n, census_id):
    out = tmp_path / "rows.txt"
    p = run(engine, family, n, 2, out)
    assert unit(p.stdout)[1] == 0
    rows = lines(out)
    entry = read_json(ROOT / f"census/{census_id}.json")
    reps = [canonical(read_json(ROOT / f"quadruples/{q}.json")["rows"]) for q in entry["representatives"]]
    classes = set()
    for line in rows:
        q = validate_rows([[1 if c == "+" else -1 for c in w] for w in line.split()], n)
        assert entry["stratum"] in memberships(q)
        classes.add(canonical(q))
    assert classes == set(reps) and len(classes) == entry["claimed_count"]

    engine_id = FAMILY_ENGINE[family]
    report = {
        "version": 1,
        "entry_id": census_id,
        "entry_sha256": digest(ROOT / f"census/{census_id}.json"),
        "engine_id": engine_id,
        "engine_sha256": census_replays.engine_digests(engine_id),
        "mode": "full",
        "status": "passed",
        "rows": rows,
    }
    result = {"checks": []}
    census_replays.apply_report(entry, result, report, report["entry_sha256"], reps)
    assert (result["bound"], result["tier"]) == ("exact", "reproduced")


@pytest.mark.parametrize("family,n", (("wil", 37), ("good", 37)))
def test_split_units_write_the_whole_run(engine, tmp_path, family, n):
    whole = tmp_path / "whole.txt"
    pairs = unit(run(engine, family, n, 2, whole).stdout)[0]
    groups = len(re.findall(r"^   group \d+: \(", run(engine, family, n, 1, env={"HKT_LIST": "1"}).stdout, re.M))
    assert groups >= 1
    rows, split_pairs = set(), 0
    for g, j, i, k in itertools.product(range(groups), range(2), range(3), range(2)):
        part = tmp_path / f"u{g}{j}{i}{k}.txt"
        env = {
            "HKT_DONLY": str(g),
            "HKT_APARTS": "2",
            "HKT_APART": str(j),
            "HKT_BPIECES": "3",
            "HKT_BPIECE": str(i),
            "HKT_SLICES": "2",
            "HKT_SLICE": str(k),
        }
        split_pairs += unit(run(engine, family, n, 2, part, env=env).stdout)[0]
        rows |= set(lines(part))
    assert split_pairs == pairs
    assert sorted(rows) == lines(whole)


def test_driver_plan_split_and_stop(tmp_path):
    out = tmp_path / "w41"
    env = {"OUT": str(out), "MEM_MB": "96", "UNIT_PAIRS": "1e6"}
    run("bash", SRC / "run_census.sh", "wil", 41, 2, env=env)
    plan = (out / "plan.txt").read_text()
    # the small budgets force several B pieces and slice ranges
    assert "bpieces=2" in plan and "slices=2" in plan
    assert len(list(out.glob("*.done"))) > 3
    assert "nullity-capped solves 0" in (out / "total.txt").read_text()
    whole = tmp_path / "whole.txt"
    run(
        "bash",
        SRC / "run_census.sh",
        "wil",
        41,
        2,
        env={"OUT": str(tmp_path / "one"), "APARTS": "1", "BPIECES": "1", "SLICES": "1"},
    )
    run(tmp_path / "one" / "bin" / os.uname().nodename / "hkt_census", "wil", 41, 2, whole)
    assert lines(out / "rows.txt") == lines(whole) == lines(tmp_path / "one" / "rows.txt")
    # a STOP file stops the driver before its next unit
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "STOP").touch()
    p = subprocess.run(
        ["bash", str(SRC / "run_census.sh"), "good", "21", "1"],
        capture_output=True,
        text=True,
        env={**os.environ, "OUT": str(tmp_path / "s")},
    )
    assert p.returncode == 3 and not list((tmp_path / "s").glob("*.done"))
