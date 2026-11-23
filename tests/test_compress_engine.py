"""The compression census engine (research/starter/compress) at small orders.

Each case compresses, lifts with both builds of pc_lift.c, and compares the
equiv-v1 classes of the lifted rows with the board's census of the cell. The
staged path (enumeration pieces, merge, match shards with forced passes) and the
sparse count table must write the same instances as the one-shot run.
"""

import os
import subprocess

import pytest

from verify import census_replays
from verify.common import ROOT, digest, read_json
from verify.equivalence import canonical
from verify.gs_verify import memberships, validate_rows

SRC = ROOT / "research/starter/compress"
CASES = (
    ("ksss", 21, 3, "good-n21"),
    ("ksss", 25, 5, "good-n25"),
    ("ksss", 27, 9, "good-n27"),
    ("ksss", 33, 11, "good-n33"),
    ("ksss", 35, 5, "good-n35"),
    ("ksss", 39, 3, "good-n39"),
    ("kkss", 21, 7, "vxvware-kkss-n21"),
    ("kkss", 25, 5, "vxvware-kkss-n25"),
    ("kkss", 33, 3, "vxvware-kkss-n33"),
    ("kkss", 35, 5, "vxvware-kkss-n35"),
    ("kkss", 39, 3, "vxvware-kkss-n39"),
    ("kkks", 21, 3, "best-n21"),
    ("kkks", 21, 7, "best-n21"),
    ("ssss", 25, 5, "williamson-n25"),
    ("ssss", 27, 3, "williamson-n27"),
    ("ssss", 33, 3, "williamson-n33"),
    ("ssss", 35, 5, "williamson-n35"),
    ("ssss", 39, 3, "williamson-n39"),
)
STRATUM_ENGINE = {
    "good": "good-compress-v1",
    "kkss": "kkss-compress-v1",
    "best": "best-compress-v1",
    "williamson": "williamson-compress-v3",
}


def run(*args, env=None):
    full = {k: v for k, v in os.environ.items() if not k.startswith("PC_")}
    full.update(env or {})
    p = subprocess.run([str(a) for a in args], capture_output=True, text=True, env=full, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:] + p.stdout[-2000:]
    return p


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    d = tmp_path_factory.mktemp("compress")
    out = {}
    for name, source, flags in (
        ("compress", "pc_compress.c", ()),
        ("lift", "pc_lift.c", ()),
        ("lift_wide", "pc_lift.c", ("-DPC_WIDE",)),
    ):
        exe = d / name
        run("gcc", "-O3", "-march=native", "-fopenmp", *flags, SRC / source, "-lm", "-o", exe)
        out[name] = exe
    return out


def lines(path):
    return sorted(path.read_text().splitlines())


def board_classes(census_id):
    entry = read_json(ROOT / f"census/{census_id}.json")
    reps = [read_json(ROOT / f"quadruples/{q}.json")["rows"] for q in entry["representatives"]]
    return entry, [canonical(r) for r in reps]


@pytest.mark.parametrize("pattern,n,d,census_id", CASES)
def test_classes_equal_board_census(engines, tmp_path, pattern, n, d, census_id):
    inst = tmp_path / "inst.txt"
    run(engines["compress"], "all", pattern, n, d, 2, inst)
    rows = {}
    for build in ("lift", "lift_wide"):
        out = tmp_path / f"{build}.txt"
        run(engines[build], n, d, inst, 2, out)
        rows[build] = lines(out)
    assert rows["lift"] == rows["lift_wide"]
    entry, reps = board_classes(census_id)
    classes = set()
    for line in rows["lift"]:
        q = validate_rows([[1 if c == "+" else -1 for c in w] for w in line.split()], n)
        assert entry["stratum"] in memberships(q)
        classes.add(canonical(q))
    assert classes == set(reps) and len(classes) == entry["claimed_count"]

    # the same rows pass the registered-replay check of the engine for this stratum
    engine = STRATUM_ENGINE[entry["stratum"]]
    report = {
        "version": 1,
        "entry_id": census_id,
        "entry_sha256": digest(ROOT / f"census/{census_id}.json"),
        "engine_id": engine,
        "engine_sha256": census_replays.engine_digests(engine),
        "mode": "full",
        "status": "passed",
        "rows": rows["lift"],
        "compression": d,
    }
    result = {"checks": []}
    census_replays.apply_report(entry, result, report, report["entry_sha256"], reps)
    assert (result["bound"], result["tier"]) == ("exact", "reproduced")


def test_staged_and_sparse_runs_write_the_same_instances(engines, tmp_path):
    pattern, n, d = "ksss", 33, 3
    one = tmp_path / "one.txt"
    run(engines["compress"], "all", pattern, n, d, 2, one)
    pieces = []
    for i in range(3):
        pieces.append(tmp_path / f"lists-{i}.txt")
        run(engines["compress"], "enum", pattern, n, d, 2, pieces[-1], i, 3)
    merged = tmp_path / "lists.txt"
    with pytest.raises(AssertionError):  # pieces that leave a gap are refused
        run(engines["compress"], "merge", merged, *pieces[:2])
    run(engines["compress"], "merge", merged, *pieces)
    staged = []
    for k in range(2):
        part = tmp_path / f"inst-{k}.txt"
        p = run(engines["compress"], "match", merged, 2, part, k, 2, env={"PC_MEM_MB": "0.005"})
        assert " 1 passes" not in p.stdout  # the small budget forces several passes
        staged += lines(part)
    assert sorted(staged) == lines(one)
    sparse = tmp_path / "sparse.txt"
    run(engines["compress"], "all", pattern, n, d, 2, sparse, env={"PC_SPARSE": "1"})
    assert lines(sparse) == lines(one)


# Table 1 of bright2019good (baselines/references.bib): compressed good
# quadruples by 3, one per equivalence class, at the orders divisible by 3.
PUBLISHED_GOOD_INSTANCES = {9: 2, 15: 11, 21: 39, 27: 186, 33: 840, 39: 1934, 45: 19205}


@pytest.mark.parametrize("n,count", sorted(PUBLISHED_GOOD_INSTANCES.items()))
def test_good_instance_counts_equal_published(engines, tmp_path, n, count):
    inst = tmp_path / "inst.txt"
    run(engines["compress"], "all", "ksss", n, 3, 2, inst)
    assert len(inst.read_text().splitlines()) == count
