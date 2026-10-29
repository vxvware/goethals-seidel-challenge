"""Bounded full census replays. No submission-provided executable is accepted."""

from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from math import gcd
from pathlib import Path
import os
import subprocess
import tempfile
from verify.common import ROOT, TRACKS
from verify.equivalence import canonical, branch_canonical
from verify.gs_verify import memberships, paf, validate_rows


@lru_cache(maxsize=16)
def small_exact(n, stratum):
    if n == 1 and stratum in TRACKS:
        # The single row (1) is symmetric and skew, so the one quadruple at
        # n=1 belongs to all six strata.
        return {((1,),) * 4}
    if stratum != "good" or n not in range(3, 16, 2):
        raise ValueError("small-exact-v1 supports good odd orders <=15 and every stratum at n=1")
    h = n // 2
    skew = {}
    symmetric = []
    for code in range(1 << h):
        half = tuple(1 if code & (1 << j) else -1 for j in range(h))
        a = (1,) + half + tuple(-x for x in half[::-1])
        b = (1,) + half + half[::-1]
        skew.setdefault(paf(a)[1 : h + 1], []).append(a)
        symmetric.append((b, paf(b)[1 : h + 1], sum(b) ** 2))
    found = set()
    for b, c, d in combinations_with_replacement(symmetric, 3):
        if b[2] + c[2] + d[2] != 4 * n - 1:
            continue
        need = tuple(-sum(x) for x in zip(b[1], c[1], d[1]))
        for a in skew.get(need, ()):
            q = (a, b[0], c[0], d[0])
            validate_rows(q)
            found.add(canonical(q))
    return found


@lru_cache(maxsize=8)
def best_census(n):
    if n not in (3, 7, 13, 21, 31, 43):
        raise ValueError("best-census-c-v1 supports 3,7,13,21,31,43")
    source = ROOT / "research/starter/best/census.c"
    with tempfile.TemporaryDirectory(prefix="gs-census-") as tmp:
        exe = Path(tmp) / "census"
        out = Path(tmp) / "solutions.txt"
        build = subprocess.run(
            ["gcc", "-O3", "-march=native", "-fopenmp", str(source), "-lm", "-o", str(exe)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if build.returncode:
            raise ValueError("census compilation failed: " + build.stderr[-2000:])
        run = subprocess.run(
            [str(exe), str(n), "2", str(out)], capture_output=True, text=True, timeout=240
        )
        if run.returncode or not out.exists():
            raise ValueError("census replay failed: " + run.stderr[-2000:])
        rows = [
            tuple(tuple(1 if c == "+" else -1 for c in word) for word in line.split())
            for line in out.read_text().splitlines()
        ]
    for q in rows:
        validate_rows(q, n)
    reps = {canonical(q) for q in rows}
    branches = {branch_canonical(q) for q in rows}
    expanded = set()
    for q in branches:
        for t in range(1, n):
            if gcd(t, n) != 1:
                continue
            images = [tuple(r[(t * j) % n] for j in range(n)) for r in q]
            for abc in permutations(images[:3]):
                for sign in (-1, 1):
                    expanded.add(abc + (tuple(sign * x for x in images[3]),))
    return reps, {"branch_orbits": len(branches), "expanded_quadruples": len(expanded)}


# Pinned C enumerators replayed inline on every full verification. The order
# ranges are a time budget: the last order of each takes under a minute on two
# threads, and the next one takes several minutes.
C_ENGINES = {
    "good-census-c-v1": {
        "stratum": "good",
        "source": "research/starter/good/good_c.c",
        "flags": ("-DGS_WIDE", "-DGC_WIDE"),
        "orders": range(3, 44, 2),
        "env": ("GOOD_", "TILE_"),
    },
    "williamson-census-c-v1": {
        "stratum": "williamson",
        "source": "research/starter/williamson/williamson_c.c",
        "flags": (),
        "orders": range(3, 38, 2),
        "env": (),
    },
}


@lru_cache(maxsize=len(C_ENGINES))
def c_exe(engine):
    """Build a pinned C census once per process; several censuses replay it."""
    spec = C_ENGINES[engine]
    exe = Path(tempfile.mkdtemp(prefix="gs-census-build-")) / "census"
    build = subprocess.run(
        [
            "gcc",
            "-O3",
            "-march=native",
            "-fopenmp",
            *spec["flags"],
            str(ROOT / spec["source"]),
            "-lm",
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if build.returncode:
        raise ValueError(f"{engine} compilation failed: " + build.stderr[-2000:])
    return exe


def c_census(engine, n):
    """Replay a pinned C census at odd order n and return its equiv-v1 classes.

    The engines' environment variables only tune performance, but any with the
    engine's prefixes are stripped so that a shard setting inherited from a
    campaign shell cannot narrow a replay.
    """
    spec = C_ENGINES[engine]
    if n not in spec["orders"]:
        raise ValueError(
            f"{engine} supports odd orders {spec['orders'][0]} to {spec['orders'][-1]}"
        )
    env = {k: v for k, v in os.environ.items() if not k.startswith(spec["env"] or ("\0",))}
    with tempfile.TemporaryDirectory(prefix="gs-census-") as tmp:
        out = Path(tmp) / "solutions.txt"
        run = subprocess.run(
            [str(c_exe(engine)), str(n), "2", str(out)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=tmp,
            env=env,
        )
        if run.returncode or not out.exists():
            raise ValueError(f"{engine} replay failed: " + run.stderr[-2000:])
        rows = [
            tuple(tuple(1 if c == "+" else -1 for c in word) for word in line.split())
            for line in out.read_text().splitlines()
            if line and line[0] in "+-"
        ]
    for q in rows:
        validate_rows(q, n)
        if spec["stratum"] not in memberships(q):
            raise ValueError(f"{engine} produced a row outside the {spec['stratum']} stratum")
    return {canonical(q) for q in rows}


def good_census(n):
    return c_census("good-census-c-v1", n)


def williamson_census(n):
    return c_census("williamson-census-c-v1", n)


def verify(entry, quadruples, reproduce=True):
    reps = []
    for identifier in entry["representatives"]:
        if identifier not in quadruples:
            raise ValueError(f"census references missing quadruple: {identifier}")
        q = quadruples[identifier]
        if q["n"] != entry["n"] or entry["stratum"] not in q["tracks"]:
            raise ValueError("census representative has wrong order or stratum")
        reps.append(q["canonical_rows"])
    unique = set(reps)
    if len(unique) != len(reps):
        raise ValueError("census repeats equivalent representatives")
    if entry["claimed_count"] < len(unique):
        raise ValueError("claimed census count is smaller than verified representatives")
    result = {
        "id": entry["id"],
        "type": "census",
        "n": entry["n"],
        "stratum": entry["stratum"],
        "equivalence": entry["equivalence"],
        "count": len(unique),
        "bound": "lower",
        "tier": None,
        "claimed_count": entry["claimed_count"],
        "claim_kind": entry["claim_kind"],
        "checks": ["representative-existence", "pairwise-inequivalence"],
    }
    if entry["claim_kind"] == "computed-exact" and reproduce:
        engine = entry.get("enumerator")
        if engine == "best-census-c-v1" and entry["stratum"] == "best":
            expected, secondary = best_census(entry["n"])
            result["secondary"] = secondary
        elif engine in C_ENGINES and entry["stratum"] == C_ENGINES[engine]["stratum"]:
            expected = c_census(engine, entry["n"])
        elif engine == "small-exact-v1":
            expected = small_exact(entry["n"], entry["stratum"])
        else:
            raise ValueError("no trusted enumerator for this census")
        if expected != unique or entry["claimed_count"] != len(expected):
            raise ValueError("exhaustive census disagrees with submitted representatives/count")
        result.update(bound="exact", tier="reproduced")
        result["checks"].append("full-enumeration-replayed")
    return result
