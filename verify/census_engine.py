"""Bounded full census replays. No submission-provided executable is accepted."""

from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from math import gcd
from pathlib import Path
import subprocess
import tempfile
from verify.common import ROOT
from verify.equivalence import canonical, branch_canonical
from verify.gs_verify import paf, validate_rows


@lru_cache(maxsize=16)
def small_exact(n, stratum):
    if stratum != "good" or n not in range(1, 16, 2):
        raise ValueError("small-exact-v1 supports good odd orders <=15")
    if n == 1:
        return {((1,),) * 4}
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
        elif engine == "small-exact-v1":
            expected = small_exact(entry["n"], entry["stratum"])
        else:
            raise ValueError("no trusted enumerator for this census")
        if expected != unique or entry["claimed_count"] != len(expected):
            raise ValueError("exhaustive census disagrees with submitted representatives/count")
        result.update(bound="exact", tier="reproduced")
        result["checks"].append("full-enumeration-replayed")
    return result
