"""Bounded full census replays. No submission-provided executable is accepted."""

import os
import re
import subprocess
import tempfile
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from math import gcd
from pathlib import Path
from typing import NotRequired, TypedDict

from verify.common import ROOT, TRACKS, ById, JSONObject, PathLike, Row, Rows
from verify.equivalence import CANONICAL, VERSION, VERSION_SHIFTS, branch_canonical, canonical
from verify.gs_verify import memberships, paf, validate_rows


@lru_cache(maxsize=16)
def small_exact(n: int, stratum: str) -> set[Rows]:
    if n == 1 and stratum in TRACKS:
        # The single row (1) is symmetric and skew, so the one quadruple at
        # n=1 belongs to all six strata.
        return {((1,),) * 4}
    if stratum != "good" or n not in range(3, 16, 2):
        raise ValueError("small-exact-v1 supports good odd orders <=15 and every stratum at n=1")
    h = n // 2
    skew: dict[tuple[int, ...], list[Row]] = {}
    symmetric: list[tuple[Row, tuple[int, ...], int]] = []
    for code in range(1 << h):
        half = tuple(1 if code & (1 << j) else -1 for j in range(h))
        a = (1, *half, *(-x for x in half[::-1]))
        b = (1, *half, *half[::-1])
        skew.setdefault(paf(a)[1 : h + 1], []).append(a)
        symmetric.append((b, paf(b)[1 : h + 1], sum(b) ** 2))
    found: set[Rows] = set()
    for sb, sc, sd in combinations_with_replacement(symmetric, 3):
        if sb[2] + sc[2] + sd[2] != 4 * n - 1:
            continue
        need = tuple(-sum(x) for x in zip(sb[1], sc[1], sd[1], strict=True))
        for a in skew.get(need, ()):
            q = (a, sb[0], sc[0], sd[0])
            validate_rows(q)
            found.add(canonical(q))
    return found


# The even orders the Williamson enumerator below replays inline: order 20 takes about two seconds, and the
# pair table grows fourfold with each step of two.
WILLIAMSON_EVEN_ORDERS = range(2, 21, 2)


@lru_cache(maxsize=len(WILLIAMSON_EVEN_ORDERS))
def williamson_even(n: int) -> set[Rows]:
    """Every equiv-v1 class of Williamson quadruples at an even order n, from the definition.

    A symmetric row with first entry +1 is fixed by its entries at 1 to n/2, so there are 2^(n/2) of them.
    Every unordered pair of rows is filed under the sum of its two PAF vectors at shifts 1 to n/2, and two
    pairs form a quadruple exactly when their sums cancel. Nothing is pruned: no row-sum or spectral
    filter is applied, and the arithmetic is exact. Every class returned is rechecked from its rows.
    """
    if n not in WILLIAMSON_EVEN_ORDERS:
        raise ValueError(
            f"williamson-even-exact-v1 supports even orders {WILLIAMSON_EVEN_ORDERS[0]} to {WILLIAMSON_EVEN_ORDERS[-1]}"
        )
    h = n // 2
    rows: list[tuple[Row, tuple[int, ...]]] = []
    for code in range(1 << h):
        half = tuple(1 if code & (1 << j) else -1 for j in range(h))
        row = (1, *half, *half[-2::-1])
        rows.append((row, paf(row)[1 : h + 1]))
    pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i, (_, first) in enumerate(rows):
        for j in range(i, len(rows)):
            pairs.setdefault(tuple(x + y for x, y in zip(first, rows[j][1], strict=True)), []).append((i, j))
    found: set[Rows] = set()
    for total, left in pairs.items():
        opposite = tuple(-x for x in total)
        # A sum and its negative are joined once; the zero sum is joined with itself.
        if opposite < total or opposite not in pairs:
            continue
        for i, j in left:
            for k, m in pairs[opposite]:
                found.add(canonical((rows[i][0], rows[j][0], rows[k][0], rows[m][0])))
    for q in found:
        validate_rows(q, n)
        if "williamson" not in memberships(q):
            raise ValueError("williamson-even-exact-v1 produced a row that is not symmetric")
    return found


@lru_cache(maxsize=8)
def best_census(n: int) -> tuple[set[Rows], dict[str, int]]:
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
        run = subprocess.run([str(exe), str(n), "2", str(out)], capture_output=True, text=True, timeout=240)
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
    expanded: set[Rows] = set()
    for q in branches:
        for t in range(1, n):
            if gcd(t, n) != 1:
                continue
            images = [tuple(r[(t * j) % n] for j in range(n)) for r in q]
            for abc in permutations(images[:3]):
                for sign in (-1, 1):
                    expanded.add((*abc, tuple(sign * x for x in images[3])))
    return reps, {"branch_orbits": len(branches), "expanded_quadruples": len(expanded)}


# Pinned C enumerators replayed inline on every full verification. The order
# ranges are a time budget: the last order of each takes under a minute on two
# threads, and the next one takes several minutes.
class CBuild(TypedDict):
    """How to compile a pinned C census: preprocessor flags and the source path."""

    flags: tuple[str, ...]
    source: str


class CEngine(CBuild):
    """A pinned C census, the orders it replays and the environment prefixes it strips."""

    stratum: str
    orders: range
    env: tuple[str, ...]
    equivalence: NotRequired[str]
    count_orders: NotRequired[range]
    count_key: NotRequired[str]


C_ENGINES: dict[str, CEngine] = {
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
    # The unconstrained tracks are counted under equiv-v2, which also identifies
    # independent cyclic shifts; under equiv-v1 their counts explode (514 and
    # 152 classes at n=7 against 4 and 4) and would need an entry per class.
    "general-census-c-v1": {
        "stratum": "general",
        "source": "research/starter/general/gencensus.c",
        "flags": ("-DGEN_SHIFTCLASS",),
        "orders": range(3, 14, 2),
        "env": ("GEN_",),
        "equivalence": VERSION_SHIFTS,
        "count_orders": range(15, 20, 2),
        "count_key": "classes_all",
    },
    "skew-sds-census-c-v1": {
        "stratum": "skew-sds",
        "source": "research/starter/general/gencensus.c",
        "flags": ("-DGEN_SHIFTCLASS", "-DGEN_SKEW"),
        "orders": range(3, 14, 2),
        "env": ("GEN_",),
        "equivalence": VERSION_SHIFTS,
        "count_orders": range(15, 20, 2),
        "count_key": "classes_skew",
    },
}


def engine_equivalence(engine: str) -> str:
    return C_ENGINES[engine].get("equivalence", VERSION)


# The count-only build of gencensus.c. It runs the same enumeration of PAF
# 4-tuples as the row-writing build and counts the classes of each product by
# Burnside's lemma instead of writing them, so one run yields the general and
# the skew-sds totals.
COUNT_SOURCE = "research/starter/general/gencensus.c"
COUNT_FLAGS = ("-DGEN_SHIFTCLASS", "-DGEN_COUNT_ONLY")


@lru_cache(maxsize=len(C_ENGINES) + 1)
def c_exe(engine: str) -> Path:
    """Build a pinned C census once per process; several censuses replay it."""
    spec: CBuild = {"flags": COUNT_FLAGS, "source": COUNT_SOURCE} if engine == "count" else C_ENGINES[engine]
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


def c_census(engine: str, n: int) -> set[Rows]:
    """Replay a pinned C census at odd order n and return its classes.

    Classes are taken under the engine's declared equivalence version.

    The engines' environment variables only tune performance, but any with the
    engine's prefixes are stripped so that a shard setting inherited from a
    campaign shell cannot narrow a replay.
    """
    spec = C_ENGINES[engine]
    if n not in spec["orders"]:
        raise ValueError(f"{engine} supports odd orders {spec['orders'][0]} to {spec['orders'][-1]}")
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
    canonical_form = CANONICAL[engine_equivalence(engine)]
    return {canonical_form(q) for q in rows}


@lru_cache(maxsize=8)
def shift_class_totals(n: int) -> dict[str, int]:
    """Run the count-only build at order n and return its two class totals."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEN_")}
    with tempfile.TemporaryDirectory(prefix="gs-census-") as tmp:
        run = subprocess.run(
            [str(c_exe("count")), str(n), "2", str(Path(tmp) / "empty.txt")],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=tmp,
            env=env,
        )
    found = re.findall(rf"^n={n} .*classes_all=(\d+) classes_skew=(\d+) ", run.stderr, re.M)
    if run.returncode or len(found) != 1:
        raise ValueError("count-only census replay failed: " + run.stderr[-2000:])
    return {"classes_all": int(found[0][0]), "classes_skew": int(found[0][1])}


def c_census_count(engine: str, n: int) -> int:
    """Class count of a pinned engine at an order where it counts without writing rows."""
    spec = C_ENGINES[engine]
    if n not in spec.get("count_orders", ()):
        raise ValueError(f"{engine} has no count-only replay at order {n}")
    return shift_class_totals(n)[spec["count_key"]]


def verify_count_only(entry: JSONObject, reproduce: bool) -> JSONObject:
    """A census that states a class count and stores no representatives.

    The count is exact only when the pinned engine's count-only build returns
    it here. The classes themselves are not re-derived by the board's canonical
    form at these orders; the two agree at every order where rows are written.
    """
    engine = entry.get("enumerator")
    if entry["representatives"] or entry["claim_kind"] != "computed-exact":
        raise ValueError("a count-only census is computed-exact and lists no representatives")
    if engine not in C_ENGINES or C_ENGINES[engine]["stratum"] != entry["stratum"]:
        raise ValueError("no trusted enumerator for this census")
    if engine_equivalence(engine) != entry["equivalence"]:
        raise ValueError("census equivalence differs from the enumerator's")
    if entry["n"] not in C_ENGINES[engine].get("count_orders", ()):
        raise ValueError(f"{engine} has no count-only replay at order {entry['n']}")
    result = {
        "id": entry["id"],
        "type": "census",
        "n": entry["n"],
        "stratum": entry["stratum"],
        "equivalence": entry["equivalence"],
        "count": entry["claimed_count"],
        "bound": "claimed",
        "tier": None,
        "claimed_count": entry["claimed_count"],
        "claim_kind": entry["claim_kind"],
        "count_only": True,
        "checks": [],
    }
    if reproduce:
        if c_census_count(engine, entry["n"]) != entry["claimed_count"]:
            raise ValueError("count-only census replay disagrees with the claimed count")
        result.update(bound="exact", tier="reproduced")
        result["checks"].append("full-enumeration-count-replayed")
        result["limitations"] = [
            "The count is the pinned engine's Burnside total over its exhaustive enumeration; "
            "no representatives are stored and the classes are not re-derived here."
        ]
    return result


def good_census(n: int) -> set[Rows]:
    return c_census("good-census-c-v1", n)


def williamson_census(n: int) -> set[Rows]:
    return c_census("williamson-census-c-v1", n)


def verify(entry: JSONObject, quadruples: ById, reproduce: bool = True, root: PathLike = ROOT) -> JSONObject:
    # Representatives are stored under equiv-v1; a census declared under
    # equiv-v2 compares them, and the enumerator's rows, under that version.
    if entry.get("catalogue"):
        from verify import catalogue

        return catalogue.verify(entry, quadruples, root, reproduce)
    if entry.get("published_classes"):
        raise ValueError("published_classes describes a catalogue")
    if entry.get("count_only"):
        return verify_count_only(entry, reproduce)
    canonical_form = CANONICAL[entry["equivalence"]]
    reps = []
    for identifier in entry["representatives"]:
        if identifier not in quadruples:
            raise ValueError(f"census references missing quadruple: {identifier}")
        q = quadruples[identifier]
        if q["n"] != entry["n"] or entry["stratum"] not in q["tracks"]:
            raise ValueError("census representative has wrong order or stratum")
        reps.append(canonical_form(q["canonical_rows"]))
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
        if engine in C_ENGINES and engine_equivalence(engine) != entry["equivalence"]:
            raise ValueError("census equivalence differs from the enumerator's")
        if engine not in C_ENGINES and entry["equivalence"] != VERSION:
            raise ValueError("no trusted enumerator for this census")
        if engine == "best-census-c-v1" and entry["stratum"] == "best":
            expected, secondary = best_census(entry["n"])
            result["secondary"] = secondary
        elif engine in C_ENGINES and entry["stratum"] == C_ENGINES[engine]["stratum"]:
            expected = c_census(engine, entry["n"])
        elif engine == "small-exact-v1":
            expected = small_exact(entry["n"], entry["stratum"])
        elif engine == "williamson-even-exact-v1" and entry["stratum"] == "williamson":
            expected = williamson_even(entry["n"])
        else:
            raise ValueError("no trusted enumerator for this census")
        if expected != unique or entry["claimed_count"] != len(expected):
            raise ValueError("exhaustive census disagrees with submitted representatives/count")
        result.update(bound="exact", tier="reproduced")
        result["checks"].append("full-enumeration-replayed")
    return result
