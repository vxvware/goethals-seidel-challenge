"""Apply maintainer-registered full census replays of pinned offline engines.

A census whose order is too slow to enumerate inside the verifier can still be
exact on the board when a maintainer has run a pinned engine to completion
offline and registered the result. The registered report carries the raw
quadruples the engine wrote, and this module re-derives their equivalence
classes with the board's own canonical form on every load, so the only trust
placed in the report is that the engine ran to completion on the stated
sources; the count and the class set are recomputed here, never copied.
"""

from math import gcd
from pathlib import Path
from typing import NotRequired, TypedDict

from verify.common import ROOT, ById, JSONObject, Rows, artifact, digest, read_json
from verify.equivalence import canonical
from verify.gs_verify import memberships, validate_rows

# pc_compress.c writes the compressed instances of one block pattern and
# pc_lift.c lifts them back; the stratum fixes the pattern (ssss, ksss, kkss,
# kkks), and a report records the divisor d of the order it compressed by.
COMPRESS_SOURCES = (
    "research/starter/compress/pc_compress.c",
    "research/starter/compress/pc_lift.c",
)
# pc2_inst.c writes the instances at one compression length and pc2_join.c
# lifts them through a second, coprime one; a report records both divisors.
COMPRESS2_SOURCES = (
    "research/starter/compress2/pc2_inst.c",
    "research/starter/compress2/pc2_join.c",
)


# hkt_census.c runs both HKT censuses with a family argument (wil or good)
# on the wide solver header; run_census.sh drives it as restartable units.
HKT_SOURCES = (
    "research/starter/hkt/hkt_census.c",
    "research/starter/kkss/hkt_solve64.h",
)


class ReplayEngine(TypedDict):
    """A pinned offline census engine: its stratum, sources, orders and whether it compresses."""

    stratum: str
    sources: tuple[str, ...]
    orders: range | tuple[int, ...]
    composite: bool
    divisors: NotRequired[int]


ENGINES: dict[str, ReplayEngine] = {
    "williamson-census-c-v1": {
        "stratum": "williamson",
        "sources": ("research/starter/williamson/williamson_c.c",),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "williamson-compress-v1": {
        "stratum": "williamson",
        "sources": (
            "research/starter/williamson/wc_compress.c",
            "research/starter/williamson/wc_lift.c",
        ),
        "orders": range(9, 66, 2),
        "composite": True,
    },
    "williamson-compress-v2": {
        "stratum": "williamson",
        "sources": (
            "research/starter/williamson/wc_compress.c",
            "research/starter/williamson/wc_lift.c",
            "research/starter/williamson/wc_lift5.c",
        ),
        "orders": range(9, 66, 2),
        "composite": True,
    },
    "williamson-hkt-v1": {
        "stratum": "williamson",
        "sources": (
            "research/starter/williamson/wil_hkt.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "williamson-hkt-v2": {
        "stratum": "williamson",
        "sources": (
            "research/starter/williamson/wil_hkt2.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "williamson-hkt-v3": {
        "stratum": "williamson",
        "sources": (
            "research/starter/williamson/wil_hkt3.c",
            "research/starter/kkss/hkt_solve64.h",
        ),
        "orders": range(3, 128, 2),
        "composite": False,
    },
    "kkss-census-c-v1": {
        "stratum": "kkss",
        "sources": (
            "research/starter/kkss/gmat_c.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "kkss-census-c-v2": {
        "stratum": "kkss",
        "sources": (
            "research/starter/kkss/gmat_c_v2.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "kkss-census-c-v3": {
        "stratum": "kkss",
        "sources": (
            "research/starter/kkss/gmat_c_v3.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "best-census-hkt-v1": {
        "stratum": "best",
        "sources": ("research/starter/best/best_hkt.c",),
        "orders": (3, 7, 13, 21, 31, 43, 57),
        "composite": False,
    },
    "best-census-hkt-v2": {
        "stratum": "best",
        "sources": ("research/starter/best/best_hkt2.c",),
        "orders": (3, 7, 13, 21, 31, 43, 57, 73),
        "composite": False,
    },
    "good-census-c-v1": {
        "stratum": "good",
        "sources": ("research/starter/good/good_c.c",),
        "orders": range(3, 68, 2),
        "composite": False,
    },
    "good-hkt-v1": {
        "stratum": "good",
        "sources": (
            "research/starter/good/good_hkt.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
    "williamson-hkt-v4": {
        "stratum": "williamson",
        "sources": HKT_SOURCES,
        "orders": range(3, 128, 2),
        "composite": False,
    },
    "good-hkt-v2": {
        "stratum": "good",
        "sources": HKT_SOURCES,
        "orders": range(3, 128, 2),
        "composite": False,
    },
    "williamson-compress-v3": {
        "stratum": "williamson",
        "sources": COMPRESS_SOURCES,
        "orders": range(9, 126, 2),
        "composite": True,
    },
    "good-compress-v1": {
        "stratum": "good",
        "sources": COMPRESS_SOURCES,
        "orders": range(9, 126, 2),
        "composite": True,
    },
    "kkss-compress-v1": {
        "stratum": "kkss",
        "sources": COMPRESS_SOURCES,
        "orders": range(9, 126, 2),
        "composite": True,
    },
    "best-compress-v1": {
        "stratum": "best",
        "sources": COMPRESS_SOURCES,
        "orders": range(9, 126, 2),
        "composite": True,
    },
    "williamson-compress2-v1": {
        "stratum": "williamson",
        "sources": COMPRESS2_SOURCES,
        "orders": range(15, 128, 2),
        "composite": True,
        "divisors": 2,
    },
    "good-compress2-v1": {
        "stratum": "good",
        "sources": COMPRESS2_SOURCES,
        "orders": range(15, 128, 2),
        "composite": True,
        "divisors": 2,
    },
    "kkss-compress2-v1": {
        "stratum": "kkss",
        "sources": COMPRESS2_SOURCES,
        "orders": range(15, 128, 2),
        "composite": True,
        "divisors": 2,
    },
    "best-compress2-v1": {
        "stratum": "best",
        "sources": COMPRESS2_SOURCES,
        "orders": range(15, 128, 2),
        "composite": True,
        "divisors": 2,
    },
}
REPORT_DIR = "evidence/census-replays/"


def engine_digests(engine: str, root: Path = ROOT) -> dict[str, str]:
    return {path: digest(root / path) for path in ENGINES[engine]["sources"]}


def two_divisors(d: object, n: int) -> bool:
    """[d1, d2]: odd coprime divisors of n, at least 3, whose product divides n and leaves lengths n/d at least 3."""
    if type(d) is not list or len(d) != 2 or any(type(x) is not int for x in d):
        return False
    d1, d2 = d
    return (
        all(x >= 3 and x % 2 == 1 and n % x == 0 and n // x >= 3 for x in (d1, d2))
        and gcd(d1, d2) == 1
        and n % (d1 * d2) == 0
    )


def replayed_classes(report: JSONObject, n: int, stratum: str) -> set[Rows]:
    """Canonical classes of the report's raw quadruples, each exactly rechecked."""
    classes: set[Rows] = set()
    for line in report["rows"]:
        words = line.split()
        if len(words) != 4 or any(len(w) != n or set(w) - set("+-") for w in words):
            raise ValueError("registered replay row is not four +/- rows of the order")
        rows = validate_rows([[1 if ch == "+" else -1 for ch in w] for w in words], n)
        if stratum not in memberships(rows):
            raise ValueError("registered replay row is outside the census stratum")
        classes.add(canonical(rows))
    return classes


def apply_report(
    entry: JSONObject,
    result: JSONObject,
    report: JSONObject,
    entry_sha256: str,
    representatives: list[Rows],
    root: Path = ROOT,
) -> None:
    if report.get("entry_id") != entry["id"] or report.get("entry_sha256") != entry_sha256:
        raise ValueError("registered census replay is stale or bound to a different entry")
    engine = report.get("engine_id")
    if engine not in ENGINES:
        raise ValueError("unknown census replay engine")
    spec = ENGINES[engine]
    n = entry["n"]
    if entry["equivalence"] != "equiv-v1":
        raise ValueError("registered census replays are defined only under equiv-v1")
    if entry["stratum"] != spec["stratum"] or n not in spec["orders"]:
        raise ValueError("census replay engine does not cover this cell")
    if report.get("engine_sha256") != engine_digests(engine, root):
        raise ValueError("registered census replay engine differs from the pinned sources")
    if spec.get("divisors") == 2:
        if not two_divisors(report.get("compression"), n):
            raise ValueError("two-divisor compression replay must record coprime divisors d1, d2 with n/(d1 d2) whole")
    elif spec["composite"]:
        d = report.get("compression")
        if type(d) is not int or not 1 < d < n or n % d:
            raise ValueError("compression replay must record a proper divisor of the order")
    if report.get("status") != "passed" or report.get("mode") != "full":
        raise ValueError("registered census replay is not a passed full replay")
    if type(report.get("rows")) is not list or len(report["rows"]) > 65536:
        raise ValueError("registered census replay must list its raw rows")
    classes = replayed_classes(report, n, entry["stratum"])
    if classes != set(representatives) or entry["claimed_count"] != len(classes):
        raise ValueError("registered census replay disagrees with the representatives or count")
    result["checks"] = sorted(set(result["checks"] + ["registered-full-census-replay"]))
    result["replay"] = {
        key: report[key]
        for key in ("engine_id", "compression", "host", "threads", "date", "wall_seconds")
        if key in report
    }
    result["replay"]["raw_rows"] = len(report["rows"])
    result.update(bound="exact", tier="reproduced")
    result["limitations"] = [
        "Full offline replay of a pinned engine, re-derived from its raw rows here; "
        "the same engine can reproduce its own systematic bugs."
    ]


def apply_registered(documents: ById, results: ById, root: Path = ROOT) -> None:
    registry = read_json(root / "baselines/census_replays.json")
    if registry.get("version") != 1:
        raise ValueError("unsupported census replay registry version")
    seen: set[str] = set()
    for ref in registry["reports"]:
        if not ref["path"].startswith(REPORT_DIR):
            raise ValueError(f"registered census replay must live under {REPORT_DIR}")
        report = read_json(artifact(root, ref))
        ident = report.get("entry_id")
        if ident not in documents or documents[ident]["type"] != "census":
            raise ValueError("registered census replay refers to an absent census")
        if ident in seen:
            raise ValueError("census replay registered twice")
        seen.add(ident)
        entry = documents[ident]
        representatives = [results[q]["canonical_rows"] for q in entry["representatives"]]
        apply_report(
            entry,
            results[ident],
            report,
            digest(root / f"census/{ident}.json"),
            representatives,
            root,
        )
