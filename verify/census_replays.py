"""Apply maintainer-registered full census replays of pinned offline engines.

A census whose order is too slow to enumerate inside the verifier can still be
exact on the board when a maintainer has run a pinned engine to completion
offline and registered the result. The registered report carries the raw
quadruples the engine wrote, and this module re-derives their equivalence
classes with the board's own canonical form on every load, so the only trust
placed in the report is that the engine ran to completion on the stated
sources; the count and the class set are recomputed here, never copied.
"""

from verify.common import ROOT, artifact, digest, read_json
from verify.equivalence import canonical
from verify.gs_verify import memberships, validate_rows

ENGINES = {
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
    "kkss-census-c-v1": {
        "stratum": "kkss",
        "sources": (
            "research/starter/kkss/gmat_c.c",
            "research/starter/kkss/hkt_solve.h",
        ),
        "orders": range(3, 64, 2),
        "composite": False,
    },
}
REPORT_DIR = "evidence/census-replays/"


def engine_digests(engine, root=ROOT):
    return {path: digest(root / path) for path in ENGINES[engine]["sources"]}


def replayed_classes(report, n, stratum):
    """Canonical classes of the report's raw quadruples, each exactly rechecked."""
    classes = set()
    for line in report["rows"]:
        words = line.split()
        if len(words) != 4 or any(len(w) != n or set(w) - set("+-") for w in words):
            raise ValueError("registered replay row is not four +/- rows of the order")
        rows = validate_rows([[1 if ch == "+" else -1 for ch in w] for w in words], n)
        if stratum not in memberships(rows):
            raise ValueError("registered replay row is outside the census stratum")
        classes.add(canonical(rows))
    return classes


def apply_report(entry, result, report, entry_sha256, representatives, root=ROOT):
    if report.get("entry_id") != entry["id"] or report.get("entry_sha256") != entry_sha256:
        raise ValueError("registered census replay is stale or bound to a different entry")
    engine = report.get("engine_id")
    if engine not in ENGINES:
        raise ValueError("unknown census replay engine")
    spec = ENGINES[engine]
    n = entry["n"]
    if entry["stratum"] != spec["stratum"] or n not in spec["orders"]:
        raise ValueError("census replay engine does not cover this cell")
    if report.get("engine_sha256") != engine_digests(engine, root):
        raise ValueError("registered census replay engine differs from the pinned sources")
    if spec["composite"]:
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


def apply_registered(documents, results, root=ROOT):
    registry = read_json(root / "baselines/census_replays.json")
    if registry.get("version") != 1:
        raise ValueError("unsupported census replay registry version")
    seen = set()
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
