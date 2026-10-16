"""Fail-closed receipt validation and small, exactly checked proof rules."""

from math import isqrt
from verify.common import artifact, read_json


def intervals_union(intervals, total):
    current = 0
    covered = 0
    gaps = []
    for start, end in sorted(intervals):
        if any(type(x) is not int for x in (start, end)) or not 0 <= start < end <= total:
            raise ValueError("invalid half-open coverage interval")
        if start < current:
            raise ValueError("overlapping coverage intervals")
        if start > current:
            gaps.append([current, start])
        covered += end - start
        current = end
    if current < total:
        gaps.append([current, total])
    return covered, gaps


def good_decompositions(n):
    # Origin-normalized symmetric rows have sum == n (mod 4).
    choices = [s if s % 4 == n % 4 else -s for s in range(1, isqrt(4 * n - 1) + 1, 2)]
    from itertools import combinations_with_replacement

    return [
        t for t in combinations_with_replacement(choices, 3) if sum(s * s for s in t) == 4 * n - 1
    ]


def expected_passes(n):
    return sorted({max(t, key=abs) for t in good_decompositions(n)})


def in_scope(rows, scope):
    if scope["kind"] == "full":
        return True
    n = len(rows[0])
    d = scope["subgroup_order"]
    subgroup = [x for x in range(1, n) if pow(x, d, n) == 1]
    if len(subgroup) != d:
        return False
    from verify.equivalence import normalize

    rows = [normalize(tuple(r)) for r in rows]
    return all(all(row[(h * j) % n] == row[j] for h in subgroup for j in range(n)) for row in rows)


def proof(entry):
    n = entry["n"]
    if entry["claim"] != "empty":
        raise ValueError("proof must assert empty in its stated scope")
    if entry["method"] == "row-sum-theorem":
        if entry["stratum"] != "best" or entry["scope"] != {"kind": "full"}:
            raise ValueError("best row-sum rule requires full best stratum")
        if isqrt(4 * n - 3) ** 2 == 4 * n - 3:
            raise ValueError("row-sum theorem does not rule out this order")
        return "3 + sum(D)^2 = 4n, but 4n-3 is not a square"
    scope = entry["scope"]
    if scope["kind"] != "same-subgroup-coset-unions" or entry["stratum"] != "skew-sds":
        raise ValueError("coset proof needs same-subgroup skew-sds scope")
    if n < 3 or any(n % d == 0 for d in range(2, isqrt(n) + 1)):
        raise ValueError("coset proof v1 requires prime n")
    d = scope["subgroup_order"]
    if d % 2 != 1 or (n - 1) % d:
        raise ValueError("subgroup must have odd order dividing n-1")
    zero = (0, 1) if scope["include_zero"] else (0,)
    sums = {n - 2 * (d * u + e) for u in range((n - 1) // d + 1) for e in zero}
    squares = {s * s for s in sums}
    pairs = {a + b for a in squares for b in squares}
    if any(4 * n - 1 - c in pairs for c in squares):
        raise ValueError("row-sum obstruction does not rule out this coset scope")
    return "No three admissible coset-block squared row sums add to 4n-1"


def verify_good_records(entry, root):
    passes = {}
    checks = ["artifact-sha256", "receipt-identity", "zero-pending", "coverage-intervals"]
    for ref in entry["records"]:
        rec = read_json(artifact(root, ref))
        for key in ("n", "sB", "start", "classes", "order_digest", "pending", "unsettled"):
            if key not in rec:
                raise ValueError(f"receipt missing required field {key}")
        if rec["n"] != entry["n"] or rec["pending"] != [] or rec["unsettled"] != 0:
            raise ValueError("receipt identity mismatch or unsettled work")
        if "end" not in rec and "done" not in rec:
            raise ValueError("receipt missing end/done cursor")
        if rec.get("solutions", rec.get("sols", 0)) != 0:
            raise ValueError("nonempty receipt cannot support a null claim")
        end = rec.get("end", rec.get("done"))
        s = rec["sB"]
        if s not in expected_passes(entry["n"]):
            raise ValueError("receipt has an inadmissible row-sum pass")
        identity = (rec["classes"], rec["order_digest"])
        if type(identity[0]) is not int or identity[0] <= 0:
            raise ValueError("invalid class count")
        import re

        if not isinstance(identity[1], str) or not re.fullmatch(r"[0-9a-f]{16}", identity[1]):
            raise ValueError("legacy good-engine order digest must be 16 hex characters")
        p = passes.setdefault(s, {"identity": identity, "intervals": []})
        if p["identity"] != identity:
            raise ValueError("class-order digest/count mismatch across receipt seam")
        p["intervals"].append([rec["start"], end])
    if not passes:
        raise ValueError("no coverage records")
    coverage = []
    complete = set(passes) == set(expected_passes(entry["n"]))
    for s, p in sorted(passes.items()):
        covered, gaps = intervals_union(p["intervals"], p["identity"][0])
        complete &= not gaps
        coverage.append(
            {
                "pass": s,
                "classes": p["identity"][0],
                "covered": covered,
                "order_digest": p["identity"][1],
                "digest_algorithm": "sha256-prefix-64",
                "intervals": p["intervals"],
                "uncovered": gaps,
            }
        )
    if entry["claim"] == "empty" and not complete:
        raise ValueError("full emptiness claim has missing passes or coverage gaps")
    return coverage, checks


def verify(entry, root):
    result = {
        "id": entry["id"],
        "type": "certificate",
        "n": entry["n"],
        "group": entry["group"],
        "stratum": entry["stratum"],
        "claim": entry["claim"],
        "scope": entry["scope"],
        "tier": None,
        "checks": [],
        "gaps": list(entry["gaps"]),
        "coverage": [],
    }
    for ref in entry["records"]:
        artifact(root, ref)
    if entry["method"].endswith("theorem"):
        result.update(tier="proven", proof=proof(entry), checks=["exact-row-sum-obstruction"])
    elif entry["method"] == "literature":
        result["checks"] = ["citation-present"]
        result["provenance"] = "literature"
    elif entry["method"] == "unreplayed-census":
        # An exhaustive census reported by a cited, archived source that the
        # board has not replayed. It is an attributed claim with no tier, like
        # a literature citation, but it is not a published result.
        if entry["scope"] != {"kind": "full"}:
            raise ValueError("unreplayed census claims cover the full stratum")
        if not any("artifact" in source for source in entry["sources"]):
            raise ValueError("unreplayed census requires an archived source file")
        result["checks"] = ["citation-present", "source-artifact-sha256"]
        result["provenance"] = "unreplayed-census"
        result["gaps"].append("The reported census has not been replayed on this board.")
    elif entry["method"] in ("good-coverage-v1", "archived-good-coverage-v1"):
        if entry["stratum"] != "good" or entry["scope"] != {"kind": "full"}:
            raise ValueError("good-engine receipts support only the full good-matrix search space")
        if "engine" not in entry:
            raise ValueError("coverage entry requires a pinned engine snapshot")
        artifact(root, entry["engine"]["artifact"])
        archived = entry["method"] == "archived-good-coverage-v1"
        checked = {**entry, "claim": "partial"} if archived else entry
        result["coverage"], result["checks"] = verify_good_records(checked, root)
        result["tier"] = None if archived else "receipt-backed"
        if archived:
            result["provenance"] = "archived-computation"
            result["gaps"].append(
                "Receipt coverage is incomplete; full emptiness is a historical claim."
            )
        result["checks"].append("engine-snapshot-sha256")
        result["limitations"] = [
            "Class-order digest is compared across receipts, not independently regenerated.",
            "Legacy receipts do not record execution commit or explicit solution count.",
            "No random slices or full computation have been replayed by this check.",
        ]
    elif entry["method"] == "best-coverage-v1":
        if entry["stratum"] != "best" or entry["scope"] != {"kind": "full"}:
            raise ValueError("best archive supports only the full best-matrix search space")
        if "engine" not in entry or not entry["records"]:
            raise ValueError("best archive requires engine and coverage records")
        artifact(root, entry["engine"]["artifact"])
        # Legacy fleet records use heterogeneous decks; preserve the complete
        # evidence without claiming a uniform, fully identified coverage union.
        for ref in entry["records"]:
            if not ref["path"].endswith(".json"):
                continue
            record = read_json(artifact(root, ref), max_bytes=2 * 1024 * 1024)
            if "cursor" in record and (record.get("sols") != 0 or record.get("unsettled_n") != 0):
                raise ValueError("best checkpoint is not settled empty")
        result["checks"] = ["artifact-sha256", "checkpoint-zero-solutions-and-unsettled"]
        result["gaps"] = sorted(
            set(
                result["gaps"]
                + [
                    "104500 legacy classes lack modern execution identity/digest; full coverage not certified.",
                    "Enumeration audit and heterogeneous deck unions require independent replay.",
                ]
            )
        )
        result["provenance"] = "archived-computation"
    return result
