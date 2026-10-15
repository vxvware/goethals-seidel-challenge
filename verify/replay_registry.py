"""Apply maintainer-registered replay reports, never submission-declared tiers."""

from verify.common import ROOT, artifact, digest, read_json
from verify.evidence import expected_passes
from verify.recheck_slices import ENGINE, sample_classes, validate_replayed


def apply_report(entry, result, report, entry_sha256):
    if report.get("entry_id") != entry["id"] or report.get("entry_sha256") != entry_sha256:
        raise ValueError("registered replay is stale or bound to a different entry")
    if report.get("engine_id") != "good-gpu-v4" or report.get("engine_sha256") != digest(ENGINE):
        raise ValueError("registered replay engine differs from the pinned engine")
    if report.get("status") != "passed":
        raise ValueError(
            "registered replay failed or is incomplete; evidence cannot retain its tier"
        )
    samples = report.get("samples", [])
    if not samples or report.get("sample_count") != len(samples):
        raise ValueError("registered replay has missing samples")
    if entry["method"] not in ("good-coverage-v1", "archived-good-coverage-v1"):
        raise ValueError("unsupported registered replay method")
    coverage = {p["pass"]: p for p in result["coverage"]}
    full = report.get("mode") == "full"
    if full:
        if sorted(r["sB"] for r in samples) != expected_passes(entry["n"]):
            raise ValueError("full replay is missing a pass or repeats a pass")
        tasks = [(r["sB"], 0, r["classes"]) for r in samples]
    elif report.get("mode") == "sample":
        if type(report.get("seed")) is not int:
            raise ValueError("sample replay must record its integer random seed")
        tasks = [
            (s, i, i + 1)
            for s, i in sample_classes(result["coverage"], len(samples), report["seed"])
        ]
    else:
        raise ValueError("unknown replay mode")
    for receipt, (s, start, end) in zip(samples, tasks, strict=True):
        validate_replayed(receipt, entry["n"], s, start, end, coverage.get(s))
    result["checks"] = sorted(
        set(result["checks"] + ["registered-pinned-engine-replay", "regenerated-class-order"])
    )
    result["replay"] = {k: report[k] for k in ("mode", "seed", "sample_count", "status")}
    if full:
        result["tier"] = "reproduced"
        result["limitations"] = [
            "Full replay uses the same archived engine; systematic engine bugs remain possible."
        ]
    else:
        # A sample cannot repair gaps in an archived full-coverage claim.
        if result["tier"] == "receipt-backed":
            result["tier"] = "spot-checked"
        population = sum(p["covered"] for p in result["coverage"])
        result["replay"]["covered_population"] = population
        result["replay"]["single_bad_class_detection_probability"] = len(samples) / population
        result["limitations"] = [
            "Only sampled classes were replayed; unrecorded coverage gaps remain.",
            "The same engine can reproduce its own systematic bugs.",
        ]


def apply_registered(documents, results, root=ROOT):
    registry = read_json(root / "baselines/rechecks.json")
    if registry.get("version") != 1:
        raise ValueError("unsupported replay registry version")
    for ref in registry["reports"]:
        if not ref["path"].startswith("evidence/rechecks/"):
            raise ValueError("registered report must live under evidence/rechecks/")
        report = read_json(artifact(root, ref))
        ident = report.get("entry_id")
        if ident not in documents or documents[ident]["type"] != "certificate":
            raise ValueError("registered replay refers to an absent certificate")
        apply_report(
            documents[ident], results[ident], report, digest(root / f"certificates/{ident}.json")
        )
