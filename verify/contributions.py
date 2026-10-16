"""Attach curated origin and credit without changing mathematical verdicts."""

from jsonschema import Draft202012Validator, FormatChecker
from verify.common import ROOT, read_json
from verify.seed_inventory import inventory

LABELS = {
    "participant-result": "Challenge participant",
    "independent-reproduction": "Independent reproduction",
    "published-import": "Published / catalogue import",
}


def apply(documents, results, root=ROOT):
    registry = read_json(root / "baselines/contributions.json")
    validator = Draft202012Validator(
        read_json(ROOT / "schema/contributions.schema.json"), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(registry), key=lambda e: str(e.path))
    if errors:
        raise ValueError(f"contribution schema: {errors[0].message}")
    artifacts = {a["id"]: a for a in inventory(root)["artifacts"]}
    seen = set()
    for result in results.values():
        result["contributions"] = []
    for position, group in enumerate(registry["contributions"]):
        if group["id"] in seen:
            raise ValueError(f"duplicate contribution: {group['id']}")
        seen.add(group["id"])
        ids = set(group["entry_ids"])
        for ident in group["seed_artifacts"]:
            if ident not in artifacts:
                raise ValueError(f"unknown contribution seed: {ident}")
            ids.update(artifacts[ident]["representatives"])
        if not ids or ids - documents.keys():
            raise ValueError(f"unknown or empty contribution entries: {group['id']}")
        item = {k: v for k, v in group.items() if k != "seed_artifacts"}
        item["entry_ids"] = sorted(ids)
        item["label"] = LABELS[group["relation"]]
        item["position"] = position
        item["recorded_at"] = registry["recorded_at"]
        for ident in sorted(ids):
            result = results[ident]
            focus = group.get("stratum")
            tracks = result.get("tracks", [result.get("stratum")])
            if focus and focus not in tracks:
                raise ValueError(f"contribution stratum mismatch: {ident}")
            result["contributions"].append(item)


def collect(results):
    """Stable, deduplicated credits for a cell or the whole board."""
    groups = {}
    for result in results:
        for group in result.get("contributions", []):
            groups[group["id"]] = group
    return sorted(groups.values(), key=lambda x: (not x["featured"], x["position"]))
