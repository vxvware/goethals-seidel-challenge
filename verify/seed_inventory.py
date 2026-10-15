"""Combine the original catalogue and later participant archives."""

from pathlib import Path
from verify.common import ROOT, read_json


def inventory(root=ROOT):
    artifacts = {}
    total = 0
    representatives = set()
    for name in ("seed-manifest.json", "participant-seed-manifest.json"):
        manifest = read_json(Path(root) / "baselines" / name, max_bytes=4 * 1024 * 1024)
        ids = {ident for a in manifest["artifacts"] for ident in a["representatives"]}
        if (
            sum(a["rows"] for a in manifest["artifacts"]) != manifest["raw_rows"]
            or len(ids) != manifest["representative_count"]
        ):
            raise ValueError(f"seed inventory totals disagree: {name}")
        for record in manifest["artifacts"]:
            if record["id"] in artifacts:
                raise ValueError(f"duplicate seed artifact: {record['id']}")
            artifacts[record["id"]] = record
        total += manifest["raw_rows"]
        representatives.update(ids)
    return {
        "artifacts": list(artifacts.values()),
        "raw_rows": total,
        "representative_count": len(representatives),
    }
