#!/usr/bin/env python3
"""Check every imported raw matrix, its archive hash, count, and dedup mapping."""

import gzip
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, artifact, read_json
from verify.equivalence import fingerprint
from verify.gs_verify import validate_rows


def audit(root=ROOT):
    manifest = read_json(root / "baselines/seed-manifest.json", max_bytes=4 * 1024 * 1024)
    total = 0
    all_ids = set()
    for record in manifest["artifacts"]:
        count = 0
        ids = set()
        with gzip.open(artifact(root, record["archive"]), "rt") as stream:
            for line in stream:
                if len(line) > 256 * 1024:
                    raise ValueError("seed row exceeds limit")
                row = json.loads(line)
                q = validate_rows(row["rows"], row["n"])
                expected = f"n{row['n']}-{fingerprint(q)[3:23]}"
                if row["representative"] != expected:
                    raise ValueError("bad seed equivalence mapping")
                doc = read_json(root / f"quadruples/{expected}.json")
                if fingerprint(doc["rows"]) != fingerprint(q):
                    raise ValueError("representative does not match raw row")
                count += 1
                ids.add(expected)
        if count != record["rows"] or sorted(ids) != record["representatives"]:
            raise ValueError("seed manifest count/membership mismatch")
        total += count
        all_ids.update(ids)
    if total != manifest["raw_rows"] or len(all_ids) != manifest["representative_count"]:
        raise ValueError("seed total mismatch")
    return {
        "verified_raw_rows": total,
        "representatives": len(all_ids),
        "artifacts": len(manifest["artifacts"]),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
