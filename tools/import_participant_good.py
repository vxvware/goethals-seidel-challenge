#!/usr/bin/env python3
"""Import Vincent Russo's pinned order-53/55 rows, without executing source code."""

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, digest, write_json
from verify.equivalence import canonical, fingerprint
from verify.gs_verify import memberships, validate_rows
from verify.model import validate

COMMIT = "020601f6e15db4e2ca023b602bc460204ce40236"
SOURCE_HASHES = {
    53: "79deb6ae7f913732e86d74786d1e453b522ae1c999083bc3eb20fcf3a8be7f60",
    55: "b7b7d33e460c7eea6e5d2d1ffdf1cec7bfbff7dd948be40b8cc73cf3e0c1dde7",
}


def import_rows(source_dir):
    records = []
    all_ids = set()
    for n in (53, 55):
        original = source_dir / f"good-{n}.txt"
        if digest(original) != SOURCE_HASHES[n]:
            raise ValueError("source differs from the pinned Vincent Russo archive")
        rows = []
        for line in original.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            words = line.split()
            if any(set(word) - set("+-") for word in words):
                raise ValueError("source rows must contain signs only")
            q = validate_rows([[1 if ch == "+" else -1 for ch in word] for word in words], n)
            if "good" not in memberships(q):
                raise ValueError("source does not contain good matrices")
            rows.append(q)
        if not rows:
            raise ValueError("empty participant construction source")
        folder = ROOT / "evidence/participants/vprusso-good"
        folder.mkdir(parents=True, exist_ok=True)
        raw = folder / original.name
        if raw.exists() and raw.read_bytes() != original.read_bytes():
            raise ValueError("refusing to overwrite a different source archive")
        raw.write_bytes(original.read_bytes())
        source = {
            "title": f"Vincent Russo: good matrices at n={n}",
            "url": f"https://github.com/vxvware/saw-best-matrix/blob/{COMMIT}/data/good/good-{n}.txt",
            "commit": COMMIT,
            "source_path": f"data/good/good-{n}.txt",
            "artifact": {"path": raw.relative_to(ROOT).as_posix(), "sha256": digest(raw)},
        }
        common = {
            "schema_version": 1,
            "n": n,
            "group": {"kind": "cyclic", "invariant_factors": [n]},
            "submitted_by": "trevor-vincent",
            "submitted_at": "2026-10-15",
            "discovered_by": ["Vincent Russo"],
            "sources": [source],
            "equivalence": "equiv-v1",
        }
        mappings = []
        ids = set()
        for q in rows:
            rep = canonical(q)
            ident = f"n{n}-{fingerprint(rep)[3:23]}"
            doc = validate(
                {**common, "id": ident, "type": "quadruple", "rows": [list(r) for r in rep]}
            )
            path = ROOT / f"quadruples/{ident}.json"
            if path.exists() and json.loads(path.read_text()) != doc:
                raise ValueError("refusing to overwrite a different contribution")
            write_json(path, doc)
            ids.add(ident)
            mappings.append({"n": n, "rows": q, "representative": ident})
        archive = folder / f"good-{n}.jsonl.gz"
        payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in mappings).encode()
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as stream:
            stream.write(payload)
        if archive.exists() and archive.read_bytes() != compressed.getvalue():
            raise ValueError("refusing to overwrite a different raw-row archive")
        archive.write_bytes(compressed.getvalue())
        census = validate(
            {
                **common,
                "id": f"russo-good-n{n}",
                "type": "census",
                "stratum": "good",
                "representatives": sorted(ids),
                "claimed_count": len(ids),
                "claim_kind": "lower-bound",
                "secondary": {"stored_rows": len(rows)},
                "notes": "The source reports a full census. This board verifies the representatives and a lower bound; the full order has not been replayed here.",
            }
        )
        path = ROOT / f"census/{census['id']}.json"
        if path.exists() and json.loads(path.read_text()) != census:
            raise ValueError("refusing to overwrite a different census")
        write_json(path, census)
        records.append(
            {
                "id": f"russo-good-n{n}",
                "source": source,
                "rows": len(rows),
                "representatives": sorted(ids),
                "archive": {
                    "path": archive.relative_to(ROOT).as_posix(),
                    "sha256": digest(archive),
                },
            }
        )
        all_ids.update(ids)
        print(f"n={n}: {len(rows)} verified quadruples -> {len(ids)} classes")
    write_json(
        ROOT / "baselines/participant-seed-manifest.json",
        {
            "version": 1,
            "artifacts": records,
            "raw_rows": sum(r["rows"] for r in records),
            "representative_count": len(all_ids),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    import_rows(p.parse_args().source_dir)
