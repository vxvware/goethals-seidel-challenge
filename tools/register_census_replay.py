"""Write and register a full offline census replay report.

A maintainer runs a pinned census engine to completion, then records its raw
output here:

    uv run python tools/register_census_replay.py williamson-n39 \
        --engine williamson-census-c-v1 --rows wil-39.txt \
        --host bos-game --threads 16 --date 2026-11-12 --wall 35

The report is checked exactly as verify/census_replays.py will check it on
every load (engine sources, entry bytes, re-derived classes against the
representatives and the claimed count) before anything is written. It lands
under evidence/census-replays/ and is registered in
baselines/census_replays.json; both are trusted paths, so this belongs in a
maintenance PR, never in an entry PR.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import census_replays  # noqa: E402
from verify.board import load  # noqa: E402
from verify.common import ROOT, digest, read_json, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("entry")
    p.add_argument("--engine", required=True, choices=sorted(census_replays.ENGINES))
    p.add_argument("--rows", required=True, help="engine output: four +/- rows per line")
    p.add_argument("--compression", type=int, help="d for a compression engine (n = d m)")
    p.add_argument("--host", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--wall", type=float, required=True, help="wall-clock seconds")
    p.add_argument("--command", default="")
    a = p.parse_args()
    documents, results, _ = load(reproduce=False)
    entry = documents[a.entry]
    if entry["type"] != "census":
        raise SystemExit(f"{a.entry} is not a census")
    rows = [
        line.strip()
        for line in Path(a.rows).read_text().splitlines()
        if line.strip() and line[0] in "+-"
    ]
    report = {
        "version": 1,
        "entry_id": a.entry,
        "entry_sha256": digest(ROOT / f"census/{a.entry}.json"),
        "engine_id": a.engine,
        "engine_sha256": census_replays.engine_digests(a.engine),
        "mode": "full",
        "status": "passed",
        "host": a.host,
        "threads": a.threads,
        "date": a.date,
        "wall_seconds": a.wall,
        "command": a.command,
        "rows": rows,
    }
    if a.compression is not None:
        report["compression"] = a.compression
    representatives = [results[q]["canonical_rows"] for q in entry["representatives"]]
    trial = dict(results[a.entry])
    census_replays.apply_report(entry, trial, report, report["entry_sha256"], representatives)
    path = ROOT / census_replays.REPORT_DIR / f"{a.entry}.json"
    write_json(path, report)
    registry = read_json(ROOT / "baselines/census_replays.json")
    ref = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
    registry["reports"] = [r for r in registry["reports"] if r["path"] != ref["path"]] + [ref]
    registry["reports"].sort(key=lambda r: r["path"])
    write_json(ROOT / "baselines/census_replays.json", registry)
    print(
        f"{a.entry}: {len(rows)} raw rows, {trial['count']} classes, bound {trial['bound']}, "
        f"registered at {ref['path']}"
    )


if __name__ == "__main__":
    main()
