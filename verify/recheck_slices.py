#!/usr/bin/env python3
"""Replay uniformly sampled classes using a pinned, bundled GPU engine."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import secrets
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, read_json, digest, write_json
from verify.evidence import verify, expected_passes


ENGINE = ROOT / "research/starter/good_gpu/good_gpu_complete.py"


def sample_classes(coverage, k, seed):
    intervals = [(p["pass"], a, b) for p in coverage for a, b in p["intervals"]]
    total = sum(b - a for _, a, b in intervals)
    if not 1 <= k <= total:
        raise ValueError("sample count must be between 1 and covered class count")
    offsets = random.Random(seed).sample(range(total), k)
    selected = []
    for offset in offsets:
        for s, a, b in intervals:
            if offset < b - a:
                selected.append((s, a + offset))
                break
            offset -= b - a
    return sorted(selected)


def validate_replayed(receipt, n, s, start, end, expected):
    for k in ("n", "sB", "start", "end", "classes", "order_digest", "unsettled", "pending"):
        if k not in receipt:
            raise ValueError(f"replay receipt lacks {k}")
    if (receipt["n"], receipt["sB"], receipt["start"], receipt["end"]) != (n, s, start, end):
        raise ValueError("replayed interval/identity mismatch")
    if receipt["pending"] != [] or receipt["unsettled"] != 0:
        raise ValueError("replay left pending work")
    if receipt.get("solutions", receipt.get("sols", 0)) != 0:
        raise ValueError("replay found a solution")
    if any(
        type(receipt[k]) is not int for k in ("n", "sB", "start", "end", "classes", "unsettled")
    ):
        raise ValueError("invalid replay integer fields")
    if not 0 <= start < end <= receipt["classes"]:
        raise ValueError("invalid replay bounds")
    if expected and (receipt["classes"], receipt["order_digest"]) != (
        expected["classes"],
        expected["order_digest"],
    ):
        raise ValueError("regenerated class count/order digest differs from archived receipt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("entry", type=Path)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, help="Omit for fresh OS randomness after the entry is fixed")
    p.add_argument("--full", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = output = None
    try:
        from verify.model import validate

        entry = validate(read_json(args.entry))
        if entry["method"] not in ("good-coverage-v1", "archived-good-coverage-v1"):
            raise ValueError("the bundled replay adapter supports good-matrix coverage entries")
        from verify.check_manifest import check

        check()
        result = verify(entry, ROOT)
        seed = secrets.randbits(128) if args.seed is None else args.seed
        coverage = {x["pass"]: x for x in result["coverage"]}
        tasks = (
            [(s, 0, None) for s in expected_passes(entry["n"])]
            if args.full
            else [(s, i, i + 1) for s, i in sample_classes(result["coverage"], args.samples, seed)]
        )
        report = {
            "version": 1,
            "entry_id": entry["id"],
            "entry_sha256": digest(args.entry),
            "engine_id": "good-gpu-v4",
            "engine_sha256": digest(ENGINE),
            "mode": "full" if args.full else "sample",
            "seed": seed,
            "sample_count": len(tasks),
            "samples": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        output = args.output or ROOT / ".gs/rechecks" / f"{entry['id']}-{seed}.json"
        write_json(output, report)
        for s, start, end in tasks:
            with tempfile.TemporaryDirectory(prefix="gs-recheck-") as tmp:
                command = [
                    args.python,
                    str(ENGINE),
                    str(entry["n"]),
                    "--pass-sB",
                    str(s),
                    "--start",
                    str(start),
                ]
                if end is not None:
                    command += ["--dmax", str(end)]
                proc = subprocess.run(
                    command, cwd=tmp, capture_output=True, text=True, timeout=args.timeout
                )
                receipt_path = Path(tmp) / f"coverage_n{entry['n']}_sB{s}.json"
                if (
                    proc.returncode
                    or not receipt_path.exists()
                    or list(Path(tmp).glob("good_solution_*.npy"))
                ):
                    report.update(status="failed", error=(proc.stderr or proc.stdout)[-4000:])
                    write_json(output, report)
                    raise ValueError(
                        f"engine failed, found a solution, or omitted receipt; see {output}"
                    )
                receipt = read_json(receipt_path)
                validate_replayed(
                    receipt,
                    entry["n"],
                    s,
                    start,
                    end if end is not None else receipt["classes"],
                    coverage.get(s),
                )
                report["samples"].append(receipt)
                write_json(output, report)
        report.update(
            status="passed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            covered_population=sum(x["covered"] for x in result["coverage"]),
        )
        write_json(output, report)
        print(
            json.dumps(
                {"report": str(output), "status": "passed", "mode": report["mode"]}, indent=2
            )
        )
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        if report is not None and output is not None:
            report.update(
                status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat()
            )
            write_json(output, report)
        p.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
