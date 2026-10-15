#!/usr/bin/env python3
"""Validate every entry, replay supported censuses, and check the whole board."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.board import load, summary
from verify.common import ROOT, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Skip census replay; census results remain lower bounds",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        d, r, b = load(args.root, reproduce=not args.quick)
        s = summary(d, r, b)
        if args.output:
            write_json(args.output, {"summary": s, "entries": r})
        print(
            json.dumps(
                {
                    k: s[k]
                    for k in (
                        "entry_count",
                        "quadruples",
                        "certificates",
                        "censuses",
                        "tier_counts",
                        "ungraded_claims",
                    )
                },
                indent=2,
            )
        )
    except (ValueError, KeyError, OSError) as e:
        p.exit(1, f"FAIL: {e}\n")


if __name__ == "__main__":
    main()
