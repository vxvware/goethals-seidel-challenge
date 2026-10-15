"""Exact existence verification using arbitrary-precision integer rotations."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import MAX_N, ROOT, read_json
from verify.equivalence import VERSION, canonical, fingerprint


def row_types(row):
    n = len(row)
    result = set()
    if all(row[j] == row[-j] for j in range(1, n)):
        result.add("s")
    if all(row[j] == -row[-j] for j in range(1, n)):
        result.add("k")
    return result


def memberships(rows):
    from itertools import permutations

    types = [row_types(r) for r in rows]
    result = ["general"]
    if any("k" in t for t in types):
        result.insert(0, "skew-sds")
    for track, pattern in (
        ("williamson", "ssss"),
        ("good", "ksss"),
        ("best", "kkks"),
        ("kkss", "kkss"),
    ):
        if any(all(kind in t for kind, t in zip(pattern, order)) for order in permutations(types)):
            result.insert(0, track)
    return result


def paf(row):
    n = len(row)
    bits = sum((v == -1) << j for j, v in enumerate(row))
    mask = (1 << n) - 1
    return tuple(
        n - 2 * (bits ^ (((bits << s) | (bits >> (n - s))) & mask)).bit_count() for s in range(n)
    )


def validate_rows(rows, n=None):
    if not isinstance(rows, (list, tuple)) or len(rows) != 4:
        raise ValueError("exactly four rows are required")
    n = len(rows[0]) if n is None else n
    if type(n) is not int or not 1 <= n <= MAX_N or n % 2 != 1:
        raise ValueError(f"order must be odd, between 1 and {MAX_N}")
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError("row length does not match order")
        if any(type(v) is not int or v not in (-1, 1) for v in row):
            raise ValueError("rows must contain integer +1 or -1 (no booleans/floats)")
    totals = tuple(map(sum, zip(*(paf(row) for row in rows))))
    if totals != (4 * n,) + (0,) * (n - 1):
        shift = next(s for s, v in enumerate(totals) if v != (4 * n if s == 0 else 0))
        raise ValueError(f"PAF identity fails at shift {shift}: {totals[shift]}")
    return tuple(tuple(row) for row in rows)


def decode_rows(entry):
    n = entry["n"]
    if entry["group"] != {"kind": "cyclic", "invariant_factors": [n]}:
        raise ValueError("v1 accepts only the cyclic group with invariant factors [n]")
    if ("rows" in entry) == ("blocks" in entry):
        raise ValueError("supply exactly one of rows or negative-index blocks")
    if "rows" in entry:
        return validate_rows(entry["rows"], n)
    blocks = entry["blocks"]
    if len(blocks) != 4:
        raise ValueError("exactly four blocks are required")
    for block in blocks:
        if any(type(j) is not int or not 0 <= j < n for j in block) or len(set(block)) != len(
            block
        ):
            raise ValueError("block indices must be distinct integers in [0,n)")
    return validate_rows([[(-1 if j in block else 1) for j in range(n)] for block in blocks], n)


def verify(entry, baseline=None):
    rows = decode_rows(entry)
    rep = canonical(rows)
    tracks = memberships(rep)
    baseline = baseline if baseline is not None else read_json(ROOT / "baselines/skew-gaps.json")
    flags = []
    if "skew-sds" in tracks and 4 * entry["n"] in baseline["orders"]:
        flags.append(
            {
                "kind": "fills-a-baseline-gap",
                "matrix_type": "skew-Hadamard",
                "order": 4 * entry["n"],
                "baseline": baseline["id"],
                "baseline_date": baseline["date"],
                "discovery_confirmed": False,
            }
        )
    return {
        "id": entry.get("id"),
        "type": "quadruple",
        "n": entry["n"],
        "group": entry["group"],
        "tracks": tracks,
        "tier": "reproduced",
        "checks": ["integer-domain", "exact-PAF", "symmetry", "canonical-form"],
        "equivalence": VERSION,
        "fingerprint": fingerprint(rows),
        "canonical_rows": rep,
        "row_sums": [sum(r) for r in rep],
        "pattern": "".join(
            "k" if "k" in row_types(r) else "s" if "s" in row_types(r) else "·" for r in rep
        ),
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(read_json(args.entry)), indent=2, ensure_ascii=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
