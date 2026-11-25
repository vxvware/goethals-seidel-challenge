#!/usr/bin/env python3
"""Write the best-matrix row-sum certificates for a range of odd orders.

Best matrices of order n need 4n - 3 to be an odd square. For every odd n in
the range where it is not, this writes the certificate that the verifier
checks by that arithmetic, in the form of the certificates already on the
board at n up to 131.

    uv run python tools/import_best_row_sums.py --min 133 --max 299 --date 2026-11-25
"""

import argparse
import sys
from math import isqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store
from verify.common import ROOT
from verify.model import validate

SUBMITTER = "vprusso"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min", type=int, required=True)
    p.add_argument("--max", type=int, required=True)
    p.add_argument("--date", required=True)
    a = p.parse_args()
    for n in range(a.min | 1, a.max + 1, 2):
        if isqrt(4 * n - 3) ** 2 == 4 * n - 3:
            continue
        doc = validate(
            {
                "schema_version": 1,
                "id": f"best-n{n}-row-sums",
                "type": "certificate",
                "n": n,
                "group": {"kind": "cyclic", "invariant_factors": [n]},
                "submitted_by": SUBMITTER,
                "submitted_at": a.date,
                "discovered_by": ["Best-matrix row-sum theorem"],
                "sources": [{"title": "Best row-sum necessary condition", "url": "https://arxiv.org/abs/1907.04987"}],
                "stratum": "best",
                "claim": "empty",
                "scope": {"kind": "full"},
                "method": "row-sum-theorem",
                "records": [],
                "gaps": [],
            }
        )
        store(ROOT / f"certificates/{doc['id']}.json", doc)
        print(f"best n={n}: 4n-3 = {4 * n - 3} is not a square")
