#!/usr/bin/env python3
"""
Exact verification of best matrix quadruples (order n, circulant form).

Standalone: numpy only. A quadruple of first rows (a, b, c, d) is a best
matrix quadruple iff
  - a, b, c are skew type: x_0 = 1, x_{n-j} = -x_j;
  - d is symmetric: d_{n-j} = d_j;
  - PAF_a(s) + PAF_b(s) + PAF_c(s) + PAF_d(s) = 0 for s = 1..n-1
    (equivalently AA^T + BB^T + CC^T + DD^T = 4nI for the circulants).
"""
import json
import sys

import numpy as np


def paf(x):
    n = len(x)
    return np.array([int(np.dot(x, np.roll(x, -s))) for s in range(n)])


def verify_quadruple(ra, rb, rc, rd, n):
    tot = paf(ra) + paf(rb) + paf(rc) + paf(rd)
    if tot[0] != 4 * n or np.any(tot[1:]):
        return False
    for r in (ra, rb, rc):
        if r[0] != 1 or any(r[j] != -r[n - j] for j in range(1, n)):
            return False
    return all(rd[j] == rd[n - j] for j in range(1, n))


def main():
    """Verify a solution JSON file: verify.py solution.json"""
    with open(sys.argv[1]) as f:
        sol = json.load(f)
    rows = [np.array(sol[k], dtype=np.int64) for k in ("A", "B", "C", "D")]
    n = len(rows[0])
    ok = verify_quadruple(*rows, n)
    print(f"order {n}: {'VALID best matrix quadruple' if ok else 'INVALID'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
