#!/usr/bin/env python3
"""
Exact verification of GOOD matrix quadruples (circulant defining rows).

Good matrices (Bright-Djokovic-Kotsireas-Ganesh, AAAI-19, Def. 1 with the
circulant convention (a')): A skew type (a_0 = 1, a_{n-i} = -a_i);
B, C, D symmetric (x_{n-i} = x_i); and
    PAF_A(s) + PAF_B(s) + PAF_C(s) + PAF_D(s) = 0 for s = 1..n-1
(equivalently AA^T + B^2 + C^2 + D^2 = 4nI).
Row-sum corollary: sum(B)^2 + sum(C)^2 + sum(D)^2 = 4n - 1, each
sum == n (mod 4) after b_0 = c_0 = d_0 normalization via negation.
"""
import json
import sys

import numpy as np


def paf(x):
    n = len(x)
    return np.array([int(np.dot(x, np.roll(x, -s))) for s in range(n)])


def verify_good(ra, rb, rc, rd, n):
    tot = paf(ra) + paf(rb) + paf(rc) + paf(rd)
    if tot[0] != 4 * n or np.any(tot[1:]):
        return False
    if ra[0] != 1 or any(ra[j] != -ra[n - j] for j in range(1, n)):
        return False
    for r in (rb, rc, rd):
        if any(r[j] != r[n - j] for j in range(1, n)):
            return False
    return True


def main():
    with open(sys.argv[1]) as f:
        sol = json.load(f)
    rows = [np.array(sol[k], dtype=np.int64) for k in ("A", "B", "C", "D")]
    n = len(rows[0])
    ok = verify_good(*rows, n)
    print(f"order {n}: {'VALID good matrix quadruple' if ok else 'INVALID'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
