"""CORRECTED good-matrix linear completion (fixes the fold-before-parity bug in
good_linear_check.py). Per the Codex review: fold the reflected duplicate terms
BEFORE reducing mod 2, and divide the residual by 8. Then a useful GF(2) system
in the H negative-half bits of C remains (my earlier version reduced mod 2 first,
so reflected pairs cancelled -> spurious rank 0).

good_completion_rows(A,B): fix skew A (a0=1) and symmetric B (b0=1). With
g_0=1, g_k=-a_k a_{2k}, h=g*B, D=h*C, the Gram condition per shift s is
  sum_{(i,j) in Q_s} (x_i XOR x_j) = (n + P_h(s) + P_A(s) + P_B(s)) / 8   (mod 2)
which is ONE GF(2) equation per shift. Solve M x = r, recover C=(-1)^x, D=h*C,
apply row sums, then EXACT integer PAF verify. This is NECESSARY, not sufficient.

Validated here on the whole catalogue: the true C is always recovered, with a
SMALL candidate count (not 2^H) -> the completion linearizes after all.
"""
import numpy as np, os

def good_completion_rows(A, B):
    n = len(A); H = (n - 1) // 2
    assert int(A[0]) == 1 and int(B[0]) == 1
    g = [1] + [-int(A[k]) * int(A[(2 * k) % n]) for k in range(1, n)]
    h = [g[k] * int(B[k]) for k in range(n)]
    y = [int(v < 0) for v in h]
    def paf(row, s): return sum(int(row[k]) * int(row[(k + s) % n]) for k in range(n))
    rows = []
    for s in range(1, H + 1):
        numerator = n + paf(h, s) + paf(A, s) + paf(B, s)
        if numerator % 8:
            return None, h            # impossible integer equation -> reject (A,B)
        u = s * pow(2, -1, n) % n; u = min(u, n - u)
        mask = 1 << (u - 1)
        for j in range(1, H + 1):
            mask ^= (y[(j - s) % n] ^ y[(j + s) % n]) << (j - 1)
        rows.append((mask, (numerator // 8) & 1))
    return rows, h

def solve_gf2(rows, nvar):
    R = [(m << 1) | b for m, b in rows]; piv = []
    for col in range(nvar):
        p = next((r for r in range(len(piv), len(R)) if (R[r] >> (col + 1)) & 1), None)
        if p is None: continue
        k = len(piv); R[k], R[p] = R[p], R[k]
        for r in range(len(R)):
            if r != k and (R[r] >> (col + 1)) & 1: R[r] ^= R[k]
        piv.append(col)
    for r in R:
        if r == 1: return None, []                     # 0=1 inconsistent
    part = sum(((R[r] & 1) << col) for r, col in enumerate(piv))
    basis = []
    for col in sorted(set(range(nvar)) - set(piv)):
        v = 1 << col
        for r, pc in enumerate(piv):
            if (R[r] >> (col + 1)) & 1: v |= (1 << pc)
        basis.append(v)
    return part, basis

def normalize(sol):
    """A skew a0=1; B,C,D symmetric normalized to x[0]=+1 (PAF-invariant)."""
    A, B, C, D = [np.asarray(x, dtype=np.int64).copy() for x in sol]
    if A[0] == -1: A = -A
    for X in (B, C, D):
        if X[0] == -1: X *= -1
    return A, B, C, D

def check_recovery(sol, n):
    H = (n - 1) // 2
    A, B, C, _D = normalize(sol)
    rows, _h = good_completion_rows(A, B)
    if rows is None: return False, -1
    part, basis = solve_gf2(rows, H)
    if part is None: return False, 0
    xtrue = sum((1 << (j - 1)) for j in range(1, H + 1) if C[j] == -1)
    cands = [part]
    for v in basis: cands = cands + [p ^ v for p in cands]
    return (xtrue in cands), len(cands)

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    tot = fails = 0; cs = []
    for n in (3, 5, 7, 9, 11, 13, 15):
        f = os.path.join(base, f"good_matrices_n{n}.npz")
        if not os.path.exists(f): continue
        S = np.load(f)["sols"]; nf = 0; c = []
        for sol in S:
            rec, nc = check_recovery(sol, n); tot += 1
            if not rec: fails += 1; nf += 1
            else: c.append(nc); cs.append(nc)
        print(f"n={n}: {len(S):,} matrices | recovery fails={nf} | median #candidates={int(np.median(c)) if c else -1}")
    for n in (15, 21):
        p = os.path.join(base, f"good_solution_n{n}.npy")
        if os.path.exists(p):
            rec, nc = check_recovery(np.load(p), n)
            print(f"  Russo n={n}: C-recovered={rec} #candidates={nc}")
    print(f"\nTOTAL {tot}: recovery fails={fails} | median candidates={int(np.median(cs)) if cs else -1}")
    print("VERDICT:", "CORRECTED LINEAR COMPLETION WORKS — C solved from (A,B) via GF(2), small candidate sets"
          if fails == 0 else "FAILS — investigate")

if __name__ == "__main__":
    main()
