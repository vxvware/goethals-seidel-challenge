#!/usr/bin/env python3
"""
Order-parameterized SAW search: the same kernel as saw_fast.py with the
order n as a runtime argument instead of a module constant.

Purpose: run the identical algorithm on the orders where best matrices are
already known (n = 3, 7, 13, 21, 31, 43, 57) and see how far up that ladder
it reaches. n = 73 is the open case; a technique that cannot solve 43 or 57
is not going to solve 73, and the ladder says which one it is.

Equivalence: at n = 73, with the same seed, keys and table size, this
reproduces saw_fast.saw_fast_run integer-for-integer (test_saw_n.py checks
it), so results here transfer to the production kernel.

Layout for order n, half = (n - 1) // 2:
    bits 0 .. 3*half-1   sequences A, B, C (skew: x_0 = 1, x_{n-j} = -x_j)
    bits 3*half .. 4*half sequence D (symmetric: x_{n-j} = x_j), incl. d_0
so nbits = 4*half + 1 free bits and the search space is 2^(4*half+1).

Usage: saw_n.py --n 13 [--workers W] [--steps S] [--table-bits B]
                       [--walks-per-worker K] [--seed-base S]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from numba import njit


@njit(cache=True)
def _bit_info(b, n, half):
    if b < 3 * half:
        m = b // half
        j = b % half + 1
        return m, j, n - j
    j = b - 3 * half
    if j == 0:
        return 3, 0, -1
    return 3, j, n - j


@njit(cache=True)
def _init_state(x, n, half):
    for m in range(3):
        x[m, 0] = 1
        for j in range(1, half + 1):
            v = 1 if np.random.random() < 0.5 else -1
            x[m, j] = v
            x[m, n - j] = -v
    x[3, 0] = 1 if np.random.random() < 0.5 else -1
    for j in range(1, half + 1):
        v = 1 if np.random.random() < 0.5 else -1
        x[3, j] = v
        x[3, n - j] = v


@njit(cache=True)
def _compute_T(x, T, n, half):
    for s in range(1, half + 1):
        acc = 0
        for m in range(4):
            for j in range(n):
                acc += x[m, j] * x[m, (j + s) % n]
        T[s] = acc


@njit(cache=True)
def _dT_row(x, b, out, n, half):
    """dT vector for flipping bit b at the current state (x unchanged)."""
    m, p1, p2 = _bit_info(b, n, half)
    for s in range(1, half + 1):
        out[s] = 0
    for s in range(1, half + 1):
        out[s] += -2 * x[m, p1] * (x[m, (p1 - s) % n] + x[m, (p1 + s) % n])
    x[m, p1] = -x[m, p1]
    if p2 >= 0:
        for s in range(1, half + 1):
            out[s] += -2 * x[m, p2] * (x[m, (p2 - s) % n] + x[m, (p2 + s) % n])
    x[m, p1] = -x[m, p1]


@njit(cache=True)
def _apply_flip(x, b, n, half):
    m, p1, p2 = _bit_info(b, n, half)
    x[m, p1] = -x[m, p1]
    if p2 >= 0:
        x[m, p2] = -x[m, p2]


@njit(cache=True)
def _visited(table, mask, h):
    if h == 0:
        return True
    i = h & mask
    while table[i] != 0:
        if table[i] == h:
            return True
        i = (i + 1) & mask
    return False


@njit(cache=True)
def _insert(table, mask, used, h, cap):
    if h == 0 or used[0] > cap:
        return
    i = h & mask
    while table[i] != 0:
        if table[i] == h:
            return
        i = (i + 1) & mask
    table[i] = h
    used[0] += 1


@njit(cache=True)
def _seq_bits(m, half):
    """(start, count) of bit indices belonging to sequence m."""
    if m < 3:
        return m * half, half
    return 3 * half, half + 1


@njit(cache=True)
def saw_n_run(seed, n_steps, keys, tbits, n):
    """One SAW walk at order n. Returns (best_E, found, x_final, x_best,
    steps_taken)."""
    half = (n - 1) // 2
    nbits = 4 * half + 1
    np.random.seed(seed)
    tsize = 1 << tbits
    mask = tsize - 1
    cap = (tsize * 9) // 10
    x = np.empty((4, n), dtype=np.int64)
    _init_state(x, n, half)
    T = np.zeros(half + 1, dtype=np.int64)
    _compute_T(x, T, n, half)
    E = 0
    for s in range(1, half + 1):
        E += T[s] * T[s]
    best_E = E
    best_x = x.copy()
    if E == 0:
        return 0, True, x, best_x, 0

    h = np.uint64(0)
    for b in range(nbits):
        m, p1, _ = _bit_info(b, n, half)
        if x[m, p1] < 0:
            h ^= keys[b]
    table = np.zeros(tsize, dtype=np.uint64)
    used = np.zeros(1, dtype=np.int64)
    _insert(table, mask, used, h, cap)

    dTmat = np.zeros((nbits, half + 1), dtype=np.int64)
    q = np.zeros(nbits, dtype=np.int64)
    pvec = np.zeros(nbits, dtype=np.int64)
    for b in range(nbits):
        _dT_row(x, b, dTmat[b], n, half)
        acc_q = 0
        acc_p = 0
        for s in range(1, half + 1):
            acc_q += dTmat[b, s] * dTmat[b, s]
            acc_p += T[s] * dTmat[b, s]
        q[b] = acc_q
        pvec[b] = acc_p

    delta = np.zeros(half + 1, dtype=np.int64)

    for step in range(n_steps):
        best_b = -1
        best_dE = np.int64(1) << 60
        for b in range(nbits):
            dE = 2 * pvec[b] + q[b]
            if dE < best_dE:
                if not _visited(table, mask, h ^ keys[b]):
                    best_dE = dE
                    best_b = b
        if best_b < 0:
            return best_E, False, x, best_x, step
        a = best_b
        am, _, _ = _bit_info(a, n, half)
        for s in range(half + 1):
            delta[s] = dTmat[a, s]
        _apply_flip(x, a, n, half)
        for s in range(1, half + 1):
            T[s] += delta[s]
        E += best_dE
        h ^= keys[a]
        _insert(table, mask, used, h, cap)
        if E < best_E:
            best_E = E
            best_x[:, :] = x
        if E == 0:
            return 0, True, x, best_x, step + 1

        for m in range(4):
            if m == am:
                continue
            start, cnt = _seq_bits(m, half)
            for b in range(start, start + cnt):
                acc = 0
                for s in range(1, half + 1):
                    acc += delta[s] * dTmat[b, s]
                pvec[b] += acc
        start, cnt = _seq_bits(am, half)
        for b in range(start, start + cnt):
            _dT_row(x, b, dTmat[b], n, half)
            acc_q = 0
            acc_p = 0
            for s in range(1, half + 1):
                acc_q += dTmat[b, s] * dTmat[b, s]
                acc_p += T[s] * dTmat[b, s]
            q[b] = acc_q
            pvec[b] = acc_p

    return best_E, False, x, best_x, n_steps


def make_keys(nbits):
    rng = np.random.default_rng(12345)
    return rng.integers(1, 1 << 63, size=nbits, dtype=np.uint64)


def save_solution(x, n, path):
    from verify import verify_quadruple
    ra, rb, rc, rd = (np.array(x[m]) for m in range(4))
    if not verify_quadruple(ra, rb, rc, rd, n):
        print(f"n={n}: E=0 FAILED exact verification (bug!)", flush=True)
        return False
    with open(path, "w") as f:
        json.dump({"A": ra.tolist(), "B": rb.tolist(),
                   "C": rc.tolist(), "D": rd.tolist()}, f)
    return True


def worker(args):
    (wid, n, n_steps, tbits, seed_base, max_walks, max_secs, marker,
     sol_path, t_start) = args
    half = (n - 1) // 2
    keys = make_keys(4 * half + 1)
    ppid0 = os.getppid()
    all_best = 1 << 60
    walk = 0
    steps_done = 0
    walk_bests = []
    while max_walks == 0 or walk < max_walks:
        if os.path.exists(marker) or os.getppid() != ppid0:
            break
        if max_secs and time.time() - t_start > max_secs:
            break
        walk += 1
        seed = seed_base + wid * 100_003 + walk
        best_E, found, x, best_x, steps = saw_n_run(seed, n_steps, keys,
                                                    tbits, n)
        steps_done += steps
        if found:
            dt = time.time() - t_start
            if save_solution(x, n, sol_path):
                open(marker, "w").close()
                print(f"n={n} SOLVED by w{wid:02d} on walk {walk} "
                      f"({steps} steps, {dt:.1f}s) -> {sol_path}", flush=True)
            walk_bests.append(0)
            return (wid, walk, steps_done, 0, True, walk_bests)
        walk_bests.append(int(best_E))
        if best_E < all_best:
            all_best = best_E
    return (wid, walk, steps_done, all_best, False, walk_bests)


def solve_order(n, workers, steps, tbits, seed_base, max_walks, max_secs,
                outdir):
    """Run a fleet at order n; return a result dict."""
    half = (n - 1) // 2
    nbits = 4 * half + 1
    marker = os.path.join(outdir, f"FOUND_n{n}")
    sol_path = os.path.join(outdir, f"solution_n{n}.json")
    if os.path.exists(marker):
        os.remove(marker)
    t0 = time.time()
    jobs = [(w, n, steps, tbits, seed_base, max_walks, max_secs, marker,
             sol_path, t0) for w in range(workers)]
    if workers == 1:
        res = [worker(jobs[0])]
    else:
        with Pool(workers) as pool:
            res = pool.map(worker, jobs)
    dt = time.time() - t0
    solved = any(r[4] for r in res)
    walks = sum(r[1] for r in res)
    steps_done = sum(r[2] for r in res)
    best = min(r[3] for r in res)
    bests = sorted(e for r in res for e in r[5])
    hist = {}
    for e in bests:
        hist[e] = hist.get(e, 0) + 1
    if os.path.exists(marker):
        os.remove(marker)
    return {"n": n, "nbits": nbits, "space": f"2^{nbits}", "solved": solved,
            "walks": walks, "steps": steps_done,
            "best_E": 0 if solved else best,
            "median_walk_E": bests[len(bests) // 2] if bests else None,
            "E_hist": {str(k): v for k, v in sorted(hist.items())[:12]},
            "secs": round(dt, 1),
            "steps_per_sec": round(steps_done / dt) if dt > 0 else 0,
            "solution": sol_path if solved else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="order (odd)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200_000,
                    help="max steps per walk")
    ap.add_argument("--table-bits", type=int, default=22)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--walks-per-worker", type=int, default=0,
                    help="0 = unlimited")
    ap.add_argument("--max-secs", type=float, default=0,
                    help="wall-clock budget for this order (0 = unlimited)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    if args.n % 2 == 0:
        sys.exit("n must be odd")
    r = solve_order(args.n, args.workers, args.steps, args.table_bits,
                    args.seed_base, args.walks_per_worker, args.max_secs,
                    args.outdir)
    print(json.dumps(r), flush=True)
    return 0 if r["solved"] else 1


if __name__ == "__main__":
    sys.exit(main())
