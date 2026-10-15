#!/usr/bin/env python3
"""
Fast SAW search for order-73 best matrices — incremental-evaluation kernel.

Semantics identical to saw_search.py (best-unvisited neighbor over 145
single-bit flips, Zobrist visited-memory, E = 0 <=> verified best matrix),
but per-step cost is cut by maintaining, for every candidate bit b:
    dTmat[b]  — the T-delta vector a flip of b would cause now
    q[b]      — sum_s dTmat[b,s]^2
    pvec[b]   — sum_s T[s]*dTmat[b,s]
so dE_b = 2*pvec[b] + q[b] is O(1) during neighbor selection. After a
commit, only same-sequence rows are rebuilt; other rows get a cheap
pvec += dot(Delta, dT) update. Also: visited table enlarged (default 2^25
slots) — the 2^22 table saturated its 90% cap mid-walk, slowing probes and
weakening the self-avoiding memory.

With --table-bits 22 and the same seed, trajectories match saw_search.py
EXACTLY (validated integer-for-integer before deployment).

Every walk's record state is appended to a records file (--records), so a
deep walk's configuration survives the run instead of only its energy
appearing in a log line. --seed-base offsets the seed stream: two hosts
running the same command with the same base would repeat each other's
walks exactly, so give every host a distinct base.

Usage: saw_fast.py [--workers W] [--steps S] [--table-bits B]
                   [--seed-base K] [--records FILE] [--tag NAME]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from numba import njit

N = 73
HALF = 36
NBITS = 3 * HALF + (HALF + 1)  # 145
RECORD_THRESHOLD = 320  # walks at or below the quasi-floor get their state kept


@njit(cache=True)
def _bit_info(b):
    if b < 3 * HALF:
        m = b // HALF
        j = b % HALF + 1
        return m, j, N - j
    j = b - 3 * HALF
    if j == 0:
        return 3, 0, -1
    return 3, j, N - j


@njit(cache=True)
def _init_state(x):
    for m in range(3):
        x[m, 0] = 1
        for j in range(1, HALF + 1):
            v = 1 if np.random.random() < 0.5 else -1
            x[m, j] = v
            x[m, N - j] = -v
    x[3, 0] = 1 if np.random.random() < 0.5 else -1
    for j in range(1, HALF + 1):
        v = 1 if np.random.random() < 0.5 else -1
        x[3, j] = v
        x[3, N - j] = v


@njit(cache=True)
def _compute_T(x, T):
    for s in range(1, HALF + 1):
        acc = 0
        for m in range(4):
            for j in range(N):
                acc += x[m, j] * x[m, (j + s) % N]
        T[s] = acc


@njit(cache=True)
def _dT_row(x, b, out):
    """dT vector for flipping bit b at the current state (x unchanged)."""
    m, p1, p2 = _bit_info(b)
    for s in range(1, HALF + 1):
        out[s] = 0
    for s in range(1, HALF + 1):
        out[s] += -2 * x[m, p1] * (x[m, (p1 - s) % N] + x[m, (p1 + s) % N])
    x[m, p1] = -x[m, p1]
    if p2 >= 0:
        for s in range(1, HALF + 1):
            out[s] += -2 * x[m, p2] * (x[m, (p2 - s) % N] + x[m, (p2 + s) % N])
        x[m, p2] = -x[m, p2]
        x[m, p2] = -x[m, p2]
    x[m, p1] = -x[m, p1]


@njit(cache=True)
def _apply_flip(x, b):
    m, p1, p2 = _bit_info(b)
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
def _seq_bits(m):
    """(start, count) of bit indices belonging to sequence m."""
    if m < 3:
        return m * HALF, HALF
    return 3 * HALF, HALF + 1


@njit(cache=True)
def saw_fast_run(seed, n_steps, keys, tbits):
    np.random.seed(seed)
    tsize = 1 << tbits
    mask = tsize - 1
    cap = (tsize * 9) // 10
    x = np.empty((4, N), dtype=np.int64)
    _init_state(x)
    T = np.zeros(HALF + 1, dtype=np.int64)
    _compute_T(x, T)
    E = 0
    for s in range(1, HALF + 1):
        E += T[s] * T[s]
    best_E = E
    best_x = x.copy()
    if E == 0:
        return 0, True, x, best_x

    h = np.uint64(0)
    for b in range(NBITS):
        m, p1, _ = _bit_info(b)
        if x[m, p1] < 0:
            h ^= keys[b]
    table = np.zeros(tsize, dtype=np.uint64)
    used = np.zeros(1, dtype=np.int64)
    _insert(table, mask, used, h, cap)

    # incremental structures
    dTmat = np.zeros((NBITS, HALF + 1), dtype=np.int64)
    q = np.zeros(NBITS, dtype=np.int64)
    pvec = np.zeros(NBITS, dtype=np.int64)
    for b in range(NBITS):
        _dT_row(x, b, dTmat[b])
        acc_q = 0
        acc_p = 0
        for s in range(1, HALF + 1):
            acc_q += dTmat[b, s] * dTmat[b, s]
            acc_p += T[s] * dTmat[b, s]
        q[b] = acc_q
        pvec[b] = acc_p

    delta = np.zeros(HALF + 1, dtype=np.int64)

    for _ in range(n_steps):
        best_b = -1
        best_dE = np.int64(1) << 60
        for b in range(NBITS):
            dE = 2 * pvec[b] + q[b]
            if dE < best_dE:
                if not _visited(table, mask, h ^ keys[b]):
                    best_dE = dE
                    best_b = b
        if best_b < 0:
            break
        a = best_b
        am, _, _ = _bit_info(a)
        for s in range(HALF + 1):
            delta[s] = dTmat[a, s]
        _apply_flip(x, a)
        for s in range(1, HALF + 1):
            T[s] += delta[s]
        E += best_dE
        h ^= keys[a]
        _insert(table, mask, used, h, cap)
        if E < best_E:
            best_E = E
            best_x[:, :] = x
        if E == 0:
            return 0, True, x, best_x

        # cross-sequence rows: dT unchanged, pvec += dot(delta, dT)
        for m in range(4):
            if m == am:
                continue
            start, cnt = _seq_bits(m)
            for b in range(start, start + cnt):
                acc = 0
                for s in range(1, HALF + 1):
                    acc += delta[s] * dTmat[b, s]
                pvec[b] += acc
        # same-sequence rows: rebuild
        start, cnt = _seq_bits(am)
        for b in range(start, start + cnt):
            _dT_row(x, b, dTmat[b])
            acc_q = 0
            acc_p = 0
            for s in range(1, HALF + 1):
                acc_q += dTmat[b, s] * dTmat[b, s]
                acc_p += T[s] * dTmat[b, s]
            q[b] = acc_q
            pvec[b] = acc_p

    return best_E, False, x, best_x


def verify_and_save(x):
    from verify import verify_quadruple
    ra, rb, rc, rd = (np.array(x[m]) for m in range(4))
    if not verify_quadruple(ra, rb, rc, rd, N):
        print("E=0 FAILED exact verification (bug!)", flush=True)
        return False
    with open("saw_solution.json", "w") as f:
        json.dump({"A": ra.tolist(), "B": rb.tolist(),
                   "C": rc.tolist(), "D": rd.tolist()}, f)
    open("FOUND_SAW", "w").close()
    print("*** VERIFIED BEST MATRIX OF ORDER 73 -> saw_solution.json ***",
          flush=True)
    return True


def worker(args):
    wid, n_steps, tbits, seed_base, records, tag = args
    rng = np.random.default_rng(12345)
    keys = rng.integers(1, 1 << 63, size=NBITS, dtype=np.uint64)
    ppid0 = os.getppid()
    all_best = 1 << 60
    restart = 0
    while not os.path.exists("FOUND_SAW"):
        if os.getppid() != ppid0:
            return
        restart += 1
        seed = seed_base + wid * 100_003 + restart
        best_E, found, x, best_x = saw_fast_run(seed, n_steps, keys, tbits)
        if found:
            verify_and_save(x)
            return
        if best_E < all_best:
            all_best = best_E
            print(f"f{wid:02d} best E={best_E} (walk {restart})", flush=True)
        if records and best_E <= RECORD_THRESHOLD:
            rec = {"tag": tag, "worker": wid, "walk": restart, "seed": seed,
                   "E": int(best_E), "steps": n_steps,
                   "x": [[int(v) for v in best_x[m]] for m in range(4)]}
            with open(records, "a") as f:
                f.write(json.dumps(rec) + "\n")
        elif wid == 0 and restart % 20 == 0:
            print(f"heartbeat: f00 walk {restart}, all-time E={all_best}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--table-bits", type=int, default=25)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="per-host seed offset; distinct per machine")
    ap.add_argument("--records", default="saw_records.jsonl",
                    help="append record states with E <= 320 here ('' = off)")
    ap.add_argument("--tag", default="local", help="host label in records")
    args = ap.parse_args()
    if os.path.exists("FOUND_SAW"):
        os.remove("FOUND_SAW")
    print(f"fast SAW: {args.workers} walkers, {args.steps} steps/walk, "
          f"2^{args.table_bits} visited table, seed base {args.seed_base}",
          flush=True)
    with Pool(args.workers) as pool:
        pool.map(worker, [(w, args.steps, args.table_bits, args.seed_base,
                           args.records, args.tag)
                          for w in range(args.workers)])


if __name__ == "__main__":
    sys.exit(main())
