#!/usr/bin/env python3
"""
Exact join pipeline for circulant GOOD matrices (A skew; B, C, D symmetric).

Identity: PAF_A + PAF_B + PAF_C + PAF_D = 0 at every shift s != 0
(<=> AA^T + B^2 + C^2 + D^2 = 4nI). Row sums: sum(A) = 1 forced;
{sum(B), sum(C), sum(D)} ranges over the decompositions of 4n - 1 into
three odd squares with each sum == n (mod 4) (after negation
normalization x_0 = +1 for B, C, D).

Architecture per decomposition (sB >= sC >= sD by |.|):
  outer  : pool(sB) — enumerate, spectral-viability filter
           (PSD <= 4n - 1, since PSD_A >= 1), decimation classes;
  middle : per outer row O with budget R = 4n - PSD_O, filter pool(sC)
           and pool(sD) by PSD <= R - 1; scan pairs (c, d) with
           PSD_c + PSD_d <= R - 1 pointwise;
  close  : required PAF_a = -(PAF_O + PAF_c + PAF_d); look up in a
           fingerprint index of the FULL skew pool 2^H (A needs no
           spectral filter: linearity forces its spectrum to fit);
  verify : exact integer PAF check + full axiom verification.

Complete over scanned classes. CPU/numpy implementation (multiprocessing
over outer classes).

Usage: good_join.py <n> [--dmax K] [--start S] [--workers W]
"""
import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

MASK = (1 << 62) - 1


def decompositions(n):
    """Sign-fixed {sB, sC, sD}: odd, == n mod 4, squares summing 4n-1."""
    tgt = 4 * n - 1
    res = set()
    mx = int(tgt ** 0.5) + 1
    odds = [v for v in range(1, mx + 1, 2)]
    for x in odds:
        for y in odds:
            z2 = tgt - x * x - y * y
            if z2 < 1:
                continue
            z = int(round(z2 ** 0.5))
            if z * z != z2 or z % 2 == 0:
                continue
            trip = []
            ok = True
            for v in (x, y, z):
                if v % 4 == n % 4:
                    trip.append(v)
                elif (-v) % 4 == n % 4:
                    trip.append(-v)
                else:
                    ok = False
                    break
            if ok:
                res.add(tuple(sorted(trip, key=abs, reverse=True)))
    return sorted(res, key=lambda t: tuple(map(abs, t)), reverse=True)


def sym_pool_combos(H, sigma):
    """Negative-position combos for symmetric rows with x_0=+1, sum sigma."""
    m_neg2 = H - (sigma - 1) // 2
    if m_neg2 % 2:
        return None, None
    k = m_neg2 // 2
    if not (0 <= k <= H):
        return None, None
    return k, None


def sym_rows_from_combos(combos, H, n):
    m = len(combos)
    free = np.ones((m, H))
    if combos.shape[1]:
        np.put_along_axis(free, combos.astype(np.int64), -1.0, axis=1)
    rows = np.empty((m, n))
    rows[:, 0] = 1.0
    rows[:, 1:H + 1] = free
    rows[:, H + 1:] = free[:, ::-1]
    return rows


def build_sym_pool(H, n, sigma, chunk=1 << 17):
    """(rows int8 (m,n), psd float32 (m,H), paf int16 (m,H))."""
    k, _ = sym_pool_combos(H, sigma)
    assert k is not None, f"no pool for sigma={sigma}"
    combos_iter = itertools.combinations(range(H), k)
    rows_l, psd_l, paf_l = [], [], []
    while True:
        block = np.array(list(itertools.islice(combos_iter, chunk)),
                         dtype=np.int16)
        if block.size == 0 and len(block) == 0:
            break
        rows = sym_rows_from_combos(block.reshape(len(block), k), H, n)
        F = np.fft.rfft(rows, axis=1)
        p = np.abs(F) ** 2
        rows_l.append(rows.astype(np.int8))
        psd_l.append(p[:, 1:].astype(np.float32))
        paf_l.append(np.round(np.fft.irfft(p, n=n, axis=1)[:, 1:H + 1])
                     .astype(np.int16))
        if len(block) < chunk:
            break
    return (np.concatenate(rows_l), np.concatenate(psd_l),
            np.concatenate(paf_l))


def build_skew_index(H, n, hv, chunk=1 << 18):
    """Fingerprint index over the full skew pool 2^H: PAF int8 + sorted fp."""
    m = 1 << H
    paf = np.empty((m, H), dtype=np.int8)
    fp = np.empty(m, dtype=np.int64)
    for i0 in range(0, m, chunk):
        ar = np.arange(i0, min(i0 + chunk, m))
        bits = ((ar[:, None] >> np.arange(H)) & 1).astype(np.int8)
        f = (2 * bits - 1).astype(np.float64)
        rows = np.ones((len(ar), n))
        rows[:, 1:H + 1] = f
        rows[:, H + 1:] = -f[:, ::-1]
        p = np.abs(np.fft.rfft(rows, axis=1)) ** 2
        pf = np.round(np.fft.irfft(p, n=n, axis=1)[:, 1:H + 1])
        paf[i0:i0 + len(ar)] = pf.astype(np.int8)
        fp[i0:i0 + len(ar)] = (pf.astype(np.int64) @ hv) & MASK
    order = np.argsort(fp, kind="stable").astype(np.int64)
    return paf, fp[order], order


def skew_row_from_index(idx, H, n):
    bits = np.array([(idx >> b) & 1 for b in range(H)])
    f = 2 * bits - 1
    r = np.ones(n, dtype=np.int64)
    r[1:H + 1] = f
    r[H + 1:] = -f[::-1]
    return r


def decim_classes(psd_pool, combos_masks, n, H, viable_idx):
    """Decimation class reps among viable outer rows via free-bit masks."""
    ts = [t for t in range(1, n) if np.gcd(t, n) == 1]
    vm = combos_masks[viable_idx]
    best = vm.copy()
    for t in ts:
        perm = np.empty(H, dtype=np.int64)
        for b in range(H):
            j = (t * (b + 1)) % n
            perm[b] = min(j, n - j) - 1
        img = np.zeros_like(vm)
        for b in range(H):
            img |= (((vm >> np.uint64(b)) & np.uint64(1))
                    << np.uint64(perm[b]))
        best = np.minimum(best, img)
    return viable_idx[best == vm]


_G = {}


def _init_worker(payload):
    _G.update(payload)


def _scan_outer(oi):
    n, H = _G["n"], _G["H"]
    psd_o = _G["psd_B"][oi].astype(np.float64)
    paf_o = _G["paf_B"][oi].astype(np.int64)
    R = 4 * n - psd_o
    pC = _G["psd_C"]
    pD = _G["psd_D"]
    Cidx = np.nonzero((pC <= (R - 1)[None, :] + 1e-6).all(axis=1))[0]
    Didx = np.nonzero((pD <= (R - 1)[None, :] + 1e-6).all(axis=1))[0]
    if len(Cidx) == 0 or len(Didx) == 0:
        return None, 0
    target = -(paf_o)
    pafC = _G["paf_C"][Cidx].astype(np.int64)
    pafD = _G["paf_D"][Didx].astype(np.int64)
    fpD = (pafD @ _G["hv"]) & MASK
    psdC = pC[Cidx]
    psdD = pD[Didx]
    fp_sorted, order = _G["fp_sorted"], _G["order"]
    hv = _G["hv"]
    tgt_fp = (target @ hv) & MASK
    npairs = 0
    for ci in range(len(Cidx)):
        keep = (psdD <= (R - 1e0)[None, :] - psdC[ci][None, :] + 1e-6
                ).all(axis=1)
        dk = np.nonzero(keep)[0]
        if len(dk) == 0:
            continue
        npairs += len(dk)
        need = (tgt_fp - (pafC[ci] @ hv) - fpD[dk]) & MASK
        pos = np.searchsorted(fp_sorted, need)
        pos = np.clip(pos, 0, len(fp_sorted) - 1)
        hits = np.nonzero(fp_sorted[pos] == need)[0]
        for q in hits:
            di = dk[q]
            p = pos[q]
            needvec = target - pafC[ci] - pafD[di].astype(np.int64)
            while p < len(fp_sorted) and fp_sorted[p] == need[q]:
                ai = int(order[p])
                if np.array_equal(_G["paf_A"][ai].astype(np.int64), needvec):
                    return (oi, ai, int(Cidx[ci]), int(Didx[di])), npairs
                p += 1
    return None, npairs


def main():
    from verify_good import verify_good
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--dmax", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    n = args.n
    H = (n - 1) // 2

    for dec in decompositions(n):
        sB, sC, sD = dec
        print(f"n={n}: decomposition sums (B,C,D) = {dec}", flush=True)
        t0 = time.time()
        # pools carry the b_0 = c_0 = d_0 = +1 normalization (negation
        # equivalence); signed sums are handled directly by the combo count
        rows_B, psd_B, paf_B = build_sym_pool(H, n, sB)
        rows_C, psd_C, paf_C = build_sym_pool(H, n, sC)
        rows_D, psd_D, paf_D = build_sym_pool(H, n, sD)
        print(f"  pools B/C/D: {len(rows_B):,}/{len(rows_C):,}/"
              f"{len(rows_D):,} [{time.time()-t0:.0f}s]", flush=True)

        rng = np.random.default_rng(777)
        hv = rng.integers(1, 1 << 61, size=H, dtype=np.int64)
        t0 = time.time()
        paf_A, fp_sorted, order = build_skew_index(H, n, hv)
        print(f"  skew index: {1 << H:,} rows [{time.time()-t0:.0f}s]",
              flush=True)

        viable = np.nonzero(psd_B.max(axis=1) <= 4 * n - 1 + 1e-9)[0]
        masks = np.zeros(len(rows_B), dtype=np.uint64)
        neg = rows_B[:, 1:H + 1] < 0
        for b in range(H):
            masks |= (neg[:, b].astype(np.uint64) << np.uint64(b))
        reps = decim_classes(psd_B, masks, n, H, viable)
        print(f"  outer viable {len(viable):,}, classes {len(reps):,}",
              flush=True)
        if args.dmax:
            reps = reps[:args.dmax]

        # fork inheritance (Linux): populate parent-side globals BEFORE the
        # Pool so workers share the tables copy-on-write — never pickle them
        _G.update(n=n, H=H, psd_B=psd_B, paf_B=paf_B, psd_C=psd_C,
                  paf_C=paf_C, psd_D=psd_D, paf_D=paf_D, paf_A=paf_A,
                  fp_sorted=fp_sorted, order=order, hv=hv)
        t0 = time.time()
        done = 0
        tot_pairs = 0.0
        with Pool(args.workers) as pool:
            for res, npairs in pool.imap_unordered(
                    _scan_outer, reps[args.start:], chunksize=1):
                done += 1
                tot_pairs += npairs
                if done % 200 == 0:
                    print(f"  {args.start + done}/{len(reps)} outer classes, "
                          f"{tot_pairs:.2e} pairs "
                          f"[{time.time()-t0:.0f}s]", flush=True)
                    with open(f"progress_good_n{n}.json", "w") as f:
                        json.dump(dict(n=n, dec=list(dec),
                                       start=args.start,
                                       done=args.start + done,
                                       classes=int(len(reps)),
                                       pairs=tot_pairs,
                                       elapsed_s=round(time.time()-t0, 1)), f)
                if res is not None:
                    oi, ai, ci, di = res
                    x = np.stack([
                        skew_row_from_index(ai, H, n),
                        rows_B[oi].astype(np.int64),
                        rows_C[ci].astype(np.int64),
                        rows_D[di].astype(np.int64)])
                    ok = verify_good(x[0], x[1], x[2], x[3], n)
                    print(f"*** SOLUTION n={n} dec={dec}: verified={ok} "
                          f"({done} classes, {time.time()-t0:.0f}s) ***",
                          flush=True)
                    if ok:
                        np.save(f"good_solution_n{n}.npy", x)
                        pool.terminate()
                        return
        print(f"  decomposition {dec}: no solution in {done} classes "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"n={n}: ALL decompositions exhausted — no circulant good "
          f"matrices of order {n} (within scanned classes)", flush=True)


if __name__ == "__main__":
    main()
