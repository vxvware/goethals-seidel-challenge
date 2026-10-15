#!/usr/bin/env python3
"""
GPU LINEAR-COMPLETION engine for circulant GOOD matrices (Codex §3 path).

Per outer symmetric B decimation class:
  1. Skew A pool by sine-MITM under the spectral budget PSD_A <= 4n - PSD_B
     (C,D symmetric only guarantee PSD >= 0; float64 budgets — conservative).
  2. One GPU thread per A: all-integer sign-bit pipeline
       sa -> g (g_k = -a_k a_2k, XOR) -> h = g*B -> PAF via 128-bit popcounts
       -> mod-8 consistency (early exit) -> GF(2) system (the CORRECTED folded
       completion, good_completion.py convention) -> in-register elimination
       -> enumerate the small nullspace -> exact integer PAF verify in-kernel.
  3. Hits re-verified on host by verify_good (exact); nullity > FMAX recorded
     and settled on host via the exact CPU completion (fail-closed).

Replaces the Theta(mC*mD) C-by-D pair scan with Theta(|A-pool|) completions.
B-first covers every decomposition sharing sigma_B in one pass (C<->D relabel
is free; row sums are implied by the PAF identities).

Usage: good_gpu_complete.py <n> [--all] [--dmax K] [--start S]
"""
import argparse
import json
import os
import time

import numpy as np
import cupy as cp

from good_join import decompositions, build_sym_pool, decim_classes
from mitm_skew import half_tables, mitm_pool_fused
from good_completion import good_completion_rows, solve_gf2
from verify_good import verify_good

MAXH = 48
FMAX = 12          # in-kernel nullity cap (2^12 candidates); larger -> host
HIT_CAP = 1 << 16
UNS_CAP = 1 << 16

KERNEL = r"""
typedef unsigned long long u64;
typedef unsigned __int128 u128;
// N_ / H_ are substituted per order at kernel-build time: compile-time loop
// bounds (full unroll, constant indexing, right-sized arrays) and NO integer
// modulo anywhere in the hot path (division is ~30-40 cycles on GPU and the
// v1 sg-loop did n of them per system -- it dominated the whole kernel).
#define N_ %%N%%
#define H_ %%H%%
#define FMAX 12

__device__ __forceinline__ int pc128(u128 v)
{ return __popcll((u64)v) + __popcll((u64)(v >> 64)); }

__device__ __forceinline__ u128 rotn(u128 v, int s, u128 maskn)
{ return ((v >> s) | (v << (N_ - s))) & maskn; }

extern "C" __global__ void complete_scan(
    const u64* __restrict__ Am,     // (nA) neg-half masks of A (bit j-1 <-> a_j<0)
    const long long nA,
    const u64 sbLo, const u64 sbHi, // B sign mask (n bits; bit k=1 iff b_k<0)
    const int* __restrict__ pafB,   // (H+1) integer PAF_B[s], index s=1..H
    const int* __restrict__ ufold,  // (H+1) u(s)=fold(s*inv2 mod n), 1..H
    const int n_unused, const int H_unused,
    unsigned int* counts,           // [0]=consistent,[1]=hits,[2]=unsettled
    u64* __restrict__ hitbuf,       // (HIT_CAP,2): (Amask, xC)
    const int hitcap,
    u64* __restrict__ unsbuf, const int unscap)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nA) return;
    const u64 m = Am[i];
    const u128 maskn = ((((u128)1) << N_) - 1);
    const u128 sb = ((u128)sbLo) | (((u128)sbHi) << 64);

    // skew A sign row: bit k=1 iff a_k<0; a_0=+1, a_{n-j}=-a_j
    u128 sa = 0;
    #pragma unroll
    for (int j = 1; j <= H_; j++) {
        u64 b = (m >> (j - 1)) & 1ull;
        sa |= ((u128)b) << j;
        sa |= ((u128)(b ^ 1ull)) << (N_ - j);
    }
    // g_k = -a_k a_{2k} (k!=0), g_0=+1: sign = 1 ^ sa_k ^ sa_{2k}
    // (2k mod n) via conditional subtract -- no division
    u128 sg = 0;
    #pragma unroll
    for (int k = 1; k < N_; k++) {
        int k2 = 2 * k; if (k2 >= N_) k2 -= N_;
        u64 b = 1ull ^ (u64)((sa >> k) & 1) ^ (u64)((sa >> k2) & 1);
        sg |= ((u128)b) << k;
    }
    const u128 sh = sg ^ sb;                       // h = g*B (h_0=+1)

    // per-shift: mod-8 invariant + GF(2) row build (corrected folded completion).
    // Row masks built bit-parallel (codex_review.md #2, algebra pre-validated
    // 26,600 rows / 0 mismatches): bit j of rotn(sh,s) is y[j+s], bit j of
    // rotn(sh,N_-s) is y[j-s]; shift by 1 and keep H bits to select j=1..H.
    // Replaces the O(H^2) per-bit inner loop with O(H) whole-row ops. The mod-8
    // check is kept as a cheap invariant (measured: never fires -- divisibility
    // is implied by the h construction; do NOT plan around it as a prefilter).
    const u64 maskH = (1ull << H_) - 1ull;
    int pafAB[H_ + 1];
    u64 rows[H_];
    for (int s = 1; s <= H_; s++) {
        int ph = N_ - 2 * pc128(sh ^ rotn(sh, s, maskn));
        int pa = N_ - 2 * pc128(sa ^ rotn(sa, s, maskn));
        int num = N_ + ph + pa + pafB[s];
        if (num & 7) return;                       // invariant; never observed
        pafAB[s] = pa + pafB[s];
        u128 nbr = rotn(sh, s, maskn) ^ rotn(sh, N_ - s, maskn);
        u64 mask = (((u64)(nbr >> 1)) & maskH) ^ (1ull << (ufold[s] - 1));
        rows[s - 1] = (mask << 1) | ((u64)((num >> 3) & 1));
    }

    // GF(2) elimination (bit 0 = rhs, bit c+1 = variable c)
    int piv[H_], r = 0;
    for (int col = 0; col < H_; col++) {
        u64 bit = 2ull << col;
        int p = -1;
        for (int q = r; q < H_; q++) if (rows[q] & bit) { p = q; break; }
        if (p < 0) continue;
        u64 tmp = rows[p]; rows[p] = rows[r]; rows[r] = tmp;
        for (int q = 0; q < H_; q++)
            if (q != r && (rows[q] & bit)) rows[q] ^= rows[r];
        piv[r++] = col;
    }
    for (int q = r; q < H_; q++) if (rows[q] == 1ull) return;   // 0 = 1
    atomicAdd(&counts[0], 1u);
    int freeCnt = H_ - r;
    if (freeCnt > FMAX) {                          // rare: settle on host
        unsigned idx = atomicAdd(&counts[2], 1u);
        if (idx < (unsigned)unscap) unsbuf[idx] = m;
        return;
    }
    u64 part = 0;
    for (int q = 0; q < r; q++) if (rows[q] & 1ull) part |= 1ull << piv[q];
    u64 basis[FMAX]; int nb = 0;
    int pi = 0;
    for (int col = 0; col < H_; col++) {
        if (pi < r && piv[pi] == col) { pi++; continue; }
        u64 v = 1ull << col;
        for (int q = 0; q < r; q++)
            if (rows[q] & (2ull << col)) v |= 1ull << piv[q];
        basis[nb++] = v;
    }
    for (long long t = 0; t < (1ll << freeCnt); t++) {
        u64 x = part;
        for (int b = 0; b < nb; b++) if ((t >> b) & 1) x ^= basis[b];
        u128 sc = 0;                               // symmetric C, c_0=+1
        #pragma unroll
        for (int j = 1; j <= H_; j++) {
            u64 cb = (x >> (j - 1)) & 1ull;
            sc |= ((u128)cb) << j;
            sc |= ((u128)cb) << (N_ - j);
        }
        u128 sd = sh ^ sc;                         // D = h*C (symmetric)
        bool good = true;
        for (int s = 1; s <= H_; s++) {
            int pcs = N_ - 2 * pc128(sc ^ rotn(sc, s, maskn));
            int pds = N_ - 2 * pc128(sd ^ rotn(sd, s, maskn));
            if (pafAB[s] + pcs + pds != 0) { good = false; break; }
        }
        if (good) {
            unsigned idx = atomicAdd(&counts[1], 1u);
            if (idx < (unsigned)hitcap) {
                hitbuf[2 * idx] = m; hitbuf[2 * idx + 1] = x;
            }
        }
    }
}
"""


def fold(x, n):
    x %= n
    return min(x, n - x)


def rows_from_masks(amask, xmask, Brow, n, H):
    """Reconstruct (A,B,C,D) int rows from kernel hit (Amask, xC)."""
    A = np.ones(n, dtype=np.int64)
    for j in range(1, H + 1):
        A[j] = -1 if (int(amask) >> (j - 1)) & 1 else 1
        A[n - j] = -A[j]
    C = np.ones(n, dtype=np.int64)
    for j in range(1, H + 1):
        C[j] = -1 if (int(xmask) >> (j - 1)) & 1 else 1
        C[n - j] = C[j]
    B = Brow.astype(np.int64)
    g = np.empty(n, dtype=np.int64); g[0] = 1
    for k in range(1, n):
        g[k] = -A[k] * A[(2 * k) % n]
    D = g * B * C
    return A, B, C, D


def settle_host(amask, Brow, n, H):
    """Exact CPU completion for a nullity>FMAX system. Returns list of solutions."""
    A = np.ones(n, dtype=np.int64)
    for j in range(1, H + 1):
        A[j] = -1 if (int(amask) >> (j - 1)) & 1 else 1
        A[n - j] = -A[j]
    rows, h = good_completion_rows(A, Brow.astype(np.int64))
    if rows is None:
        return []
    part, basis = solve_gf2(rows, H)
    if part is None:
        return []
    h = np.asarray(h, dtype=np.int64)
    sols = []
    cands = [part]
    for v in basis:
        cands = cands + [p ^ v for p in cands]
    for x in cands:
        C = np.ones(n, dtype=np.int64)
        for j in range(1, H + 1):
            C[j] = -1 if (x >> (j - 1)) & 1 else 1
            C[n - j] = C[j]
        D = h * C
        if verify_good(A, Brow.astype(np.int64), C, D, n):
            sols.append(np.stack([A, Brow.astype(np.int64), C, D]))
    return sols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--all", action="store_true",
                    help="collect ALL verified solutions (gate mode); default stops at first")
    ap.add_argument("--dmax", type=int, default=0,
                    help="bound classes of the FIRST executed pass only")
    ap.add_argument("--start", type=int, default=0,
                    help="resume cursor for the FIRST executed pass only; "
                         "later passes always start at 0")
    ap.add_argument("--pass-sB", dest="pass_sB", type=int, default=None,
                    help="run ONLY the pass with this outer B row-sum "
                         "(fleet chunking: pass + interval, explicit)")
    args = ap.parse_args()
    n = args.n
    H = (n - 1) // 2
    assert H <= MAXH
    H1 = H // 2
    inv2 = pow(2, -1, n)
    ufold_h = np.zeros(H + 1, dtype=np.int32)
    for s in range(1, H + 1):
        ufold_h[s] = fold(s * inv2, n)
    ufold_g = cp.asarray(ufold_h)

    src = KERNEL.replace("%%N%%", str(n)).replace("%%H%%", str(H))
    kern = cp.RawKernel(src, "complete_scan", options=("--device-int128",))
    # a1=+1 normalization: decimation t=n-1 fixes symmetric B,C,D and complements
    # A's half-mask, so masks pair (m, ~m) with identical completion outcomes —
    # validated 10082/10082 on the catalogue. Production halves BEFORE the join
    # (codex_review.md #1): restrict the low half-table to indices with bit0==1
    # (f_1=+1 -> a_1=+1), materialized contiguously; the kernel's compact index
    # cg2 maps back to the original low index as 2*cg2+1 (an off-by-one here
    # would silently produce wrong A rows -- gated by set-equality below).
    # --all keeps the full pool so the CPU set-equality gate stays exact.
    halve = not args.all
    counts = cp.zeros(3, dtype=cp.uint32)
    hitbuf = cp.zeros((HIT_CAP, 2), dtype=cp.uint64)
    unsbuf = cp.zeros(UNS_CAP, dtype=cp.uint64)

    t00 = time.time()
    G, Hh = half_tables(n, H, H1)
    if halve:
        Gj = cp.ascontiguousarray(G[1::2])   # low halves with f_1=+1 only
    else:
        Gj = G
    print(f"n={n} H={H}: MITM half tables 2^{H1}/2^{H-H1}"
          + (" (low half a1=+1-restricted)" if halve else "")
          + f" [{time.time()-t00:.1f}s]", flush=True)

    all_sols = []
    exhaustive = (args.start == 0 and args.dmax == 0 and args.pass_sB is None)
    all_pending = []
    sBs = sorted(set(dec[0] for dec in decompositions(n)),
                 key=abs, reverse=True)
    for pass_idx, sB in enumerate(sBs):
        if args.pass_sB is not None and sB != args.pass_sB:
            continue                      # fleet mode: run one named pass only
        # COVERAGE FIX (codex_review.md): --start/--dmax bound ONLY the first
        # executed pass (the resume/chunk target); later passes start at 0 and
        # run in full. Without this, a resume like `--start 290600` silently
        # skipped the first 290,600 classes of every subsequent sB pass while
        # still printing an exhaustion message.
        first = (pass_idx == 0) or (args.pass_sB is not None)
        p_start = args.start if first else 0
        p_dmax = args.dmax if first else 0
        print(f"n={n}: outer B row-sum {sB} (pass {pass_idx}, start={p_start}"
              + (f", dmax={p_dmax}" if p_dmax else "") + ")", flush=True)
        t0 = time.time()
        import glob
        hit = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f"pools_n{n}_{sB}_*.npz"))
        if hit:
            z = np.load(hit[0])          # reuse pair-scan cache (B side + reps)
            rows_B, psd_B, reps = z["rB"], z["pB"], z["reps"]
            print(f"  loaded cached B pool {os.path.basename(hit[0])}", flush=True)
        else:
            rows_B, psd_B, _ = build_sym_pool(H, n, sB)
            viable = np.nonzero(psd_B.max(axis=1) <= 4 * n - 1 + 1e-9)[0]
            masks = np.zeros(len(rows_B), dtype=np.uint64)
            neg = rows_B[:, 1:H + 1] < 0
            for b in range(H):
                masks |= (neg[:, b].astype(np.uint64) << np.uint64(b))
            reps = decim_classes(psd_B, masks, n, H, viable)
        # cheapest-first (tightest budget -> smallest A pool)
        reps = reps[np.argsort(-psd_B[reps].max(axis=1), kind="stable")]
        # coverage identity: digest of the ordered class list (a resume cursor is
        # only meaningful against the identical ordering)
        import hashlib
        digest = hashlib.sha256(np.ascontiguousarray(reps).tobytes()).hexdigest()[:16]
        print(f"  B pool {len(rows_B):,}, classes {len(reps):,}, "
              f"order-digest {digest} [{time.time()-t0:.0f}s]", flush=True)
        n_classes = len(reps)
        if p_dmax:
            reps = reps[:p_dmax]

        t0 = time.time()
        tot_sys = tot_cons = tot_uns = 0
        done = p_start
        pending = []                      # fail-closed: anything not fully settled
        for ci in range(p_start, len(reps)):
            oi = reps[ci]
            Brow = rows_B[oi].astype(np.int64)
            F = np.fft.rfft(Brow)
            psd64 = (np.abs(F) ** 2)[1:H + 1]           # float64, conservative
            R = 4 * n - psd64
            if (R < 1.0 - 1e-9).any():
                done += 1
                continue                                 # PSD_A >= 1 always
            pafB_h = np.rint(np.fft.irfft(np.abs(np.fft.rfft(Brow.astype(np.float64))) ** 2,
                                          n=n))[0:H + 1].astype(np.int32)
            pafB_h[0] = 0
            pafB_g = cp.asarray(pafB_h)
            sbm = 0
            for k in range(n):
                if Brow[k] < 0: sbm |= 1 << k
            sbLo = np.uint64(sbm & ((1 << 64) - 1))
            sbHi = np.uint64(sbm >> 64)

            pool, _M = mitm_pool_fused(Gj, Hh, R, H1, nb=min(64, 1 << H1))
            if halve:
                # remap compact low index back to the original: cg = 2*cg2 + 1
                lm = np.int64((1 << H1) - 1)
                pool = ((pool & lm) * 2 + 1) + ((pool >> H1) << H1)
            Am = (~pool) & np.int64((1 << H) - 1)        # bit=1 in pool means +1
            nA = int(Am.shape[0])
            tot_sys += nA
            if nA:
                Am = Am.astype(cp.uint64)
                counts[:] = 0
                CH = 1 << 22
                for s0 in range(0, nA, CH):
                    blk = Am[s0:s0 + CH]
                    g = (int(blk.shape[0]) + 127) // 128
                    kern((g,), (128,),
                         (blk, np.int64(blk.shape[0]), sbLo, sbHi,
                          pafB_g, ufold_g, np.int32(n), np.int32(H),
                          counts, hitbuf, np.int32(HIT_CAP),
                          unsbuf, np.int32(UNS_CAP)))
                cp.cuda.Device().synchronize()
                ccons, chits, cuns = (int(counts[0]), int(counts[1]),
                                      int(counts[2]))
                tot_cons += ccons; tot_uns += cuns
                if chits:
                    nh = min(chits, HIT_CAP)
                    if chits > HIT_CAP:
                        pending.append(("hit_overflow", int(oi)))
                    hb = cp.asnumpy(hitbuf[:nh])
                    for amask, xmask in hb:
                        A, B, C, D = rows_from_masks(amask, xmask, Brow, n, H)
                        if verify_good(A, B, C, D, n):
                            sol = np.stack([A, B, C, D])
                            if args.all:
                                all_sols.append(sol)
                            else:
                                np.save(f"good_solution_n{n}.npy", sol)
                                print(f"*** VERIFIED GOOD MATRICES n={n} "
                                      f"(completion, class {ci}) "
                                      f"[{time.time()-t0:.0f}s] ***", flush=True)
                                return
                if cuns:
                    nu = min(cuns, UNS_CAP)
                    if cuns > UNS_CAP:
                        pending.append(("uns_overflow", int(oi)))
                    ub = cp.asnumpy(unsbuf[:nu])
                    for amask in ub:
                        for sol in settle_host(amask, Brow, n, H):
                            if args.all:
                                all_sols.append(sol)
                            else:
                                np.save(f"good_solution_n{n}.npy", sol)
                                print(f"*** VERIFIED GOOD MATRICES n={n} "
                                      f"(host completion, class {ci}) ***",
                                      flush=True)
                                return
            done += 1
            if (done - p_start) % 200 == 0:
                el = time.time() - t0
                # per-pass cursor: (n, sB, interval, order digest) — a resume is
                # only valid against the same pass AND the same ordered list
                with open(f"progress_complete_n{n}_sB{sB}.json", "w") as f:
                    json.dump(dict(n=n, sB=sB, start=p_start, done=done,
                                   classes=n_classes, order_digest=digest,
                                   systems=tot_sys,
                                   consistent=tot_cons, unsettled=tot_uns,
                                   pending=pending, elapsed_s=round(el, 1),
                                   rate=round((done - p_start) / el, 3)), f)
                print(f"  {done}/{len(reps)} classes, {tot_sys:,} systems, "
                      f"{tot_cons} consistent, {tot_uns} nullity>{FMAX}, "
                      f"{(done-p_start)/el:.2f} cls/s [{el:.0f}s]", flush=True)
        covered = f"[{p_start},{len(reps)}) of {n_classes}"
        print(f"  sB={sB}: covered {covered}, digest {digest}, {tot_sys:,} "
              f"systems, {tot_cons} consistent, {len(pending)} pending "
              f"[{time.time()-t0:.0f}s]", flush=True)
        all_pending.extend((sB, p) for p in pending)
        if pending:
            print(f"  WARNING pending (not exhausted): {pending[:10]}...",
                  flush=True)
        with open(f"coverage_n{n}_sB{sB}.json", "w") as f:
            json.dump(dict(n=n, sB=sB, start=p_start, end=len(reps),
                           classes=n_classes, order_digest=digest,
                           systems=tot_sys, consistent=tot_cons,
                           unsettled=tot_uns, pending=pending), f)
    if args.all:
        if all_sols:
            np.save(f"good_all_n{n}.npy", np.stack(all_sols))
        print(f"n={n}: --all collected {len(all_sols)} verified solutions",
              flush=True)
    elif exhaustive and not all_pending:
        # full nonexistence claim ONLY for an unrestricted, pending-free run
        print(f"n={n}: ALL outer sums exhausted — no circulant good matrices "
              f"of order {n}", flush=True)
    else:
        why = []
        if not exhaustive:
            why.append(f"restricted run (start={args.start}, dmax={args.dmax}, "
                       f"pass={args.pass_sB})")
        if all_pending:
            why.append(f"{len(all_pending)} pending items")
        print(f"n={n}: scanned range complete, 0 solutions — NOT an exhaustion "
              f"claim ({'; '.join(why)}); combine coverage_n{n}_sB*.json "
              f"receipts for the full claim", flush=True)


if __name__ == "__main__":
    main()
