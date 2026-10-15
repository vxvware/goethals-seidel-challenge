"""MITM construction of the spectrally-filtered SKEW pool (standalone port of
best-matrix-campaign/benchmark/mitm_pool.py, minus its n=57/73 validation mains).

For a skew sequence a = [1, f, -rev(f)] with f in {+-1}^H:
    FFT_a(k) = 1 + 2i * S_k,   S_k = sum_j f_j sin(2*pi*j*k/n)
    PSD_a(k) = 1 + 4*S_k^2                       (linear S in the bits!)
Budget PSD_a(k) <= R(k)  <=>  |S_k| <= T_k = sqrt(R(k)-1)/2.

Split f = (g, h) into H1 low + H2 high bits; range-join half tables on the
tightest frequency; check remaining frequencies in a fused kernel. Output:
bit-indices (bit b <-> f_{b+1}; bit=1 means f=+1), identical to the brute
filtered pool, built in ~2^(H/2)-scale work instead of 2^H.

Good-matrix use: per outer symmetric B class, the skew A budget is
R(k) = 4n - PSD_B(k) (C,D symmetric only guarantee PSD >= 0).
"""
import numpy as np
import cupy as cp

EPS = 1e-9

CHECK_KERNEL = r"""
extern "C" __global__ void check_cands(
    const double* __restrict__ G,   // (mg, H)
    const double* __restrict__ Hh,  // (mh, H)
    const double* __restrict__ T,   // (H) thresholds, |S_k| <= T_k
    const int* __restrict__ kord,   // (H) test order, tightest first
    const long long* __restrict__ cg,
    const long long* __restrict__ ch,
    const long long nc, const int H,
    unsigned char* __restrict__ okout)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    const double* g = G + cg[i] * H;
    const double* h = Hh + ch[i] * H;
    unsigned char ok = 1;
    for (int q = 0; q < H; q++) {
        int k = kord[q];
        double s = g[k] + h[k];
        if (s < -T[k] || s > T[k]) { ok = 0; break; }
    }
    okout[i] = ok;
}
"""
_check_kern = None


def _get_check_kern():
    global _check_kern
    if _check_kern is None:
        _check_kern = cp.RawKernel(CHECK_KERNEL, "check_cands")
    return _check_kern


def sin_table(n, H):
    j = np.arange(1, H + 1)[:, None]
    k = np.arange(1, H + 1)[None, :]
    return np.sin(2.0 * np.pi * j * k / n)  # (H, H) [j-1, k-1]


def half_tables(n, H, H1):
    """G (2^H1, H) and Hh (2^H2, H) float64 GPU tables of half-sums."""
    st = cp.asarray(sin_table(n, H))
    outs = []
    for lo, hb in ((0, H1), (H1, H - H1)):
        idx = cp.arange(1 << hb, dtype=cp.int64)
        bits = ((idx[:, None] >> cp.arange(hb, dtype=cp.int64)) & 1)
        f = (2.0 * bits - 1.0).astype(cp.float64)
        outs.append(f @ st[lo:lo + hb])
    return outs  # G, Hh


FUSED_SRC = r"""
extern "C" __global__ void fused(
    const long long* __restrict__ order, const long long* __restrict__ hp,
    const long long* __restrict__ plo,   const long long* __restrict__ cum,
    long long P,
    const double* __restrict__ G, const double* __restrict__ Hh,
    const double* __restrict__ T, const int* __restrict__ kord,
    int Hf, int H1,
    long long q0, long long qn,
    long long* __restrict__ out, unsigned long long* __restrict__ outcount, long long outcap)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= qn) return;
    long long i = q0 + t;                 // global candidate index in [0,Q)
    // smallest p with cum[p] > i  (cum = inclusive prefix sums of per-query counts)
    long long lo = 0, hi = P;
    while (lo < hi) { long long mid = (lo + hi) >> 1; if (cum[mid] > i) hi = mid; else lo = mid + 1; }
    long long p = lo;
    long long base = (p > 0) ? cum[p - 1] : 0;
    long long gpos = plo[p] + (i - base);
    long long cg = order[gpos];
    long long h  = hp[p];
    const double* g  = G  + cg * Hf;
    const double* hh = Hh + h  * Hf;
    for (int q = 0; q < Hf; ++q) { int k = kord[q]; double s = g[k] + hh[k]; if (s < -T[k] || s > T[k]) return; }
    unsigned long long idx = atomicAdd(outcount, 1ULL);
    if (idx < (unsigned long long)outcap) out[idx] = cg + (h << H1);
}
"""
_fused = None


def _get_fused():
    global _fused
    if _fused is None:
        _fused = cp.RawKernel(FUSED_SRC, "fused")
    return _fused


def mitm_pool_fused(G, Hh, R, H1, nb=64, outcap=1 << 27, qchunk=1 << 28):
    """Same output SET as mitm_pool, built by a fused expand/check/compact kernel
    (port of best-matrix mitm_fused.py, identity gvals): no cg/pos materialization
    over Q~1e9 candidates -- the per-class ~6 GB of expansion traffic that
    dominates good-matrix class cost at depth. Raises on overflow (never truncates)."""
    T = cp.sqrt(cp.maximum(cp.asarray(R) - 1.0 + EPS, 0.0)) / 2.0
    tt = cp.asnumpy(cp.argsort(T)); k1, k2 = int(tt[0]), int(tt[1])
    mg = len(G); g2 = G[:, k2]
    ei = cp.asarray(np.linspace(0, mg - 1, nb + 1).astype(np.int64)[1:-1])
    edges = cp.sort(g2)[ei]; bid = cp.searchsorted(edges, g2).astype(cp.float64)
    g1 = G[:, k1]
    W = 2.0 * (float(cp.abs(g1).max().get()) + float(cp.abs(Hh[:, k1]).max().get())
               + float(T[k1].get()) + 10.0)
    key = bid * W + (g1 + W / 2.0); order = cp.argsort(key); keys = key[order]
    lo1 = -T[k1] - Hh[:, k1]; hi1 = T[k1] - Hh[:, k1]
    bl = cp.searchsorted(edges, -T[k2] - Hh[:, k2]).astype(cp.int64)
    bh = cp.searchsorted(edges, T[k2] - Hh[:, k2], side="right").astype(cp.int64)
    nbins_h = cp.maximum(bh - bl + 1, 0); P = int(nbins_h.sum().get())
    if P == 0:
        return cp.empty(0, dtype=cp.int64), 0
    hp = cp.repeat(cp.arange(len(Hh), dtype=cp.int64), nbins_h)
    pstart = cp.cumsum(nbins_h) - nbins_h
    bp = (cp.arange(P, dtype=cp.int64) - cp.repeat(pstart, nbins_h)
          + cp.repeat(bl, nbins_h)).astype(cp.float64)
    plo = cp.searchsorted(keys, bp * W + (lo1[hp] + W / 2.0)).astype(cp.int64)
    phi = cp.searchsorted(keys, bp * W + (hi1[hp] + W / 2.0), side="right").astype(cp.int64)
    counts = (phi - plo).astype(cp.int64)
    cum = cp.cumsum(counts)                     # inclusive prefix sums, int64
    Q = int(cum[-1].get())
    if Q == 0:
        return cp.empty(0, dtype=cp.int64), 0
    tcpu = cp.asnumpy(T)
    kord_np = np.argsort(tcpu, kind="stable")
    kord_np = np.concatenate([kord_np[kord_np != k1], [k1]])
    kord_g = cp.asarray(kord_np.astype(np.int32)); Tg = cp.asarray(tcpu)
    out = cp.empty(outcap, dtype=cp.int64)
    outcount = cp.zeros(1, dtype=cp.uint64)
    kern = _get_fused()
    q0 = 0
    while q0 < Q:
        qn = min(qchunk, Q - q0)
        blocks = (qn + 255) // 256
        kern((int(blocks),), (256,),
             (order, hp, plo, cum, np.int64(P), G, Hh, Tg, kord_g,
              np.int32(G.shape[1]), np.int32(H1), np.int64(q0), np.int64(qn),
              out, outcount, np.int64(outcap)))
        q0 += qn
    ntot = int(outcount[0].get())
    if ntot > outcap:
        raise RuntimeError(f"fused pool overflow: {ntot} > outcap {outcap}")
    return out[:ntot], Q


def mitm_pool(G, Hh, R, H1, nb=1024, chunk=1 << 24):
    """Bit-indices of all f with PSD(k) <= R(k)+EPS for all k.

    Two-frequency bucket join: g-halves are bucketed by G at the
    2nd-tightest frequency (equal-population bins) and sorted within
    each bucket by G at the tightest frequency, encoded as one global
    monotone composite key. Each h queries only the buckets its k2
    interval intersects, with a range search on k1 inside each — so
    candidate volume scales like s1*s2 instead of s1. Expansion is
    chunked; memory stays bounded for any class size."""
    T = cp.sqrt(cp.maximum(cp.asarray(R) - 1.0 + EPS, 0.0)) / 2.0
    tt = cp.asnumpy(cp.argsort(T))
    k1, k2 = int(tt[0]), int(tt[1])
    mg = len(G)
    g2 = G[:, k2]
    ei = cp.asarray(np.linspace(0, mg - 1, nb + 1).astype(np.int64)[1:-1])
    edges = cp.sort(g2)[ei]
    bid = cp.searchsorted(edges, g2).astype(cp.float64)
    g1 = G[:, k1]
    W = 2.0 * (float(cp.abs(g1).max().get())
               + float(cp.abs(Hh[:, k1]).max().get())
               + float(T[k1].get()) + 10.0)
    key = bid * W + (g1 + W / 2.0)
    order = cp.argsort(key)
    keys = key[order]
    lo1 = -T[k1] - Hh[:, k1]
    hi1 = T[k1] - Hh[:, k1]
    bl = cp.searchsorted(edges, -T[k2] - Hh[:, k2]).astype(cp.int64)
    bh = cp.searchsorted(edges, T[k2] - Hh[:, k2], side="right").astype(cp.int64)
    nbins_h = cp.maximum(bh - bl + 1, 0)
    P = int(nbins_h.sum().get())
    hp = cp.repeat(cp.arange(len(Hh), dtype=cp.int64), nbins_h)
    pstart = cp.cumsum(nbins_h) - nbins_h
    bp = (cp.arange(P, dtype=cp.int64) - cp.repeat(pstart, nbins_h)
          + cp.repeat(bl, nbins_h)).astype(cp.float64)
    plo = cp.searchsorted(keys, bp * W + (lo1[hp] + W / 2.0))
    phi = cp.searchsorted(keys, bp * W + (hi1[hp] + W / 2.0), side="right")
    counts = (phi - plo).astype(cp.int64)
    M = int(counts.sum().get())
    cum = cp.cumsum(counts)
    cum_h = cp.asnumpy(cum)
    kern = _get_check_kern()
    tcpu = cp.asnumpy(T)
    kord_np = np.argsort(tcpu, kind="stable")
    kord_np = np.concatenate([kord_np[kord_np != k1], [k1]])
    kord_g = cp.asarray(kord_np.astype(np.int32))
    Tg = cp.asarray(tcpu)
    out = []
    p0 = 0
    done_before = 0
    while p0 < P:
        p1 = int(np.searchsorted(cum_h, done_before + chunk, side="right"))
        p1 = max(p1, p0 + 1)
        c = counts[p0:p1]
        mblk = int(c.sum().get())
        if mblk:
            st = cp.cumsum(c) - c
            ch = cp.repeat(hp[p0:p1], c)
            pos = (cp.arange(mblk, dtype=cp.int64) - cp.repeat(st, c)
                   + cp.repeat(plo[p0:p1], c))
            cg = order[pos]
            ok8 = cp.empty(mblk, dtype=cp.uint8)
            g = (mblk + 255) // 256
            kern((g,), (256,),
                 (G, Hh, Tg, kord_g, cg, ch, np.int64(mblk),
                  np.int32(G.shape[1]), ok8))
            out.append((cg + (ch << H1))[ok8.view(cp.bool_)])
        done_before += mblk
        p0 = p1
    if not out:
        return cp.empty(0, dtype=cp.int64), M
    return cp.concatenate(out), M
