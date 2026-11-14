/* gmat_c_v3.c -- G-matrices (kkss): exhaustive census by the HKT mod-4 solve.
 *
 * Four circulant (-1,1) matrices of odd order n with A A^T + B B^T + C C^T +
 * D D^T = 4nI, A and B skew (x_0 = 1, x_{n-j} = -x_j), C and D symmetric.
 * Same CLI, environment and output as gmat_c_v2.c:
 *
 *   build: gcc -O3 -march=native -fopenmp [-DNLEV=32] gmat_c_v3.c -lm -o gmat
 *   run:   ./gmat n [threads] [out]
 *   env:   NW_LIST=1, NW_DONLY=i, NW_CCK=k, NW_NODEC=1, GMAT_REV=0/1,
 *          GMAT_CLO/GMAT_CHI (skew-code range of A), NW_REPS=path,
 *          NW_SCALARAND=1 as in v2; new: GMAT_SWAP=0 (keep every (A, B)
 *          ordering), GMAT_GTAB=0/1 (store G(t) in the table), GMAT_SAMPLE=k
 *          (solve every k-th chunk of A codes and project), GMAT_BPIECES=p
 *          GMAT_BPIECE=i (run B rows [i/p, (i+1)/p) of the sorted table).
 *   out:   one line per quadruple found, four words of n '+'/'-' (A B C D).
 *
 * What changed against v2, in order of effect:
 *
 *   swap quotient    A and B are both skew with row sum 1, so (A, B, C, D) and
 *                    (B, A, C, D) are the same class.  A is enumerated as the
 *                    least code of its decimation orbit; the B table is
 *                    sorted by the least code of each row's orbit, and a pair
 *                    is scanned only when that key is at least code(A).  Half
 *                    the pairs go, and no class is lost: of the two skew rows
 *                    of any solution, the one with the smaller orbit minimum
 *                    is the enumerated A.  GMAT_SWAP=0 restores v2's set.
 *   one pass         all row-sum decompositions share one pass over the
 *                    pairs (the skew table does not depend on them); the
 *                    row-sum filter accepts any admissible (c, d).  v2 ran
 *                    the pair loop once per decomposition.
 *   folded test      the stage-2 right-hand side is G(t) XOR X with X from
 *                    two words of A and two of B (wil_hkt2.c), so the table
 *                    test reads no mc and no u values; the B record is 16
 *                    bytes.
 *   elimination      pivoted columns are skipped (wil_elim8), the tail is
 *                    lane-vectorised (wil_tail8).
 *   memory           no per-row PAF or spectrum arrays: B PAFs are
 *                    recomputed for the few row-sum matches; the table build
 *                    is parallel.
 *
 * Every closed form is checked at startup against the reference computation
 * for the order in hand.  Solve calls dropped by the nullity cap are counted
 * and reported; the census is exhaustive only when that count is zero.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(__AVX2__)
#error "gmat_c_v3.c needs AVX2 (build with -march=native on an AVX2 host)"
#endif

static int N, H;
static int NW_DECCANON = 1;
static int USE_SWAP = 1;
#define SCALE 16
#define CEIL16 (4 * N * SCALE)

#include "hkt_solve.h"

#ifndef NLEV
#define NLEV 256
#endif
#ifndef ACHUNK
#define ACHUNK 8192
#endif
#ifndef NW_CHUNK
#define NW_CHUNK 32
#endif

static double wnow(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double tcpun(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#ifdef GM_SPLIT
#include <x86intrin.h>
#define GM_TICK() __rdtsc()
#else
#define GM_TICK() 0ULL
#endif
static unsigned long long C_FILT, C_PRE, C_FLUSH, C_TAIL;

/* ------------------------------------------------- per-row prefix halves

   PAF(k) is congruent to n mod 4 for every lag of a (-1,1) row of odd
   length, so PAF_B(k) = n - 4 u_k with u_k >= 0.  With A_k = -PAF_A(k) - n -
   2 FIXED_[k] the solver's working value is v = A_k + 4 u_k, its stage-1
   right-hand side bit is bit2(A_k) ^ bit0(M1CONST_[k]) ^ bit0(u_k), so
   rhs1 = RA ^ RB and t = T(RA) ^ T(RB) since T is linear.  The stage-2
   target is ta_k + u_k with ta_k = A_k >> 2. */
static inline uint32_t nw_tmap(uint32_t r) {
    uint32_t t = 0;
    for (int b = 0; b < TNB_; b++)
        t ^= TBYTE_[(size_t)b * 256 + ((r >> (8 * b)) & 0xffu)];
    return t;
}
static inline void nw_arow(const int8_t *pafA, int *ta, uint32_t *raraw) {
    uint32_t r = 0;
    for (int k = 1; k <= H; k++) {
        int ak = -(int)pafA[k - 1] - N - 2 * FIXED_[k];
        ta[k] = ak >> 2;
        if ((((ak >> 1) - 2 * M1CONST_[k]) >> 1) & 1) r |= 1u << (k - 1);
    }
    *raraw = r;
}
static inline uint32_t nw_bvec_ref(const int16_t *mc, const int *ta, const int8_t *su) {
    uint32_t bvec = 0;
    for (int kb = 1; kb <= H; kb++)
        if (((mc[kb - 1] - ta[kb] - su[kb - 1]) >> 1) & 1) bvec |= 1u << (kb - 1);
    return bvec;
}

/* ------------------------------------------------- eight-lane elimination

   hkt_elim8v with the columns already pivoted in every lane skipped: a
   pivoted column is a unit vector no later step can touch. */
static inline void gm_elim8(uint32_t *c, uint32_t *bb, uint32_t *pivb, uint32_t *pm) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i p = zero;
    __m256i vb = _mm256_loadu_si256((const __m256i *)bb);
    int pend[64], npend = 0;
    for (int cc = 0; cc < H; cc++) {
        __m256i colv = _mm256_loadu_si256((const __m256i *)(c + (size_t)cc * 8));
        __m256i m    = _mm256_andnot_si256(p, colv);
        __m256i bit  = _mm256_and_si256(m, _mm256_sub_epi32(zero, m));
        __m256i R    = _mm256_xor_si256(colv, bit);
        p = _mm256_or_si256(p, bit);
        _mm256_storeu_si256((__m256i *)(pivb + (size_t)cc * 8), bit);
        for (int k = cc; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * 8));
            __m256i ns = _mm256_cmpeq_epi32(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * 8), v);
        }
        for (int i = 0; i < npend; i++) {
            int k = pend[i];
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * 8));
            __m256i ns = _mm256_cmpeq_epi32(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * 8), v);
        }
        __m256i nb = _mm256_cmpeq_epi32(_mm256_and_si256(vb, bit), zero);
        vb = _mm256_xor_si256(vb, _mm256_andnot_si256(nb, R));
        if (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(bit, zero))))
            pend[npend++] = cc;
    }
    _mm256_storeu_si256((__m256i *)bb, vb);
    _mm256_storeu_si256((__m256i *)pm, p);
}

/* Lane-vectorised tail: part, free-column mask, and the null vector of the
   first free column for all eight lanes.  Lanes with two or more free
   columns are left to hkt_tail_m by the caller. */
static inline void gm_tail8(const uint32_t *ct, const uint32_t *pivb, const uint32_t *bbv,
                            uint32_t *part, uint32_t *fm, uint32_t *basis0, int *nf) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i vb = _mm256_loadu_si256((const __m256i *)bbv);
    __m256i vpart = zero, vfm = zero;
    for (int cc = 0; cc < H; cc++) {
        __m256i bit = _mm256_loadu_si256((const __m256i *)(pivb + (size_t)cc * 8));
        __m256i e   = _mm256_set1_epi32((int)(1u << cc));
        __m256i nz  = _mm256_cmpeq_epi32(_mm256_and_si256(vb, bit), zero);
        vpart = _mm256_or_si256(vpart, _mm256_andnot_si256(nz, e));
        vfm   = _mm256_or_si256(vfm, _mm256_and_si256(_mm256_cmpeq_epi32(bit, zero), e));
    }
    _mm256_storeu_si256((__m256i *)part, vpart);
    _mm256_storeu_si256((__m256i *)fm, vfm);
    uint32_t colf[8];
    for (int L = 0; L < 8; L++) {
        nf[L] = __builtin_popcount(fm[L]);
        int f = fm[L] ? __builtin_ctz(fm[L]) : 0;
        colf[L] = fm[L] ? ct[(size_t)f * 8 + L] : 0u;
    }
    __m256i vc = _mm256_loadu_si256((const __m256i *)colf);
    __m256i vw = _mm256_and_si256(vfm, _mm256_sub_epi32(zero, vfm));
    for (int cc = 0; cc < H; cc++) {
        __m256i bit = _mm256_loadu_si256((const __m256i *)(pivb + (size_t)cc * 8));
        __m256i e   = _mm256_set1_epi32((int)(1u << cc));
        __m256i nz  = _mm256_cmpeq_epi32(_mm256_and_si256(vc, bit), zero);
        vw = _mm256_or_si256(vw, _mm256_andnot_si256(nz, e));
    }
    _mm256_storeu_si256((__m256i *)basis0, vw);
}

/* ------------------------------------------------- the folded right side

   bb = [MC1(t) ^ (D0(t) & ~MC0(t))] ^ [WA1 ^ WB1 ^ (TA0 & ~D0)] with
   D0 = rhs1 ^ M0W, rhs1 = M1FWD t (wil_hkt2.c).  The first bracket is
   G(t), the second is X. */
static uint32_t M0W;
static uint32_t *M1FWD;
static inline uint32_t gf2_parity_mul(const uint32_t *rows, uint32_t v) {
    uint32_t r = 0;
    for (int k = 0; k < H; k++) r |= (uint32_t)__builtin_parity(rows[k] & v) << k;
    return r;
}
static int fwd_init(void) {
    uint32_t inv[64], aug[64];
    for (int k = 0; k < H; k++) { aug[k] = M1INVCOL_[k]; inv[k] = 1u << k; }
    int rank = 0;
    for (int c = 0; c < H; c++) {
        int pr = -1;
        for (int r2 = rank; r2 < H; r2++) if ((aug[r2] >> c) & 1) { pr = r2; break; }
        if (pr < 0) continue;
        uint32_t tt = aug[pr]; aug[pr] = aug[rank]; aug[rank] = tt;
        tt = inv[pr]; inv[pr] = inv[rank]; inv[rank] = tt;
        for (int r2 = 0; r2 < H; r2++)
            if (r2 != rank && ((aug[r2] >> c) & 1)) { aug[r2] ^= aug[rank]; inv[r2] ^= inv[rank]; }
        rank++;
    }
    if (rank != H) return 0;
    M1FWD = calloc(H, sizeof(uint32_t));
    for (int k = 0; k < H; k++) M1FWD[k] = inv[k];
    M0W = 0;
    for (int k = 1; k <= H; k++) if (M1CONST_[k] & 1) M0W |= 1u << (k - 1);
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0x1234567ULL ^ (uint64_t)N;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t r = (uint32_t)(seed & hm);
        if (gf2_parity_mul(M1FWD, nw_tmap(r)) != r) return 0;
    }
    return 1;
}

/* ------------------------------------------------- stage-2 closed forms */
static inline uint64_t nw_uw(uint32_t t) {
    uint32_t rv = t;
    rv = ((rv >> 1) & 0x55555555u) | ((rv & 0x55555555u) << 1);
    rv = ((rv >> 2) & 0x33333333u) | ((rv & 0x33333333u) << 2);
    rv = ((rv >> 4) & 0x0f0f0f0fu) | ((rv & 0x0f0f0f0fu) << 4);
    rv = __builtin_bswap32(rv) >> (32 - H);
    return 1ULL | ((uint64_t)t << 1) | ((uint64_t)rv << (H + 1));
}
static void nw_rows_ref(uint32_t t, int16_t *mc, uint32_t *col) {
    const uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    const uint32_t hmask = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t uw = nw_uw(t);
    uint32_t a[64];
    for (int i = 0; i < 64; i++) a[i] = 0;
    for (int k = 1; k <= H; k++) {
        uint64_t kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
        uint64_t G    = kept & ~CONSTALL_[k] & maskn;
        uint64_t oddu = (G ^ hkt_rot(G, k)) & maskn;
        a[k - 1] = (uint32_t)(oddu >> 1) & hmask;
        mc[k - 1] = (int16_t)(__builtin_popcountll(kept & NFIXM_[k]) / 2);
    }
    hkt_tr(a, H, col);
}
static int nw_leftnull(uint32_t t, uint32_t *out, int maxout) {
    int16_t mc[64]; uint32_t col[64];
    nw_rows_ref(t, mc, col);
    uint32_t bt[32];
    for (int j = 0; j < H; j++) bt[j] = 1u << j;
    uint32_t pv = 0;
    for (int c = 0; c < H; c++) {
        uint32_t m = col[c] & ~pv;
        if (!m) continue;
        int q = __builtin_ctz(m);
        pv |= 1u << q;
        uint32_t R = col[c] & ~(1u << q);
        for (int c2 = 0; c2 < H; c2++)
            col[c2] ^= R & (uint32_t) - (int32_t)((col[c2] >> q) & 1u);
        for (int j = 0; j < H; j++)
            bt[j] ^= R & (uint32_t) - (int32_t)((bt[j] >> q) & 1u);
    }
    int nc = 0;
    for (int q = 0; q < H; q++) {
        if ((pv >> q) & 1u) continue;
        if (nc < maxout) {
            uint32_t colq = 0;
            for (int j = 0; j < H; j++) if ((bt[j] >> q) & 1u) colq |= 1u << j;
            out[nc] = colq;
        }
        nc++;
    }
    return nc;
}
static uint64_t NT1_[64], NT2_[64], CJ_[64];
static __m256i *NT1V_, *NT2V_, *CJV_;
static void cols_init(void) {
    uint64_t hb = 0;
    for (int k = 1; k <= H; k++) hb |= 1ULL << k;
    for (int j = 1; j <= H; j++) {
        uint64_t t1 = 0, t2 = 0;
        for (int k = 1; k <= H; k++) {
            if ((CONSTALL_[k] >> j) & 1) t1 |= 1ULL << k;
            int jj = ((j - k) % N + N) % N;
            if ((CONSTALL_[k] >> jj) & 1) t2 |= 1ULL << k;
        }
        NT1_[j] = ~t1 & hb; NT2_[j] = ~t2 & hb; CJ_[j] = NT1_[j] ^ NT2_[j];
    }
    NT1V_ = aligned_alloc(32, 64 * sizeof(__m256i));
    NT2V_ = aligned_alloc(32, 64 * sizeof(__m256i));
    CJV_  = aligned_alloc(32, 64 * sizeof(__m256i));
    for (int j = 0; j < 64; j++) {
        NT1V_[j] = _mm256_set1_epi64x((long long)NT1_[j]);
        NT2V_[j] = _mm256_set1_epi64x((long long)NT2_[j]);
        CJV_[j]  = _mm256_set1_epi64x((long long)CJ_[j]);
    }
}
static inline void nw_mc_from_t(uint32_t t, int16_t *mc) {
    const uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    uint64_t uw = nw_uw(t);
    for (int k = 1; k <= H; k++) {
        uint64_t r = ((uw >> k) | (uw << (N - k))) & maskn;
        mc[k - 1] = (int16_t)(H - __builtin_popcountll(uw ^ r) / 2);
    }
    for (int k = H; k < 64; k++) mc[k] = 0;
}
static inline void nw_mcbits(uint32_t t, uint32_t *mc0, uint32_t *mc1) {
    const uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    uint64_t uw = nw_uw(t);
    uint32_t a0 = 0, a1 = 0;
    for (int k = 1; k <= H; k++) {
        uint64_t r = ((uw >> k) | (uw << (N - k))) & maskn;
        int m = H - __builtin_popcountll(uw ^ r) / 2;
        a0 |= (uint32_t)(m & 1) << (k - 1);
        a1 |= (uint32_t)((m >> 1) & 1) << (k - 1);
    }
    *mc0 = a0; *mc1 = a1;
}
static inline void nw_mcbits4(const uint32_t *t4, uint32_t *mc0, uint32_t *mc1) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i nib = _mm256_set1_epi8(0x0f), zero = _mm256_setzero_si256();
    const __m256i maskn = _mm256_set1_epi64x((long long)((N >= 64) ? ~0ULL : ((1ULL << N) - 1)));
    const __m256i hh = _mm256_set1_epi64x(H), one = _mm256_set1_epi64x(1);
    __m256i uw = _mm256_setr_epi64x((long long)nw_uw(t4[0]), (long long)nw_uw(t4[1]),
                                    (long long)nw_uw(t4[2]), (long long)nw_uw(t4[3]));
    __m256i a0 = zero, a1 = zero;
    for (int k = 1; k <= H; k++) {
        __m128i ck = _mm_cvtsi32_si128(k), cn = _mm_cvtsi32_si128(N - k), sh = _mm_cvtsi32_si128(k - 1);
        __m256i r = _mm256_and_si256(_mm256_or_si256(_mm256_srl_epi64(uw, ck),
                                                     _mm256_sll_epi64(uw, cn)), maskn);
        __m256i x = _mm256_xor_si256(uw, r);
        __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(x, nib));
        __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), nib));
        __m256i pc = _mm256_sad_epu8(_mm256_add_epi8(lo, hi), zero);
        __m256i m  = _mm256_sub_epi64(hh, _mm256_srli_epi64(pc, 1));
        a0 = _mm256_or_si256(a0, _mm256_sll_epi64(_mm256_and_si256(m, one), sh));
        a1 = _mm256_or_si256(a1, _mm256_sll_epi64(_mm256_and_si256(_mm256_srli_epi64(m, 1), one), sh));
    }
    uint64_t o0[4], o1[4];
    _mm256_storeu_si256((__m256i *)o0, a0);
    _mm256_storeu_si256((__m256i *)o1, a1);
    for (int u = 0; u < 4; u++) { mc0[u] = (uint32_t)o0[u]; mc1[u] = (uint32_t)o1[u]; }
}
static inline void nw_cols_scalar(uint32_t t, uint32_t *col) {
    const uint32_t hmask = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t uw = nw_uw(t);
    for (int j = 1; j <= H; j++) {
        uint64_t P = uw >> j;
        uint64_t Q = (uw << j) | (uw >> (N - j));
        uint64_t s = (uint64_t)0 - ((uw >> j) & 1ULL);
        uint64_t X = (P & NT1_[j]) ^ (Q & NT2_[j]) ^ (CJ_[j] & ~s);
        col[j - 1] = (uint32_t)(X >> 1) & hmask;
    }
}
static inline void nw_cols4(const uint32_t *t4, uint32_t *ct, int L0) {
    const __m128i hmask4 = _mm_set1_epi32((H >= 32) ? (int)0xffffffffu : (int)((1u << H) - 1u));
    const __m256i one = _mm256_set1_epi64x(1), zero = _mm256_setzero_si256();
    const __m256i pick = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    __m256i uw = _mm256_setr_epi64x((long long)nw_uw(t4[0]), (long long)nw_uw(t4[1]),
                                    (long long)nw_uw(t4[2]), (long long)nw_uw(t4[3]));
    for (int j = 1; j <= H; j++) {
        __m128i cj = _mm_cvtsi32_si128(j), cn = _mm_cvtsi32_si128(N - j);
        __m256i P = _mm256_srl_epi64(uw, cj);
        __m256i Q = _mm256_or_si256(_mm256_sll_epi64(uw, cj), _mm256_srl_epi64(uw, cn));
        __m256i s = _mm256_sub_epi64(zero, _mm256_and_si256(P, one));
        __m256i X = _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(P, NT1V_[j]),
                                                      _mm256_and_si256(Q, NT2V_[j])),
                                     _mm256_andnot_si256(s, CJV_[j]));
        __m256i Y = _mm256_permutevar8x32_epi32(_mm256_srli_epi64(X, 1), pick);
        __m128i lo = _mm_and_si128(_mm256_castsi256_si128(Y), hmask4);
        _mm_storeu_si128((__m128i *)(ct + (size_t)(j - 1) * 8 + L0), lo);
    }
}
static void cols_selfcheck(void) {
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0x2545F4914F6CDD1DULL ^ (uint64_t)N;
    for (int i = 0; i < 2048; i++) {
        uint32_t t4[4];
        for (int u = 0; u < 4; u++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            t4[u] = (uint32_t)(seed & hm);
        }
        uint32_t ct[64 * 8]; uint32_t q0[4], q1[4];
        nw_cols4(t4, ct, 0);
        nw_mcbits4(t4, q0, q1);
        for (int u = 0; u < 4; u++) {
            int16_t m1[64], m2[64]; uint32_t c1[64], c2[64], b0, b1;
            nw_rows_ref(t4[u], m1, c1);
            nw_mc_from_t(t4[u], m2);
            nw_cols_scalar(t4[u], c2);
            nw_mcbits(t4[u], &b0, &b1);
            for (int k = 0; k < H; k++) {
                int bad = (m1[k] != m2[k]) || (c1[k] != c2[k]) || (ct[(size_t)k * 8 + u] != c1[k]);
                bad |= (((b0 >> k) & 1) != (uint32_t)(m1[k] & 1)) || (((b1 >> k) & 1) != (uint32_t)((m1[k] >> 1) & 1));
                bad |= (((q0[u] >> k) & 1) != (uint32_t)(m1[k] & 1)) || (((q1[u] >> k) & 1) != (uint32_t)((m1[k] >> 1) & 1));
                if (bad) {
                    fprintf(stderr, "n=%d: direct column build disagrees with the row build at t=%u lag %d\n",
                            N, t4[u], k + 1);
                    exit(1);
                }
            }
        }
    }
}

/* ------------------------------------------------- consistency table

   K uint32 words per pattern t: a left null vector of the stage-2 matrix in
   bits 0..H-1 with the parity of G(t) against it in bit 31 (CC_G); a zero
   word ends the list.  For H <= 30, 0xffffffff in word 0 marks a pattern of
   nullity above the cap, dropped outright.  At H = 31 there is no room for
   that mark and such patterns go to the elimination, which drops them.
   Optionally (GTAB) the entry also stores G(t) after the K words. */
/* Nullity cap.  hkt_solve.h caps the stage-2 nullity at 6 (64 candidates),
   and at the composite orders the corank runs higher: sampled coranks are
   up to 5 at 45, 51, 57 and up to 7 at 63 (37% of patterns there), so the
   header's cap would silently drop those systems.  v3 carries its own tail
   with the cap at GM_MAXFREE and counts every system still above it. */
#ifndef GM_MAXFREE
#define GM_MAXFREE 12
#endif
#define GM_MAXC (1 << GM_MAXFREE)
#define S2_MAXFREE GM_MAXFREE
#define CC_REJECT 0xffffffffu
#define CC_G      0x80000000u
static uint32_t *CC = NULL;
static int CCK = 0, GTAB = 0, CCS = 0;
static long long CC_REJECTED = 0;

static inline uint32_t nw_G(uint32_t t) {
    uint32_t mc0, mc1;
    nw_mcbits(t, &mc0, &mc1);
    uint32_t d0 = gf2_parity_mul(M1FWD, t) ^ M0W;
    return mc1 ^ (d0 & ~mc0);
}
static int cc_choose_k(void) {
    long hist[40] = {0}; int ns = 4096;
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    uint32_t tmp[64];
    for (int i = 0; i < ns; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        int c = nw_leftnull((uint32_t)(seed & hm), tmp, 64);
        hist[c > S2_MAXFREE ? S2_MAXFREE : c]++;
    }
    long acc = 0;
    for (int k = 0; k <= S2_MAXFREE; k++) {
        acc += hist[k];
        if (acc * 100 >= ns * 99) return k > 0 ? k : 1;
    }
    return S2_MAXFREE;
}
static void cc_build(void) {
    size_t nt = (size_t)1 << H;
    CCS = CCK + GTAB;
    size_t bytes = nt * (size_t)CCS * sizeof(uint32_t);
    bytes = (bytes + (1u << 21) - 1) & ~(size_t)((1u << 21) - 1);
    CC = aligned_alloc(1u << 21, bytes);
    if (!CC) {
        fprintf(stderr, "out of memory for the consistency table (%.2f GB)\n",
                nt * (double)CCS * 4 / 1073741824.0);
        exit(1);
    }
    madvise(CC, bytes, MADV_HUGEPAGE);
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    long long rej = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+:rej)
    for (long long ti = 0; ti < (long long)nt; ti++) {
        uint32_t t = (uint32_t)ti;
        uint32_t nv[S2_MAXFREE + 1];
        uint32_t *e = CC + (size_t)t * CCS;
        int c = nw_leftnull(t, nv, S2_MAXFREE + 1);
        if (c > S2_MAXFREE) {
            rej++;
            if (H <= 30) { e[0] = CC_REJECT; for (int i = 1; i < CCS; i++) e[i] = 0; continue; }
            for (int i = 0; i < CCS; i++) e[i] = 0;   /* no mark at H = 31: the elimination drops it */
            if (GTAB) e[CCK] = nw_G(t);
            continue;
        }
        uint32_t G = nw_G(t);
        if (GTAB) e[CCK] = G;
        for (int i = 0; i < CCK; i++) {
            if (i < c) {
                uint32_t w = nv[i] & hm;
                uint32_t ew = w | (__builtin_parity(w & G) ? CC_G : 0u);
                if (H <= 30 && ew == CC_REJECT) ew = 0;   /* cannot happen for H <= 30 (bit 30 clear) */
                e[i] = ew;
            } else e[i] = 0;
        }
    }
    CC_REJECTED = rej;
}

/* hkt_tail_m with the cap at GM_MAXFREE; cand holds GM_MAXC words. */
static inline int gm_tail_m(const uint32_t *col, uint32_t bb, uint32_t pivoted,
                            const uint32_t *pivb, uint32_t *cand) {
    if (bb & ~pivoted) return 0;
    int nul = H - __builtin_popcount(pivoted);
    if (nul > GM_MAXFREE) return -1;
    uint32_t part = 0;
    for (int c = 0; c < H; c++)
        if (bb & pivb[c]) part |= 1u << c;
    uint32_t basis[64]; int nf = 0;
    for (int c = 0; c < H; c++) {
        if (pivb[c]) continue;
        uint32_t w = 1u << c;
        for (int c2 = 0; c2 < H; c2++)
            if (col[c] & pivb[c2]) w |= 1u << c2;
        basis[nf++] = w;
    }
    int ncand = 1 << nf;
    cand[0] = part;
    for (int i = 0; i < nf; i++)                 /* Gray-free doubling: block i copies block < i */
        for (int m = 0; m < (1 << i); m++) cand[(1 << i) + m] = cand[m] ^ basis[i];
    return ncand;
}

/* ------------------------------------------------------------ row filter

   ALLOWD[pc] is the set of popcounts pd such that (pc, pd) are the popcounts
   of some admissible (C, D).  For pc == pd (c == d) the candidate set holds
   both (C, D) and (D, C); codeC <= codeD keeps one. */
static uint32_t ALLOWD[64];
static inline int gm_keep(uint32_t v, uint32_t s0, uint32_t maskH) {
    int a = H - __builtin_popcount(v & maskH);
    uint32_t am = ALLOWD[a];
    if (!am) return 0;
    uint32_t cD = (s0 ^ v) & maskH;
    int b = __builtin_popcount(cD);
    if (!((am >> b) & 1u)) return 0;
    return a != b || ((~v) & maskH) <= cD;
}
static inline int gm_rowfilter(uint32_t *cand, int nc, uint32_t s, uint32_t maskH) {
    int keep = 0;
    for (int i = 0; i < nc; i++)
        if (gm_keep(cand[i], s, maskH)) cand[keep++] = cand[i];
    return keep;
}

/* ------------------------------------------------------------------ rows */
static inline uint64_t skew_mask(uint32_t code) {
    uint64_t m = 1ULL;
    for (int j = 1; j <= H; j++) {
        if ((code >> (j - 1)) & 1) m |= 1ULL << j;
        else                       m |= 1ULL << (N - j);
    }
    return m;
}
static inline uint64_t sym_mask(uint32_t code) {
    uint64_t m = 1ULL;
    for (int j = 1; j <= H; j++)
        if ((code >> (j - 1)) & 1) { m |= 1ULL << j; m |= 1ULL << (N - j); }
    return m;
}
static inline void paf_mask(uint64_t m, int8_t *paf) {
    uint64_t full = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    for (int s = 1; s <= H; s++) {
        uint64_t r = ((m >> s) | (m << (N - s))) & full;
        paf[s - 1] = (int8_t)(N - 2 * (int)__builtin_popcountll(m ^ r));
    }
}
static double *COSTAB;
static void cos_init(void) {
    COSTAB = malloc(sizeof(double) * (size_t)(H + 1) * H);
    for (int k = 0; k <= H; k++)
        for (int s = 1; s <= H; s++)
            COSTAB[(size_t)k * H + s - 1] = 2.0 * cos(2.0 * M_PI * s * k / N);
}
static inline double psd_at(const int8_t *paf, int k) {
    double acc = N;
    const double *row = COSTAB + (size_t)k * H;
    for (int s = 0; s < H; s++) acc += paf[s] * row[s];
    return acc;
}
static double psd_all(const int8_t *paf, float *out) {
    double m = -1e18;
    for (int k = 1; k <= H; k++) {
        double v = psd_at(paf, k);
        out[k - 1] = (float)v;
        if (v > m) m = v;
    }
    return m;
}

/* Folded right-hand side against the reference on random skew row pairs. */
static void fold_selfcheck(void) {
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0xD1B54A32D192ED03ULL ^ (uint64_t)N;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t ca = (uint32_t)(seed & hm);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t cb = (uint32_t)(seed & hm);
        int8_t pa[64], pb[64], su[64]; int ta[64];
        paf_mask(skew_mask(ca), pa); paf_mask(skew_mask(cb), pb);
        uint32_t raraw; nw_arow(pa, ta, &raraw);
        uint32_t wa1 = 0, ta0 = 0, rbraw = 0, wb1 = 0;
        for (int k = 1; k <= H; k++) {
            if (ta[k] & 1) ta0 |= 1u << (k - 1);
            if (ta[k] & 2) wa1 |= 1u << (k - 1);
            int u = (N - (int)pb[k - 1]) >> 2;
            su[k - 1] = (int8_t)u;
            if (u & 1) rbraw |= 1u << (k - 1);
            if (u & 2) wb1 |= 1u << (k - 1);
        }
        uint32_t rhs1 = raraw ^ rbraw;
        uint32_t t = nw_tmap(rhs1);
        int16_t mc[64]; nw_mc_from_t(t, mc);
        uint32_t ref = nw_bvec_ref(mc, ta, su);
        uint32_t d0 = rhs1 ^ M0W;
        uint32_t X = wa1 ^ wb1 ^ (ta0 & ~d0);
        uint32_t got = (nw_G(t) ^ X) & hm;
        if (got != (ref & hm) || gf2_parity_mul(M1FWD, t) != rhs1) {
            fprintf(stderr, "n=%d: folded right-hand side disagrees with nw_bvec (A=%u B=%u ref=%08x got=%08x)\n",
                    N, ca, cb, ref, got);
            exit(1);
        }
    }
}

/* -------------------------------------------- decimation on skew codes

   x'_j = x_{tj mod n} for a unit t.  For j in 1..H let m = tj mod n: if
   m <= H the new code bit j-1 is the old bit m-1, otherwise x_m = -x_{n-m}
   and the bit is complemented.  So a decimation is a bit permutation
   followed by an XOR with a fixed flip word, tabulated by bytes.  Unit t =
   n-1 is the reversal, so reversal is in the quotient; rotation does not
   preserve skewness and is not. */
static int NDEC;                 /* phi(n), including the identity at index 0 */
static int *DECUNIT;
static uint32_t *DBYTE;          /* [NDEC][DNB][256] */
static uint32_t *DFLIP;          /* [NDEC] */
static int DNB;
static int gcd_int(int a, int b) { while (b) { int q = a % b; a = b; b = q; } return a; }
static void dec_init(void) {
    int cnt = 0;
    for (int t = 1; t < N; t++) if (gcd_int(t, N) == 1) cnt++;
    NDEC = cnt;
    DECUNIT = malloc(sizeof(int) * (size_t)cnt);
    DNB = (H + 7) / 8;
    DBYTE = calloc((size_t)NDEC * DNB * 256, sizeof(uint32_t));
    DFLIP = calloc((size_t)NDEC, sizeof(uint32_t));
    if (!DBYTE || !DFLIP || !DECUNIT) { fprintf(stderr, "out of memory for the decimation tables\n"); exit(1); }
    int d = 0;
    for (int t = 1; t < N; t++) {
        if (gcd_int(t, N) != 1) continue;
        DECUNIT[d] = t;
        int src[64]; uint32_t flip = 0;
        for (int j = 1; j <= H; j++) {
            int m = (int)(((long long)t * j) % N);
            if (m <= H) src[j - 1] = m - 1;
            else { src[j - 1] = N - m - 1; flip |= 1u << (j - 1); }
        }
        uint32_t dest_of_src[64];
        for (int p = 0; p < H; p++) dest_of_src[p] = 0;
        for (int j = 0; j < H; j++) dest_of_src[src[j]] |= 1u << j;
        for (int b = 0; b < DNB; b++)
            for (int v = 0; v < 256; v++) {
                uint32_t acc = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int pos = b * 8 + bit;
                    if (pos < H && ((v >> bit) & 1)) acc |= dest_of_src[pos];
                }
                DBYTE[(((size_t)d * DNB) + b) * 256 + v] = acc;
            }
        DFLIP[d] = flip;
        d++;
    }
}
static inline uint32_t dec_code(uint32_t code, int d) {
    const uint32_t *t = DBYTE + (size_t)d * DNB * 256;
    uint32_t r = 0;
    for (int b = 0; b < DNB; b++) r |= t[(size_t)b * 256 + ((code >> (8 * b)) & 0xffu)];
    return r ^ DFLIP[d];
}
static inline int a_is_canonical(uint32_t code) {
    for (int d = 1; d < NDEC; d++)
        if (dec_code(code, d) < code) return 0;
    return 1;
}
static inline uint32_t orbit_min(uint32_t code) {
    uint32_t m = code;
    for (int d = 1; d < NDEC; d++) { uint32_t r = dec_code(code, d); if (r < m) m = r; }
    return m;
}
/* startup check of the code decimation against the mask form */
static void dec_selfcheck(void) {
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0xA5A5A5A5DEADBEEFULL ^ (uint64_t)N;
    for (int i = 0; i < 512; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t c = (uint32_t)(seed & hm);
        uint64_t m = skew_mask(c);
        for (int d = 0; d < NDEC; d++) {
            int t = DECUNIT[d];
            uint64_t r = 0;
            for (int j = 0; j < N; j++) if ((m >> ((t * j) % N)) & 1) r |= 1ULL << j;
            uint32_t rc = 0;
            for (int j = 1; j <= H; j++) if ((r >> j) & 1) rc |= 1u << (j - 1);
            if ((r & 1) == 0 || rc != dec_code(c, d) || skew_mask(rc) != r) {
                fprintf(stderr, "n=%d: code decimation disagrees with the mask form (code %u, unit %d)\n", N, c, t);
                exit(1);
            }
        }
    }
}

/* ------------------------------------------------------------ row sums */
typedef struct { int a, b, c, d; } Decomp;
static int sym_popcount_for(int s) {
    int num = s - 1 + 2 * H;
    if (num < 0 || (num & 3)) return -1;
    int p = num / 4;
    return (p >= 0 && p <= H) ? p : -1;
}
static int decomps(Decomp *out, int maxout) {
    int nd = 0, target = 4 * N - 2;
    int lim = (int)(sqrt((double)target) + 1e-9);
    for (int c = -lim; c <= lim; c++) {
        if (sym_popcount_for(c) < 0) continue;
        for (int d = -lim; d <= c; d++) {
            if (sym_popcount_for(d) < 0) continue;
            if (c * c + d * d != target) continue;
            if (nd < maxout) { out[nd].a = 1; out[nd].b = 1; out[nd].c = c; out[nd].d = d; nd++; }
        }
    }
    return nd;
}

/* --------------------------------------------------------- the B table

   Every skew code that passes the PSD ceiling, sorted by the least code of
   its decimation orbit (the swap key), with one 16-byte record per row and
   the quantised-spectrum bitmap index of v2 over the sorted positions. */
static int lev_of(double x, int fourn) {
    int v = (int)(x * NLEV / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > NLEV - 1) v = NLEV - 1;
    return v;
}
typedef struct { uint32_t rbt, rbraw, wb1, pad; } BRec;
typedef struct {
    int n;                 /* rows in this piece */
    long long nall;        /* rows passing the ceiling, all pieces */
    uint32_t *code;        /* [n], table order */
    uint32_t *key;         /* [n], ascending */
    BRec     *rec;         /* [n] */
    int cwords;            /* padded to a multiple of 4 */
    uint64_t *bmp;         /* [H][NLEV][cwords], prefix-ored over levels */
} Table;
typedef struct { uint32_t key, code; } KC;
static int cmp_kc(const void *x, const void *y) {
    const KC *a = x, *b = y;
    if (a->key != b->key) return a->key < b->key ? -1 : 1;
    return a->code < b->code ? -1 : (a->code > b->code);
}

static int BSUB_M = 1, BSUB_R = 0;   /* GMAT_BSUB=m,r: keep codes with code mod m == r (sampling only) */
static void table_build(Table *T, int fourn, int npieces, int piece) {
    long long ncomb = 1LL << H;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    /* two passes over the skew codes: count the rows each thread range keeps,
       then fill the key array in place, so the peak is the array itself */
    long long *ln = calloc(nth + 1, sizeof(long long));
    long long per = (ncomb + nth - 1) / nth;
#pragma omp parallel num_threads(nth)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        long long lo = (long long)t * per, hi = lo + per;
        if (hi > ncomb) hi = ncomb;
        long long cnt = 0;
        int8_t paf[64]; float psd[64];
        for (long long ci = lo; ci < hi; ci++) {
            if (BSUB_M > 1 && (ci % BSUB_M) != BSUB_R) continue;
            paf_mask(skew_mask((uint32_t)ci), paf);
            if (!(psd_all(paf, psd) > fourn + 1e-6)) cnt++;
        }
        ln[t] = cnt;
    }
    long long total = 0;
    for (int t = 0; t < nth; t++) { long long c = ln[t]; ln[t] = total; total += c; }
    ln[nth] = total;
    if (total > 0x7fffffffLL) { fprintf(stderr, "B table too large for int indices\n"); exit(1); }
    KC *kc = malloc(sizeof(KC) * (size_t)(total + 1));
    if (!kc) { fprintf(stderr, "out of memory for the B keys (%lld rows)\n", total); exit(1); }
#pragma omp parallel num_threads(nth)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        long long lo = (long long)t * per, hi = lo + per;
        if (hi > ncomb) hi = ncomb;
        long long off = ln[t];
        int8_t paf[64]; float psd[64];
        for (long long ci = lo; ci < hi; ci++) {
            if (BSUB_M > 1 && (ci % BSUB_M) != BSUB_R) continue;
            paf_mask(skew_mask((uint32_t)ci), paf);
            if (!(psd_all(paf, psd) > fourn + 1e-6)) kc[off++].code = (uint32_t)ci;
        }
        if (off != ln[t + 1]) { fprintf(stderr, "B table count mismatch\n"); exit(1); }
    }
    free(ln);
#pragma omp parallel for schedule(dynamic, 4096)
    for (long long i = 0; i < total; i++)
        kc[i].key = USE_SWAP ? orbit_min(kc[i].code) : 0u;
    qsort(kc, (size_t)total, sizeof(KC), cmp_kc);
    T->nall = total;
    long long plo = total * piece / npieces, phi = total * (piece + 1) / npieces;
    int n = (int)(phi - plo);
    T->n = n;
    T->code = malloc(sizeof(uint32_t) * (size_t)(n + 1));
    T->key  = malloc(sizeof(uint32_t) * (size_t)(n + 1));
    for (int i = 0; i < n; i++) { T->code[i] = kc[plo + i].code; T->key[i] = kc[plo + i].key; }
    free(kc);
    {   size_t rb = ((size_t)(n + 4) * sizeof(BRec) + (1u << 21) - 1) & ~(size_t)((1u << 21) - 1);
        T->rec = aligned_alloc(1u << 21, rb);
        if (T->rec) madvise(T->rec, rb, MADV_HUGEPAGE); }
    int cw = (n + 63) >> 6;
    size_t cw4 = ((size_t)cw + 3) & ~(size_t)3;
    size_t bmpn = (size_t)H * NLEV * cw4;
    T->bmp = calloc(bmpn + 4, sizeof(uint64_t));
    if (!T->code || !T->key || !T->rec || !T->bmp) {
        fprintf(stderr, "out of memory for the B table (%d rows, index %.1f GB)\n",
                n, bmpn * 8.0 / (1024.0 * 1024.0 * 1024.0));
        exit(1);
    }
    T->cwords = (int)cw4;
#pragma omp parallel for schedule(dynamic, 4)
    for (int w = 0; w < cw; w++) {
        int8_t paf[64]; float psd[64];
        int hi = (w + 1) * 64; if (hi > n) hi = n;
        for (int bi = w * 64; bi < hi; bi++) {
            paf_mask(skew_mask(T->code[bi]), paf);
            psd_all(paf, psd);
            uint32_t rb = 0, wb1 = 0;
            for (int k = 1; k <= H; k++) {
                int u = (N - (int)paf[k - 1]) >> 2;
                if (u & 1) rb |= 1u << (k - 1);
                if (u & 2) wb1 |= 1u << (k - 1);
            }
            T->rec[bi].rbt = nw_tmap(rb);
            T->rec[bi].rbraw = rb;
            T->rec[bi].wb1 = wb1;
            T->rec[bi].pad = 0;
            for (int k = 0; k < H; k++) {
                int lv = lev_of(psd[k], fourn);
                T->bmp[(((size_t)k * NLEV) + lv) * T->cwords + w] |= 1ULL << (bi & 63);
            }
        }
    }
#pragma omp parallel for schedule(static)
    for (int k = 0; k < H; k++)
        for (int t = 1; t < NLEV; t++) {
            uint64_t *dst = T->bmp + (((size_t)k * NLEV) + t) * T->cwords;
            const uint64_t *src = dst - T->cwords;
            for (int w = 0; w < T->cwords; w++) dst[w] |= src[w];
        }
}
static void table_free(Table *T) {
    free(T->code); free(T->key); free(T->rec); free(T->bmp);
    memset(T, 0, sizeof *T);
}
static int lower_bound_u32(const uint32_t *a, int n, uint32_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}

/* ------------------------------------------------- class canonical form

   v2's form for the class count: decimate all four rows together, put each
   in its least rotation/negation (and reversal unless GMAT_REV=0), sort. */
static int USE_REV = 1;
static uint64_t nw_rot(uint64_t m, int s) {
    uint64_t full = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    return ((m >> s) | (m << (N - s))) & full;
}
static uint64_t row_rev(uint64_t m) {
    uint64_t r = m & 1ULL;
    for (int j = 1; j < N; j++) if ((m >> j) & 1) r |= 1ULL << (N - j);
    return r;
}
static uint64_t nw_row_canon(uint64_t m) {
    uint64_t full = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    uint64_t best = ~0ULL;
    uint64_t src[2] = { m, row_rev(m) };
    int nsrc = USE_REV ? 2 : 1;
    for (int t = 0; t < nsrc; t++)
        for (int neg = 0; neg < 2; neg++) {
            uint64_t y = neg ? ((~src[t]) & full) : src[t];
            for (int s = 0; s < N; s++) {
                uint64_t r = nw_rot(y, s);
                if (r < best) best = r;
            }
        }
    return best;
}
static uint64_t nw_decimate_mask(uint64_t m, int t) {
    uint64_t r = 0;
    for (int j = 0; j < N; j++) if ((m >> ((t * j) % N)) & 1) r |= 1ULL << j;
    return r;
}
static void nw_canon(const uint64_t *q, uint64_t *out) {
    uint64_t best[4]; int have = 0;
    for (int i = 0; i < NDEC; i++) {
        uint64_t r[4];
        for (int u = 0; u < 4; u++) r[u] = nw_row_canon(nw_decimate_mask(q[u], DECUNIT[i]));
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3 - a; b++)
                if (r[b] > r[b + 1]) { uint64_t t = r[b]; r[b] = r[b + 1]; r[b + 1] = t; }
        if (!have || memcmp(r, best, sizeof best) < 0) { memcpy(best, r, sizeof best); have = 1; }
    }
    memcpy(out, best, sizeof best);
}
static uint64_t *HSET; static size_t HCAP, HCNT;
static void hset_init(size_t cap) { HCAP = cap; HCNT = 0; HSET = calloc(HCAP * 4, sizeof(uint64_t)); }
static int hset_add(const uint64_t *k) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < 4; i++) { h ^= k[i]; h *= 1099511628211ULL; }
    size_t i = (size_t)(h & (HCAP - 1));
    for (;;) {
        uint64_t *slot = HSET + i * 4;
        if (!slot[0] && !slot[1] && !slot[2] && !slot[3]) {
            memcpy(slot, k, 4 * sizeof(uint64_t)); HCNT++; return 1;
        }
        if (!memcmp(slot, k, 4 * sizeof(uint64_t))) return 0;
        i = (i + 1) & (HCAP - 1);
    }
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gmat <n> [threads] [out]\n"); return 1; }
    N = atoi(argv[1]);
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    const char *outpath = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    if (N < 3 || (N % 2) == 0) { fprintf(stderr, "n must be odd and >= 3\n"); return 1; }
    H = (N - 1) / 2;
    if (N > 63) { fprintf(stderr, "n > 63 needs wider words\n"); return 1; }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    cos_init();
    if (!hkt_build()) { fprintf(stderr, "hkt_build failed\n"); return 1; }
    {   uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
        unsigned seed = 12345u;
        for (int trial = 0; trial < 4096; trial++) {
            seed = seed * 1103515245u + 12345u; uint32_t x = (seed >> 8) & hm;
            seed = seed * 1103515245u + 12345u; uint32_t y = (seed >> 8) & hm;
            if (nw_tmap(x ^ y) != (nw_tmap(x) ^ nw_tmap(y))) {
                fprintf(stderr, "n=%d: the t map is not linear\n", N);
                return 1;
            }
        }
    }
    for (int k = 1; k <= H; k++)
        if (((FIXED_[k] + N) & 1) != 0) {
            fprintf(stderr, "n=%d: FIXED_[%d]=%d breaks the mod-4 invariant\n", N, k, FIXED_[k]);
            return 1;
        }
    if (!fwd_init()) { fprintf(stderr, "n=%d: stage-1 forward matrix failed\n", N); return 1; }
    cols_init();
    cols_selfcheck();
    fold_selfcheck();
    dec_init();
    dec_selfcheck();
    { const char *cv = getenv("NW_NODEC"); if (cv) NW_DECCANON = 0; }
    { const char *rv = getenv("GMAT_REV"); if (rv) USE_REV = atoi(rv); }
    { const char *sv = getenv("GMAT_SWAP"); if (sv) USE_SWAP = atoi(sv) ? 1 : 0; }
    if (!NW_DECCANON) USE_SWAP = 0;    /* the swap key is the orbit minimum, which needs a canonical A */
    int scalar_and = 0;
    { const char *v = getenv("NW_SCALARAND"); if (v) scalar_and = atoi(v); }
    int sample = 1;
    { const char *v = getenv("GMAT_SAMPLE"); if (v) { sample = atoi(v); if (sample < 1) sample = 1; } }
    { const char *v = getenv("GMAT_BSUB"); if (v) { if (sscanf(v, "%d,%d", &BSUB_M, &BSUB_R) != 2 || BSUB_M < 1 || BSUB_R < 0 || BSUB_R >= BSUB_M) { fprintf(stderr, "GMAT_BSUB wants m,r\n"); return 1; } } }
    int npieces = 1, piece = 0;
    { const char *v = getenv("GMAT_BPIECES"); if (v) { npieces = atoi(v); if (npieces < 1) npieces = 1; }
      const char *w = getenv("GMAT_BPIECE"); if (w) { piece = atoi(w); if (piece < 0 || piece >= npieces) { fprintf(stderr, "GMAT_BPIECE out of range\n"); return 1; } } }

    int fourn = 4 * N;
    printf("n=%d: 4n = %d, H = %d, %d decimations\n", N, fourn, H, NDEC);
    Decomp dec[64];
    int nd = decomps(dec, 64);
    printf("   %d row-sum decompositions\n", nd);
    int donly = -1;
    { const char *dv = getenv("NW_DONLY"); if (dv) { donly = atoi(dv); printf("   running decomposition %s only\n", dv); } }
    if (getenv("NW_LIST")) {
        for (int i = 0; i < nd; i++)
            printf("   decomposition %d: a=%d b=%d c=%d d=%d\n", i, dec[i].a, dec[i].b, dec[i].c, dec[i].d);
        return 0;
    }
    memset(ALLOWD, 0, sizeof ALLOWD);
    for (int i = 0; i < nd; i++) {
        if (donly >= 0 && i != donly) continue;
        int pc = sym_popcount_for(dec[i].c), pdd = sym_popcount_for(dec[i].d);
        ALLOWD[pc] |= 1u << pdd; ALLOWD[pdd] |= 1u << pc;
        printf("   decomposition %d: c=%d d=%d (popcounts %d %d)\n", i, dec[i].c, dec[i].d, pc, pdd);
    }
    if (!nd) { printf("n=%d: 0 quadruples, no admissible row sums\n", N); if (outpath) { FILE *f = fopen(outpath, "w"); if (f) fclose(f); } return 0; }
    printf("   swap quotient %s, A canonical under decimation %s\n", USE_SWAP ? "on (B key >= code(A))" : "off", NW_DECCANON ? "on" : "off");
    if (npieces > 1) printf("   B piece %d of %d\n", piece, npieces);
    if (BSUB_M > 1) printf("   SAMPLING: B table restricted to codes = %d mod %d; not a census\n", BSUB_R, BSUB_M);
    if (sample > 1) printf("   sampling: pair filter and solve on every %d-th chunk of %d A codes\n", sample, ACHUNK);
    fflush(stdout);

    double t0 = wnow();
    {
        int kk = -1;
        const char *v = getenv("NW_CCK"); if (v) kk = atoi(v);
        if (kk < 0) kk = cc_choose_k();
        if (kk > S2_MAXFREE) kk = S2_MAXFREE;
        CCK = kk;
        GTAB = (CCK > 0 && H <= 26) ? 1 : 0;
        const char *gv = getenv("GMAT_GTAB"); if (gv && CCK > 0) GTAB = atoi(gv) ? 1 : 0;
    }
    if (CCK) {
        double tb = wnow();
        cc_build();
        printf("   consistency table: K=%d words%s per pattern, %.1f MB, %lld patterns nullity-capped, built in %.2fs\n",
               CCK, GTAB ? " + G(t)" : "", (double)CCS * 4.0 * (double)((size_t)1 << H) / 1048576.0, CC_REJECTED, wnow() - tb);
    } else {
        printf("   no consistency table: every survivor goes to the elimination\n");
    }
    fflush(stdout);

    Table T;
    double ttb = wnow();
    table_build(&T, fourn, npieces, piece);
    double idx_mb = (double)H * NLEV * T.cwords * 8.0 / 1048576.0;
    printf("   skew codes passing the ceiling: %lld of %lld (%.1f%%); this piece %d rows, index %.0f MB, records %.0f MB, built in %.1fs\n",
           T.nall, 1LL << H, 100.0 * T.nall / (double)(1LL << H), T.n, idx_mb,
           (double)T.n * sizeof(BRec) / 1048576.0, wnow() - ttb);
    fflush(stdout);

    long long ncomb = 1LL << H;
    long long CLO = 0, CHI = ncomb;
    {   const char *v = getenv("GMAT_CLO"); if (v) CLO = atoll(v);
        const char *w = getenv("GMAT_CHI");
        if (w) { long long h = atoll(w); if (h >= 0 && h < CHI) CHI = h; }
        if (CLO < 0) CLO = 0;
        if (CLO || CHI != ncomb) fprintf(stderr, "   skew range [%lld, %lld) of %lld\n", CLO, CHI, ncomb);
    }
    const uint32_t maskH = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    long long solcap = 1024, nsol = 0;
    uint64_t *sol = malloc((size_t)solcap * 4 * sizeof(uint64_t));
    long long nA = 0, nAs = 0, pairs = 0, pass = 0, raw = 0, filt = 0, found = 0, capped = 0;
    double tscan = 0, tfilt = 0, tsolve = 0;

    if (T.n) {
#pragma omp parallel
    {
        int8_t pafa[64]; float psda[64];
        uint32_t tb[2 * NW_CHUNK + 8]; int ta[64];
        uint32_t pend_t[8], pend_X[8], pend_d0[8], pend_G[8]; int pend_b[8]; int npend = 0;
        int *surv = malloc(sizeof(int) * (size_t)T.n);
        long long lA = 0, lAs = 0, lpairs = 0, lpass = 0, lraw = 0, lfilt = 0, lfound = 0, lcapped = 0;
        double lscan = 0, lfilt_t = 0, lsolve = 0;
        unsigned long long cfilt = 0, cpre = 0, cflush = 0, ctail = 0; (void)cfilt; (void)cpre; (void)cflush; (void)ctail;
#pragma omp for schedule(dynamic, 1)
        for (long long c0 = CLO; c0 < CHI; c0 += ACHUNK) {
          long long chi = c0 + ACHUNK; if (chi > CHI) chi = CHI;
          int solve_this = ((c0 / ACHUNK) % sample) == 0;
          double tc0 = tcpun();
          for (long long ci = c0; ci < chi; ci++) {
            uint32_t acode = (uint32_t)ci;
            if (NW_DECCANON && !a_is_canonical(acode)) continue;
            uint64_t am = skew_mask(acode);
            paf_mask(am, pafa);
            if (psd_all(pafa, psda) > fourn + 1e-6) continue;
            lA++;
            if (!solve_this) continue;
            lAs++;
            double ts0 = tcpun(); lscan += ts0 - tc0;
            uint32_t raraw; nw_arow(pafa, ta, &raraw);
            uint32_t rat = nw_tmap(raraw);
            uint32_t wa1 = 0, ta0 = 0;
            for (int k = 1; k <= H; k++) {
                if (ta[k] & 1) ta0 |= 1u << (k - 1);
                if (ta[k] & 2) wa1 |= 1u << (k - 1);
            }
            const uint32_t xa = wa1, da = raraw ^ M0W;
            int nsurv = 0;
            unsigned long long _c0 = GM_TICK();
            {
                const uint64_t *bp[64]; int nbp = 0, dead = 0;
                for (int k = 0; k < H && !dead; k++) {
                    double cap = (double)fourn - psda[k];
                    if (cap < 0) { dead = 1; break; }
                    int t = lev_of(cap, fourn);
                    if (t >= NLEV - 1) continue;
                    bp[nbp++] = T.bmp + (((size_t)k * NLEV) + t) * T.cwords;
                }
                int lo = USE_SWAP ? lower_bound_u32(T.key, T.n, acode) : 0;
                int w0 = lo >> 6;
                if (!dead && lo < T.n) {
                    uint64_t lomask = (lo & 63) ? (~0ULL << (lo & 63)) : ~0ULL;
                    int wlast = (T.n - 1) >> 6;
                    uint64_t himask = (T.n & 63) ? ((1ULL << (T.n & 63)) - 1) : ~0ULL;
                    int w = w0 & ~3;
                    if (!scalar_and) {
                        for (; w < T.cwords; w += 4) {
                            __m256i fv = _mm256_set1_epi64x(-1);
                            for (int i = 0; i < nbp; i++)
                                fv = _mm256_and_si256(fv, _mm256_loadu_si256((const __m256i *)(bp[i] + w)));
                            if (_mm256_testz_si256(fv, fv)) continue;
                            uint64_t v4[4];
                            _mm256_storeu_si256((__m256i *)v4, fv);
                            for (int q = 0; q < 4; q++) {
                                int ww = w + q;
                                if (ww < w0 || ww > wlast) continue;
                                uint64_t v = v4[q];
                                if (ww == w0) v &= lomask;
                                if (ww == wlast) v &= himask;
                                while (v) {
                                    int b = __builtin_ctzll(v);
                                    v &= v - 1;
                                    surv[nsurv++] = (ww << 6) + b;
                                }
                            }
                        }
                    } else {
                        for (w = w0; w <= wlast; w++) {
                            uint64_t v = ~0ULL;
                            for (int i = 0; i < nbp; i++) v &= bp[i][w];
                            if (w == w0) v &= lomask;
                            if (w == wlast) v &= himask;
                            while (v) {
                                int b = __builtin_ctzll(v);
                                v &= v - 1;
                                surv[nsurv++] = (w << 6) + b;
                            }
                        }
                    }
                }
            }
            lpairs += nsurv;
            unsigned long long _c1 = GM_TICK(); cfilt += _c1 - _c0;
            double tf1 = tcpun(); lfilt_t += tf1 - ts0;

#define GM_EMIT(NCD, CANDS, SBITS, BIDX)                                       \
            do {                                                               \
                int8_t pfb[64]; paf_mask(skew_mask(T.code[BIDX]), pfb);        \
                for (int cc = 0; cc < (NCD); cc++) {                           \
                    uint32_t vv = (CANDS)[cc];                                 \
                    uint32_t codeC = (~vv) & maskH;                            \
                    uint32_t codeD = ((SBITS) ^ vv) & maskH;                   \
                    uint64_t wc = hkt_symword(codeC), wd = hkt_symword(codeD); \
                    int good = 1;                                              \
                    for (int k = 1; k <= H && good; k++) {                     \
                        int t2 = (int)pafa[k - 1] + (int)pfb[k - 1] + 2 * N    \
                               - 2 * (__builtin_popcountll(wc ^ hkt_rotk(wc, k)) \
                                    + __builtin_popcountll(wd ^ hkt_rotk(wd, k))); \
                        if (t2 != 0) good = 0;                                 \
                    }                                                          \
                    lfilt++;                                                   \
                    if (!good) continue;                                       \
                    lfound++;                                                  \
                    _Pragma("omp critical")                                    \
                    {                                                          \
                        if (nsol == solcap) {                                  \
                            solcap *= 2;                                       \
                            sol = realloc(sol, (size_t)solcap * 4 * sizeof(uint64_t)); \
                        }                                                      \
                        uint64_t *r = sol + (size_t)nsol * 4;                  \
                        r[0] = am; r[1] = T.code[BIDX];                        \
                        r[2] = codeC; r[3] = codeD;                            \
                        nsol++;                                                \
                    }                                                          \
                }                                                              \
            } while (0)

#define GM_FLUSH()                                                             \
            do {                                                               \
                uint32_t ct[64 * 8], pb8[64 * 8], pm8[8], bbv[8];               \
                uint32_t q0[8] = {0}, q1[8] = {0};                              \
                unsigned long long _f0 = GM_TICK(); cpre += _f0 - _k0;          \
                for (int L = npend; L < 8; L++) { pend_t[L] = 0; pend_X[L] = 0; pend_d0[L] = 0; pend_G[L] = 0; } \
                for (int k = 0; k < H; k++)                                     \
                    for (int L = npend; L < 8; L++) ct[(size_t)k * 8 + L] = 0;  \
                nw_cols4(pend_t, ct, 0);                                        \
                if (npend > 4) nw_cols4(pend_t + 4, ct, 4);                     \
                if (GTAB) {                                                     \
                    for (int L = 0; L < 8; L++)                                 \
                        bbv[L] = L < npend ? (pend_G[L] ^ pend_X[L]) & maskH : 0u; \
                } else {                                                        \
                    nw_mcbits4(pend_t, q0, q1);                                 \
                    if (npend > 4) nw_mcbits4(pend_t + 4, q0 + 4, q1 + 4);      \
                    for (int L = 0; L < 8; L++)                                 \
                        bbv[L] = L < npend ? (q1[L] ^ (pend_d0[L] & ~q0[L]) ^ pend_X[L]) & maskH : 0u; \
                }                                                               \
                gm_elim8(ct, bbv, pb8, pm8);                                    \
                unsigned long long _f1 = GM_TICK(); cflush += _f1 - _f0;        \
                uint32_t part[8], fm[8], bas0[8]; int nf[8];                    \
                gm_tail8(ct, pb8, bbv, part, fm, bas0, nf);                     \
                for (int L = 0; L < npend; L++) {                               \
                    if (bbv[L] & ~pm8[L]) continue;                             \
                    if (nf[L] > S2_MAXFREE) { lcapped++; continue; }            \
                    uint32_t cand[GM_MAXC]; int nc;                             \
                    if (nf[L] <= 1) {                                           \
                        cand[0] = part[L]; nc = 1;                              \
                        if (nf[L]) { cand[1] = part[L] ^ bas0[L]; nc = 2; }     \
                    } else {                                                    \
                        uint32_t colL[64], pivL[64];                            \
                        for (int k = 0; k < H; k++) {                           \
                            colL[k] = ct[(size_t)k * 8 + L];                    \
                            pivL[k] = pb8[(size_t)k * 8 + L];                   \
                        }                                                       \
                        nc = gm_tail_m(colL, bbv[L], pm8[L], pivL, cand);       \
                        if (nc < 0) { lcapped++; continue; }                    \
                    }                                                           \
                    lraw += nc;                                                 \
                    int nk = gm_rowfilter(cand, nc, pend_t[L], maskH);          \
                    if (nk) GM_EMIT(nk, cand, pend_t[L], pend_b[L]);            \
                }                                                               \
                npend = 0;                                                      \
                _k0 = GM_TICK(); ctail += _k0 - _f1;                            \
            } while (0)

            uint32_t *tcur = tb, *tnext = tb + NW_CHUNK;
            unsigned long long _k0 = GM_TICK(); (void)_k0;
            int mch = nsurv < NW_CHUNK ? nsurv : NW_CHUNK;
            for (int u = 0; u < mch; u++) {
                uint32_t t = rat ^ T.rec[surv[u]].rbt;
                tcur[u] = t;
                if (CCK) __builtin_prefetch(CC + (size_t)t * CCS);
                if (NW_CHUNK + u < nsurv) __builtin_prefetch(&T.rec[surv[NW_CHUNK + u]]);
            }
            for (int s0i = 0; s0i < nsurv; s0i += NW_CHUNK) {
                int s1i = s0i + NW_CHUNK;
                int mnext = nsurv - s1i; if (mnext > NW_CHUNK) mnext = NW_CHUNK;
                int s2i = s1i + NW_CHUNK;
                int m2 = nsurv - s2i; if (m2 > NW_CHUNK) m2 = NW_CHUNK;
                for (int u = 0; u < mch; u++) {
                    if (u < m2) __builtin_prefetch(&T.rec[surv[s2i + u]]);
                    if (u < mnext) {
                        uint32_t t = rat ^ T.rec[surv[s1i + u]].rbt;
                        tnext[u] = t;
                        if (CCK) __builtin_prefetch(CC + (size_t)t * CCS);
                    }
                    uint32_t t = tcur[u];
                    int b0i = surv[s0i + u];
                    const BRec *r = &T.rec[b0i];
                    uint32_t d0 = da ^ r->rbraw;
                    uint32_t X = xa ^ r->wb1 ^ (ta0 & ~d0);
                    uint32_t gt = 0;
                    if (CCK) {
                        const uint32_t *e = CC + (size_t)t * CCS;
                        uint32_t e0 = e[0];
                        if (H <= 30 && e0 == CC_REJECT) { lcapped++; continue; }
                        int bad = 0;
                        for (int i = 0; i < CCK; i++) {
                            uint32_t ei = i ? e[i] : e0;
                            uint32_t w = ei & maskH;
                            if (!w) break;
                            if ((__builtin_parity(w & X) ^ (ei >> 31)) & 1u) { bad = 1; break; }
                        }
                        if (bad) continue;
                        if (GTAB) gt = e[CCK];
                    }
                    lpass++;
                    pend_t[npend] = t; pend_X[npend] = X; pend_d0[npend] = d0; pend_b[npend] = b0i;
                    pend_G[npend] = gt;
                    if (++npend == 8) GM_FLUSH();
                }
                { uint32_t *sw = tcur; tcur = tnext; tnext = sw; }
                mch = mnext;
            }
            if (npend) GM_FLUSH();
            cpre += GM_TICK() - _k0;
#undef GM_EMIT
#undef GM_FLUSH
            tc0 = tcpun(); lsolve += tc0 - tf1;
          }
          lscan += tcpun() - tc0;
        }
        free(surv);
#pragma omp atomic
        C_FILT += cfilt;
#pragma omp atomic
        C_PRE += cpre;
#pragma omp atomic
        C_FLUSH += cflush;
#pragma omp atomic
        C_TAIL += ctail;
#pragma omp atomic
        nA += lA;
#pragma omp atomic
        nAs += lAs;
#pragma omp atomic
        pairs += lpairs;
#pragma omp atomic
        pass += lpass;
#pragma omp atomic
        raw += lraw;
#pragma omp atomic
        filt += lfilt;
#pragma omp atomic
        found += lfound;
#pragma omp atomic
        capped += lcapped;
#pragma omp atomic
        tscan += lscan;
#pragma omp atomic
        tfilt += lfilt_t;
#pragma omp atomic
        tsolve += lsolve;
    }
    }
    double tsearch = wnow() - t0;
    table_free(&T);

    printf("   canonical A rows passing the ceiling: %lld\n", nA);
    printf("   pairs %lld, past the consistency test %lld (%.1f%%), lifts %lld, row-sum matches %lld, nullity-capped %lld\n",
           pairs, pass, pairs ? 100.0 * pass / (double)pairs : 0.0, raw, filt, capped);
    printf("   thread-seconds: A prep %.2f | survivors %.2f | solve %.2f; %.0f ns per pair\n",
           tscan, tfilt, tsolve, pairs ? 1e9 * (tsolve + tfilt) / (double)pairs : 0.0);
    if (capped)
        printf("   WARNING: %lld solve calls were dropped by the nullity cap; the census is not exhaustive\n", capped);
#ifdef GM_SPLIT
    {   double tot = (double)(C_FILT + C_PRE + C_FLUSH + C_TAIL);
        printf("   cycles: filter %.1f%% | test %.1f%% | columns+elimination %.1f%% | tail %.1f%%  (%.0f per pair, %.0f elim per passed)\n",
               100.0 * C_FILT / tot, 100.0 * C_PRE / tot, 100.0 * C_FLUSH / tot, 100.0 * C_TAIL / tot,
               pairs ? tot / (double)pairs : 0.0, pass ? (double)C_FLUSH / (double)pass : 0.0); }
#endif
    if (sample > 1)
        printf("   sampled %lld of %lld canonical A: %lld pairs, %.1f pairs per A; projected %.4g pairs, filter+solve %.0f thread-s\n",
               nAs, nA, pairs, nAs ? (double)pairs / (double)nAs : 0.0,
               nAs ? (double)pairs * (double)nA / (double)nAs : 0.0,
               nAs ? (tsolve + tfilt) * (double)nA / (double)nAs : 0.0);
    {
        FILE *st = fopen("/proc/self/status", "r");
        char line[256];
        if (st) {
            while (fgets(line, sizeof line, st))
                if (!strncmp(line, "VmHWM:", 6)) { printf("   peak RSS %s", line + 6); break; }
            fclose(st);
        }
    }
    printf("n=%d: %lld quadruples (%lld candidates from the solve), %.2fs\n", N, found, raw, tsearch);
    {
        double tc = wnow();
        size_t cap = 1; while (cap < (size_t)nsol * 4 + 16) cap <<= 1;
        hset_init(cap);
        FILE *rf = NULL;
        { const char *rp = getenv("NW_REPS"); if (rp) rf = fopen(rp, "w"); }
        uint64_t q[4], c[4];
        for (long long i = 0; i < nsol; i++) {
            uint64_t *r = sol + (size_t)i * 4;
            q[0] = r[0];
            q[1] = skew_mask((uint32_t)r[1]);
            q[2] = sym_mask((uint32_t)r[2]);
            q[3] = sym_mask((uint32_t)r[3]);
            nw_canon(q, c);
            if (hset_add(c) && rf) {
                for (int u = 0; u < 4; u++) {
                    if (u) fputc(' ', rf);
                    for (int j = 0; j < N; j++) fputc(((c[u] >> j) & 1) ? '+' : '-', rf);
                }
                fputc('\n', rf);
            }
        }
        if (rf) fclose(rf);
        printf("   inequivalent classes: %zu   (canonicalised in %.2fs)\n", HCNT, wnow() - tc);
    }
    if (outpath) {
        FILE *f = fopen(outpath, "w");
        if (!f) { perror(outpath); return 1; }
        for (long long i = 0; i < nsol; i++) {
            uint64_t *r = sol + (size_t)i * 4;
            uint64_t w[4];
            w[0] = r[0];
            w[1] = skew_mask((uint32_t)r[1]);
            w[2] = sym_mask((uint32_t)r[2]);
            w[3] = sym_mask((uint32_t)r[3]);
            for (int u = 0; u < 4; u++) {
                if (u) fputc(' ', f);
                for (int j = 0; j < N; j++) fputc(((w[u] >> j) & 1) ? '+' : '-', f);
            }
            fputc('\n', f);
        }
        fclose(f);
        printf("   solutions -> %s\n", outpath);
    }
    return 0;
}
