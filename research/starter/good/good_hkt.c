/* good_hkt.c -- exhaustive census of circulant good matrices of odd order n
 * by the mod-4 linearisation of Holzmann, Kharaghani and Tayfeh-Rezaie (2008).
 *
 * A good quadruple is one skew row A (a_0 = 1, a_j = -a_{n-j}) and three
 * symmetric rows B, C, D (x_j = x_{n-j}, normalised to x_0 = 1) of length n
 * with PAF_A + PAF_B + PAF_C + PAF_D = 0 at every shift 1..(n-1)/2.  The row
 * sums satisfy 1 + b^2 + c^2 + d^2 = 4n.
 *
 * Enumeration: A runs over the skew codes that are least in their orbit under
 * the phi(n) decimations (t and -t act differently on a skew row) and pass the
 * PSD ceiling.  B runs over a table of the symmetric codes of the popcount
 * fixed by the largest symmetric row sum b, filtered per A by
 * PSD_A + PSD_B <= 4n through an index of H bitmaps; every (c, d) with
 * c^2 + d^2 = 4n - 1 - b^2 is accepted in one pass, and codeC <= codeD keeps
 * one of each swapped pair when c = d.  The board's equivalence on a good
 * quadruple reduces to the phi(n) multipliers applied to all four rows, so a
 * class is an orbit of (A, {B, C, D}) under them and restricting A to
 * canonical codes loses no class.
 *
 * Per (A, B) pair, with the solver of hkt_solve.h (it needs only that PAF(k)
 * is congruent to n mod 4, which holds for a skew row as for a symmetric one;
 * the solve path follows research/starter/williamson/wil_hkt2.c):
 *
 *   consistency test   the stage-2 matrix depends only on the sign pattern t,
 *                      and so does its left null space; a pair is consistent
 *                      exactly when its right-hand side bb is orthogonal to
 *                      every left null vector.  A table of 2^H entries holds
 *                      those vectors (K words each) and the parity of the
 *                      t-only half of bb against them.  At prime orders the
 *                      corank is 1 and half the pairs fail; at composite
 *                      orders about 87% fail.
 *   folded right side  bb = G(t) XOR X with G a function of t alone and X
 *                      formed from two words of the A row, one of the B row
 *                      and the stage-1 right side (derivation above M0W).
 *   direct columns     the stage-2 matrix of a pair that passes is built in
 *                      closed form into the transposed eight-lane layout.
 *   elimination        eight pairs per AVX2 elimination; each candidate
 *                      (C, D) is checked exactly before it is written.
 *
 * The table is 4 bytes per pattern and null vector (plus 4 for G(t) when
 * stored): 32 MB at n = 47, 512 MB at n = 55, 4 GB at n = 61.  It is off above
 * H = 30, where every survivor of the pair filter goes to the elimination.
 *
 * Every closed form is checked at startup against the reference computation
 * for the order in hand.  The stage-2 nullity cap of the solver (nullity > 6)
 * is counted rather than assumed: the run prints how many solve calls it
 * dropped, and the census is exhaustive only when that count is zero.
 *
 *   build: gcc -O3 -march=native -fopenmp -o good_hkt good_hkt.c -lm
 *          (needs AVX2 and hkt_solve.h; orders 3 <= n <= 63)
 *   run:   ./good_hkt n threads out.txt
 *   env:   every GOOD_ variable tunes speed or memory only; none changes the
 *          set of quadruples written.
 *          GOOD_CCK=k     null-vector words per table entry, 0 = no table
 *                         (default chosen from a sample of patterns)
 *          GOOD_GTAB=0/1  store G(t) in the table (default on for H <= 26)
 *          GOOD_NLEV=L    PSD quantisation levels of the B index, 2..256
 *                         (default 256; fewer levels filter less and use
 *                         less memory)
 *
 * Output: one line per distinct quadruple found, four words of n '+'/'-'
 * characters: A (skew) first, then B, C, D (symmetric) in ascending code
 * order.  Quadruples are not expanded over the decimation orbit; compare
 * outputs as sets of classes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#if !defined(__AVX2__)
#error "good_hkt.c needs AVX2 (build with -march=native on an AVX2 host)"
#endif
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int N, H;
#define SCALE 16
#define CEIL16 (4 * N * SCALE)

#include "hkt_solve.h"

static double wnow(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double tcpun(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ------------------------------------------------- per-row prefix halves

   PAF(k) is congruent to n mod 4 for every lag of a (-1,1) row of odd length,
   so PAF_B(k) = n - 4 u_k with u_k >= 0.  With A_k = -PAF_A(k) - n - 2 FIXED_[k]
   (even, depends only on A) the solver's working value is v = A_k + 4 u_k, its
   stage-1 right-hand side bit is bit 1 of (v >> 1) - 2 M1CONST_[k], which is
   bit2(A_k) ^ bit0(M1CONST_[k]) ^ bit0(u_k), so rhs1 = RA ^ RB with one word
   per row, and the stage-1 solution is t = T(rhs1) = T(RA) ^ T(RB) since T is
   linear.  The stage-2 target is ta_k + u_k with ta_k = A_k >> 2. */
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
/* Reference right-hand side of stage 2: bit 1 of
   mc_k - ta_k - u_k at each lag. */
static inline uint32_t nw_bvec_ref(const int16_t *mc, const int *ta, const int8_t *su) {
    uint32_t bvec = 0;
    for (int kb = 1; kb <= H; kb++)
        if (((mc[kb - 1] - ta[kb] - su[kb - 1]) >> 1) & 1) bvec |= 1u << (kb - 1);
    return bvec;
}

/* ------------------------------------------------- eight-lane elimination

   hkt_elim8v's construction (columns transposed, ct[c*8 + lane], one-hot
   pivot masks, dead lanes carry zero) with one change: a column that has
   taken its pivot in every lane is a unit vector e_q with q in p, no later
   pivot bit can match it, and it never changes again, so the inner loop runs
   over the columns not yet reached plus the few columns some lane left
   without a pivot.  At the prime orders the corank is 1 and the free column
   is the last one in half the patterns and the second-to-last in a quarter,
   so that list is almost always empty and the update count is about H^2 / 2
   rather than H^2.  Output identical to hkt_elim8v. */
static inline void hk_elim8(uint32_t *c, uint32_t *bb, uint32_t *pivb, uint32_t *pm) {
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
            pend[npend++] = cc;              /* some lane has no pivot here */
    }
    _mm256_storeu_si256((__m256i *)bb, vb);
    _mm256_storeu_si256((__m256i *)pm, p);
}

/* Lane-vectorised tail for the reduced eight-lane system.  For every lane:
   part has bit c when the pivot row of column c carries the right-hand side,
   fm has bit c when column c has no pivot, and for the first free column f
   basis0 is the null vector e_f + sum of e_c over the pivot columns c whose
   pivot row appears in the reduced column f.  Lanes with two or more free
   columns are left to the scalar hkt_tail_m by the caller (nf[L] >= 2). */
static inline void hk_tail8(const uint32_t *ct, const uint32_t *pivb, const uint32_t *bbv,
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
    __m256i vw = _mm256_and_si256(vfm, _mm256_sub_epi32(zero, vfm));   /* e_f */
    for (int cc = 0; cc < H; cc++) {
        __m256i bit = _mm256_loadu_si256((const __m256i *)(pivb + (size_t)cc * 8));
        __m256i e   = _mm256_set1_epi32((int)(1u << cc));
        __m256i nz  = _mm256_cmpeq_epi32(_mm256_and_si256(vc, bit), zero);
        vw = _mm256_or_si256(vw, _mm256_andnot_si256(nz, e));
    }
    _mm256_storeu_si256((__m256i *)basis0, vw);
}

/* ------------------------------------------------- the folded right side

   Write d_k = ta_k + u_k.  Bit 1 of mc_k - d_k is
       bit1(mc_k) ^ bit1(d_k) ^ (bit0(d_k) & ~bit0(mc_k))
   (the last term is the borrow out of bit 0), and
       bit1(d_k) = bit1(ta_k) ^ bit1(u_k) ^ (bit0(ta_k) & bit0(u_k)).
   The stage-1 right side is rhs1_k = bit2(A_k) ^ m_k ^ bit0(u_k) with
   m_k = bit0(M1CONST_[k]) and bit2(A_k) = bit0(ta_k), so
       bit0(d_k) = bit0(ta_k) ^ bit0(u_k) = rhs1_k ^ m_k =: D0_k,
   a function of t alone (rhs1 = T^{-1} t), and
       bit0(ta_k) & bit0(u_k) = bit0(ta_k) & ~D0_k.
   Hence, as H-bit words,
       bb = [MC1(t) ^ (D0(t) & ~MC0(t))] ^ [WA1 ^ WB1 ^ (TA0 & ~D0)]
   with MC0, MC1 the bit-0 and bit-1 words of mc(t), WA1 = bit1(ta),
   TA0 = bit0(ta), WB1 = bit1(u).  The first bracket is G(t); the second is X,
   which a pair forms from two words of A, one word of B and rhs1 = RA ^ RB.
   For a left null vector w of the stage-2 matrix of t, parity(bb & w) =
   parity(G(t) & w) ^ parity(X & w), and the table stores the first term. */
static uint32_t M0W;              /* bit0(M1CONST_[k]) as a word over lags */
static uint32_t *M1FWD;           /* forward stage-1 matrix: rhs1_k = parity(M1FWD[k] & t) */

static inline uint32_t gf2_parity_mul(const uint32_t *rows, uint32_t v) {
    uint32_t r = 0;
    for (int k = 0; k < H; k++) r |= (uint32_t)__builtin_parity(rows[k] & v) << k;
    return r;
}
/* Invert the map t = T(rhs1) by elimination on its byte-table columns. */
static int fwd_init(void) {
    uint32_t inv[64], aug[64];
    for (int k = 0; k < H; k++) { inv[k] = 0; }
    /* rows of T: t_c = parity(M1INVCOL_[c] & rhs1), so row c of T is M1INVCOL_[c] */
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
    /* check: T(M1FWD t) = t and M1FWD T(r) = r on random words */
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0x1234567ULL ^ (uint64_t)N;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t r = (uint32_t)(seed & hm);
        if (gf2_parity_mul(M1FWD, nw_tmap(r)) != r) return 0;
    }
    return 1;
}

/* ------------------------------------------------- stage-2 reference build */
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
    for (int k = H; k < 64; k++) mc[k] = 0;
    hkt_tr(a, H, col);
}
/* Left null space of the stage-2 matrix of pattern t, as masks over the lags. */
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

/* ------------------------------------------------- direct columns and mc
   (as in research/starter/kkss/gmat_c_v2.c).  Writing j = c + 1, the row build gives
       M[k][c] = ([u_j = u_{j+k}] & ~CA_k[j]) ^ ([u_j = u_{j-k}] & ~CA_k[j-k])
   with CA_k = CONSTALL_[k]; over k at fixed j the first bracket reads bits of
   uw >> j and the second bits of rot(uw, j), so
       col_j = ((P & NT1) ^ (Q & NT2) ^ (u_j ? 0 : NT1 ^ NT2)) >> 1.
   And kept_k always contains the one fixed point of j -> -j-k, so
       mc_k = H - popcount(uw ^ rot(uw, k)) / 2. */
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
/* MC0 and MC1 words (bits 0 and 1 of mc_k over the lags) for one pattern. */
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
/* mc for four patterns at once, nibble-table popcount. */
static inline void nw_mc4(const uint32_t *t4, int16_t (*mc4)[64]) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i nib = _mm256_set1_epi8(0x0f), zero = _mm256_setzero_si256();
    const __m256i maskn = _mm256_set1_epi64x((long long)((N >= 64) ? ~0ULL : ((1ULL << N) - 1)));
    const __m256i hh = _mm256_set1_epi64x(H);
    __m256i uw = _mm256_setr_epi64x((long long)nw_uw(t4[0]), (long long)nw_uw(t4[1]),
                                    (long long)nw_uw(t4[2]), (long long)nw_uw(t4[3]));
    for (int k = 1; k <= H; k++) {
        __m128i ck = _mm_cvtsi32_si128(k), cn = _mm_cvtsi32_si128(N - k);
        __m256i r = _mm256_and_si256(_mm256_or_si256(_mm256_srl_epi64(uw, ck),
                                                     _mm256_sll_epi64(uw, cn)), maskn);
        __m256i x = _mm256_xor_si256(uw, r);
        __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(x, nib));
        __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), nib));
        __m256i pc = _mm256_sad_epu8(_mm256_add_epi8(lo, hi), zero);
        __m256i m  = _mm256_sub_epi64(hh, _mm256_srli_epi64(pc, 1));
        mc4[0][k - 1] = (int16_t)_mm256_extract_epi64(m, 0);
        mc4[1][k - 1] = (int16_t)_mm256_extract_epi64(m, 1);
        mc4[2][k - 1] = (int16_t)_mm256_extract_epi64(m, 2);
        mc4[3][k - 1] = (int16_t)_mm256_extract_epi64(m, 3);
    }
    for (int k = H; k < 64; k++) mc4[0][k] = mc4[1][k] = mc4[2][k] = mc4[3][k] = 0;
}
/* MC0 and MC1 words for four patterns at once: the bits 0 and 1 of mc_k are
   masked out of the 64-bit lane and shifted into place, so no value leaves
   the vector unit until the end. */
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
/* Four patterns at once into the transposed layout ct[c*8 + lane]. */
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
        uint32_t ct[64 * 8]; int16_t mc4[4][64]; uint32_t q0[4], q1[4];
        nw_cols4(t4, ct, 0);
        nw_mc4(t4, mc4);
        nw_mcbits4(t4, q0, q1);
        for (int u = 0; u < 4; u++) {
            int16_t m1[64], m2[64]; uint32_t c1[64], c2[64], b0, b1;
            nw_rows_ref(t4[u], m1, c1);
            nw_mc_from_t(t4[u], m2);
            nw_cols_scalar(t4[u], c2);
            nw_mcbits(t4[u], &b0, &b1);
            for (int k = 0; k < H; k++) {
                int bad = (m1[k] != m2[k]) || (c1[k] != c2[k]);
                bad |= (ct[(size_t)k * 8 + u] != c1[k]) || (mc4[u][k] != m1[k]);
                bad |= (((b0 >> k) & 1) != (m1[k] & 1)) || (((b1 >> k) & 1) != ((m1[k] >> 1) & 1));
                bad |= (q0[u] != b0) || (q1[u] != b1);
                if (bad) {
                    fprintf(stderr, "n=%d: direct column build disagrees with the "
                            "row build at t=%u lag %d\n", N, t4[u], k + 1);
                    exit(1);
                }
            }
        }
    }
}

/* ------------------------------------------------- consistency table (CC)

   K uint32 words per pattern t:
     bits 0..H-1   a left null vector w of the stage-2 matrix; zero ends the
                   list
     bit 31        parity(G(t) & w), the t-only half of the test
     bit 30        on the last stored word: the null space has more vectors
                   than fit, so a pair that passes the stored ones still runs
                   the elimination (which rechecks consistency)
     0xffffffff    in word 0: nullity above S2_MAXFREE; the pattern cannot be
                   completed under the cap and the pair is counted as capped
   Needs H <= 30. */
#define S2_MAXFREE 6
#define CC_REJECT 0xffffffffu
#define CC_MORE   0x40000000u
#define CC_G      0x80000000u
static uint32_t *CC = NULL;
static int CCK = 0;              /* null-vector words per pattern; 0 = no table */
static int GTAB = 0;             /* 1: the entry also stores G(t) after the K words */
static int CCS = 0;              /* entry stride in words, CCK + GTAB */
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
            e[0] = CC_REJECT; for (int i = 1; i < CCS; i++) e[i] = 0; rej++; continue;
        }
        uint32_t G = nw_G(t);
        if (GTAB) e[CCK] = G;
        for (int i = 0; i < CCK; i++) {
            if (i < c) {
                uint32_t w = nv[i] & hm;
                e[i] = w | (__builtin_parity(w & G) ? CC_G : 0u);
            } else e[i] = 0;
        }
        if (c > CCK) {
            uint32_t w = e[CCK - 1] | CC_MORE;
            /* a stored vector that would read as CC_REJECT is dropped; the
               flag alone still sends the pair to the full solve */
            e[CCK - 1] = (w == CC_REJECT) ? CC_MORE : w;
        }
    }
    CC_REJECTED = rej;
}

/* -------------------------------------------------------------- row filter

   ALLOWD[pc] is the set of popcounts pd such that (pc, pd) are the popcounts
   of some (C, D) completing the current (a, b).  When C and D have the same
   popcount the candidate set contains both (C, D) and (D, C), so codeC <=
   codeD keeps one of them. */
static uint32_t ALLOWD[64];
static inline int hk_keep(uint32_t v, uint32_t s0, uint32_t maskH) {
    int a = H - __builtin_popcount(v & maskH);        /* popcount of codeC */
    uint32_t am = ALLOWD[a];
    if (!am) return 0;
    uint32_t cD = (s0 ^ v) & maskH;
    int b = __builtin_popcount(cD);
    if (!((am >> b) & 1u)) return 0;
    return a != b || ((~v) & maskH) <= cD;
}
static inline int hk_rowfilter(uint32_t *cand, int nc, uint32_t s, uint32_t maskH) {
    int keep = 0;
    for (int i = 0; i < nc; i++)
        if (hk_keep(cand[i], s, maskH)) cand[keep++] = cand[i];
    return keep;
}

/* ------------------------------------------------------------------ rows */
static void sym_row(uint32_t code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v; x[N - j] = v;
    }
}
static inline uint64_t sym_mask(uint32_t code) {
    uint64_t m = 1ULL;
    for (int j = 1; j <= H; j++)
        if ((code >> (j - 1)) & 1) { m |= 1ULL << j; m |= 1ULL << (N - j); }
    return m;
}
/* skew row from a code: x_0 = 1, bit j-1 set means x_j = +1 and x_{n-j} = -1 */
static void skew_row(uint32_t code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v; x[N - j] = (int8_t)-v;
    }
}
static inline uint64_t skew_mask(uint32_t code) {
    uint64_t m = 1ULL;
    for (int j = 1; j <= H; j++) {
        if ((code >> (j - 1)) & 1) m |= 1ULL << j;
        else                       m |= 1ULL << (N - j);
    }
    return m;
}
static inline void paf_mask(uint64_t m, int8_t *paf) {
    uint64_t full = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    for (int s = 1; s <= H; s++) {
        uint64_t r = ((m >> s) | (m << (N - s))) & full;
        paf[s - 1] = (int8_t)(N - 2 * (int)__builtin_popcountll(m ^ r));
    }
}
static double *COSTAB;                 /* [H+1][H] */
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

/* Startup check of the folded right-hand side against nw_bvec_ref on random
   (skew A, symmetric B) row pairs. */
static void fold_selfcheck(void) {
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0xD1B54A32D192ED03ULL ^ (uint64_t)N;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t ca = (uint32_t)(seed & hm);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        uint32_t cb = (uint32_t)(seed & hm);
        int8_t pa[64], pb[64], su[64]; int ta[64];
        paf_mask(skew_mask(ca), pa); paf_mask(sym_mask(cb), pb);
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
            fprintf(stderr, "n=%d: folded right-hand side disagrees with nw_bvec_ref "
                    "(A=%u B=%u ref=%08x got=%08x)\n", N, ca, cb, ref, got);
            exit(1);
        }
    }
}

/* -------------------------------------------------- decimation on codes

   x'_{tj} = x_j for a unit t of Z_n.  The source position j lands on
   q = tj mod n; when q <= H the bit moves to q - 1, and when q > H the stored
   position is n - q, where a symmetric row repeats the value and a skew row
   negates it.  So the image of a code is a bit permutation (the same for both
   kinds) followed, for the skew kind, by an XOR with the mask of the wrapped
   positions.  All phi(n) units are kept, index 0 being t = 1, because t and
   -t differ on a skew row. */
static int NDEC;
static uint32_t *DBYTE;          /* [NDEC][DNB][256] */
static uint32_t *DFLIP;          /* [NDEC] */
static int DNB;
static int gcd_int(int a, int b) { while (b) { int q = a % b; a = b; b = q; } return a; }
static void dec_init(void) {
    int cnt = 0;
    for (int t = 1; t < N; t++) if (gcd_int(t, N) == 1) cnt++;
    NDEC = cnt;
    int *map = malloc(sizeof(int) * (size_t)cnt * H);
    DFLIP = calloc((size_t)cnt, sizeof(uint32_t));
    if (!map || !DFLIP) { fprintf(stderr, "out of memory for the decimation maps\n"); exit(1); }
    int i = 0;
    for (int t = 1; t < N; t++) {
        if (gcd_int(t, N) != 1) continue;
        for (int j = 1; j <= H; j++) {
            int q = (int)(((long long)t * j) % N);
            if (q <= H) map[(size_t)i * H + j - 1] = q - 1;
            else { map[(size_t)i * H + j - 1] = N - q - 1; DFLIP[i] |= 1u << (N - q - 1); }
        }
        i++;
    }
    DNB = (H + 7) / 8;
    DBYTE = calloc((size_t)NDEC * DNB * 256, sizeof(uint32_t));
    if (!DBYTE) { fprintf(stderr, "out of memory for the decimation tables\n"); exit(1); }
    for (int d = 0; d < NDEC; d++)
        for (int b = 0; b < DNB; b++)
            for (int v = 0; v < 256; v++) {
                uint32_t acc = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int pos = b * 8 + bit;
                    if (pos < H && ((v >> bit) & 1))
                        acc |= 1u << map[(size_t)d * H + pos];
                }
                DBYTE[(((size_t)d * DNB) + b) * 256 + v] = acc;
            }
    free(map);
}
static inline uint32_t dec_scatter(uint32_t code, int d) {
    const uint32_t *t = DBYTE + (size_t)d * DNB * 256;
    uint32_t r = 0;
    for (int b = 0; b < DNB; b++) r |= t[(size_t)b * 256 + ((code >> (8 * b)) & 0xffu)];
    return r;
}
static inline uint32_t dec_skew(uint32_t code, int d) { return dec_scatter(code, d) ^ DFLIP[d]; }
static inline uint32_t dec_sym(uint32_t code, int d)  { return dec_scatter(code, d); }
/* Canonical A: the least code in its decimation orbit. */
static inline int a_is_canonical(uint32_t code) {
    for (int d = 1; d < NDEC; d++)
        if (dec_skew(code, d) < code) return 0;
    return 1;
}
/* the decimation tables against the definition, on a few codes */
static void dec_selfcheck(void) {
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    unsigned seed = 777u;
    int8_t x[128], y[128];
    for (int trial = 0; trial < 256; trial++) {
        seed = seed * 1103515245u + 12345u; uint32_t c = (seed >> 8) & hm;
        int ti = 0;
        for (int t = 1; t < N; t++) {
            if (gcd_int(t, N) != 1) continue;
            for (int kind = 0; kind < 2; kind++) {
                if (kind) skew_row(c, x); else sym_row(c, x);
                for (int j = 0; j < N; j++) y[(int)(((long long)t * j) % N)] = x[j];
                uint32_t img = kind ? dec_skew(c, ti) : dec_sym(c, ti);
                for (int j = 1; j <= H; j++) {
                    int bit = (img >> (j - 1)) & 1;
                    if ((y[j] > 0) != bit || y[0] != 1 ||
                        y[N - j] != (kind ? -y[j] : y[j])) {
                        fprintf(stderr, "n=%d: decimation table disagrees with the "
                                "definition at t=%d\n", N, t);
                        exit(1);
                    }
                }
            }
            ti++;
        }
    }
}

/* --------------------------------------------------------- combinations */
static long long BINOM[64][64];
static void binom_init(void) {
    for (int i = 0; i < 64; i++) {
        BINOM[i][0] = 1;
        for (int j = 1; j < 64; j++)
            BINOM[i][j] = (j > i) ? 0 : BINOM[i-1][j-1] + BINOM[i-1][j];
    }
}
static uint32_t unrank_colex(long long rank, int k) {
    uint32_t m = 0;
    for (int i = k; i >= 1; i--) {
        int c = i - 1;
        while (BINOM[c + 1][i] <= rank) c++;
        m |= 1u << c;
        rank -= BINOM[c][i];
    }
    return m;
}
static inline uint32_t gosper_next(uint32_t v) {
    uint32_t t = v | (v - 1);
    return (t + 1) | (((~t & (t + 1)) - 1) >> (__builtin_ctz(v) + 1));
}

/* ------------------------------------------------------------ row sums */
static int signed_sum(int s) {
    return (((s - N) % 4) + 4) % 4 == 0 ? s : -s;
}
static int popcount_for(int s) {
    int num = s - 1 + 2 * H;
    if (num < 0 || (num & 3)) return -1;
    int p = num / 4;
    return (p >= 0 && p <= H) ? p : -1;
}
/* A symmetric row with x_0 = +1 has row sum 1 + 4p - 2H for p of its H free
   positions +1; the skew row sums to 1, so b^2 + c^2 + d^2 = 4n - 1. */
typedef struct { int s[3]; int p[3]; } Decomp;      /* s[0] >= s[1] >= s[2] > 0 */
static int decomps(Decomp *out, int maxout) {
    int nd = 0, target = 4 * N - 1;
    int lim = (int)(sqrt((double)target) + 1e-9);
    for (int b = 1; b <= lim; b += 2)
        for (int c = 1; c <= b; c += 2)
            for (int d = 1; d <= c; d += 2) {
                if (b*b + c*c + d*d != target) continue;
                if (nd >= maxout) continue;
                Decomp *D = &out[nd];
                D->s[0] = b; D->s[1] = c; D->s[2] = d;
                int ok = 1;
                for (int u = 0; u < 3; u++) {
                    D->p[u] = popcount_for(signed_sum(D->s[u]));
                    if (D->p[u] < 0) ok = 0;
                }
                if (ok) nd++;
            }
    return nd;
}

/* --------------------------------------------------------- the B table */
static int NLEV = 256;
#define ACHUNK 4096             /* A codes per scheduling chunk */
#define NW_CHUNK 32             /* survivors per prefetch round */
static int lev_of(double x, int fourn) {
    int v = (int)(x * NLEV / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > NLEV - 1) v = NLEV - 1;
    return v;
}
typedef struct { uint32_t rbt, rbraw, wb1, pad; } BRec;
typedef struct {
    int n;
    uint32_t *code;        /* ascending */
    int8_t   *paf;         /* [n][H], read on candidates only */
    BRec     *rec;         /* [n] */
    int cwords;
    uint64_t *bmp;         /* [H][NLEV][cwords], prefix-ored over levels */
    long long ncomb;
} Table;

static void table_build(Table *T, int pb, int fourn) {
    long long ncomb = BINOM[H][pb];
    T->ncomb = ncomb;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    uint32_t **lc = calloc(nth, sizeof(uint32_t *));
    long long *ln = calloc(nth, sizeof(long long));
#pragma omp parallel num_threads(nth)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        long long per = (ncomb + nth - 1) / nth;
        long long lo = (long long)t * per, hi = lo + per;
        if (hi > ncomb) hi = ncomb;
        long long cap = 1024, cnt = 0;
        uint32_t *buf = malloc(sizeof(uint32_t) * (size_t)cap);
        int8_t paf[64]; float psd[64];
        if (lo < hi) {
            uint32_t c = unrank_colex(lo, pb);
            for (long long ci = lo; ci < hi; ci++) {
                paf_mask(sym_mask(c), paf);
                if (!(psd_all(paf, psd) > fourn + 1e-6)) {
                    if (cnt == cap) { cap *= 2; buf = realloc(buf, sizeof(uint32_t) * (size_t)cap); }
                    buf[cnt++] = c;
                }
                if (ci + 1 < hi) c = gosper_next(c);
            }
        }
        lc[t] = buf; ln[t] = cnt;
    }
    long long total = 0;
    for (int t = 0; t < nth; t++) total += ln[t];
    if (total > 0x7fffffffLL) { fprintf(stderr, "B table too large for int indices\n"); exit(1); }
    T->n = (int)total;
    T->code = malloc(sizeof(uint32_t) * (size_t)(total + 1));
    {   long long off = 0;
        for (int t = 0; t < nth; t++) {
            memcpy(T->code + off, lc[t], sizeof(uint32_t) * (size_t)ln[t]);
            off += ln[t]; free(lc[t]);
        }
    }
    free(lc); free(ln);
    T->paf = malloc((size_t)(total + 1) * H);
    T->rec = aligned_alloc(64, ((size_t)(total + 4) * sizeof(BRec) + 63) & ~(size_t)63);
    T->cwords = (int)((total + 63) >> 6);
    /* the AND chain reads four words at a time; pad to a multiple of four */
    size_t cw4 = ((size_t)T->cwords + 3) & ~(size_t)3;
    size_t bmpn = (size_t)H * NLEV * cw4;
    T->bmp = calloc(bmpn + 4, sizeof(uint64_t));
    if (!T->paf || !T->rec || !T->bmp) {
        fprintf(stderr, "out of memory for the B table (%lld rows, index %.1f GB)\n",
                total, bmpn * 8.0 / (1024.0 * 1024.0 * 1024.0));
        exit(1);
    }
    T->cwords = (int)cw4;
#pragma omp parallel for schedule(dynamic, 4)
    for (int w = 0; w < T->cwords; w++) {
        int8_t paf[64]; float psd[64];
        int hi = (w + 1) * 64; if (hi > T->n) hi = T->n;
        for (int bi = w * 64; bi < hi; bi++) {
            paf_mask(sym_mask(T->code[bi]), paf);
            psd_all(paf, psd);
            memcpy(T->paf + (size_t)bi * H, paf, H);
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
    free(T->code); free(T->paf); free(T->rec); free(T->bmp);
    memset(T, 0, sizeof *T);
}

/* ------------------------------------------------------------- solutions */
typedef struct { uint32_t c[4]; } Quad;
static Quad *SOL; static long long NSOL, SOLCAP;
static void sol_push(const uint32_t *c) {
    if (NSOL == SOLCAP) {
        SOLCAP = SOLCAP ? 2 * SOLCAP : 1024;
        SOL = realloc(SOL, sizeof(Quad) * (size_t)SOLCAP);
    }
    memcpy(SOL[NSOL++].c, c, sizeof(uint32_t) * 4);
}
/* sort the three symmetric codes; the skew code stays in front */
static void sort3(uint32_t *r) {
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2 - a; b++)
            if (r[b] > r[b + 1]) { uint32_t t = r[b]; r[b] = r[b + 1]; r[b + 1] = t; }
}
static int cmp_quad(const void *x, const void *y) {
    return memcmp(((const Quad *)x)->c, ((const Quad *)y)->c, sizeof(uint32_t) * 4);
}
static void canon_class(const uint32_t *q, uint32_t *out) {
    uint32_t best[4]; int have = 0;
    for (int d = 0; d < NDEC; d++) {
        uint32_t r[4];
        r[0] = dec_skew(q[0], d);
        for (int u = 1; u < 4; u++) r[u] = dec_sym(q[u], d);
        sort3(r + 1);
        if (!have || memcmp(r, best, sizeof best) < 0) { memcpy(best, r, sizeof best); have = 1; }
    }
    memcpy(out, best, sizeof best);
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: good_hkt n threads out.txt\n"); return 1; }
    N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    const char *outpath = argv[3];
    if (N < 3 || (N % 2) == 0) { fprintf(stderr, "n must be odd and >= 3\n"); return 1; }
    if (threads < 1) threads = 1;
    H = (N - 1) / 2;
    if (N > 63) { fprintf(stderr, "n > 63 needs wider words\n"); return 1; }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    binom_init();
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
    int fourn = 4 * N;
    printf("n=%d: 4n - 1 = %d, H = %d, %d decimations\n", N, fourn - 1, H, NDEC);

    Decomp dec[256];
    int nd = decomps(dec, 256);
    const int i0 = 0, o0 = 1, o1 = 2;      /* enumerate the largest sum as B */
    typedef struct { int sb, pb; int nmem; int mem[16]; } Group;
    Group grp[256]; int ng = 0;
    for (int i = 0; i < nd; i++) {
        int g;
        for (g = 0; g < ng; g++) if (grp[g].sb == dec[i].s[i0]) break;
        if (g == ng) {
            grp[g].sb = dec[i].s[i0]; grp[g].pb = dec[i].p[i0];
            grp[g].nmem = 0; ng++;
        }
        if (grp[g].nmem < 16) grp[g].mem[grp[g].nmem++] = i;
    }
    printf("   %d decompositions b >= c >= d of 4n - 1, %d groups\n", nd, ng);
    for (int g = 0; g < ng; g++) {
        printf("   group %d: b = %d (signed %d) with", g, grp[g].sb, signed_sum(grp[g].sb));
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            printf(" (%d, %d)", D->s[o0], D->s[o1]);
        }
        printf("   popcount %d of %d\n", grp[g].pb, H);
    }
    { const char *v = getenv("GOOD_NLEV"); if (v) { NLEV = atoi(v); if (NLEV < 2) NLEV = 2; if (NLEV > 256) NLEV = 256; } }
    if (NLEV != 256) printf("   B index quantised to %d levels\n", NLEV);

    double t0 = wnow();
    {
        int kk = -1;
        const char *v = getenv("GOOD_CCK"); if (v) kk = atoi(v);
        if (H > 30) { if (kk > 0) fprintf(stderr, "H > 30: the consistency table is off\n"); kk = 0; }
        if (kk < 0) kk = cc_choose_k();
        if (kk > S2_MAXFREE) kk = S2_MAXFREE;
        CCK = kk;
        /* G(t) in the table saves the mc bits of every pair that passes the
           test, at four more bytes per pattern.  Default on up to H = 26
           (512 MB); GOOD_GTAB=0/1 overrides. */
        GTAB = (CCK > 0 && H <= 26) ? 1 : 0;
        const char *gv = getenv("GOOD_GTAB"); if (gv && CCK > 0) GTAB = atoi(gv) ? 1 : 0;
    }
    if (CCK) {
        double tb = wnow();
        cc_build();
        printf("   consistency table: K=%d null-vector words%s per pattern, %.1f MB, %lld patterns nullity-capped, built in %.2fs\n",
               CCK, GTAB ? " + G(t)" : "", (double)CCS * 4.0 * (double)((size_t)1 << H) / 1048576.0, CC_REJECTED, wnow() - tb);
    } else {
        printf("   no consistency table: every survivor goes to the elimination\n");
    }
    fflush(stdout);

    const uint32_t maskH = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    long long tot_pairs = 0, tot_pass = 0, tot_raw = 0, tot_filt = 0, tot_found = 0, tot_capped = 0, tot_A = 0;
    double max_index_mb = 0;

    const long long ncombA = 1LL << H;          /* every skew code */
    for (int g = 0; g < ng; g++) {
        double tg = wnow();
        int pb = grp[g].pb;
        memset(ALLOWD, 0, sizeof ALLOWD);
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            ALLOWD[D->p[o0]] |= 1u << D->p[o1];
            ALLOWD[D->p[o1]] |= 1u << D->p[o0];
        }
        Table T;
        double ttb = wnow();
        table_build(&T, pb, fourn);
        double idx_mb = (double)H * NLEV * T.cwords * 8.0 / 1048576.0;
        if (idx_mb > max_index_mb) max_index_mb = idx_mb;
        printf("   group %d (b = %d): B table %d of %lld codes (%.1f%%), index %.0f MB, built in %.1fs; A over %lld skew codes\n",
               g, grp[g].sb, T.n, T.ncomb, 100.0 * T.n / (double)T.ncomb, idx_mb,
               wnow() - ttb, ncombA);
        fflush(stdout);
        if (!T.n) { table_free(&T); continue; }

        long long nA = 0, pairs = 0, pass = 0, raw = 0, filt = 0, found = 0, capped = 0;
        double tsolve = 0, tscan = 0, tfilt = 0;
#pragma omp parallel
        {
            int8_t pafa[64]; float psda[64];
            uint32_t tb[2 * NW_CHUNK + 8]; int ta[64];
            uint32_t pend_t[8], pend_X[8], pend_d0[8], pend_G[8]; int pend_b[8]; int npend = 0;
            int *surv = malloc(sizeof(int) * (size_t)T.n);
            long long lA = 0, lpairs = 0, lpass = 0, lraw = 0, lfilt = 0, lfound = 0, lcapped = 0;
            double lsolve = 0, lscan = 0, lfilt_t = 0;
#pragma omp for schedule(dynamic, 1)
            for (long long c0 = 0; c0 < ncombA; c0 += ACHUNK) {
              long long chi = c0 + ACHUNK; if (chi > ncombA) chi = ncombA;
              double tc0 = tcpun();
              for (long long ci = c0; ci < chi; ci++) {
                uint32_t am = (uint32_t)ci;
                if (!a_is_canonical(am)) continue;
                paf_mask(skew_mask(am), pafa);
                if (psd_all(pafa, psda) > fourn + 1e-6) continue;
                lA++;
                double ts0 = tcpun(); lscan += ts0 - tc0;
                uint32_t raraw; nw_arow(pafa, ta, &raraw);
                uint32_t rat = nw_tmap(raraw);
                uint32_t wa1 = 0, ta0 = 0;
                for (int k = 1; k <= H; k++) {
                    if (ta[k] & 1) ta0 |= 1u << (k - 1);
                    if (ta[k] & 2) wa1 |= 1u << (k - 1);
                }
                const uint32_t xa = wa1, da = raraw ^ M0W;   /* X = xa ^ wb1 ^ (ta0 & ~(da ^ rbraw)) */
                /* pair filter: rows whose level at every frequency is at most
                   the level of 4n - PSD_A */
                int nsurv = 0;
                {
                    const uint64_t *bp[64]; int nbp = 0, dead = 0;
                    for (int k = 0; k < H && !dead; k++) {
                        double cap = (double)fourn - psda[k];
                        if (cap < 0) { dead = 1; break; }
                        int t = lev_of(cap, fourn);
                        if (t >= NLEV - 1) continue;
                        bp[nbp++] = T.bmp + (((size_t)k * NLEV) + t) * T.cwords;
                    }
                    if (!dead) {
                        int wlast = (T.n - 1) >> 6;
                        uint64_t himask = (T.n & 63) ? ((1ULL << (T.n & 63)) - 1) : ~0ULL;
                        for (int w = 0; w < T.cwords; w += 4) {
                            __m256i fv = _mm256_set1_epi64x(-1);
                            for (int i = 0; i < nbp; i++)
                                fv = _mm256_and_si256(fv, _mm256_loadu_si256((const __m256i *)(bp[i] + w)));
                            if (_mm256_testz_si256(fv, fv)) continue;
                            uint64_t v4[4];
                            _mm256_storeu_si256((__m256i *)v4, fv);
                            for (int q = 0; q < 4; q++) {
                                int ww = w + q;
                                if (ww > wlast) continue;
                                uint64_t v = v4[q];
                                if (ww == wlast) v &= himask;
                                while (v) {
                                    int b = __builtin_ctzll(v);
                                    v &= v - 1;
                                    surv[nsurv++] = (ww << 6) + b;
                                }
                            }
                        }
                    }
                }
                lpairs += nsurv;
                double tf1 = tcpun(); lfilt_t += tf1 - ts0;

#define HK_EMIT(NCD, CANDS, SBITS, BIDX)                                      \
                do {                                                           \
                    const int8_t *pf = T.paf + (size_t)(BIDX) * H;             \
                    for (int cc = 0; cc < (NCD); cc++) {                       \
                        uint32_t vv = (CANDS)[cc];                             \
                        uint32_t codeC = (~vv) & maskH;                        \
                        uint32_t codeD = ((SBITS) ^ vv) & maskH;               \
                        uint64_t wc = hkt_symword(codeC), wd = hkt_symword(codeD); \
                        int good = 1;                                          \
                        for (int k = 1; k <= H && good; k++) {                 \
                            int t2 = (int)pafa[k - 1] + (int)pf[k - 1] + 2 * N \
                                   - 2 * (__builtin_popcountll(wc ^ hkt_rotk(wc, k)) \
                                        + __builtin_popcountll(wd ^ hkt_rotk(wd, k))); \
                            if (t2 != 0) good = 0;                             \
                        }                                                      \
                        lfilt++;                                               \
                        if (!good) continue;                                   \
                        lfound++;                                              \
                        uint32_t q[4] = { am, T.code[BIDX], codeC, codeD };    \
                        _Pragma("omp critical")                                \
                        sol_push(q);                                           \
                    }                                                          \
                } while (0)

/* Eight queued pairs through one elimination: columns straight into the
   transposed layout, bb = G(t) ^ X per lane, dead lanes zero. */
#define HK_FLUSH()                                                            \
                do {                                                           \
                    uint32_t ct[64 * 8], pb8[64 * 8], pm8[8], bbv[8];           \
                    uint32_t q0[8] = {0}, q1[8] = {0};                          \
                    for (int L = npend; L < 8; L++) { pend_t[L] = 0; pend_X[L] = 0; pend_d0[L] = 0; pend_G[L] = 0; } \
                    for (int k = 0; k < H; k++)                                 \
                        for (int L = npend; L < 8; L++) ct[(size_t)k * 8 + L] = 0; \
                    nw_cols4(pend_t, ct, 0);                                    \
                    if (npend > 4) nw_cols4(pend_t + 4, ct, 4);                 \
                    if (GTAB) {                                                 \
                        for (int L = 0; L < 8; L++)                             \
                            bbv[L] = L < npend ? (pend_G[L] ^ pend_X[L]) & maskH : 0u; \
                    } else {                                                    \
                        nw_mcbits4(pend_t, q0, q1);                             \
                        if (npend > 4) nw_mcbits4(pend_t + 4, q0 + 4, q1 + 4);  \
                        for (int L = 0; L < 8; L++)                             \
                            bbv[L] = L < npend ? (q1[L] ^ (pend_d0[L] & ~q0[L]) ^ pend_X[L]) & maskH : 0u; \
                    }                                                           \
                    hk_elim8(ct, bbv, pb8, pm8);                               \
                    uint32_t part[8], fm[8], bas0[8]; int nf[8];                \
                    hk_tail8(ct, pb8, bbv, part, fm, bas0, nf);                \
                    for (int L = 0; L < npend; L++) {                           \
                        if (bbv[L] & ~pm8[L]) continue;                         \
                        if (nf[L] > S2_MAXFREE) { lcapped++; continue; }        \
                        uint32_t cand[HKT_MAXC]; int nc;                        \
                        if (nf[L] <= 1) {                                       \
                            cand[0] = part[L]; nc = 1;                          \
                            if (nf[L]) { cand[1] = part[L] ^ bas0[L]; nc = 2; } \
                        } else {                                                \
                            uint32_t colL[64], pivL[64];                        \
                            for (int k = 0; k < H; k++) {                       \
                                colL[k] = ct[(size_t)k * 8 + L];                \
                                pivL[k] = pb8[(size_t)k * 8 + L];               \
                            }                                                   \
                            nc = hkt_tail_m(colL, bbv[L], pm8[L], pivL, cand);  \
                        }                                                       \
                        lraw += nc;                                             \
                        int nk = hk_rowfilter(cand, nc, pend_t[L], maskH);     \
                        HK_EMIT(nk, cand, pend_t[L], pend_b[L]);               \
                    }                                                           \
                    npend = 0;                                                  \
                } while (0)

                /* Two passes per chunk of survivors: the first forms each
                   pattern and prefetches its table word, with the B records
                   of the next chunk prefetched alongside; the second does the
                   test.  The next chunk's patterns are formed while this one
                   is tested, so every table line has a chunk of work to
                   arrive in. */
                uint32_t *tcur = tb, *tnext = tb + NW_CHUNK;
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
                            if (e0 == CC_REJECT) { lcapped++; continue; }
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
                        if (++npend == 8) HK_FLUSH();
                    }
                    { uint32_t *sw = tcur; tcur = tnext; tnext = sw; }
                    mch = mnext;
                }
                if (npend) HK_FLUSH();
#undef HK_EMIT
#undef HK_FLUSH
                tc0 = tcpun(); lsolve += tc0 - tf1;
              }
              lscan += tcpun() - tc0;
            }
            free(surv);
#pragma omp atomic
            tsolve += lsolve;
#pragma omp atomic
            tfilt += lfilt_t;
#pragma omp atomic
            tscan += lscan;
#pragma omp atomic
            nA += lA;
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
        }
        table_free(&T);
        printf("   group %d (b = %d): %lld canonical A, %lld pairs, %lld (%.1f%%) past the consistency test, %lld lifts, %lld row-sum matches, %lld quadruples, %lld capped, %.1fs\n",
               g, grp[g].sb, nA, pairs, pass, pairs ? 100.0 * pass / (double)pairs : 0.0,
               raw, filt, found, capped, wnow() - tg);
        printf("   group %d: thread-seconds scan %.1f, filter %.1f, solve %.1f; %.0f ns per pair\n",
               g, tscan, tfilt, tsolve, pairs ? 1e9 * (tsolve + tfilt) / (double)pairs : 0.0);
        fflush(stdout);
        tot_A += nA; tot_pairs += pairs; tot_pass += pass; tot_raw += raw; tot_filt += filt;
        tot_found += found; tot_capped += capped;
    }

    double tsearch = wnow() - t0;
    /* distinct quadruples: the same (A, {B, C, D}) can arrive under more than
       one labelling when symmetric sums coincide */
    Quad *keyed = malloc(sizeof(Quad) * (size_t)(NSOL + 1));
    long long ndistinct = 0;
    for (long long i = 0; i < NSOL; i++) sort3(SOL[i].c + 1);
    qsort(SOL, (size_t)NSOL, sizeof(Quad), cmp_quad);
    for (long long i = 0; i < NSOL; i++) {
        if (i && !memcmp(SOL[i].c, SOL[i - 1].c, sizeof(uint32_t) * 4)) continue;
        keyed[ndistinct++] = SOL[i];
    }
    long long nclasses = 0;
    {
        Quad *cl = malloc(sizeof(Quad) * (size_t)(ndistinct + 1));
        for (long long i = 0; i < ndistinct; i++) canon_class(keyed[i].c, cl[i].c);
        qsort(cl, (size_t)ndistinct, sizeof(Quad), cmp_quad);
        for (long long i = 0; i < ndistinct; i++)
            if (!i || memcmp(cl[i].c, cl[i - 1].c, sizeof(uint32_t) * 4)) nclasses++;
        free(cl);
    }
    printf("   canonical A rows passing the ceiling: %lld\n", tot_A);
    printf("   pairs %lld, past the consistency test %lld, lifts %lld, row-sum matches %lld, nullity-capped calls %lld\n",
           tot_pairs, tot_pass, tot_raw, tot_filt, tot_capped);
    if (tot_capped)
        printf("   WARNING: %lld solve calls were dropped by the nullity cap; the census is not exhaustive\n",
               tot_capped);
    printf("   largest B index %.0f MB\n", max_index_mb);
    {
        FILE *st = fopen("/proc/self/status", "r");
        char line[256];
        if (st) {
            while (fgets(line, sizeof line, st))
                if (!strncmp(line, "VmHWM:", 6)) { printf("   peak RSS %s", line + 6); break; }
            fclose(st);
        }
    }
    printf("n=%d: %lld quadruples found (%lld distinct), %lld inequivalent classes, %.2fs\n",
           N, tot_found, ndistinct, nclasses, tsearch);
    FILE *f = fopen(outpath, "w");
    if (!f) { perror(outpath); return 1; }
    int8_t x[128];
    for (long long i = 0; i < ndistinct; i++) {
        for (int u = 0; u < 4; u++) {
            if (u) fputc(' ', f);
            if (u == 0) skew_row(keyed[i].c[u], x); else sym_row(keyed[i].c[u], x);
            for (int j = 0; j < N; j++) fputc(x[j] > 0 ? '+' : '-', f);
        }
        fputc('\n', f);
    }
    fclose(f);
    printf("   solutions -> %s\n", outpath);
    return 0;
}
