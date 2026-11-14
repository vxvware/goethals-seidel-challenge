/* G-matrices: Goethals-Seidel difference families of symmetry type (kkss).
 *
 * Four circulant (-1,1) matrices of odd order v with A A^T + B B^T + C C^T +
 * D D^T = 4vI, where A and B are skew (x_0 = 1, x_j = -x_{v-j}) and C and D are
 * symmetric.  Djokovic and Kotsireas, arXiv:1802.00556, classify these for odd
 * v < 50 and name them after J. Seberry Wallis; v = 51 and beyond is open.
 *
 * This is nw_c.c with the two enumerated blocks changed from (free, symmetric)
 * to (skew, skew).  The solver is untouched, because it only ever sees a target
 * vector and the type of the two blocks it solves for, and those are still
 * symmetric.  What changes:
 *
 *   row sums   a skew row sums to 1, so the decomposition is c^2 + d^2 = 4v - 2
 *              rather than a^2 + b^2 + c^2 + d^2 = 4v with one free part
 *   A loop     skew codes, quotiented by decimation only: rotating a skew row
 *              destroys skewness, so the rotation quotient of finding 409/410
 *              does not apply here
 *   B table    skew codes rather than symmetric ones of a fixed row sum, so
 *              there is no popcount filter and the table is larger
 *
 * Revision of 2026-11-14 (agent_gmat).  One solve path replaces the cached
 * and NW_NOCACHE pair; results identical to the stored censuses at
 * n = 37, 41, 43, 45.  What changed, in order of effect:
 *
 *   consistency table   2^H entries of K uint32 words holding the left null
 *                       vectors of the stage-2 matrix (cc_build); a survivor
 *                       whose right-hand side fails a parity against them is
 *                       dropped before any elimination.  4K bytes per entry
 *                       against 64 + 4H for the old stage-2 cache: 24 MB at
 *                       n=43, 2.1 GB at n=59 and 4.3 GB at n=61 (corank 1
 *                       there, K=1).  NW_CCK overrides K; NW_CCK=0 runs
 *                       without the table.
 *   direct columns      the stage-2 matrix is built by columns in closed
 *                       form (nw_cols4, four patterns per AVX2 register)
 *                       straight into the transposed layout the eight-lane
 *                       elimination reads; the row build and hkt_tr are gone
 *                       from the hot path.  mc has a closed form as well
 *                       (nw_mc4).  Both are checked against the row build at
 *                       startup.
 *   memory layout       the t-map word and the u values of a B row share one
 *                       record (BREC); table words and B records are
 *                       prefetched one and two survivor chunks ahead; the
 *                       table is madvised to huge pages.
 *   AVX2 AND chain      the pair filter intersects four bitmap words per
 *                       register (NW_SCALARAND=1 restores the scalar chain).
 *
 * Measured on bos-game, four niced threads under a load of about 20, against
 * the previous source built the two old ways (see the report for the runs):
 *   n=43   NW_NOCACHE 42-47 s   cached 19-22 s   this 20-22 s
 *   n=45   NW_NOCACHE 122-130 s  cached 46-54 s   this 48-50 s
 * The NW_NOCACHE build was the only one that could reach n = 59 and 61.
 * -DGM_SPLIT prints an rdtsc split of the solve.  Build as before:
 *   gcc -O3 -march=native -fopenmp -Inearwill -o gmat nearwill/gmat_c.c -lm
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

static int N, H;
static int NW_DECCANON = 1;   /* quotient A by decimation too (finding 404) */
#define SCALE 16
#define CEIL16 (4 * N * SCALE)

#include "hkt_solve.h"


/* ------------------------------------------------- consistency table (CC)

   The stage-2 matrix is a function of the sign pattern t alone, and so is its
   left null space.  A system is consistent exactly when the right-hand side
   is orthogonal to every left null vector, and that test rejects 87% of the
   survivors at n=43 and 92% at n=45 (finding 476), before any elimination.
   The old cached build stored the whole stage-2 entry (mc, the masks, the
   pivot map and the null basis) at 64 + 4H bytes per pattern, which is why
   it stopped fitting at n=57.  This table stores only the left null vectors,
   K words per pattern, so it is 4K bytes per entry: 25 MB at n=43 and 2.1 GB
   at n=59, where the corank is 1 for every pattern sampled and K=1.  Every
   other quantity the solve needs is rebuilt from t when, and only when, the
   system has passed the test.

   Encoding, K uint32 words per pattern:
     word i < K   a left null vector over the lags (bits 0..H-1), nonzero; a
                  zero word ends the list
     bit 31 set   on the last stored word: more vectors exist than fit, so a
                  pattern that passes the stored ones still has to be solved
                  in full (the elimination rechecks consistency anyway)
     0xffffffff   in word 0: the nullity exceeds S2_MAXFREE, so the pattern can
                  never be completed and is rejected outright
   H <= 31 is guaranteed by the N <= 63 guard in main, so bit 31 is free. */
#define S2_MAXFREE 6            /* the nullity cap, as hkt_tail_m applies it */
#define CC_REJECT 0xffffffffu
#define CC_MORE   0x80000000u
static uint32_t *CC = NULL;
static int CCK = 0;              /* words per pattern; 0 = no table */

/* Unfolded symmetric word from the sign pattern (finding 315): position j in
   1..H reads bit j-1 of t and position N-j reads the same bit. */
static inline uint64_t nw_uw(uint32_t t) {
    uint32_t rv = t;
    rv = ((rv >> 1) & 0x55555555u) | ((rv & 0x55555555u) << 1);
    rv = ((rv >> 2) & 0x33333333u) | ((rv & 0x33333333u) << 2);
    rv = ((rv >> 4) & 0x0f0f0f0fu) | ((rv & 0x0f0f0f0fu) << 4);
    rv = __builtin_bswap32(rv) >> (32 - H);
    return 1ULL | ((uint64_t)t << 1) | ((uint64_t)rv << (H + 1));
}

/* Reference row build, used only to fill the table and to self-check the
   direct column build at startup. */
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

/* Left null space of the stage-2 matrix of pattern t, as masks over the lags.
   Returns the corank, writing up to maxout masks. */
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

/* Pick K from a sample of patterns: the smallest word count that holds the
   whole left null space of at least 99% of them.  Any K gives the same
   census; this only decides how many patterns take the full solve. */
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
    size_t bytes = nt * (size_t)CCK * sizeof(uint32_t);
    bytes = (bytes + (1u << 21) - 1) & ~(size_t)((1u << 21) - 1);
    CC = aligned_alloc(1u << 21, bytes);
    if (CC) madvise(CC, bytes, MADV_HUGEPAGE);  /* random reads: keep the TLB out of it */
    if (!CC) {
        fprintf(stderr, "out of memory for the consistency table (%.2f GB)\n",
                nt * (double)CCK * 4 / 1073741824.0);
        exit(1);
    }
#pragma omp parallel for schedule(dynamic, 4096)
    for (size_t t = 0; t < nt; t++) {
        uint32_t nv[S2_MAXFREE + 1];
        uint32_t *e = CC + t * (size_t)CCK;
        int c = nw_leftnull((uint32_t)t, nv, S2_MAXFREE + 1);
        if (c > S2_MAXFREE) { e[0] = CC_REJECT; for (int i = 1; i < CCK; i++) e[i] = 0; continue; }
        for (int i = 0; i < CCK; i++) e[i] = i < c ? nv[i] : 0u;
        if (c > CCK) {
            /* more vectors than words: flag the last word; the one case where
               flag and mask would read as CC_REJECT drops that mask instead */
            uint32_t w = nv[CCK - 1] | CC_MORE;
            e[CCK - 1] = (w == CC_REJECT) ? CC_MORE : w;
        }
    }
}

/* ------------------------------------------------- direct column build

   The elimination wants the matrix by columns, one word per unknown c with
   bit k-1 set when lag k involves c, and good_c and the old gmat_c built it
   by rows and transposed (finding 473 says why vectorising that was a loss:
   the transpose is the cost).  The columns have a closed form of their own.
   Writing j = c + 1, the row build gives

       M[k][c] = ([u_j = u_{j+k}] & ~CA_k[j]) ^ ([u_j = u_{j-k}] & ~CA_k[j-k])

   with CA_k = CONSTALL_[k].  Over k at fixed j, u_{j+k} is bit k of uw >> j
   (no wrap: j + k <= 2H < N), and u_{j-k} = u_{k-j} by symmetry is bit k of
   uw rotated left by j.  The two CA terms are constants per j, NT1 and NT2
   below with their bits 1..H kept, and [a = b] & m is (a ^ b ^ 1) & m, so

       col_j = ((P & NT1) ^ (Q & NT2) ^ (u_j ? 0 : NT1 ^ NT2)) >> 1

   with P = uw >> j and Q = rot(uw, j).  Thirteen word operations per column
   and no transpose.  Checked against nw_rows_ref at startup for the order in
   hand, and identical on 1.6 million systems over every odd order 7..63.

   mc has a closed form too.  kept_k always contains the one fixed point of
   j -> -j-k (its two positions carry equal bits by symmetry), so
   popcount(kept & ~FIXM) = N - popcount(uw ^ rot(uw, k)) - 1, and
   mc_k = H - popcount(uw ^ rot(uw, k)) / 2. */
static uint64_t NT1_[64], NT2_[64], CJ_[64];
#if defined(__AVX2__)
static __m256i *NT1V_, *NT2V_, *CJV_;        /* the same, pre-broadcast */
#endif
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
#if defined(__AVX2__)
    NT1V_ = aligned_alloc(32, 64 * sizeof(__m256i));
    NT2V_ = aligned_alloc(32, 64 * sizeof(__m256i));
    CJV_  = aligned_alloc(32, 64 * sizeof(__m256i));
    for (int j = 0; j < 64; j++) {
        NT1V_[j] = _mm256_set1_epi64x((long long)NT1_[j]);
        NT2V_[j] = _mm256_set1_epi64x((long long)NT2_[j]);
        CJV_[j]  = _mm256_set1_epi64x((long long)CJ_[j]);
    }
#endif
}

static inline void nw_mc_from_t(uint32_t t, int16_t *mc) {
    const uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    uint64_t uw = nw_uw(t);
    int k = 1;
    for (; k + 1 <= H; k += 2) {
        uint64_t r0 = ((uw >> k) | (uw << (N - k))) & maskn;
        uint64_t r1 = ((uw >> (k + 1)) | (uw << (N - k - 1))) & maskn;
        mc[k - 1] = (int16_t)(H - __builtin_popcountll(uw ^ r0) / 2);
        mc[k]     = (int16_t)(H - __builtin_popcountll(uw ^ r1) / 2);
    }
    for (; k <= H; k++) {
        uint64_t r = ((uw >> k) | (uw << (N - k))) & maskn;
        mc[k - 1] = (int16_t)(H - __builtin_popcountll(uw ^ r) / 2);
    }
    for (k = H; k < 64; k++) mc[k] = 0;           /* nw_bvec reads past H */
}

#if defined(__AVX2__)
/* mc for four patterns at once.  The popcount is the nibble-table form, two
   shuffles and a byte sum per lag; the four results are written as int16 so
   nw_bvec reads them unchanged.  NW_SCALARMC=1 keeps the scalar form. */
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
#endif

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

#if defined(__AVX2__)
/* Four patterns at once into the transposed layout ct[c*8 + lane], lanes
   L0..L0+3.  The shift counts are uniform across lanes, so they come from an
   XMM register; the low dword of each 64-bit lane is compacted with one
   permute and stored as 128 bits. */
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
#endif

/* Startup self-check of the two closed forms against the reference build. */
static void cols_selfcheck(void) {
    uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint64_t seed = 0x2545F4914F6CDD1DULL ^ (uint64_t)N;
    for (int i = 0; i < 2048; i++) {
        uint32_t t4[4];
        for (int u = 0; u < 4; u++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            t4[u] = (uint32_t)(seed & hm);
        }
        uint32_t ct[64 * 8]; int16_t mc4[4][64];
#if defined(__AVX2__)
        nw_cols4(t4, ct, 0);
        nw_mc4(t4, mc4);
#endif
        for (int u = 0; u < 4; u++) {
            int16_t m1[64], m2[64]; uint32_t c1[64], c2[64];
            nw_rows_ref(t4[u], m1, c1);
            nw_mc_from_t(t4[u], m2);
            nw_cols_scalar(t4[u], c2);
            for (int k = 0; k < H; k++) {
                int bad = (m1[k] != m2[k]) || (c1[k] != c2[k]);
#if defined(__AVX2__)
                bad |= (ct[(size_t)k * 8 + u] != c1[k]) || (mc4[u][k] != m1[k]);
#endif
                if (bad) {
                    fprintf(stderr, "n=%d: direct column build disagrees with the "
                            "row build at t=%u lag %d\n", N, t4[u], k + 1);
                    exit(1);
                }
            }
        }
    }
}

/* ------------------------------------------------- per-pair prefix halves

   The prefix splits into a part that depends only on A and a part that
   depends only on B, so the per-pair work is one XOR (finding 393).

   PAF(k) is congruent to n mod 4 for every lag of every (-1,1) circulant row
   (finding 392), so PAF_B(k) = n - 4 u_k with u_k a non-negative integer.
   Writing A_k = -PAF_A(k) - n - 2 FIXED_[k], which depends only on A, the
   prefix's working value is

       v = -(PAF_A(k) + PAF_B(k)) - 2 FIXED_[k] = A_k + 4 u_k

   and A_k is even.  The bit that goes into rhs1 is bit 1 of
   (v >> 1) - 2 M1CONST_[k] = X_k + 2 u_k with X_k = (A_k >> 1) - 2 M1CONST_[k];
   adding 2u cannot carry into bit 1 from below, so that bit is just
   bit1(X_k) XOR bit0(u_k), one term per side.  Hence rhs1 = RA XOR RB[b]
   with RA built once per A row and RB once per B row.  And
   v >> 2 = (A_k >> 2) + u_k, so the target the bvec loop wants is TA[k] + u_k
   and never has to be materialised. */
/* rhs1 -> t is a GF(2) matrix-vector product, sliced into byte tables, so it
   is linear and T(0) = 0; t = T(RA) ^ T(RB) (finding 397).  main() checks the
   linearity at startup for the order in hand. */
static inline uint32_t nw_tmap(uint32_t r) {
    uint32_t t = 0;
    for (int b = 0; b < TNB_; b++)
        t ^= TBYTE_[(size_t)b * 256 + ((r >> (8 * b)) & 0xffu)];
    return t;
}

static inline void nw_arow(const int8_t *pafA, int *ta, uint32_t *ra) {
    uint32_t r = 0;
    for (int k = 1; k <= H; k++) {
        int ak = -(int)pafA[k - 1] - N - 2 * FIXED_[k];
        ta[k] = ak >> 2;
        if ((((ak >> 1) - 2 * M1CONST_[k]) >> 1) & 1) r |= 1u << (k - 1);
    }
    *ra = nw_tmap(r);
}

/* Right-hand side of the stage-2 system: bit 1 of mc_k - TA[k] - u_k per
   lag.  mc widens from int16 and su from int8; a shift left by 30 puts the
   bit where movemask_ps collects eight lags at once.  The loop runs
   ceil(H/8) times; the lags past H read zeros in mc, inside a 64-slot array
   for ta, and inside the padded u field of the B record; the bits they
   produce are masked off (finding 395). */
static inline uint32_t nw_bvec(const int16_t *mc, const int *ta, const int8_t *su) {
    uint32_t bvec = 0;
    int kb = 1;
#if defined(__AVX2__)
    for (; kb <= H; kb += 8) {
        __m256i vm = _mm256_cvtepi16_epi32(
                         _mm_loadu_si128((const __m128i *)(mc + kb - 1)));
        __m256i vt = _mm256_loadu_si256((const __m256i *)(ta + kb));
        __m256i vu = _mm256_cvtepi8_epi32(
                         _mm_loadl_epi64((const __m128i *)(su + kb - 1)));
        __m256i vd = _mm256_sub_epi32(_mm256_sub_epi32(vm, vt), vu);
        uint32_t mm = (uint32_t)_mm256_movemask_ps(
                          _mm256_castsi256_ps(_mm256_slli_epi32(vd, 30)));
        bvec |= mm << (kb - 1);
    }
    bvec &= (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
#else
    for (; kb <= H; kb++)
        if (((mc[kb - 1] - ta[kb] - su[kb - 1]) >> 1) & 1) bvec |= 1u << (kb - 1);
#endif
    return bvec;
}

#if !defined(__AVX2__)
/* Scalar stand-in for hkt_elim8v on the same transposed layout: one
   hkt_elim1m per lane.  Only the build without AVX2 uses it. */
static inline void nw_elim8_scalar(uint32_t *c, uint32_t *bb, uint32_t *pivb,
                                   uint32_t *pm) {
    for (int L = 0; L < 8; L++) {
        uint32_t col[64], pv[64];
        for (int k = 0; k < H; k++) col[k] = c[(size_t)k * 8 + L];
        hkt_elim1m(col, &bb[L], pv, &pm[L]);
        for (int k = 0; k < H; k++) {
            c[(size_t)k * 8 + L] = col[k];
            pivb[(size_t)k * 8 + L] = pv[k];
        }
    }
}
#define NW_ELIM8 nw_elim8_scalar
#else
#define NW_ELIM8 hkt_elim8v
#endif

/* Row-sum filter on the solve's candidates.  When the decomposition gives C
   and D the same row sum, swapping them is a symmetry of the whole problem
   and every solution is found twice; keeping codeC <= codeD keeps exactly
   one of each (finding 406). */
static inline int nw_rowfilter(uint32_t *cand, int nc, uint32_t s,
                               int pc, int pdd, uint32_t maskH) {
    const int swapsym = (pc == pdd);
    int keep = 0;
    for (int i = 0; i < nc; i++) {
        uint32_t v = cand[i];
        int a = H - __builtin_popcount(v & maskH);
        if (a != pc && a != pdd) continue;
        uint32_t cD = (s ^ v) & maskH;
        int b = __builtin_popcount(cD);
        if ((a == pc && b == pdd) || (a == pdd && b == pc)) {
            if (!swapsym || ((~v) & maskH) <= cD) cand[keep++] = v;
        }
    }
    return keep;
}


static double GM_SURV, GM_SOLVE, GM_CHUNK;  /* finding-468 split, permanent */
static long long GM_NSURV, GM_NPASS;        /* pair-filter survivors; of them, past the table */
#ifdef GM_SPLIT
/* -DGM_SPLIT: rdtsc split of the solve into the table precheck (mc, bvec,
   parities), the batched build-and-eliminate, and the tail (de-transpose,
   null-space tail, row filter, exact check).  Instrument only. */
#include <x86intrin.h>
static unsigned long long GM_CPRE, GM_CELIM, GM_CTAIL;
#define GM_TICK() __rdtsc()
#else
#define GM_TICK() 0ULL
#endif
static int SCALAR_AND = 0, SCALAR_MC = 0;
static double tcpun(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double wnow(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ------------------------------------------------------------------ rows */

/* symmetric row from a code: x_0 = 1, bit j-1 set means x_j = x_{n-j} = +1 */
static void sym_row(uint32_t code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v; x[N - j] = v;
    }
}

/* circulant row from a bitmask over positions 0..n-1; bit set means +1 */
static void circ_row(uint64_t m, int8_t *x) {
    for (int j = 0; j < N; j++) x[j] = (m >> j) & 1 ? 1 : -1;
}

/* PAF straight from the mask.  For x_j = +1 exactly where bit j is set,
   PAF(s) counts agreements minus disagreements, and the disagreements are the
   set bits of m ^ rot(m, s), so PAF(s) = N - 2 popcount(m ^ rot(m, s)).  The
   loop it replaces was O(N H) with a modulo in its inner statement: 351
   divisions per A row at n=27 against 13 popcounts here (finding 398). */
/* mask of a skew row from its H free bits: x_0 = 1 and x_{v-j} = -x_j */
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

static void paf_row(const int8_t *x, int8_t *paf) {
    for (int s = 1; s <= H; s++) {
        int acc = 0;
        for (int j = 0; j < N; j++) acc += x[j] * x[(j + s) % N];
        paf[s - 1] = (int8_t)acc;
    }
}

/* PSD at frequency k from the autocorrelation, as census.c and good_c do */
static double *COSTAB;                 /* [H+1][H] */
static void cos_init(void) {
    COSTAB = malloc(sizeof(double) * (size_t)(H + 1) * H);
    for (int k = 0; k <= H; k++)
        for (int s = 1; s <= H; s++)
            COSTAB[(size_t)k * H + s - 1] = 2.0 * cos(2.0 * M_PI * s * k / N);
}
static double psd_at(const int8_t *paf, int k) {
    double acc = N;
    const double *row = COSTAB + (size_t)k * H;
    for (int s = 0; s < H; s++) acc += paf[s] * row[s];
    return acc;
}
/* max over the nonzero frequencies; frequency 0 is the row sum squared */
/* The ceiling test wants the largest value over the frequencies and the pair
   filter wants all of them, and both callers were computing the whole spectrum
   twice over: once inside psd_max, once in the loop that filled the array
   (finding 394).  One pass does both. */
static double psd_max_only(const int8_t *paf) {
    double m = -1e18;
    for (int k = 1; k <= H; k++) { double v = psd_at(paf, k); if (v > m) m = v; }
    return m;
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

/* ------------------------------------------------------- A canonical form */

/* A is defined up to rotation and negation (a circulant signed permutation
   matrix, in the paper's terms), so only the least such form is enumerated.
   Rotation and negation act on A alone, which is why they cannot be folded
   into the decimation the way they are for the symmetric rows. */
static void decim_init(void);
static uint64_t nw_decimate(uint64_t m, const int *map);
extern int *DECIM_FWD; extern int NDECIM_FWD;

/* Byte tables for the decimations, so the canonicity test does not rebuild
   each decimated mask one bit at a time.  nw_decimate gathers, bit j of the
   result coming from bit map[j] of the source; inverting the map turns it into
   a scatter, and a scatter is what tabulates: for each byte of the source, the
   bits it contributes to the result.  ceil(N/8) lookups replace N iterations
   (finding 408).  The table is phi(N) * ceil(N/8) * 256 words, 246 KB at both
   n=31 and n=39. */
static uint64_t *DBYTE;                 /* [ndec][nbytes][256] */
static int DNB;
static void dbyte_init(void) {
    DNB = (N + 7) / 8;
    DBYTE = calloc((size_t)NDECIM_FWD * DNB * 256, sizeof(uint64_t));
    if (!DBYTE) { fprintf(stderr, "out of memory for the decimation tables\n"); exit(1); }
    for (int i = 0; i < NDECIM_FWD; i++) {
        const int *map = DECIM_FWD + (size_t)i * N;
        int inv[64];
        for (int j = 0; j < N; j++) inv[map[j]] = j;
        for (int b = 0; b < DNB; b++)
            for (int v = 0; v < 256; v++) {
                uint64_t acc = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int pos = b * 8 + bit;
                    if (pos < N && ((v >> bit) & 1)) acc |= 1ULL << inv[pos];
                }
                DBYTE[(((size_t)i * DNB) + b) * 256 + v] = acc;
            }
    }
}
static inline uint64_t decim_fast(uint64_t m, int i) {
    const uint64_t *t = DBYTE + (size_t)i * DNB * 256;
    uint64_t r = 0;
    for (int b = 0; b < DNB; b++) r |= t[(size_t)b * 256 + ((m >> (8 * b)) & 0xffu)];
    return r;
}

/* A rows up to rotation and decimation.

   Rotating A alone leaves every autocorrelation unchanged, so the solution set
   is closed under it and enumerating one rotation per orbit loses nothing.
   Decimation is the other half and acts on all four rows at once: with
   A'_j = A_{tj} for t coprime to n, PAF_{A'}(k) = PAF_A(tk), so a decimated
   solution is a solution; B, C and D stay symmetric because t(n-j) = -tj, and
   x_0 is fixed because t0 = 0.  Every equivalence class therefore contains a
   member whose A row is minimal under both, and enumerating only those finds
   every class.

   What it does not preserve is the raw quadruple count, which drops by about
   phi(n): the census reports classes, and those are what the published table
   holds (finding 404). */
static int a_is_canonical_dec(uint64_t m) {
    /* Decimation only.  A skew row rotated is no longer skew, so the rotation
       quotient that findings 404, 409 and 410 apply to a free A row is not
       available here; decimation does preserve skewness, since t(v-j) = -tj. */
    for (int i = 1; i < NDECIM_FWD; i++)
        if (decim_fast(m, i) < m) return 0;
    return 1;
}

/* Enumerate the masks of one popcount directly rather than filtering every
   integer.  The A loop tested popcount on all 2^N masks to keep the C(N, pa)
   that have the right one: 2.1e9 tested for 3.0e8 kept at n=31, and the ratio
   grows with the order.  Gosper's hack steps from one to the next in a few
   instructions, and its sequence is the masks in increasing numeric order,
   which is colex order on the bit positions, so a chunk can be entered at any
   rank by unranking once (finding 407). */
static long long BINOM[64][64];
static void binom_init(void) {
    for (int i = 0; i < 64; i++) {
        BINOM[i][0] = 1;
        for (int j = 1; j < 64; j++)
            BINOM[i][j] = (j > i) ? 0 : BINOM[i-1][j-1] + BINOM[i-1][j];
    }
}
static uint64_t unrank_colex(long long rank, int k) {
    uint64_t m = 0;
    for (int i = k; i >= 1; i--) {
        int c = i - 1;
        while (BINOM[c + 1][i] <= rank) c++;
        m |= 1ULL << c;
        rank -= BINOM[c][i];
    }
    return m;
}
static inline uint64_t gosper_next(uint64_t v) {
    uint64_t t = v | (v - 1);
    return (t + 1) | (((~t & (uint64_t)(-(int64_t)~t)) - 1) >> (__builtin_ctzll(v) + 1));
}

static int a_is_canonical(uint64_t m) {
    /* Rotation only.  Negation is the other half of the circulant signed
       permutation freedom, and it is taken care of by fixing a >= 0 in the
       decomposition: negating A flips the sign of its row sum, so enumerating
       only the positive sum already picks one representative per negation
       class. */
    uint64_t full = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    for (int s = 1; s < N; s++) {
        uint64_t r = ((m >> s) | (m << (N - s))) & full;
        if (r < m) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------ row sums */


/* ------------------------------------------------------ row sums

   The four row sums satisfy a^2 + b^2 + c^2 + d^2 = 4n, and for a symmetric row
   normalised to x_0 = 1 the sum is 1 + 4p - 2H for p of the H free positions
   set, so b, c and d are congruent to n modulo 4 automatically.  Fixing the
   decomposition is what makes the search cheap: it pins A to one row sum and B
   to one, and the tables for a single sum are smaller than the whole space by
   roughly a factor of H each.  Enumerating without it costs a factor of H^2 in
   pairs, which is the difference between minutes and days by n=27.  */
typedef struct { int a, b, c, d; } Decomp;

static int sym_popcount_for(int s) {      /* p with 1 + 4p - 2H == s, or -1 */
    int num = s - 1 + 2 * H;
    if (num < 0 || (num & 3)) return -1;
    int p = num / 4;
    return (p >= 0 && p <= H) ? p : -1;
}

static int decomps(Decomp *out, int maxout) {
    /* Both skew blocks have row sum 1, so what is left is c^2 + d^2 = 4v - 2
       with c and d the row sums of the two symmetric blocks, each congruent to
       v mod 4.  C and D are interchangeable, so take c >= d. */
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

static int rowsum_of_mask(uint64_t m) {
    int p = __builtin_popcountll(m);
    return 2 * p - N;
}


/* ---------------------------------------------------------------- chain

   The pair filter compares two spectra frequency by frequency for every (A, B)
   pair, and at n=27 that is 3.0e9 pairs of 13 float compares.  The B table is
   only a few hundred rows, so a bitmap over it is a handful of words: index the
   table by quantised spectrum once per decomposition, and the filter for one A
   becomes an intersection of H bitmaps rather than a scan of the whole table.
   census.c and good_c have both been built around this for a long time; what
   makes it worth more here is that the table being filtered is small.

   Quantising loses nothing: levels are floors, so selecting the rows whose
   level is at most the level of the cap is a superset of those whose spectrum
   is at most the cap, and the final autocorrelation check is exact, so a looser
   filter costs work and cannot cost solutions. */
#ifndef ACHUNK
#define ACHUNK 8192
#endif
#ifndef NW_CHUNK
#define NW_CHUNK 24
#endif
/* Spectrum quantisation for the pair filter.  Levels are floors, so a coarser
   grid admits rows the exact test would reject: at NLEV=64, 6.8% of the
   survivors at n=27 were quantisation slack rather than genuine, and each one
   costs a solve call.  256 recovers most of that for four times the bitmap,
   which is 0.7 MB at n=27 and 28 MB at n=37 (finding 402). */
#ifndef NLEV
#define NLEV 256
#endif
static int lev_of(double x, int fourn) {
    int v = (int)(x * NLEV / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > NLEV - 1) v = NLEV - 1;
    return v;
}

/* ------------------------------------------------------ canonical form

   Two quadruples are equivalent when an automorphism of Z_n and four circulant
   signed permutation matrices carry one to the other (arXiv 2605.08661, Sec 1),
   that is: decimate all four together, and rotate or negate each row on its own,
   with the four rows unordered.  Rotation and negation act per row, so each row
   has its own cheap canonical form, the permutation becomes a sort, and only
   the decimation is searched.

   Doing this in a fixed order is what does not work: rotating A can make it
   symmetric, which only then lets the distinguished role move, so a single pass
   splits classes.  Treating the object as an unordered quadruple of circulants
   sidesteps that entirely, since "three of them are symmetric" is then a
   property of the class rather than of the representative. */
static uint64_t nw_rot(uint64_t m, int s) {
    uint64_t full = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    return ((m >> s) | (m << (N - s))) & full;
}
/* Reversal j -> -j of a single row.  It leaves every autocorrelation alone, so
   it is a symmetry of the defining equation, and it does nothing to a symmetric
   row while acting non-trivially on a skew one.  Djokovic and Kotsireas include
   it among their elementary transformations, which is why an equivalence class
   of theirs can hold two of the "small classes" the earlier literature counted;
   the near Williamson convention (rotate and negate only) does not include it.
   GMAT_REV=0 drops it and reproduces the finer count. */
static int USE_REV = 1;
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
static int *DECIM;                    /* [phi(n)][n] position maps */
static int NDECIM;
int *DECIM_FWD; int NDECIM_FWD;
static void decim_init(void) {
    int cnt = 0;
    for (int t = 1; t < N; t++) { int a = t, b = N; while (b) { int q = a % b; a = b; b = q; } if (a == 1) cnt++; }
    NDECIM = cnt; NDECIM_FWD = cnt;
    DECIM = malloc(sizeof(int) * (size_t)cnt * N);
    DECIM_FWD = DECIM; NDECIM_FWD = cnt;
    int i = 0;
    for (int t = 1; t < N; t++) {
        int a = t, b = N; while (b) { int q = a % b; a = b; b = q; }
        if (a != 1) continue;
        for (int j = 0; j < N; j++) DECIM[(size_t)i * N + j] = (t * j) % N;
        i++;
    }
}
static uint64_t nw_decimate(uint64_t m, const int *map) {
    uint64_t r = 0;
    for (int j = 0; j < N; j++) if ((m >> map[j]) & 1) r |= 1ULL << j;
    return r;
}
static void nw_canon(const uint64_t *q, uint64_t *out) {
    uint64_t best[4]; int have = 0;
    for (int i = 0; i < NDECIM; i++) {
        const int *map = DECIM + (size_t)i * N;
        uint64_t r[4];
        for (int u = 0; u < 4; u++) r[u] = nw_row_canon(nw_decimate(q[u], map));
        for (int a = 0; a < 3; a++)          /* sort 4 */
            for (int b = 0; b < 3 - a; b++)
                if (r[b] > r[b + 1]) { uint64_t t = r[b]; r[b] = r[b + 1]; r[b + 1] = t; }
        if (!have || memcmp(r, best, sizeof best) < 0) { memcpy(best, r, sizeof best); have = 1; }
    }
    memcpy(out, best, sizeof best);
}
/* open-addressed set of canonical quadruples */
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
static uint64_t sym_mask(uint32_t code) {
    uint64_t m = 1ULL;                       /* x_0 = +1 */
    for (int j = 1; j <= H; j++)
        if ((code >> (j - 1)) & 1) { m |= 1ULL << j; m |= 1ULL << (N - j); }
    return m;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: nw_c <n> [threads] [out]\n"); return 1; }
    N = atoi(argv[1]);
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    const char *outpath = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    if (N < 3 || (N % 2) == 0) { fprintf(stderr, "n must be odd and >= 3\n"); return 1; }
    H = (N - 1) / 2;
    /* Raising this limit is not a one-line change.  Three things in this file
       assume it, and none of them fails loudly if it is lifted:
         - the bit reversal `__builtin_bswap32(rv) >> (32 - H)` truncates bit 32
           and shifts by a negative amount once H > 32, and is then wrong on
           every input, not on some;
         - `1ULL << j` over positions is undefined for j >= 64, and x86 masks
           the count so bit 64 silently becomes bit 0;
         - the row masks read (N >= 64) ? ~0ULL : ((1ULL << N) - 1), which is
           the wrong mask above 64.
       good_c.c raised its equivalent limit with -DGC_WIDE/-DGS_WIDE and did not
       update the first two.  The n=65 and n=67 censuses came back empty, were
       published as new results, and had to be withdrawn (finding 522); a
       regression on a published order-127 family later showed that build
       recovering zero candidates for a quadruple that exists.  If this guard is
       lifted, fix those three first and add a known-answer fixture above the
       new boundary, because reproducing a stored census cannot test a path that
       only runs past where the stored censuses stop. */
    if (N > 63) { fprintf(stderr, "n > 63 needs wider words\n"); return 1; }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    binom_init();
    cos_init();
    if (!hkt_build()) { fprintf(stderr, "hkt_build failed\n"); return 1; }
    /* The invariant finding 392 removes the per-call test for.  Checked here
       rather than assumed: it holds at every order from 7 to 45, but it is a
       property of the construction, not of the definition. */
    {   /* Finding 397 rests on t = T(rhs1) being linear; check it here rather
           than assume it, as with the mod-4 invariant above. */
        uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
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
            fprintf(stderr, "n=%d: FIXED_[%d]=%d breaks the mod-4 invariant\n",
                    N, k, FIXED_[k]);
            return 1;
        }
    cols_init();
    cols_selfcheck();
    {   const char *kv = getenv("NW_CCK");
        CCK = kv ? atoi(kv) : cc_choose_k();
        if (CCK < 0) CCK = 0;
        if (CCK > S2_MAXFREE) CCK = S2_MAXFREE;
        if (getenv("NW_SCALARAND")) SCALAR_AND = atoi(getenv("NW_SCALARAND"));
        if (getenv("NW_SCALARMC")) SCALAR_MC = atoi(getenv("NW_SCALARMC"));
    }
    double tcc = wnow();
    if (CCK) cc_build();
    printf("   consistency table: K=%d words per pattern, %.1f MB, built in %.2fs\n",
           CCK, CCK ? ((double)CCK * 4 * (1ULL << H)) / 1048576.0 : 0.0, wnow() - tcc);
    double t0 = wnow();

    /* Row sums: a^2 + b^2 + c^2 + d^2 = 4n with b, c, d congruent to n mod 4,
       which is automatic once each symmetric row is normalised to x_0 = 1. */
    int fourn = 4 * N;
    printf("n=%d: 4n = %d\n", N, fourn);

    Decomp dec[64];
    int nd = decomps(dec, 64);
    printf("   %d row-sum decompositions\n", nd);
    { const char *dv = getenv("NW_DONLY");
      if (dv) printf("   running decomposition %s only\n", dv); }
    /* Report the decompositions and stop.  A runner that needs the count to
       loop over them must not have to run the census to find it out. */
    if (getenv("NW_LIST")) {
        for (int i = 0; i < nd; i++)
            printf("   decomposition %d: a=%d b=%d c=%d d=%d\n",
                   i, dec[i].a, dec[i].b, dec[i].c, dec[i].d);
        return 0;
    }

    decim_init();          /* needed by the A-row canonicity test below */
    dbyte_init();
    long long nA = 0, tested = 0, found = 0;
    long long solcap = 1024, nsol = 0;
    uint64_t *sol = malloc((size_t)solcap * 4 * sizeof(uint64_t));
    uint32_t nsym = 1u << H;
    /* Size the B table by how many skew codes actually pass the spectral
       ceiling rather than by 2^H.  At v=51 the difference is between 5.2 GB
       and a fraction of it, which decides whether the order runs at all. */
    long long bkeep = 0;
    {   int8_t paf[64]; float scratch[64];
        /* The same predicate as the fill pass below, computed by the same
           function.  Using a different one here (psd_max_only) let the two
           disagree on a borderline row by one rounding step, so nkeep could
           exceed the allocation and the B-table indices ran off the end. */
        for (uint32_t c = 0; c < nsym; c++) {
            paf_mask(skew_mask(c), paf);
            if (!(psd_all(paf, scratch) > 4.0 * N + 1e-6)) bkeep++;
        }
        printf("   skew codes passing the ceiling: %lld of %u (%.1f%%)\n",
               bkeep, nsym, 100.0 * bkeep / nsym);
    }
    size_t btab = (size_t)bkeep + 1;
    int8_t *SPAF = malloc(btab * H);
    /* Spectra are kept beside the autocorrelations rather than recomputed:
       the pair filter used to call psd_at, itself O(H), once per frequency
       per pair, so the filter was O(H^2) per pair and rebuilt A's spectrum
       inside the B loop. */
    float *SPSD = malloc(btab * H * sizeof(float));
    uint32_t *SCODE = malloc(btab * sizeof(uint32_t));
    /* Per-B half of the prefix (finding 393): u_k = (n - PAF_B(k))/4 for the
       bvec loop, and its low bits packed into one word, already carried
       through the t map (finding 397) so the pair loop is a single XOR. */
    /* The two live in one record per B row, the t-map word first and the u
       values behind it, so a survivor touches one line rather than two.  The
       u field is padded to the eight-lane reads nw_bvec makes, and the stride
       is a multiple of 8: 32 bytes through H=24, 40 through H=32. */
    const size_t BSTRIDE = (size_t)((4 + ((H + 7) & ~7) + 7) & ~7);
    uint8_t *BREC = aligned_alloc(64, (btab + 1) * BSTRIDE);
    if (!SPAF || !SPSD || !SCODE || !BREC) {
        fprintf(stderr, "out of memory for the B table (%lld rows)\n", bkeep);
        exit(1);
    }

    /* A single decomposition can be run on its own, which is how a long run is
       made restartable: there is no checkpoint inside one, but there are eight
       to twelve of them per order and each writes its own solutions, so a run
       that dies resumes at the next one rather than the beginning. */
    int donly = -1;
    { const char *dv = getenv("NW_DONLY"); if (dv) donly = atoi(dv); }
    { const char *cv = getenv("NW_NODEC"); if (cv) NW_DECCANON = 0; }
    { const char *rv = getenv("GMAT_REV"); if (rv) USE_REV = atoi(rv); }
    for (int di = 0; di < nd; di++) {
        if (donly >= 0 && di != donly) continue;

        /* Only masks with bit N-1 clear can be rotation-canonical: with pa < N
           there is a zero somewhere, and rotating it to the top gives a
           smaller value.  Bit 0 must be set for the same kind of reason: if it
           were clear then rot(m,1) is just m >> 1, which is smaller.  So the
           enumeration runs over the pa-1 free bits in positions 1..N-2, which
           is N/(N-pa) times (N-1)/pa fewer masks than all of them, four to six
           times fewer here (findings 409 and 410). */
        long long ncomb = 1LL << H;      /* skew codes */
        /* The decomposition also fixes what C and D's row sums must be, and for
           a symmetric row normalised to x_0 = 1 the row sum is 1 + 4p - 2H, so
           the sum is a popcount of the code.  The solve works modulo 4 and most
           of its lifts are not solutions: at n=31 it emits 1.4e10 candidates for
           1.4e6 quadruples, so 99.99% are rejected, and each rejection used to
           cost H popcount-and-rotate steps.  Checking the two popcounts first
           costs two instructions and throws most of them out. */
        int pc = sym_popcount_for(dec[di].c), pdd = sym_popcount_for(dec[di].d);
        /* B table: skew rows that pass the ceiling.  Every skew row has row
           sum 1, so this is all 2^H of them rather than one weight class. */
        int nkeep = 0;
        {
            int8_t paf[64];
            for (uint32_t c = 0; c < nsym; c++) {
                /* every skew row has row sum 1, so unlike the symmetric table
                   in nw_c.c there is no row-sum filter to thin it with */
                paf_mask(skew_mask(c), paf);
                if (psd_all(paf, SPSD + (size_t)nkeep * H) > fourn + 1e-6) continue;
                memcpy(SPAF + (size_t)nkeep * H, paf, H);
                {   uint32_t rb = 0;
                    uint8_t *rec = BREC + (size_t)nkeep * BSTRIDE;
                    memset(rec, 0, BSTRIDE);
                    for (int k = 1; k <= H; k++) {
                        int u = (N - (int)paf[k - 1]) >> 2;
                        rec[4 + k - 1] = (uint8_t)(int8_t)u;
                        if (u & 1) rb |= 1u << (k - 1);
                    }
                    uint32_t rbt = nw_tmap(rb);
                    memcpy(rec, &rbt, 4); }
                SCODE[nkeep] = c; nkeep++;
                if ((size_t)nkeep > btab) {
                    fprintf(stderr, "B table overflow: counted %lld, filling %d\n",
                            bkeep, nkeep);
                    exit(1);
                }
            }
        }
        if (!nkeep) continue;

        /* index the B table: bmp[k][t] holds the rows whose level at frequency
           k is at most t, prefix-ored so one bitmap per frequency answers the
           whole constraint */
        int cwords = (nkeep + 63) >> 6;
            size_t bmpn = (size_t)H * NLEV * cwords;
        uint64_t *bmp = calloc(bmpn, sizeof(uint64_t));
        if (!bmp) {
            /* Unchecked, this returned NULL at n=59 and the fill loop below
               segfaulted with nothing on stdout, since the ceiling line was
               still in the buffer.  The index is H x NLEV bitmaps over the
               kept rows, so it grows with nkeep: 344 MB at n=43, where
               511,750 codes of 2^21 pass the ceiling, against 62 GB at
               n=59, where 71,985,426 of 2^29 do.  At the default NLEV that
               is what puts the order out of reach, not the search; build
               with a smaller NLEV to fit it. */
            fprintf(stderr,
                    "B-table index needs %.1f GB (%d rows, H=%d, NLEV=%d); "
                    "order %d is out of reach for this index\n",
                    bmpn * 8.0 / (1024.0*1024.0*1024.0), nkeep, H, NLEV, N);
            exit(1);
        }
        for (int bi = 0; bi < nkeep; bi++)
            for (int k = 0; k < H; k++) {
                int lv = lev_of(SPSD[(size_t)bi * H + k], fourn);
                bmp[(((size_t)k * NLEV) + lv) * cwords + (bi >> 6)] |= 1ULL << (bi & 63);
            }
        for (int k = 0; k < H; k++)
            for (int t = 1; t < NLEV; t++) {
                uint64_t *dst = bmp + (((size_t)k * NLEV) + t) * cwords;
                const uint64_t *src = dst - cwords;
                for (int w = 0; w < cwords; w++) dst[w] |= src[w];
            }

        /* Finding 460.  A range over the skew index, so a long run can be
           split into pieces that are run one after another and concatenated.
           gmat_c has no checkpoint, and NW_DONLY only splits by decomposition,
           which is no help at n=59 or 61 where there is exactly one.  Without
           this a two-day run that dies has nothing to resume from. */
        long long CLO = 0, CHI = ncomb;
        {   const char *v = getenv("GMAT_CLO"); if (v) CLO = atoll(v);
            const char *w = getenv("GMAT_CHI");
            if (w) { long long h = atoll(w); if (h >= 0 && h < CHI) CHI = h; }
            if (CLO < 0) CLO = 0;
            if (CLO || CHI != ncomb)
                fprintf(stderr, "   skew range [%lld, %lld) of %lld\n",
                        CLO, CHI, ncomb);
        }
#pragma omp parallel
        {
            int8_t pafa[64];
            float psda[64];
            uint32_t cand[HKT_MAXC];
            uint32_t tb[2 * NW_CHUNK + 8]; int ta[64]; int16_t mc4[4][64]; (void)mc4;
            uint32_t pend_t[8], pend_bb[8]; int pend_b[8]; int npend = 0;
            int *surv = malloc(sizeof(int) * (size_t)nkeep);
            long long lA = 0, ltested = 0, lfound = 0, lsurvn = 0, lpass = 0;
            unsigned long long cpre = 0, celim = 0, ctail = 0; (void)cpre; (void)celim; (void)ctail;
            double lsurv = 0, lsolve = 0, lchunk = 0;
#pragma omp for schedule(dynamic, 1)
            for (long long c0 = CLO; c0 < CHI; c0 += ACHUNK) {
              long long chi = c0 + ACHUNK; if (chi > CHI) chi = CHI;
              double _tc0 = tcpun();
              for (long long ci = c0; ci < chi; ci++) {
                /* A skew row is its H free bits; every one has row sum 1, so
                   there is no popcount to enumerate by. */
                uint64_t am = skew_mask((uint32_t)ci);
                if (NW_DECCANON ? !a_is_canonical_dec(am)
                                : !a_is_canonical(am)) continue;
                paf_mask(am, pafa);
                if (psd_all(pafa, psda) > fourn + 1e-6) continue;
                lA++;
                uint32_t ra; nw_arow(pafa, ta, &ra);
                double _t1 = tcpun();
                /* Collect the survivors of the pair filter, then run the
                   eliminations two at a time.  The two are independent, so one
                   lane's work fills the other's stalls; good_c's pair loop has
                   done this since finding 326 and this program did not. */
                int nsurv = 0;
                {
                    const uint64_t *bp[64]; int nbp = 0, dead = 0;
                    for (int k = 0; k < H && !dead; k++) {
                        double cap = (double)fourn - psda[k];
                        if (cap < 0) { dead = 1; break; }
                        int t = lev_of(cap, fourn);
                        if (t >= NLEV - 1) continue;      /* no constraint here */
                        bp[nbp++] = bmp + (((size_t)k * NLEV) + t) * cwords;
                    }
                    if (!dead) {
                        /* Finding 508 in good_c: the AND chain has no useful
                           early exit, so four words go through one AVX2
                           register.  The masks one A row touches are
                           nbp x cwords words, L2-resident, so instruction
                           count sets the cost.  NW_SCALARAND=1 keeps the
                           old scalar chain for comparison. */
                        int w = 0;
#if defined(__AVX2__)
                        if (!SCALAR_AND && nbp > 0) {
                            for (; w + 4 <= cwords; w += 4) {
                                __m256i fv = _mm256_loadu_si256((const __m256i *)(bp[0] + w));
                                for (int i = 1; i < nbp; i++)
                                    fv = _mm256_and_si256(fv, _mm256_loadu_si256((const __m256i *)(bp[i] + w)));
                                if (_mm256_testz_si256(fv, fv)) continue;
                                uint64_t v4[4];
                                _mm256_storeu_si256((__m256i *)v4, fv);
                                for (int q = 0; q < 4; q++) {
                                    uint64_t v = v4[q];
                                    while (v) {
                                        int b = __builtin_ctzll(v);
                                        v &= v - 1;
                                        surv[nsurv++] = ((w + q) << 6) + b;
                                    }
                                }
                            }
                        }
#endif
                        for (; w < cwords; w++) {
                            uint64_t v = ~0ULL;
                            for (int i = 0; i < nbp; i++) v &= bp[i][w];
                            if (w == cwords - 1 && (nkeep & 63))
                                v &= (1ULL << (nkeep & 63)) - 1;
                            while (v) {
                                int b = __builtin_ctzll(v);
                                v &= v - 1;
                                surv[nsurv++] = (w << 6) + b;
                            }
                        }
                    }
                }
                double _t2 = tcpun(); lsurv += _t2 - _t1;
                lsurvn += nsurv;
                uint32_t maskH = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);

#define NW_EMIT(NCD, CANDS, SBITS, BIDX)                                       \
                do {                                                           \
                    const int8_t *pf = SPAF + (size_t)(BIDX) * H;              \
                    for (int ci = 0; ci < (NCD); ci++) {                        \
                        uint32_t vv = (CANDS)[ci];                             \
                        uint32_t codeC = (~vv) & maskH;                        \
                        uint32_t codeD = ((SBITS) ^ vv) & maskH;               \
                        uint64_t wc = hkt_symword(codeC), wd = hkt_symword(codeD); \
                        int good = 1;                                           \
                        for (int k = 1; k <= H && good; k++) {                  \
                            int t2 = (int)pafa[k - 1] + (int)pf[k - 1] + 2 * N  \
                                   - 2 * (__builtin_popcountll(wc ^ hkt_rotk(wc, k)) \
                                        + __builtin_popcountll(wd ^ hkt_rotk(wd, k))); \
                            if (t2 != 0) good = 0;                              \
                        }                                                       \
                        if (!good) continue;                                    \
                        lfound++;                                               \
                        _Pragma("omp critical")                                 \
                        {                                                       \
                            if (nsol == solcap) {                               \
                                solcap *= 2;                                    \
                                sol = realloc(sol, (size_t)solcap * 4 * sizeof(uint64_t)); \
                            }                                                   \
                            uint64_t *r = sol + (size_t)nsol * 4;               \
                            r[0] = am; r[1] = SCODE[BIDX];                      \
                            r[2] = codeC; r[3] = codeD;                         \
                            nsol++;                                             \
                        }                                                       \
                    }                                                           \
                } while (0)

/* Columns for eight queued patterns into ct. */
#if defined(__AVX2__)
#define NW_COLS8(T8, CT, NP) do { nw_cols4((T8), (CT), 0); if ((NP) > 4) nw_cols4((T8) + 4, (CT), 4); } while (0)
#else
#define NW_COLS8(T8, CT, NP) do { for (int L = 0; L < (NP); L++) { uint32_t cl[64]; nw_cols_scalar((T8)[L], cl); for (int k = 0; k < H; k++) (CT)[(size_t)k * 8 + L] = cl[k]; } } while (0)
#endif
/* Eight queued survivors through one elimination.  Columns go straight into
   the transposed layout ct[c*8 + lane] four lanes at a time; dead lanes carry
   zero columns and a zero right-hand side, which makes their updates the
   identity.  Finding 450's consistency test runs before the de-transpose. */
#define NW_FLUSH()                                                             \
                do {                                                           \
                    uint32_t ct[64 * 8], pb8[64 * 8], pm8[8], bbv[8];           \
                    for (int L = npend; L < 8; L++) pend_t[L] = 0;              \
                    for (int k = 0; k < H; k++)                                 \
                        for (int L = npend; L < 8; L++) ct[(size_t)k * 8 + L] = 0; \
                    unsigned long long _f0 = GM_TICK();                         \
                    NW_COLS8(pend_t, ct, npend);                                \
                    for (int L = 0; L < 8; L++) bbv[L] = L < npend ? pend_bb[L] : 0u; \
                    NW_ELIM8(ct, bbv, pb8, pm8);                                \
                    unsigned long long _f1 = GM_TICK(); celim += _f1 - _f0;     \
                    for (int L = 0; L < npend; L++) {                           \
                        if (bbv[L] & ~pm8[L]) continue;                         \
                        uint32_t colL[64], pivL[64];                            \
                        for (int k = 0; k < H; k++) {                           \
                            colL[k] = ct[(size_t)k * 8 + L];                    \
                            pivL[k] = pb8[(size_t)k * 8 + L];                   \
                        }                                                       \
                        int nc = hkt_tail_m(colL, bbv[L], pm8[L], pivL, cand);  \
                        ltested += nc;                                          \
                        int nk = nw_rowfilter(cand, nc, pend_t[L], pc, pdd, maskH); \
                        NW_EMIT(nk, cand, pend_t[L], pend_b[L]);                \
                    }                                                           \
                    ctail += GM_TICK() - _f1;                                   \
                    npend = 0;                                                  \
                } while (0)


                /* Two passes over a chunk of survivors.  The first computes
                   each one's pattern t and prefetches its table words and its
                   B-row u values; the second does the work that reads them,
                   by which time the lines have had a chunk's arithmetic to
                   arrive in (finding 391).  A survivor whose system the table
                   proves inconsistent stops there.  The rest queue up eight
                   at a time for the batched elimination (finding 454), with
                   their columns built directly in the transposed layout. */
                /* The patterns of chunk i+1 are formed, and their table words
                   and B records prefetched, while chunk i is being worked, so
                   every line has a whole chunk of work to arrive in rather
                   than the few instructions the single-chunk form gave the
                   first entries; the prefetches are spread one per survivor
                   so they do not burst past the fill buffers. */
                uint32_t *tcur = tb, *tnext = tb + NW_CHUNK;
                int mch = nsurv < NW_CHUNK ? nsurv : NW_CHUNK;
                for (int u = 0; u < mch; u++) {
                    int b = surv[u];
                    uint32_t rbt; memcpy(&rbt, BREC + (size_t)b * BSTRIDE, 4);
                    uint32_t t = ra ^ rbt;
                    tcur[u] = t;
                    if (CCK) __builtin_prefetch(CC + (size_t)t * CCK);
                    if (NW_CHUNK + u < nsurv)
                        __builtin_prefetch(BREC + (size_t)surv[NW_CHUNK + u] * BSTRIDE);
                }
                for (int s0i = 0; s0i < nsurv; s0i += NW_CHUNK) {
                    int s1i = s0i + NW_CHUNK;
                    int mnext = nsurv - s1i; if (mnext > NW_CHUNK) mnext = NW_CHUNK;
                    unsigned long long _k0 = GM_TICK();
                    int s2i = s1i + NW_CHUNK;
                    int m2 = nsurv - s2i; if (m2 > NW_CHUNK) m2 = NW_CHUNK;
                    for (int u = 0; u < mch; u++) {
                        /* B records two chunks ahead, so the t-map word read
                           one chunk ahead finds its line already present */
                        if (u < m2) __builtin_prefetch(BREC + (size_t)surv[s2i + u] * BSTRIDE);
                        if (u < mnext) {
                            int b = surv[s1i + u];
                            uint32_t rbt; memcpy(&rbt, BREC + (size_t)b * BSTRIDE, 4);
                            uint32_t t = ra ^ rbt;
                            tnext[u] = t;
                            if (CCK) __builtin_prefetch(CC + (size_t)t * CCK);
                        }
                        uint32_t t = tcur[u];
                        int b0i = surv[s0i + u];
                        const int8_t *su = (const int8_t *)(BREC + (size_t)b0i * BSTRIDE + 4);
                        int16_t mcs[64];
                        const int16_t *mc = mcs;
                        uint32_t bb;
#if defined(__AVX2__)
                        /* four patterns' mc at a time; tcur has slack past
                           mch, so the group may run over the chunk end */
                        if (!SCALAR_MC) {
                            if ((u & 3) == 0) nw_mc4(tcur + u, mc4);
                            mc = mc4[u & 3];
                        } else
#endif
                        nw_mc_from_t(t, mcs);
                        if (CCK) {
                            const uint32_t *e = CC + (size_t)t * CCK;
                            if (e[0] == CC_REJECT) continue;
                            bb = nw_bvec(mc, ta, su);
                            int bad = 0;
                            for (int i = 0; i < CCK; i++) {
                                uint32_t w = e[i] & ~CC_MORE;
                                if (!w) break;
                                if (__builtin_parity(bb & w)) { bad = 1; break; }
                            }
                            if (bad) continue;
                        } else {
                            bb = nw_bvec(mc, ta, su);
                        }
                        lpass++;
                        pend_t[npend] = t; pend_bb[npend] = bb; pend_b[npend] = b0i;
                        if (++npend == 8) {
                            unsigned long long _k1 = GM_TICK(); cpre += _k1 - _k0;
                            NW_FLUSH();
                            _k0 = GM_TICK();
                        }
                    }
                    cpre += GM_TICK() - _k0;
                    { uint32_t *sw = tcur; tcur = tnext; tnext = sw; }
                    mch = mnext;
                }
                if (npend) { NW_FLUSH(); }
                lsolve += tcpun() - _t2;
#undef NW_EMIT
#undef NW_FLUSH
              }
              lchunk += tcpun() - _tc0;
            }
            free(surv);
#pragma omp atomic
            GM_SURV += lsurv;
#pragma omp atomic
            GM_SOLVE += lsolve;
#pragma omp atomic
            GM_CHUNK += lchunk;
#pragma omp atomic
            nA += lA;
#pragma omp atomic
            tested += ltested;
#pragma omp atomic
            found += lfound;
#pragma omp atomic
            GM_NSURV += lsurvn;
#pragma omp atomic
            GM_NPASS += lpass;
#ifdef GM_SPLIT
#pragma omp atomic
            GM_CPRE += cpre;
#pragma omp atomic
            GM_CELIM += celim;
#pragma omp atomic
            GM_CTAIL += ctail;
#endif
        }
        free(bmp);
    }

    printf("   canonical A rows passing the ceiling: %lld\n", nA);
    printf("n=%d: %lld quadruples (%lld candidates from the solve), %.2fs\n",
           N, found, tested, wnow() - t0);
    if (GM_CHUNK > 0)
        printf("   thread-seconds: A prep %.2f | survivors %.2f | solve %.2f "
               "(prep = chunk total minus the other two)\n",
               GM_CHUNK - GM_SURV - GM_SOLVE, GM_SURV, GM_SOLVE);
    printf("   pair-filter survivors: %lld, of which %lld (%.1f%%) reached the elimination\n",
           GM_NSURV, GM_NPASS, GM_NSURV ? 100.0 * GM_NPASS / GM_NSURV : 0.0);
#ifdef GM_SPLIT
    {   double tot = (double)(GM_CPRE + GM_CELIM + GM_CTAIL);
        printf("   solve cycles: precheck %.1f%% | columns+elimination %.1f%% | tail %.1f%%"
               "  (%.0f cycles per survivor, %.0f per eliminated system)\n",
               100.0 * GM_CPRE / tot, 100.0 * GM_CELIM / tot, 100.0 * GM_CTAIL / tot,
               GM_NSURV ? tot / GM_NSURV : 0.0, GM_NPASS ? (double)GM_CELIM / GM_NPASS : 0.0); }
#endif
    {
        double tc = wnow();
        size_t cap = 1; while (cap < (size_t)nsol * 4 + 16) cap <<= 1;
        hset_init(cap);
        /* One line per inequivalent class, written as the canonical
           representative: the four rows are decimated together, each is put in
           its own least rotation-or-negation, and the four are then sorted, so
           which row is the unconstrained one is not recoverable from the line
           and need not be.  This is what makes a census storable: at n=27 it is
           6,579 lines rather than 347,418. */
        FILE *rf = NULL;
        { const char *rp = getenv("NW_REPS"); if (rp) rf = fopen(rp, "w"); }
        uint64_t q[4], c[4];
        for (long long i = 0; i < nsol; i++) {
            uint64_t *r = sol + (size_t)i * 4;
            q[0] = r[0];                                  /* A, skew mask */
            q[1] = skew_mask((uint32_t)r[1]);             /* B, skew code   */
            q[2] = sym_mask((uint32_t)r[2]);              /* C, symmetric   */
            q[3] = sym_mask((uint32_t)r[3]);              /* D, symmetric   */
            nw_canon(q, c);
            if (hset_add(c) && rf) {
                for (int u = 0; u < 4; u++) {
                    if (u) fputc(' ', rf);
                    for (int j = 0; j < N; j++)
                        fputc(((c[u] >> j) & 1) ? '+' : '-', rf);
                }
                fputc('\n', rf);
            }
        }
        if (rf) fclose(rf);
        printf("   inequivalent classes: %zu   (canonicalised in %.2fs)\n", HCNT, wnow() - tc);
    }
    if (outpath) {
        FILE *f = fopen(outpath, "w");
        for (long long i = 0; i < nsol; i++) {
            uint64_t *r = sol + (size_t)i * 4;
            uint64_t w[4];
            w[0] = r[0];
            w[1] = skew_mask((uint32_t)r[1]);
            w[2] = sym_mask((uint32_t)r[2]);
            w[3] = sym_mask((uint32_t)r[3]);
            for (int u = 0; u < 4; u++) {
                if (u) fputc(' ', f);
                for (int j = 0; j < N; j++)
                    fputc(((w[u] >> j) & 1) ? '+' : '-', f);
            }
            fputc('\n', f);
        }
        fclose(f);
        printf("   solutions -> %s\n", outpath);
    }
    return 0;
}
