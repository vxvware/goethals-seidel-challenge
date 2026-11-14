/* wil_hkt.c -- exhaustive census of Williamson matrices of odd order n by the
 * mod-4 linearisation of Holzmann, Kharaghani and Tayfeh-Rezaie (2008).
 *
 * A Williamson quadruple is four symmetric (-1,1) sequences A, B, C, D of
 * length n (x_j = x_{n-j}) with PAF_A + PAF_B + PAF_C + PAF_D = 0 at every
 * shift 1..(n-1)/2.  The row sums satisfy a^2 + b^2 + c^2 + d^2 = 4n with all
 * four odd.  Every row is normalised to x_0 = +1, which fixes the sign of its
 * row sum (s = n mod 4), so the positive decomposition a >= b >= c >= d maps
 * to one signed decomposition and one popcount per row.
 *
 * This is nw_c.c with the two enumerated rows both symmetric.  The solver in
 * hkt_solve.h is untouched: it takes the two known autocorrelations and
 * returns the symmetric C, D with x_0 = +1.  What changes:
 *
 *   rows      A and B are symmetric codes (H free bits) of the popcounts the
 *             decomposition fixes, drawn from the kind of table nw_c.c builds
 *             for B; there is no free circulant row
 *   A loop    quotiented by decimation only.  j -> sj for a unit s acts on all
 *             four rows at once and keeps them symmetric, and s and -s act the
 *             same way on a symmetric row, so A runs over the least code in
 *             its orbit under the phi(n)/2 folded decimations.  Rotation is
 *             not a symmetry of a symmetric row and is not used.
 *   a = b     B is restricted to codes >= code(A).  Of the two labellings of
 *             the equal-sum rows, the one whose row has the smaller orbit
 *             minimum satisfies it, so no class is lost.
 *   (c, d)    for a fixed (a, b) every (c, d) with c^2 + d^2 = 4n - a^2 - b^2
 *             is accepted in one pass over the (A, B) pairs
 *   c = d     the candidate set is closed under C <-> D, so codeC <= codeD
 *             keeps exactly one of each pair
 *
 * Other equal-sum coincidences (b = c and the like) yield the same quadruple
 * from more than one labelling.  The raw list is deduplicated as a set of four
 * codes before it is written, and the class count printed at the end
 * quotients by decimation as well, which is what classes.py counts.
 *
 * The stage-2 nullity cap of the solver (nullity > 6) is counted rather than
 * assumed: the run prints how many solve calls were dropped by it, and the
 * census is exhaustive only when that count is zero.
 *
 *   build: gcc -O3 -march=native -fopenmp -o wil_hkt wil_hkt.c -lm
 *   run:   ./wil_hkt n [threads] [out]
 *   env:   WIL_LIST=1    list the (a, b) groups and exit
 *          WIL_DONLY=i   run group i only
 *          WIL_NOCACHE=1 per-pair elimination (eight lanes, AVX2) instead of
 *                        the 2^H stage-2 table; =0 forces the table.  Default
 *                        is the table for H <= 24.
 *          WIL_AB=ij     which two of the sorted sums are enumerated
 *                        (default 01, the two largest)
 *          WIL_NLEV=L    PSD quantisation levels of the B index (default 256;
 *                        coarser is looser, never lossy)
 *          WIL_SAMPLE=k  solve every k-th chunk of A codes only and report
 *                        the sampled pair count and solve time per pair
 *
 * Output: one line per distinct quadruple found, four words of n '+'/'-'
 * characters (A B C D), the format the nearwill engines write.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static int N, H;
#define SCALE 16
#define CEIL16 (4 * N * SCALE)

#include "hkt_solve.h"

/* ------------------------------------------------- stage-2 elimination cache

   Verbatim from nw_c.c.  The stage-2 matrix depends on the sign pattern t
   alone, so its elimination, pivot layout and null-space basis are tabulated
   once for all 2^H patterns; the hot path is then a lookup plus a linear map
   on the right-hand side.  Hot fields (mc, chk, nchk, ok) are what every call
   reads; pt and basis are read only by the calls that pass the consistency
   check and live in a second array. */
static uint8_t *S2HOT, *S2COLD;
static size_t SHOT, SCOLD;
static size_t OFF_CHK, OFF_NCHK, OFF_OK, OFF_BASIS, OFF_NF;
#define S2_MAXFREE 6
static size_t CHKW;
static long long S2_NOTOK;          /* patterns rejected by the nullity cap */

static void s2_layout(void) {
    CHKW      = ((size_t)H + 7) / 8;
    OFF_CHK   = (size_t)H * 2;
    OFF_NCHK  = OFF_CHK + S2_MAXFREE * CHKW;
    OFF_OK    = OFF_NCHK + 1;
    SHOT      = OFF_OK + 1;
    if (SHOT < OFF_CHK + (S2_MAXFREE - 1) * CHKW + 4)
        SHOT = OFF_CHK + (S2_MAXFREE - 1) * CHKW + 4;
    SHOT      = (SHOT + 63) & ~(size_t)63;
    OFF_BASIS = (size_t)H * sizeof(uint32_t);
    OFF_NF    = OFF_BASIS + S2_MAXFREE * sizeof(uint32_t);
    SCOLD     = (OFF_NF + 1 + 3) & ~(size_t)3;
}

static void s2_build(void) {
    size_t nt = (size_t)1 << H;
    S2HOT  = aligned_alloc(64, nt * SHOT);
    S2COLD = malloc(nt * SCOLD);
    if (!S2HOT || !S2COLD) {
        fprintf(stderr, "out of memory for the stage-2 cache\n"); exit(1);
    }
    uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    uint32_t hmask = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    long long notok = 0;
#pragma omp parallel for schedule(static) reduction(+:notok)
    for (long long ti = 0; ti < (long long)nt; ti++) {
        size_t t = (size_t)ti;
        uint8_t *hot = S2HOT + t * SHOT, *cold = S2COLD + t * SCOLD;
        int16_t  *mc    = (int16_t *)hot;
        uint8_t  *chk   = hot + OFF_CHK;
        int8_t   *nchkp = (int8_t *)(hot + OFF_NCHK);
        int8_t   *okp   = (int8_t *)(hot + OFF_OK);
        uint32_t *pt    = (uint32_t *)cold;
        uint32_t *basis = (uint32_t *)(cold + OFF_BASIS);
        int8_t   *nfp   = (int8_t *)(cold + OFF_NF);
        uint32_t rv = (uint32_t)t;
        rv = ((rv >> 1) & 0x55555555u) | ((rv & 0x55555555u) << 1);
        rv = ((rv >> 2) & 0x33333333u) | ((rv & 0x33333333u) << 2);
        rv = ((rv >> 4) & 0x0f0f0f0fu) | ((rv & 0x0f0f0f0fu) << 4);
        rv = __builtin_bswap32(rv) >> (32 - H);
        uint64_t uw = 1ULL | ((uint64_t)t << 1) | ((uint64_t)rv << (H + 1));
        uint32_t a[64];
        for (int i = 0; i < 64; i++) a[i] = 0;
        for (int k = 1; k <= H; k++) {
            uint64_t kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
            uint64_t G    = kept & ~CONSTALL_[k] & maskn;
            uint64_t oddu = (G ^ hkt_rot(G, k)) & maskn;
            a[k - 1] = (uint32_t)(oddu >> 1) & hmask;
            mc[k - 1] = (int16_t)(__builtin_popcountll(kept & NFIXM_[k]) / 2);
        }
        uint32_t col[64];
        hkt_tr(a, H, col);
        uint32_t bt[32];
        for (int j = 0; j < H; j++) bt[j] = 1u << j;
        int piv[64]; for (int c = 0; c < H; c++) piv[c] = -1;
        uint32_t pv = 0; int rank = 0;
        for (int c = 0; c < H; c++) {
            uint32_t m = col[c] & ~pv;
            if (!m) continue;
            int q = __builtin_ctz(m);
            pv |= 1u << q; piv[c] = q; rank++;
            uint32_t R = col[c] & ~(1u << q);
            for (int c2 = 0; c2 < H; c2++)
                col[c2] ^= R & (uint32_t) - (int32_t)((col[c2] >> q) & 1u);
            for (int j = 0; j < H; j++)
                bt[j] ^= R & (uint32_t) - (int32_t)((bt[j] >> q) & 1u);
        }
        int nul = H - rank;
        if (nul > S2_MAXFREE || (1 << nul) > HKT_MAXC) {
            *okp = 0; *nchkp = 0; *nfp = 0; notok++; continue;
        }
        {   int nc = 0;
            for (int q = 0; q < H && nc < S2_MAXFREE; q++) {
                if ((pv >> q) & 1u) continue;
                uint32_t colq = 0;
                for (int j = 0; j < H; j++) if ((bt[j] >> q) & 1u) colq |= 1u << j;
                memcpy(chk + (size_t)nc * CHKW, &colq, CHKW);
                nc++;
            }
            *nchkp = (int8_t)nc;
        }
        for (int j = 0; j < H; j++) {
            uint32_t pj = 0;
            for (int c = 0; c < H; c++) {
                int q = piv[c];
                if (q >= 0 && ((bt[j] >> q) & 1u)) pj |= 1u << c;
            }
            pt[j] = pj;
        }
        int nf = 0;
        for (int c = 0; c < H; c++) {
            if (piv[c] >= 0) continue;
            uint32_t w = 1u << c;
            for (int c2 = 0; c2 < H; c2++) {
                int q = piv[c2];
                if (q >= 0 && ((col[c] >> q) & 1u)) w |= 1u << c2;
            }
            basis[nf++] = w;
        }
        *nfp = (int8_t)nf;
        *okp = 1;
    }
    S2_NOTOK = notok;
}

/* The per-row halves of the prefix (nw_c.c findings 393 and 397): the sign
   pattern is t = T(RA) ^ T(RB) with one half per row, and the target the bvec
   loop subtracts is TA[k] + u_k with u_k = (n - PAF_B(k)) / 4. */
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

static inline uint32_t nw_bvec(const uint8_t *hot, const int *ta,
                               const int8_t *su) {
    const int16_t *mc = (const int16_t *)hot;
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

/* Cache-free rebuild of mc and the stage-2 matrix from t (nw_c.c finding
   457), for the orders whose 2^H table does not fit. */
static inline void nw_rows_from_t(uint32_t t, int16_t *mc, uint32_t *col) {
    const uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    const uint32_t hmask = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    uint32_t rv = t;
    rv = ((rv >> 1) & 0x55555555u) | ((rv & 0x55555555u) << 1);
    rv = ((rv >> 2) & 0x33333333u) | ((rv & 0x33333333u) << 2);
    rv = ((rv >> 4) & 0x0f0f0f0fu) | ((rv & 0x0f0f0f0fu) << 4);
    rv = __builtin_bswap32(rv) >> (32 - H);
    uint64_t uw = 1ULL | ((uint64_t)t << 1) | ((uint64_t)rv << (H + 1));
    uint32_t a[64];
    for (int i = 0; i < 64; i++) { a[i] = 0; mc[i] = 0; }
    int k = 1;
    for (; k + 1 <= H; k += 2) {
        const uint64_t nc0 = ~CONSTALL_[k] & maskn, nc1 = ~CONSTALL_[k + 1] & maskn;
        uint64_t r0 = ((uw << (N - k)) | (uw >> k)) & maskn;
        uint64_t r1 = ((uw << (N - k - 1)) | (uw >> (k + 1))) & maskn;
        uint64_t kept0 = (~(uw ^ r0)) & maskn, kept1 = (~(uw ^ r1)) & maskn;
        uint64_t G0 = kept0 & nc0, G1 = kept1 & nc1;
        uint64_t o0 = (G0 ^ (((G0 << k) | (G0 >> (N - k))) & maskn)) & maskn;
        uint64_t o1 = (G1 ^ (((G1 << (k + 1)) | (G1 >> (N - k - 1))) & maskn)) & maskn;
        a[k - 1] = (uint32_t)(o0 >> 1) & hmask;
        a[k]     = (uint32_t)(o1 >> 1) & hmask;
        mc[k - 1] = (int16_t)(__builtin_popcountll(kept0 & NFIXM_[k]) / 2);
        mc[k]     = (int16_t)(__builtin_popcountll(kept1 & NFIXM_[k + 1]) / 2);
    }
    for (; k <= H; k++) {
        uint64_t kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
        uint64_t G    = kept & ~CONSTALL_[k] & maskn;
        uint64_t oddu = (G ^ hkt_rot(G, k)) & maskn;
        a[k - 1] = (uint32_t)(oddu >> 1) & hmask;
        mc[k - 1] = (int16_t)(__builtin_popcountll(kept & NFIXM_[k]) / 2);
    }
    hkt_tr(a, H, col);
}

/* -------------------------------------------------------------- row filter

   ALLOWD[pc] is the set of popcounts pd such that (pc, pd) are the popcounts
   of some (C, D) completing the current (a, b): every (c, d) with
   c^2 + d^2 = 4n - a^2 - b^2, in both orders.  When C and D have the same
   popcount the candidate set contains both (C, D) and (D, C), so codeC <=
   codeD keeps one of them (nw_c.c finding 406). */
static uint32_t ALLOWD[64];

static inline int wil_keep(uint32_t v, uint32_t s0, uint32_t maskH) {
    int a = H - __builtin_popcount(v & maskH);        /* popcount of codeC */
    uint32_t am = ALLOWD[a];
    if (!am) return 0;
    uint32_t cD = (s0 ^ v) & maskH;
    int b = __builtin_popcount(cD);
    if (!((am >> b) & 1u)) return 0;
    return a != b || ((~v) & maskH) <= cD;
}

static inline int wil_rowfilter(uint32_t *cand, int nc, uint32_t s, uint32_t maskH) {
    int keep = 0;
    for (int i = 0; i < nc; i++)
        if (wil_keep(cand[i], s, maskH)) cand[keep++] = cand[i];
    return keep;
}

/* The cached solve, nw_c.c's nw_solve_at with the row filter above.  capped
   counts the calls that hit a pattern the nullity cap rejected. */
static inline int wil_solve_at(const uint8_t *hot, const int *ta, const int8_t *su,
                               uint32_t *cand, uint32_t s0, uint32_t maskH,
                               long long *raw, long long *capped) {
    if (!*(const int8_t *)(hot + OFF_OK)) { (*capped)++; return 0; }
    uint32_t bb = nw_bvec(hot, ta, su);
    {   int nchk = *(const int8_t *)(hot + OFF_NCHK);
        const uint8_t *chk = hot + OFF_CHK;
        uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
        for (int i = 0; i < nchk; i++) {
            uint32_t c;
            memcpy(&c, chk + (size_t)i * CHKW, 4);
            if (__builtin_parity(bb & c & hm)) return 0;
        }
    }
    const uint8_t *cold = S2COLD + (size_t)s0 * SCOLD;
    const uint32_t *pt = (const uint32_t *)cold;
    uint32_t part = 0;
    for (uint32_t t = bb; t; t &= t - 1) part ^= pt[__builtin_ctz(t)];
    const uint32_t *basis = (const uint32_t *)(cold + OFF_BASIS);
    int nf = *(const int8_t *)(cold + OFF_NF), ncand = 1 << nf;
    *raw += ncand;
    uint32_t v = part;
    int keep = 0;
    for (int i = 0; ; ) {
        if (wil_keep(v, s0, maskH)) cand[keep++] = v;
        if (++i >= ncand) break;
        v ^= basis[__builtin_ctz((unsigned)i)];
    }
    return keep;
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
static inline uint64_t sym_mask(uint32_t code) {
    uint64_t m = 1ULL;
    for (int j = 1; j <= H; j++)
        if ((code >> (j - 1)) & 1) { m |= 1ULL << j; m |= 1ULL << (N - j); }
    return m;
}

/* PAF(s) = N - 2 popcount(m ^ rot(m, s)) for the +1 mask m */
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
/* all nonzero frequencies, and the maximum over them */
static double psd_all(const int8_t *paf, float *out) {
    double m = -1e18;
    for (int k = 1; k <= H; k++) {
        double v = psd_at(paf, k);
        out[k - 1] = (float)v;
        if (v > m) m = v;
    }
    return m;
}

/* -------------------------------------------------- decimation on codes

   x'_j = x_{sj} for a unit s of Z_n.  On a symmetric row the position sj folds
   to min(sj mod n, n - sj mod n), s and n - s give the same map, and s = 1 is
   the identity, so there are phi(n)/2 distinct maps, indexed here with 0 the
   identity.  The tables scatter each byte of the code to its image, so a
   decimation is ceil(H/8) lookups.  Using the map for s or for its inverse
   gives the same orbit, which is all the canonical form needs. */
static int NDEC;
static int *DECMAP;              /* [NDEC][H] */
static uint32_t *DBYTE;          /* [NDEC][DNB][256] */
static int DNB;
static int gcd_int(int a, int b) { while (b) { int q = a % b; a = b; b = q; } return a; }
static void dec_init(void) {
    int cnt = 0;
    for (int s = 1; s <= H; s++) if (gcd_int(s, N) == 1) cnt++;
    NDEC = cnt;
    DECMAP = malloc(sizeof(int) * (size_t)cnt * H);
    int i = 0;
    for (int s = 1; s <= H; s++) {
        if (gcd_int(s, N) != 1) continue;
        for (int j = 1; j <= H; j++) {
            int m = (int)(((long long)s * j) % N);
            if (m > H) m = N - m;
            DECMAP[(size_t)i * H + j - 1] = m - 1;
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
                        acc |= 1u << DECMAP[(size_t)d * H + pos];
                }
                DBYTE[(((size_t)d * DNB) + b) * 256 + v] = acc;
            }
}
static inline uint32_t dec_code(uint32_t code, int d) {
    const uint32_t *t = DBYTE + (size_t)d * DNB * 256;
    uint32_t r = 0;
    for (int b = 0; b < DNB; b++) r |= t[(size_t)b * 256 + ((code >> (8 * b)) & 0xffu)];
    return r;
}
/* Canonical A: the least code in its decimation orbit.  Every class has a
   member whose A row is canonical, since decimating all four rows preserves
   the Williamson property, symmetry, x_0 and the row sums. */
static inline int a_is_canonical(uint32_t code) {
    for (int d = 1; d < NDEC; d++)
        if (dec_code(code, d) < code) return 0;
    return 1;
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
/* Gosper's sequence of k-subsets of H bits is increasing, which is colex
   order on the positions, so a chunk can be entered at any rank. */
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
    uint32_t t = v | (v - 1);          /* ~t & (t + 1) is the lowest clear bit of t */
    return (t + 1) | (((~t & (t + 1)) - 1) >> (__builtin_ctz(v) + 1));
}

/* ------------------------------------------------------------ row sums

   A symmetric row with x_0 = +1 has row sum 1 + 4p - 2H for p of its H free
   positions +1, so the sum is congruent to n mod 4 and p = (n + s - 2) / 4.
   For an odd |s| exactly one of +|s|, -|s| is congruent to n mod 4, so the
   positive decomposition fixes the sign of every row and its popcount. */
static int signed_sum(int s) {           /* the sign that x_0 = +1 forces */
    return (((s - N) % 4) + 4) % 4 == 0 ? s : -s;
}
static int popcount_for(int s) {         /* p with 1 + 4p - 2H == s, or -1 */
    int num = s - 1 + 2 * H;
    if (num < 0 || (num & 3)) return -1;
    int p = num / 4;
    return (p >= 0 && p <= H) ? p : -1;
}

typedef struct { int s[4]; int p[4]; } Decomp;      /* s[0] >= ... >= s[3] > 0 */
static int decomps(Decomp *out, int maxout) {
    int nd = 0, fourn = 4 * N;
    int lim = (int)(sqrt((double)fourn) + 1e-9);
    for (int a = 1; a <= lim; a += 2)
        for (int b = 1; b <= a; b += 2)
            for (int c = 1; c <= b; c += 2)
                for (int d = 1; d <= c; d += 2) {
                    if (a*a + b*b + c*c + d*d != fourn) continue;
                    if (nd >= maxout) continue;
                    Decomp *D = &out[nd];
                    D->s[0] = a; D->s[1] = b; D->s[2] = c; D->s[3] = d;
                    int ok = 1;
                    for (int u = 0; u < 4; u++) {
                        D->p[u] = popcount_for(signed_sum(D->s[u]));
                        if (D->p[u] < 0) ok = 0;
                    }
                    if (ok) nd++;
                }
    return nd;
}

/* --------------------------------------------------------- the B table

   Symmetric codes of one popcount that pass the PSD ceiling, in ascending
   code order (Gosper order, which the a = b restriction relies on), with the
   per-row halves of the solve and an index of H bitmaps by quantised PSD
   level so the pair filter for one A is an intersection of H bitmaps. */
/* Levels are floors, so a coarser grid only admits more pairs to the exact
   solve; it never loses one.  Runtime so the index can be shrunk at the orders
   where 256 levels over a multi-million-row table would not fit (WIL_NLEV). */
static int NLEV = 256;
/* WIL_SAMPLE=k: scan every A row for canonicity and the ceiling, but run the
   pair filter and the solve on every k-th chunk of A codes only.  The run then
   reports the sampled pair count and the solve thread-time per pair, which is
   what a cost projection for an order that is not going to be run needs. */
static int SAMPLE = 1;
static double tcpun(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#ifndef ACHUNK
#define ACHUNK 4096
#endif
#ifndef NW_CHUNK
#define NW_CHUNK 24
#endif
static int lev_of(double x, int fourn) {
    int v = (int)(x * NLEV / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > NLEV - 1) v = NLEV - 1;
    return v;
}

typedef struct {
    int n;                 /* rows kept */
    uint32_t *code;        /* ascending */
    int8_t   *paf;         /* [n][H] */
    int8_t   *su;          /* [n][H] + slack, u_k = (n - PAF(k)) / 4 */
    uint32_t *rb;          /* T(parity bits of u) */
    int cwords;
    uint64_t *bmp;         /* [H][NLEV][cwords], prefix-ored over levels */
    uint8_t  *lv8;         /* [n][H] 256-level PSD, sampling runs only */
    long long ncomb;
} Table;
static int lev256(double x, int fourn) {
    int v = (int)(x * 256 / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return v;
}

static void table_build(Table *T, int pb, int fourn) {
    long long ncomb = BINOM[H][pb];
    T->ncomb = ncomb;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    uint32_t **lc = calloc(nth, sizeof(uint32_t *));
    long long *ln = calloc(nth, sizeof(long long));
    /* pass 1: which codes pass the ceiling */
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
    T->su  = malloc((size_t)(total + 1) * H + 64);
    T->rb  = malloc(sizeof(uint32_t) * (size_t)(total + 1));
    T->cwords = (int)((total + 63) >> 6);
    size_t bmpn = (size_t)H * NLEV * T->cwords;
    T->bmp = calloc(bmpn, sizeof(uint64_t));
    T->lv8 = (SAMPLE > 1) ? malloc((size_t)(total + 1) * H) : NULL;
    if (!T->paf || !T->su || !T->rb || !T->bmp || (SAMPLE > 1 && !T->lv8)) {
        fprintf(stderr, "out of memory for the B table (%lld rows, index %.1f GB)\n",
                total, bmpn * 8.0 / (1024.0 * 1024.0 * 1024.0));
        exit(1);
    }
    /* pass 2: per-row data and the level index, one bitmap word per block of
       64 rows so the ORs never share a word across threads */
#pragma omp parallel for schedule(dynamic, 4)
    for (int w = 0; w < T->cwords; w++) {
        int8_t paf[64]; float psd[64];
        int hi = (w + 1) * 64; if (hi > T->n) hi = T->n;
        for (int bi = w * 64; bi < hi; bi++) {
            paf_mask(sym_mask(T->code[bi]), paf);
            psd_all(paf, psd);
            memcpy(T->paf + (size_t)bi * H, paf, H);
            uint32_t rb = 0;
            for (int k = 1; k <= H; k++) {
                int u = (N - (int)paf[k - 1]) >> 2;
                T->su[(size_t)bi * H + k - 1] = (int8_t)u;
                if (u & 1) rb |= 1u << (k - 1);
            }
            T->rb[bi] = nw_tmap(rb);
            for (int k = 0; k < H; k++) {
                int lv = lev_of(psd[k], fourn);
                T->bmp[(((size_t)k * NLEV) + lv) * T->cwords + w] |= 1ULL << (bi & 63);
                if (T->lv8) T->lv8[(size_t)bi * H + k] = (uint8_t)lev256(psd[k], fourn);
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
    free(T->code); free(T->paf); free(T->su); free(T->rb); free(T->bmp); free(T->lv8);
    memset(T, 0, sizeof *T);
}
static int lower_bound_u32(const uint32_t *a, int n, uint32_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
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
static void sort4(uint32_t *r) {
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3 - a; b++)
            if (r[b] > r[b + 1]) { uint32_t t = r[b]; r[b] = r[b + 1]; r[b + 1] = t; }
}
static int cmp_quad(const void *x, const void *y) {
    return memcmp(((const Quad *)x)->c, ((const Quad *)y)->c, sizeof(uint32_t) * 4);
}
/* classes.py's equivalence: decimate all four together, x_0 = +1 (already),
   rows unordered.  No rotation, no reversal. */
static void canon_class(const uint32_t *q, uint32_t *out) {
    uint32_t best[4]; int have = 0;
    for (int d = 0; d < NDEC; d++) {
        uint32_t r[4];
        for (int u = 0; u < 4; u++) r[u] = dec_code(q[u], d);
        sort4(r);
        if (!have || memcmp(r, best, sizeof best) < 0) { memcpy(best, r, sizeof best); have = 1; }
    }
    memcpy(out, best, sizeof best);
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: wil_hkt <n> [threads] [out]\n"); return 1; }
    N = atoi(argv[1]);
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    const char *outpath = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    if (N < 3 || (N % 2) == 0) { fprintf(stderr, "n must be odd and >= 3\n"); return 1; }
    H = (N - 1) / 2;
    /* 64-bit row masks and the 32-bit solve: hkt_build refuses H > 32 on its
       own, and (1ULL << N) - 1 needs N < 64. */
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
            fprintf(stderr, "n=%d: FIXED_[%d]=%d breaks the mod-4 invariant\n",
                    N, k, FIXED_[k]);
            return 1;
        }
    dec_init();
    int fourn = 4 * N;
    printf("n=%d: 4n = %d, H = %d, %d folded decimations\n", N, fourn, H, NDEC);

    /* decompositions, grouped by the two enumerated sums */
    Decomp dec[256];
    int nd = decomps(dec, 256);
    int i0 = 0, i1 = 1;
    {   const char *ab = getenv("WIL_AB");
        if (ab && strlen(ab) == 2 && ab[0] >= '0' && ab[0] <= '3' && ab[1] >= '0' && ab[1] <= '3' && ab[0] < ab[1]) {
            i0 = ab[0] - '0'; i1 = ab[1] - '0';
        }
    }
    int o0 = -1, o1 = -1;
    for (int u = 0; u < 4; u++) if (u != i0 && u != i1) { if (o0 < 0) o0 = u; else o1 = u; }
    /* groups: distinct (s[i0], s[i1]) */
    typedef struct { int sa, sb, pa, pb; int nmem; int mem[16]; } Group;
    Group grp[256]; int ng = 0;
    for (int i = 0; i < nd; i++) {
        int g;
        for (g = 0; g < ng; g++)
            if (grp[g].sa == dec[i].s[i0] && grp[g].sb == dec[i].s[i1]) break;
        if (g == ng) {
            grp[g].sa = dec[i].s[i0]; grp[g].sb = dec[i].s[i1];
            grp[g].pa = dec[i].p[i0]; grp[g].pb = dec[i].p[i1];
            grp[g].nmem = 0; ng++;
        }
        if (grp[g].nmem < 16) grp[g].mem[grp[g].nmem++] = i;
    }
    printf("   %d decompositions a >= b >= c >= d, %d (a, b) groups (enumerating positions %d and %d)\n",
           nd, ng, i0, i1);
    for (int g = 0; g < ng; g++) {
        printf("   group %d: (%d, %d) with", g, grp[g].sa, grp[g].sb);
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            printf(" (%d, %d)", D->s[o0], D->s[o1]);
        }
        printf("   popcounts %d %d\n", grp[g].pa, grp[g].pb);
    }
    if (getenv("WIL_LIST")) return 0;
    int donly = -1;
    { const char *dv = getenv("WIL_DONLY"); if (dv) donly = atoi(dv); }
    { const char *v = getenv("WIL_NLEV"); if (v) { NLEV = atoi(v); if (NLEV < 2) NLEV = 2; if (NLEV > 256) NLEV = 256; } }
    { const char *v = getenv("WIL_SAMPLE"); if (v) { SAMPLE = atoi(v); if (SAMPLE < 1) SAMPLE = 1; } }
    if (SAMPLE > 1) printf("   sampling: pair filter and solve on every %d-th chunk of %d A codes\n", SAMPLE, ACHUNK);
    if (NLEV != 256) printf("   B index quantised to %d levels\n", NLEV);

    /* the solve path */
    int use_cache = (H <= 24);
    { const char *cv = getenv("WIL_NOCACHE"); if (cv) use_cache = !atoi(cv); }
#if !defined(__AVX2__)
    if (!use_cache) { fprintf(stderr, "the cache-free path needs AVX2; using the table\n"); use_cache = 1; }
#endif
    double t0 = wnow();
    if (use_cache) {
        s2_layout();
        double tb = wnow();
        s2_build();
        printf("   stage-2 table: 2^%d patterns, %.0f MB hot + %.0f MB cold, %lld nullity-capped, built in %.1fs\n",
               H, (double)SHOT * (double)((size_t)1 << H) / 1048576.0,
               (double)SCOLD * (double)((size_t)1 << H) / 1048576.0,
               S2_NOTOK, wnow() - tb);
    } else {
        printf("   cache-free solve (eight-lane elimination per survivor chunk)\n");
    }

    const uint32_t maskH = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    long long tot_pairs = 0, tot_raw = 0, tot_filt = 0, tot_found = 0, tot_capped = 0, tot_A = 0;
    double max_index_mb = 0;

    for (int g = 0; g < ng; g++) {
        if (donly >= 0 && g != donly) continue;
        double tg = wnow();
        int pa = grp[g].pa, pb = grp[g].pb;
        int equal_ab = (grp[g].sa == grp[g].sb);
        memset(ALLOWD, 0, sizeof ALLOWD);
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            ALLOWD[D->p[o0]] |= 1u << D->p[o1];
            ALLOWD[D->p[o1]] |= 1u << D->p[o0];
        }
        Table T;
        table_build(&T, pb, fourn);
        double idx_mb = (double)H * NLEV * T.cwords * 8.0 / 1048576.0;
        if (idx_mb > max_index_mb) max_index_mb = idx_mb;
        long long ncombA = BINOM[H][pa];
        printf("   group %d (%d, %d): B table %d of %lld codes (%.1f%%), index %.0f MB; A over %lld codes%s\n",
               g, grp[g].sa, grp[g].sb, T.n, T.ncomb, 100.0 * T.n / (double)T.ncomb, idx_mb,
               ncombA, equal_ab ? ", B >= A" : "");
        fflush(stdout);
        if (!T.n) { table_free(&T); continue; }

        long long nA = 0, pairs = 0, raw = 0, filt = 0, found = 0, capped = 0;
        long long nAs = 0, pairs256 = 0; double tsolve = 0, tscan = 0;
#pragma omp parallel
        {
            int8_t pafa[64]; float psda[64];
            uint32_t cand[HKT_MAXC];
            uint32_t tb[NW_CHUNK]; int ta[64];
            int *surv = malloc(sizeof(int) * (size_t)T.n);
            long long lA = 0, lpairs = 0, lraw = 0, lfilt = 0, lfound = 0, lcapped = 0;
            long long lAs = 0, lpairs256 = 0; double lsolve = 0, lscan = 0;
#pragma omp for schedule(dynamic, 1)
            for (long long c0 = 0; c0 < ncombA; c0 += ACHUNK) {
              long long chi = c0 + ACHUNK; if (chi > ncombA) chi = ncombA;
              int solve_this = ((c0 / ACHUNK) % SAMPLE) == 0;
              uint32_t code = unrank_colex(c0, pa);
              double tc0 = tcpun();
              for (long long ci = c0; ci < chi; ci++) {
                uint32_t am = code;
                if (ci + 1 < chi) code = gosper_next(code);
                if (!a_is_canonical(am)) continue;
                paf_mask(sym_mask(am), pafa);
                if (psd_all(pafa, psda) > fourn + 1e-6) continue;
                lA++;
                if (!solve_this) continue;
                lAs++;
                double ts0 = tcpun(); lscan += ts0 - tc0;
                uint32_t ra; nw_arow(pafa, ta, &ra);
                /* pair filter: rows whose level at every frequency is at most
                   the level of 4n - PSD_A; for a = b only rows at or above A */
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
                    int lo = equal_ab ? lower_bound_u32(T.code, T.n, am) : 0;
                    int w0 = lo >> 6;
                    if (!dead) {
                        for (int w = w0; w < T.cwords; w++) {
                            uint64_t v = ~0ULL;
                            for (int i = 0; i < nbp; i++) v &= bp[i][w];
                            if (w == w0 && (lo & 63)) v &= ~0ULL << (lo & 63);
                            if (w == T.cwords - 1 && (T.n & 63))
                                v &= (1ULL << (T.n & 63)) - 1;
                            while (v) {
                                int b = __builtin_ctzll(v);
                                v &= v - 1;
                                surv[nsurv++] = (w << 6) + b;
                            }
                        }
                    }
                }
                lpairs += nsurv;
                if (T.lv8) {
                    /* how many survivors the 256-level grid would have kept */
                    uint8_t capl[64];
                    for (int k = 0; k < H; k++) capl[k] = (uint8_t)lev256((double)fourn - psda[k], fourn);
                    for (int i = 0; i < nsurv; i++) {
                        const uint8_t *l = T.lv8 + (size_t)surv[i] * H;
                        int ok = 1;
                        for (int k = 0; k < H && ok; k++) if (l[k] > capl[k]) ok = 0;
                        lpairs256 += ok;
                    }
                }

#define WIL_EMIT(NCD, CANDS, SBITS, BIDX)                                      \
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

                for (int s0i = 0; s0i < nsurv; s0i += NW_CHUNK) {
                    int mch = nsurv - s0i;
                    if (mch > NW_CHUNK) mch = NW_CHUNK;
                    for (int u = 0; u < mch; u++) {
                        uint32_t t = ra ^ T.rb[surv[s0i + u]];
                        tb[u] = t;
                        if (use_cache) __builtin_prefetch(S2HOT + (size_t)t * SHOT);
                    }
                    if (use_cache) {
                        for (int u = 0; u < mch; u++) {
                            uint32_t s0 = tb[u];
                            int b0i = surv[s0i + u];
                            int n0 = wil_solve_at(S2HOT + (size_t)s0 * SHOT, ta,
                                                  T.su + (size_t)b0i * H, cand, s0,
                                                  maskH, &lraw, &lcapped);
                            WIL_EMIT(n0, cand, s0, b0i);
                        }
                    }
#if defined(__AVX2__)
                    else {
                        /* eight survivors per elimination, columns transposed
                           (nw_c.c finding 457, gmat_c.c finding 454) */
                        for (int u0 = 0; u0 < mch; u0 += 8) {
                            int nbu = mch - u0; if (nbu > 8) nbu = 8;
                            uint32_t bcand[8][HKT_MAXC]; int bn[8];
                            uint32_t ct[64 * 8], bbv[8], sv[8];
                            for (int j = 0; j < nbu; j++) {
                                uint32_t c0[64]; int16_t mcb[64];
                                uint32_t s0 = tb[u0 + j];
                                nw_rows_from_t(s0, mcb, c0);
                                uint32_t b0 = nw_bvec((const uint8_t *)mcb, ta,
                                                      T.su + (size_t)surv[s0i + u0 + j] * H);
                                for (int kk = 0; kk < H; kk++) ct[(size_t)kk * 8 + j] = c0[kk];
                                bbv[j] = b0; sv[j] = s0; bn[j] = 0;
                            }
                            for (int L = nbu; L < 8; L++) {
                                bbv[L] = 0; sv[L] = 0;
                                for (int kk = 0; kk < H; kk++) ct[(size_t)kk * 8 + L] = 0;
                            }
                            uint32_t pb8[64 * 8], pm8[8];
                            hkt_elim8v(ct, bbv, pb8, pm8);
                            for (int L = 0; L < nbu; L++) {
                                if (bbv[L] & ~pm8[L]) continue;        /* inconsistent */
                                if (H - __builtin_popcount(pm8[L]) > S2_MAXFREE) { lcapped++; continue; }
                                uint32_t colL[64], pivL[64];
                                for (int kk = 0; kk < H; kk++) {
                                    colL[kk] = ct[(size_t)kk * 8 + L];
                                    pivL[kk] = pb8[(size_t)kk * 8 + L];
                                }
                                int nc = hkt_tail_m(colL, bbv[L], pm8[L], pivL, bcand[L]);
                                lraw += nc;
                                bn[L] = wil_rowfilter(bcand[L], nc, sv[L], maskH);
                            }
                            for (int j = 0; j < nbu; j++)
                                WIL_EMIT(bn[j], bcand[j], sv[j], surv[s0i + u0 + j]);
                        }
                    }
#endif
                }
#undef WIL_EMIT
                tc0 = tcpun(); lsolve += tc0 - ts0;
              }
              lscan += tcpun() - tc0;
            }
            free(surv);
#pragma omp atomic
            nAs += lAs;
#pragma omp atomic
            pairs256 += lpairs256;
#pragma omp atomic
            tsolve += lsolve;
#pragma omp atomic
            tscan += lscan;
#pragma omp atomic
            nA += lA;
#pragma omp atomic
            pairs += lpairs;
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
        printf("   group %d (%d, %d): %lld canonical A, %lld pairs solved, %lld lifts, %lld row-sum matches, %lld quadruples, %lld capped, %.1fs\n",
               g, grp[g].sa, grp[g].sb, nA, pairs, raw, filt, found, capped, wnow() - tg);
        printf("   group %d: thread-seconds scan %.1f, filter+solve %.1f; %.0f ns per pair solved\n",
               g, tscan, tsolve, pairs ? 1e9 * tsolve / (double)pairs : 0.0);
        if (SAMPLE > 1)
            printf("   group %d: sampled %lld of %lld canonical A: %lld pairs (%lld at 256 levels), %.1f pairs per A; projected %.3g pairs, %.3g at 256 levels, solve %.0f thread-s\n",
                   g, nAs, nA, pairs, pairs256, nAs ? (double)pairs / (double)nAs : 0.0,
                   nAs ? (double)pairs * (double)nA / (double)nAs : 0.0,
                   nAs ? (double)pairs256 * (double)nA / (double)nAs : 0.0,
                   nAs ? tsolve * (double)nA / (double)nAs : 0.0);
        fflush(stdout);
        tot_A += nA; tot_pairs += pairs; tot_raw += raw; tot_filt += filt;
        tot_found += found; tot_capped += capped;
    }

    double tsearch = wnow() - t0;
    /* distinct quadruples: the same set of four rows can arrive under more
       than one labelling when sums coincide */
    Quad *keyed = malloc(sizeof(Quad) * (size_t)(NSOL + 1));
    Quad *orig  = malloc(sizeof(Quad) * (size_t)(NSOL + 1));
    long long ndistinct = 0;
    {
        /* sort pairs (key, original) together */
        typedef struct { Quad key, o; } KQ;
        KQ *kq = malloc(sizeof(KQ) * (size_t)(NSOL + 1));
        for (long long i = 0; i < NSOL; i++) {
            kq[i].o = SOL[i]; kq[i].key = SOL[i]; sort4(kq[i].key.c);
        }
        qsort(kq, (size_t)NSOL, sizeof(KQ), cmp_quad);   /* key is first in KQ */
        for (long long i = 0; i < NSOL; i++) {
            if (i && !memcmp(kq[i].key.c, kq[i - 1].key.c, sizeof(uint32_t) * 4)) continue;
            keyed[ndistinct] = kq[i].key; orig[ndistinct] = kq[i].o; ndistinct++;
        }
        free(kq);
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
    printf("   pairs solved %lld, lifts %lld, row-sum matches %lld, nullity-capped calls %lld\n",
           tot_pairs, tot_raw, tot_filt, tot_capped);
    if (tot_capped)
        printf("   WARNING: %lld solve calls were dropped by the nullity cap; the census is not exhaustive\n",
               tot_capped);
    printf("   largest B index %.0f MB%s\n", max_index_mb,
           use_cache ? "" : " (no stage-2 table)");
    {   /* peak resident set, from the kernel's own accounting */
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
    if (outpath) {
        FILE *f = fopen(outpath, "w");
        if (!f) { perror(outpath); return 1; }
        int8_t x[128];
        for (long long i = 0; i < ndistinct; i++) {
            for (int u = 0; u < 4; u++) {
                if (u) fputc(' ', f);
                sym_row(orig[i].c[u], x);
                for (int j = 0; j < N; j++) fputc(x[j] > 0 ? '+' : '-', f);
            }
            fputc('\n', f);
        }
        fclose(f);
        printf("   solutions -> %s\n", outpath);
    }
    return 0;
}
