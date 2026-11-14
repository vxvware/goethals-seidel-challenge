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
static int NW_DECCANON = 1;   /* quotient A by decimation too (finding 404) */
#define SCALE 16
#define CEIL16 (4 * N * SCALE)

#include "hkt_solve.h"


/* ------------------------------------------------- stage-2 elimination cache

   hkt_prefix builds the stage-2 matrix from the sign pattern t and nothing
   else: the target only ever reaches the right-hand side.  So the elimination
   of that matrix, the pivot layout it produces and the null-space basis
   hkt_tail rebuilds from it are all functions of t and the order, and none of
   them depend on A, B or the decomposition.  They were being recomputed once
   per solve call.

   Measured at n=25: 4,096 distinct sign patterns over 94,660,700 solves, a mean
   reuse of 23,111.  Every one of those repeats redid an H by H GF(2)
   elimination and an H by H basis build.  Precomputing all 2^H of them costs
   one elimination each and turns the hot path into a lookup plus a linear map
   on the right-hand side.

   The right-hand side is transformed by the same operation sequence as the
   matrix, so that map is linear and is stored as its images of the unit
   vectors. */
/* The stage-2 cache, split by how often each field is read.

   Every solve call reads mc, and 69% of them then die on the consistency
   check, which needs only chk, nchk and ok.  Those four fields are the hot
   half, and at n=37 they are 62 bytes, so one 64-byte line answers two calls
   in three.  pt and basis are read only by the 31% that survive, and live in
   a second array.

   Both are laid out at a stride computed from H rather than from the largest
   H the program accepts.  The single struct this replaces was 260 bytes
   whatever the order, which at n=37 is a 68 MB table entered at random, five
   lines touched per call; the same table is now 17 MB hot plus 26 MB cold,
   with one line touched on the common path. */
static uint8_t *S2HOT, *S2COLD;
static size_t SHOT, SCOLD;
static size_t OFF_CHK, OFF_NCHK, OFF_OK, OFF_BASIS, OFF_NF;
#define S2_MAXFREE 6            /* the nullity cap; also bounds nchk */

static size_t CHKW;                /* bytes per consistency mask */
static void s2_layout(void) {
    /* The masks are H bits, so storing them in four bytes each wastes room
       that decides whether the hot half fits one cache line.  At H=19,
       2H + 6*4 + 2 is 66 and the stride rounds to 128, which measures 6%
       slower than 64 (finding 411); at ceil(H/8) bytes each it is 58 and the
       stride stays 64 through H=22.  They are read with an unaligned 4-byte
       load and masked, so the layout needs four bytes of slack past the last
       one. */
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
    s2_layout();
#ifdef NW_NOCACHE
    /* The ablation build never reads the table, and at H=27 allocating it is
       25 GB, which the kernel answers with SIGKILL.  Skip it: that is what
       makes the large orders reachable at all. */
    S2HOT = NULL; S2COLD = NULL;
    return;
#endif
    S2HOT  = aligned_alloc(64, nt * SHOT);
    S2COLD = malloc(nt * SCOLD);
    if (!S2HOT || !S2COLD) {
        fprintf(stderr, "out of memory for the stage-2 cache\n"); exit(1);
    }
    uint64_t maskn = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    uint32_t hmask = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
    for (size_t t = 0; t < nt; t++) {
        uint8_t *hot = S2HOT + t * SHOT, *cold = S2COLD + t * SCOLD;
        int16_t  *mc    = (int16_t *)hot;
        uint8_t  *chk   = hot + OFF_CHK;
        int8_t   *nchkp = (int8_t *)(hot + OFF_NCHK);
        int8_t   *okp   = (int8_t *)(hot + OFF_OK);
        uint32_t *pt    = (uint32_t *)cold;
        uint32_t *basis = (uint32_t *)(cold + OFF_BASIS);
        int8_t   *nfp   = (int8_t *)(cold + OFF_NF);
        /* rebuild the matrix exactly as hkt_prefix does */
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
            /* Findings 442 and 447.  The c2 term cancelled against itself
               here in the source already; the two that remained are one,
               because kept & FIXM_[k] is a subset of kept. */
            mc[k - 1] = (int16_t)(__builtin_popcountll(kept & NFIXM_[k]) / 2);
        }
        uint32_t col[64];
        hkt_tr(a, H, col);
        /* eliminate, tracking what the same operations do to the unit vectors */
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
        /* The nullity decides admissibility, and it also bounds the number of
           non-pivot rows, so it is settled before chk is written: the array
           holds S2_MAXFREE of them and a rejected pattern can have more. */
        int nul = H - rank;
        if (nul > S2_MAXFREE || (1 << nul) > HKT_MAXC) {
            *okp = 0; *nchkp = 0; *nfp = 0; continue;
        }
        {   int nc = 0;
            for (int q = 0; q < H && nc < S2_MAXFREE; q++) {
                if ((pv >> q) & 1u) continue;              /* pivot row */
                uint32_t colq = 0;
                for (int j = 0; j < H; j++) if ((bt[j] >> q) & 1u) colq |= 1u << j;
                memcpy(chk + (size_t)nc * CHKW, &colq, CHKW);
                nc++;
            }
            *nchkp = (int8_t)nc;
        }
        /* Composite of the RHS transform and the pivot-bit read; see 388. */
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
}

/* Target vector and sign pattern only.  hkt_prefix also builds the stage-2
   matrix and transposes it, which is precisely the work the table already
   holds, so the hot path stops before that: it needs the target, the pattern t,
   and then bvec, which the stored mc[] reduces to one subtraction per lag. */
/* The prefix splits into a part that depends only on A and a part that
   depends only on B, so the per-pair work is one XOR (finding 393).

   PAF(k) is congruent to n mod 4 for every lag of every (-1,1) circulant row
   (finding 392), so PAF_B(k) = n - 4 u_k with u_k a non-negative integer.
   Writing A_k = -PAF_A(k) - n - 2 FIXED_[k], which depends only on A, the
   prefix's working value is

       v = -(PAF_A(k) + PAF_B(k)) - 2 FIXED_[k] = A_k + 4 u_k

   and A_k is even.  Two consequences.  The bit that goes into rhs1 is bit 1 of
   (v >> 1) - 2 M1CONST_[k] = X_k + 2 u_k with X_k = (A_k >> 1) - 2 M1CONST_[k];
   adding 2u cannot carry into bit 1 from below, so that bit is just
   bit1(X_k) XOR bit0(u_k), one term per side.  Hence

       rhs1 = RA XOR RB[b]

   with RA built once per A row and RB once per B row.  And
   v >> 2 = (A_k >> 2) + u_k, so the target the bvec loop wants is TA[k] + u_k
   and never has to be materialised at all.

   What this removes from every one of the 568 million calls at n=27 is the
   whole first vector loop: two autocorrelation loads, two constant loads, the
   arithmetic, a movemask and a store, replaced by one XOR. */
/* rhs1 -> t is a GF(2) matrix-vector product, sliced into byte tables, so it
   is linear and T(0) = 0.  Finding 393 made rhs1 = RA ^ RB, so
   t = T(RA) ^ T(RB) and both halves can be transformed once per row instead of
   the byte lookups running once per pair (finding 397).  main() checks the
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

/* Second half of the prefix: it needs the cached row, so it runs only once the
   caller has had the chance to prefetch it (finding 391).  The target it
   subtracts is TA[k] + u_k, the two halves of finding 393, never materialised. */
static inline uint32_t nw_bvec(const uint8_t *hot, const int *ta,
                               const int8_t *su) {
    const int16_t *mc = (const int16_t *)hot;
    uint32_t bvec = 0;
    int kb = 1;
#if defined(__AVX2__)
    /* mc widens from int16 and su from int8; the bit that goes into bvec is
       bit 1 of the difference, and a shift left by 30 puts it where
       movemask_ps collects eight lags at once.

       The loop runs ceil(H/8) times rather than floor(H/8), so at H=13 it is
       two vector steps instead of one plus five scalar ones, and at H=18 three
       instead of two plus two.  The lags past H read within the hot line for
       mc, inside a 64-slot array for ta, and into the next row of SU, which is
       allocated with a slack row for the purpose; the bits they produce are
       masked off at the end (finding 395). */
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

static inline int nw_solve_at(const uint8_t *hot, const int *ta, const int8_t *su,
                              uint32_t *cand, uint32_t *sout,
                              int pc, int pdd, uint32_t maskH, long long *raw);

static inline int nw_rowfilter(uint32_t *cand, int nc, uint32_t s,
                               int pc, int pdd, uint32_t maskH) {
    int keep = 0;
    for (int i = 0; i < nc; i++) {
        uint32_t v = cand[i];
        int a = H - __builtin_popcount(v & maskH);
        if (a != pc && a != pdd) continue;
        int b = __builtin_popcount((s ^ v) & maskH);
        if ((a == pc && b == pdd) || (a == pdd && b == pc)) cand[keep++] = v;
    }
    return keep;
}

/* Finding 455.  What the cache-free path actually has to rebuild from t.

   Finding 393 split the sign pattern into a per-A half and a per-B half, so
   t = ra ^ SRB[b], and the survivor loop already computes exactly that into
   tb[] in order to prefetch the cache line.  The cache-free path was then
   throwing it away and calling hkt_prefix, which recomputes the target vector
   from the two autocorrelations and derives t again through the byte tables.
   None of that is needed: t is in hand, and the only things the stage-2 entry
   held that cannot be got from the per-A and per-B halves are mc, the matrix,
   and the elimination.  This rebuilds the first two and lets hkt_elim8v do the
   third, so the whole target-vector loop disappears from the path. */
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
    /* Finding 456.  Two lags interleaved.  The chain kept -> G -> oddu is
       three deep and each lag is independent of the next, so two of them fill
       each other's latency.  Finding 447 measured this as noise inside
       good_c, but there finding 450 had shown the row build hides entirely
       under the target-vector loop's memory stalls; here finding 455 removed
       that loop, so the lag loop is exposed and the question is a different
       one. */
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

static inline int nw_solve(const int8_t *pafA, const int8_t *pafB,
                           uint32_t *cand, uint32_t *sout,
                           int pc, int pdd, uint32_t maskH, long long *raw) {
#ifdef NW_NOCACHE
    /* Ablation: the original per-call elimination, for measuring whether the
       2^H table still pays once it stops fitting the last level cache. */
    {   uint32_t col[64], bb2; int piv[64], rank;
        if (!hkt_prefix(pafA, pafB, sout, col, &bb2)) return 0;
        uint32_t p = 0;
        for (int c = 0; c < H; c++) piv[c] = -1;
        rank = 0;
        for (int c = 0; c < H; c++) {
            uint32_t m = col[c] & ~p;
            if (!m) continue;
            int q = __builtin_ctz(m);
            p |= 1u << q; piv[c] = q; rank++;
            uint32_t R = col[c] & ~(1u << q);
            for (int c2 = 0; c2 < H; c2++)
                col[c2] ^= R & (uint32_t) - (int32_t)((col[c2] >> q) & 1u);
            bb2 ^= R & (uint32_t) - (int32_t)((bb2 >> q) & 1u);
        }
        {   int nc = hkt_tail(col, bb2, rank, piv, p, cand);
            *raw += nc;
            return nw_rowfilter(cand, nc, *sout, pc, pdd, maskH); }
    }
#endif
    int ta[64]; int8_t su[64]; uint32_t ra;
    nw_arow(pafA, ta, &ra);
    uint32_t rb = 0;
    for (int k = 1; k <= H; k++) {
        su[k - 1] = (int8_t)((N - (int)pafB[k - 1]) >> 2);
        if (su[k - 1] & 1) rb |= 1u << (k - 1);
    }
    uint32_t t = ra ^ nw_tmap(rb);
    *sout = t;
    return nw_solve_at(S2HOT + (size_t)t * SHOT, ta, su, cand, sout,
                       pc, pdd, maskH, raw);
}

/* The half of the solve that reads the cached row, taking the row's address
   from the caller so it can have been prefetched. */
static inline int nw_solve_at(const uint8_t *hot, const int *ta, const int8_t *su,
                              uint32_t *cand, uint32_t *sout,
                              int pc, int pdd, uint32_t maskH, long long *raw) {
    if (!*(const int8_t *)(hot + OFF_OK)) return 0;
    uint32_t bb = nw_bvec(hot, ta, su);
    /* Decide consistency before building r: each non-pivot row is one parity.
       Everything read so far is in the hot line, so the two calls in three
       that stop here never touch the cold array. */
    {   int nchk = *(const int8_t *)(hot + OFF_NCHK);
        const uint8_t *chk = hot + OFF_CHK;
        uint32_t hm = (H >= 32) ? 0xffffffffu : ((1u << H) - 1u);
        for (int i = 0; i < nchk; i++) {
            uint32_t c;
            memcpy(&c, chk + (size_t)i * CHKW, 4);
            if (__builtin_parity(bb & c & hm)) return 0;
        }
    }
    const uint8_t *cold = S2COLD + (size_t)*sout * SCOLD;
    const uint32_t *pt = (const uint32_t *)cold;
    uint32_t part = 0;
    for (uint32_t t = bb; t; t &= t - 1) part ^= pt[__builtin_ctz(t)];
    const uint32_t *basis = (const uint32_t *)(cold + OFF_BASIS);
    int nf = *(const int8_t *)(cold + OFF_NF), ncand = 1 << nf;
    *raw += ncand;
    /* Gray code: one XOR per candidate instead of nf, and the row-sum test
       runs on v before it is stored. */
    uint32_t s0 = *sout, v = part;
    int keep = 0;
    /* When the decomposition gives C and D the same row sum, swapping them is
       a symmetry of the whole problem, so every solution is found twice.  Both
       members of a swapped pair are solutions and the candidate set contains
       every solution, so keeping the one with codeC <= codeD keeps exactly one
       of each and cannot drop a class (finding 406). */
    const int swapsym = (pc == pdd);
    for (int i = 0; ; ) {
        int a = H - __builtin_popcount(v & maskH);
        if (a == pc || a == pdd) {
            uint32_t cD = (s0 ^ v) & maskH;
            int b = __builtin_popcount(cD);
            if ((a == pc && b == pdd) || (a == pdd && b == pc)) {
                if (!swapsym || ((~v) & maskH) <= cD) cand[keep++] = v;
            }
        }
        if (++i >= ncand) break;
        v ^= basis[__builtin_ctz((unsigned)i)];
    }
    return keep;
}

static double GM_SURV, GM_SOLVE, GM_CHUNK;  /* finding-468 split, permanent */
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
    s2_build();
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
    uint64_t amask = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
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
    int8_t   *SU  = malloc(btab * H + 64);   /* slack: see nw_bvec */
    uint32_t *SRB = malloc(btab * sizeof(uint32_t));
    if (!SPAF || !SPSD || !SCODE || !SU || !SRB) {
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
        int a = dec[di].a, b = dec[di].b;
        int pa = (N + a) / 2;                 /* popcount of A */
        /* Only masks with bit N-1 clear can be rotation-canonical: with pa < N
           there is a zero somewhere, and rotating it to the top gives a
           smaller value.  Bit 0 must be set for the same kind of reason: if it
           were clear then rot(m,1) is just m >> 1, which is smaller.  So the
           enumeration runs over the pa-1 free bits in positions 1..N-2, which
           is N/(N-pa) times (N-1)/pa fewer masks than all of them, four to six
           times fewer here (findings 409 and 410). */
        long long ncomb = 1LL << H;      /* skew codes */
        int pb = sym_popcount_for(b);
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
                    for (int k = 1; k <= H; k++) {
                        int u = (N - (int)paf[k - 1]) >> 2;
                        SU[(size_t)nkeep * H + k - 1] = (int8_t)u;
                        if (u & 1) rb |= 1u << (k - 1);
                    }
                    SRB[nkeep] = nw_tmap(rb); }
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
            uint32_t cand[HKT_MAXC], cand1[HKT_MAXC];
            uint32_t tb[NW_CHUNK]; int ta[64];
            int *surv = malloc(sizeof(int) * (size_t)nkeep);
            long long lA = 0, ltested = 0, lfound = 0;
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
                        for (int w = 0; w < cwords; w++) {
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

                /* One call per survivor.  Pairing two eliminations in
                   lockstep (finding 372) is pointless now that the elimination
                   is a table lookup: there is nothing left to overlap. */
                /* Two passes over a chunk of survivors.  The first computes
                   each one's cache index and prefetches its row; the second
                   does the work that reads the row, by which time the line has
                   had a whole chunk's arithmetic to arrive in.  The addresses
                   are random and known well in advance, which is the case
                   software prefetch is for (finding 391). */
                for (int s0i = 0; s0i < nsurv; s0i += NW_CHUNK) {
                    int mch = nsurv - s0i;
                    if (mch > NW_CHUNK) mch = NW_CHUNK;
                    for (int u = 0; u < mch; u++) {
                        uint32_t t = ra ^ SRB[surv[s0i + u]];
                        tb[u] = t;
                        __builtin_prefetch(S2HOT + (size_t)t * SHOT);
                    }
#if defined(NW_NOCACHE) && defined(__AVX2__)
                    /* Finding 454.  The cache-free path is the only one that
                       reaches n = 59 and 61, where the 2^H table would want 37
                       and 75 GB, and it was doing one elimination per pair.
                       Eight survivors go through hkt_elim8v together instead,
                       with the columns transposed, exactly as good_c.c does.
                       Solutions are still emitted in survivor order. */
                    for (int u0 = 0; u0 < mch; u0 += 8) {
                        int nbu = mch - u0; if (nbu > 8) nbu = 8;
                        uint32_t bcand[8][HKT_MAXC], bsv[8]; int bn[8];
                        uint32_t ct[64 * 8], bbv[8], sv[8];
                        int lane_of[8], nl = 0;
                        for (int j = 0; j < nbu; j++) { bn[j] = 0; bsv[j] = 0; }
                        for (int j = 0; j < nbu; j++) {
                            uint32_t c0[64]; int16_t mcb[64];
                            uint32_t s0 = tb[u0 + j];      /* finding 455 */
                            nw_rows_from_t(s0, mcb, c0);
                            uint32_t b0 = nw_bvec((const uint8_t *)mcb, ta,
                                                  SU + (size_t)surv[s0i + u0 + j] * H);
                            for (int k = 0; k < H; k++) ct[(size_t)k * 8 + nl] = c0[k];
                            bbv[nl] = b0; sv[nl] = s0; lane_of[nl] = j; nl++;
                        }
                        if (nl) {
                            uint32_t pb8[64 * 8], pm8[8];
                            for (int L = nl; L < 8; L++) {
                                bbv[L] = 0;
                                for (int k = 0; k < H; k++) ct[(size_t)k * 8 + L] = 0;
                            }
                            hkt_elim8v(ct, bbv, pb8, pm8);
                            for (int L = 0; L < nl; L++) {
                                int j = lane_of[L];
                                bsv[j] = sv[L];
                                /* finding 450: consistency before the de-transpose */
                                if (bbv[L] & ~pm8[L]) { bn[j] = 0; continue; }
                                uint32_t colL[64], pivL[64];
                                for (int k = 0; k < H; k++) {
                                    colL[k] = ct[(size_t)k * 8 + L];
                                    pivL[k] = pb8[(size_t)k * 8 + L];
                                }
                                int nc = hkt_tail_m(colL, bbv[L], pm8[L], pivL,
                                                    bcand[j]);
                                ltested += nc;
                                bn[j] = nw_rowfilter(bcand[j], nc, sv[L],
                                                     pc, pdd, maskH);
                            }
                        }
                        for (int j = 0; j < nbu; j++) {
                            int b0i = surv[s0i + u0 + j];
                            NW_EMIT(bn[j], bcand[j], bsv[j], b0i);
                        }
                    }
#else
                    for (int u = 0; u < mch; u++) {
                        uint32_t s0 = tb[u];
                        int b0i = surv[s0i + u];
#ifdef NW_NOCACHE
                        {   uint32_t s1;
                            int n1 = nw_solve(pafa, SPAF + (size_t)b0i * H,
                                              cand, &s1, pc, pdd, maskH, &ltested);
                            NW_EMIT(n1, cand, s1, b0i);
                            continue; }
#endif
                        int n0 = nw_solve_at(S2HOT + (size_t)s0 * SHOT, ta,
                                             SU + (size_t)b0i * H, cand, &s0,
                                             pc, pdd, maskH, &ltested);
                        NW_EMIT(n0, cand, s0, b0i);
                    }
#endif
                }
                lsolve += tcpun() - _t2;
#undef NW_EMIT
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
        int8_t x[128];
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
