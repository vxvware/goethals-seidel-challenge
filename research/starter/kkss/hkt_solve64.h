/* hkt_solve64.h -- the mod-4 solve of hkt_solve.h with a compile-time word
   width, so the census engines can run past n = 63.

   Two word types run through the solve, and the 63 limit comes from both:

     H-bit words   one bit per free position of a symmetric row, bit k-1 for
                   k in 1..H: the codes, the sign pattern t, rhs1, the
                   columns of the stage-2 matrix, the right-hand side bb, the
                   pivot masks and the candidates.  hkt_solve.h holds these in
                   uint32_t, so H <= 32 (and sets 1u << (k-1) for k up to H,
                   which is 1u << 32 at n = 67: finding 483).
     N-bit words   one bit per position of the unfolded row, 0..N-1: FIXM_,
                   CONSTALL_, CONSTM_, NFIXM_, the unfolded pattern word uw
                   and the row masks of the exact check.  hkt_solve.h holds
                   these in uint64_t, so N <= 63.

   Both arrive together: n = 65 has H = 32 and N = 65, so the N-bit side
   overflows first, and n = 67 has H = 33.  The width is chosen at compile
   time:

     default        hw_t = uint32_t, nw_t = uint64_t, HKT_LANES = 8.  Every
                    routine below reduces to the hkt_solve.h text with the
                    types spelled through typedefs and macros; the inner
                    loops of hkt_elim8v and hkt_tail_m are the same
                    instructions.  hkt_build refuses N > 63 or H > 32.
     -DHKT_WIDE     hw_t = uint64_t, nw_t = unsigned __int128, HKT_LANES = 4.
                    The lane-vectorised elimination runs four 64-bit lanes
                    per AVX2 register instead of eight 32-bit ones (epi64
                    compare, subtract, variable shift and movemask_pd in
                    place of their epi32 forms), so one batch eliminates four
                    systems and costs about the same per batch; per system it
                    is twice the narrow cost, times (H/H_ref)^2 for the
                    larger matrix.  hkt_build refuses N > 127 or H > 64.

   What changes under HKT_WIDE and what does not:
     - unchanged: the mathematics, the stage-1 inversion, the byte-table map
       for t, the closed-form row build, the one-hot pivot elimination, the
       tail and the nullity cap (HKT_MAXC, nullity > 6); the transposed lane
       layout c[k * HKT_LANES + L]; every function name and argument order.
     - changed: hw_t, nw_t and HKT_LANES; the bit reversal hkt_revh works on
       64 bits (bswap64) instead of 32; hkt_tr has no AVX2 form for 64-bit
       rows and takes the scalar loop (it is startup-only in the census
       engines: nw_rows_ref, the left null space, the self-checks); the
       N-bit mask is (nw_t)1 << N minus one up to N = 127; popcount of an
       N-bit word is two popcountll.
     - the callers must size their own arrays from HKT_LANES, use hw_t for
       codes and pattern words and nw_t for unfolded masks, and keep the
       2^H consistency table off above H = 30 (it does not fit anyway).

   The wide path is tested by building the engines with -DHKT_WIDE at small
   orders, where it must give the same class sets as the narrow build; that
   is the only way to exercise it on orders that finish quickly.

   Requires: N, H, and SCALE/CEIL16 from the including file. */
#ifndef HKT_SOLVE64_H
#define HKT_SOLVE64_H

#include <immintrin.h>

/* ------------------------------------------------------------ word types */
#ifdef HKT_WIDE
#define HKT_HBITS 64
#define HKT_NBITS 128
#define HKT_LANES 4
typedef uint64_t hw_t;
typedef unsigned __int128 nw_t;
#define HW_ONE 1ULL
#define hw_popcount(x) __builtin_popcountll((uint64_t)(x))
#define hw_ctz(x)      __builtin_ctzll((uint64_t)(x))
#define hw_parity(x)   __builtin_parityll((uint64_t)(x))
static inline int nw_popcount(nw_t x) {
    return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64));
}
#if defined(__AVX2__)
#define HV_SET1(x)      _mm256_set1_epi64x((long long)(x))
#define HV_SUB          _mm256_sub_epi64
#define HV_CMPEQ        _mm256_cmpeq_epi64
#define HV_SRLV         _mm256_srlv_epi64
#define HV_MOVEMASK(v)  _mm256_movemask_pd(_mm256_castsi256_pd(v))
#endif
#else
#define HKT_HBITS 32
#define HKT_NBITS 64
#define HKT_LANES 8
typedef uint32_t hw_t;
typedef uint64_t nw_t;
#define HW_ONE 1u
#define hw_popcount(x) __builtin_popcount((uint32_t)(x))
#define hw_ctz(x)      __builtin_ctz((uint32_t)(x))
#define hw_parity(x)   __builtin_parity((uint32_t)(x))
#define nw_popcount(x) __builtin_popcountll((uint64_t)(x))
#if defined(__AVX2__)
#define HV_SET1(x)      _mm256_set1_epi32((int)(x))
#define HV_SUB          _mm256_sub_epi32
#define HV_CMPEQ        _mm256_cmpeq_epi32
#define HV_SRLV         _mm256_srlv_epi32
#define HV_MOVEMASK(v)  _mm256_movemask_ps(_mm256_castsi256_ps(v))
#endif
#endif
#define NW_ONE ((nw_t)1)
/* All H bits set, and all N bits set. */
static inline hw_t hkt_hmask(void) {
    return (H >= HKT_HBITS) ? ~(hw_t)0 : ((HW_ONE << H) - HW_ONE);
}
static inline nw_t hkt_nmask(void) {
    return (N >= HKT_NBITS) ? ~(nw_t)0 : ((NW_ONE << N) - NW_ONE);
}

/* ---------------------------------------------------------------------------
   HKT mod-4 linearization (finding 300).  Given the skew row A and one
   symmetric row B, solve for the other two symmetric rows rather than
   enumerating them.  Two stages, both GF(2) in practice:

     stage 1  divide the PAF equations by 2 and use xy = x+y-1 (mod 4) to get
              c_u + d_u = q_u (mod 4), which fixes d_u = +- c_u exactly;
     stage 2  substitute, divide by 4, linearise again, solve for c.

   The matrix of stage 1 depends only on n and the shift, never on A or B, so
   it is inverted once at startup.  Verified against brute force at n=7,9,11,13
   and against every published good matrix at 31,33,35,43,45,57 (finding 300).
   --------------------------------------------------------------------------- */
static int8_t *EU_, *EW_, *ECONST_;
static hw_t *EMASK_;
static int *FIXED_, *M1CONST_;
/* Byte tables for t, finding 328.  See hkt_build. */
static hw_t *TBYTE_ = NULL;   /* [NT][256] */
static int TNB_ = 0;
static hw_t *M1INVCOL_;
static nw_t *CONSTALL_, *FIXM_, *CONSTM_, *NFIXM_;

static int hkt_mvar(int j) { int t = N - j; return j < t ? j : t; }
static inline nw_t hkt_rot(nw_t x, int k) {
    nw_t m = hkt_nmask();
    x &= m; return ((x << k) | (x >> (N - k))) & m;
}

static int hkt_build(void) {
    /* The row accumulators, TBYTE_, the elimination and both tail routines
       hold an H-bit vector in an hw_t and set bit k-1 for k up to H; the
       unfolded masks hold an N-bit vector in an nw_t.  Past either width the
       shift is undefined and x86 reduces the count, so bit H lands on bit 0
       and the output is wrong on every input rather than on some.  good_c hit
       exactly this at n = 67 and produced wrong output for fourteen
       core-hours before the types were read (finding 483).  Refuse rather
       than truncate. */
    if (H > HKT_HBITS || N > HKT_NBITS - 1) {
        fprintf(stderr,
                "hkt_solve64.h: H = %d, n = %d exceed the %d-bit/%d-bit build "
                "(H <= %d, n <= %d); %s\n", H, N, HKT_HBITS, HKT_NBITS,
                HKT_HBITS, HKT_NBITS - 1,
                HKT_HBITS == 32 ? "rebuild with -DHKT_WIDE" : "no build reaches this order");
        exit(1);
    }
    EU_ = malloc((size_t)(H + 1) * H); EW_ = malloc((size_t)(H + 1) * H);
    ECONST_ = malloc((size_t)(H + 1) * H);
    EMASK_ = malloc(sizeof(hw_t) * (size_t)(H + 1) * H);
    FIXED_ = calloc(H + 1, sizeof(int)); M1CONST_ = calloc(H + 1, sizeof(int));
    CONSTALL_ = calloc(H + 1, sizeof(nw_t));
    FIXM_     = calloc(H + 1, sizeof(nw_t));
    CONSTM_   = calloc(H + 1, sizeof(nw_t));

    NFIXM_    = calloc(H + 1, sizeof(nw_t));
    if (!CONSTALL_ || !FIXM_ || !CONSTM_ || !NFIXM_) return 0;
    hw_t *M1ROW = calloc(H + 1, sizeof(hw_t));
    char *seen = malloc((size_t)N * N);
    if (!EU_ || !EW_ || !ECONST_ || !EMASK_ || !FIXED_ || !M1CONST_ || !M1ROW || !seen)
        return 0;
    for (int k = 1; k <= H; k++) {
        memset(seen, 0, (size_t)N * N); int nr = 0, fx = 0;
        for (int j = 0; j < N; j++) {
            int jj = (((N - j - k) % N) + N) % N;
            int a = j < jj ? j : jj, b = j < jj ? jj : j;
            if (seen[a * N + b]) continue;
            seen[a * N + b] = 1;
            if (j == jj) { fx++; FIXM_[k] |= NW_ONE << j; CONSTALL_[k] |= NW_ONE << j; continue; }
            int u = hkt_mvar(j), w = hkt_mvar((j + k) % N);
            EU_[(k - 1) * H + nr] = (int8_t)u; EW_[(k - 1) * H + nr] = (int8_t)w;
            hw_t m = 0; int8_t isc = 0;
            if (u == 0 && w == 0) isc = 1;
            else if (u == 0) m = HW_ONE << (w - 1);
            else if (w == 0) m = HW_ONE << (u - 1);
            else if (u == w) isc = 1;
            else m = (HW_ONE << (u - 1)) | (HW_ONE << (w - 1));
            EMASK_[(k - 1) * H + nr] = m; ECONST_[(k - 1) * H + nr] = isc;
            if (isc) { CONSTM_[k] |= NW_ONE << j; CONSTALL_[k] |= NW_ONE << j;
                       CONSTALL_[k] |= NW_ONE << jj; }
            nr++;
        }
        for (int i = nr; i < H; i++) {           /* pad so the row loop is fixed */
            EU_[(k - 1) * H + i] = 0; EW_[(k - 1) * H + i] = 0;
            EMASK_[(k - 1) * H + i] = 0; ECONST_[(k - 1) * H + i] = 0;
        }
        FIXED_[k] = fx;
        NFIXM_[k] = ~FIXM_[k];       /* finding 447 */
        int mult[64]; memset(mult, 0, sizeof mult); int cst = 0;
        for (int i = 0; i < nr; i++) {
            int u = EU_[(k - 1) * H + i], w = EW_[(k - 1) * H + i];
            if (u == 0 && w == 0) cst++;
            else if (u == 0) mult[w]++;
            else if (w == 0) mult[u]++;
            else if (u == w) cst++;
            else { mult[u]++; mult[w]++; cst--; }
        }
        hw_t r = 0;
        for (int u = 1; u <= H; u++) if (mult[u] & 1) r |= HW_ONE << (u - 1);
        M1ROW[k] = r; M1CONST_[k] = cst;
    }
    free(seen);
    /* invert the stage-1 matrix once: it never depends on A or B */
    hw_t aug[64], inv[64];
    for (int k = 0; k < H; k++) { aug[k] = M1ROW[k + 1]; inv[k] = HW_ONE << k; }
    int rank = 0;
    for (int c = 0; c < H; c++) {
        int pr = -1;
        for (int r2 = rank; r2 < H; r2++) if ((aug[r2] >> c) & 1) { pr = r2; break; }
        if (pr < 0) continue;
        hw_t t = aug[pr]; aug[pr] = aug[rank]; aug[rank] = t;
        t = inv[pr]; inv[pr] = inv[rank]; inv[rank] = t;
        for (int r2 = 0; r2 < H; r2++)
            if (r2 != rank && ((aug[r2] >> c) & 1)) { aug[r2] ^= aug[rank]; inv[r2] ^= inv[rank]; }
        rank++;
    }
    free(M1ROW);
    if (rank != H) return 0;               /* stage 1 must be invertible */
    M1INVCOL_ = calloc(H, sizeof(hw_t));
    if (!M1INVCOL_) return 0;
    for (int r2 = 0; r2 < H; r2++)
        for (int c = 0; c < H; c++)
            if ((inv[r2] >> c) & 1) M1INVCOL_[r2] |= HW_ONE << c;
    /* Byte tables for t, finding 328.  t_c is the parity of
       M1INVCOL_[c] AND rhs1, which is a GF(2) matrix-vector product, so t is
       the XOR of those rows of the transpose that rhs1 selects.  Splitting
       rhs1 into bytes and precomputing the XOR for each byte value replaces H
       parity operations with ceil(H/8) loads and as many XORs.  The tables are
       ceil(H/8) by 256 words, three kilobytes at H=21, so they sit in L1. */
    {
        hw_t rowt[64];
        for (int i = 0; i < H; i++) {
            hw_t w = 0;
            for (int c = 0; c < H; c++) if ((M1INVCOL_[c] >> i) & 1) w |= HW_ONE << c;
            rowt[i] = w;
        }
        TNB_ = (H + 7) / 8;
        TBYTE_ = calloc((size_t)TNB_ * 256, sizeof(hw_t));
        if (!TBYTE_) return 0;
        for (int b = 0; b < TNB_; b++)
            for (int v = 1; v < 256; v++) {
                int low = __builtin_ctz((unsigned)v);
                int idx = 8 * b + low;
                hw_t base = TBYTE_[(size_t)b * 256 + (unsigned)(v & (v - 1))];
                TBYTE_[(size_t)b * 256 + v] = base ^ (idx < H ? rowt[idx] : (hw_t)0);
            }
    }
    return 1;
}

/* Emit up to HKT_MAXC candidate c-bit-vectors and the sign pattern.  Returns
   the number of candidates; 0 means this (A, B) pair admits no completion. */
#define HKT_MAXC 64
/* Bit-matrix transpose of an H x H GF(2) matrix, finding 313.  The scalar form
   is H^2 shift-and-test iterations; movemask_ps reads bit 31 of each dword of a
   vector, so pre-shifting the rows left by 32-H puts column H-1 on top and each
   further shift by one walks down a column at a time, eight rows per movemask.
   Rows H..31 of the input must be zero.  Only the row groups that exist are
   touched, which is what makes the small-H case worth separating.  The
   64-bit build has no vector form and takes the scalar loop; the census
   engines call this at startup only. */
static inline void hkt_tr(const hw_t *a, int h, hw_t *col) {
#if defined(__AVX2__) && HKT_HBITS == 32
    int s = 32 - h;
    if (h <= 8) {
        __m256i v0 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)a), s);
        for (int c = h - 1; c >= 0; c--) {
            col[c] = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v0));
            v0 = _mm256_slli_epi32(v0, 1);
        }
        return;
    }
    if (h <= 16) {
        __m256i v0 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 0)), s);
        __m256i v1 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 8)), s);
        for (int c = h - 1; c >= 0; c--) {
            col[c] = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v0))
                   | ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v1)) << 8);
            v0 = _mm256_slli_epi32(v0, 1); v1 = _mm256_slli_epi32(v1, 1);
        }
        return;
    }
    if (h <= 24) {
        __m256i v0 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 0)), s);
        __m256i v1 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 8)), s);
        __m256i v2 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 16)), s);
        for (int c = h - 1; c >= 0; c--) {
            col[c] = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v0))
                   | ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v1)) << 8)
                   | ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v2)) << 16);
            v0 = _mm256_slli_epi32(v0, 1); v1 = _mm256_slli_epi32(v1, 1);
            v2 = _mm256_slli_epi32(v2, 1);
        }
        return;
    }
    if (h <= 32) {
        __m256i v0 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 0)), s);
        __m256i v1 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 8)), s);
        __m256i v2 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 16)), s);
        __m256i v3 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(a + 24)), s);
        for (int c = h - 1; c >= 0; c--) {
            col[c] = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v0))
                   | ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v1)) << 8)
                   | ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v2)) << 16)
                   | ((uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(v3)) << 24);
            v0 = _mm256_slli_epi32(v0, 1); v1 = _mm256_slli_epi32(v1, 1);
            v2 = _mm256_slli_epi32(v2, 1); v3 = _mm256_slli_epi32(v3, 1);
        }
        return;
    }
#endif
    for (int c = 0; c < h; c++) {
        hw_t m = 0;
        for (int r = 0; r < h; r++) if ((a[r] >> c) & 1) m |= HW_ONE << r;
        col[c] = m;
    }
}

/* Reverse the low H bits of an H-bit word (finding 315 uses it to unfold a
   symmetric pattern).  32-bit: swap within bytes, then bswap32 and drop the
   32-H unused bits; 64-bit the same through bswap64. */
static inline hw_t hkt_revh(hw_t x) {
#if HKT_HBITS == 64
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
    return __builtin_bswap64(x) >> (64 - H);
#else
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    return __builtin_bswap32(x) >> (32 - H);
#endif
}

/* Two-lane solve, finding 326.  Finding 325 measured the elimination
   latency-bound and two interleaved eliminations 1.28x to 1.69x, with the
   caller offering 608 independent solves per outer row on average.  The solve
   is therefore split: hkt_prefix does everything up to and including the
   transpose, hkt_elim2 runs two eliminations in lockstep branchless so an
   inactive column contributes R = 0 and its inner loop is a no-op, and
   hkt_tail finishes each lane on its own.  hkt_solve keeps its old signature
   as the one-lane path for the odd survivor. */
static inline int hkt_prefix(const int8_t *pafA, const int8_t *pafB,
                             hw_t *sout, hw_t *col, hw_t *bbout) {
    /* Target vector, finding 323.  Eight lags at a time: the two PAF bytes
       widen with cvtepi8_epi32, FIXED_ and M1CONST_ are int arrays read at
       consecutive k so they load straight, and the bit that goes into rhs1 is
       bit 1 of r, which a shift left by 30 moves to bit 31 where movemask_ps
       collects all eight at once.  The mod-4 reject becomes an OR accumulated
       across the block and tested after it, which finding 321 measured never
       fires at n=43 anyway. */
    int tgt[72]; hw_t rhs1 = 0;
    {
        int k = 1;
#if defined(__AVX2__)
        __m256i vbad = _mm256_setzero_si256();
        const __m256i v3 = _mm256_set1_epi32(3);
        const __m256i vz = _mm256_setzero_si256();
        for (; k + 7 <= H; k += 8) {
            __m256i va = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(pafA + k - 1)));
            __m256i vb = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(pafB + k - 1)));
            __m256i vf = _mm256_loadu_si256((const __m256i *)(FIXED_ + k));
            __m256i vm = _mm256_loadu_si256((const __m256i *)(M1CONST_ + k));
            __m256i v  = _mm256_sub_epi32(vz, _mm256_add_epi32(va, vb));
            v = _mm256_sub_epi32(v, _mm256_slli_epi32(vf, 1));
            vbad = _mm256_or_si256(vbad, _mm256_and_si256(v, v3));
            __m256i vt = _mm256_srai_epi32(v, 1);
            _mm256_storeu_si256((__m256i *)(tgt + k), vt);
            __m256i vr = _mm256_sub_epi32(vt, _mm256_slli_epi32(vm, 1));
            uint32_t mm = (uint32_t)_mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32(vr, 30)));
            rhs1 |= (hw_t)mm << (k - 1);
        }
        if (!_mm256_testz_si256(vbad, vbad)) return 0;
#endif
        for (; k <= H; k++) {
            int v = -((int)pafA[k - 1] + (int)pafB[k - 1]) - 2 * FIXED_[k];
            if (v & 3) return 0;
            tgt[k] = v >> 1;
            int r = tgt[k] - 2 * M1CONST_[k];
            if ((r >> 1) & 1) rhs1 |= HW_ONE << (k - 1);
        }
    }
    hw_t t = TBYTE_[rhs1 & 0xffu];
    for (int b = 1; b < TNB_; b++)
        t ^= TBYTE_[(size_t)b * 256 + ((rhs1 >> (8 * b)) & 0xffu)];
    *sout = t;
    /* Closed-form row build, finding 308.  In the unfolded n-bit space with
       u_j = t_{m(j)}, which is symmetric, a representative at j is kept exactly
       when bit j of ~(u XOR rot(u, k)) is set, the row mask is G XOR rot(G, k)
       for G = kept AND NOT const read at bits 1..H, and the counts are
       popcounts.  Two rotates, two XORs and three popcounts per shift instead
       of H iterations with two byte gathers each, verified identical on all
       three outputs and 13.8x to 20.3x faster as a kernel. */
    nw_t maskn = hkt_nmask();
    /* Closed form for the unfolded word, finding 315.  With n = 2H+1 and
       u_j = t_{min(j, n-j)}, position j in 1..H reads bit j-1 of t and position
       H+1+i in the upper half reads bit H-1-i, which is bit i of t with its low
       H bits reversed.  So the whole word is 1, t shifted up one, and the
       reversal shifted up H+1, which also removes the tb[] byte array that
       existed only to feed this loop. */
    hw_t rv = hkt_revh(t);
    nw_t uw = NW_ONE | ((nw_t)t << 1) | ((nw_t)rv << (H + 1));
    hw_t hmask = hkt_hmask();
    hw_t a[64]; hw_t bvec = 0;
    for (int i = 0; i < 64; i++) a[i] = 0;
    for (int k = 1; k <= H; k++) {
        nw_t kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
        nw_t G    = kept & ~CONSTALL_[k] & maskn;
        nw_t oddu = (G ^ hkt_rot(G, k)) & maskn;
        hw_t odd  = (hw_t)(oddu >> 1) & hmask;
        /* Findings 442 and 447.  The CONSTM_ popcount entered rr twice with
           opposite signs and cancels; of the two that remained, kept & FIXM_[k]
           is a subset of kept, so their difference is one popcount of
           kept & ~FIXM_[k].  Three popcounts per lag become one. */
        int rr = nw_popcount(kept & NFIXM_[k]) / 2 - (tgt[k] >> 1);
        a[k - 1] = odd; if ((rr >> 1) & 1) bvec |= HW_ONE << (k - 1);
    }
    /* Column-major elimination, finding 310.  col[c] is a bitmask over rows,
       so the pivot search is one ctz and eliminating a column is a branchless
       masked XOR across columns, with no row swaps.  The row-major version did
       about 700 L1-resident word operations at 5 ns each, which was the pivot
       scan's data-dependent break and the conditional XOR, not the arithmetic;
       this measured 10.4x faster as a kernel with identical solution sets. */
    hkt_tr(a, H, col);
    *bbout = bvec;
    return 1;
}

/* Two eliminations in lockstep.  Branchless per lane: an inactive column takes
   R = 0, which makes its inner-loop XOR a no-op, so the two chains stay in
   step with no divergence and the machine can overlap one lane's stalls with
   the other's work. */
static inline void hkt_elim2(hw_t *c0, hw_t *bb0, int *rk0, int *pv0,
                             hw_t *c1, hw_t *bb1, int *rk1, int *pv1,
                             hw_t *pm0, hw_t *pm1) {
    hw_t p0 = 0, p1 = 0, b0 = *bb0, b1 = *bb1;
    int r0 = 0, r1 = 0;
    for (int c = 0; c < H; c++) { pv0[c] = -1; pv1[c] = -1; }
    for (int c = 0; c < H; c++) {
        hw_t m0 = c0[c] & ~p0, m1 = c1[c] & ~p1;
        int a0 = m0 != 0, a1 = m1 != 0;
        int q0 = a0 ? hw_ctz(m0) : 0, q1 = a1 ? hw_ctz(m1) : 0;
        hw_t R0 = a0 ? (c0[c] & ~(HW_ONE << q0)) : 0;
        hw_t R1 = a1 ? (c1[c] & ~(HW_ONE << q1)) : 0;
        p0 |= a0 ? (HW_ONE << q0) : (hw_t)0; p1 |= a1 ? (HW_ONE << q1) : (hw_t)0;
        pv0[c] = a0 ? q0 : -1;      pv1[c] = a1 ? q1 : -1;
        r0 += a0; r1 += a1;
        for (int k = 0; k < H; k++) {
            c0[k] ^= R0 & ((hw_t)0 - (hw_t)((c0[k] >> q0) & 1u));
            c1[k] ^= R1 & ((hw_t)0 - (hw_t)((c1[k] >> q1) & 1u));
        }
        b0 ^= R0 & ((hw_t)0 - (hw_t)((b0 >> q0) & 1u));
        b1 ^= R1 & ((hw_t)0 - (hw_t)((b1 >> q1) & 1u));
    }
    *bb0 = b0; *bb1 = b1; *rk0 = r0; *rk1 = r1; *pm0 = p0; *pm1 = p1;
}

/* HKT_LANES eliminations at once, vectorised across lanes rather than across
   columns.  Finding 438 showed the across-columns form is null: H is 21 to 26,
   so the loop is two or three vector iterations plus a tail, and the per-pivot
   setup eats the gain.  Across lanes the vector is always full.

   The columns are stored transposed, c[k * HKT_LANES + lane], so a fixed k
   loads all lanes contiguously.  Each lane pivots at its own column, so the
   shift is per-lane and goes through a variable shift; a lane with no pivot
   left carries R = 0, which makes its update the identity and lets the whole
   batch run branchless in lockstep.

   Lanes beyond nlane are filled with R = 0 the same way, so a partial batch
   costs the same as a full one but computes nothing for the dead lanes. */
#if defined(__AVX2__)
static inline void hkt_elim8(hw_t *c, hw_t *bb, int *rk,
                             int pv[HKT_LANES][64], hw_t *pm, int nlane) {
    hw_t p[HKT_LANES];
    for (int L = 0; L < HKT_LANES; L++) {
        p[L] = 0; rk[L] = 0;
        for (int k = 0; k < H; k++) pv[L][k] = -1;
    }
    for (int cc = 0; cc < H; cc++) {
        hw_t Rl[HKT_LANES], ql[HKT_LANES];
        for (int L = 0; L < HKT_LANES; L++) {
            if (L >= nlane) { Rl[L] = 0; ql[L] = 0; continue; }
            hw_t col = c[(size_t)cc * HKT_LANES + L];
            hw_t m = col & ~p[L];
            if (!m) { Rl[L] = 0; ql[L] = 0; continue; }
            int q = hw_ctz(m);
            ql[L] = (hw_t)q;
            Rl[L] = col & ~(HW_ONE << q);
            p[L] |= HW_ONE << q;
            pv[L][cc] = q;
            rk[L]++;
        }
        __m256i vR = _mm256_loadu_si256((const __m256i *)Rl);
        __m256i vq = _mm256_loadu_si256((const __m256i *)ql);
        const __m256i one = HV_SET1(1);
        const __m256i zero = _mm256_setzero_si256();
        for (int k = 0; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * HKT_LANES));
            __m256i bit = _mm256_and_si256(HV_SRLV(v, vq), one);
            v = _mm256_xor_si256(v, _mm256_and_si256(vR, HV_SUB(zero, bit)));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * HKT_LANES), v);
        }
        __m256i vb = _mm256_loadu_si256((const __m256i *)bb);
        __m256i bit = _mm256_and_si256(HV_SRLV(vb, vq), one);
        vb = _mm256_xor_si256(vb, _mm256_and_si256(vR, HV_SUB(zero, bit)));
        _mm256_storeu_si256((__m256i *)bb, vb);
    }
    for (int L = 0; L < HKT_LANES; L++) pm[L] = p[L];
}
#endif

#if defined(__AVX2__)
/* Finding 454.  HKT_LANES eliminations at once, the construction good_c.c
   uses.  Columns are stored transposed, c[k*HKT_LANES+L], so a fixed k loads
   all lanes contiguously; the pivot is a one-hot mask so the whole pivot
   search vectorises, and membership is an AND rather than a shift by a
   per-lane count.  A lane with no pivot carries bit = 0 and updates by the
   identity.  Callers that solve one pair at a time can batch survivors into
   this; the scalar one-hot form on its own is not faster than the ctz
   version, so the vector width is where the gain is. */
static inline void hkt_elim8v(hw_t *c, hw_t *bb, hw_t *pivb, hw_t *pm) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i p = zero;
    __m256i vb = _mm256_loadu_si256((const __m256i *)bb);
    for (int cc = 0; cc < H; cc++) {
        __m256i colv = _mm256_loadu_si256((const __m256i *)(c + (size_t)cc * HKT_LANES));
        __m256i m    = _mm256_andnot_si256(p, colv);
        __m256i bit  = _mm256_and_si256(m, HV_SUB(zero, m));
        __m256i R    = _mm256_xor_si256(colv, bit);
        p = _mm256_or_si256(p, bit);
        _mm256_storeu_si256((__m256i *)(pivb + (size_t)cc * HKT_LANES), bit);
        for (int k = 0; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * HKT_LANES));
            __m256i ns = HV_CMPEQ(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * HKT_LANES), v);
        }
        __m256i nb = HV_CMPEQ(_mm256_and_si256(vb, bit), zero);
        vb = _mm256_xor_si256(vb, _mm256_andnot_si256(nb, R));
    }
    _mm256_storeu_si256((__m256i *)bb, vb);
    _mm256_storeu_si256((__m256i *)pm, p);
}
#endif

/* Finding 453.  One-lane elimination carrying the pivot as a one-hot mask
   instead of an index, which is finding 444's construction reduced to a single
   system.  1 << q is the lowest set bit of col & ~p, so bit = m & -m, the
   pivot row never has to be named, and the membership test becomes an AND
   against the mask rather than a shift by a runtime count.  A column with no
   pivot gets bit = 0, and (v & 0) != 0 is false for every v, so its update is
   the identity: the branch and the ctz both go, and the loop is branchless.

   good_c.c batches eight of these into one AVX2 elimination.  The programs
   that call the solve one pair at a time cannot, but the scalar form is still
   cheaper than the ctz version it replaces. */
static inline void hkt_elim1m(hw_t *col, hw_t *bb, hw_t *pivb, hw_t *pm) {
    hw_t p = 0, b = *bb;
    for (int c = 0; c < H; c++) {
        hw_t m   = col[c] & ~p;
        hw_t bit = m & (~m + HW_ONE);
        hw_t R   = col[c] ^ bit;
        p |= bit;
        pivb[c] = bit;
        for (int c2 = 0; c2 < H; c2++) {
            hw_t v = col[c2];
            col[c2] = v ^ (R & ((hw_t)0 - (hw_t)((v & bit) != 0)));
        }
        b ^= R & ((hw_t)0 - (hw_t)((b & bit) != 0));
    }
    *bb = b; *pm = p;
}

/* Tail on one-hot pivot masks: a column is a pivot column when its mask is
   nonzero, membership is an AND, and the rank is the popcount of the mask. */
static inline int hkt_tail_m(const hw_t *col, hw_t bb, hw_t pivoted,
                             const hw_t *pivb, hw_t *cand) {
    if (bb & ~pivoted) return 0;
    int nul = H - hw_popcount(pivoted);
    if (nul > 6) return 0;
    hw_t part = 0;
    for (int c = 0; c < H; c++)
        if (bb & pivb[c]) part |= HW_ONE << c;
    hw_t basis[64]; int nf = 0;
    for (int c = 0; c < H; c++) {
        if (pivb[c]) continue;
        hw_t w = HW_ONE << c;
        for (int c2 = 0; c2 < H; c2++)
            if (col[c] & pivb[c2]) w |= HW_ONE << c2;
        basis[nf++] = w;
    }
    int ncand = 1 << nf; if (ncand > HKT_MAXC) return 0;
    for (int mask = 0; mask < ncand; mask++) {
        hw_t v = part;
        for (int i = 0; i < nf; i++) if ((mask >> i) & 1) v ^= basis[i];
        cand[mask] = v;
    }
    return ncand;
}

/* Finding 330.  The elimination already knows which rows it pivoted, and
   finding 324 established that the OR of the reduced columns is exactly that
   mask, checked over 560,000 systems.  Finding 326's split threw the mask away
   and had the tail rebuild it from piv_of_col and then recompute the OR, two
   H-iteration loops per call on every call, ahead of the 87% of calls that
   finding 321 measured inconsistent.  Both go by passing the mask through. */
static inline int hkt_tail(hw_t *col, hw_t bb, int rank,
                           const int *piv_of_col, hw_t pivoted,
                           hw_t *cand) {
    if (bb & ~pivoted) return 0;                 /* inconsistent */
    int nul = H - rank;
    if (nul > 6) return 0;
    hw_t part = 0;
    for (int c = 0; c < H; c++) {
        int p = piv_of_col[c];
        if (p >= 0 && ((bb >> p) & 1u)) part |= HW_ONE << c;
    }
    hw_t basis[64]; int nf = 0;
    for (int c = 0; c < H; c++) {
        if (piv_of_col[c] >= 0) continue;
        hw_t w = HW_ONE << c;
        for (int c2 = 0; c2 < H; c2++) {
            int p = piv_of_col[c2];
            if (p >= 0 && ((col[c] >> p) & 1u)) w |= HW_ONE << c2;
        }
        basis[nf++] = w;
    }
    int ncand = 1 << nf; if (ncand > HKT_MAXC) return 0;
    for (int mask = 0; mask < ncand; mask++) {
        hw_t v = part;
        for (int i = 0; i < nf; i++) if ((mask >> i) & 1) v ^= basis[i];
        cand[mask] = v;
    }
    return ncand;
}

static inline int hkt_solve(const int8_t *pafA, const int8_t *pafB,
                            hw_t *cand, hw_t *sout) {
    hw_t col[64], bb; int piv[64], rank;
    if (!hkt_prefix(pafA, pafB, sout, col, &bb)) return 0;
    hw_t p = 0;
    for (int c = 0; c < H; c++) piv[c] = -1;
    rank = 0;
    for (int c = 0; c < H; c++) {
        hw_t m = col[c] & ~p;
        if (!m) continue;
        int q = hw_ctz(m);
        p |= HW_ONE << q; piv[c] = q; rank++;
        hw_t R = col[c] & ~(HW_ONE << q);
        for (int c2 = 0; c2 < H; c2++)
            col[c2] ^= R & ((hw_t)0 - (hw_t)((col[c2] >> q) & 1u));
        bb ^= R & ((hw_t)0 - (hw_t)((bb >> q) & 1u));
    }
    return hkt_tail(col, bb, rank, piv, p, cand);
}

/* Bit-word verification, finding 320.  A plus-minus-one row is stored as an
   N-bit word whose bit j is set when x_j is -1, so x_j x_{j+k} is -1 exactly
   where w and w rotated by k differ and PAF_x(k) = N - 2 popcount(w XOR
   rot(w,k)).  A is skew and B and C and D are symmetric, which only changes
   whether the upper half repeats the lower half or complements it.  The A and B
   PAFs are already in hand as the solve's own arguments, so only C and D are
   computed here. */
static inline nw_t hkt_symword(hw_t code) {
    hw_t hm = hkt_hmask();
    hw_t c = code & hm, nc = (~c) & hm;
    return ((nw_t)nc << 1) | ((nw_t)hkt_revh(nc) << (H + 1));
}
static inline nw_t hkt_rotk(nw_t w, int k) {
    nw_t m = hkt_nmask();
    return ((w >> k) | (w << (N - k))) & m;
}

static void paf_of(const int8_t *x, int8_t *paf) {
    for (int s = 1; s <= H; s++) {
        int acc = 0;
        for (int j = 0; j < N; j++) acc += x[j] * x[(j + s) % N];
        paf[s - 1] = (int8_t)acc;
    }
}

#endif
