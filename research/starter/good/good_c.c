/*
 * good_c.c -- exhaustive census of circulant good matrices of odd order n.
 *
 * Good matrices are four circulant {-1,+1} matrices A, B, C, D of order n with
 * A skew (a_0 = 1 and a_j = -a_{n-j}) and B, C, D symmetric (x_j = x_{n-j}),
 * satisfying A A^T + B^2 + C^2 + D^2 = 4n I.  Seberry's construction turns a
 * set of them into a skew Hadamard matrix of order 4n.
 *
 * Equivalently, with PAF_x(s) = sum_j x_j x_{j+s} the periodic autocorrelation
 * and PSD_x(k) = |sum_j x_j w^{jk}|^2 for w = exp(2 pi i / n):
 *
 *     PAF_A(s) + PAF_B(s) + PAF_C(s) + PAF_D(s) = 0     for s != 0,
 *     PSD_A(k) + PSD_B(k) + PSD_C(k) + PSD_D(k) = 4n    for every k.
 *
 * The row sums satisfy sum(A) = 1 and sum(B)^2 + sum(C)^2 + sum(D)^2 = 4n - 1,
 * with every row sum congruent to n modulo 4.  That three-square condition
 * admits several decompositions rather than pinning one weight, which is the
 * structural difference from best matrices and the reason the outer loop here
 * is large.
 *
 * Structure, following census.c: build tables of surviving candidates once,
 * quantise their spectra into levels, and for each pair of an outer and a skew
 * row reach the admissible partners by intersecting one bitmap per frequency
 * rather than scanning, then find the fourth sequence by an exact hash on its
 * autocorrelation.  The skew rows are cut down to one per decimation orbit and
 * the solutions expanded again at the end.
 *
 *   build: gcc -O3 -march=native -fopenmp -o good_c good_c.c -lm
 *   run:   ./good_c <n> [threads] [out]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static int N, H;                    /* order, (n-1)/2 */
/* Spectra are carried as int16 scaled by SCALE, as census.c does.  PSD is at
   most 4n, which is 212 at n=53, so 16ths fit an int16 with room.  Each value
   is floored, so a sum of floors is at most the floor of the sum: the integer
   filter is a superset of the exact one, never dropping a candidate the exact
   test would keep.  The hash on the autocorrelation is exact, so a looser
   filter costs work and cannot cost solutions. */
#define SCALE 16
#define CEIL16 (4 * N * SCALE)
#ifndef NLEV_DEFAULT
#define NLEV_DEFAULT 64         /* default quantisation steps for the bitmaps */
#endif
/* Selected at startup rather than fixed: the bitmaps are linear in the level
   count, so the best value depends on whether the chain still fits the last
   level cache.  N is already a runtime value, so LEVEL() divides at runtime
   either way, and the only cost of making this a variable is a multiply in the
   bitmap index, outside the word loop, measured at 0.1% over four pairs.

   A threshold that dropped to 32 levels at order 51 and above lived here
   briefly and is gone: it won 26.9% at n=53 against an untiled chain, but
   tiling the partner space wins 2.09x on its own, and with tiling in place 32
   levels measures 3.1% slower than 64 over three pairs.  The two do not
   compose, because both were shrinking the same working set and only tiling
   does it without costing filter resolution.  Coarsening remains a trap in its
   own right: untiled at n=53, 16 levels is 11% slower than 64 and 8 levels is
   3.6 times slower.  GOOD_LEVELS stays for sweeping a new machine. */
static int NLEV = NLEV_DEFAULT;
/* Slices of the outer table per decomposition.  One is the old behaviour. */
static int NOUTER = 1;
/* Live slice of the compacted bitmap set, in bytes, per thread.  Without
   tiling the chain streams 24.7 MB per outer row at n=53 (finding 250) and
   misses cache on every pass.  census.c has tiled its partner space since
   finding 141; this is the same idea.

   cbm is allocated inside the parallel region, so what competes for the shared
   last level cache is threads * TILE_BYTES, and the budget is a share of that
   cache rather than a constant.  Swept at n=53, thread-seconds per row:

       tile        1 MB    2 MB    4 MB    8 MB
       4 threads   42.41   38.38   39.64   59.78
       2 threads   36.22   33.27   31.76   34.07

   The optimum halves as the threads double, and both optima are the same
   aggregate: 4 x 2 MB and 2 x 4 MB are each 8 MB against 16 MB of L3.  An
   aggregate of a full L3 is worse at both thread counts, and a quarter is
   worse at both, so the rule is half the last level cache shared out.  A fixed
   2 MB tuned at 4 threads would ask for 28 MB on a 14 thread run against 8 MB
   of L3, which is the regime where the 8 MB column above costs 56%. */
static long TILE_BYTES = 0;             /* 0 means derive it at startup */

/* Largest last level cache the running CPU reports, in bytes, or 0. */
static long l3_bytes(void) {
    long best = 0;
    for (int i = 0; i < 16; i++) {
        char path[128];
        FILE *f;
        int level = 0;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        if (!(f = fopen(path, "r"))) break;
        if (fscanf(f, "%d", &level) != 1) level = 0;
        fclose(f);
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        if (!(f = fopen(path, "r"))) continue;
        long sz = 0; char unit = 0;
        if (fscanf(f, "%ld%c", &sz, &unit) >= 1) {
            if (unit == 'K') sz *= 1024;
            else if (unit == 'M') sz *= 1024L * 1024L;
            if (level >= 3 && sz > best) best = sz;
        }
        fclose(f);
    }
    return best;
}
#ifndef VW
#define VW 32                       /* words intersected per pass */
#endif
#define LEVEL(x) ((NLEV * (int)(x)) / (CEIL16 + 1))
static double *COS;                 /* [H][H] cos table, COS[k][s] for lag s+1 */

/* ---------------------------------------------------------------- helpers */

static void build_cos(void) {
    COS = malloc((size_t)H * H * sizeof(double));
    for (int k = 0; k < H; k++)
        for (int s = 0; s < H; s++)
            COS[(size_t)k * H + s] = cos(2.0 * M_PI * (k + 1) * (s + 1) / N);
}

/* PSD at the H independent nonzero frequencies, from the H autocorrelations.
   For a sequence with PAF values paf[0..H-1] at lags 1..H and row sum known,
   PSD(k) = sum_j x_j x_{j+0} + 2 sum_s paf[s] cos(...) = n + 2 sum_s ... */
static void psd_from_paf(const int8_t *paf, int16_t *out) {
    for (int k = 0; k < H; k++) {
        const double *row = COS + (size_t)k * H;
        double acc = N;
        for (int s = 0; s < H; s++) acc += 2.0 * paf[s] * row[s];
        double v = floor(acc * SCALE);
        if (v < 0) v = 0;
        if (v > 32767) v = 32767;
        out[k] = (int16_t)v;
    }
}

/* periodic autocorrelation at lags 1..H of a full length-n sequence */

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
/* The solve word: a bitmask over the H positions of a row or column.  32 bits
   reaches H = 32, that is n = 65, which is where the good-matrix census stops.
   Building with -DGS_WIDE carries them in 64 bits and reaches H = 63, n = 127,
   at the cost of halving every vector lane count.  The narrow word stays the
   default because every order through 65 fits it. */
#ifdef GS_WIDE
typedef uint64_t gsw;
#define GSW_BITS 64
#define GS_MAXH  63
#else
typedef uint32_t gsw;
#define GSW_BITS 32
#define GS_MAXH  32
#endif
#define GSW_ONE  ((gsw)1)
#ifdef GS_WIDE
#define GSW_POPCNT(x) __builtin_popcountll(x)
#else
#define GSW_POPCNT(x) __builtin_popcount(x)
#endif
#ifdef GS_WIDE
#define GSW_CTZ(x) __builtin_ctzll(x)
#else
#define GSW_CTZ(x) __builtin_ctz(x)
#endif
#define GSW_HMASK ((H >= GSW_BITS) ? ~(gsw)0 : ((GSW_ONE << H) - 1))

#ifdef GC_VERIFY
/* Substitution check.  The solve returns candidate vectors s over H unknowns
   for the system whose column c is the row-bitmask col0[c], with right-hand
   side bb0: s is a solution exactly when the XOR of the col0[c] for the set
   bits c of s equals bb0.  That statement does not depend on how the solve is
   implemented, so it catches a truncated word at any H, which the H <= 32
   regression cannot: every path in this file is correct below 33 bits. */
static unsigned long long VF_OK = 0, VF_BAD = 0;
static void gc_verify(const gsw *col0, gsw bb0, const gsw *cand, int ncd) {
    for (int i = 0; i < ncd; i++) {
        gsw acc = 0, v = cand[i] & GSW_HMASK;
        for (int c = 0; c < H; c++) if ((v >> c) & 1) acc ^= col0[c];
        if (acc == bb0)
#pragma omp atomic
            VF_OK++;
        else
#pragma omp atomic
            VF_BAD++;
    }
}
#endif
static gsw *EMASK_;
static int *FIXED_, *M1CONST_;
/* Byte tables for t, finding 328.  See hkt_build. */
static gsw *TBYTE_ = NULL;        /* [NT][256], one solve word per entry */
static int TNB_ = 0;
static gsw *M1INVCOL_;
/* The unfolded row word.  Rows are n bits, so orders above 63 need more than
   a uint64_t; building with -DGC_WIDE carries them in an unsigned __int128 and
   raises the order limit to 127.  The narrow word is the default because every
   order below 65 fits it and 128-bit shifts and popcounts are not free. */
#ifdef GC_WIDE
typedef unsigned __int128 gcw;
#define GCW_ONE ((gcw)1)
#define GCW_POP(x) (__builtin_popcountll((uint64_t)(x)) \
                    + __builtin_popcountll((uint64_t)((x) >> 64)))
#define GC_MAXN 127
#else
typedef uint64_t gcw;
#define GCW_ONE 1ULL
#define GCW_POP(x) __builtin_popcountll(x)
#define GC_MAXN 63
#endif
static gcw *CONSTALL_, *FIXM_, *CONSTM_, *NFIXM_;

static int hkt_mvar(int j) { int t = N - j; return j < t ? j : t; }
static inline gcw hkt_rot(gcw x, int k) {
    gcw m = (N >= (int)(8 * sizeof(gcw))) ? ~(gcw)0 : ((GCW_ONE << N) - 1);
    x &= m; return ((x << k) | (x >> (N - k))) & m;
}

/* Reverse the low H bits of a solve word, leaving the result in the low H
   bits.  The old helper was unconditionally 32-bit: in a GS_WIDE build it
   discarded positions 32..H-1 and evaluated 32-H as a negative shift. */
static inline gsw hkt_revh(gsw x) {
#ifdef GS_WIDE
    x = ((x >> 1) & UINT64_C(0x5555555555555555))
      | ((x & UINT64_C(0x5555555555555555)) << 1);
    x = ((x >> 2) & UINT64_C(0x3333333333333333))
      | ((x & UINT64_C(0x3333333333333333)) << 2);
    x = ((x >> 4) & UINT64_C(0x0f0f0f0f0f0f0f0f))
      | ((x & UINT64_C(0x0f0f0f0f0f0f0f0f)) << 4);
    return (gsw)(__builtin_bswap64((uint64_t)x) >> (64 - H));
#else
    x = ((x >> 1) & UINT32_C(0x55555555)) | ((x & UINT32_C(0x55555555)) << 1);
    x = ((x >> 2) & UINT32_C(0x33333333)) | ((x & UINT32_C(0x33333333)) << 2);
    x = ((x >> 4) & UINT32_C(0x0f0f0f0f)) | ((x & UINT32_C(0x0f0f0f0f)) << 4);
    return (gsw)(__builtin_bswap32((uint32_t)x) >> (32 - H));
#endif
}

static int hkt_build(void) {
    EU_ = malloc((size_t)(H + 1) * H); EW_ = malloc((size_t)(H + 1) * H);
    ECONST_ = malloc((size_t)(H + 1) * H);
    EMASK_ = malloc(sizeof(gsw) * (size_t)(H + 1) * H);
    FIXED_ = calloc(H + 1, sizeof(int)); M1CONST_ = calloc(H + 1, sizeof(int));
    CONSTALL_ = calloc(H + 1, sizeof(gcw));
    FIXM_     = calloc(H + 1, sizeof(gcw));
    CONSTM_   = calloc(H + 1, sizeof(gcw));
    NFIXM_    = calloc(H + 1, sizeof(gcw));
    if (!CONSTALL_ || !FIXM_ || !CONSTM_ || !NFIXM_) return 0;
    gsw *M1ROW = calloc(H + 1, sizeof(gsw));
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
            if (j == jj) { fx++; FIXM_[k] |= GCW_ONE << j;
                           CONSTALL_[k] |= GCW_ONE << j; continue; }
            int u = hkt_mvar(j), w = hkt_mvar((j + k) % N);
            EU_[(k - 1) * H + nr] = (int8_t)u; EW_[(k - 1) * H + nr] = (int8_t)w;
            gsw m = 0; int8_t isc = 0;
            if (u == 0 && w == 0) isc = 1;
            else if (u == 0) m = GSW_ONE << (w - 1);
            else if (w == 0) m = GSW_ONE << (u - 1);
            else if (u == w) isc = 1;
            else m = (GSW_ONE << (u - 1)) | (GSW_ONE << (w - 1));
            EMASK_[(k - 1) * H + nr] = m; ECONST_[(k - 1) * H + nr] = isc;
            if (isc) { CONSTM_[k] |= GCW_ONE << j; CONSTALL_[k] |= GCW_ONE << j;
                       CONSTALL_[k] |= GCW_ONE << jj; }
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
        gsw r = 0;
        for (int u = 1; u <= H; u++) if (mult[u] & 1) r |= GSW_ONE << (u - 1);
        M1ROW[k] = r; M1CONST_[k] = cst;
    }
    free(seen);
    /* invert the stage-1 matrix once: it never depends on A or B */
    gsw aug[64], inv[64];
    for (int k = 0; k < H; k++) { aug[k] = M1ROW[k + 1]; inv[k] = GSW_ONE << k; }
    int rank = 0;
    for (int c = 0; c < H; c++) {
        int pr = -1;
        for (int r2 = rank; r2 < H; r2++) if ((aug[r2] >> c) & 1) { pr = r2; break; }
        if (pr < 0) continue;
        gsw t = aug[pr]; aug[pr] = aug[rank]; aug[rank] = t;
        t = inv[pr]; inv[pr] = inv[rank]; inv[rank] = t;
        for (int r2 = 0; r2 < H; r2++)
            if (r2 != rank && ((aug[r2] >> c) & 1)) { aug[r2] ^= aug[rank]; inv[r2] ^= inv[rank]; }
        rank++;
    }
    free(M1ROW);
    if (rank != H) return 0;               /* stage 1 must be invertible */
    M1INVCOL_ = calloc(H, sizeof(gsw));
    if (!M1INVCOL_) return 0;
    for (int r2 = 0; r2 < H; r2++)
        for (int c = 0; c < H; c++)
            if ((inv[r2] >> c) & 1) M1INVCOL_[r2] |= GSW_ONE << c;
    /* Byte tables for t, finding 328.  t_c is the parity of
       M1INVCOL_[c] AND rhs1, which is a GF(2) matrix-vector product, so t is
       the XOR of those rows of the transpose that rhs1 selects.  Splitting
       rhs1 into bytes and precomputing the XOR for each byte value replaces H
       parity operations with ceil(H/8) loads and as many XORs.  The tables are
       ceil(H/8) by 256 words, three kilobytes at H=21, so they sit in L1. */
    {
        gsw rowt[64];
        for (int i = 0; i < H; i++) {
            gsw w = 0;
            for (int c = 0; c < H; c++) if ((M1INVCOL_[c] >> i) & 1) w |= GSW_ONE << c;
            rowt[i] = w;
        }
        TNB_ = (H + 7) / 8;
        TBYTE_ = calloc((size_t)TNB_ * 256, sizeof(gsw));
        if (!TBYTE_) return 0;
        for (int b = 0; b < TNB_; b++)
            for (int v = 1; v < 256; v++) {
                int low = __builtin_ctz((unsigned)v);
                int idx = 8 * b + low;
                gsw base = TBYTE_[(size_t)b * 256 + (unsigned)(v & (v - 1))];
                TBYTE_[(size_t)b * 256 + v] = base ^ (idx < H ? rowt[idx] : (gsw)0);
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
   touched, which is what makes the small-H case worth separating. */

/* Finding 478.  Reproducing the census byte-identically proves the program is
   correct, not that a conditional path ran.  -DGS_WIDE is validated at orders
   whose H fits a 32-bit word regardless, so without this the wide build could
   be measuring the narrow path twice.  It aborts rather than warns, because a
   check that only prints is one someone has to go looking for. */
static void gs_check_word(void) {
    if (sizeof(gsw) * 8 != GSW_BITS) {
        fprintf(stderr, "solve word is %zu bits, GSW_BITS says %d\n",
                sizeof(gsw) * 8, GSW_BITS);
        exit(1);
    }
    if (H > GS_MAXH) {
        fprintf(stderr, "H = %d exceeds %d, the widest this build reaches;"
                " rebuild with -DGS_WIDE\n", H, GS_MAXH);
        exit(1);
    }
#ifdef GS_WIDE
    fprintf(stderr, "   solve word: 64 bits (wide build), H = %d\n", H);
#else
    if (H > 32) { fprintf(stderr, "narrow build cannot reach H = %d\n", H); exit(1); }
#endif
}

static inline void hkt_tr(const gsw *a, int h, gsw *col) {
#ifdef GC_TRSTUB
    /* Ablation only: same h loads and h stores, no bit shuffling.  Results are
       wrong; this exists to price the transpose inside the rebuild phase. */
    for (int c = 0; c < h; c++) col[c] = a[c];
    return;
#endif
#if defined(__AVX2__) && defined(GS_WIDE)
    /* Wide transpose, by halves.  The obvious wide analogue of the 32-bit path
       uses movemask_pd, which reads the top bit of four 64-bit lanes against
       movemask_ps's eight 32-bit ones, so it needs ceil(h/4) groups per column
       instead of ceil(h/8): at h = 33 that is nine movemasks a column rather
       than five, and the ablation priced the whole transpose at 29% of the wide
       build's runtime at both n = 43 and n = 45.

       Splitting each row into its low and high 32 bits removes the pd path
       entirely.  Column c < 32 reads bit c of the low halves and column 32 + c
       reads bit c of the high halves, both through movemask_ps.  Below h = 32
       the high pass does not run at all, so the wide build costs the same as
       the narrow one there. */
    if (h <= 64) {
        int ng = (h + 7) >> 3;
        uint32_t lo[64], hi[64];
        for (int i = 0; i < h; i++) {
            lo[i] = (uint32_t)a[i]; hi[i] = (uint32_t)(a[i] >> 32);
        }
        for (int i = h; i < 8 * ng; i++) { lo[i] = 0; hi[i] = 0; }
        __m256i v[8];
        int nlo = h < 32 ? h : 32, nhi = h - 32;
        for (int g = 0; g < ng; g++)
            v[g] = _mm256_slli_epi32(
                       _mm256_loadu_si256((const __m256i *)(lo + 8 * g)), 32 - nlo);
        for (int c = nlo - 1; c >= 0; c--) {
            gsw m = 0;
            for (int g = 0; g < ng; g++) {
                m |= (gsw)(unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(v[g]))
                     << (8 * g);
                v[g] = _mm256_slli_epi32(v[g], 1);
            }
            col[c] = m;
        }
        if (nhi > 0) {
            for (int g = 0; g < ng; g++)
                v[g] = _mm256_slli_epi32(
                           _mm256_loadu_si256((const __m256i *)(hi + 8 * g)), 32 - nhi);
            for (int c = nhi - 1; c >= 0; c--) {
                gsw m = 0;
                for (int g = 0; g < ng; g++) {
                    m |= (gsw)(unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(v[g]))
                         << (8 * g);
                    v[g] = _mm256_slli_epi32(v[g], 1);
                }
                col[32 + c] = m;
            }
        }
        return;
    }
#endif
#if defined(__AVX2__) && !defined(GS_WIDE)
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
    /* The accumulator and the shift must both be the solve word.  This read
       uint32_t m and 1u << r, which is undefined once r reaches 32 and would
       have silently dropped every row past 31 at n=67, the order the wide
       build exists for.  No validation below H=32 could have caught it. */
    for (int c = 0; c < h; c++) {
        gsw m = 0;
        for (int r = 0; r < h; r++) if ((a[r] >> c) & 1) m |= GSW_ONE << r;
        col[c] = m;
    }
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
                             gsw *sout, gsw *col, gsw *bbout) {
    /* Target vector, finding 323.  Eight lags at a time: the two PAF bytes
       widen with cvtepi8_epi32, FIXED_ and M1CONST_ are int arrays read at
       consecutive k so they load straight, and the bit that goes into rhs1 is
       bit 1 of r, which a shift left by 30 moves to bit 31 where movemask_ps
       collects all eight at once.  The mod-4 reject becomes an OR accumulated
       across the block and tested after it, which finding 321 measured never
       fires at n=43 anyway. */
    int tgt[64]; gsw rhs1 = 0;
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
            rhs1 |= (gsw)mm << (k - 1);
        }
        if (!_mm256_testz_si256(vbad, vbad)) return 0;
#endif
        for (; k <= H; k++) {
            int v = -((int)pafA[k - 1] + (int)pafB[k - 1]) - 2 * FIXED_[k];
            if (v & 3) return 0;
            tgt[k] = v >> 1;
            int r = tgt[k] - 2 * M1CONST_[k];
            if ((r >> 1) & 1) rhs1 |= GSW_ONE << (k - 1);
        }
    }
    gsw t = TBYTE_[rhs1 & 0xffu];
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
    gcw maskn = (N >= (int)(8 * sizeof(gcw))) ? ~(gcw)0 : ((GCW_ONE << N) - 1);
    /* Closed form for the unfolded word, finding 315.  With n = 2H+1 and
       u_j = t_{min(j, n-j)}, position j in 1..H reads bit j-1 of t and position
       H+1+i in the upper half reads bit H-1-i, which is bit i of t with its low
       H bits reversed.  So the whole word is 1, t shifted up one, and the
       reversal shifted up H+1, which also removes the tb[] byte array that
       existed only to feed this loop. */
    gsw rv = hkt_revh(t);
    gcw uw = GCW_ONE | ((gcw)t << 1) | ((gcw)rv << (H + 1));
    gsw hmask = GSW_HMASK;
    gsw a[64]; gsw bvec = 0;
    for (int i = 0; i < 64; i++) a[i] = 0;
    for (int k = 1; k <= H; k++) {
        gcw kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
        gcw G    = kept & ~CONSTALL_[k] & maskn;
        gcw oddu = (G ^ hkt_rot(G, k)) & maskn;
        gsw odd = (gsw)(oddu >> 1) & hmask;
        /* Finding 442.  The CONSTM_ popcount entered rr twice with opposite
           signs, once through mono and once through the target term, so it
           cancels identically: rr = (keptall - kfix)/2 - (tgt[k] >> 1).
           Finding 447.  The two that remained are one, because kept & FIXM_[k]
           is a subset of kept, so popcount(kept) - popcount(kept & FIXM_[k])
           is popcount(kept & ~FIXM_[k]) exactly.  The mask is complemented
           once at build time and the AND is a single ANDN, so this removes a
           popcount per lag and adds nothing. */
        int rr = GCW_POP(kept & NFIXM_[k]) / 2 - (tgt[k] >> 1);
        a[k - 1] = odd; if ((rr >> 1) & 1) bvec |= GSW_ONE << (k - 1);
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

/* Finding 458.  The target vector splits into a per-A half and a per-B half.

   With ak = -PAF_A(k) - n - 2*FIXED_[k] and bk = n - PAF_B(k), the quantity
   hkt_prefix forms is v = -(PAF_A + PAF_B) - 2*FIXED_[k] = ak + bk, and both
   ak and bk are divisible by 4 because every PAF is n mod 4.  Writing them as
   4a and 4b, the bit that goes into rhs1 is

       ((v/2 - 2*M1CONST_[k]) >> 1) & 1  =  (a & 1) ^ (b & 1) ^ (M1CONST_ & 1)

   so rhs1 is the XOR of a per-A word and a per-B word, and since the byte
   tables are a linear map, t is the XOR of their images.  nw_c.c and gmat_c.c
   have used this since finding 393; good_c.c never did, and paid the whole
   target-vector loop on every pair to rebuild something that is one XOR of two
   precomputed words.  The A halves are computed once for the whole canonical
   A table and the B half once per outer row. */
static inline gsw gc_tmap(gsw r) {
    gsw t = 0;
    for (int b = 0; b < TNB_; b++)
        t ^= TBYTE_[(size_t)b * 256 + ((r >> (8 * b)) & 0xffu)];
    return t;
}

/* ta is stored as int8_t.  ta[k] = (-paf[k-1] - N - 2*FIXED_[k]) >> 2 with
   paf in [-N, N] and FIXED_[k] the number of j with 2j = -k mod N, which is at
   most 2 for odd N, so |ta| <= (2N + 4)/4 and the widest supported order,
   GC_MAXN = 127, gives at most 64.  Measured range is far tighter still:
   [-18,-4] at n=43 and n=45, [-21,-5] at n=51.  The narrowing is for traffic,
   not space: ATA is read once per survivor at a scattered index, so at n=67 it
   was three cache lines a survivor and is now one. */
static inline void gc_arow(const int8_t *pafA, int8_t *ta, gsw *ra) {
    gsw r = 0;
    for (int k = 1; k <= H; k++) {
        int ak = -(int)pafA[k - 1] - N - 2 * FIXED_[k];
        ta[k] = (int8_t)(ak >> 2);
        if ((((ak >> 1) - 2 * M1CONST_[k]) >> 1) & 1) r |= GSW_ONE << (k - 1);
    }
    for (int k = H + 1; k < H + 8; k++) ta[k] = 0;
    *ra = gc_tmap(r);
}

static inline void gc_brow(const int8_t *pafB, int8_t *su, gsw *rb) {
    gsw r = 0;
    for (int k = 1; k <= H; k++) {
        int u = (N - (int)pafB[k - 1]) >> 2;
        su[k - 1] = (int8_t)u;
        if (u & 1) r |= GSW_ONE << (k - 1);
    }
    for (int k = H; k < H + 8; k++) su[k] = 0;
    *rb = gc_tmap(r);
}

/* Rebuild from t what hkt_prefix would have built: mc, for the right-hand
   side, and the matrix.  Two lags interleaved, finding 456. */
static inline void gc_rows_from_t(gsw t, int16_t *mc, gsw *col) {
    const gcw maskn = (N >= (int)(8 * sizeof(gcw))) ? ~(gcw)0 : ((GCW_ONE << N) - 1);
    const gsw hmask = GSW_HMASK;
    gsw rv = hkt_revh(t);
    gcw uw = GCW_ONE | ((gcw)t << 1) | ((gcw)rv << (H + 1));
    gsw a[64];
    for (int i = 0; i < 64; i++) { a[i] = 0; mc[i] = 0; }
    int k = 1;
    for (; k + 1 <= H; k += 2) {
        const gcw nc0 = ~CONSTALL_[k] & maskn, nc1 = ~CONSTALL_[k + 1] & maskn;
        gcw r0 = ((uw << (N - k)) | (uw >> k)) & maskn;
        gcw r1 = ((uw << (N - k - 1)) | (uw >> (k + 1))) & maskn;
        gcw kept0 = (~(uw ^ r0)) & maskn, kept1 = (~(uw ^ r1)) & maskn;
        gcw G0 = kept0 & nc0, G1 = kept1 & nc1;
        gcw o0 = (G0 ^ (((G0 << k) | (G0 >> (N - k))) & maskn)) & maskn;
        gcw o1 = (G1 ^ (((G1 << (k + 1)) | (G1 >> (N - k - 1))) & maskn)) & maskn;
        a[k - 1] = (gsw)(o0 >> 1) & hmask;
        a[k]     = (gsw)(o1 >> 1) & hmask;
        mc[k - 1] = (int16_t)(GCW_POP(kept0 & NFIXM_[k]) / 2);
        mc[k]     = (int16_t)(GCW_POP(kept1 & NFIXM_[k + 1]) / 2);
    }
    for (; k <= H; k++) {
        gcw kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
        gcw G    = kept & ~CONSTALL_[k] & maskn;
        gcw oddu = (G ^ hkt_rot(G, k)) & maskn;
        a[k - 1] = (gsw)(oddu >> 1) & hmask;
        mc[k - 1] = (int16_t)(GCW_POP(kept & NFIXM_[k]) / 2);
    }
    hkt_tr(a, H, col);
}

/* bvec bit k-1 is bit 1 of mc[k-1] - ta[k] - su[k-1]. */
static inline gsw gc_bvec(const int16_t *mc, const int8_t *ta, const int8_t *su) {
    gsw bvec = 0;
    int kb = 1;
    /* The lane arithmetic here is 32-bit regardless of the solve word: mc, ta
       and su are int16/int/int8 arrays and each group yields eight bits.  Only
       the accumulator's width depends on GS_WIDE, so the wide build used to
       fall to the scalar loop for no reason.  The overrun past H reads the
       zero padding gc_arow and gc_brow leave in ta and su and the zeroed tail
       of mc, and GSW_HMASK discards those bits. */
#if defined(__AVX2__)
    for (; kb <= H; kb += 8) {
        __m256i vm = _mm256_cvtepi16_epi32(
                         _mm_loadu_si128((const __m128i *)(mc + kb - 1)));
        __m256i vt = _mm256_cvtepi8_epi32(
                         _mm_loadl_epi64((const __m128i *)(ta + kb)));
        __m256i vu = _mm256_cvtepi8_epi32(
                         _mm_loadl_epi64((const __m128i *)(su + kb - 1)));
        __m256i vd = _mm256_sub_epi32(_mm256_sub_epi32(vm, vt), vu);
        uint32_t mm = (uint32_t)_mm256_movemask_ps(
                          _mm256_castsi256_ps(_mm256_slli_epi32(vd, 30)));
        bvec |= (gsw)mm << (kb - 1);
    }
    bvec &= GSW_HMASK;
#else
    for (; kb <= H; kb++)
        if (((mc[kb - 1] - ta[kb] - su[kb - 1]) >> 1) & 1) bvec |= GSW_ONE << (kb - 1);
#endif
    return bvec;
}

/* Two eliminations in lockstep.  Branchless per lane: an inactive column takes
   R = 0, which makes its inner-loop XOR a no-op, so the two chains stay in
   step with no divergence and the machine can overlap one lane's stalls with
   the other's work. */
static inline void hkt_elim2(gsw *c0, gsw *bb0, int *rk0, int *pv0,
                             uint32_t *c1, uint32_t *bb1, int *rk1, int *pv1,
                             uint32_t *pm0, uint32_t *pm1) {
    uint32_t p0 = 0, p1 = 0, b0 = *bb0, b1 = *bb1;
    int r0 = 0, r1 = 0;
    for (int c = 0; c < H; c++) { pv0[c] = -1; pv1[c] = -1; }
    for (int c = 0; c < H; c++) {
        uint32_t m0 = c0[c] & ~p0, m1 = c1[c] & ~p1;
        int a0 = m0 != 0, a1 = m1 != 0;
        int q0 = a0 ? __builtin_ctz(m0) : 0, q1 = a1 ? __builtin_ctz(m1) : 0;
        uint32_t R0 = a0 ? (c0[c] & ~(1u << q0)) : 0;
        uint32_t R1 = a1 ? (c1[c] & ~(1u << q1)) : 0;
        p0 |= a0 ? (1u << q0) : 0u; p1 |= a1 ? (1u << q1) : 0u;
        pv0[c] = a0 ? q0 : -1;      pv1[c] = a1 ? q1 : -1;
        r0 += a0; r1 += a1;
        for (int k = 0; k < H; k++) {
            c0[k] ^= R0 & (uint32_t) - (int32_t)((c0[k] >> q0) & 1u);
            c1[k] ^= R1 & (uint32_t) - (int32_t)((c1[k] >> q1) & 1u);
        }
        b0 ^= R0 & (uint32_t) - (int32_t)((b0 >> q0) & 1u);
        b1 ^= R1 & (uint32_t) - (int32_t)((b1 >> q1) & 1u);
    }
    *bb0 = b0; *bb1 = b1; *rk0 = r0; *rk1 = r1; *pm0 = p0; *pm1 = p1;
}

#if defined(__AVX2__) && !defined(GS_WIDE)
static inline void hkt_elim8(gsw *c, gsw *bb, int *rk,
                             int pv[8][64], uint32_t *pm, int nlane) {
    uint32_t p[8];
    for (int L = 0; L < 8; L++) {
        p[L] = 0; rk[L] = 0;
        for (int k = 0; k < H; k++) pv[L][k] = -1;
    }
    for (int cc = 0; cc < H; cc++) {
        uint32_t Rl[8], ql[8];
        for (int L = 0; L < 8; L++) {
            if (L >= nlane) { Rl[L] = 0; ql[L] = 0; continue; }
            uint32_t col = c[(size_t)cc * 8 + L];
            uint32_t m = col & ~p[L];
            if (!m) { Rl[L] = 0; ql[L] = 0; continue; }
            int q = __builtin_ctz(m);
            ql[L] = (uint32_t)q;
            Rl[L] = col & ~(1u << q);
            p[L] |= 1u << q;
            pv[L][cc] = q;
            rk[L]++;
        }
        __m256i vR = _mm256_loadu_si256((const __m256i *)Rl);
        __m256i vq = _mm256_loadu_si256((const __m256i *)ql);
        const __m256i one = _mm256_set1_epi32(1);
        const __m256i zero = _mm256_setzero_si256();
        for (int k = 0; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * 8));
            __m256i bit = _mm256_and_si256(_mm256_srlv_epi32(v, vq), one);
            v = _mm256_xor_si256(v, _mm256_and_si256(vR, _mm256_sub_epi32(zero, bit)));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * 8), v);
        }
        __m256i vb = _mm256_loadu_si256((const __m256i *)bb);
        __m256i bit = _mm256_and_si256(_mm256_srlv_epi32(vb, vq), one);
        vb = _mm256_xor_si256(vb, _mm256_and_si256(vR, _mm256_sub_epi32(zero, bit)));
        _mm256_storeu_si256((__m256i *)bb, vb);
    }
    for (int L = 0; L < 8; L++) pm[L] = p[L];
}
#endif


/* Finding 444.  hkt_elim8 still did the pivot search one lane at a time:
   mask off the used rows, ctz for the pivot index, build R, set the bit in p,
   record the index.  That is about eight scalar ops per lane per column, so
   64 across the batch, against only H vector ops for the update that follows,
   and at H = 21 the scalar half is the larger one.

   All of it vectorises once the pivot is carried as a one-hot mask instead of
   an index, because 1 << q is exactly the lowest set bit of col & ~p, which is
   m & -m.  Then R = col ^ bit, p |= bit, and the update tests membership with
   (v & bit) == 0 rather than shifting by a per-lane count, which also drops the
   srlv.  A lane with no pivot has bit = 0, and then (v & 0) == 0 for every v,
   so its update is the identity with no special case.  Nothing scalar is left
   in the loop and the batch runs entirely in registers, the right-hand side
   included.

   hkt_tail_m reads the same one-hot masks, and the rank it needs is the
   popcount of the pivot mask, so no index is ever materialised. */
/* Four lanes of the elimination for the wide solve word.  The eight-lane
   version below is 32-bit-lane throughout and cannot run over a 64-bit word,
   so the wide build used the scalar path and paid 2.76x at n=41 and 3.08x at
   n=43.  Every operation in it has a 64-bit counterpart: sub_epi32 becomes
   sub_epi64 for the low-bit isolate, cmpeq_epi32 becomes cmpeq_epi64 for the
   zero test, and the rest are bitwise and width-agnostic.  Four lanes instead
   of eight, so the batch is two groups. */
#if defined(__AVX2__) && defined(GS_WIDE)
static inline void hkt_elim4v(gsw *c, gsw *bb, gsw *pivb, gsw *pm, int base) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i p = zero;
    __m256i vb = _mm256_loadu_si256((const __m256i *)(bb + base));
    for (int cc = 0; cc < H; cc++) {
        __m256i colv = _mm256_loadu_si256((const __m256i *)(c + (size_t)cc * 8 + base));
        __m256i m    = _mm256_andnot_si256(p, colv);
        __m256i bit  = _mm256_and_si256(m, _mm256_sub_epi64(zero, m));
        __m256i R    = _mm256_xor_si256(colv, bit);
        p = _mm256_or_si256(p, bit);
        _mm256_storeu_si256((__m256i *)(pivb + (size_t)cc * 8 + base), bit);
        for (int k = 0; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * 8 + base));
            __m256i ns = _mm256_cmpeq_epi64(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * 8 + base), v);
        }
        __m256i nb = _mm256_cmpeq_epi64(_mm256_and_si256(vb, bit), zero);
        vb = _mm256_xor_si256(vb, _mm256_andnot_si256(nb, R));
    }
    _mm256_storeu_si256((__m256i *)(bb + base), vb);
    _mm256_storeu_si256((__m256i *)(pm + base), p);
}
static inline void hkt_elim8v(gsw *c, gsw *bb, gsw *pivb, gsw *pm) {
    hkt_elim4v(c, bb, pivb, pm, 0);
    hkt_elim4v(c, bb, pivb, pm, 4);
}
#endif

/* These eight-lane kernels are 32-bit-lane throughout, so they cannot run over
   a 64-bit solve word: reading eight 32-bit lanes out of an array of 64-bit
   values silently mixes halves.  The wide build takes the scalar per-lane path
   instead, which is why it is slower.  Correct 4-lane versions through
   cmpeq_epi64 and movemask_pd are possible and not written yet. */
#if defined(__AVX2__) && !defined(GS_WIDE)
static inline void hkt_elim8v(gsw *c, gsw *bb, gsw *pivb,
                              gsw *pm) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i p = zero;
    __m256i vb = _mm256_loadu_si256((const __m256i *)bb);
    for (int cc = 0; cc < H; cc++) {
        __m256i colv = _mm256_loadu_si256((const __m256i *)(c + (size_t)cc * 8));
        __m256i m    = _mm256_andnot_si256(p, colv);
        __m256i bit  = _mm256_and_si256(m, _mm256_sub_epi32(zero, m));
        __m256i R    = _mm256_xor_si256(colv, bit);
        p = _mm256_or_si256(p, bit);
        _mm256_storeu_si256((__m256i *)(pivb + (size_t)cc * 8), bit);
        for (int k = 0; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * 8));
            __m256i ns = _mm256_cmpeq_epi32(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * 8), v);
        }
        __m256i nb = _mm256_cmpeq_epi32(_mm256_and_si256(vb, bit), zero);
        vb = _mm256_xor_si256(vb, _mm256_andnot_si256(nb, R));
    }
    _mm256_storeu_si256((__m256i *)bb, vb);
    _mm256_storeu_si256((__m256i *)pm, p);
}
#endif

/* Tail on one-hot pivot masks.  Identical to hkt_tail except that a column is
   a pivot column when its mask is nonzero, membership is an AND against the
   mask, and the rank is the popcount of the pivot mask. */
static inline int hkt_tail_m(const gsw *col, gsw bb, gsw pivoted,
                             const gsw *pivb, gsw *cand) {
    if (bb & ~pivoted) return 0;
    int nul = H - GSW_POPCNT(pivoted);
    if (nul > 6) return 0;
    gsw part = 0;
    for (int c = 0; c < H; c++)
        if (bb & pivb[c]) part |= GSW_ONE << c;
    gsw basis[64]; int nf = 0;
    for (int c = 0; c < H; c++) {
        if (pivb[c]) continue;
        gsw w = GSW_ONE << c;
        for (int c2 = 0; c2 < H; c2++)
            if (col[c] & pivb[c2]) w |= GSW_ONE << c2;
        basis[nf++] = w;
    }
    int ncand = 1 << nf; if (ncand > HKT_MAXC) return 0;
    for (int mask = 0; mask < ncand; mask++) {
        gsw v = part;
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
static inline int hkt_tail(gsw *col, gsw bb, int rank,
                           const int *piv_of_col, gsw pivoted,
                           gsw *cand) {
    if (bb & ~pivoted) return 0;                 /* inconsistent */
    int nul = H - rank;
    if (nul > 6) return 0;
    gsw part = 0;
    for (int c = 0; c < H; c++) {
        int p = piv_of_col[c];
        if (p >= 0 && ((bb >> p) & 1u)) part |= GSW_ONE << c;
    }
    gsw basis[64]; int nf = 0;
    for (int c = 0; c < H; c++) {
        if (piv_of_col[c] >= 0) continue;
        gsw w = GSW_ONE << c;
        for (int c2 = 0; c2 < H; c2++) {
            int p = piv_of_col[c2];
            if (p >= 0 && ((col[c] >> p) & 1u)) w |= GSW_ONE << c2;
        }
        basis[nf++] = w;
    }
    int ncand = 1 << nf; if (ncand > HKT_MAXC) return 0;
    for (int mask = 0; mask < ncand; mask++) {
        gsw v = part;
        for (int i = 0; i < nf; i++) if ((mask >> i) & 1) v ^= basis[i];
        cand[mask] = v;
    }
    return ncand;
}

static inline int hkt_solve(const int8_t *pafA, const int8_t *pafB,
                            gsw *cand, gsw *sout) {
    gsw col[64], bb; int piv[64], rank;
    if (!hkt_prefix(pafA, pafB, sout, col, &bb)) return 0;
#ifdef GC_VERIFY
    gsw col0[64], bb0 = bb;
    for (int c = 0; c < H; c++) col0[c] = col[c];
#endif
    gsw p = 0;
    for (int c = 0; c < H; c++) piv[c] = -1;
    rank = 0;
    for (int c = 0; c < H; c++) {
        gsw m = col[c] & ~p;
        if (!m) continue;
        int q = GSW_CTZ(m);
        p |= GSW_ONE << q; piv[c] = q; rank++;
        gsw R = col[c] & ~(GSW_ONE << q);
        for (int c2 = 0; c2 < H; c2++)
            col[c2] ^= R & (gsw) - (long long)((col[c2] >> q) & 1u);
        bb ^= R & (gsw) - (long long)((bb >> q) & 1u);
    }
    {   int nc = hkt_tail(col, bb, rank, piv, p, cand);
#ifdef GC_VERIFY
        gc_verify(col0, bb0, cand, nc);
#endif
        return nc; }
}

/* Bit-word verification, finding 320.  A plus-minus-one row is stored as an
   N-bit word whose bit j is set when x_j is -1, so x_j x_{j+k} is -1 exactly
   where w and w rotated by k differ and PAF_x(k) = N - 2 popcount(w XOR
   rot(w,k)).  A is skew and B and C and D are symmetric, which only changes
   whether the upper half repeats the lower half or complements it.  The A and B
   PAFs are already in hand as the solve's own arguments, so only C and D are
   computed here. */
static inline gcw hkt_symword(gsw code) {
    gsw c = code & GSW_HMASK, nc = (~c) & GSW_HMASK;
    return ((gcw)nc << 1) | ((gcw)hkt_revh(nc) << (H + 1));
}
static inline gcw hkt_rotk(gcw w, int k) {
    gcw m = (N >= (int)(8 * sizeof(gcw))) ? ~(gcw)0 : ((GCW_ONE << N) - 1);
    return ((w >> k) | (w << (N - k))) & m;
}

/* The mask of the sequence, so PAF can come from popcounts.  paf_of below is
   O(N H) and has a modulo in its innermost statement; the identity
   PAF(s) = N - 2 popcount(m ^ rot(m, s)) is H popcounts and no expansion
   (finding 399, the same change nw_c.c took in 398). */
static gcw sym_mask_of(uint64_t code) {
    gcw m = GCW_ONE;                         /* x_0 = +1 */
    for (int j = 1; j <= H; j++)
        if ((code >> (j - 1)) & 1) {
            m |= GCW_ONE << j; m |= GCW_ONE << (N - j);
        }
    return m;
}
static gcw skew_mask_of(uint64_t code) {
    gcw m = GCW_ONE;                         /* x_0 = +1, x_{N-j} = -x_j */
    for (int j = 1; j <= H; j++) {
        if ((code >> (j - 1)) & 1) m |= GCW_ONE << j;
        else                       m |= GCW_ONE << (N - j);
    }
    return m;
}
static inline void paf_of_mask(gcw m, int8_t *paf) {
    gcw full = (N >= (int)(8 * sizeof(gcw))) ? ~(gcw)0 : ((GCW_ONE << N) - 1);
    for (int s = 1; s <= H; s++) {
        gcw r = ((m >> s) | (m << (N - s))) & full;
        paf[s - 1] = (int8_t)(N - 2 * GCW_POP(m ^ r));
    }
}

static void paf_of(const int8_t *x, int8_t *paf) {
    for (int s = 1; s <= H; s++) {
        int acc = 0;
        for (int j = 0; j < N; j++) acc += x[j] * x[(j + s) % N];
        paf[s - 1] = (int8_t)acc;
    }
}

/* ------------------------------------------------------------ enumeration */

/* skew: a_0 = 1, a_j = -a_{n-j}; free bits are j = 1..H */
/* Decimation.  For t coprime to n the map (x_j) -> (x_{tj mod n}) sends
   PAF_x(s) to PAF_x(ts), so it permutes the autocorrelation and carries a good
   matrix quadruple to another one when applied to all four rows at once.  It
   preserves skewness, symmetry, row sums and the spectrum, the last as a
   permutation of frequencies, so the ceiling filter is invariant too and an
   orbit survives or fails as a whole.

   Both encodings store bits for positions 1..H.  Under decimation position j
   takes its value from q = tj mod n, which is either a stored position or the
   negative of one, so the action on a code is a bit permutation together with
   a fixed XOR mask: the mask is zero for the symmetric encoding, where
   x_{n-q} = x_q, and marks the wrapped positions for the skew encoding, where
   x_{n-q} = -x_q. */
/* rows finished, across all threads, reset per decomposition */
/* Phase timing.  good_c has never been profiled in this repo; census.c has
   carried a phase line for most of its life and it is what made findings 189
   and 203 possible there.  Wall time per phase, summed over the run. */
static const char *CKPT = NULL;     /* checkpoint path; NULL disables all of it */
static double CKPT_EVERY = 300.0;   /* seconds between writes */
static double CKPT_LAST = 0.0;
static uint64_t *CKPT_DONE = NULL;  /* one bit per outer row of the live decomposition */
static int CKPT_NROW = 0;
static int CKPT_DI = 0;             /* decomposition currently in progress */
static int CKPT_OC = 0;             /* outer-table slice within it */

/* Checkpointing.  A run at n=53 is tens of core-hours and n=55 is hundreds,
   with nothing resumable: a reboot loses all of it, and a dropped connection
   cannot even be told apart from a dead run.  The state that has to survive is
   small.  Rows complete out of order under dynamic scheduling, so completion is
   a bitmap rather than a high-water mark.  Solutions are held only in memory
   until the run ends, which is its own loss risk, so they go in too.

   Written to a temporary file and renamed: rename is atomic within a
   directory, so an interrupted write cannot leave a half-file to resume from. */
static void ckpt_write(long long found, long long tested,
                       int nsol, const long long *sol) {
    if (!CKPT) return;
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", CKPT);
    FILE *g = fopen(tmp, "w");
    if (!g) return;
    fprintf(g, "good_c ckpt v1\n");
    fprintf(g, "n %d\n", N);
    fprintf(g, "decomp %d\n", CKPT_DI);
    fprintf(g, "chunk %d\n", CKPT_OC);
    fprintf(g, "nrow %d\n", CKPT_NROW);
    fprintf(g, "found %lld\n", found);
    fprintf(g, "tested %lld\n", tested);
    int words = (CKPT_NROW + 63) >> 6;
    fprintf(g, "done %d", words);
    for (int w = 0; w < words; w++)
        fprintf(g, " %llx", (unsigned long long)(CKPT_DONE ? CKPT_DONE[w] : 0ULL));
    fprintf(g, "\n");
    fprintf(g, "nsol %d\n", nsol);
    for (int i = 0; i < nsol; i++)
        fprintf(g, "sol %lld %lld %lld %lld\n",
                sol[(size_t)i * 4], sol[(size_t)i * 4 + 1],
                sol[(size_t)i * 4 + 2], sol[(size_t)i * 4 + 3]);
    if (fflush(g) != 0 || fclose(g) != 0) { remove(tmp); return; }
    rename(tmp, CKPT);
}

/* Returns 1 if a checkpoint for this same order was loaded.

   Parsed with fscanf rather than fgets: the done line carries one hex word per
   64 outer rows, which at n=53 is tens of kilobytes on a single line.  A
   line-buffered reader silently truncates it and the resume then repeats work
   it had already finished, which the final census does not reveal because
   redoing rows still produces the right answer. */
static int ckpt_read(int *di, int *oc, int *nrow, uint64_t **done,
                     long long *found, long long *tested,
                     int *nsol, int *solcap, long long **sol) {
    if (!CKPT) return 0;
    FILE *g = fopen(CKPT, "r");
    if (!g) return 0;
    char head[64];
    if (!fgets(head, sizeof head, g) || strncmp(head, "good_c ckpt v1", 14) != 0) {
        fclose(g); return 0;
    }
    int n = -1, d = -1, nr = -1, ok = 1, oc_ = 0;
    uint64_t *bits = NULL;
    char key[64];
    while (ok && fscanf(g, "%63s", key) == 1) {
        if (!strcmp(key, "n"))           { if (fscanf(g, "%d", &n) != 1) ok = 0; }
        else if (!strcmp(key, "decomp")) { if (fscanf(g, "%d", &d) != 1) ok = 0; }
        else if (!strcmp(key, "chunk"))  { if (fscanf(g, "%d", &oc_) != 1) ok = 0; }
        else if (!strcmp(key, "nrow"))   { if (fscanf(g, "%d", &nr) != 1) ok = 0; }
        else if (!strcmp(key, "found"))  { if (fscanf(g, "%lld", found) != 1) ok = 0; }
        else if (!strcmp(key, "tested")) { if (fscanf(g, "%lld", tested) != 1) ok = 0; }
        else if (!strcmp(key, "done")) {
            int words = 0;
            if (fscanf(g, "%d", &words) != 1 || words < 0) { ok = 0; break; }
            free(bits);
            bits = calloc((size_t)(words > 0 ? words : 1), sizeof(uint64_t));
            if (!bits) { ok = 0; break; }
            for (int w = 0; w < words; w++) {
                unsigned long long v = 0;
                if (fscanf(g, "%llx", &v) != 1) { ok = 0; break; }
                bits[w] = (uint64_t)v;
            }
        }
        else if (!strcmp(key, "nsol")) {
            int ns = 0;
            if (fscanf(g, "%d", &ns) != 1 || ns < 0) { ok = 0; break; }
            if (ns > 0) {
                free(*sol);
                *sol = malloc((size_t)ns * 4 * sizeof(long long));
                if (!*sol) { ok = 0; break; }
                *solcap = ns; *nsol = 0;
            }
        }
        else if (!strcmp(key, "sol")) {
            long long a, b, c, e;
            if (fscanf(g, "%lld %lld %lld %lld", &a, &b, &c, &e) != 4) { ok = 0; break; }
            if (*sol && *nsol < *solcap) {
                long long *r = *sol + (size_t)(*nsol) * 4;
                r[0] = a; r[1] = b; r[2] = c; r[3] = e;
                (*nsol)++;
            }
        }
        else { ok = 0; break; }
    }
    fclose(g);
    if (!ok || n != N || d < 0) { free(bits); return 0; }
    *di = d; *oc = oc_; *nrow = nr; *done = bits;
    return 1;
}

static double PH_TABLE, PH_COMPACT, PH_REBUILD, PH_PAIR;
static double PH_SURV, PH_SOLVE;   /* the finding-468 upstream split */

#ifdef GC_TARANGE
/* ATA is int at H+8 per candidate, three cache lines a survivor at n=67.  If
   ta fits a narrower type the per-survivor read shrinks by the same factor,
   which is what matters if the run is memory-bound (finding 485). */
static int TA_MIN = 1 << 30, TA_MAX = -(1 << 30);
#endif
/* rdtsc, for the phase instruments.  This used to live inside the GC_TSTAT
   block while GC_SOLVESPLIT also used it, so building the split instrument on
   its own failed to compile and every split measurement was silently taken
   with both flags set. */
#define GC_TICK() __rdtsc()
#ifdef GC_SOLVESPLIT
/* Findings 469 and 470 put good_c at 92% solve and named the solve loop's
   table traffic as the target, but never split the solve itself.  These three
   accumulators cut the eight-lane batch at its two internal boundaries: the
   per-lane rebuild of the matrix from t, the batched elimination, and the
   de-transpose plus tail.  Whatever they leave is the candidate loop. */
static double PH_RB, PH_EL, PH_TL, PH_ALL;
#endif
#ifdef GC_TSTAT
/* How often does the rebuild see a t it has already been given?  gmat_c caches
   this and good_c does not; whether that is an oversight or a correct choice
   depends on the reuse rate, which nobody has measured. */
static long long TS_CALLS;
static uint64_t *TS_SEEN;
static long long TS_DISTINCT;
static long long TS_ROWS, TS_RBDIST;
static long long RK_HIST[40], RK_INCONS, RK_TOTAL;
static uint64_t *PM_SEEN; static long long PM_DIST;
static uint64_t *TS_RBSEEN;
/* The first version read CLOCK_THREAD_CPUTIME_ID and took an OpenMP atomic six
   times per eight-lane batch, which doubled the solve it was meant to measure:
   12.24 thread-seconds against 6.39 at n=39.  rdtsc costs tens of cycles and
   the batch totals are reduced into the shared accumulators once per outer
   row, so the atomics fall from one per batch to three per row. */
#endif
/* Wall clock, for anything that has to track real elapsed time: the checkpoint
   interval, and the table-build phase which is measured once outside the
   parallel region. */
static double wnow(void) {
    struct timeval t; gettimeofday(&t, 0);
    return t.tv_sec + 1e-6 * t.tv_usec;
}
/* Per-thread CPU, for the phase timers.  They previously used wall clock, which
   made them read high whenever the machine was busy: measured on identical work
   at n=39, 0.49 thread-seconds on a quiet machine, 0.59 under moderate load and
   1.05 with another job holding half the cores, a factor of 2.1 (finding 245).
   Every phase split and every slice-derived cost taken that way was an upper
   bound rather than a measurement.  CLOCK_THREAD_CPUTIME_ID counts only the
   calling thread's own CPU, so a busy machine slows the run without moving the
   number. */
static double cpunow(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static int PROGRESS;
#ifndef PROGRESS_EVERY
#define PROGRESS_EVERY 2000
#endif
static int NT;                      /* number of t coprime to n */
static int8_t *DSRC;                /* [NT][H] source bit for each position */
static long long *DFLIP;            /* [NT] skew sign mask */

static int gcd_i(int a, int b) { while (b) { int r = a % b; a = b; b = r; } return a; }

static void build_decim(void) {
    NT = 0;
    for (int t = 1; t < N; t++) if (gcd_i(t, N) == 1) NT++;
    DSRC = malloc((size_t)NT * H);
    DFLIP = malloc((size_t)NT * sizeof(long long));
    int i = 0;
    for (int t = 1; t < N; t++) {
        if (gcd_i(t, N) != 1) continue;
        long long flip = 0;
        for (int j = 1; j <= H; j++) {
            int q = (int)(((long long)t * j) % N);
            if (q <= H) DSRC[(size_t)i * H + j - 1] = (int8_t)(q - 1);
            else { DSRC[(size_t)i * H + j - 1] = (int8_t)(N - q - 1); flip |= 1LL << (j - 1); }
        }
        DFLIP[i++] = flip;
    }
}

static long long decimate(long long code, int i, int skew) {
    const int8_t *src = DSRC + (size_t)i * H;
    long long r = 0;
    for (int j = 0; j < H; j++) r |= ((code >> src[j]) & 1LL) << j;
    return skew ? r ^ DFLIP[i] : r;
}

/* true when this code is the smallest in its decimation orbit */
static int decim_canonical(long long code) {
    for (int i = 1; i < NT; i++)
        if (decimate(code, i, 1) < code) return 0;
    return 1;
}

static void skew_seq(long long code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v;
        x[N - j] = (int8_t)-v;
    }
}

/* symmetric: x_0 = 1, x_j = x_{n-j}; the code selects which pairs are +1 */
static void sym_seq(long long code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v;
        x[N - j] = v;
    }
}

/* row sum of a symmetric sequence with x_0 = 1 and p pairs set to +1 */
static int sym_rowsum(int p) { return 4 * p - (N - 2); }

/* ----------------------------------------------------------------- tables */

typedef struct {
    int      cnt;
    int8_t  *paf;      /* [cnt][H] */
    int16_t *psd;      /* [cnt][H]; NULL when the table stores lev instead */
    int8_t  *lev;      /* [cnt][H]; NULL when the table stores psd instead */
    gsw     *code;     /* [cnt]; one bit for each of the H free positions */
} Table;

static void table_free(Table *t) {
    free(t->paf); free(t->psd); free(t->lev); free(t->code);
    t->paf = NULL; t->psd = NULL; t->lev = NULL; t->code = NULL; t->cnt = 0;
}

static long long build_raw;         /* ceiling survivors before decimation */

/* Build the table of sequences of one kind that survive the spectral ceiling.
   A sequence with any PSD value above 4n cannot appear in a good matrix,
   because the four PSD values are nonnegative and sum to 4n.  This is the
   analogue of finding 57 in the best-matrix census, and it is what keeps the
   tables from being the whole space.

   For the skew kind the decimation test is applied here rather than to a
   finished table.  Every solution orbit contains one whose skew row is the
   smallest in its own decimation orbit, so only those need storing, and
   storing only those keeps the table a factor phi(n) smaller than the set of
   ceiling survivors instead of allocating that set and then discarding it.

   The enumeration is split over the threads by contiguous ranges of the code,
   and the ranges are concatenated in order, so the table is in ascending code
   order exactly as a serial build leaves it. */
/* keep_psd chooses what the table carries per row alongside paf.  The skew
   table needs the int16 spectrum, because the candidate loop reads it per
   candidate.  The outer table does not: its spectrum was read once per row and
   is now rebuilt there by psd_from_paf, so it carries the quantised level
   instead, at H bytes a row rather than 2H, and that level is what the filter
   wanted anyway.  Storing it here also removes the separate olev pass and its
   allocation, which is why this cuts the outer table from 3H bytes a row to H:
   57 GB to 28.6 at n=71, and it lowers the peak rather than only the
   steady state. */
/* [clo, chi) is the range of codes to enumerate, so the caller can build the
   table one slice at a time instead of whole.  Pass 0 and 1<<H for the whole
   thing.  The internal nchunk split subdivides whatever range it is given,
   and the slices come out in ascending code order as before, so a sequence
   of ranges covering [0, 1<<H) yields exactly the rows of one full build in
   the same order.  This is what lets n=71 run without holding 28 GB of outer
   table at once; see docs/good_matrices_plan.md. */
static Table build_table(int is_skew, int want_p, int threads, int keep_psd,
                         long long clo, long long chi) {
    double _t0 = wnow();
    long long cap = chi - clo;
    Table t; memset(&t, 0, sizeof(t));
    if (threads < 1) threads = 1;
    int nchunk = threads * 8;
    int8_t   **cpaf  = calloc((size_t)nchunk, sizeof(int8_t *));
    int16_t  **cpsd  = calloc((size_t)nchunk, sizeof(int16_t *));
    int8_t   **clev  = calloc((size_t)nchunk, sizeof(int8_t *));
    gsw      **ccode = calloc((size_t)nchunk, sizeof(gsw *));
    int *ccnt = calloc((size_t)nchunk, sizeof(int));
    long long *craw = calloc((size_t)nchunk, sizeof(long long));
    if (!cpaf || !cpsd || !clev || !ccode || !ccnt || !craw) {
        fprintf(stderr, "out of memory\n"); exit(1);
    }
#pragma omp parallel for schedule(dynamic)
    for (int ch = 0; ch < nchunk; ch++) {
        long long lo = clo + cap * ch / nchunk,
                  hi = clo + cap * (ch + 1) / nchunk;
        int alloc = 1 << 10, cnt = 0;
        long long raw = 0;
        int8_t   *paf_a  = malloc((size_t)alloc * H);
        int16_t  *psd_a  = malloc((size_t)alloc * H * sizeof(int16_t));
        int8_t   *lev_a  = NULL;
        gsw      *code_a = malloc((size_t)alloc * sizeof(gsw));
        int8_t *x = malloc(N), *paf = malloc(H);
        int16_t *psd = malloc((size_t)H * sizeof(int16_t));
        if (!paf_a || !psd_a || !code_a || !x || !paf || !psd) {
            fprintf(stderr, "out of memory\n"); exit(1);
        }
        for (long long code = lo; code < hi; code++) {
            if (!is_skew) {
                /* symmetric sequences are grouped by weight: only the codes with
                   the right popcount have the row sum this decomposition needs */
                if (__builtin_popcountll((unsigned long long)code) != want_p) continue;
                paf_of_mask(sym_mask_of(code), paf);
            } else {
                paf_of_mask(skew_mask_of(code), paf);
            }
            psd_from_paf(paf, psd);
            int ok = 1;
            for (int k = 0; k < H; k++)
                if (psd[k] > CEIL16) { ok = 0; break; }
            if (!ok) continue;
            raw++;
            if (is_skew && !decim_canonical(code)) continue;
            if (cnt == alloc) {
                alloc *= 2;
                paf_a  = realloc(paf_a,  (size_t)alloc * H);
                psd_a  = realloc(psd_a,  (size_t)alloc * H * sizeof(int16_t));
                code_a = realloc(code_a, (size_t)alloc * sizeof(gsw));
                if (!paf_a || !psd_a || !code_a) {
                    fprintf(stderr, "out of memory\n"); exit(1);
                }
            }
            memcpy(paf_a + (size_t)cnt * H, paf, H);
            memcpy(psd_a + (size_t)cnt * H, psd, (size_t)H * sizeof(int16_t));
            code_a[cnt] = (gsw)code;
            cnt++;
        }
        free(x); free(paf); free(psd);
        if (!keep_psd) {
            /* Quantise the chunk in one tight contiguous pass and drop its
               spectrum here, so the whole-table array never exists.  LEVEL is
               an integer divide, and doing it in this shape rather than inside
               the enumeration above keeps it the same loop the old olev pass
               was, which the compiler handles far better. */
            lev_a = malloc((size_t)(cnt ? cnt : 1) * H);
            if (!lev_a) { fprintf(stderr, "out of memory\n"); exit(1); }
            for (size_t i = 0; i < (size_t)cnt * H; i++)
                lev_a[i] = (int8_t)LEVEL(psd_a[i]);
            free(psd_a); psd_a = NULL;
        }
        cpaf[ch] = paf_a; cpsd[ch] = psd_a; clev[ch] = lev_a; ccode[ch] = code_a;
        ccnt[ch] = cnt; craw[ch] = raw;
    }
    build_raw = 0;
    for (int ch = 0; ch < nchunk; ch++) { t.cnt += ccnt[ch]; build_raw += craw[ch]; }
    t.paf  = malloc((size_t)(t.cnt ? t.cnt : 1) * H);
    t.psd  = keep_psd
        ? malloc((size_t)(t.cnt ? t.cnt : 1) * H * sizeof(int16_t)) : NULL;
    t.lev  = keep_psd ? NULL : malloc((size_t)(t.cnt ? t.cnt : 1) * H);
    t.code = malloc((size_t)(t.cnt ? t.cnt : 1) * sizeof(gsw));
    if (!t.paf || (keep_psd ? !t.psd : !t.lev) || !t.code) {
        fprintf(stderr, "out of memory\n"); exit(1);
    }
    int at = 0;
    for (int ch = 0; ch < nchunk; ch++) {
        memcpy(t.paf + (size_t)at * H, cpaf[ch], (size_t)ccnt[ch] * H);
        if (keep_psd)
            memcpy(t.psd + (size_t)at * H, cpsd[ch],
                   (size_t)ccnt[ch] * H * sizeof(int16_t));
        else
            memcpy(t.lev + (size_t)at * H, clev[ch], (size_t)ccnt[ch] * H);
        memcpy(t.code + at, ccode[ch], (size_t)ccnt[ch] * sizeof(gsw));
        at += ccnt[ch];
        free(cpaf[ch]); free(cpsd[ch]); free(clev[ch]); free(ccode[ch]);
    }
    free(cpaf); free(cpsd); free(clev); free(ccode); free(ccnt); free(craw);
    PH_TABLE += wnow() - _t0;
    return t;
}

/* ------------------------------------------------------------ hash on PAF */

typedef struct { uint32_t key; int32_t idx; } Slot;

/* Eight bytes at a time.  The caller guarantees that ceil(len/8) whole words
   are readable and that the bytes past len are zero, so the value depends on
   the len bytes alone.  Any hash is safe here: a slot is accepted only after an
   exact memcmp on the autocorrelation. */
static uint64_t hash_paf(const int8_t *p, int len) {
    uint64_t h = 1469598103934665603ULL;
    int nw = (len + 7) >> 3;
    for (int i = 0; i < nw; i++) {
        uint64_t w;
        memcpy(&w, p + (size_t)i * 8, sizeof w);
        h = (h ^ w) * 1099511628211ULL;
        h ^= h >> 31;
    }
    return h;
}

/* ------------------------------------------------------------------- main */

/* the three-square decompositions of 4n-1 with every part congruent to n mod 4 */
static int decomps(int out[][3]) {
    int cnt = 0, target = 4 * N - 1;
    for (int a = -(2 * H + 3); a <= 2 * H + 3; a++) {
        if (((a % 2) + 2) % 2 == 0) continue;
        if ((((a - N) % 4) + 4) % 4) continue;
        for (int b = a; b <= 2 * H + 3; b++) {
            if (((b % 2) + 2) % 2 == 0) continue;
            if ((((b - N) % 4) + 4) % 4) continue;
            for (int c = b; c <= 2 * H + 3; c++) {
                if (((c % 2) + 2) % 2 == 0) continue;
                if ((((c - N) % 4) + 4) % 4) continue;
                if (a * a + b * b + c * c != target) continue;
                out[cnt][0] = a; out[cnt][1] = b; out[cnt][2] = c;
                cnt++;
            }
        }
    }
    return cnt;
}

static int p_for_sum(int s) {
    int num = s + N - 2;
    if (num % 4) return -1;
    int p = num / 4;
    return (p >= 0 && p <= H) ? p : -1;
}

#ifdef GC_FILTSTAT
static long long FS_OUTER, FS_PAIRS, FS_AFTER_BITMAP, FS_AFTER_CEIL;
/* Word operations actually executed in the bitmap chain, and the masks the
   chain was given.  The chain stops on the first word that empties, so the
   ANDs performed are far fewer than nap * awords and the difference is the
   whole point; counting them is the only way to know what the filter costs.
   Finding 503 made this the largest block in the program. */
static long long FS_ANDS, FS_NAP, FS_WORDS;
#endif
#ifdef GC_SOLVESTAT
static long long SS_PREFIX, SS_REJECT, SS_ELIM2, SS_SINGLE;
#endif

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr,
        "usage: good_c <n> [threads] [out] [checkpoint] [seconds]\n"); return 1; }
    N = atoi(argv[1]);
    if (N < 3 || N % 2 == 0) { fprintf(stderr, "n must be odd and at least 3\n"); return 1; }
    /* Rows are carried in uint64_t throughout, and the row masks read
       (N >= 64) ? ~0ULL : ((1ULL << N) - 1), which is the wrong mask above 64
       and a shift by 64 or more at N = 65.  Refuse rather than answer wrongly;
       nw_c.c already refuses on the same grounds. */
    if (N > GC_MAXN) { fprintf(stderr, "n > %d needs a wider word;"
        " rebuild with -DGC_WIDE\n", GC_MAXN); return 1; }
    H = (N - 1) / 2;
#ifdef GC_NULLPROBE
    /* Finding 475 wants the consistency test moved ahead of the rebuild.  With
       corank 1 the column space is a hyperplane, so consistency is one parity
       check w(t) . bb = 0 against the left null vector of M(t), and M depends
       on t alone.  The question is whether w(t) is cheap.  This probe builds
       M(t) for sampled t, extracts w by eliminating on the transpose, and tests
       whether t -> w(t) is GF(2)-linear, which would make it a byte-table
       lookup on the order of five loads instead of an elimination. */
    if (!hkt_build()) { fprintf(stderr, "hkt_build failed\n"); return 1; }
    {
        int NS = argc > 2 ? atoi(argv[2]) : 400;
        unsigned long long rs = 88172645463325252ULL;
        int corank1 = 0, samples = 0, lin_ok = 0, lin_bad = 0, wzero = 0;
        int chist[64]; for (int i = 0; i < 64; i++) chist[i] = 0;
        gsw wt[4096]; gsw tv[4096];
        if (NS > 4000) NS = 4000;
        for (int it = 0; it < NS; it++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            gsw t = (gsw)rs & GSW_HMASK;
            int16_t mcb[64]; gsw c0[64];
            gc_rows_from_t(t, mcb, c0);
            /* c0[c] is column c as a bitmask over the H rows.  The left null
               space is the null space of M^T, whose column r is the bitmask
               over c of bit r of c0[c]. */
            gsw rows[64];
            for (int r = 0; r < H; r++) {
                gsw m = 0;
                for (int c = 0; c < H; c++) if ((c0[c] >> r) & 1) m |= GSW_ONE << c;
                rows[r] = m;
            }
            /* eliminate over the r index to find w with sum_r w_r rows[r] = 0 */
            gsw red[64], tag[64]; int nr = 0;
            for (int r = 0; r < H; r++) { red[nr] = rows[r]; tag[nr] = GSW_ONE << r; nr++; }
            char used[64]; for (int r = 0; r < nr; r++) used[r] = 0;
            for (int c = 0; c < H; c++) {
                int pr = -1;
                for (int r = 0; r < nr; r++)
                    if (!used[r] && ((red[r] >> c) & 1)) { pr = r; break; }
                if (pr < 0) continue;
                used[pr] = 1;
                for (int r = 0; r < nr; r++)
                    if (r != pr && ((red[r] >> c) & 1)) { red[r] ^= red[pr]; tag[r] ^= tag[pr]; }
            }
            int nnull = 0; gsw w = 0;
            for (int r = 0; r < nr; r++) if (red[r] == 0) { nnull++; if (nnull == 1) w = tag[r]; }
            samples++;
            if (nnull == 1) corank1++;
            chist[nnull < 63 ? nnull : 63]++;
            if (w == 0) wzero++;
            if (it < 4000) { wt[it] = w; tv[it] = t; }
        }
        /* linearity: w(t1 ^ t2) == w(t1) ^ w(t2) on sampled pairs */
        for (int i = 0; i + 1 < samples && i < 600; i += 2) {
            gsw t3 = tv[i] ^ tv[i + 1];
            int16_t mcb[64]; gsw c0[64];
            gc_rows_from_t(t3, mcb, c0);
            gsw rows[64];
            for (int r = 0; r < H; r++) {
                gsw m = 0;
                for (int c = 0; c < H; c++) if ((c0[c] >> r) & 1) m |= GSW_ONE << c;
                rows[r] = m;
            }
            gsw red[64], tag[64]; int nr = 0;
            for (int r = 0; r < H; r++) { red[nr] = rows[r]; tag[nr] = GSW_ONE << r; nr++; }
            char used[64]; for (int r = 0; r < nr; r++) used[r] = 0;
            for (int c = 0; c < H; c++) {
                int pr = -1;
                for (int r = 0; r < nr; r++)
                    if (!used[r] && ((red[r] >> c) & 1)) { pr = r; break; }
                if (pr < 0) continue;
                used[pr] = 1;
                for (int r = 0; r < nr; r++)
                    if (r != pr && ((red[r] >> c) & 1)) { red[r] ^= red[pr]; tag[r] ^= tag[pr]; }
            }
            gsw w3 = 0; int nn = 0;
            for (int r = 0; r < nr; r++) if (red[r] == 0) { nn++; if (nn == 1) w3 = tag[r]; }
            if (nn != 1) continue;
            if (w3 == (wt[i] ^ wt[i + 1])) lin_ok++; else lin_bad++;
        }
        /* Pivot-mask concentration.  Consistency is bb & ~pivoted, and bb needs
           only the popcount half of the rebuild.  If pivoted takes few values
           and is cheap to know, the test moves ahead of oddu and the transpose
           for the rejected majority.  Count distinct pivot masks and the share
           taken by the commonest. */
        {
            gsw seen[4096]; long long cnt[4096]; int nd = 0;
            unsigned long long rs2 = 0x9e3779b97f4a7c15ULL;
            for (int it = 0; it < NS; it++) {
                rs2 ^= rs2 << 13; rs2 ^= rs2 >> 7; rs2 ^= rs2 << 17;
                gsw t = (gsw)rs2 & GSW_HMASK;
                int16_t mcb[64]; gsw c0[64];
                gc_rows_from_t(t, mcb, c0);
                gsw pv = 0;
                for (int c = 0; c < H; c++) {
                    gsw m = c0[c] & ~pv;
                    if (!m) continue;
                    int q = GSW_CTZ(m);
                    pv |= GSW_ONE << q;
                    gsw R = c0[c] & ~(GSW_ONE << q);
                    for (int c2 = 0; c2 < H; c2++)
                        c0[c2] ^= R & (gsw) - (long long)((c0[c2] >> q) & 1u);
                }
                int f = -1;
                for (int j = 0; j < nd; j++) if (seen[j] == pv) { f = j; break; }
                if (f < 0) { if (nd < 4096) { seen[nd] = pv; cnt[nd] = 1; nd++; } }
                else cnt[f]++;
            }
            long long top = 0; for (int j = 0; j < nd; j++) if (cnt[j] > top) top = cnt[j];
            /* Rows that are non-pivot under EVERY observed mask.  bb must be
               zero there for any consistent system, whichever mask applies, so
               testing bb against this set is a sound necessary condition that
               needs no elimination.  ALWAYS = complement of the OR of the
               pivot masks. */
            gsw orpiv = 0;
            for (int j = 0; j < nd; j++) orpiv |= seen[j];
            gsw always = ~orpiv & GSW_HMASK;
            printf("N=%d H=%d  distinct pivot masks %d of %d, commonest %.1f%%, "
                   "never-pivot rows %d of %d -> would reject %.1f%%\n",
                   N, H, nd, NS, 100.0 * top / NS,
                   GSW_POPCNT(always), H,
                   100.0 * (1.0 - 1.0 / (double)(1ULL << GSW_POPCNT(always))));
        }
        printf("N=%d H=%d n=%d  corank:", N, H, samples);
        for (int i = 0; i < 8; i++) if (chist[i])
            printf(" %d:%.1f%%", i, 100.0 * chist[i] / samples);
        printf("   corank1=%.1f%%  linear ok=%d bad=%d\n",
               100.0 * corank1 / samples, lin_ok, lin_bad);
        return 0;
    }
#endif
#ifdef GC_TSTAT
    TS_SEEN = calloc(((size_t)1 << H) / 64 + 1, 8);
    TS_RBSEEN = calloc(((size_t)1 << H) / 64 + 1, 8);
    PM_SEEN = calloc(((size_t)1 << H) / 64 + 1, 8);
#endif
    /* H = 32 is fine: every 1u << H in the masks is already guarded by
       H >= 32, and the code word is a uint32_t which holds exactly 32
       bits.  Past that the code word itself overflows. */
    gs_check_word();
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    const char *outpath = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    CKPT = (argc > 4 && argv[4][0] != '-') ? argv[4] : NULL;
    if (argc > 5) CKPT_EVERY = atof(argv[5]);
    /* GOOD_LEVELS overrides the quantisation for sweeps.  Any value is sound,
       since the level filter is a superset of the exact test at every level
       count, so this can only trade filter sharpness against bitmap size. */
    {   const char *e = getenv("GOOD_OUTER_CHUNKS");
        if (e && *e) {
            int v = atoi(e);
            if (v < 1 || v > 4096) {
                fprintf(stderr, "GOOD_OUTER_CHUNKS must be between 1 and 4096;"
                        " got %s\n", e);
                return 1;
            }
            NOUTER = v;
        }
    }
    {   const char *e = getenv("GOOD_LEVELS");
        if (e) {
            int v = atoi(e);
            /* Levels are stored in int8_t (alev, olev) and used as array
               indices, so anything above 127 wraps negative and indexes the
               bitmap out of bounds.  The bound used to read 256, and
               GOOD_LEVELS=192 segfaulted with no output at all, which is what
               the comment inviting a sweep on a new machine would have hit
               first.  Refuse what the storage cannot represent. */
            if (v < 2 || v > 127) {
                fprintf(stderr, "GOOD_LEVELS must be between 2 and 127;"
                        " levels are stored in int8_t\n");
                return 1;
            }
            NLEV = v;
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    /* Half the last level cache, shared among the threads that contend for it.
       TILE_KB overrides, which is how a machine gets swept before a long run:
       the detected cache is a single figure and says nothing about how many
       cache instances the cores are divided across. */
    if (!hkt_build()) { fprintf(stderr, "hkt_build failed at n=%d\n", N); exit(1); }
    {   const char *tv = getenv("TILE_KB");
        if (tv) {
            TILE_BYTES = atol(tv) * 1024L;
        } else {
            long l3 = l3_bytes();
            if (l3 <= 0) l3 = 8L * 1024 * 1024;
            TILE_BYTES = l3 / 2 / (threads > 0 ? threads : 1);
        }
        if (TILE_BYTES < 256L * 1024) TILE_BYTES = 256L * 1024;
        if (TILE_BYTES > 8L * 1024 * 1024) TILE_BYTES = 8L * 1024 * 1024;
    }
    build_cos();
    build_decim();

    int dec[64][3];
    int ndec = decomps(dec);
    printf("n=%d: 4n-1 = %d, %d row-sum decomposition%s\n",
           N, 4 * N - 1, ndec, ndec == 1 ? "" : "s");
    for (int i = 0; i < ndec; i++)
        printf("   (%d, %d, %d)\n", dec[i][0], dec[i][1], dec[i][2]);
    if (ndec == 0) { printf("n=%d: no valid row sums, no good matrices\n", N); return 0; }

    Table A = build_table(1, 0, threads, 1, 0, 1LL << H);
    long long araw = build_raw;
    printf("   skew A: %lld of %lld survive the ceiling, %d after decimation "
           "(%d maps, %.1fx)\n",
           araw, 1LL << H, A.cnt, NT, araw / (double)(A.cnt ? A.cnt : 1));

    /* Finding 458.  The A half of the target vector, once for the whole
       canonical skew table.  ta needs eight ints of slack past H because
       gc_bvec loads eight at a time from ta + kb. */
    int ATAW = H + 8;
    gsw *ARA = malloc((size_t)A.cnt * sizeof(gsw));
    int8_t *ATA = malloc((size_t)A.cnt * ATAW * sizeof(int8_t));
    if (!ARA || !ATA) { fprintf(stderr, "out of memory for the A halves\n"); return 1; }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < A.cnt; i++) {
        gc_arow(A.paf + (size_t)i * H, ATA + (size_t)i * ATAW, &ARA[i]);
#ifdef GC_TARANGE
        for (int k = 1; k <= H; k++) {
            int v = ATA[(size_t)i * ATAW + k];
#pragma omp critical
            {   if (v < TA_MIN) TA_MIN = v;
                if (v > TA_MAX) TA_MAX = v; }
        }
#endif
    }

    /* Level bitmaps over the skew table, the same mechanism applied one
       level up: for a fixed outer value the admissible A are contained in the
       intersection over k of {lv_A <= NLEV-1-lv_O(k)}, so the scan over the
       whole skew table becomes a word-at-a-time intersection followed by a
       bit walk.  The quantisation is against 4n and so does not depend on the
       row-sum decomposition; these are built once. */
    int awords = (A.cnt + 63) >> 6;
    uint64_t atail = (A.cnt & 63) ? (~0ULL) >> (64 - (A.cnt & 63)) : ~0ULL;
    uint64_t *abm = calloc((size_t)H * NLEV * awords, sizeof(uint64_t));
    int8_t *alev = malloc((size_t)A.cnt * H);
    if (!abm || !alev) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int i = 0; i < A.cnt; i++)
        for (int k = 0; k < H; k++)
            alev[(size_t)i * H + k] = (int8_t)LEVEL(A.psd[(size_t)i * H + k]);
    for (int k = 0; k < H; k++) {
        uint64_t *bk = abm + (size_t)k * NLEV * awords;
        for (int i = 0; i < A.cnt; i++)
            bk[(size_t)alev[(size_t)i * H + k] * awords + (i >> 6)]
                |= 1ULL << (i & 63);
        for (int t = 1; t < NLEV; t++) {
            uint64_t *dst = bk + (size_t)t * awords;
            const uint64_t *src = dst - awords;
            for (int w = 0; w < awords; w++) dst[w] |= src[w];
        }
    }

    FILE *f = outpath ? fopen(outpath, "w") : NULL;
    long long found = 0, tested = 0;
    long long *sol = NULL;
    int nsol = 0, solcap = 0;

    int resume_di = 0, resume_oc = 0, resume_nrow = 0;
    uint64_t *resume_done = NULL;
    if (ckpt_read(&resume_di, &resume_oc, &resume_nrow, &resume_done, &found,
                  &tested, &nsol, &solcap, &sol)) {
        fprintf(stderr, "   resuming from %s: decomposition %d of %d, "
                "slice %d, %d solutions already found\n",
                CKPT, resume_di, ndec, resume_oc, nsol);
    } else {
        resume_di = -1;
    }
    CKPT_LAST = wnow();

    for (int di = 0; di < ndec; di++) {
        if (resume_di >= 0 && di < resume_di) continue;
        CKPT_DI = di;
        int s1 = dec[di][0], s2 = dec[di][1], s3 = dec[di][2];
        int p1 = p_for_sum(s1), p2 = p_for_sum(s2), p3 = p_for_sum(s3);
        if (p1 < 0 || p2 < 0 || p3 < 0) continue;
        /* Only one symmetric table is built.  The solve enumerates (A, B)
           and derives the other two rows, and finding 300 showed their row
           sums fall out of the shift equations rather than needing tables of
           their own, so mid and hash have no role.  The plan note's rule picks
           the outer as the sequence with the largest row sum in absolute
           value, which is the smallest list, and that needs no build to
           decide.  This is where the method's memory saving actually lives:
           finding 304 removed the mid level array and bitmaps for -26%, but
           left all three tables built because role selection needed them. */
        int sbest = s1, pbest = p1;
        if (abs(s2) > abs(sbest)) { sbest = s2; pbest = p2; }
        if (abs(s3) > abs(sbest)) { sbest = s3; pbest = p3; }
        /* Outer table in slices.  Holding it whole is what stops n=71: its
           paf, level and code arrays come to about 28 GB there against
           bos-game's 28 GB of RAM, while the skew side the filter reads is
           independent of it and stays resident throughout.  GOOD_OUTER_CHUNKS
           sets the count; 1 reproduces the unsliced program exactly. */
        long long dfound = 0, dtested = 0;
#ifdef GC_ABL
        /* Declared out here, not beside the pragma: the outer-table slicing
           loop opens between this point and the parallel region, so a sink
           declared inside it is out of scope at the print below.  That is why
           GC_ABL stopped compiling when slicing landed. */
        long long dsink = 0;
#endif
        PROGRESS = 0;
        for (int oc = 0; oc < NOUTER; oc++) {
        if (resume_di >= 0 && di == resume_di && oc < resume_oc) continue;
        CKPT_OC = oc;
        long long _clo = (1LL << H) * oc / NOUTER,
                  _chi = (1LL << H) * (oc + 1) / NOUTER;
        Table T1 = build_table(0, pbest, threads, 0, _clo, _chi);
        printf("   (%d,%d,%d): outer row sum %d, symmetric table %d\n",
               s1, s2, s3, sbest, T1.cnt);
        Table *out_t = &T1, *mid_t = &T1, *hash_t = &T1;
        (void)mid_t; (void)hash_t;
        /* Per-row resume bitmap.  Without this CKPT_DONE stays NULL, which
           makes the periodic write at the bottom of the row loop and the skip
           at the top both dead: checkpoints then happen only at decomposition
           boundaries, and at n=67 one decomposition is about 250 core-hours,
           so a crash costs all of it.  The bit for a row is set after its
           solutions are appended, so a row whose bit is missing is merely
           redone on resume. */
        if (CKPT) {
            CKPT_NROW = out_t->cnt;
            size_t cw = ((size_t)CKPT_NROW + 63) >> 6;
            free(CKPT_DONE);
            CKPT_DONE = calloc(cw ? cw : 1, sizeof(uint64_t));
            if (!CKPT_DONE) { fprintf(stderr, "out of memory\n"); exit(1); }
            if (di == resume_di && oc == resume_oc && resume_done
                && resume_nrow == CKPT_NROW) {
                memcpy(CKPT_DONE, resume_done, cw * sizeof(uint64_t));
                int already = 0;
                for (size_t w = 0; w < cw; w++)
                    already += __builtin_popcountll(CKPT_DONE[w]);
                fprintf(stderr, "   resuming decomposition %d slice %d with "
                        "%d of %d outer rows already done\n",
                        di, oc, already, CKPT_NROW);
            }
            if (di == resume_di && oc == resume_oc) {
                free(resume_done); resume_done = NULL;
            }
        }
        /* the PAF hash over hash_t is gone with the fourth-row lookup */
        int words = (mid_t->cnt + 63) >> 6;
        /* build_table already quantised these, so there is no olev pass and no
           olev allocation: the levels are the table.  The outer spectrum is not
           stored at all now, and the one place that wants it, once per outer
           row, rebuilds it with psd_from_paf for H*H multiply-adds.  A.psd is a
           different matter and stays, since the candidate loop reads it per
           candidate rather than per row. */
        const int8_t *olev = out_t->lev;
        /* mlev and the mid bitmaps are not built: the solve reads neither.
           Only olev survives, because TO[] and the A chain are computed from
           the outer row's levels.  The bm calloc that used to sit above this
           comment went with them: it was never written or read, only allocated
           and freed, at H * NLEV * words words.  That is 25 GB of untouched
           address space at n=67 and 114 GB at n=71, where it would likely be
           refused outright.  Being calloc and never touched it cost no resident
           memory, which is why it survived so long. */
        int8_t *mlev = NULL; (void)mlev;


#ifdef GC_ABL
        /* Ablation stops.  -DGC_ABL=k runs the first k stages of the per-row
           work and nothing after, which is how the stage shares are measured.
           The sink rides the existing reduction so the compiler cannot delete
           the work being timed. */
#pragma omp parallel reduction(+:dfound,dtested,dsink)
#else
#pragma omp parallel reduction(+:dfound,dtested)
#endif
        {
        /* Scratch for the per-outer compaction, one set per thread.  The outer
           value alone rules out most of the partner table, measured at 92% of
           it at n=39, so the chain is run against a compacted copy of what
           survives rather than against the whole table.  That shrinks the word
           count every pair scans by more than an order of magnitude, and the
           compaction is amortised over the hundreds of skew rows that share
           the outer value. */
        /* The compacted list holds well under an eighth of the table in
           practice, and both scratch arrays grow if an outer value needs more,
           so they are sized from that rather than from the table. */
        int ccap = mid_t->cnt / 8 + 128;
        int cwmax = (ccap + 63) >> 6;
        int cicap = ccap;
        int *cidx = malloc((size_t)cicap * sizeof(int));
#define GC_CHUNK 32
#ifndef GC_APF
#define GC_APF 8
#endif
#ifndef GC_RPF
#define GC_RPF 8
#endif
        /* Survivors now accumulate across every block of the tile rather
           than one block at a time, so the buffer is sized to the tile and
           grown on demand. */
        int sbcap = 4096, *sbuf = malloc((size_t)sbcap * sizeof(int));
        if (!sbuf) { fprintf(stderr, "out of memory\n"); exit(1); }
        int ascap = 1024, *asurv = malloc((size_t)ascap * sizeof(int));
        if (!asurv) { fprintf(stderr, "out of memory\n"); exit(1); }
        /* VW words of slack: the block loop reads a whole block past the end of
           a row, and those lanes are discarded rather than used. */
        uint64_t *cbm = malloc(((size_t)H * NLEV * cwmax + VW) * sizeof(uint64_t));
        uint64_t *omask = malloc((size_t)words * sizeof(uint64_t));
        uint64_t *univ = malloc(((size_t)cwmax + VW) * sizeof(uint64_t));
        if (!cidx || !cbm || !omask || !univ) { fprintf(stderr, "out of memory\n"); exit(1); }
#pragma omp for schedule(dynamic)
        for (int io = 0; io < out_t->cnt; io++) {
            if (CKPT_DONE && ((CKPT_DONE[io >> 6] >> (io & 63)) & 1)) continue;
            /* Progress.  A run prints one line per row-sum decomposition, and
               n=53 has two, so the first sign of life arrives near the halfway
               mark and n=55's three would take most of a day.  A periodic line
               makes a multi-day run observable instead of a black box; it is
               one atomic increment against an outer loop whose body scans
               thousands of words. */
            if ((__atomic_add_fetch(&PROGRESS, 1, __ATOMIC_RELAXED)
                 % PROGRESS_EVERY) == 0)
#pragma omp critical
                fprintf(stderr, "   (%d,%d,%d): %d of %d outer rows, "
                        "%.0f%%"
#ifdef GC_VERIFY
                        "  [verify ok %llu bad %llu]"
#endif
                        "\n", s1, s2, s3, PROGRESS, out_t->cnt,
                        100.0 * PROGRESS / out_t->cnt
#ifdef GC_VERIFY
                        , VF_OK, VF_BAD
#endif
                        );
            int16_t pObuf[80];
            psd_from_paf(out_t->paf + (size_t)io * H, pObuf);
            const int16_t *pO = pObuf;
            const int8_t *fO = out_t->paf + (size_t)io * H;
            int8_t suO[80]; gsw rbO;
            gc_brow(fO, suO, &rbO);           /* finding 458, once per outer row */
#ifdef GC_TSTAT
            {
#pragma omp atomic
                TS_ROWS++;
                uint64_t w = (uint64_t)rbO >> 6, b = 1ULL << (rbO & 63);
                if (!(__atomic_fetch_or(&TS_RBSEEN[w], b, __ATOMIC_RELAXED) & b))
#pragma omp atomic
                    TS_RBDIST++;
            }
#endif
            const int8_t *lvO = olev + (size_t)io * H;
            int8_t want[64];
            memset(want, 0, sizeof want);
            /* Test frequencies most-constrained first.  The bitmap chain and
               the exact cap test below both stop on the first failure, so the
               order decides how many of each the average rejected candidate
               costs.  The outer value's own spectrum says which frequencies
               leave the least room, and that ordering is fixed for the whole
               outer iteration. */
            int forder[64];
            for (int k = 0; k < H; k++) forder[k] = k;
            for (int i = 1; i < H; i++) {
                int v = forder[i], j = i - 1;
                while (j >= 0 && pO[forder[j]] < pO[v]) { forder[j + 1] = forder[j]; j--; }
                forder[j + 1] = v;
            }
            /* Ceiling budget, finding 327.  The exact cap test is
               pO[k] + pA[k] > CEIL16, which since both spectra lie in
               [0, CEIL16] and CEIL16 is 64n rewrites exactly as
               pA[k] > CEIL16 - pO[k] with no overflow, and that right side
               depends only on the outer row.  Finding 327 measured the
               early exit reaching 19.01 of H=21 on average, so the ordered
               scalar walk was running nearly to completion anyway and there
               is nothing to lose by testing all H at once. */
            int16_t bud[64];
            for (int k = 0; k < H; k++) bud[k] = (int16_t)(CEIL16 - (int)pO[k]);
#if defined(__AVX2__)
            __m256i vb0, vb1; int use_v = (H >= 16);
            if (use_v) {
                vb0 = _mm256_loadu_si256((const __m256i *)bud);
                vb1 = _mm256_loadu_si256((const __m256i *)(bud + H - 16));
            } else { vb0 = _mm256_setzero_si256(); vb1 = vb0; }
#endif
            double _c0 = cpunow();
            /* The compaction is gone with the chain it fed. */

            double _rb = 0.0, _pr = 0.0, _c1_first = 0.0;
            /* TO[k] and the A chain depend only on the outer row, so they are
               hoisted out of the tile loop rather than recomputed per tile. */
            int TO[64];
            for (int k = 0; k < H; k++) TO[k] = NLEV - 1 - lvO[k];
            const uint64_t *ap[64];
            int nap = 0, askip = 0;
            for (int q = 0; q < H; q++) {
                int k = forder[q];
                int t = NLEV - 1 - lvO[k];
                if (t >= NLEV - 1) continue;
                if (t < 0) { askip = 1; break; }
                ap[nap++] = abm + ((size_t)k * NLEV + t) * awords;
            }
            if (askip) continue;
            /* The A chain and the (O, A) ceiling test depend on the outer row
               and the candidate, never on the tile, so the survivors are found
               once here rather than rescanned for every tile.  Untiled this
               was free; tiled it is not, because the scan is nap * awords word
               operations and the test is O(H) per survivor, and both would
               repeat once per tile.  At the budget finding 260 derives for 14
               threads the tile is small and the tile count is correspondingly
               large, which is the regime this protects. */
#ifdef GC_FILTSTAT
#pragma omp atomic
            FS_OUTER++;
#pragma omp atomic
            FS_PAIRS += (long long)A.cnt;
#endif
            int nasurv = 0;
#ifdef GC_FILTSTAT
#pragma omp atomic
            FS_NAP += nap;
#pragma omp atomic
            FS_WORDS += awords;
            long long _ands = 0;
#endif
            /* Finding 508: the AND chain does not early-exit.  Measured
               at n=45 it runs 15.03 of nap=18.2 masks per word, so the exit
               saves 1.2x, and what is left is a serial dependent chain of
               scalar ANDs.  Four words go through one AVX2 register instead,
               in chunks small enough to keep the chunk buffer in L1, with the
               bit extraction reading the chunk afterwards.  The per-word exit
               is given up (1.2x) and the width returns 4x.  The masks a single
               outer row touches are nap x awords, 92 KB at n=45, so the chain
               is L2-resident and instruction count, not bandwidth, sets it. */
            enum { FCH = 256 };
            uint64_t andbuf[FCH];
            for (int fbase = 0; fbase < awords; fbase += FCH) {
                int fnw = awords - fbase < FCH ? awords - fbase : FCH;
                int fw = 0;
                if (nap == 0) {
                    for (; fw < fnw; fw++) andbuf[fw] = ~0ULL;
                } else {
#if defined(__AVX2__)
                    for (; fw + 4 <= fnw; fw += 4) {
                        __m256i fv = _mm256_loadu_si256(
                            (const __m256i *)(ap[0] + fbase + fw));
                        for (int c = 1; c < nap; c++)
                            fv = _mm256_and_si256(fv, _mm256_loadu_si256(
                                (const __m256i *)(ap[c] + fbase + fw)));
                        _mm256_storeu_si256((__m256i *)(andbuf + fw), fv);
#ifdef GC_FILTSTAT
                        _ands += 4 * (long long)nap;
#endif
                    }
#endif
                    for (; fw < fnw; fw++) {
                        uint64_t v = ~0ULL;
                        for (int c = 0; c < nap && v; c++) {
                            v &= ap[c][fbase + fw];
#ifdef GC_FILTSTAT
                            _ands++;
#endif
                        }
                        andbuf[fw] = v;
                    }
                }
                /* atail is applied after the chain rather than before it;
                   AND is commutative so the survivor set is identical. */
                if (fbase + fnw == awords) andbuf[fnw - 1] &= atail;
                for (int fwi = 0; fwi < fnw; fwi++) {
                    uint64_t av = andbuf[fwi];
                    while (av) {
                        int ia = ((fbase + fwi) << 6) + __builtin_ctzll(av);
                        av &= av - 1;
#ifdef GC_FILTSTAT
#pragma omp atomic
                        FS_AFTER_BITMAP++;
#endif
#ifdef GC_ABL
                        dsink += ia;
                        if (GC_ABL <= 1) continue;
#endif
#ifndef GC_NOCEIL
                        /* -DGC_NOCEIL drops the exact (O, A) ceiling test and
                           sends every bitmap survivor to the solve.  The test
                           rejects only 13.8% of what the bitmap already passed
                           (finding 508) and hkt_prefix rejects the same
                           candidates, so this asks whether running it is worth
                           its 627.7M random reads.  Output must be identical
                           either way; only the cost moves. */
                        const int16_t *pA = A.psd + (size_t)ia * H;
#if defined(__AVX2__)
                        if (use_v) {
                            /* Two sixteen-lane loads overlapping in the middle
                               cover all H without reading past the row. */
                            __m256i x0 = _mm256_loadu_si256((const __m256i *)pA);
                            __m256i x1 = _mm256_loadu_si256((const __m256i *)(pA + H - 16));
                            __m256i bad = _mm256_or_si256(_mm256_cmpgt_epi16(x0, vb0),
                                                          _mm256_cmpgt_epi16(x1, vb1));
                            if (!_mm256_testz_si256(bad, bad)) continue;
                        } else
#endif
                        {
                            int ok = 1;
                            for (int q = 0; q < H; q++) {
                                int k = forder[q];
                                if ((int)pO[k] + (int)pA[k] > CEIL16) { ok = 0; break; }
                            }
                            if (!ok) continue;
                        }
#endif  /* GC_NOCEIL */
                        if (nasurv == ascap) {
                            ascap *= 2;
                            asurv = realloc(asurv, (size_t)ascap * sizeof(int));
                            if (!asurv) { fprintf(stderr, "out of memory\n"); exit(1); }
                        }
#ifdef GC_ABL
                        if (GC_ABL <= 2) { dsink += 1; continue; }
#endif
#ifdef GC_FILTSTAT
#pragma omp atomic
                        FS_AFTER_CEIL++;
#endif
                        asurv[nasurv++] = ia;
                    }
                }
            }
#ifdef GC_FILTSTAT
#pragma omp atomic
            FS_ANDS += _ands;
#endif
            double _c1s = cpunow();
#pragma omp atomic
            PH_SURV += _c1s - _c0;
            if (!nasurv) continue;
            /* HKT solve (finding 300) replaces the tile, chain, survivor
               and hash machinery.  The (outer, A) pairs walked here are exactly
               the (B, A) pairs the method enumerates; instead of scanning a
               third table and hashing for the fourth, the two remaining
               symmetric rows are solved for.  Everything above is unchanged:
               table build, decimation canonicalisation, outer loop, A ceiling
               filter. */
            /* Eight-lane batch, finding 439.  Finding 438 closed vectorising
               the elimination across columns: H is 21 at n=43 and 29 at n=59,
               so the column loop is only two or three vector iterations plus a
               scalar tail and the per-pivot setup is paid H times against that.
               Vectorising across lanes instead fills the register whatever H
               is.  The survivors of one outer row are independent solves, so
               eight go through hkt_elim8 together with the columns stored
               transposed, c[k*8+L], and _mm256_srlv_epi32 supplying a
               different pivot bit per lane.  Lanes whose prefix rejects, and
               the short tail of a partial batch, carry R = 0 and compute
               nothing.  Solutions are still emitted in ai order. */
            gsw bcand[8][HKT_MAXC], bsb[8]; int bncd[8];
#ifdef GC_SOLVESPLIT
            unsigned long long _acc_rb = 0, _acc_el = 0, _acc_tl = 0;
            unsigned long long _acc_all = GC_TICK();
#endif
            for (int abase = 0; abase < nasurv; abase += 8) {
                int nb = nasurv - abase; if (nb > 8) nb = 8;
                for (int aj = 0; aj < nb; aj++) { bncd[aj] = 0; bsb[aj] = 0; }
#if defined(__AVX2__) && !defined(GC_NOELIM8)
                {
                    gsw ct[64 * 8], bbv[8], sv[8];
                    int lane_of[8], nl = 0;
#ifdef GC_SOLVESPLIT
                    unsigned long long _r0 = GC_TICK();
#endif
                    for (int aj = 0; aj < nb; aj++) {
                        int ia_ = asurv[abase + aj];
                        gsw c0[64]; int16_t mcb[64];
                        gsw s0 = ARA[ia_] ^ rbO;           /* finding 458 */
#ifdef GC_TSTAT
                        {
#pragma omp atomic
                            TS_CALLS++;
                            uint64_t w = (uint64_t)s0 >> 6, b = 1ULL << (s0 & 63);
                            uint64_t old_;
                            old_ = __atomic_fetch_or(&TS_SEEN[w], b, __ATOMIC_RELAXED);
                            if (!(old_ & b))
#pragma omp atomic
                                TS_DISTINCT++;
                        }
#endif
                        gc_rows_from_t(s0, mcb, c0);
                        gsw b0 = gc_bvec(mcb, ATA + (size_t)ia_ * ATAW, suO);
                        for (int k = 0; k < H; k++) ct[(size_t)k * 8 + nl] = c0[k];
                        bbv[nl] = b0; sv[nl] = s0; lane_of[nl] = aj; nl++;
                    }
#ifdef GC_SOLVESPLIT
                    _acc_rb += GC_TICK() - _r0;
#endif
#ifdef GC_SOLVESTAT
#pragma omp atomic
                    SS_PREFIX += nb;
#pragma omp atomic
                    SS_REJECT += nb - nl;
                    if (nl) { _Pragma("omp atomic") SS_ELIM2++; }
#endif
                    if (nl) {
                        gsw pb8[64 * 8], pm8[8];
#ifdef GC_VERIFY
                        gsw ct0[64 * 8], bb0v[8];
                        for (int k = 0; k < H * 8; k++) ct0[k] = ct[k];
                        for (int Lx = 0; Lx < 8; Lx++) bb0v[Lx] = bbv[Lx];
#endif
                        for (int Lx = nl; Lx < 8; Lx++) {
                            bbv[Lx] = 0;
                            for (int k = 0; k < H; k++) ct[(size_t)k * 8 + Lx] = 0;
                        }
#ifdef GC_SOLVESPLIT
                        unsigned long long _e0 = GC_TICK();
#endif
                        hkt_elim8v(ct, bbv, pb8, pm8);
#ifdef GC_SOLVESPLIT
                        unsigned long long _e1 = GC_TICK();
                        _acc_el += _e1 - _e0;
                        unsigned long long _t0 = _e1;
#endif
                        for (int Lx = 0; Lx < nl; Lx++) {
                            int aj = lane_of[Lx];
                            bsb[aj] = sv[Lx];
                            /* Finding 450.  The consistency test needs only
                               the reduced right-hand side and the pivot mask,
                               both already in registers, and finding 440
                               measured it killing 82% of systems.  Testing it
                               before the de-transpose spares those lanes 2H
                               strided loads out of the transposed batch. */
#ifdef GC_TSTAT
                            {   int rk = __builtin_popcount(pm8[Lx]);
                                if (rk > 39) rk = 39;
#pragma omp atomic
                                RK_HIST[rk]++;
#pragma omp atomic
                                RK_TOTAL++;
                                {   uint32_t np = ~pm8[Lx] & ((H>=32)?0xffffffffu:((1u<<H)-1u));
                                    uint64_t w = (uint64_t)np >> 6, bb = 1ULL << (np & 63);
                                    if (!(__atomic_fetch_or(&PM_SEEN[w], bb, __ATOMIC_RELAXED) & bb))
#pragma omp atomic
                                        PM_DIST++;
                                }
                                if (bbv[Lx] & ~pm8[Lx])
#pragma omp atomic
                                    RK_INCONS++;
                            }
#endif
                            if (bbv[Lx] & ~pm8[Lx]) { bncd[aj] = 0; continue; }
                            gsw colL[64], pivL[64];
                            for (int k = 0; k < H; k++) {
                                colL[k] = ct[(size_t)k * 8 + Lx];
                                pivL[k] = pb8[(size_t)k * 8 + Lx];
                            }
                            bncd[aj] = hkt_tail_m(colL, bbv[Lx], pm8[Lx], pivL,
                                                  bcand[aj]);
#ifdef GC_VERIFY
                            {   gsw c0v[64];
                                for (int k = 0; k < H; k++) c0v[k] = ct0[(size_t)k * 8 + Lx];
                                gc_verify(c0v, bb0v[Lx], bcand[aj], bncd[aj]); }
#endif
                        }
#ifdef GC_SOLVESPLIT
                        _acc_tl += GC_TICK() - _t0;
#endif
                    }
                }
#else
                for (int aj = 0; aj < nb; aj++) {
#ifdef GC_SOLVESTAT
#pragma omp atomic
                    SS_SINGLE++;
#endif
                    bncd[aj] = hkt_solve(A.paf + (size_t)asurv[abase + aj] * H,
                                         fO, bcand[aj], &bsb[aj]);
                }
#endif
                for (int aj = 0; aj < nb; aj++) {
                    int ia = asurv[abase + aj];
                    const gsw *cand = bcand[aj];
                    gsw sbits = bsb[aj];
                    int ncd = bncd[aj];
                dtested += ncd;
#ifdef GC_ABL
                if (GC_ABL <= 3) { dsink += ncd; continue; }
#endif
                gsw maskH = GSW_HMASK;
                const int8_t *pafA_ = A.paf + (size_t)ia * H;
                for (int ci = 0; ci < ncd; ci++) {
                    gsw codeC = (~cand[ci]) & maskH;
                    gsw codeD = (~(sbits ^ codeC)) & maskH;
                    /* The solve works modulo 4, so a lift need not be a real
                       solution: verify against the definition before believing
                       it.  Finding 320: PAF from popcount on the bit words, and
                       the A and B PAFs read straight off the solve's arguments
                       instead of being rebuilt per candidate. */
                    gcw wc = hkt_symword(codeC), wd = hkt_symword(codeD);
                    int good = 1;
                    for (int k = 1; k <= H && good; k++) {
                        int t2 = (int)pafA_[k - 1] + (int)fO[k - 1] + 2 * N
                               - 2 * (GCW_POP(wc ^ hkt_rotk(wc, k))
                                    + GCW_POP(wd ^ hkt_rotk(wd, k)));
                        if (t2 != 0) good = 0;
                    }
                    if (!good) continue;
                    dfound++;
#pragma omp critical
                    {
                        if (nsol == solcap) {
                            solcap = solcap ? solcap * 2 : 1024;
                            sol = realloc(sol, (size_t)solcap * 4 * sizeof(long long));
                            if (!sol) { fprintf(stderr, "out of memory\n"); exit(1); }
                        }
                        long long *r = sol + (size_t)nsol * 4;
                        r[0] = A.code[ia];
                        r[1] = out_t->code[io];
                        r[2] = (long long)codeC;
                        r[3] = (long long)codeD;
                        nsol++;
                    }
                }
            }
            }
#ifdef GC_SOLVESPLIT
            { unsigned long long _all = GC_TICK() - _acc_all;

#pragma omp atomic
              PH_RB += (double)_acc_rb;
#pragma omp atomic
              PH_EL += (double)_acc_el;
#pragma omp atomic
              PH_TL += (double)_acc_tl;
#pragma omp atomic
              PH_ALL += (double)_all; }
#endif
            { double _c2x = cpunow(); _c1_first = _c2x; _rb += 0.0; _pr += 0.0;
#pragma omp atomic
              PH_SOLVE += _c2x - _c1s; }
            if (CKPT_DONE) {
                __atomic_or_fetch(&CKPT_DONE[io >> 6], 1ULL << (io & 63),
                                  __ATOMIC_RELAXED);
                /* The unnamed critical is the same lock the solution append
                   uses, so the write sees a consistent sol/nsol. */
                if (wnow() - CKPT_LAST > CKPT_EVERY)
#pragma omp critical
                {   if (wnow() - CKPT_LAST > CKPT_EVERY) {
                        ckpt_write(found + dfound, tested + dtested, nsol, sol);
                        CKPT_LAST = wnow();
                    }
                }
            }
            {
#pragma omp atomic
                PH_COMPACT += _c1_first - _c0;
#pragma omp atomic
                PH_REBUILD += _rb;
#pragma omp atomic
                PH_PAIR += _pr;
            }
        }
        free(cidx); free(asurv); free(sbuf); free(cbm); free(omask); free(univ);
        }
        free(mlev);
        table_free(&T1);
        }   /* end outer-table slice */
        found += dfound; tested += dtested;
#ifdef GC_ABL
        fprintf(stderr, "GC_ABL=%d sink=%lld\n", GC_ABL, dsink);
#endif
        if (CKPT) {
            CKPT_DI = di + 1; CKPT_OC = 0;
            free(CKPT_DONE); CKPT_DONE = NULL; CKPT_NROW = 0;
            ckpt_write(found, tested, nsol, sol);
            CKPT_LAST = wnow();
        }
        printf("   (%d,%d,%d): %lld quadruples, %lld triples tested\n",
               s1, s2, s3, dfound, dtested);
    }

    free(abm); free(alev); free(ARA); free(ATA);

    /* Expand each solution over the decimation maps and drop repeats.  A
       quadruple whose skew row has a nontrivial stabiliser has a short orbit,
       so this deduplicates rather than assuming every orbit is full. */
    long long *all = malloc((size_t)nsol * NT * 4 * sizeof(long long));
    if (nsol && !all) { fprintf(stderr, "out of memory\n"); exit(1); }
    int nall = 0;
    for (int i = 0; i < nsol; i++)
        for (int t = 0; t < NT; t++) {
            long long *r = all + (size_t)nall * 4;
            r[0] = decimate(sol[(size_t)i * 4 + 0], t, 1);
            for (int u = 1; u < 4; u++)
                r[u] = decimate(sol[(size_t)i * 4 + u], t, 0);
            nall++;
        }
    int cmp_quad(const void *a, const void *b) {
        const long long *x = a, *y = b;
        for (int u = 0; u < 4; u++) if (x[u] != y[u]) return x[u] < y[u] ? -1 : 1;
        return 0;
    }
    qsort(all, (size_t)nall, 4 * sizeof(long long), cmp_quad);
    int nuniq = 0;
    for (int i = 0; i < nall; i++)
        if (i == 0 || cmp_quad(all + (size_t)i * 4, all + (size_t)(i - 1) * 4)) {
            if (nuniq != i) memcpy(all + (size_t)nuniq * 4, all + (size_t)i * 4,
                                   4 * sizeof(long long));
            nuniq++;
        }
    if (f) {
        int8_t x[4][256];
        for (int i = 0; i < nuniq; i++) {
            skew_seq(all[(size_t)i * 4 + 0], x[0]);
            for (int u = 1; u < 4; u++) sym_seq(all[(size_t)i * 4 + u], x[u]);
            for (int u = 0; u < 4; u++) {
                for (int j = 0; j < N; j++)
                    fputc(x[u][j] > 0 ? '+' : '-', f);
                fputc(u == 3 ? '\n' : ' ', f);
            }
        }
    }
    free(all); free(sol);
    if (f) fclose(f);
    {   double tot = PH_TABLE + PH_COMPACT;
        /* Two columns, not four.  The solve replaced the chain, so there is no
           bitmap rebuild and no pair loop to report, and printing them as 0%
           beside a "compaction" that holds 95% of the run described a program
           that no longer exists.  Splitting the per-row time into the A ceiling
           filter and the solve needs one more clock read per outer row, which
           did not resolve against this machine's noise floor over eight
           order-alternated pairs; finding 367 records the split, measured with
           a temporary instrument, as 5% filter and 92% solve at n=43. */
        if (tot > 0) printf("   thread-seconds by phase: table build %.3f (%.0f%%) | "
               "per-row work %.3f (%.0f%%)\n",
               PH_TABLE, 100*PH_TABLE/tot, PH_COMPACT, 100*PH_COMPACT/tot);
#ifdef GC_TSTAT
    fprintf(stderr, "   rebuild inputs: %lld calls, %lld distinct t of 2^%d,"
            " reuse %.1fx\n", TS_CALLS, TS_DISTINCT, H,
            TS_DISTINCT ? (double)TS_CALLS / TS_DISTINCT : 0.0);
    fprintf(stderr, "   outer rows: %lld, distinct rbO %lld, rows per rbO %.2f\n",
            TS_ROWS, TS_RBDIST, TS_RBDIST ? (double)TS_ROWS / TS_RBDIST : 0.0);
    fprintf(stderr, "   systems %lld, inconsistent %lld (%.1f%%); rank histogram:",
            RK_TOTAL, RK_INCONS, 100.0 * RK_INCONS / (RK_TOTAL ? RK_TOTAL : 1));
    for (int i = 0; i < 40; i++)
        if (RK_HIST[i]) fprintf(stderr, " %d:%.1f%%", i,
                                100.0 * RK_HIST[i] / (RK_TOTAL ? RK_TOTAL : 1));
    fprintf(stderr, "  (H=%d)\n", H);
    fprintf(stderr, "   distinct non-pivot masks: %lld\n", PM_DIST);
#endif
#ifdef GC_SOLVESPLIT
        if (PH_ALL > 0)
            fprintf(stderr, "   solve split (tsc): rebuild %.0f%%  elim %.0f%%"
                    "  tail %.0f%%  candidates %.0f%%\n",
                    100*PH_RB/PH_ALL, 100*PH_EL/PH_ALL, 100*PH_TL/PH_ALL,
                    100*(PH_ALL-PH_RB-PH_EL-PH_TL)/PH_ALL);

#endif
        if (PH_SURV + PH_SOLVE > 0)
            printf("   per-row split: A filter + survivors %.3f | solve %.3f "
                   "(continue rows counted in the filter half)\n",
                   PH_SURV, PH_SOLVE);
    }
    printf("\nn=%d: %d good-matrix quadruples (%lld on canonical skew rows, "
           "expanded over %d decimations)\n", N, nuniq, found, NT);
    printf("   triples tested: %lld\n", tested);
#ifdef GC_FILTSTAT
    fprintf(stderr, "filter: %lld outer rows x A table = %lld pairs\n"
            "        %lld survive the bitmap chain (%.4f%%)\n"
            "        %lld survive the ceiling test  (%.4f%% of pairs, %.1f%% of chain)\n",
            FS_OUTER, FS_PAIRS, FS_AFTER_BITMAP,
            100.0 * FS_AFTER_BITMAP / (FS_PAIRS ? FS_PAIRS : 1), FS_AFTER_CEIL,
            100.0 * FS_AFTER_CEIL / (FS_PAIRS ? FS_PAIRS : 1),
            100.0 * FS_AFTER_CEIL / (FS_AFTER_BITMAP ? FS_AFTER_BITMAP : 1));
    fprintf(stderr, "filter chain: %lld word ANDs over %lld words offered, "
            "%.2f per word, against nap %.1f masks average (so the early exit "
            "saves %.1fx)\n",
            FS_ANDS, FS_WORDS, (double)FS_ANDS / (FS_WORDS ? FS_WORDS : 1),
            (double)FS_NAP / (FS_OUTER ? FS_OUTER : 1),
            (double)FS_NAP / (FS_OUTER ? FS_OUTER : 1)
              / ((double)FS_ANDS / (FS_WORDS ? FS_WORDS : 1)));
#endif
#ifdef GC_SOLVESTAT
    fprintf(stderr, "solve: %lld prefix calls, %lld rejected there (%.1f%%), "
            "%lld two-lane eliminations, %lld single-lane solves\n",
            SS_PREFIX, SS_REJECT, 100.0 * SS_REJECT / (SS_PREFIX ? SS_PREFIX : 1),
            SS_ELIM2, SS_SINGLE);
#endif
    table_free(&A);
    free(COS);
#ifdef GC_VERIFY
    fprintf(stderr, "   verify: %llu candidates satisfy the system, %llu do not\n",
            VF_OK, VF_BAD);
#endif
#ifdef GC_TARANGE
    fprintf(stderr, "   ta range [%d, %d]  (int8_t holds [-128,127])\n", TA_MIN, TA_MAX);
#endif
    return 0;
}
