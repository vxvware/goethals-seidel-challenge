/* best_hkt.c -- circulant best matrices: exhaustive census up to equiv-v1 by
 * a mod-4 linearisation (the Holzmann-Kharaghani-Tayfeh-Rezaie idea) adapted
 * to two unknown skew rows.
 *
 *   build: gcc -O3 -march=native -fopenmp best_hkt.c -lm -o best_hkt
 *          (add -DWIDE for n > 63: 128-bit rows, 64-bit code words)
 *   run:   ./best_hkt <n> <threads> <out.txt>
 *   out:   one line per solution found, four words of n '+'/'-' (A B C D),
 *          A, B, C skew, D symmetric.  The lines are representatives, not
 *          yet reduced to classes: canonicalise them to count classes.
 *   env:   BEST_NLEV=l      levels of the quantised-spectrum index (default: auto)
 *          BEST_PIECES=p    split the A table into p code ranges (default: auto)
 *          BEST_MEM_MB=m    budget for one A piece, records + index (default 1000)
 *          BEST_CC_MB=m     budget for the consistency table (default 3000)
 *          BEST_CCK=k       words per pattern in that table (0 = no table)
 *          BEST_NOMAX=1     drop reduction R4 below (A need not be the peak row)
 *          BEST_NODEC=1     drop reduction R2 below (every D, not one per orbit)
 *          BEST_PROBE=a,d   sampling probe, NOT a census: a random A codes and
 *                           d random D codes through the same filters and solve,
 *                           with the pass rates and the cost per pair printed.
 *          BEST_PLANT=file  recovery test, NOT a census: file holds known
 *                           quadruples (four +/- words per line, any row order
 *                           and sign).  D is restricted to the orbit minima of
 *                           their symmetric rows and A to the multiplier images
 *                           and reversals of their skew rows; everything else
 *                           runs unchanged, so each planted class must come out.
 *
 * Problem.  n odd, A, B, C skew (x_0 = 1, x_{n-j} = -x_j), D symmetric,
 * PAF_A(k) + PAF_B(k) + PAF_C(k) + PAF_D(k) = 0 for k = 1..n-1.  PAF is even
 * in k, so k = 1..H with H = (n-1)/2 suffices.  A skew row is coded by H bits
 * y (bit u-1 set iff x_u = -1), a symmetric row by H+1 bits (bit u set iff
 * d_u = -1, u = 0..H).
 *
 * Method.  D and A are enumerated; B and C are solved for.
 *
 *   For a skew row X the terms j = 0 and j = -k of PAF_X(k) cancel, the term
 *   j = -k/2 is -1, and the other n-3 terms pair off under j -> -j-k into H-1
 *   equal pairs P_k:  PAF_X(k) = -1 + 2 sum_{P_k} x_j x_{j+k}.
 *   Every PAF of a +-1 row of odd length n is n - 4u with u >= 0 an integer;
 *   write uA, uB, uC, uD for the four rows at lag k.  The equation at lag k is
 *   uA + uB + uC + uD = n, an identity between integers.
 *
 *   Stage 1 (exact, necessary).  With beta, gamma the sign bits of B, C and
 *   sigma = beta xor gamma (symmetric in j, so H bits t), the equation mod 2
 *   after dividing by 4 reads  M1 t = r,  r_k = 1 + uA_k + uD_k (mod 2).
 *   M1 depends on n only: row k has ones at m(k) and m(k/2), so M1 = I + P for
 *   the halving permutation P and its corank is the number c of cycles of
 *   doubling on (Z_n minus 0)/+-1.  Hence c parity conditions on (A, D) (the
 *   "signature"), and 2^c patterns t for a pair that meets them.
 *
 *   Stage 2 (exact, necessary).  Given t, the pairs of P_k with sigma_j !=
 *   sigma_{j+k} contribute 0 to PAF_B + PAF_C and the others contribute
 *   4 b_j b_{j+k}.  Dividing by 4 and reducing mod 2 gives a GF(2) system
 *   M2(t) y = G(t) xor X(A, D) in the H code bits y of B, and C = y xor t.
 *   Every solution of that system is then checked against the integer
 *   equations uA + uB + uC + uD = n at all H lags, so nothing false is kept.
 *
 * Both stages are implications of the integer equations, so a solution
 * (A, B, C, D) with A and D enumerated is always among the candidates of
 * stage 2 for the pattern t = code(B) xor code(C).  The only way the solve
 * can lose it is the cap on the stage-2 nullity (MAXFREE free variables);
 * every system dropped by the cap is counted and reported, and the run is
 * exhaustive only when that count is zero.
 *
 * Exhaustiveness up to equiv-v1 (negate any row, reverse any row, permute
 * rows, one multiplier j -> tj on all four rows).  Take any best quadruple.
 * Its symmetric row is D, the other three are skew after negation where
 * needed (a skew row is fixed in sign by x_0 = 1).
 *
 *   R1  row sum of D.  Skew rows have sum 1, so sum(D)^2 = 4n-3.  Negating D
 *       makes sum(D) = +sqrt(4n-3).  Only such D are enumerated.
 *   R2  multiplier.  A multiplier maps skew rows to skew rows and symmetric
 *       rows to symmetric rows and keeps row sums, so it may be used to move
 *       D to the least code of its multiplier orbit.  Only such D are kept.
 *       (t and -t act alike on D, so units up to sign are enough.)
 *   R3  spectrum.  PSD_A + PSD_B + PSD_C + PSD_D = 4n at every frequency and
 *       a skew row has PSD = 1 + (real)^2 >= 1.  So PSD_D <= 4n-3 and
 *       PSD_A <= 4n-2-PSD_D are necessary.  These are tested in floating
 *       point, always against a bound relaxed by 1e-6 and, for the A index,
 *       rounded to a coarser level in the accepting direction; the tests can
 *       pass a pair that fails, never fail one that passes.
 *   R4  which skew row is A.  Let k* be the frequency where PSD_D is least
 *       (any rule depending on D only would do).  The three skew spectra at
 *       k* sum to 4n - PSD_D(k*), so the largest is at least a third of that.
 *       Permuting the skew rows puts that row first: A may be restricted to
 *       PSD_A(k*) >= (4n - PSD_D(k*))/3, again relaxed by 1e-6.
 *   R5  reversal of A.  Reversing a skew row complements its code and keeps
 *       its PAF and PSD, so R3 and R4 are unaffected.  Only A with a_1 = +1
 *       (code bit 0 clear) are tabulated.
 *   R6  signature.  Pairs (A, D) failing the c parity conditions of stage 1
 *       have no completion; the A table is sorted by signature and D scans
 *       the one matching range.
 *   R7  reversal of C alone complements t.  Only patterns with t_1 = 0 are
 *       solved; the kernel of M1 contains the all-ones vector, so exactly
 *       half of the 2^c patterns qualify and the other half are the
 *       reversals of C.
 *   R8  reversal of B and C together complements y and keeps t.  Only
 *       y_1 = 0 (b_1 = +1) is emitted.
 *   R9  swap of B and C maps y to y xor t and keeps t.  Only y <= y xor t is
 *       emitted.
 *
 * R1, R2 use negation of D and a multiplier; R4, R9 a row permutation; R5,
 * R7, R8 row reversals.  All are equiv-v1 moves, each normalisation is made
 * after the previous ones and preserves them (R5, R7, R8, R9 do not touch D;
 * R7, R8, R9 do not touch A; R8 keeps t; R9 keeps t and, because t_1 = 0,
 * keeps y_1), so every class has a representative that satisfies all of them,
 * and that representative is found by the solve.  No floating-point test and
 * no cap other than the counted nullity cap can remove it.
 *
 * Self-checks at startup: the stage-1 and stage-2 identities on random skew
 * pairs against integer PAFs, the closed-form column build against the row
 * build, the kernel and particular solution of M1, the multiplier tables.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WIDE
typedef unsigned __int128 uw;
typedef uint64_t cw;
#define CWB 64
static inline int popu(uw x) { return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64)); }
static inline int parc(cw x) { return __builtin_parityll(x); }
static inline int popc(cw x) { return __builtin_popcountll(x); }
static inline int ctzc(cw x) { return __builtin_ctzll(x); }
#define NMAX 127
#else
typedef uint64_t uw;
typedef uint32_t cw;
#define CWB 32
static inline int popu(uw x) { return __builtin_popcountll(x); }
static inline int parc(cw x) { return __builtin_parity(x); }
static inline int popc(cw x) { return __builtin_popcount(x); }
static inline int ctzc(cw x) { return __builtin_ctz(x); }
#define NMAX 63
#endif
#define CC_FLAG ((cw)1 << (CWB - 1))
#define HMAX 64
#ifndef MAXFREE
#define MAXFREE 20
#endif

static int N, H;
static uw FULL;
static cw MASKH;

static double wnow(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static double tcpu(void) { struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }

/* bit j of rotr(x,k) is x_{j+k}; bit j of rotl(x,k) is x_{j-k} (indices mod n) */
static inline uw rotr(uw x, int k) { return ((x >> k) | (x << (N - k))) & FULL; }
static inline uw rotl(uw x, int k) { return ((x << k) | (x >> (N - k))) & FULL; }
static inline uint64_t rev64(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
    return __builtin_bswap64(x);
}
static inline cw revH(cw x) { return (cw)(rev64((uint64_t)x) >> (64 - H)); }
/* rows as n-bit words, bit j set iff x_j = -1 */
static inline uw skew_word(cw y) { return ((uw)y << 1) | ((uw)revH((cw)(~y & MASKH)) << (H + 1)); }
static inline uw symh_word(cw t) { return ((uw)t << 1) | ((uw)revH(t) << (H + 1)); }   /* bit 0 clear */
static inline uw sym_word(uint64_t code) { return (uw)(code & 1) | symh_word((cw)(code >> 1)); }
/* u_k = (n - PAF(k))/4 for k = 1..H, as two bit planes and optionally the values */
static inline void uvals(uw w, int *u, cw *u0, cw *u1) {
    cw a0 = 0, a1 = 0;
    for (int k = 1; k <= H; k++) {
        int v = popu(w ^ rotr(w, k)) >> 1;
        if (u) u[k] = v;
        a0 |= (cw)(v & 1) << (k - 1);
        a1 |= (cw)((v >> 1) & 1) << (k - 1);
    }
    *u0 = a0; *u1 = a1;
}

/* ------------------------------------------------------------- stage tables */
static uw E_[HMAX + 1], HX_[HMAX + 1], EA_[HMAX + 1], EB_[HMAX + 1];
static cw M1ROW_[HMAX], LN_[HMAX], NBSTAR_, *NBCOMB_;
static int NNULL_, NCOMB_;
static cw *TBYTE_; static int NBY_;

static inline void s2_rows_ref(uw S, cw *row, cw *g) {
    cw gg = 0;
    for (int k = 1; k <= H; k++) {
        uw F = ~(S ^ rotr(S, k)) & E_[k];
        row[k - 1] = (cw)((F ^ rotl(F, k)) >> 1) & MASKH;
        int K = popu(F) >> 1, hx = popu(F & HX_[k]) >> 1;
        gg |= (cw)((((K + H) >> 1) ^ hx) & 1) << (k - 1);
    }
    *g = gg;
}
static inline cw s2_G(uw S) {
    cw gg = 0;
    for (int k = 1; k <= H; k++) {
        uw F = ~(S ^ rotr(S, k)) & E_[k];
        int K = popu(F) >> 1, hx = popu(F & HX_[k]) >> 1;
        gg |= (cw)((((K + H) >> 1) ^ hx) & 1) << (k - 1);
    }
    return gg;
}
/* column u-1 of M2(t) as a mask over lags (bit k-1) */
static inline void s2_cols(uw S, cw *col) {
    for (int u = 1; u <= H; u++) {
        uw su = (uw)0 - ((S >> u) & 1);
        uw P = S >> u, Q = rotl(S, u);
        uw X = (~(su ^ P) & EA_[u]) ^ (~(su ^ Q) & EB_[u]);
        col[u - 1] = (cw)(X >> 1) & MASKH;
    }
}
static inline cw tmap(cw r) {
    cw t = 0;
    for (int b = 0; b < NBY_; b++) t ^= TBYTE_[(size_t)b * 256 + ((r >> (8 * b)) & 0xff)];
    return t;
}
static inline cw gf2_mul(const cw *rows, cw v) {
    cw r = 0;
    for (int k = 0; k < H; k++) r |= (cw)parc(rows[k] & v) << k;
    return r;
}
static inline int sig_of(cw u0) {
    int s = 0;
    for (int c = 0; c < NNULL_; c++) s |= parc(LN_[c] & u0) << c;
    return s;
}
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}

static void die(const char *m) { fprintf(stderr, "best_hkt: n=%d: %s\n", N, m); exit(1); }

static void stage_init(void) {
    int inv2 = (N + 1) / 2;
    uw HI = 0;
    for (int j = H + 1; j < N; j++) HI |= (uw)1 << j;
    for (int k = 1; k <= H; k++) {
        int j0 = (int)(((long long)(N - k) * inv2) % N);
        E_[k] = FULL & ~((uw)1 | ((uw)1 << (N - k)) | ((uw)1 << j0));
        HX_[k] = HI ^ rotr(HI, k);
        M1ROW_[k - 1] = (cw)((E_[k] ^ rotl(E_[k], k)) >> 1) & MASKH;
    }
    for (int u = 1; u <= H; u++) {
        uw a = 0, b = 0;
        for (int k = 1; k <= H; k++) {
            if ((E_[k] >> u) & 1) a |= (uw)1 << k;
            if ((E_[k] >> (((u - k) % N + N) % N)) & 1) b |= (uw)1 << k;
        }
        EA_[u] = a; EB_[u] = b;
    }
    /* eliminate M1 with the row operations recorded */
    cw a[HMAX], id[HMAX]; int pivcol[HMAX], rank = 0;
    for (int k = 0; k < H; k++) { a[k] = M1ROW_[k]; id[k] = (cw)1 << k; }
    for (int c = 0; c < H; c++) {
        int pr = -1;
        for (int r = rank; r < H; r++) if ((a[r] >> c) & 1) { pr = r; break; }
        if (pr < 0) continue;
        cw x = a[pr]; a[pr] = a[rank]; a[rank] = x;
        x = id[pr]; id[pr] = id[rank]; id[rank] = x;
        for (int r = 0; r < H; r++)
            if (r != rank && ((a[r] >> c) & 1)) { a[r] ^= a[rank]; id[r] ^= id[rank]; }
        pivcol[rank++] = c;
    }
    NNULL_ = H - rank;
    if (NNULL_ < 1 || NNULL_ > 16) die("unexpected corank of the stage-1 matrix");
    for (int c = 0; c < NNULL_; c++) LN_[c] = id[rank + c];
    cw tv[HMAX];                                   /* tmap column for each bit of r */
    for (int b = 0; b < H; b++) {
        cw v = 0;
        for (int i = 0; i < rank; i++) if ((id[i] >> b) & 1) v |= (cw)1 << pivcol[i];
        tv[b] = v;
    }
    NBY_ = (H + 7) / 8;
    TBYTE_ = calloc((size_t)NBY_ * 256, sizeof(cw));
    for (int b = 0; b < NBY_; b++)
        for (int v = 1; v < 256; v++) {
            int low = __builtin_ctz((unsigned)v), idx = 8 * b + low;
            TBYTE_[(size_t)b * 256 + v] = TBYTE_[(size_t)b * 256 + (v & (v - 1))] ^ (idx < H ? tv[idx] : 0);
        }
    cw nb[HMAX]; int nn = 0; char ispiv[HMAX] = {0};
    for (int i = 0; i < rank; i++) ispiv[pivcol[i]] = 1;
    for (int f = 0; f < H; f++) {
        if (ispiv[f]) continue;
        cw v = (cw)1 << f;
        for (int i = 0; i < rank; i++) if ((a[i] >> f) & 1) v |= (cw)1 << pivcol[i];
        if (gf2_mul(M1ROW_, v)) die("stage-1 kernel vector is not in the kernel");
        nb[nn++] = v;
    }
    if (gf2_mul(M1ROW_, MASKH)) die("the all-ones vector is not in the stage-1 kernel");
    int star = -1;
    for (int i = 0; i < nn; i++) if (nb[i] & 1) { star = i; break; }
    if (star < 0) die("no stage-1 kernel vector moves t_1");
    NBSTAR_ = nb[star];
    cw rest[HMAX]; int nr = 0;
    for (int i = 0; i < nn; i++) {
        if (i == star) continue;
        rest[nr++] = (nb[i] & 1) ? (nb[i] ^ NBSTAR_) : nb[i];
    }
    NCOMB_ = 1 << nr;
    NBCOMB_ = malloc(sizeof(cw) * (size_t)NCOMB_);
    NBCOMB_[0] = 0;
    for (int i = 0; i < nr; i++)
        for (int m = 0; m < (1 << i); m++) NBCOMB_[(1 << i) + m] = NBCOMB_[m] ^ rest[i];
    /* tmap gives a solution of M1 t = r whenever r is consistent */
    for (int i = 0; i < 2048; i++) {
        cw t = (cw)rng() & MASKH, r = gf2_mul(M1ROW_, t);
        for (int c = 0; c < NNULL_; c++) if (parc(LN_[c] & r)) die("left kernel of stage 1 is wrong");
        if (gf2_mul(M1ROW_, tmap(r)) != r) die("stage-1 particular solution is wrong");
    }
    /* the two stage identities on random skew pairs, against integer PAFs */
    for (int i = 0; i < 2048; i++) {
        cw yb = (cw)rng() & MASKH, yc = (cw)rng() & MASKH, t = yb ^ yc;
        int ub[HMAX + 1], uc[HMAX + 1]; cw b0, b1, c0, c1;
        uvals(skew_word(yb), ub, &b0, &b1); uvals(skew_word(yc), uc, &c0, &c1);
        if (gf2_mul(M1ROW_, t) != (b0 ^ c0)) die("stage-1 identity fails");
        uw S = symh_word(t);
        cw row[HMAX], g, col[HMAX];
        s2_rows_ref(S, row, &g);
        s2_cols(S, col);
        if (g != s2_G(S)) die("G(t) forms disagree");
        for (int k = 1; k <= H; k++) {
            uw F = ~(S ^ rotr(S, k)) & E_[k];
            int K = popu(F) >> 1, hx = popu(F & HX_[k]) >> 1;
            int a2 = N - ub[k] - uc[k];            /* uA + uD of a completing pair */
            int v = K + H - a2;
            if (v & 1) die("stage-2 integrality fails");
            if (parc(row[k - 1] & yb) != (((v >> 1) + hx) & 1)) die("stage-2 identity fails");
            int bit1 = (a2 >> 1) & 1;
            if (parc(row[k - 1] & yb) != (int)(((g >> (k - 1)) & 1) ^ (cw)bit1)) die("stage-2 folded right side fails");
            for (int u = 1; u <= H; u++)
                if (((col[u - 1] >> (k - 1)) & 1) != ((row[k - 1] >> (u - 1)) & 1)) die("stage-2 column build disagrees with the row build");
        }
    }
}

/* ------------------------------------------------------------ spectra */
static double *SINB_, *COSB_, *SBASE_, *CBASE_;   /* [NBY][256][H], [H] */
static void trig_init(void) {
    SINB_ = calloc((size_t)NBY_ * 256 * H, sizeof(double));
    COSB_ = calloc((size_t)NBY_ * 256 * H, sizeof(double));
    SBASE_ = calloc(H, sizeof(double)); CBASE_ = calloc(H, sizeof(double));
    for (int k = 1; k <= H; k++)
        for (int u = 1; u <= H; u++) {
            SBASE_[k - 1] += sin(2.0 * M_PI * u * k / N);
            CBASE_[k - 1] += cos(2.0 * M_PI * u * k / N);
        }
    for (int b = 0; b < NBY_; b++)
        for (int v = 0; v < 256; v++)
            for (int bit = 0; bit < 8; bit++) {
                int u = 8 * b + bit + 1;
                if (u > H || !((v >> bit) & 1)) continue;
                for (int k = 1; k <= H; k++) {
                    SINB_[((size_t)b * 256 + v) * H + k - 1] -= 2.0 * sin(2.0 * M_PI * u * k / N);
                    COSB_[((size_t)b * 256 + v) * H + k - 1] -= 2.0 * cos(2.0 * M_PI * u * k / N);
                }
            }
}
/* PSD of the skew row with code y at frequencies 1..H; returns the maximum */
static inline double psd_skew(cw y, double *out) {
    double s[HMAX];
    for (int k = 0; k < H; k++) s[k] = SBASE_[k];
    for (int b = 0; b < NBY_; b++) {
        const double *t = SINB_ + ((size_t)b * 256 + ((y >> (8 * b)) & 0xff)) * H;
        for (int k = 0; k < H; k++) s[k] += t[k];
    }
    double m = 0;
    for (int k = 0; k < H; k++) { double v = 1.0 + 4.0 * s[k] * s[k]; out[k] = v; if (v > m) m = v; }
    return m;
}
static inline double psd_sym(uint64_t code, double *out) {
    cw c = (cw)(code >> 1);
    double s[HMAX], d0 = (code & 1) ? -1.0 : 1.0;
    for (int k = 0; k < H; k++) s[k] = CBASE_[k];
    for (int b = 0; b < NBY_; b++) {
        const double *t = COSB_ + ((size_t)b * 256 + ((c >> (8 * b)) & 0xff)) * H;
        for (int k = 0; k < H; k++) s[k] += t[k];
    }
    double m = 0;
    for (int k = 0; k < H; k++) { double x = d0 + 2.0 * s[k], v = x * x; out[k] = v; if (v > m) m = v; }
    return m;
}

/* ------------------------------------------------- multipliers on D codes */
static int NDEC_; static cw *DBYTE_;
static int gcd_int(int a, int b) { while (b) { int q = a % b; a = b; b = q; } return a; }
static void dec_init(void) {
    int cnt = 0;
    for (int t = 2; t <= H; t++) if (gcd_int(t, N) == 1) cnt++;
    NDEC_ = cnt;
    DBYTE_ = calloc((size_t)(cnt + 1) * NBY_ * 256, sizeof(cw));
    int d = 0;
    for (int t = 2; t <= H; t++) {
        if (gcd_int(t, N) != 1) continue;
        cw dest[HMAX]; memset(dest, 0, sizeof dest);
        for (int j = 1; j <= H; j++) {
            int m = (int)(((long long)t * j) % N); if (m > H) m = N - m;
            dest[m - 1] |= (cw)1 << (j - 1);       /* new bit j-1 = old bit m-1 */
        }
        for (int b = 0; b < NBY_; b++)
            for (int v = 0; v < 256; v++) {
                cw acc = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int pos = 8 * b + bit;
                    if (pos < H && ((v >> bit) & 1)) acc |= dest[pos];
                }
                DBYTE_[(((size_t)d * NBY_) + b) * 256 + v] = acc;
            }
        /* check against the n-bit form on random codes */
        for (int i = 0; i < 64; i++) {
            cw c = (cw)rng() & MASKH, r = 0;
            for (int b = 0; b < NBY_; b++) r |= DBYTE_[(((size_t)d * NBY_) + b) * 256 + ((c >> (8 * b)) & 0xff)];
            uw w = symh_word(c), w2 = 0;
            for (int j = 1; j < N; j++) if ((w >> (((long long)t * j) % N)) & 1) w2 |= (uw)1 << j;
            if (w2 != symh_word(r)) die("multiplier table disagrees with the row form");
        }
        d++;
    }
}
static inline int d_canonical(cw c) {
    for (int d = 0; d < NDEC_; d++) {
        const cw *t = DBYTE_ + (size_t)d * NBY_ * 256;
        cw r = 0;
        for (int b = 0; b < NBY_; b++) r |= t[(size_t)b * 256 + ((c >> (8 * b)) & 0xff)];
        if (r < c) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------- the A table */
typedef struct { cw code, u0, u1, ta; } ARec;
typedef struct {
    long long n;            /* rows */
    ARec *rec;
    long long *sigstart;    /* [NSIG+1] */
    int nlev; long long cwords;
    uint64_t *bmp;          /* [H][nlev][cwords], bit set iff level(PSD_k) <= lev */
} ATable;
static int NLEV_ = 0;
static inline int lev_of(double x, int nlev) {
    int v = (int)(x * nlev / (double)(4 * N + 1));
    if (v < 0) v = 0;
    if (v > nlev - 1) v = nlev - 1;
    return v;
}
typedef struct { cw *code; uint16_t *sig; long long n, cap; } CVec;
static void cvec_push(CVec *v, cw code, int sig) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 4096;
        v->code = realloc(v->code, sizeof(cw) * (size_t)v->cap);
        v->sig = realloc(v->sig, sizeof(uint16_t) * (size_t)v->cap);
        if (!v->code || !v->sig) die("out of memory for the A codes");
    }
    v->code[v->n] = code; v->sig[v->n] = (uint16_t)sig; v->n++;
}
/* rows from half-codes i in [ilo, ihi) (code = 2i, so a_1 = +1), or from the
   given list when probing */
static void atable_build(ATable *T, long long ilo, long long ihi, const cw *list, long long nlist, double mem_mb) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    CVec *vec = calloc(nth, sizeof(CVec));
    const double ceil_a = 4.0 * N - 2.0 + 1e-6;
    long long lo = list ? 0 : ilo, hi = list ? nlist : ihi;
#pragma omp parallel num_threads(nth)
    {
        int th = 0;
#ifdef _OPENMP
        th = omp_get_thread_num();
#endif
        double psd[HMAX];
#pragma omp for schedule(static)
        for (long long i = lo; i < hi; i++) {
            cw y = list ? list[i] : (cw)((uint64_t)i << 1);
            if (psd_skew(y, psd) > ceil_a) continue;
            cw u0, u1; uvals(skew_word(y), NULL, &u0, &u1);
            cvec_push(&vec[th], y, sig_of(u0));
        }
    }
    int nsig = 1 << NNULL_;
    long long total = 0;
    T->sigstart = calloc(nsig + 1, sizeof(long long));
    for (int t = 0; t < nth; t++) {
        total += vec[t].n;
        for (long long i = 0; i < vec[t].n; i++) T->sigstart[vec[t].sig[i] + 1]++;
    }
    for (int s = 0; s < nsig; s++) T->sigstart[s + 1] += T->sigstart[s];
    T->n = total;
    cw *codes = malloc(sizeof(cw) * (size_t)(total + 1));
    long long *pos = malloc(sizeof(long long) * nsig);
    if (!codes || !pos) die("out of memory for the A table");
    memcpy(pos, T->sigstart, sizeof(long long) * nsig);
    for (int t = 0; t < nth; t++) {
        for (long long i = 0; i < vec[t].n; i++) codes[pos[vec[t].sig[i]]++] = vec[t].code[i];
        free(vec[t].code); free(vec[t].sig);
    }
    free(vec); free(pos);
    long long cwd = (total + 63) >> 6; if (cwd < 1) cwd = 1;
    int nlev = NLEV_;
    if (!nlev) {
        double rec_mb = (double)total * sizeof(ARec) / 1048576.0;
        nlev = 256;
        while (nlev > 8 && rec_mb + (double)H * nlev * cwd * 8.0 / 1048576.0 > mem_mb) nlev >>= 1;
    }
    T->nlev = nlev; T->cwords = cwd;
    T->rec = malloc(sizeof(ARec) * (size_t)(total + 1));
    T->bmp = calloc((size_t)H * nlev * cwd, sizeof(uint64_t));
    if (!T->rec || !T->bmp) die("out of memory for the A index");
#pragma omp parallel for schedule(dynamic, 64)
    for (long long w = 0; w < cwd; w++) {
        double psd[HMAX];
        long long e = (w + 1) * 64; if (e > total) e = total;
        for (long long i = w * 64; i < e; i++) {
            cw y = codes[i];
            psd_skew(y, psd);
            ARec *r = &T->rec[i];
            r->code = y;
            uvals(skew_word(y), NULL, &r->u0, &r->u1);
            r->ta = tmap(r->u0);
            for (int k = 0; k < H; k++)
                T->bmp[((size_t)k * nlev + lev_of(psd[k], nlev)) * cwd + w] |= 1ULL << (i & 63);
        }
    }
    free(codes);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < H; k++)
        for (int l = 1; l < nlev; l++) {
            uint64_t *dst = T->bmp + ((size_t)k * nlev + l) * cwd;
            const uint64_t *src = dst - cwd;
            for (long long w = 0; w < cwd; w++) dst[w] |= src[w];
        }
}
static void atable_free(ATable *T) { free(T->rec); free(T->bmp); free(T->sigstart); memset(T, 0, sizeof *T); }

/* ------------------------------------------------------ consistency table

   For each pattern t with t_1 = 0, up to CCK left null vectors w of M2(t)
   over the lags, each with the parity of G(t) against it in the top bit.  A
   pair with right-hand side G(t) xor X is inconsistent as soon as one w has
   parity(w & X) different from that bit.  A zero word ends the list.  The
   test only rejects systems that have no solution; patterns of corank above
   CCK keep their first CCK vectors and lose nothing. */
static cw *CC_ = NULL; static int CCK_ = 0;
static int s2_leftnull(cw t, cw *out, int maxout, cw *gout) {
    uw S = symh_word(t);
    cw col[HMAX], bt[HMAX], p = 0;
    s2_cols(S, col);
    for (int j = 0; j < H; j++) bt[j] = (cw)1 << j;
    for (int c = 0; c < H; c++) {
        cw m = col[c] & ~p; if (!m) continue;
        cw bit = m & (~m + 1), R = col[c] ^ bit;
        p |= bit;
        for (int c2 = 0; c2 < H; c2++) col[c2] ^= R & ((cw)0 - (cw)((col[c2] & bit) != 0));
        for (int j = 0; j < H; j++) bt[j] ^= R & ((cw)0 - (cw)((bt[j] & bit) != 0));
    }
    int nc = 0;
    for (int q = 0; q < H; q++) {
        if ((p >> q) & 1) continue;
        if (nc < maxout) {
            cw v = 0;
            for (int j = 0; j < H; j++) if ((bt[j] >> q) & 1) v |= (cw)1 << j;
            out[nc] = v;
        }
        nc++;
    }
    if (gout) *gout = s2_G(S);
    return nc;
}
static void cc_build(void) {
    size_t nt = (size_t)1 << (H - 1);
    CC_ = malloc(nt * (size_t)CCK_ * sizeof(cw));
    if (!CC_) die("out of memory for the consistency table");
#pragma omp parallel for schedule(dynamic, 4096)
    for (long long ti = 0; ti < (long long)nt; ti++) {
        cw t = (cw)((uint64_t)ti << 1), nv[HMAX], g;
        int c = s2_leftnull(t, nv, CCK_, &g);
        cw *e = CC_ + (size_t)ti * CCK_;
        for (int i = 0; i < CCK_; i++) e[i] = (i < c) ? (nv[i] | (parc(nv[i] & g) ? CC_FLAG : 0)) : 0;
    }
}

/* ------------------------------------------------------------- the solve */
typedef struct {
    long long nD, pairs, tc, ccpass, cons, cands, found, capped;
    double t_scan, t_solve;
} Stat;
typedef struct { cw a, b, c; uint64_t d; } Sol;
static Sol *SOL_; static long long NSOL_, SOLCAP_;
static void sol_add(cw a, cw b, cw c, uint64_t d) {
#pragma omp critical(soladd)
    {
        if (NSOL_ == SOLCAP_) {
            SOLCAP_ = SOLCAP_ ? 2 * SOLCAP_ : 1024;
            SOL_ = realloc(SOL_, sizeof(Sol) * (size_t)SOLCAP_);
            if (!SOL_) die("out of memory for the solutions");
        }
        SOL_[NSOL_].a = a; SOL_[NSOL_].b = b; SOL_[NSOL_].c = c; SOL_[NSOL_].d = d; NSOL_++;
    }
}
/* one pattern t of one pair: eliminate, lift, verify */
static void solve_t(cw t, cw X, cw acode, uint64_t dcode, const int *uD, Stat *st) {
    uw S = symh_word(t);
    cw col[HMAX], pivb[HMAX], p = 0;
    s2_cols(S, col);
    cw bb = (s2_G(S) ^ X) & MASKH;
    for (int c = 0; c < H; c++) {
        cw m = col[c] & ~p;
        cw bit = m & (~m + 1);
        pivb[c] = bit;
        if (!bit) continue;
        cw R = col[c] ^ bit;
        p |= bit;
        for (int c2 = 0; c2 < H; c2++) col[c2] ^= R & ((cw)0 - (cw)((col[c2] & bit) != 0));
        bb ^= R & ((cw)0 - (cw)((bb & bit) != 0));
    }
    if (bb & ~p) return;
    st->cons++;
    cw part = 0, basis[HMAX]; int nf = 0;
    for (int c = 0; c < H; c++) if (bb & pivb[c]) part |= (cw)1 << c;
    for (int f = 0; f < H; f++) {
        if (pivb[f]) continue;
        cw w = (cw)1 << f;
        for (int c2 = 0; c2 < H; c2++) if (col[f] & pivb[c2]) w |= (cw)1 << c2;
        basis[nf++] = w;
    }
    if (nf > MAXFREE) { st->capped++; return; }
    int tgt[HMAX + 1];                              /* popcount target: uB + uC = n - uA - uD */
    { int ua[HMAX + 1]; cw x0, x1; uvals(skew_word(acode), ua, &x0, &x1);
      for (int k = 1; k <= H; k++) tgt[k] = 2 * (N - ua[k] - uD[k]); }
    cw v = part;
    for (uint64_t g = 0; g < (1ULL << nf); g++) {
        if (g) v ^= basis[__builtin_ctzll(g)];
        if (v & 1) continue;                       /* R8 */
        cw c = v ^ t;
        if (c < v) continue;                       /* R9 */
        st->cands++;
        uw wb = skew_word(v), wc = skew_word(c);
        int ok = 1;
        for (int k = 1; k <= H; k++)
            if (popu(wb ^ rotr(wb, k)) + popu(wc ^ rotr(wc, k)) != tgt[k]) { ok = 0; break; }
        if (!ok) continue;
        st->found++;
        sol_add(acode, v, c, dcode);
    }
}

static int NOMAX_ = 0;
#define BLKW 512
static void process_d(uint64_t dcode, const ATable *T, Stat *st, int *surv) {
    double psd[HMAX];
    psd_sym(dcode, psd);
    int uD[HMAX + 1]; cw ud0, ud1;
    uvals(sym_word(dcode), uD, &ud0, &ud1);
    int sreq = sig_of(ud0) ^ sig_of(MASKH);
    long long lo = T->sigstart[sreq], hi = T->sigstart[sreq + 1];
    st->nD++;
    if (lo >= hi) return;
    const int nlev = T->nlev; const long long cwd = T->cwords;
    const uint64_t *bp[HMAX], *bneg = NULL; int nbp = 0, kstar = 0;
    for (int k = 1; k < H; k++) if (psd[k] < psd[kstar]) kstar = k;
    for (int k = 0; k < H; k++) {
        double cap = 4.0 * N - 2.0 - psd[k] + 1e-6;
        if (cap < 1.0) return;
        int l = lev_of(cap, nlev);
        if (l >= nlev - 1) continue;
        bp[nbp++] = T->bmp + ((size_t)k * nlev + l) * cwd;
    }
    if (!NOMAX_) {
        int l = lev_of((4.0 * N - psd[kstar]) / 3.0 - 1e-6, nlev);
        if (l > 0) bneg = T->bmp + ((size_t)kstar * nlev + (l - 1)) * cwd;
    }
    const cw tD = tmap((cw)(~ud0 & MASKH));
    long long w0 = lo >> 6, wlast = (hi - 1) >> 6;
    uint64_t lomask = ~0ULL << (lo & 63), himask = (hi & 63) ? ((1ULL << (hi & 63)) - 1) : ~0ULL;
    for (long long wb = w0; wb <= wlast; wb += BLKW) {
        double t0 = tcpu();
        long long we = wb + BLKW - 1; if (we > wlast) we = wlast;
        int ns = 0;
        for (long long w = wb; w <= we; w++) {
            uint64_t v = ~0ULL;
            for (int i = 0; i < nbp; i++) v &= bp[i][w];
            if (bneg) v &= ~bneg[w];
            if (w == w0) v &= lomask;
            if (w == wlast) v &= himask;
            while (v) { surv[ns++] = (int)((w - wb) << 6) + __builtin_ctzll(v); v &= v - 1; }
        }
        double t1 = tcpu();
        st->t_scan += t1 - t0;
        st->pairs += ns;
        st->tc += (long long)ns * NCOMB_;
        const ARec *base = T->rec + (wb << 6);
        for (int s0 = 0; s0 < ns; s0 += 16) {
            int s1 = s0 + 16 < ns ? s0 + 16 : ns;
            cw tt[16];
            for (int s = s0; s < s1; s++) {
                const ARec *r = base + surv[s];
                cw t = r->ta ^ tD;
                if (t & 1) t ^= NBSTAR_;           /* R7 */
                tt[s - s0] = t;
                if (CC_) for (int m = 0; m < NCOMB_; m++)
                    __builtin_prefetch(CC_ + (size_t)((t ^ NBCOMB_[m]) >> 1) * CCK_);
                if (s + 16 < ns) __builtin_prefetch(base + surv[s + 16]);
            }
            for (int s = s0; s < s1; s++) {
                const ARec *r = base + surv[s];
                cw X = (r->u1 ^ ud1 ^ (r->u0 & ud0)) & MASKH;
                for (int m = 0; m < NCOMB_; m++) {
                    cw t = tt[s - s0] ^ NBCOMB_[m];
                    if (CC_) {
                        const cw *e = CC_ + (size_t)(t >> 1) * CCK_;
                        int bad = 0;
                        for (int i = 0; i < CCK_; i++) {
                            cw ei = e[i], w = ei & ~CC_FLAG;
                            if (!w) break;
                            if (parc(w & X) != (int)(ei >> (CWB - 1))) { bad = 1; break; }
                        }
                        if (bad) continue;
                    }
                    st->ccpass++;
                    solve_t(t, X, r->code, dcode, uD, st);
                }
            }
        }
        st->t_solve += tcpu() - t1;
    }
}

/* ------------------------------------------------------------- D rows */
typedef struct { uint64_t *v; long long n, cap; } DVec;
static void dvec_push(DVec *d, uint64_t x) {
    if (d->n == d->cap) { d->cap = d->cap ? 2 * d->cap : 4096; d->v = realloc(d->v, sizeof(uint64_t) * (size_t)d->cap); if (!d->v) die("out of memory for the D rows"); }
    d->v[d->n++] = x;
}
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void fput_row(FILE *f, uw w) { for (int j = 0; j < N; j++) fputc(((w >> j) & 1) ? '-' : '+', f); }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: best_hkt <n> <threads> <out.txt>\n"); return 1; }
    N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    const char *outpath = argv[3];
    if (N < 3 || !(N & 1)) { fprintf(stderr, "n must be odd and at least 3\n"); return 1; }
    if (N > NMAX) { fprintf(stderr, "n = %d needs wider words (limit %d; build with -DWIDE for up to 127)\n", N, NMAX); return 1; }
    H = (N - 1) / 2;
    FULL = (N >= (int)(8 * sizeof(uw))) ? ~(uw)0 : (((uw)1 << N) - 1);
    MASKH = (H >= CWB) ? ~(cw)0 : (((cw)1 << H) - 1);
    if (threads < 1) threads = 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    int root = (int)(sqrt((double)(4 * N - 3)) + 0.5);
    if (root * root != 4 * N - 3) {
        printf("n=%d: 4n-3 is not a square, no best matrices\n", N);
        FILE *f = fopen(outpath, "w"); if (f) fclose(f);
        return 0;
    }
    double tstart = wnow();
    stage_init();
    trig_init();
    dec_init();
    { const char *v = getenv("BEST_NOMAX"); if (v && atoi(v)) NOMAX_ = 1; }
    int nodec = 0; { const char *v = getenv("BEST_NODEC"); if (v && atoi(v)) nodec = 1; }
    { const char *v = getenv("BEST_NLEV"); if (v) NLEV_ = atoi(v); }
    double mem_mb = 1000, cc_mb = 3000;
    { const char *v = getenv("BEST_MEM_MB"); if (v) mem_mb = atof(v); }
    { const char *v = getenv("BEST_CC_MB"); if (v) cc_mb = atof(v); }
    long long probeA = 0, probeD = 0;
    { const char *v = getenv("BEST_PROBE"); if (v && sscanf(v, "%lld,%lld", &probeA, &probeD) != 2) { fprintf(stderr, "BEST_PROBE wants a,d\n"); return 1; } }
    int probe = probeA > 0;
    const char *plant = getenv("BEST_PLANT");
    DVec PD = {0}; cw *plist = NULL; long long nplist = 0;
    if (plant) {
        FILE *pf = fopen(plant, "r"); if (!pf) { perror(plant); return 1; }
        char wd[4][160]; long long cap = 0;
        while (fscanf(pf, "%159s %159s %159s %159s", wd[0], wd[1], wd[2], wd[3]) == 4) {
            int nsym = 0;
            for (int i = 0; i < 4; i++) {
                if ((int)strlen(wd[i]) != N) die("planted row has the wrong length");
                uw w = 0;
                for (int j = 0; j < N; j++) if (wd[i][j] == '-') w |= (uw)1 << j;
                int issym = 1, isskew = 1;
                for (int j = 1; j <= H; j++) { int a = (int)((w >> j) & 1), b = (int)((w >> (N - j)) & 1); if (a != b) issym = 0; else isskew = 0; }
                if (issym) {
                    nsym++;
                    if (N - 2 * popu(w) < 0) w = ~w & FULL;                   /* R1 */
                    for (int t = 1; t < N; t++) {
                        if (gcd_int(t, N) != 1) continue;
                        uw w2 = 0;
                        for (int j = 0; j < N; j++) if ((w >> (((long long)t * j) % N)) & 1) w2 |= (uw)1 << j;
                        uint64_t code = (uint64_t)(w2 & (((uw)1 << (H + 1)) - 1));
                        if (d_canonical((cw)(code >> 1))) dvec_push(&PD, code);   /* R2 */
                    }
                } else if (isskew) {
                    if (w & 1) w = ~w & FULL;
                    for (int t = 1; t < N; t++) {
                        if (gcd_int(t, N) != 1) continue;
                        uw w2 = 0;
                        for (int j = 0; j < N; j++) if ((w >> (((long long)t * j) % N)) & 1) w2 |= (uw)1 << j;
                        cw y = (cw)(w2 >> 1) & MASKH;
                        if (skew_word(y) != w2) die("planted skew image is not skew");
                        if (y & 1) y = ~y & MASKH;                             /* R5 */
                        if (nplist == cap) { cap = cap ? 2 * cap : 1024; plist = realloc(plist, sizeof(cw) * (size_t)cap); }
                        plist[nplist++] = y;
                    }
                } else die("planted row is neither skew nor symmetric");
            }
            if (nsym != 1) die("planted quadruple does not have exactly one symmetric row");
        }
        fclose(pf);
        qsort(PD.v, (size_t)PD.n, sizeof(uint64_t), cmp_u64);
        long long m = 0;
        for (long long i = 0; i < PD.n; i++) if (!m || PD.v[m - 1] != PD.v[i]) PD.v[m++] = PD.v[i];
        PD.n = m;
        printf("   PLANT (not a census): %lld D orbit minima and %lld A codes from %s\n", PD.n, nplist, plant);
    }

    /* admissible (d_0, q): d_0 + 2H - 4q = root, q = number of -1 among d_1..d_H */
    int nadm = 0, adm_d0[2], adm_q[2];
    for (int d0 = 1; d0 >= -1; d0 -= 2) {
        int num = 2 * H + d0 - root;
        if (num < 0 || (num & 3) || num / 4 > H) continue;
        adm_d0[nadm] = d0; adm_q[nadm] = num / 4; nadm++;
    }
    printf("n=%d: H=%d, row sum of D %d, stage-1 corank %d (%d signatures, %d patterns per pair), %d multipliers up to sign\n",
           N, H, root, NNULL_, 1 << NNULL_, NCOMB_, NDEC_ + 1);
    for (int i = 0; i < nadm; i++) printf("   D rows: d_0 = %+d with %d entries -1 among d_1..d_H\n", adm_d0[i], adm_q[i]);
    if (NOMAX_) printf("   R4 off: A is any of the three skew rows\n");
    if (nodec) printf("   R2 off: every D of its multiplier orbit\n");
    fflush(stdout);

    /* ---- D rows ---- */
    DVec D = {0};
    long long d_adm = 0, d_ceil = 0;
    double td0 = wnow();
    const double ceil_d = 4.0 * N - 3.0 + 1e-6;
    if (plant) {
        D = PD;
    } else if (!probe) {
        long long ncode = 1LL << H;
#pragma omp parallel
        {
            DVec loc = {0}; long long la = 0, lc = 0; double psd[HMAX];
#pragma omp for schedule(dynamic, 1 << 14) nowait
            for (long long c = 0; c < ncode; c++) {
                int q = popc((cw)c);
                for (int i = 0; i < nadm; i++) {
                    if (q != adm_q[i]) continue;
                    la++;
                    uint64_t code = ((uint64_t)c << 1) | (adm_d0[i] < 0 ? 1u : 0u);
                    if (psd_sym(code, psd) > ceil_d) continue;
                    lc++;
                    if (!nodec && !d_canonical((cw)c)) continue;
                    dvec_push(&loc, code);
                }
            }
#pragma omp critical(dmerge)
            { for (long long i = 0; i < loc.n; i++) dvec_push(&D, loc.v[i]); d_adm += la; d_ceil += lc; }
            free(loc.v);
        }
        qsort(D.v, (size_t)D.n, sizeof(uint64_t), cmp_u64);
        printf("   D rows: %lld with the row sum, %lld under the ceiling, %lld kept%s (%.1fs)\n",
               d_adm, d_ceil, D.n, nodec ? "" : " as orbit minima", wnow() - td0);
    } else {
        double psd[HMAX];
        if (!nadm) die("no admissible D row sum");
        for (long long i = 0; i < probeD; i++) {
            int a = (int)(rng() % (uint64_t)nadm), q = adm_q[a];
            cw c = 0; int have = 0;                 /* uniform q-subset of H positions */
            while (have < q) { int b = (int)(rng() % (uint64_t)H); if (!((c >> b) & 1)) { c |= (cw)1 << b; have++; } }
            d_adm++;
            uint64_t code = ((uint64_t)c << 1) | (adm_d0[a] < 0 ? 1u : 0u);
            if (psd_sym(code, psd) > ceil_d) continue;
            d_ceil++;
            dvec_push(&D, code);
        }
        printf("   PROBE (not a census): %lld random D rows, %lld under the ceiling (%.4f)\n", d_adm, d_ceil, (double)d_ceil / (double)d_adm);
    }
    fflush(stdout);

    /* ---- consistency table ---- */
    {
        int kk = -1; const char *v = getenv("BEST_CCK"); if (v) kk = atoi(v);
        long hist[HMAX + 1] = {0}; int ns = 4096; cw tmp[HMAX];
        for (int i = 0; i < ns; i++) hist[s2_leftnull((cw)(rng() & MASKH) & ~(cw)1, tmp, HMAX, NULL)]++;
        printf("   stage-2 corank over %d random patterns:", ns);
        for (int k = 0; k <= H; k++) if (hist[k]) printf(" %d:%.1f%%", k, 100.0 * hist[k] / ns);
        printf("\n");
        if (kk < 0) {
            long acc = 0; kk = 1;
            for (int k = 0; k <= H; k++) { acc += hist[k]; if (acc * 100 >= (long)ns * 99) { kk = k > 0 ? k : 1; break; } }
            if (kk > 6) kk = 6;
            while (kk > 0 && (double)((size_t)1 << (H - 1)) * kk * sizeof(cw) / 1048576.0 > cc_mb) kk--;
            if (probe || plant) kk = 0;
        }
        CCK_ = kk;
        if (CCK_ > 0) {
            double tb = wnow();
            cc_build();
            printf("   consistency table: %d words per pattern, %.1f MB, built in %.1fs\n", CCK_,
                   (double)((size_t)1 << (H - 1)) * CCK_ * sizeof(cw) / 1048576.0, wnow() - tb);
        } else printf("   no consistency table: every pattern of every pair is eliminated\n");
    }
    fflush(stdout);

    /* ---- pieces of the A table ---- */
    long long nhalf = 1LL << (H - 1);
    cw *alist = NULL;
    int pieces = 1;
    if (plant) {
        alist = plist; probeA = nplist;
    } else if (probe) {
        alist = malloc(sizeof(cw) * (size_t)probeA);
        for (long long i = 0; i < probeA; i++) alist[i] = (cw)(rng() & MASKH) & ~(cw)1;
    } else {
        const char *v = getenv("BEST_PIECES");
        if (v) pieces = atoi(v);
        else {
            double psd[HMAX]; int ns = 20000, pass = 0;
            for (int i = 0; i < ns; i++) if (psd_skew((cw)(rng() & MASKH) & ~(cw)1, psd) <= 4.0 * N - 2.0) pass++;
            double rows = (double)nhalf * pass / ns;
            double mb = rows * (sizeof(ARec) + (double)H * 32 / 8.0) / 1048576.0;   /* at 32 levels */
            pieces = (int)ceil(mb / mem_mb);
        }
        if (pieces < 1) pieces = 1;
        if (pieces > nhalf) pieces = (int)nhalf;
    }

    Stat tot = {0};
    long long arows = 0; double t_table = 0, idx_mb_max = 0; int nlev_used = 0;
    for (int pc = 0; pc < pieces; pc++) {
        ATable T = {0};
        double tb = wnow();
        atable_build(&T, nhalf * pc / pieces, nhalf * (pc + 1) / pieces, alist, probeA, mem_mb);
        t_table += wnow() - tb;
        arows += T.n; nlev_used = T.nlev;
        double mb = ((double)T.n * sizeof(ARec) + (double)H * T.nlev * T.cwords * 8.0) / 1048576.0;
        if (mb > idx_mb_max) idx_mb_max = mb;
        printf("   A piece %d of %d: %lld rows, %d levels, %.1f MB, built in %.1fs\n", pc + 1, pieces, T.n, T.nlev, mb, wnow() - tb);
        fflush(stdout);
        if (T.n > 0x7fffffffLL - 64) die("A piece too large for int offsets; raise BEST_PIECES");
#pragma omp parallel
        {
            Stat st = {0};
            int *surv = malloc(sizeof(int) * BLKW * 64);
#pragma omp for schedule(dynamic, 1) nowait
            for (long long di = 0; di < D.n; di++) process_d(D.v[di], &T, &st, surv);
            free(surv);
#pragma omp critical(statmerge)
            {
                tot.pairs += st.pairs; tot.tc += st.tc; tot.ccpass += st.ccpass; tot.cons += st.cons;
                tot.cands += st.cands; tot.found += st.found; tot.capped += st.capped;
                tot.t_scan += st.t_scan; tot.t_solve += st.t_solve;
            }
        }
        atable_free(&T);
    }
    double telapsed = wnow() - tstart;

    long long acodes = (probe || plant) ? probeA : nhalf;
    printf("   A rows under the ceiling: %lld of %lld (%.4f)\n", arows, acodes, (double)arows / (double)acodes);
    printf("   pairs (A, D) past the index: %lld (%.3g per D row, %.3g of all pairs)\n", tot.pairs,
           D.n ? (double)tot.pairs / (double)D.n : 0.0, (D.n && arows) ? (double)tot.pairs / ((double)D.n * (double)arows) : 0.0);
    printf("   patterns %lld, past the consistency table %lld (%.2f%%), consistent systems %lld (%.3f%%), candidates verified %lld\n",
           tot.tc, tot.ccpass, tot.tc ? 100.0 * tot.ccpass / (double)tot.tc : 0.0, tot.cons,
           tot.tc ? 100.0 * tot.cons / (double)tot.tc : 0.0, tot.cands);
    printf("   thread-seconds: A table %.2f (wall) | index scan %.2f | test and solve %.2f; %.0f ns per pair, %.0f ns per pattern\n",
           t_table, tot.t_scan, tot.t_solve, tot.pairs ? 1e9 * tot.t_solve / (double)tot.pairs : 0.0,
           tot.tc ? 1e9 * tot.t_solve / (double)tot.tc : 0.0);
    printf("   nullity-capped systems: %lld%s\n", tot.capped, tot.capped ? "   WARNING: the run is NOT exhaustive" : "");
    {
        FILE *st = fopen("/proc/self/status", "r"); char line[256];
        if (st) { while (fgets(line, sizeof line, st)) if (!strncmp(line, "VmHWM:", 6)) { printf("   peak RSS %s", line + 6); break; } fclose(st); }
    }
    if (probe) {
        /* projection: D orbit minima ~ (rows with the sum)/(orbit size) times the ceiling rate */
        double nD_all = 0;
        for (int i = 0; i < nadm; i++) { double b = 1; for (int j = 1; j <= adm_q[i]; j++) b = b * (H - adm_q[i] + j) / j; nD_all += b; }
        double nD = nD_all / (NDEC_ + 1) * (double)d_ceil / (double)d_adm;
        double nA = (double)nhalf * (double)arows / (double)acodes;
        double f = (D.n && arows) ? (double)tot.pairs / ((double)D.n * (double)arows) : 0.0;
        double ppairs = nD * nA * f;
        printf("   PROBE projection at %d levels: %.3g D orbit minima, %.3g A rows, %.3g pairs, %.3g patterns\n",
               nlev_used, nD, nA, ppairs, ppairs * NCOMB_);
        printf("   PROBE cost without the consistency table: scan %.3g + solve %.3g thread-seconds (this host), found in sample %lld\n",
               nD * nA / 64.0 * (tot.t_scan / ((double)D.n * ((double)arows / 64.0 + 1))), ppairs * (tot.pairs ? tot.t_solve / (double)tot.pairs : 0.0), tot.found);
    } else {
        printf("n=%d: %lld solutions written (representatives, not classes), %.2fs\n", N, tot.found, telapsed);
        FILE *f = fopen(outpath, "w");
        if (!f) { perror(outpath); return 1; }
        for (long long i = 0; i < NSOL_; i++) {
            fput_row(f, skew_word(SOL_[i].a)); fputc(' ', f);
            fput_row(f, skew_word(SOL_[i].b)); fputc(' ', f);
            fput_row(f, skew_word(SOL_[i].c)); fputc(' ', f);
            fput_row(f, sym_word(SOL_[i].d)); fputc('\n', f);
        }
        fclose(f);
    }
    return 0;
}
