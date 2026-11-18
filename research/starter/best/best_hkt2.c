/* best_hkt2.c -- circulant best matrices: exhaustive census up to equiv-v1 by
 * a mod-4 linearisation (the Holzmann-Kharaghani-Tayfeh-Rezaie idea) adapted
 * to two unknown skew rows.  Successor of best_hkt.c (best-census-hkt-v1):
 * the same search space, the same reductions R1 to R9 and the same output
 * lines, reorganised so that one pattern costs several times less.  What
 * changed is listed under "Changes from best_hkt.c" below, each with its
 * exhaustiveness argument.
 *
 *   build: gcc -O3 -march=native -fopenmp best_hkt2.c -lm -o best_hkt2
 *          (add -DWIDE for n > 63: 128-bit rows, 64-bit code words;
 *           -DVB=64 selects 64-byte vectors for an AVX-512 host and has not
 *           been run anywhere yet)
 *   run:   ./best_hkt2 <n> <threads> <out.txt>
 *   out:   one line per solution found, four words of n '+'/'-' (A B C D),
 *          A, B, C skew, D symmetric.  The lines are representatives, not
 *          yet reduced to classes: canonicalise them to count classes.
 *   env:   BEST_NLEV=l      levels of the quantised-spectrum index (default: auto)
 *          BEST_PIECES=p    split the A table into p code ranges (default: auto)
 *          BEST_MEM_MB=m    budget for one A piece, records + index (default 1000)
 *          BEST_CCK=k       words per pattern in the consistency table
 *                           (default 2; 0 = no table, every pattern is eliminated)
 *          BEST_CC_MB=m     budget for the consistency table of one class of
 *                           patterns (default 1100); the number of classes
 *                           is the least power of two that fits it
 *          BEST_QB=q        force 2^q classes of patterns
 *          BEST_PIECE=i     with BEST_PIECES=p: run only A piece i (0-based)
 *          BEST_CLASS=b     run only class b of patterns (0-based)
 *                           Both restrict a census to one unit of work, for
 *                           checkpointing (see "Units of work" below); the
 *                           output file then holds that unit's solutions and
 *                           the log that unit's pair, pattern and capped
 *                           counts.  They may be combined.
 *          BEST_DFILE=path  keep the D list of a census in this file: read if it
 *                           is there and matches n and BEST_NODEC, else
 *                           enumerated as usual and written
 *          BEST_NOMAX=1     drop reduction R4 below (A need not be the peak row)
 *          BEST_NODEC=1     drop reduction R2 below (every D, not one per orbit)
 *          BEST_PROBE=a,d   sampling probe, NOT a census: a random A codes and
 *                           d random D codes through the same filters and solve
 *                           for one class of patterns, a real A build over a
 *                           consecutive code range for its cost, and the
 *                           projected thread-seconds of a census by part.
 *          BEST_PROBE_CLS=c with BEST_PROBE: keep only the D rows of the first c
 *                           classes, so that a class holds as many D rows as
 *                           in a census (the scan then meets warm bitmaps)
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
 *       have no completion.  (See C5: the conditions bear on D alone.)
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
 * Changes from best_hkt.c.  None adds a reduction: the set of (A, D, t)
 * systems solved and the set of lines emitted are those of best_hkt.c, and
 * the sorted output is identical wherever both were run.
 *
 *   C1  Stage tables.  M2(t) is affine in t (the entry of row k, column u is
 *       [sigma_u = sigma_{u+k}] + [sigma_{u-k} = sigma_u] up to fixed
 *       exclusions, and [a = b] = 1 + a + b) and G(t) is quadratic in t (a
 *       second binary digit of a count of bits affine in t).  The columns and
 *       the right side of t = t0 xor kappa are therefore built from nibble
 *       tables and a bilinear form instead of 128-bit row arithmetic.  Both
 *       facts are identities; they are checked at startup on random patterns
 *       against the reference builds, and the run stops if either fails.
 *   C2  Batched elimination.  VW systems are eliminated together, one per
 *       vector lane, and a pivot step updates only the later columns and the
 *       right side.  The skipped updates are no-ops (see batch_elim), so the
 *       reduced system, the consistency verdict and the solution set are
 *       those of the plain elimination.
 *   C3  Two unknowns fixed in advance.  The all-ones vector and t are in the
 *       kernel of M2(t) for every t (complementing y reverses B and C
 *       together, y -> y xor t swaps them; neither changes any PAF sum, and
 *       the stage-2 identity holds for every y).  The solutions thus fall
 *       into orbits {v, not v, v xor t, not v xor t}, and for t != 0 with
 *       highest set bit mu exactly one member of each has v_1 = 0 and bit mu
 *       clear; it is the member R8 and R9 keep.  The system is therefore
 *       solved in the other H-2 unknowns with those two set to zero: it is
 *       consistent iff the full system is, and each of its solutions is the
 *       R8/R9 representative of one orbit, so the candidates are those of
 *       best_hkt.c without enumerating and filtering the kernel.  Patterns
 *       with t = 0 keep the plain solve.  A startup check runs both solves
 *       on random systems and requires the same verdict, the same nullity,
 *       and that the orbits of the reduced solutions are exactly the full
 *       solution set.  The nullity cap is applied to the full nullity.
 *   C4  Cheap integer tests first.  u(k) of a skew row is a combination of
 *       three popcounts of its code and its bit reversal (lagf; lags 1 and 2
 *       are also packed in dis12); the identities are checked at startup
 *       against the PAF for every lag.  A candidate is tested on lags 1 and
 *       2, then on all lags by lagf, and what passes is confirmed by the
 *       128-bit PAF check of best_hkt.c.  The cheap tests evaluate the same
 *       integer equations, so they reject only non-solutions; a candidate
 *       passed by lagf and refused by the PAF check stops the run.
 *   C5  R6 on D alone.  For a skew row u0 = M1 y xor const (checked at
 *       startup), and the left kernel of M1 kills M1 y, so every skew row
 *       has the same signature.  The signature test is therefore applied
 *       once to each D row, and the A table is no longer sorted by it.
 *   C6  Consistency table by classes of patterns.  For a pattern t the table
 *       holds up to BEST_CCK left null vectors w of M2(t); a system with
 *       right side b is rejected when w . b = 1 for one of them, which is a
 *       proof that it has no solution, never a heuristic.  The table of all
 *       patterns would need 2^(H-1) entries, so patterns are split into 2^q
 *       classes by q linear forms that vanish on the stage-1 kernel (see
 *       class_init): all patterns of a pair lie in one class, and that class
 *       is cls(A) xor cls(D) for two q-bit labels.  The census loops over
 *       the classes; for each it builds the table of that class, rebuilds
 *       the A pieces sorted by cls(A), and lets every D row scan the one
 *       range of A rows with cls(A) = class xor cls(D).  Each pair (A, D) is
 *       met in exactly one class, so the pairs and patterns processed are
 *       those of the single pass.  The index of a pattern within its class
 *       is a linear bijection (checked at startup) that puts the patterns of
 *       one pair in adjacent entries; a pattern whose class differs from the
 *       table's stops the run, so a wrong entry is never consulted.
 *   C7  A build.  The spectrum ceiling of R3 is tested per block of 128
 *       codes with the sine sums of the upper bytes shared, as
 *       s^2 <= (ceiling - 1)/4 with the same relaxed ceiling; the summation
 *       order differs from psd_skew by rounding of order 1e-13, far inside
 *       the 1e-6 relaxation, so no row under the true ceiling is dropped.
 *
 *   C8  Fine spectrum test.  Each pair that passes the quantised index is
 *       tested again on R3 and R4 with the spectrum of A quantised to 256
 *       levels (one byte per frequency, stored with the row).  The test
 *       compares levels of the same nondecreasing level function, level(PSD_A)
 *       <= level(cap) and level(PSD_A(k*)) >= level(threshold), with the
 *       bounds relaxed by 1e-6 as in R3 and R4, so it is one-sided like the
 *       index.  It makes the pairs solved independent of BEST_NLEV: a coarse
 *       index only costs more fine tests.  Where best_hkt.c ran with fewer
 *       than 256 index levels its R4 was looser and it could emit a class
 *       under more than one choice of A; the class sets are the same.
 *   C9  Order and bookkeeping.  D rows are sorted by class so that the rows
 *       scanning one range of an A piece follow each other; the constants of
 *       a D row (u values, G of its two completions) are computed once; the
 *       A records come from tables (arec_fill), checked at startup and on
 *       every 4096th row against the reference build; the index scan reads
 *       one bitmap at a time over a block; section times are taken from the
 *       time-stamp counter.  None of this changes which pairs are met.
 *
 * Units of work.  A pair (A, D) is solved inside exactly one iteration of the
 * two loops of a census: the class b = cls(A) xor cls(D) (C6), and the piece
 * i that holds the code of A, the pieces being the half-code ranges
 * [nhalf i/p, nhalf (i+1)/p), which are disjoint and cover every code.  All
 * D rows are met in every iteration.  So the runs with BEST_CLASS=b,
 * BEST_PIECE=i over all (b, i), or over all b alone, or over all i alone,
 * partition the pairs of the full census: the union of their outputs is the
 * output of the full run and their pair, pattern and capped counts add up
 * to its counts.  The split must use one fixed BEST_PIECES and BEST_QB (or
 * BEST_CC_MB) for all units.  A unit by class builds one consistency table;
 * a unit by piece alone builds the tables of all classes again, so at large
 * orders the class is the cheaper unit.
 *
 * Self-checks at startup: the stage-1 and stage-2 identities on random skew
 * pairs against integer PAFs, the closed-form column build against the row
 * build, the kernel and particular solution of M1, the multiplier tables,
 * and the checks named in C1 to C9.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
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
#define HMAX 64
#ifndef MAXFREE
#define MAXFREE 20
#endif

static int N, H;
static uw FULL;
static cw MASKH;

static double wnow(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
/* section timer: the time-stamp counter where there is one (the per-thread
   CPU clock is a system call, too slow for one call per D row and block);
   ticks are converted with the rate measured over the whole run */
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline double tcpu(void) { return (double)__rdtsc(); }
#define TCPU_TICKS 1
#else
static double tcpu(void) { struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
#define TCPU_TICKS 0
#endif

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
static cw M1ROW_[HMAX], LN_[HMAX], NBSTAR_, *NBCOMB_, REST_[HMAX];
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

static void die(const char *m) { fprintf(stderr, "best_hkt2: n=%d: %s\n", N, m); exit(1); }

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
    cw *rest = REST_; int nr = 0;
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


/* ------------------------------------------------- linearised stage tables

   M2(t) is affine in t and G(t) is quadratic in t (both are checked below on
   random patterns), so for t = t0 xor kappa the columns and the right side
   come from nibble tables and a few xors instead of 128-bit row arithmetic:
     cols(t) = COL0 xor LC(t),                LC linear,
     G(a xor b) = G(a) xor G(b) xor G(0) xor BIL(a, b),   BIL bilinear,
     BIL(a, b) = xor of QT_[i][j] over bits i of a and j of b.              */
typedef cw vcol __attribute__((vector_size(32)));
#define CVW ((int)(32 / sizeof(cw)))
#define HP2MAX ((HMAX + 1 + 7) & ~7)
typedef union { vcol v[HP2MAX * sizeof(cw) / 32]; cw w[HP2MAX]; } ColBuf;
static int NV_, NIB_, NR_;
static cw G0_, QT_[HMAX][HMAX];
static cw *LCN_;                 /* [NIB][16][HP2MAX] linear part of the columns */
static cw *BILK_;                /* [NR][NIB][16] BIL(., rest_i) */
static cw *QTB_, LSTAR_[HMAX];   /* QTB_[b][v][i] = BIL(e_i, v << 8b); LSTAR_[i] = BIL(e_i, NBSTAR_) */
static cw *GK_;                  /* GK_[m] = G(NBCOMB_[m]) xor G(0) */
static ColBuf *COLK_;            /* COLK_[m] = cols(NBCOMB_[m]) */
static inline cw bil_ref(cw a, cw b) { return s2_G(symh_word(a ^ b)) ^ s2_G(symh_word(a)) ^ s2_G(symh_word(b)) ^ G0_; }
static inline cw nib_lin(const cw *tab, cw x) {      /* tab[NIB][16] */
    cw r = 0;
    for (int b = 0; b < NIB_; b++) r ^= tab[b * 16 + ((x >> (4 * b)) & 15)];
    return r;
}
static void nib_build(cw *tab, const cw *lin) {      /* lin[H]: image of each unit vector */
    for (int b = 0; b < NIB_; b++) {
        tab[b * 16] = 0;
        for (int v = 1; v < 16; v++) {
            int idx = 4 * b + __builtin_ctz((unsigned)v);
            tab[b * 16 + v] = tab[b * 16 + (v & (v - 1))] ^ (idx < H ? lin[idx] : 0);
        }
    }
}
static inline void lc_cols(cw t, ColBuf *out) {      /* linear part of cols(t) */
    const cw *r0 = LCN_ + (size_t)(t & 15) * HP2MAX;
    for (int i = 0; i < NV_; i++) out->v[i] = ((const ColBuf *)r0)->v[i];
    for (int b = 1; b < NIB_; b++) {
        const ColBuf *r = (const ColBuf *)(LCN_ + ((size_t)b * 16 + ((t >> (4 * b)) & 15)) * HP2MAX);
        for (int i = 0; i < NV_; i++) out->v[i] ^= r->v[i];
    }
}
static void fast_init(void) {
    NV_ = (H + CVW - 1) / CVW;
    NIB_ = (H + 3) / 4;
    G0_ = s2_G(symh_word(0));
    cw col0[HMAX], colx[HMAX];
    s2_cols(symh_word(0), col0);
    if (posix_memalign((void **)&LCN_, 64, sizeof(cw) * (size_t)NIB_ * 16 * HP2MAX)) die("out of memory");
    memset(LCN_, 0, sizeof(cw) * (size_t)NIB_ * 16 * HP2MAX);
    cw (*delta)[HMAX] = calloc(H, sizeof *delta);
    for (int i = 0; i < H; i++) {
        s2_cols(symh_word((cw)1 << i), colx);
        for (int u = 0; u < H; u++) delta[i][u] = colx[u] ^ col0[u];
    }
    for (int b = 0; b < NIB_; b++)
        for (int v = 1; v < 16; v++) {
            int idx = 4 * b + __builtin_ctz((unsigned)v);
            cw *dst = LCN_ + ((size_t)b * 16 + v) * HP2MAX, *src = LCN_ + ((size_t)b * 16 + (v & (v - 1))) * HP2MAX;
            for (int u = 0; u < H; u++) dst[u] = src[u] ^ (idx < H ? delta[idx][u] : 0);
        }
    free(delta);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++) QT_[i][j] = (i == j) ? 0 : bil_ref((cw)1 << i, (cw)1 << j);
    QTB_ = calloc((size_t)NBY_ * 256 * HMAX, sizeof(cw));
    if (!QTB_) die("out of memory");
    for (int b = 0; b < NBY_; b++)
        for (int v = 1; v < 256; v++) {
            int j = 8 * b + __builtin_ctz((unsigned)v);
            cw *dst = QTB_ + ((size_t)b * 256 + v) * HMAX, *src = QTB_ + ((size_t)b * 256 + (v & (v - 1))) * HMAX;
            for (int i = 0; i < H; i++) dst[i] = src[i] ^ (j < H ? QT_[i][j] : 0);
        }
    for (int i = 0; i < H; i++) { cw a = 0; for (int j = 0; j < H; j++) if ((NBSTAR_ >> j) & 1) a ^= QT_[i][j]; LSTAR_[i] = a; }
    NR_ = 0; while ((1 << NR_) < NCOMB_) NR_++;
    BILK_ = calloc((size_t)(NR_ + 1) * NIB_ * 16, sizeof(cw));
    for (int r = 0; r < NR_; r++) {
        cw lin[HMAX];
        for (int i = 0; i < H; i++) { cw a = 0; for (int j = 0; j < H; j++) if ((REST_[r] >> j) & 1) a ^= QT_[i][j]; lin[i] = a; }
        nib_build(BILK_ + (size_t)r * NIB_ * 16, lin);
    }
    GK_ = malloc(sizeof(cw) * (size_t)NCOMB_);
    if (posix_memalign((void **)&COLK_, 64, sizeof(ColBuf) * (size_t)NCOMB_)) die("out of memory");
    memset(COLK_, 0, sizeof(ColBuf) * (size_t)NCOMB_);
    for (int m = 0; m < NCOMB_; m++) {
        GK_[m] = s2_G(symh_word(NBCOMB_[m])) ^ G0_;
        s2_cols(symh_word(NBCOMB_[m]), colx);
        for (int u = 0; u < H; u++) COLK_[m].w[u] = colx[u];
    }
    /* the affine and quadratic forms against the reference builds */
    for (int it = 0; it < 4096; it++) {
        cw a = (cw)rng() & MASKH, b = (cw)rng() & MASKH, c = (cw)rng() & MASKH;
        ColBuf lc; lc_cols(a, &lc);
        s2_cols(symh_word(a), colx);
        for (int u = 0; u < H; u++) if ((lc.w[u] ^ col0[u]) != colx[u]) die("stage-2 columns are not affine in the pattern");
        for (int u = H; u < NV_ * CVW; u++) if (lc.w[u]) die("column padding is not zero");
        cw q = 0;
        for (int i = 0; i < H; i++) if ((a >> i) & 1) for (int j = 0; j < H; j++) if ((b >> j) & 1) q ^= QT_[i][j];
        if (q != bil_ref(a, b)) die("G is not quadratic in the pattern (bilinear form)");
        if ((bil_ref(a ^ b, c) ^ bil_ref(a, c) ^ bil_ref(b, c)) != 0) die("G is not quadratic in the pattern (additivity)");
        for (int r = 0; r < NR_; r++) if (nib_lin(BILK_ + (size_t)r * NIB_ * 16, a) != bil_ref(a, REST_[r])) die("kernel bilinear table is wrong");
    }
}

/* ------------------------------------------------- classes of patterns

   QB_ linear forms rho_i on patterns, each in the row space of M1 (rho_i =
   psi_i M1) and reduced so that rho_i has, among the top QB_ pattern bits,
   only bit H-1-i.  The class of t is (rho_i . t)_i = (psi_i . M1 t)_i.  All
   patterns of a pair have the same M1 t = r, so the class of the pair is
   cls(u0 of A) xor cls(not u0 of D); and a pattern of a given class is fixed
   by its low H-QB_ bits, which index the consistency table of that class. */
static int QB_ = 0, ASIG_ = 0;
static cw U0C_ = 0;              /* u0 of a skew row with code y is M1 y xor U0C_ */
static cw PSI_[16], RHO_[16], LOWMASK_;
static inline int cls_r(cw r) { int c = 0; for (int i = 0; i < QB_; i++) c |= parc(PSI_[i] & r) << i; return c; }
static inline int cls_t(cw t) { int c = 0; for (int i = 0; i < QB_; i++) c |= parc(RHO_[i] & t) << i; return c; }
static void class_init(int q) {
    QB_ = q; LOWMASK_ = MASKH;
    {   /* the signature of a skew row does not depend on the row */
        cw u0, u1; uvals(skew_word(0), NULL, &u0, &u1); ASIG_ = sig_of(u0); U0C_ = u0;
        for (int i = 0; i < 4096; i++) {
            cw y = (cw)rng() & MASKH; uvals(skew_word(y), NULL, &u0, &u1);
            if (u0 != (gf2_mul(M1ROW_, y) ^ U0C_)) die("u0 of a skew row is not affine in its code");
            if (sig_of(u0) != ASIG_) die("the signature of a skew row is not constant");
        }
    }
    if (!q) return;
    if (q > 16 || q > H - 1) die("too many class bits");
    cw a[HMAX], id[HMAX]; char used[HMAX] = {0}; int piv[16];
    for (int k = 0; k < H; k++) { a[k] = M1ROW_[k]; id[k] = (cw)1 << k; }
    for (int i = 0; i < q; i++) {
        int col = H - 1 - i, pr = -1;
        for (int r = 0; r < H; r++) if (!used[r] && ((a[r] >> col) & 1)) { pr = r; break; }
        if (pr < 0) die("the top pattern bits are not independent modulo the stage-1 kernel; lower BEST_QB");
        used[pr] = 1; piv[i] = pr;
        for (int r = 0; r < H; r++) if (r != pr && ((a[r] >> col) & 1)) { a[r] ^= a[pr]; id[r] ^= id[pr]; }
    }
    LOWMASK_ = MASKH >> q;
    for (int i = 0; i < q; i++) {
        RHO_[i] = a[piv[i]]; PSI_[i] = id[piv[i]];
        if ((RHO_[i] & ~LOWMASK_) != ((cw)1 << (H - 1 - i))) die("class forms are not reduced");
        for (int m = 0; m < NCOMB_; m++) if (parc(RHO_[i] & NBCOMB_[m])) die("class form does not vanish on the stage-1 kernel");
        if (parc(RHO_[i] & NBSTAR_)) die("class form does not vanish on the stage-1 kernel");
    }
    for (int it = 0; it < 4096; it++) {
        cw t = (cw)rng() & MASKH;
        if (cls_t(t) != cls_r(gf2_mul(M1ROW_, t))) die("class of a pattern disagrees with the class of its image");
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
/* A record from its code by tables.  u0 is affine in the code (U0C_), ta is
   linear in u0, and xa = u1 xor G(ta) is quadratic in the code: u1 is a
   second binary digit of a count of bits affine in the code, G is quadratic
   in ta.  A quadratic map is the sum of its values on the nibbles of the
   argument and of its polar form on the pairs of nibbles, which are tabulated
   from the reference build; the result is checked at startup on random codes
   and on every 4096th row of every A piece. */
typedef struct { cw code, u0, xa, ta; } ARec;   /* xa = u1 xor G(ta) */
static cw *XAN_ = NULL, *XAB_ = NULL, *U0N_ = NULL, XA0_;   /* [NIB][16], [NIB][NIB][256], [NIB][16] */
static inline cw xa_ref(cw y) { cw u0, u1; uvals(skew_word(y), NULL, &u0, &u1); return u1 ^ s2_G(symh_word(tmap(u0))); }
static inline void arec_fill(ARec *r, cw y) {
    cw xa = XA0_, u0 = U0C_;
    int nb[(HMAX + 3) / 4];
    for (int a = 0; a < NIB_; a++) { nb[a] = (int)((y >> (4 * a)) & 15); xa ^= XAN_[a * 16 + nb[a]]; u0 ^= U0N_[a * 16 + nb[a]]; }
    for (int a = 0; a < NIB_; a++)
        for (int b = a + 1; b < NIB_; b++) xa ^= XAB_[((size_t)(a * NIB_ + b) << 8) + (nb[a] << 4) + nb[b]];
    r->code = y; r->u0 = u0; r->xa = xa; r->ta = tmap(u0);
}
static void arec_init(void) {
    XAN_ = calloc((size_t)NIB_ * 16, sizeof(cw)); U0N_ = calloc((size_t)NIB_ * 16, sizeof(cw));
    XAB_ = calloc((size_t)NIB_ * NIB_ * 256, sizeof(cw));
    if (!XAN_ || !U0N_ || !XAB_) die("out of memory");
    XA0_ = xa_ref(0);
    for (int a = 0; a < NIB_; a++)
        for (int v = 0; v < 16; v++) {
            cw x = ((cw)v << (4 * a)) & MASKH;
            XAN_[a * 16 + v] = xa_ref(x) ^ XA0_;
            U0N_[a * 16 + v] = gf2_mul(M1ROW_, x);
        }
    for (int a = 0; a < NIB_; a++)
        for (int b = a + 1; b < NIB_; b++)
            for (int va = 0; va < 16; va++)
                for (int vb = 0; vb < 16; vb++) {
                    cw x = ((cw)va << (4 * a)) & MASKH, z = ((cw)vb << (4 * b)) & MASKH;
                    XAB_[((size_t)(a * NIB_ + b) << 8) + (va << 4) + vb] = xa_ref(x ^ z) ^ xa_ref(x) ^ xa_ref(z) ^ XA0_;
                }
    for (int it = 0; it < 4096; it++) {
        cw y = (cw)rng() & MASKH, u0, u1; ARec r;
        uvals(skew_word(y), NULL, &u0, &u1);
        arec_fill(&r, y);
        if (r.u0 != u0 || r.ta != tmap(u0) || r.xa != xa_ref(y)) die("A record tables disagree with the reference build");
    }
}
typedef struct {
    long long n;            /* rows */
    ARec *rec;
    long long *sigstart;    /* [number of classes + 1] */
    int nlev; long long cwords;
    uint64_t *bmp;          /* [H][nlev][cwords], bit set iff level(PSD_k) <= lev */
    uint8_t *lv; int ls;    /* [n][ls]: level of PSD_k out of FLEV, for the fine test of each pair */
} ATable;
static int NLEV_ = 0;
#define FLEV 256
static inline int lev_of(double x, int nlev) {
    int v = (int)(x * nlev / (double)(4 * N + 1));
    if (v < 0) v = 0;
    if (v > nlev - 1) v = nlev - 1;
    return v;
}
typedef struct { cw *code; uint16_t *sig; long long n, cap; } CVec;   /* sig: the class of the row */
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
        if (list) {
#pragma omp for schedule(static)
            for (long long i = lo; i < hi; i++) {
                cw y = list[i];
                if (psd_skew(y, psd) > ceil_a) continue;
                cvec_push(&vec[th], y, cls_r(gf2_mul(M1ROW_, y) ^ U0C_));
            }
        } else {
            /* codes 2i in blocks of 128: the sine sums of the upper bytes are
               shared by a block, and 1 + 4 s^2 <= ceiling is tested as
               s^2 <= (ceiling - 1)/4 with the same relaxed ceiling */
            const double thr2 = (ceil_a - 1.0) / 4.0;
            const int h = H;
#pragma omp for schedule(static)
            for (long long blk = lo >> 7; blk <= (hi - 1) >> 7; blk++) {
                double sh[HMAX];
                cw yh = (cw)((uint64_t)blk << 8);
                for (int k = 0; k < h; k++) sh[k] = SBASE_[k];
                for (int b = 1; b < NBY_; b++) {
                    const double *t = SINB_ + ((size_t)b * 256 + ((yh >> (8 * b)) & 0xff)) * h;
                    for (int k = 0; k < h; k++) sh[k] += t[k];
                }
                long long i0 = blk << 7, i1 = i0 + 128;
                if (i0 < lo) i0 = lo;
                if (i1 > hi) i1 = hi;
                for (long long i = i0; i < i1; i++) {
                    const double *t = SINB_ + (size_t)((i & 127) << 1) * h;
                    int bad = 0;
                    for (int k = 0; k < h; k++) { double x = sh[k] + t[k]; bad |= (x * x > thr2); }
                    if (bad) continue;
                    cw y = (cw)((uint64_t)i << 1);
                    cvec_push(&vec[th], y, cls_r(gf2_mul(M1ROW_, y) ^ U0C_));
                }
            }
        }
    }
    int nsig = 1 << QB_;
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
        double rec_mb = (double)total * (sizeof(ARec) + ((H + 7) & ~7)) / 1048576.0;
        nlev = 256;
        while (nlev > 8 && rec_mb + (double)H * nlev * cwd * 8.0 / 1048576.0 > mem_mb) nlev >>= 1;
    }
    T->nlev = nlev; T->cwords = cwd;
    T->rec = malloc(sizeof(ARec) * (size_t)(total + 1));
    T->bmp = calloc((size_t)H * nlev * cwd, sizeof(uint64_t));
    T->ls = (H + 7) & ~7;
    T->lv = calloc((size_t)(total + 2) * T->ls, 1);
    if (!T->rec || !T->bmp || !T->lv) die("out of memory for the A index");
#pragma omp parallel for schedule(dynamic, 64)
    for (long long w = 0; w < cwd; w++) {
        double psd[HMAX];
        long long e = (w + 1) * 64; if (e > total) e = total;
        for (long long i = w * 64; i < e; i++) {
            cw y = codes[i];
            psd_skew(y, psd);
            ARec *r = &T->rec[i];
            arec_fill(r, y);
            if (!(i & 4095)) {                      /* spot check against the reference build */
                cw u0, u1; uvals(skew_word(y), NULL, &u0, &u1);
                if (u0 != r->u0 || r->xa != (u1 ^ s2_G(symh_word(tmap(u0))))) die("A record tables disagree with the reference build");
            }
            uint8_t *lv = T->lv + (size_t)i * T->ls;
            for (int k = 0; k < H; k++) {
                T->bmp[((size_t)k * nlev + lev_of(psd[k], nlev)) * cwd + w] |= 1ULL << (i & 63);
                lv[k] = (uint8_t)lev_of(psd[k], FLEV);
            }
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
static void atable_free(ATable *T) { free(T->rec); free(T->bmp); free(T->lv); free(T->sigstart); memset(T, 0, sizeof *T); }

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
/* Index of a pattern within the table of its class.  The patterns of one
   pair differ by the stage-1 kernel, so the index puts the kernel coordinates
   in its low bits: the patterns of a pair then share one group of NCOMB_
   adjacent entries, one or two cache lines instead of NCOMB_.  With the
   kernel basis reduced on pivot bits pi_i (taken among pattern bits 1 ..
   H-QB_-1), kappa_i = t[pi_i], t' = t xor sum kappa_i basis_i, and h the
   remaining bits of t' in that range pushed together, the index is
   (h << NR_) | kappa.  It is linear in t and a bijection on the patterns of
   a class with t_1 = 0 (checked at startup). */
static size_t CCN_ = 0;
static uint64_t *CCIDX_ = NULL;      /* [NBY][256] */
static cw CCINV_[HMAX]; static int CCLB_ = 0;
static uint64_t *CCKM_ = NULL;       /* index of NBCOMB_[m]: kernel coordinates only */
static inline uint64_t cc_index(cw t) {
    uint64_t r = 0;
    for (int b = 0; b < NBY_; b++) r ^= CCIDX_[(size_t)b * 256 + ((t >> (8 * b)) & 0xff)];
    return r;
}
static inline cw cc_pattern(uint64_t idx, int beta) {
    cw t = 0;
    while (idx) { t ^= CCINV_[__builtin_ctzll(idx)]; idx &= idx - 1; }
    t &= LOWMASK_;
    for (int i = 0; i < QB_; i++) t |= (cw)(((beta >> i) & 1) ^ parc(RHO_[i] & t)) << (H - 1 - i);
    return t;
}
static void cc_index_init(void) {
    int lowbits = H - QB_;                       /* pattern bits 0 .. lowbits-1, bit 0 is zero */
    cw rb[HMAX]; int piv[HMAX];
    for (int i = 0; i < NR_; i++) rb[i] = REST_[i];
    cw usedp = 0;
    for (int i = 0; i < NR_; i++) {
        int pr = -1, pos = -1;
        for (int j = 1; j < lowbits && pr < 0; j++)
            for (int r = i; r < NR_; r++) if ((rb[r] >> j) & 1) { pr = r; pos = j; break; }
        if (pr < 0) die("the stage-1 kernel has no pivots among the low pattern bits");
        cw x = rb[pr]; rb[pr] = rb[i]; rb[i] = x;
        for (int r = 0; r < NR_; r++) if (r != i && ((rb[r] >> pos) & 1)) rb[r] ^= rb[i];
        piv[i] = pos; usedp |= (cw)1 << pos;
    }
    CCLB_ = lowbits - 1;                          /* index bits */
    uint64_t img[HMAX]; int rank[HMAX], nf = 0;
    for (int j = 0; j < H; j++) { img[j] = 0; rank[j] = -1; }
    for (int j = 1; j < lowbits; j++) if (!((usedp >> j) & 1)) rank[j] = nf++;
    if (nf + NR_ != CCLB_) die("index bit count is wrong");
    for (int j = 1; j < lowbits; j++) if (rank[j] >= 0) { img[j] = 1ULL << (NR_ + rank[j]); CCINV_[NR_ + rank[j]] = (cw)1 << j; }
    for (int i = 0; i < NR_; i++) {
        uint64_t v = 1ULL << i;
        for (int j = 1; j < lowbits; j++) if (rank[j] >= 0 && ((rb[i] >> j) & 1)) v |= 1ULL << (NR_ + rank[j]);
        img[piv[i]] = v;
        CCINV_[i] = rb[i];
    }
    CCIDX_ = calloc((size_t)NBY_ * 256, sizeof(uint64_t));
    for (int b = 0; b < NBY_; b++)
        for (int v = 1; v < 256; v++) {
            int j = 8 * b + __builtin_ctz((unsigned)v);
            CCIDX_[(size_t)b * 256 + v] = CCIDX_[(size_t)b * 256 + (v & (v - 1))] ^ (j < H ? img[j] : 0);
        }
    CCKM_ = malloc(sizeof(uint64_t) * (size_t)NCOMB_);
    for (int m = 0; m < NCOMB_; m++) {
        CCKM_[m] = cc_index(NBCOMB_[m]);
        if (CCKM_[m] >= (uint64_t)NCOMB_) die("a kernel vector has coset coordinates");
        for (int m2 = 0; m2 < m; m2++) if (CCKM_[m2] == CCKM_[m]) die("kernel coordinates collide");
    }
    for (int it = 0; it < 4096; it++) {
        uint64_t idx = rng() & ((1ULL << CCLB_) - 1);
        int beta = (int)(rng() & (uint64_t)((1 << QB_) - 1));
        cw t = cc_pattern(idx, beta);
        if ((t & 1) || cls_t(t) != beta || cc_index(t) != idx) die("consistency table index is not a bijection on a class");
        int m = (int)(rng() % (uint64_t)NCOMB_);
        if (cc_index(t ^ NBCOMB_[m]) != (idx ^ CCKM_[m])) die("consistency table index is not linear");
    }
}
static void cc_build(int beta) {
    size_t nt = (size_t)1 << CCLB_;
    if (!CC_) {
        CCN_ = nt;
        size_t bytes = nt * (size_t)CCK_ * sizeof(cw), al = bytes >= ((size_t)1 << 22) ? ((size_t)1 << 21) : 64;
        bytes = (bytes + al - 1) & ~(al - 1);
        CC_ = aligned_alloc(al, bytes);
        if (!CC_) die("out of memory for the consistency table");
#ifdef MADV_HUGEPAGE
        if (al > 64) madvise(CC_, bytes, MADV_HUGEPAGE);
#endif
    }
    int bad = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(|:bad)
    for (long long ti = 0; ti < (long long)nt; ti++) {
        cw t = cc_pattern((uint64_t)ti, beta), nv[HMAX], g;
        int c = s2_leftnull(t, nv, HMAX, &g);
        (void)g;
        /* drop the all-ones direction: it is in every left kernel and every
           enumerated pair meets its condition (the row sums), so a vector
           spent on it rejects nothing.  The test stays one-sided whether or
           not that is so: every stored word is a left null vector of M2(t). */
        int star = -1;
        for (int i = 0; i < c; i++) if (nv[i] & 1) { star = i; break; }
        if (star >= 0) {
            for (int i = 0; i < c; i++) if (i != star && (nv[i] & 1)) nv[i] ^= nv[star];
            nv[star] = nv[c - 1]; c--;
        }
        cw *e = CC_ + (size_t)ti * CCK_;
        for (int i = 0; i < CCK_; i++) e[i] = (i < c) ? nv[i] : 0;
        if (!(ti & 8191)) {                         /* spot check: stored words annihilate every column of M2(t) */
            cw col[HMAX]; s2_cols(symh_word(t), col);
            if (cc_index(t) != (uint64_t)ti) bad = 1;
            for (int i = 0; i < CCK_ && i < c; i++) for (int u = 0; u < H; u++) if (parc(e[i] & col[u])) bad = 1;
        }
    }
    if (bad) die("consistency table spot check failed");
}

/* ------------------------------------------------------------- the solve */
typedef struct {
    long long nD, pairs0, pairs, tc, ccpass, cons, cands, found, capped;
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
/* The solve, VW systems at a time, one per vector lane (VB = 32 bytes for
   AVX2, 64 for AVX-512).

   Two unknowns are fixed before the elimination.  The all-ones vector and t
   are in the kernel of M2(t) for every t (reversal of B and C together, swap
   of B and C; checked at startup), so the solutions fall into orbits
   {v, not v, v xor t, not v xor t}.  With t_1 = 0 (R7) and mu the highest set
   bit of t != 0, exactly one member of each orbit has v_1 = 0 and bit mu
   clear: the first holds for one of v, not v and is kept by xor t, the
   second then picks between v and v xor t.  That member is the one R8 and R9
   keep (bit mu clear means v < v xor t).  So the system is solved in the
   H-2 unknowns other than y_1 and bit mu, with those two set to zero: it is
   consistent iff the full system is, and its solutions are the R8/R9
   representatives, one per orbit.  A lane holds the columns of the kept
   unknowns as masks over the lags in slots 0..H-3 (slot s is code bit s+1,
   except that slot mu-1 holds code bit H-1) and the right side in slot H-2.

   Step c takes the lowest row of column c not yet a pivot row and clears
   that row from the later columns and the right side.  Columns before c need
   no update: an earlier pivot column is never read again, and an earlier
   free column is supported on the pivot rows chosen before it, so it has no
   entry in the new pivot row.  The system is consistent iff the right side
   ends supported on the pivot rows; then y_c = bb[pivot row of c] with the
   free variables zero, and each free column f gives the kernel vector
   e_f + sum of e_c over the pivot columns c whose pivot row is set in
   col[f].  Lanes do not interact; an idle lane holds stale data and is
   never read back.

   Patterns with t = 0, and orders with H < 3, go through solve_plain, the
   elimination and enumeration of best_hkt.c. */
#ifndef VB
#define VB 32
#endif
#define VW ((int)(VB / sizeof(cw)))
typedef cw vcw __attribute__((vector_size(VB)));
/* a vector and its lanes; lanes are always read and written through w */
typedef union { vcw v; cw w[VB / sizeof(cw)]; } lanev;
typedef struct {
    lanev col[HMAX + 1], pivb[HMAX], p, part;
    cw t[VB / sizeof(cw)], acode[VB / sizeof(cw)];
    int n;
} Batch;
static int HS_;                                  /* H - 2 */
static inline int msb_of(cw t) { return CWB - 1 - (CWB == 64 ? __builtin_clzll((uint64_t)t) : __builtin_clz((uint32_t)t)); }
static inline void batch_fill(Batch *B, int l, const cw *lc, const cw *ck, cw bb, cw t, cw acode) {
    const int hs = HS_, mu = msb_of(t);
    for (int s = 0; s < hs; s++) B->col[s].w[l] = lc[s + 1] ^ ck[s + 1];
    if (mu < H - 1) B->col[mu - 1].w[l] = lc[H - 1] ^ ck[H - 1];
    B->col[hs].w[l] = bb;
    B->t[l] = t; B->acode[l] = acode;
}
/* slot bits to code bits */
static inline cw slots_to_code(cw x, int mu) {
    const cw y = x << 1, b = (y >> mu) & 1;
    return (y & ~((cw)1 << mu)) | (b << (H - 1));
}
static inline void batch_elim(Batch *B) {
    const int h = HS_;
    vcw p = B->col[0].v ^ B->col[0].v;
    for (int c = 0; c < h; c++) {
        vcw x = B->col[c].v, m = x & ~p, bit = m & (0 - m), R = x ^ bit;
        B->pivb[c].v = bit;
        p |= bit;
        for (int c2 = c + 1; c2 <= h; c2++) {
            vcw a = B->col[c2].v;
            B->col[c2].v = a ^ (R & ~(vcw)((a & bit) == 0));
        }
    }
    B->p.v = p;
    /* y_c = bb[pivot row of c], all lanes at once */
    const vcw bb = B->col[h].v;
    vcw part = p ^ p;
    for (int c = 0; c < h; c++) part |= ~(vcw)((bb & B->pivb[c].v) == 0) & ((cw)1 << c);
    B->part.v = part;
}
/* kernel basis of lane l after batch_elim, in slot bits; returns its size */
static inline int lane_kernel(const Batch *B, int l, cw *basis) {
    cw pv[HMAX]; int fr[HMAX], k = 0;
    for (int c = 0; c < HS_; c++) { pv[c] = B->pivb[c].w[l]; fr[k] = c; k += (pv[c] == 0); }
    for (int i = 0; i < k; i++) {
        cw w = (cw)1 << fr[i], x = B->col[fr[i]].w[l];
        for (int c2 = 0; c2 < HS_; c2++) w |= (cw)((x & pv[c2]) != 0) << c2;
        basis[i] = w;
    }
    return k;
}
/* lag 1 from the code alone: for a skew row u(1) = 1 + the number of sign
   changes between x_u and x_{u+1}, u = 1..H-1 (checked at startup against
   uvals).  Used only to reject a candidate before the full check. */
static inline int dis1(cw y) { return popc((y ^ (y >> 1)) & (MASKH >> 1)); }
/* lags 1 and 2 together, as u(1) + 256 u(2): u(2) = 2 + (sign changes between
   x_u and x_{u+2}, u = 1..H-2) - (x_{H-1} != x_H), for H >= 3 (also checked) */
static inline int dis12(cw y) {
    const cw z = y ^ (y >> 1);
    return popc(z & (MASKH >> 1)) + 256 * (popc((y ^ (y >> 2)) & (MASKH >> 2)) - (int)((z >> (H - 2)) & 1));
}
/* integer equations at every lag: 2 (uA + uB + uC) = 2 (n - uD) */
static inline int verify_bc(uw wa, cw v, cw c, const int *uD) {
    uw wb = skew_word(v), wc = skew_word(c);
    for (int k = 1; k <= H; k++)
        if (popu(wb ^ rotr(wb, k)) + popu(wc ^ rotr(wc, k)) + popu(wa ^ rotr(wa, k)) != 2 * (N - uD[k])) return 0;
    return 1;
}
/* The same equations from the codes alone.  For a skew row with code y and
   z its bit reversal, the products x_j x_{j+k} are of three kinds: both
   indices in 1..H (sign changes between y and y >> k), both of the form
   (j, n-k-j) in H-k+1..H, and both of the form (v, k-v) in 1..k-1; the last
   two carry a minus sign and are counted twice by the xor with z.  This gives
   2 (u(k) - k) = lagf(y, z, k), checked at startup against uvals for every
   lag.  lag_ok rejects on the first lag that fails; whatever it passes is
   confirmed by verify_bc, and a disagreement between the two stops the run. */
static inline int lagf(cw y, cw z, int k) {
    const cw one = 1;
    return 2 * popc((y ^ (y >> k)) & ((one << (H - k)) - 1))
         - popc(((y >> (H - k)) ^ z) & ((one << k) - 1))
         - popc((y ^ (z >> (H - k + 1))) & ((one << (k - 1)) - 1));
}
static inline int lag_ok(cw a, cw v, cw c, const int *uD) {
    const cw za = revH(a), zv = revH(v), zc = revH(c);
    for (int k = 1; k <= H; k++)
        if (lagf(a, za, k) + lagf(v, zv, k) + lagf(c, zc, k) != 2 * (N - uD[k] - 3 * k)) return 0;
    return 1;
}
static void accept_bc(cw acode, cw v, cw c, uint64_t dcode, const int *uD, Stat *st) {
    if (!lag_ok(acode, v, c, uD)) return;
    if (!verify_bc(skew_word(acode), v, c, uD)) die("the code-level lag test passed a quadruple that the PAF check rejects");
    st->found++; sol_add(acode, v, c, dcode);
}
/* one system by the plain elimination of best_hkt.c: all H unknowns, every
   kernel element, R8 and R9 as filters */
static void solve_plain(const cw *colin, cw bb, cw t, cw acode, uint64_t dcode, const int *uD, Stat *st) {
    cw col[HMAX], pivb[HMAX], p = 0;
    for (int u = 0; u < H; u++) col[u] = colin[u];
    for (int c = 0; c < H; c++) {
        cw m = col[c] & ~p, bit = m & (~m + 1);
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
    const uw wa = skew_word(acode);
    cw v = part;
    for (uint64_t g = 0; g < (1ULL << nf); g++) {
        if (g) v ^= basis[__builtin_ctzll(g)];
        if (v & 1) continue;                       /* R8 */
        cw c = v ^ t;
        if (c < v) continue;                       /* R9 */
        st->cands++;
        if (verify_bc(wa, v, c, uD)) { st->found++; sol_add(acode, v, c, dcode); }
    }
}
static void batch_run(Batch *B, uint64_t dcode, const int *uD, Stat *st) {
    if (!B->n) return;
    const int hs = HS_;
    /* idle lanes keep whatever they held: lanes do not interact and only
       lanes below B->n are read back */
    batch_elim(B);
    /* Lanes are first classified without data-dependent branches (consistent
       or not, unique solution or not, lags 1 and 2 of that solution), and
       only the few that need more work are visited again. */
    int hit[VB / sizeof(cw)], nh = 0;
    /* uA + uB + uC = n - uD at lags 1 and 2, in the packed form of dis12 */
    const int tgt12 = (N - 3 - uD[1]) + 256 * (N - 6 - uD[2]);
    for (int l = 0; l < B->n; l++) {
        const cw t = B->t[l], pl = B->p.w[l];
        const int cons = (B->col[hs].w[l] & ~pl) == 0, uniq = cons & (popc(pl) == hs);
        const cw v = slots_to_code(B->part.w[l], msb_of(t)), c = v ^ t;
        st->cons += cons;
        st->cands += uniq;
        hit[nh] = l;
        nh += (uniq & (dis12(v) + dis12(c) + dis12(B->acode[l]) == tgt12)) | (cons & !uniq);
    }
    for (int ih = 0; ih < nh; ih++) {
        const int l = hit[ih];
        const cw t = B->t[l], acode = B->acode[l];
        const int nf = hs - popc(B->p.w[l]), mu = msb_of(t);
        if (nf == 0) {
            const cw v = slots_to_code(B->part.w[l], mu), c = v ^ t;
            accept_bc(acode, v, c, dcode, uD, st);
            continue;
        }
        if (nf + 2 > MAXFREE) { st->capped++; continue; }      /* nullity of the full system */
        cw basis[HMAX];
        lane_kernel(B, l, basis);
        cw x = B->part.w[l];
        for (uint64_t g = 0; g < (1ULL << nf); g++) {
            if (g) x ^= basis[__builtin_ctzll(g)];
            const cw v = slots_to_code(x, mu), c = v ^ t;
            st->cands++;
            if (dis12(v) + dis12(c) + dis12(acode) == tgt12) accept_bc(acode, v, c, dcode, uD, st);
        }
    }
    B->n = 0;
}
/* startup check of the table-built, reduced, batched solve against a plain
   elimination of the reference columns in all H unknowns: same consistency,
   and the orbits of its solutions are exactly the full solution set */
static int cmp_cw(const void *a, const void *b) { cw x = *(const cw *)a, y = *(const cw *)b; return x < y ? -1 : x > y; }
static void solve_selfcheck(void) {
    static Batch B;
    cw tl[VB / sizeof(cw)], xl[VB / sizeof(cw)];
    HS_ = H - 2;
    for (int it = 0; it < 4096; it++) {
        cw y = (cw)rng() & MASKH, x0, x1; int u[HMAX + 1];
        uvals(skew_word(y), u, &x0, &x1);
        if (H >= 1 && u[1] != 1 + dis1(y)) die("lag-1 shortcut disagrees with the PAF");
        if (H >= 3 && dis12(y) != (u[1] - 1) + 256 * (u[2] - 2)) die("lag-2 shortcut disagrees with the PAF");
        for (int k = 1; k <= H; k++) if (lagf(y, revH(y), k) != 2 * (u[k] - k)) die("code-level lag formula disagrees with the PAF");
    }
    if (H < 3) return;
    for (int it = 0; it < 1024; it++) {
        for (int l = 0; l < VW; l++) {
            cw t0; int m = (int)(rng() % (uint64_t)NCOMB_);
            do { t0 = (cw)rng() & MASKH & ~(cw)1; } while (!(t0 ^ NBCOMB_[m]));
            if (it % 7 == 0) t0 = (((cw)1 << (1 + (int)(rng() % (uint64_t)(H - 1)))) ^ NBCOMB_[m]);   /* a single high bit */
            if (!(t0 ^ NBCOMB_[m])) t0 ^= 2;
            tl[l] = t0 ^ NBCOMB_[m];
            ColBuf lc; lc_cols(t0, &lc);
            cw bb = (cw)rng() & MASKH;
            if (it & 1) {                               /* a consistent right side */
                cw y = (cw)rng() & MASKH, col[HMAX]; s2_cols(symh_word(tl[l]), col);
                bb = 0; for (int u = 0; u < H; u++) if ((y >> u) & 1) bb ^= col[u];
            }
            xl[l] = bb;
            batch_fill(&B, l, lc.w, COLK_[m].w, bb, tl[l], 0);
        }
        batch_elim(&B);
        for (int l = 0; l < VW; l++) {
            const cw t = tl[l]; const int mu = msb_of(t);
            cw col[HMAX], chk[HMAX], pivb[HMAX], p = 0, bb = xl[l];
            s2_cols(symh_word(t), col); memcpy(chk, col, sizeof chk);
            cw k1 = 0, k2 = 0;
            for (int u = 0; u < H; u++) { k1 ^= chk[u]; if ((t >> u) & 1) k2 ^= chk[u]; }
            if (k1 || k2) die("the all-ones vector or t is not in the stage-2 kernel");
            for (int c = 0; c < H; c++) {
                cw m = col[c] & ~p, bit = m & (~m + 1);
                pivb[c] = bit;
                if (!bit) continue;
                cw R = col[c] ^ bit;
                p |= bit;
                for (int c2 = 0; c2 < H; c2++) col[c2] ^= R & ((cw)0 - (cw)((col[c2] & bit) != 0));
                bb ^= R & ((cw)0 - (cw)((bb & bit) != 0));
            }
            int cons = !(bb & ~p);
            cw part = 0, basis[HMAX]; int nf = 0;
            for (int c = 0; c < H; c++) if (bb & pivb[c]) part |= (cw)1 << c;
            for (int f = 0; f < H; f++) {
                if (pivb[f]) continue;
                cw w = (cw)1 << f;
                for (int c2 = 0; c2 < H; c2++) if (col[f] & pivb[c2]) w |= (cw)1 << c2;
                basis[nf++] = w;
            }
            int cons2 = (B.col[HS_].w[l] & ~B.p.w[l]) == 0;
            if (cons != cons2) die("reduced solve disagrees on consistency");
            if (!cons) continue;
            cw basis2[HMAX]; int nf2 = lane_kernel(&B, l, basis2);
            if (nf2 != HS_ - popc(B.p.w[l])) die("reduced solve: kernel size is not the corank");
            if (nf != nf2 + 2) die("reduced solve disagrees on the nullity");
            if (nf > 8) continue;
            cw s1[256], s2[256];
            for (int g = 0; g < (1 << nf); g++) {
                cw a = part;
                for (int i = 0; i < nf; i++) if ((g >> i) & 1) a ^= basis[i];
                s1[g] = a;
            }
            for (int g = 0; g < (1 << nf2); g++) {
                cw x = B.part.w[l];
                for (int i = 0; i < nf2; i++) if ((g >> i) & 1) x ^= basis2[i];
                const cw v = slots_to_code(x, mu);
                if ((v & 1) || ((v >> mu) & 1) || (v ^ t) < v) die("reduced solve: a solution is not the R8/R9 representative");
                cw acc = 0; for (int u = 0; u < H; u++) if ((v >> u) & 1) acc ^= chk[u];
                if (acc != xl[l]) die("reduced solve returned a non-solution");
                s2[4 * g] = v; s2[4 * g + 1] = ~v & MASKH; s2[4 * g + 2] = v ^ t; s2[4 * g + 3] = ~(v ^ t) & MASKH;
            }
            qsort(s1, (size_t)1 << nf, sizeof(cw), cmp_cw); qsort(s2, (size_t)1 << nf, sizeof(cw), cmp_cw);
            if (memcmp(s1, s2, sizeof(cw) << nf)) die("reduced solve disagrees on the solution set");
        }
    }
}

static int NOMAX_ = 0;
#define BLKW 512
typedef struct { uint64_t code; cw ud0, tD, gD[2]; int cls; uint8_t uD[HMAX + 1]; } DRec;   /* gD[e] = G(d_e) xor G(0) xor u1 of D */
static int cmp_drec(const void *a, const void *b) {
    const DRec *x = a, *y = b;
    if (x->cls != y->cls) return x->cls < y->cls ? -1 : 1;
    return x->code < y->code ? -1 : x->code > y->code;
}
static void process_d(const DRec *dr, int beta, const ATable *T, Stat *st, int *surv, Batch *B) {
    const uint64_t dcode = dr->code;
    const int key = beta ^ dr->cls;
    long long lo = T->sigstart[key], hi = T->sigstart[key + 1];
    st->nD++;
    if (lo >= hi) return;
    double psd[HMAX];
    psd_sym(dcode, psd);
    const cw ud0 = dr->ud0, tD = dr->tD;
    const int nlev = T->nlev; const long long cwd = T->cwords;
    const uint64_t *bp[HMAX], *bneg = NULL; int nbp = 0, kstar = 0, blev[HMAX];
    /* the same two tests of R3 and R4 at FLEV levels, applied to each pair
       that passes the index: a row passes iff level(PSD_A(k)) <= level(cap_k)
       for every k and level(PSD_A(k*)) >= level(threshold); the level is a
       nondecreasing function, so a row that meets the real inequalities
       (against the relaxed bounds) always passes */
    uint8_t capL[HMAX + 8]; int thrL = 0;
    for (int k = 1; k < H; k++) if (psd[k] < psd[kstar]) kstar = k;
    for (int k = 0; k < H; k++) capL[k] = (uint8_t)lev_of(4.0 * N - 2.0 - psd[k] + 1e-6, FLEV);
    if (!NOMAX_) thrL = lev_of((4.0 * N - psd[kstar]) / 3.0 - 1e-6, FLEV);
    for (int k = 0; k < H; k++) {
        double cap = 4.0 * N - 2.0 - psd[k] + 1e-6;
        if (cap < 1.0) return;
        int l = lev_of(cap, nlev);
        if (l >= nlev - 1) continue;
        /* lowest level first: the scan stops at the first empty word */
        int i = nbp++;
        const uint64_t *b = T->bmp + ((size_t)k * nlev + l) * cwd;
        while (i > 0 && blev[i - 1] > l) { blev[i] = blev[i - 1]; bp[i] = bp[i - 1]; i--; }
        blev[i] = l; bp[i] = b;
    }
    if (!NOMAX_) {
        int l = lev_of((4.0 * N - psd[kstar]) / 3.0 - 1e-6, nlev);
        if (l > 0) bneg = T->bmp + ((size_t)kstar * nlev + (l - 1)) * cwd;
    }
    /* per-D tables, built at the first surviving pair.
       d_e completes a row with (ta & 1) = e to a pattern with t_1 = 0 (R7) */
    int ready = 0, uD[HMAX + 1];
    cw dE[2] = {0, 0}, bld[2][((HMAX + 3) / 4) * 16];
    const cw *gD = dr->gD;
    long long w0 = lo >> 6, wlast = (hi - 1) >> 6;
    uint64_t lomask = ~0ULL << (lo & 63), himask = (hi & 63) ? ((1ULL << (hi & 63)) - 1) : ~0ULL;
    for (long long wb = w0; wb <= wlast; wb += BLKW) {
        double t0 = tcpu();
        long long we = wb + BLKW - 1; if (we > wlast) we = wlast;
        int ns = 0;
        /* one bitmap at a time over the block: each is read as one stream */
        uint64_t vb[BLKW]; const int nw = (int)(we - wb + 1);
        vb[0] = vb[nw - 1] = 0;
        if (bneg) for (int j = 0; j < nw; j++) vb[j] = ~bneg[wb + j];
        else for (int j = 0; j < nw; j++) vb[j] = ~0ULL;
        if (wb == w0) vb[0] &= lomask;
        if (we == wlast) vb[nw - 1] &= himask;
        for (int i = 0; i < nbp; i++) { const uint64_t *b = bp[i] + wb; for (int j = 0; j < nw; j++) vb[j] &= b[j]; }
        for (int j = 0; j < nw; j++) {
            uint64_t v = vb[j];
            while (v) { surv[ns++] = (j << 6) + __builtin_ctzll(v); v &= v - 1; }
        }
        double t1 = tcpu();
        st->t_scan += t1 - t0;
        st->pairs0 += ns;
        if (ns && !ready) {
            for (int k = 1; k <= H; k++) uD[k] = dr->uD[k];
            /* lin[i] = BIL(e_i, tD) from byte tables; the other completion differs by NBSTAR_ */
            cw lin[HMAX], lin2[HMAX];
            for (int i = 0; i < H; i++) lin[i] = 0;
            for (int b = 0; b < NBY_; b++) {
                const cw *q = QTB_ + ((size_t)b * 256 + ((tD >> (8 * b)) & 0xff)) * HMAX;
                for (int i = 0; i < H; i++) lin[i] ^= q[i];
            }
            for (int i = 0; i < H; i++) lin2[i] = lin[i] ^ LSTAR_[i];
            for (int e = 0; e < 2; e++) {
                const int same = (int)(tD & 1) == e;
                dE[e] = same ? tD : (tD ^ NBSTAR_);
                nib_build(bld[e], same ? lin : lin2);
            }
            ready = 1;
        }
        const ARec *base = T->rec + (wb << 6);
        /* look-ahead by LA pairs: the pattern, its table index and the
           prefetch of its table lines are issued one pair at a time, LA pairs
           before the pair is solved (a burst of prefetches would be dropped) */
        enum { LA = 12, RING = 32 };
        cw tbuf[RING]; uint64_t ibuf[RING]; char skip[RING];
        const size_t grp = (size_t)NCOMB_ * CCK_ * sizeof(cw);
        const uint8_t *lvb = T->lv + (size_t)(wb << 6) * T->ls; const int ls = T->ls;
        for (int s = 0; s < 2 * LA && s < ns; s++) { __builtin_prefetch(base + surv[s]); __builtin_prefetch(lvb + (size_t)surv[s] * ls); }
        for (int s = -LA; s < ns; s++) {
            const int sp = s + LA;
            if (sp < ns) {
                if (sp + LA < ns) { __builtin_prefetch(base + surv[sp + LA]); __builtin_prefetch(lvb + (size_t)surv[sp + LA] * ls); }
                const uint8_t *la = lvb + (size_t)surv[sp] * ls;
                int bad = la[kstar] < thrL;
                for (int k = 0; k < H; k++) bad |= la[k] > capL[k];
                skip[sp & (RING - 1)] = (char)bad;
                if (bad) goto prepped;
                const ARec *r = base + surv[sp];
                cw t = r->ta ^ dE[r->ta & 1];
                tbuf[sp & (RING - 1)] = t;
                if (QB_ && cls_t(t) != beta) die("a pattern is outside the class of its consistency table");
                if (CC_) {
                    const uint64_t ix = cc_index(t);
                    ibuf[sp & (RING - 1)] = ix;
                    const char *g = (const char *)(CC_ + (size_t)(ix & ~(uint64_t)(NCOMB_ - 1)) * CCK_);
                    for (size_t o = 0; o < grp; o += 64) __builtin_prefetch(g + o);
                }
            }
        prepped:
            if (s < 0 || skip[s & (RING - 1)]) continue;
            st->pairs++;
            st->tc += NCOMB_;
            {
                const ARec *r = base + surv[s];
                const int e = (int)(r->ta & 1);
                const cw t0p = tbuf[s & (RING - 1)];
                const cw xbase = r->xa ^ (r->u0 & ud0) ^ gD[e] ^ nib_lin(bld[e], r->ta);
                cw bk[16];
                for (int i = 0; i < NR_; i++) bk[i] = nib_lin(BILK_ + (size_t)i * NIB_ * 16, t0p);
                /* right sides of all patterns of the pair and the table test,
                   without a data-dependent branch; then only the survivors */
                cw gm = 0;
                for (int m0 = 0; m0 < NCOMB_; m0 += 64) {
                    cw bbs[64]; uint64_t pass = 0;
                    const int m1 = m0 + 64 < NCOMB_ ? m0 + 64 : NCOMB_;
                    for (int m = m0; m < m1; m++) {
                        const int mg = m ^ (m >> 1);                   /* Gray order over the kernel */
                        if (m) gm ^= bk[__builtin_ctz((unsigned)m)];
                        const cw bb = (xbase ^ GK_[mg] ^ gm) & MASKH;
                        bbs[m - m0] = bb;
                        int bad = 0;
                        if (CC_) {
                            const cw *e2 = CC_ + (size_t)(ibuf[s & (RING - 1)] ^ CCKM_[mg]) * CCK_;
                            for (int i = 0; i < CCK_; i++) bad |= parc(e2[i] & bb);
                        }
                        pass |= (uint64_t)(!bad) << (m - m0);
                    }
                    if (!pass) continue;
                    st->ccpass += __builtin_popcountll(pass);
                    ColBuf lc; lc_cols(t0p, &lc);
                    while (pass) {
                        const int m = m0 + __builtin_ctzll(pass), mg = m ^ (m >> 1);
                        pass &= pass - 1;
                        const ColBuf *ck = &COLK_[mg];
                        const cw t = t0p ^ NBCOMB_[mg];
                        if (__builtin_expect(t == 0 || H < 3, 0)) {
                            cw colp[HMAX];
                            for (int u = 0; u < H; u++) colp[u] = lc.w[u] ^ ck->w[u];
                            solve_plain(colp, bbs[m - m0], t, r->code, dcode, uD, st);
                            continue;
                        }
                        batch_fill(B, B->n, lc.w, ck->w, bbs[m - m0], t, r->code);
                        if (++B->n == VW) batch_run(B, dcode, uD, st);
                    }
                }
            }
        }
        if (wb + BLKW > wlast) batch_run(B, dcode, uD, st);
        st->t_solve += tcpu() - t1;
    }
}

/* ------------------------------------------------------------- D rows */
typedef struct { uint64_t *v; long long n, cap; } DVec;
static void dvec_push(DVec *d, uint64_t x) {
    if (d->n == d->cap) { d->cap = d->cap ? 2 * d->cap : 4096; d->v = realloc(d->v, sizeof(uint64_t) * (size_t)d->cap); if (!d->v) die("out of memory for the D rows"); }
    d->v[d->n++] = x;
}
/* The sorted D list of a census, kept in a file between the units of one
   census (BEST_DFILE): header n, R2 flag, the two counts and the length, then
   the codes.  A file that does not match n and the R2 setting, or is short,
   is ignored and rewritten; it is written under a temporary name and renamed. */
static int dfile_load(const char *path, int nodec, DVec *D, long long *adm, long long *ceil_) {
    if (!path) return 0;
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    long long h[6]; int ok = fread(h, sizeof h, 1, f) == 1 && h[0] == 0x4268327444LL && h[1] == N && h[2] == nodec && h[5] >= 0;
    if (ok) {
        D->v = malloc(sizeof(uint64_t) * (size_t)(h[5] + 1)); D->n = D->cap = h[5];
        ok = D->v && fread(D->v, sizeof(uint64_t), (size_t)h[5], f) == (size_t)h[5] && fgetc(f) == EOF;
        for (long long i = 1; ok && i < h[5]; i++) if (D->v[i - 1] >= D->v[i]) ok = 0;
        if (!ok) { free(D->v); D->v = NULL; D->n = D->cap = 0; }
        else { *adm = h[3]; *ceil_ = h[4]; }
    }
    fclose(f);
    return ok;
}
static void dfile_save(const char *path, int nodec, const DVec *D, long long adm, long long ceil_) {
    if (!path) return;
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.tmp%ld", path, (long)getpid());
    FILE *f = fopen(tmp, "wb"); if (!f) return;
    long long h[6] = {0x4268327444LL, N, nodec, adm, ceil_, D->n};
    int ok = fwrite(h, sizeof h, 1, f) == 1 && fwrite(D->v, sizeof(uint64_t), (size_t)D->n, f) == (size_t)D->n;
    ok = (fclose(f) == 0) && ok;
    if (ok) rename(tmp, path); else remove(tmp);
}
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void fput_row(FILE *f, uw w) { for (int j = 0; j < N; j++) fputc(((w >> j) & 1) ? '-' : '+', f); }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: best_hkt2 <n> <threads> <out.txt>\n"); return 1; }
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
    double tstart = wnow(), tick0 = tcpu();
    stage_init();
    fast_init();
    solve_selfcheck();
    trig_init();
    dec_init();
    { const char *v = getenv("BEST_NOMAX"); if (v && atoi(v)) NOMAX_ = 1; }
    int nodec = 0; { const char *v = getenv("BEST_NODEC"); if (v && atoi(v)) nodec = 1; }
    { const char *v = getenv("BEST_NLEV"); if (v) NLEV_ = atoi(v); }
    double mem_mb = 1000, cc_mb = 1100;
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
    } else if (!probe && dfile_load(getenv("BEST_DFILE"), nodec, &D, &d_adm, &d_ceil)) {
        printf("   D rows: %lld with the row sum, %lld under the ceiling, %lld kept%s (read from %s)\n",
               d_adm, d_ceil, D.n, nodec ? "" : " as orbit minima", getenv("BEST_DFILE"));
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
        dfile_save(getenv("BEST_DFILE"), nodec, &D, d_adm, d_ceil);
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

    /* ---- consistency table: words per pattern and class bits ---- */
    double t_cc = 0;
    {
        int kk = -1, qb = -1; const char *v = getenv("BEST_CCK"); if (v) kk = atoi(v);
        v = getenv("BEST_QB"); if (v) qb = atoi(v);
        long hist[HMAX + 1] = {0}; int ns = 4096; cw tmp[HMAX];
        for (int i = 0; i < ns; i++) hist[s2_leftnull((cw)(rng() & MASKH) & ~(cw)1, tmp, HMAX, NULL)]++;
        printf("   stage-2 corank over %d random patterns:", ns);
        for (int k = 0; k <= H; k++) if (hist[k]) printf(" %d:%.1f%%", k, 100.0 * hist[k] / ns);
        printf("\n");
        if (kk < 0) kk = 2;
        if (kk > H) kk = H;
        if (kk == 0) qb = 0;
        if (qb < 0) {
            qb = 0;
            while (qb < H / 3 && qb < 12 && (double)((size_t)1 << (H - 1 - qb)) * kk * sizeof(cw) / 1048576.0 > cc_mb) qb++;
            if ((double)((size_t)1 << (H - 1 - qb)) * kk * sizeof(cw) / 1048576.0 > cc_mb) die("the consistency table does not fit BEST_CC_MB; raise it or set BEST_CCK=0");
        }
        CCK_ = kk;
        class_init(qb);
        arec_init();
        if (CCK_ > 0) cc_index_init();
        if (CCK_ > 0) printf("   consistency table: %d words per pattern, %d classes of %.1f MB\n", CCK_, 1 << QB_,
                             (double)((size_t)1 << (H - 1 - QB_)) * CCK_ * sizeof(cw) / 1048576.0);
        else printf("   no consistency table: every pattern of every pair is eliminated\n");
    }
    fflush(stdout);

    /* ---- D records: R6 is a condition on D alone ---- */
    long long d_all = D.n;
    int probe_cls = 0;                              /* probe only: keep the D rows of the first classes */
    { const char *v = getenv("BEST_PROBE_CLS"); if (v && probe) probe_cls = atoi(v); }
    if (probe_cls > 0 && probe_cls < (1 << QB_)) d_all = 0; else probe_cls = 0;
    DRec *DR = malloc(sizeof(DRec) * (size_t)(D.n + 1)); long long nDR = 0;
    if (!DR) die("out of memory for the D records");
    for (long long i = 0; i < D.n; i++) {
        int uD[HMAX + 1]; cw ud0, ud1;
        uvals(sym_word(D.v[i]), uD, &ud0, &ud1);
        if (probe_cls) { if (cls_r((cw)(~ud0 & MASKH)) >= probe_cls) continue; d_all++; }
        if ((sig_of(ud0) ^ sig_of(MASKH)) != ASIG_) continue;
        DRec *r = &DR[nDR++];
        r->code = D.v[i]; r->ud0 = ud0;
        r->tD = tmap((cw)(~ud0 & MASKH));
        r->cls = cls_r((cw)(~ud0 & MASKH));
        for (int e = 0; e < 2; e++) {
            cw de = ((int)(r->tD & 1) == e) ? r->tD : (r->tD ^ NBSTAR_);
            r->gD[e] = s2_G(symh_word(de)) ^ G0_ ^ ud1;
        }
        r->uD[0] = 0;
        for (int k = 1; k <= H; k++) r->uD[k] = (uint8_t)uD[k];
    }
    /* rows of one class scan the same range of every A piece: keep them together */
    qsort(DR, (size_t)nDR, sizeof(DRec), cmp_drec);
    printf("   D rows with the signature of a skew row (R6): %lld of %lld\n", nDR, d_all);
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
            double mb = rows * (sizeof(ARec) + ((H + 7) & ~7) + (double)H * 32 / 8.0) / 1048576.0;   /* at 32 levels */
            pieces = (int)ceil(mb / mem_mb);
        }
        if (pieces < 1) pieces = 1;
        if (pieces > nhalf) pieces = (int)nhalf;
    }

    Stat tot = {0};
    long long arows = 0; double t_table = 0, idx_mb_max = 0; int nlev_used = 0;
    double t_range = 0; long long n_range = 0;
    if (probe) {
        /* cost of the real A build: a range of probeA consecutive half-codes at a random offset */
        n_range = probeA < nhalf ? probeA : nhalf;
        long long off = (n_range < nhalf) ? (long long)(rng() % (uint64_t)(nhalf - n_range)) : 0;
        ATable T = {0};
        double tb = wnow();
        atable_build(&T, off, off + n_range, NULL, 0, mem_mb);
        t_range = wnow() - tb;
        printf("   PROBE A build: %lld consecutive codes, %lld rows (%.4f), %d levels, %.2fs wall with %d threads\n",
               n_range, T.n, (double)T.n / (double)n_range, T.nlev, t_range, threads);
        atable_free(&T);
    }
    int nslice = 1 << QB_;
    if (probe && nslice > 1) nslice = 1;            /* a probe samples one class */
    int only_class = -1, only_piece = -1;
    if (!probe && !plant) {
        const char *v = getenv("BEST_CLASS");
        if (v) { only_class = atoi(v); if (only_class < 0 || only_class >= nslice) die("BEST_CLASS out of range"); }
        v = getenv("BEST_PIECE");
        if (v) {
            if (!getenv("BEST_PIECES")) die("BEST_PIECE needs BEST_PIECES");
            only_piece = atoi(v); if (only_piece < 0 || only_piece >= pieces) die("BEST_PIECE out of range");
        }
        if (only_class >= 0 || only_piece >= 0) {
            printf("   UNIT of a census:");
            if (only_class >= 0) printf(" class %d of %d", only_class, nslice);
            if (only_piece >= 0) printf(" A piece %d of %d", only_piece, pieces);
            printf(" (0-based)\n"); fflush(stdout);
        }
    }
    int first_beta = 1;
    for (int beta = 0; beta < nslice; beta++) {
        if (only_class >= 0 && beta != only_class) continue;
        if (CCK_ > 0) {
            double tb = wnow();
            cc_build(beta);
            t_cc += wnow() - tb;
            if (first_beta || (1 << QB_) <= 16) { printf("   class %d of %d: consistency table built in %.1fs\n", beta + 1, 1 << QB_, wnow() - tb); fflush(stdout); }
        }
        for (int pc = 0; pc < pieces; pc++) {
            if (only_piece >= 0 && pc != only_piece) continue;
            ATable T = {0};
            double tb = wnow();
            atable_build(&T, nhalf * pc / pieces, nhalf * (pc + 1) / pieces, alist, probeA, mem_mb);
            t_table += wnow() - tb;
            if (first_beta) arows += T.n;
            nlev_used = T.nlev;
            double mb = ((double)T.n * (sizeof(ARec) + T.ls) + (double)H * T.nlev * T.cwords * 8.0) / 1048576.0;
            if (mb > idx_mb_max) idx_mb_max = mb;
            if (first_beta) { printf("   A piece %d of %d: %lld rows, %d levels, %.1f MB, built in %.1fs\n", pc + 1, pieces, T.n, T.nlev, mb, wnow() - tb); fflush(stdout); }
            if (T.n > 0x7fffffffLL - 64) die("A piece too large for int offsets; raise BEST_PIECES");
#pragma omp parallel
            {
                Stat st = {0};
                int *surv = malloc(sizeof(int) * BLKW * 64);
                Batch *B = aligned_alloc(64, (sizeof(Batch) + 63) & ~(size_t)63);
                memset(B, 0, sizeof(Batch));
#pragma omp for schedule(dynamic, 8) nowait
                for (long long di = 0; di < nDR; di++) process_d(&DR[di], beta, &T, &st, surv, B);
                free(surv); free(B);
#pragma omp critical(statmerge)
                {
                    tot.pairs0 += st.pairs0; tot.pairs += st.pairs; tot.tc += st.tc; tot.ccpass += st.ccpass; tot.cons += st.cons;
                    tot.cands += st.cands; tot.found += st.found; tot.capped += st.capped;
                    tot.t_scan += st.t_scan; tot.t_solve += st.t_solve;
                }
            }
            atable_free(&T);
        }
        first_beta = 0;
        if ((1 << QB_) > 16 && !probe && only_class < 0 && ((beta + 1) & ((1 << (QB_ - 4)) - 1)) == 0) {
            printf("   classes done: %d of %d, %.0fs, %lld solutions so far\n", beta + 1, 1 << QB_, wnow() - tstart, tot.found); fflush(stdout);
        }
    }
    double telapsed = wnow() - tstart;
    if (TCPU_TICKS) { double per = telapsed / (tcpu() - tick0); tot.t_scan *= per; tot.t_solve *= per; }

    long long acodes = (probe || plant) ? probeA : nhalf;
    double scale = probe ? (double)(1 << QB_) : 1.0;  /* a probe saw one class of the A sample per D row */
    printf("   A rows under the ceiling: %lld of %lld (%.4f)\n", arows, acodes, (double)arows / (double)acodes);
    printf("   pairs (A, D) past the index: %lld, past the %d-level test: %lld (%.3g per D row, %.3g of all pairs)\n", tot.pairs0, FLEV, tot.pairs,
           d_all ? scale * (double)tot.pairs / (double)d_all : 0.0, (d_all && arows) ? scale * (double)tot.pairs / ((double)d_all * (double)arows) : 0.0);
    printf("   patterns %lld, past the consistency table %lld (%.2f%%), consistent systems %lld (%.3f%%), candidates verified %lld\n",
           tot.tc, tot.ccpass, tot.tc ? 100.0 * tot.ccpass / (double)tot.tc : 0.0, tot.cons,
           tot.tc ? 100.0 * tot.cons / (double)tot.tc : 0.0, tot.cands);
    printf("   thread-seconds: A table %.2f (wall) | consistency table %.2f (wall) | index scan %.2f | test and solve %.2f; %.0f ns per pair, %.0f ns per pattern\n",
           t_table, t_cc, tot.t_scan, tot.t_solve, tot.pairs ? 1e9 * tot.t_solve / (double)tot.pairs : 0.0,
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
        double nDsig = nD * (d_all ? (double)nDR / (double)d_all : 0.0);
        double nA = (double)nhalf * (double)arows / (double)acodes;
        double f = (d_all && arows) ? scale * (double)tot.pairs / ((double)d_all * (double)arows) : 0.0;
        double ppairs = nD * nA * f, ncls = (double)(1 << QB_);
        printf("   PROBE projection at %d levels: %.3g D orbit minima (%.3g with the signature), %.3g A rows, %.3g pairs, %.3g patterns\n",
               nlev_used, nD, nDsig, nA, ppairs, ppairs * NCOMB_);
        /* each D row with the signature scans every A row once over all classes */
        double c_scan = nDR ? tot.t_scan / ((double)nDR * ((double)arows / scale / 64.0 + 1)) : 0.0;
        double p_scan = nDsig * nA / 64.0 * c_scan;
        double p_solve = ppairs * (tot.pairs ? tot.t_solve / (double)tot.pairs : 0.0);
        double p_atab = ncls * (double)nhalf * (t_range * threads / (double)n_range);
        double p_cc = CCK_ ? (double)nhalf * (t_cc * threads / (double)CCN_) : 0.0;
        double p_tot = p_scan + p_solve + p_atab + p_cc;
        printf("   PROBE cost in thread-seconds (this host): scan %.3g + solve %.3g + A tables (%d sweeps) %.3g + consistency tables %.3g = %.3g (%.3g thread-years); found in sample %lld\n",
               p_scan, p_solve, 1 << QB_, p_atab, p_cc, p_tot, p_tot / 31557600.0, tot.found);
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
