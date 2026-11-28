/* best_cover.c -- independent coverage counts for the class split of the
 * best-matrix census (best_hkt2.c, BEST_QB=q).  Written from the definitions
 * below; it shares no code with best_hkt2.c.
 *
 *   build: gcc -O3 -march=native -fopenmp best_cover.c -lm -o best_cover
 *   run:   ./best_cover <n> <q> <threads> [d|a|da]
 *          (d: the D rows only, a: the A rows only, da: both, the default)
 *
 * Rows.  H = (n-1)/2.  A skew row x has x_0 = 1 and x_{n-u} = -x_u; its code
 * y has bit u-1 set iff x_u = -1 (u = 1..H).  A symmetric row d has
 * d_{n-u} = d_u; its code has bit 0 set iff d_0 = -1 and bit u set iff
 * d_u = -1.  For a row x, u_k(x) = (n - PAF_x(k))/4 (k = 1..H) and u0(x) is the
 * H-bit word whose bit k-1 is the parity of u_k(x).  Every PAF here is
 * computed from the +-1 entries.
 *
 * D list (R1, R2, R3 of best_hkt2.c).  Symmetric rows with row sum s, s^2 =
 * 4n - 3, s > 0; PSD_D(k) <= 4n - 3 at k = 1..H; least code in the orbit of
 * the multipliers j -> t j, t a unit.  Then R6: with M1 the H x H matrix over
 * GF(2) whose column u-1 is u0(e_u) xor u0(0) for the skew rows of code e_u
 * (bit u-1 alone) and 0, a pair (A, D) has stage-1 parity conditions that can
 * hold iff r = u0(A) xor not u0(D) lies in the column space of M1.  u0(A) =
 * M1 y xor u0(0) (checked on random rows), so the condition bears on D alone.
 *
 * A list (R3, R5).  Skew rows with x_1 = +1 (code bit 0 clear) and PSD_A(k)
 * <= 4n - 2 at k = 1..H.
 *
 * Classes (best_hkt2.c, C6).  Gaussian elimination on the rows of M1 (row k-1
 * is lag k, bit u-1 is pattern bit u): for i = 0..q-1, column c = H-1-i, the
 * pivot is the first row not yet used with bit c set, and that row is added to
 * every other row with bit c set while the same additions are recorded on the
 * identity.  rho_i is the pivot row after the q steps and psi_i the recorded
 * combination, so rho_i = psi_i M1.  The label of a row word r is the q bits
 * (psi_i . r); cls(A) = label(u0(A)), cls(D) = label(not u0(D)), and the pair
 * (A, D) belongs to class cls(A) xor cls(D).  Every pair has exactly one class.
 *
 * Output: the class forms in hex, the D counts at each step, the A count, the
 * number of rows within 1e-6 of a spectral bound, each such A row with its
 * largest PSD (line "anear <code> <psd>"), and the number of D and A rows of
 * every label.
 * The number of (A, D) pairs of class b is the sum over labels l of
 * nD(l) nA(l xor b).
 *
 * The spectra are evaluated in double precision.  A skew row has PSD
 * 1 + 4 s_k^2 with s_k = sum_u x_u sin(2 pi u k / n); a symmetric row has
 * DFT d_0 + 2 sum_u d_u cos(2 pi u k / n).  Each test is made against the bound
 * plus 1e-6, as in best_hkt2.c, and the rows within 1e-6 of the bound are
 * counted separately.  The rounding error of a PSD here is below 1e-11, so
 * the relaxed test keeps every row under the bound; a row it keeps above the
 * bound only adds work, never removes a solution.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NMAX 125
#define HMAX 62
#define QMAX 16
typedef uint64_t word;

static int n, H, Q;
static word MASK;

static int parity(word x) { return __builtin_parityll(x); }

/* ---------------------------------------------------------------- PAF */
static void skew_entries(word y, int *x) {
    x[0] = 1;
    for (int u = 1; u <= H; u++) {
        int v = ((y >> (u - 1)) & 1) ? -1 : 1;
        x[u] = v;
        x[n - u] = -v;
    }
}
static void sym_entries(word code, int *x) {
    x[0] = (code & 1) ? -1 : 1;
    for (int u = 1; u <= H; u++) {
        int v = ((code >> u) & 1) ? -1 : 1;
        x[u] = v;
        x[n - u] = v;
    }
}
static word u0_of(const int *x) {
    word r = 0;
    for (int k = 1; k <= H; k++) {
        int p = 0;
        for (int j = 0; j < n; j++) p += x[j] * x[(j + k) % n];
        if ((n - p) % 4) { fprintf(stderr, "PAF not congruent to n mod 4\n"); exit(1); }
        r |= (word)(((n - p) / 4) & 1) << (k - 1);
    }
    return r;
}
static word u0_skew(word y) { int x[NMAX]; skew_entries(y, x); return u0_of(x); }
static word u0_sym(word code) { int x[NMAX]; sym_entries(code, x); return u0_of(x); }

/* ---------------------------------------------------------------- GF(2) */
static word M1col[HMAX], M1row[HMAX], U0ZERO;
static word basis[HMAX]; static int nbasis;            /* column space, distinct leading bits */
static word PSI[QMAX], RHO[QMAX];
static int in_colspace(word r) {
    for (int i = 0; i < nbasis; i++) if ((r ^ basis[i]) < r) r ^= basis[i];
    return r == 0;
}
static int label(word r) { int c = 0; for (int i = 0; i < Q; i++) c |= parity(PSI[i] & r) << i; return c; }
static word mul(word t) { word r = 0; for (int k = 0; k < H; k++) r |= (word)parity(M1row[k] & t) << k; return r; }
static uint64_t rs = 0x2545F4914F6CDD1DULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static void gf2_init(void) {
    U0ZERO = u0_skew(0);
    for (int u = 0; u < H; u++) M1col[u] = u0_skew((word)1 << u) ^ U0ZERO;
    for (int k = 0; k < H; k++) {
        M1row[k] = 0;
        for (int u = 0; u < H; u++) M1row[k] |= ((M1col[u] >> k) & 1) << u;
    }
    for (int it = 0; it < 2000; it++) {           /* u0 of a skew row is affine in its code */
        word y = rnd() & MASK;
        if (u0_skew(y) != (mul(y) ^ U0ZERO)) { fprintf(stderr, "u0 of a skew row is not affine\n"); exit(1); }
    }
    nbasis = 0;
    for (int u = 0; u < H; u++) {
        word v = M1col[u];
        for (int i = 0; i < nbasis; i++) if ((v ^ basis[i]) < v) v ^= basis[i];
        if (!v) continue;
        basis[nbasis++] = v;
        for (int i = nbasis - 1; i > 0 && basis[i] > basis[i - 1]; i--) { word t = basis[i]; basis[i] = basis[i - 1]; basis[i - 1] = t; }
    }
    word a[HMAX], id[HMAX]; int used[HMAX] = {0};
    for (int k = 0; k < H; k++) { a[k] = M1row[k]; id[k] = (word)1 << k; }
    for (int i = 0; i < Q; i++) {
        int c = H - 1 - i, p = -1;
        for (int r = 0; r < H; r++) if (!used[r] && ((a[r] >> c) & 1)) { p = r; break; }
        if (p < 0) { fprintf(stderr, "no pivot for class form %d\n", i); exit(1); }
        used[p] = 1;
        for (int r = 0; r < H; r++) if (r != p && ((a[r] >> c) & 1)) { a[r] ^= a[p]; id[r] ^= id[p]; }
        RHO[i] = (word)p;                         /* pivot row index for now */
    }
    for (int i = 0; i < Q; i++) { int p = (int)RHO[i]; RHO[i] = a[p]; PSI[i] = id[p]; }
    for (int i = 0; i < Q; i++) {                 /* rho_i = psi_i M1 and the reduced shape */
        word r = 0;
        for (int k = 0; k < H; k++) if ((PSI[i] >> k) & 1) r ^= M1row[k];
        if (r != RHO[i]) { fprintf(stderr, "rho is not psi M1\n"); exit(1); }
        if ((RHO[i] >> (H - Q)) != ((word)1 << (Q - 1 - i))) { fprintf(stderr, "class form not reduced\n"); exit(1); }
    }
}

/* ---------------------------------------------------------------- D list */
static double COSD[HMAX + 1][HMAX];             /* cos(2 pi u k / n), k = 1..H at index k-1 */
static int inv_[NMAX];
static int fold(long v) { v %= n; if (v < 0) v += n; return v > H ? n - (int)v : (int)v; }
static long long d_all, d_ceil, d_near, d_min, d_sig;
static long long *dlab;
static int D0, MNEG;

static int orbit_least(const int *pos, int m, word mask) {
    for (int t = 2; t < n; t++) {
        if (!inv_[t]) continue;
        word img = 0;
        for (int i = 0; i < m; i++) img |= (word)1 << (fold((long)inv_[t] * pos[i]) - 1);
        if (img < mask) return 0;
    }
    return 1;
}
static void d_leaf(const double *c, const int *pos, word mask, long long *la, long long *lc, long long *ln, long long *lm, long long *lsig, long long *lab) {
    (*la)++;
    double mx = 0;
    for (int k = 0; k < H; k++) { double dft = (double)D0 - 1.0 - 4.0 * c[k]; if (dft * dft > mx) mx = dft * dft; }
    if (fabs(mx - (4.0 * n - 3.0)) <= 1e-6) (*ln)++;
    if (mx > 4.0 * n - 3.0 + 1e-6) return;
    (*lc)++;
    if (!orbit_least(pos, MNEG, mask)) return;
    (*lm)++;
    word code = (mask << 1) | (D0 < 0 ? 1 : 0);
    word r = ~u0_sym(code) & MASK;
    if (!in_colspace(r ^ U0ZERO)) return;
    (*lsig)++;
    lab[label(r)]++;
}
static void d_rec(int depth, int start, const double *c, int *pos, word mask, long long *la, long long *lc, long long *ln, long long *lm, long long *lsig, long long *lab) {
    if (depth == MNEG) { d_leaf(c, pos, mask, la, lc, ln, lm, lsig, lab); return; }
    double nc[HMAX];
    for (int u = start; u <= H - (MNEG - depth) + 1; u++) {
        for (int k = 0; k < H; k++) nc[k] = c[k] + COSD[u][k];
        pos[depth] = u;
        d_rec(depth + 1, u + 1, nc, pos, mask | ((word)1 << (u - 1)), la, lc, ln, lm, lsig, lab);
    }
}
static void d_list(void) {
    int s = (int)lround(sqrt(4.0 * n - 3.0));
    if (s * s != 4 * n - 3) { fprintf(stderr, "4n-3 is not a square\n"); exit(1); }
    for (int u = 1; u <= H; u++) for (int k = 1; k <= H; k++) COSD[u][k - 1] = cos(2.0 * M_PI * u * k / n);
    for (int t = 1; t < n; t++) for (int v = 1; v < n; v++) if ((long)t * v % n == 1) inv_[t] = v;
    dlab = calloc((size_t)1 << Q, sizeof(long long));
    /* row sum d_0 + 2 (H - 2 m) = s with m entries -1 among d_1..d_H */
    for (D0 = 1; D0 >= -1; D0 -= 2) {
        int num = D0 + 2 * H - s;
        if (num < 0 || num % 4) continue;
        MNEG = num / 4;
        if (MNEG > H) continue;
        printf("D rows: d_0 = %+d with %d entries -1 among d_1..d_H\n", D0, MNEG);
        fflush(stdout);
#pragma omp parallel
        {
            long long la = 0, lc = 0, ln = 0, lm = 0, lsig = 0;
            long long *lab = calloc((size_t)1 << Q, sizeof(long long));
            int pos[HMAX];
#pragma omp for schedule(dynamic, 1) collapse(2)
            for (int u1 = 1; u1 <= H; u1++)
                for (int u2 = 1; u2 <= H; u2++) {
                    if (MNEG < 2) {
                        if (u2 != 1) continue;
                        double c[HMAX] = {0};
                        if (MNEG == 0) { if (u1 == 1) d_rec(0, 1, c, pos, 0, &la, &lc, &ln, &lm, &lsig, lab); continue; }
                        for (int k = 0; k < H; k++) c[k] = COSD[u1][k];
                        pos[0] = u1;
                        d_rec(1, u1 + 1, c, pos, (word)1 << (u1 - 1), &la, &lc, &ln, &lm, &lsig, lab);
                        continue;
                    }
                    if (u2 <= u1 || u2 > H - (MNEG - 2)) continue;
                    double c[HMAX];
                    for (int k = 0; k < H; k++) c[k] = COSD[u1][k] + COSD[u2][k];
                    pos[0] = u1; pos[1] = u2;
                    d_rec(2, u2 + 1, c, pos, ((word)1 << (u1 - 1)) | ((word)1 << (u2 - 1)), &la, &lc, &ln, &lm, &lsig, lab);
                }
#pragma omp critical
            {
                d_all += la; d_ceil += lc; d_near += ln; d_min += lm; d_sig += lsig;
                for (int l = 0; l < (1 << Q); l++) dlab[l] += lab[l];
            }
            free(lab);
        }
    }
}

/* ---------------------------------------------------------------- A list */
static long long a_codes, a_ceil, a_near, a_near_all;
static long long *alab;
#define NEARMAX 4096
static word a_near_code[NEARMAX]; static double a_near_psd[NEARMAX];
static void a_list(void) {
    /* free bits are 1..H-1; low part bits 1..L, high part bits L+1..H-1 */
    int L = (H - 1) / 2 < 16 ? (H - 1) / 2 : 16, HB = H - 1 - L;
    int KP = (H + 3) & ~3;                        /* frequencies padded to a multiple of 4 */
    double base[HMAX], sn[HMAX + 1][HMAX];
    for (int k = 1; k <= H; k++) {
        base[k - 1] = 0;
        for (int u = 1; u <= H; u++) { sn[u][k - 1] = sin(2.0 * M_PI * u * k / n); base[k - 1] += sn[u][k - 1]; }
    }
    size_t nl = (size_t)1 << L, nh = (size_t)1 << HB;
    double *TL = aligned_alloc(64, ((nl * KP * sizeof(double)) + 63) & ~(size_t)63);
    int *labL = malloc(nl * sizeof(int));
    int labbit[HMAX];
    for (int u = 0; u < H; u++) labbit[u] = label(M1col[u]);
    int lab0 = label(U0ZERO);
    for (size_t l = 0; l < nl; l++) {
        int lb = 0;
        for (int k = 0; k < KP; k++) TL[l * KP + k] = 0;
        for (int b = 0; b < L; b++)
            if ((l >> b) & 1) {
                int u = b + 2;                    /* code bit b+1 is position u = b+2 */
                for (int k = 0; k < H; k++) TL[l * KP + k] -= 2.0 * sn[u][k];
                lb ^= labbit[b + 1];
            }
        labL[l] = lb;
    }
    for (int it = 0; it < 2000; it++) {           /* the label of A from the tables against direct PAFs */
        word y = (rnd() & MASK) & ~(word)1;
        int lb = lab0;
        for (int u = 0; u < H; u++) if ((y >> u) & 1) lb ^= labbit[u];
        if (lb != label(u0_skew(y))) { fprintf(stderr, "label of A is not affine\n"); exit(1); }
    }
    const double lim = (4.0 * n - 3.0) / 4.0;     /* 1 + 4 s^2 <= 4n - 2 */
    const double hi = lim + 0.25e-6, lo = lim - 0.25e-6;
    alab = calloc((size_t)1 << Q, sizeof(long long));
#pragma omp parallel
    {
        long long lc = 0, ln = 0;
        long long *lab = calloc((size_t)1 << Q, sizeof(long long));
        double sh[HMAX + 4] __attribute__((aligned(64)));
#pragma omp for schedule(dynamic, 16)
        for (size_t h = 0; h < nh; h++) {
            int lbh = lab0;
            for (int k = 0; k < KP; k++) sh[k] = k < H ? base[k] : 0;
            for (int b = 0; b < HB; b++)
                if ((h >> b) & 1) {
                    int u = L + b + 2;
                    for (int k = 0; k < H; k++) sh[k] -= 2.0 * sn[u][k];
                    lbh ^= labbit[L + b + 1];
                }
            for (size_t l = 0; l < nl; l++) {
                const double *t = TL + l * KP;
                double mx = 0;
                int k = 0;
                for (; k < KP; k += 4) {
                    double v0 = sh[k] + t[k], v1 = sh[k + 1] + t[k + 1], v2 = sh[k + 2] + t[k + 2], v3 = sh[k + 3] + t[k + 3];
                    double m01 = fmax(v0 * v0, v1 * v1), m23 = fmax(v2 * v2, v3 * v3), m = fmax(m01, m23);
                    if (m > hi) break;
                    if (m > mx) mx = m;
                }
                if (k < KP) continue;
                lc++;
                if (mx >= lo) {
                    ln++;
#pragma omp critical(anear)
                    if (a_near < NEARMAX) {
                        word y = ((word)l << 1) | ((word)h << (L + 1));
                        a_near_code[a_near] = y; a_near_psd[a_near] = 1.0 + 4.0 * mx; a_near++;
                    }
                }
                lab[lbh ^ labL[l]]++;
            }
        }
#pragma omp critical
        {
            a_ceil += lc; a_near_all += ln;
            for (int l = 0; l < (1 << Q); l++) alab[l] += lab[l];
        }
        free(lab);
    }
    a_codes = (long long)1 << (H - 1);
    free(TL); free(labL);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: best_cover <n> <q> <threads> [d|a|da]\n"); return 2; }
    n = atoi(argv[1]); Q = atoi(argv[2]);
    const char *what = argc > 4 ? argv[4] : "da";
#ifdef _OPENMP
    omp_set_num_threads(atoi(argv[3]));
#endif
    if (n < 7 || n > NMAX || n % 2 == 0 || Q < 0 || Q > QMAX) { fprintf(stderr, "unsupported n or q\n"); return 2; }
    H = (n - 1) / 2;
    if (Q > H - 1) { fprintf(stderr, "q too large\n"); return 2; }
    MASK = ((word)1 << H) - 1;
    gf2_init();
    printf("n %d H %d q %d stage-1 rank %d corank %d\n", n, H, Q, nbasis, H - nbasis);
    for (int i = 0; i < Q; i++) printf("form %d psi %016llx rho %016llx\n", i, (unsigned long long)PSI[i], (unsigned long long)RHO[i]);
    fflush(stdout);
    if (strchr(what, 'd')) {
        d_list();
        printf("D with the row sum %lld\nD under the ceiling %lld\nD near the ceiling %lld\nD orbit minima %lld\nD with the signature %lld\n",
               d_all, d_ceil, d_near, d_min, d_sig);
        for (int l = 0; l < (1 << Q); l++) printf("dlabel %d %lld\n", l, dlab[l]);
        fflush(stdout);
    }
    if (strchr(what, 'a')) {
        a_list();
        for (long long i = 1; i < a_near; i++)       /* sort the near rows by code */
            for (long long j = i; j > 0 && a_near_code[j] < a_near_code[j - 1]; j--) {
                word c = a_near_code[j]; a_near_code[j] = a_near_code[j - 1]; a_near_code[j - 1] = c;
                double p = a_near_psd[j]; a_near_psd[j] = a_near_psd[j - 1]; a_near_psd[j - 1] = p;
            }
        printf("A codes %lld\nA under the ceiling %lld\nA near the ceiling %lld\n", a_codes, a_ceil, a_near_all);
        for (long long i = 0; i < a_near && i < NEARMAX; i++) printf("anear %016llx %.9f\n", (unsigned long long)a_near_code[i], a_near_psd[i]);
        for (int l = 0; l < (1 << Q); l++) printf("alabel %d %lld\n", l, alab[l]);
    }
    return 0;
}
