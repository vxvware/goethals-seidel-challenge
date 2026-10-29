/*
 * williamson_c.c -- exhaustive census of Williamson quadruples of odd order n.
 *
 * Williamson matrices are four symmetric circulant {-1,+1} matrices A, B, C, D
 * of odd order n with A^2 + B^2 + C^2 + D^2 = 4n I.  With PAF_x(s) the
 * periodic autocorrelation and PSD_x(k) = |sum_j x_j w^{jk}|^2:
 *
 *     PAF_A(s) + PAF_B(s) + PAF_C(s) + PAF_D(s) = 0     for s = 1 .. (n-1)/2,
 *     PSD_A(k) + PSD_B(k) + PSD_C(k) + PSD_D(k) = 4n    for every k,
 *
 * and the row sums are odd with squares summing to 4n.
 *
 * Every sequence is normalised to x_0 = +1 (negating a block is a symmetry),
 * which leaves 2^h symmetric sequences, h = (n-1)/2.  Those whose spectrum
 * exceeds 4n anywhere are dropped, the rest are bucketed by |row sum|, and
 * each decomposition 4n = a^2+b^2+c^2+d^2 with a <= b <= c <= d is solved by
 * a meet in the middle: the pairs (C, D) from buckets c, d are stored under
 * the hash of PAF_C + PAF_D, and each pair (A, B) from buckets a, b looks up
 * -(PAF_A + PAF_B).  A quadruple is emitted once, as the tuple sorted by
 * (|row sum|, index), so block permutations are quotiented out; multipliers
 * are not, and the caller reduces to equivalence classes.
 *
 * Spectra are carried as floor(16 PSD - 0.01), so a sum of stored values never
 * exceeds the stored sum and the integer filter is a superset of the exact
 * one.  The PAF match is exact integer arithmetic, and every emitted quadruple
 * is rechecked from its rows before it is written.
 *
 *   build: gcc -O3 -march=native -fopenmp -o williamson_c williamson_c.c -lm
 *   run:   ./williamson_c <n> [threads] [out]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define HMAX 32

static int N, H;

typedef struct {
    uint32_t code;      /* bits j-1 give x_j for j = 1..h */
    int s;              /* row sum, odd, may be negative */
    int8_t paf[HMAX];
    int16_t psd[HMAX];  /* floor(16 PSD(k) - 0.01), k = 1..h */
} Seq;

static Seq *seqs;
static long nseq;

static void rows_of(uint32_t code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        x[j] = (code >> (j - 1)) & 1 ? 1 : -1;
        x[N - j] = x[j];
    }
}

static int paf_of(const int8_t *x, int s) {
    int acc = 0;
    for (int j = 0; j < N; j++) {
        int k = j + s;
        if (k >= N) k -= N;
        acc += x[j] * x[k];
    }
    return acc;
}

/* All sequences with every PSD at most 4n, in code order. */
static void build_table(void) {
    long total = 1L << H;
    double *cs = malloc(sizeof(double) * (size_t)(H + 1) * (size_t)(H + 1));
    for (int k = 1; k <= H; k++)
        for (int s = 1; s <= H; s++)
            cs[k * (H + 1) + s] = cos(2.0 * M_PI * (double)s * (double)k / (double)N);
    unsigned char *keep = calloc((size_t)total, 1);
    long count = 0;
#pragma omp parallel for schedule(static) reduction(+ : count)
    for (long code = 0; code < total; code++) {
        int8_t x[2 * HMAX + 1];
        int paf[HMAX + 1];
        rows_of((uint32_t)code, x);
        for (int s = 1; s <= H; s++) paf[s] = paf_of(x, s);
        int ok = 1;
        for (int k = 1; k <= H && ok; k++) {
            double v = N;
            for (int s = 1; s <= H; s++) v += 2.0 * paf[s] * cs[k * (H + 1) + s];
            if (v > 4.0 * N + 1e-6) ok = 0;
        }
        if (ok) { keep[code] = 1; count++; }
    }
    seqs = malloc(sizeof(Seq) * (size_t)count);
    nseq = 0;
    for (long code = 0; code < total; code++)
        if (keep[code]) seqs[nseq++].code = (uint32_t)code;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < nseq; i++) {
        int8_t x[2 * HMAX + 1];
        Seq *q = &seqs[i];
        rows_of(q->code, x);
        q->s = 0;
        for (int j = 0; j < N; j++) q->s += x[j];
        for (int s = 1; s <= H; s++) q->paf[s - 1] = (int8_t)paf_of(x, s);
        for (int k = 1; k <= H; k++) {
            double v = N;
            for (int s = 1; s <= H; s++) v += 2.0 * q->paf[s - 1] * cs[k * (H + 1) + s];
            double f = floor(16.0 * v - 0.01);
            q->psd[k - 1] = (int16_t)(f < 0 ? 0 : f);
        }
    }
    free(keep);
    free(cs);
}

static uint64_t mix(const int8_t *v) {
    uint64_t hsh = 1469598103934665603ULL;
    for (int i = 0; i < H; i++) {
        hsh ^= (uint8_t)v[i];
        hsh *= 1099511628211ULL;
        hsh ^= hsh >> 29;
    }
    return hsh;
}

typedef struct { uint64_t key; uint32_t c, d; } Pair;

static int cmp_pair(const void *a, const void *b) {
    uint64_t x = ((const Pair *)a)->key, y = ((const Pair *)b)->key;
    return x < y ? -1 : x > y;
}

static int pair_ok(const Seq *p, const Seq *q) {
    int cap = 64 * N;
    for (int k = 0; k < H; k++)
        if (p->psd[k] + q->psd[k] > cap) return 0;
    return 1;
}

static FILE *out;
static long emitted;

static void emit(const Seq *a, const Seq *b, const Seq *c, const Seq *d) {
    const Seq *q[4] = {a, b, c, d};
    int8_t x[4][2 * HMAX + 1];
    for (int i = 0; i < 4; i++) rows_of(q[i]->code, x[i]);
    for (int s = 1; s <= H; s++) {
        int t = 0;
        for (int i = 0; i < 4; i++) t += paf_of(x[i], s);
        if (t != 0) { fprintf(stderr, "internal error: false match\n"); exit(2); }
    }
#pragma omp critical
    {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < N; j++) fputc(x[i][j] > 0 ? '+' : '-', out);
            fputc(i == 3 ? '\n' : ' ', out);
        }
        emitted++;
    }
}

/* Indices of sequences with |row sum| = v. */
static long *bucket(int v, long *count) {
    long m = 0;
    for (long i = 0; i < nseq; i++) if (abs(seqs[i].s) == v) m++;
    long *idx = malloc(sizeof(long) * (size_t)(m ? m : 1));
    m = 0;
    for (long i = 0; i < nseq; i++) if (abs(seqs[i].s) == v) idx[m++] = i;
    *count = m;
    return idx;
}

static void solve(int a, int b, int c, int d) {
    long na, nb, nc, nd;
    long *ia = bucket(a, &na), *ib = bucket(b, &nb), *ic = bucket(c, &nc), *id = bucket(d, &nd);
    /* right side: pairs (C, D), C <= D by index when c == d */
    long cap = 1 << 16, np = 0;
    Pair *pairs = malloc(sizeof(Pair) * (size_t)cap);
#pragma omp parallel
    {
        long lcap = 1 << 12, ln = 0;
        Pair *loc = malloc(sizeof(Pair) * (size_t)lcap);
#pragma omp for schedule(dynamic, 16)
        for (long i = 0; i < nc; i++) {
            const Seq *C = &seqs[ic[i]];
            for (long j = (c == d ? i : 0); j < nd; j++) {
                const Seq *D = &seqs[id[j]];
                if (!pair_ok(C, D)) continue;
                int8_t sum[HMAX];
                for (int k = 0; k < H; k++) sum[k] = (int8_t)(C->paf[k] + D->paf[k]);
                if (ln == lcap) { lcap *= 2; loc = realloc(loc, sizeof(Pair) * (size_t)lcap); }
                loc[ln].key = mix(sum);
                loc[ln].c = (uint32_t)i;
                loc[ln].d = (uint32_t)j;
                ln++;
            }
        }
#pragma omp critical
        {
            while (np + ln > cap) { cap *= 2; pairs = realloc(pairs, sizeof(Pair) * (size_t)cap); }
            memcpy(pairs + np, loc, sizeof(Pair) * (size_t)ln);
            np += ln;
        }
        free(loc);
    }
    qsort(pairs, (size_t)np, sizeof(Pair), cmp_pair);
    fprintf(stderr, "   (%d,%d,%d,%d): buckets %ld %ld %ld %ld, %ld right pairs\n",
            a, b, c, d, na, nb, nc, nd, np);
    /* left side: pairs (A, B), A <= B by index when a == b, B <= C when b == c */
#pragma omp parallel for schedule(dynamic, 4)
    for (long i = 0; i < na; i++) {
        const Seq *A = &seqs[ia[i]];
        for (long j = (a == b ? i : 0); j < nb; j++) {
            const Seq *B = &seqs[ib[j]];
            if (!pair_ok(A, B)) continue;
            int8_t need[HMAX];
            for (int k = 0; k < H; k++) need[k] = (int8_t)(-(A->paf[k] + B->paf[k]));
            uint64_t key = mix(need);
            long lo = 0, hi = np;
            while (lo < hi) {
                long mid = (lo + hi) / 2;
                if (pairs[mid].key < key) lo = mid + 1; else hi = mid;
            }
            for (long p = lo; p < np && pairs[p].key == key; p++) {
                long ci = pairs[p].c, di = pairs[p].d;
                if (b == c && j > ci) continue;
                const Seq *C = &seqs[ic[ci]], *D = &seqs[id[di]];
                int same = 1;
                for (int k = 0; k < H && same; k++)
                    if (C->paf[k] + D->paf[k] != need[k]) same = 0;
                if (same) emit(A, B, C, D);
            }
        }
    }
    free(pairs); free(ia); free(ib); free(ic); free(id);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s n [threads] [out]\n", argv[0]); return 1; }
    N = atoi(argv[1]);
    if (N < 3 || N % 2 == 0 || N > 2 * HMAX + 1) { fprintf(stderr, "n must be odd, 3 <= n <= %d\n", 2 * HMAX + 1); return 1; }
    H = (N - 1) / 2;
#ifdef _OPENMP
    if (argc > 2) omp_set_num_threads(atoi(argv[2]));
#endif
    out = argc > 3 ? fopen(argv[3], "w") : stdout;
    if (!out) { perror("output"); return 1; }
    build_table();
    fprintf(stderr, "n=%d: %ld of %ld symmetric sequences survive the ceiling\n", N, nseq, 1L << H);
    for (int a = 1; a * a <= 4 * N; a += 2)
        for (int b = a; a * a + b * b <= 4 * N; b += 2)
            for (int c = b; a * a + b * b + c * c <= 4 * N; c += 2) {
                int r = 4 * N - a * a - b * b - c * c;
                int d = (int)lround(sqrt((double)r));
                if (d < c || d % 2 == 0 || d * d != r) continue;
                solve(a, b, c, d);
            }
    if (fclose(out)) { perror("output"); return 1; }
    fprintf(stderr, "n=%d: %ld sorted quadruples\n", N, emitted);
    return 0;
}
