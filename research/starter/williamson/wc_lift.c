/* Lift compressed Williamson quadruples back to order n = d m.
 *
 * Reads the instances written by wc_compress (one compressed quadruple per
 * line with its row sums).  For each instance every full symmetric sequence
 * with the given compression and row sum is enumerated coset by coset, kept
 * when every PSD is at most 4n, and the four lists are matched on the PAF sum
 * exactly as williamson_c does: right pairs (C, D) in a sorted table, sharded
 * to WIL_PAIR_BYTES (default 8 GiB), left pairs (A, B) looked up.  Each match
 * is rechecked from its rows before it is written as four +/- rows.
 *
 *   build: gcc -O3 -march=native -fopenmp -o wc_lift wc_lift.c -lm
 *   run:   ./wc_lift n d instances [threads] [out] [first line] [last line]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef unsigned __int128 u128;
static int N, D, M, H, HM;
static u128 MASK;
static uint64_t PAIR_BYTES = 8ULL << 30;
static double *cs;

static inline u128 rot(u128 x, int s) { return ((x << s) | (x >> (N - s))) & MASK; }
static inline int pc(u128 x) { return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64)); }

typedef struct { u128 bits; int s; int8_t paf[32]; int16_t psd[32]; } Seq;
typedef struct { Seq *v; long n, cap; } List;

static void push(List *L, const Seq *q) {
    if (L->n == L->cap) { L->cap = L->cap ? 2 * L->cap : 1024; L->v = realloc(L->v, sizeof(Seq) * L->cap); }
    L->v[L->n++] = *q;
}

/* coset structure: for coset j, the free indices (0..H) it touches, with multiplicity */
static int cidx[32][16], cmul[32][16], clen[32];
static void cosets(void) {
    for (int j = 0; j < HM; j++) {
        clen[j] = 0;
        for (int k = 0; k < D; k++) {
            int p = (j + k * M) % N, f = p <= H ? p : N - p, t;
            for (t = 0; t < clen[j]; t++) if (cidx[j][t] == f) { cmul[j][t]++; break; }
            if (t == clen[j]) { cidx[j][t] = f; cmul[j][t] = 1; clen[j]++; }
        }
    }
}

/* enumerate lifts of compression c with positive row sum s */
static void lift_rec(int j, int t, int acc, const int *c, int s, uint64_t *assign, List *out) {
    if (j == HM) {
        /* build the sequence: x_f = assign bit f, f = 0..H */
        u128 x = 0;
        for (int f = 0; f <= H; f++) if ((*assign >> f) & 1) { x |= ((u128)1) << f; if (f) x |= ((u128)1) << (N - f); }
        int paf[33];
        for (int sh = 1; sh <= H; sh++) paf[sh] = N - 2 * pc(x ^ rot(x, sh));
        Seq q; q.bits = x; q.s = s;
        for (int k = 1; k <= H; k++) {
            double v = N;
            for (int sh = 1; sh <= H; sh++) v += paf[sh] * cs[k * (H + 1) + sh];
            if (v > 4.0 * N + 1e-6) return;
            double f = floor(16.0 * v - 0.01);
            q.psd[k - 1] = (int16_t)(f < 0 ? 0 : f);
        }
        for (int sh = 1; sh <= H; sh++) q.paf[sh - 1] = (int8_t)paf[sh];
        push(out, &q);
        return;
    }
    if (t == clen[j]) { if (acc == c[j]) lift_rec(j + 1, 0, 0, c, s, assign, out); return; }
    /* remaining capacity pruning */
    int rem = 0; for (int u = t; u < clen[j]; u++) rem += cmul[j][u];
    if (acc + rem < c[j] || acc - rem > c[j]) return;
    int f = cidx[j][t];
    *assign |= (1ULL << f); lift_rec(j, t + 1, acc + cmul[j][t], c, s, assign, out);
    *assign &= ~(1ULL << f); lift_rec(j, t + 1, acc - cmul[j][t], c, s, assign, out);
}

static uint64_t mix(const int8_t *v) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < H; i++) { h ^= (uint8_t)v[i]; h *= 1099511628211ULL; h ^= h >> 29; }
    return h;
}
typedef struct { uint64_t key; uint32_t c, d; } Pair;
static int cmp_pair(const void *a, const void *b) { uint64_t x = ((const Pair *)a)->key, y = ((const Pair *)b)->key; return x < y ? -1 : x > y; }
static int pair_ok(const Seq *p, const Seq *q) { int cap = 64 * N; for (int k = 0; k < H; k++) if (p->psd[k] + q->psd[k] > cap) return 0; return 1; }

static FILE *out; static long emitted;
static void emit(const Seq *a, const Seq *b, const Seq *c, const Seq *d) {
    const Seq *q[4] = {a, b, c, d};
    for (int sh = 1; sh <= H; sh++) { int t = 0; for (int i = 0; i < 4; i++) t += q[i]->paf[sh - 1]; if (t) { fprintf(stderr, "internal error: false match\n"); exit(2); } }
#pragma omp critical
    {
        for (int i = 0; i < 4; i++) { for (int j = 0; j < N; j++) fputc(((q[i]->bits >> j) & 1) ? '+' : '-', out); fputc(i == 3 ? '\n' : ' ', out); }
        emitted++;
    }
}

static void solve(List *A, List *B, List *C, List *Dl, int ab_same, int cd_same, int bc_same) {
    long total = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : total)
    for (long i = 0; i < C->n; i++) for (long j = (cd_same ? i : 0); j < Dl->n; j++) if (pair_ok(&C->v[i], &Dl->v[j])) total++;
    int shards = (int)((total * (long)sizeof(Pair) + (long)PAIR_BYTES - 1) / (long)PAIR_BYTES); if (shards < 1) shards = 1;
    for (int sh = 0; sh < shards; sh++) {
        long cap = 1 << 16, np = 0; Pair *pairs = malloc(sizeof(Pair) * cap);
#pragma omp parallel
        {
            long lcap = 1 << 12, ln = 0; Pair *loc = malloc(sizeof(Pair) * lcap);
#pragma omp for schedule(dynamic, 16)
            for (long i = 0; i < C->n; i++) for (long j = (cd_same ? i : 0); j < Dl->n; j++) {
                const Seq *Cq = &C->v[i], *Dq = &Dl->v[j];
                if (!pair_ok(Cq, Dq)) continue;
                int8_t sum[32]; for (int k = 0; k < H; k++) sum[k] = (int8_t)(Cq->paf[k] + Dq->paf[k]);
                uint64_t key = mix(sum);
                if ((int)((key >> 40) % (uint64_t)shards) != sh) continue;
                if (ln == lcap) { lcap *= 2; loc = realloc(loc, sizeof(Pair) * lcap); }
                loc[ln].key = key; loc[ln].c = (uint32_t)i; loc[ln].d = (uint32_t)j; ln++;
            }
#pragma omp critical
            { while (np + ln > cap) { cap *= 2; pairs = realloc(pairs, sizeof(Pair) * cap); } memcpy(pairs + np, loc, sizeof(Pair) * ln); np += ln; }
            free(loc);
        }
        qsort(pairs, np, sizeof(Pair), cmp_pair);
#pragma omp parallel for schedule(dynamic, 4)
        for (long i = 0; i < A->n; i++) for (long j = (ab_same ? i : 0); j < B->n; j++) {
            const Seq *Aq = &A->v[i], *Bq = &B->v[j];
            if (!pair_ok(Aq, Bq)) continue;
            int8_t need[32]; for (int k = 0; k < H; k++) need[k] = (int8_t)(-(Aq->paf[k] + Bq->paf[k]));
            uint64_t key = mix(need);
            if ((int)((key >> 40) % (uint64_t)shards) != sh) continue;
            long lo = 0, hi = np;
            while (lo < hi) { long mid = (lo + hi) / 2; if (pairs[mid].key < key) lo = mid + 1; else hi = mid; }
            for (long p = lo; p < np && pairs[p].key == key; p++) {
                long ci = pairs[p].c, di = pairs[p].d;
                if (bc_same && j > ci) continue;
                const Seq *Cq = &C->v[ci], *Dq = &Dl->v[di]; int same = 1;
                for (int k = 0; k < H && same; k++) if (Cq->paf[k] + Dq->paf[k] != need[k]) same = 0;
                if (same) emit(Aq, Bq, Cq, Dq);
            }
        }
        free(pairs);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: wc_lift n d instances [threads] [out] [first] [last]\n"); return 1; }
    N = atoi(argv[1]); D = atoi(argv[2]); M = N / D; H = (N - 1) / 2; HM = (M + 1) / 2; MASK = (((u128)1) << N) - 1;
    omp_set_num_threads(argc > 4 ? atoi(argv[4]) : 16);
    out = argc > 5 ? fopen(argv[5], "w") : stdout;
    long first = argc > 6 ? atol(argv[6]) : 0, last = argc > 7 ? atol(argv[7]) : (1L << 60);
    const char *pb = getenv("WIL_PAIR_BYTES"); if (pb && *pb) PAIR_BYTES = strtoull(pb, NULL, 10);
    cs = malloc(sizeof(double) * (H + 1) * (H + 1));
    for (int k = 1; k <= H; k++) for (int s = 1; s <= H; s++) cs[k * (H + 1) + s] = 2.0 * cos(2.0 * M_PI * s * k / N);
    cosets();
    FILE *in = fopen(argv[3], "r"); if (!in) { perror(argv[3]); return 1; }
    char line[4096]; long ln = -1, done = 0; double t0 = omp_get_wtime();
    while (fgets(line, sizeof line, in)) {
        ln++; if (ln < first || ln > last) continue;
        int s4[4], c[4][32]; char *p = line;
        for (int i = 0; i < 4; i++) s4[i] = (int)strtol(p, &p, 10);
        for (int i = 0; i < 4; i++) for (int j = 0; j < HM; j++) c[i][j] = (int)strtol(p, &p, 10);
        List L[4]; memset(L, 0, sizeof L);
        for (int i = 0; i < 4; i++) {
            int reuse = -1;
            for (int u = 0; u < i; u++) if (s4[u] == s4[i] && !memcmp(c[u], c[i], sizeof(int) * HM)) reuse = u;
            if (reuse >= 0) { L[i] = L[reuse]; continue; }
            uint64_t assign = 0;
            lift_rec(0, 0, 0, c[i], s4[i], &assign, &L[i]);
        }
        int ab = s4[0] == s4[1] && !memcmp(c[0], c[1], sizeof(int) * HM);
        int cd = s4[2] == s4[3] && !memcmp(c[2], c[3], sizeof(int) * HM);
        int bc = s4[1] == s4[2] && !memcmp(c[1], c[2], sizeof(int) * HM);
        long before = emitted;
        solve(&L[0], &L[1], &L[2], &L[3], ab, cd, bc);
        fprintf(stderr, "instance %ld (%d,%d,%d,%d): lifts %ld %ld %ld %ld, %ld quadruples, %.1f s\n", ln, s4[0], s4[1], s4[2], s4[3],
                L[0].n, L[1].n, L[2].n, L[3].n, emitted - before, omp_get_wtime() - t0);
        for (int i = 0; i < 4; i++) { int dup = 0; for (int u = 0; u < i; u++) if (L[u].v == L[i].v) dup = 1; if (!dup) free(L[i].v); }
        done++;
    }
    if (out != stdout) fclose(out);
    fprintf(stderr, "%ld instances, %ld quadruples, %.1f s\n", done, emitted, omp_get_wtime() - t0);
    return 0;
}
