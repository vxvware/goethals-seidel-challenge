/* Lift compressed Williamson quadruples back to order n = d m.
 *
 * Same input and output as wc_lift, with three prunings that cannot lose a
 * solution because the exact length-n PAF match at the end is unchanged:
 *
 *  - second compression: every lifted sequence is also compressed by m (to
 *    length d), which preserves the PAF, so the lists are grouped by their
 *    length-d PAF vector and only pairs of groups that some pair of groups on
 *    the other side can meet are enumerated;
 *  - split choice: when the four lists are distinct, the meet in the middle
 *    pairs the lists in whichever of the three splits scans fewest pairs;
 *  - the PSD pair test runs on 16-lane vectors (AVX2), and the scans are
 *    tiled eight sequences at a time so each partner is read once per tile;
 *  - the lift of one compressed sequence is parallel over its first cosets.
 *
 * Lifted lists are cached across consecutive instances that share a
 * compressed sequence.
 *
 *   build: gcc -O3 -march=native -fopenmp -o wc_lift5 wc_lift5.c -lm
 *   run:   ./wc_lift5 n d instances [threads] [out] [first line] [last line]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

typedef unsigned __int128 u128;
static int N, D, M, H, HM, HD;
static u128 MASK;
static uint64_t PAIR_BYTES = 8ULL << 30;
#define TILE 8
static double *cs;
static long stat_scanned, stat_kept, stat_gk, stat_gt;
static double t_lift, t_count, t_right, t_index, t_left;

static inline u128 rot(u128 x, int s) { return ((x << s) | (x >> (N - s))) & MASK; }
static inline int pc(u128 x) { return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64)); }

typedef struct { u128 bits; int s; int g; int8_t paf[32]; int16_t psd[32]; int16_t p2[16]; } Seq;
typedef struct {
    Seq *v; long n, cap; int ng; long *gs; int16_t *gk;   /* groups: starts (ng+1) and keys (16 per group) */
    long nt, *t0, *t1;                                   /* tiles of at most TILE sequences inside one group */
    int s, c[32]; long used;                             /* cache key and last use */
} List;

static void push(List *L, const Seq *q) {
    if (L->n == L->cap) { L->cap = L->cap ? 2 * L->cap : 1024; L->v = realloc(L->v, sizeof(Seq) * L->cap); }
    L->v[L->n++] = *q;
}

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

static void second_paf(u128 x, int16_t *p2) {
    int c2[64];
    for (int j = 0; j < D; j++) c2[j] = 0;
    for (int i = 0; i < N; i++) c2[i % D] += ((x >> i) & 1) ? 1 : -1;
    for (int s = 1; s <= HD; s++) { int v = 0; for (int j = 0; j < D; j++) v += c2[j] * c2[(j + s) % D]; p2[s - 1] = (int16_t)v; }
}

#define TASK_COSETS 2
typedef struct { uint64_t assign; } Task;
static Task *tasks; static long ntasks, ctasks;
static void lift_rec(int j, int t, int acc, const int *c, int s, uint64_t *assign, List *out) {
    if (j == TASK_COSETS && t == 0 && out == NULL) {   /* collecting tasks */
        if (ntasks == ctasks) { ctasks = ctasks ? 2 * ctasks : 256; tasks = realloc(tasks, sizeof(Task) * ctasks); }
        tasks[ntasks++].assign = *assign;
        return;
    }
    if (j == HM) {
        u128 x = 0;
        for (int f = 0; f <= H; f++) if ((*assign >> f) & 1) { x |= ((u128)1) << f; if (f) x |= ((u128)1) << (N - f); }
        int paf[33];
        for (int sh = 1; sh <= H; sh++) paf[sh] = N - 2 * pc(x ^ rot(x, sh));
        Seq q; memset(&q, 0, sizeof q); q.bits = x; q.s = s;
        for (int k = 1; k <= H; k++) {
            double v = N;
            for (int sh = 1; sh <= H; sh++) v += paf[sh] * cs[k * (H + 1) + sh];
            if (v > 4.0 * N + 1e-6) return;
            double f = floor(16.0 * v - 0.01);
            q.psd[k - 1] = (int16_t)(f < 0 ? 0 : f);
        }
        for (int sh = 1; sh <= H; sh++) q.paf[sh - 1] = (int8_t)paf[sh];
        second_paf(x, q.p2);
        push(out, &q);
        return;
    }
    if (t == clen[j]) { if (acc == c[j]) lift_rec(j + 1, 0, 0, c, s, assign, out); return; }
    int rem = 0; for (int u = t; u < clen[j]; u++) rem += cmul[j][u];
    if (acc + rem < c[j] || acc - rem > c[j]) return;
    int f = cidx[j][t];
    *assign |= (1ULL << f); lift_rec(j, t + 1, acc + cmul[j][t], c, s, assign, out);
    *assign &= ~(1ULL << f); lift_rec(j, t + 1, acc - cmul[j][t], c, s, assign, out);
}

static int cmp_seq_group(const void *a, const void *b) {
    const Seq *p = a, *q = b;
    for (int i = 0; i < HD; i++) if (p->p2[i] != q->p2[i]) return p->p2[i] < q->p2[i] ? -1 : 1;
    return 0;
}
/* group a list by second-level PAF */
static void index_list(List *L) {
    qsort(L->v, L->n, sizeof(Seq), cmp_seq_group);
    L->gs = malloc(sizeof(long) * (L->n + 1)); L->gk = malloc(sizeof(int16_t) * 16 * (L->n + 1));
    int ng = 0;
    for (long i = 0; i < L->n; i++) {
        if (i == 0 || cmp_seq_group(&L->v[i - 1], &L->v[i])) { L->gs[ng] = i; memcpy(L->gk + 16 * ng, L->v[i].p2, sizeof(int16_t) * 16); ng++; }
        L->v[i].g = ng - 1;
    }
    L->gs[ng] = L->n; L->ng = ng;
    L->t0 = malloc(sizeof(long) * (L->n + 1)); L->t1 = malloc(sizeof(long) * (L->n + 1)); L->nt = 0;
    for (int g = 0; g < ng; g++)
        for (long a = L->gs[g]; a < L->gs[g + 1]; a += TILE) { L->t0[L->nt] = a; L->t1[L->nt] = a + TILE < L->gs[g + 1] ? a + TILE : L->gs[g + 1]; L->nt++; }
}

static uint64_t vhash(const int *v) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < HD; i++) { h ^= (uint64_t)(uint32_t)(v[i] + 100000); h *= 1099511628211ULL; h ^= h >> 31; }
    return h | 1;
}
typedef struct { uint64_t *t; uint64_t mask; } Set;
static Set set_new(long n) { Set s; uint64_t cap = 16; while (cap < (uint64_t)n * 2) cap <<= 1; s.t = calloc(cap, 8); s.mask = cap - 1; return s; }
static void set_add(Set *s, uint64_t h) { uint64_t i = h & s->mask; while (s->t[i] && s->t[i] != h) i = (i + 1) & s->mask; s->t[i] = h; }
static int set_has(const Set *s, uint64_t h) { uint64_t i = h & s->mask; while (s->t[i]) { if (s->t[i] == h) return 1; i = (i + 1) & s->mask; } return 0; }
static Set pair_sums(const List *X, const List *Y) {
    Set s = set_new((long)X->ng * Y->ng);
    for (int a = 0; a < X->ng; a++) for (int b = 0; b < Y->ng; b++) {
        int v[16]; for (int i = 0; i < HD; i++) v[i] = X->gk[16 * a + i] + Y->gk[16 * b + i];
        set_add(&s, vhash(v));
    }
    return s;
}
/* groups b of Y with -(X_a + Y_b) in S, flat with offsets; *work = sum of |X_a||Y_b| over kept pairs */
static int *compat_groups(const List *X, const List *Y, const Set *S, int **off, double *work) {
    int *o = malloc(sizeof(int) * (X->ng + 1)); long cap = 1024, nn = 0; int *flat = malloc(sizeof(int) * cap); double w = 0;
    for (int a = 0; a < X->ng; a++) {
        o[a] = (int)nn;
        for (int b = 0; b < Y->ng; b++) {
            int v[16]; for (int i = 0; i < HD; i++) v[i] = -(X->gk[16 * a + i] + Y->gk[16 * b + i]);
            if (set_has(S, vhash(v))) {
                if (nn == cap) { cap *= 2; flat = realloc(flat, sizeof(int) * cap); }
                flat[nn++] = b; w += (double)(X->gs[a + 1] - X->gs[a]) * (Y->gs[b + 1] - Y->gs[b]);
            }
        }
    }
    o[X->ng] = (int)nn; *off = o; *work = w; return flat;
}

static uint64_t mix(const int8_t *v) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < H; i++) { h ^= (uint8_t)v[i]; h *= 1099511628211ULL; h ^= h >> 29; }
    return h;
}
typedef struct { uint64_t key; uint32_t c, d; } Pair;
static __m256i CAPV;
static inline int pair_ok(const Seq *p, const Seq *q) {
    __m256i a0 = _mm256_loadu_si256((const __m256i *)p->psd), a1 = _mm256_loadu_si256((const __m256i *)(p->psd + 16));
    __m256i b0 = _mm256_loadu_si256((const __m256i *)q->psd), b1 = _mm256_loadu_si256((const __m256i *)(q->psd + 16));
    __m256i m = _mm256_or_si256(_mm256_cmpgt_epi16(_mm256_add_epi16(a0, b0), CAPV), _mm256_cmpgt_epi16(_mm256_add_epi16(a1, b1), CAPV));
    return _mm256_testz_si256(m, m);
}

static FILE *out; static long emitted;
static void emit(const Seq *q[4]) {
    for (int sh = 1; sh <= H; sh++) { int t = 0; for (int i = 0; i < 4; i++) t += q[i]->paf[sh - 1]; if (t) { fprintf(stderr, "internal error: false match\n"); exit(2); } }
#pragma omp critical
    {
        for (int i = 0; i < 4; i++) { for (int j = 0; j < N; j++) fputc(((q[i]->bits >> j) & 1) ? '+' : '-', out); fputc(i == 3 ? '\n' : ' ', out); }
        emitted++;
    }
}

/* meet in the middle: left lists P, Q against right lists R, S; pos[] says where each sits in the output quadruple */
static void solve(List *P, List *Q, List *R, List *S, int pq_same, int rs_same, int qr_same, const int pos[4]) {
    double tt = omp_get_wtime();
    Set SL = pair_sums(P, Q), SR = pair_sums(R, S);
    double wr, wl; int *offR, *compR = compat_groups(R, S, &SL, &offR, &wr); int *offL, *compL = compat_groups(P, Q, &SR, &offL, &wl);
    stat_gk += offR[R->ng] + offL[P->ng]; stat_gt += (long)R->ng * S->ng + (long)P->ng * Q->ng;
    long total = 0, scanned = 0;
    if (wr * sizeof(Pair) > (double)PAIR_BYTES)
#pragma omp parallel for schedule(dynamic, 2) reduction(+ : total, scanned)
    for (long tile = 0; tile < R->nt; tile++) {
        long i0 = R->t0[tile], i1 = R->t1[tile]; int g = R->v[i0].g;
        for (int t = offR[g]; t < offR[g + 1]; t++) {
            int gd = compR[t]; long j0 = S->gs[gd], j1 = S->gs[gd + 1]; if (rs_same && j0 < i0) j0 = i0;
            for (long j = j0; j < j1; j++) { const Seq *y = &S->v[j]; for (long i = i0; i < i1; i++) { if (rs_same && j < i) continue; scanned++; if (pair_ok(&R->v[i], y)) total++; } }
        }
    }
    stat_scanned += scanned; stat_kept += total;
    t_count += omp_get_wtime() - tt;
    int shards = (int)((total * (long)sizeof(Pair) + (long)PAIR_BYTES - 1) / (long)PAIR_BYTES); if (shards < 1) shards = 1;
    for (int sh = 0; sh < shards; sh++) {
        tt = omp_get_wtime();
        long cap = 1 << 16, np = 0; Pair *pairs = malloc(sizeof(Pair) * cap);
#pragma omp parallel
        {
            long lcap = 1 << 12, ln = 0; Pair *loc = malloc(sizeof(Pair) * lcap);
#pragma omp for schedule(dynamic, 2)
            for (long tile = 0; tile < R->nt; tile++) {
                long i0 = R->t0[tile], i1 = R->t1[tile]; int g = R->v[i0].g;
                for (int t = offR[g]; t < offR[g + 1]; t++) {
                    int gd = compR[t]; long j0 = S->gs[gd], j1 = S->gs[gd + 1]; if (rs_same && j0 < i0) j0 = i0;
                    for (long j = j0; j < j1; j++) for (long i = i0; i < i1; i++) {
                        const Seq *x = &R->v[i], *y = &S->v[j];
                        if (rs_same && j < i) continue;
                        if (!pair_ok(x, y)) continue;
                        int8_t sum[32]; for (int k = 0; k < H; k++) sum[k] = (int8_t)(x->paf[k] + y->paf[k]);
                        uint64_t key = mix(sum);
                        if ((int)((key >> 40) % (uint64_t)shards) != sh) continue;
                        if (ln == lcap) { lcap *= 2; loc = realloc(loc, sizeof(Pair) * lcap); }
                        loc[ln].key = key; loc[ln].c = (uint32_t)i; loc[ln].d = (uint32_t)j; ln++;
                    }
                }
            }
#pragma omp critical
            { while (np + ln > cap) { cap *= 2; pairs = realloc(pairs, sizeof(Pair) * cap); } memcpy(pairs + np, loc, sizeof(Pair) * ln); np += ln; }
            free(loc);
        }
        t_right += omp_get_wtime() - tt; tt = omp_get_wtime();
        int bb = 1; while ((1L << bb) < np) bb++; bb += 1; long nb = 1L << bb; int shift = 64 - bb;
        long *start = calloc(nb + 1, sizeof(long));
        for (long p = 0; p < np; p++) start[(pairs[p].key >> shift) + 1]++;
        for (long b = 0; b < nb; b++) start[b + 1] += start[b];
        Pair *sorted = malloc(sizeof(Pair) * (np ? np : 1)); long *fill = malloc(sizeof(long) * nb); memcpy(fill, start, sizeof(long) * nb);
        for (long p = 0; p < np; p++) sorted[fill[pairs[p].key >> shift]++] = pairs[p];
        free(fill); free(pairs); pairs = sorted;
        t_index += omp_get_wtime() - tt; tt = omp_get_wtime();
        long lscanned = 0, lkept = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : lscanned, lkept)
        for (long tile = 0; tile < P->nt; tile++) {
            long i0 = P->t0[tile], i1 = P->t1[tile]; int g = P->v[i0].g;
            for (int t = offL[g]; t < offL[g + 1]; t++) {
                int gq = compL[t]; long j0 = Q->gs[gq], j1 = Q->gs[gq + 1]; if (pq_same && j0 < i0) j0 = i0;
                for (long j = j0; j < j1; j++) for (long i = i0; i < i1; i++) {
                    const Seq *x = &P->v[i], *y = &Q->v[j];
                    if (pq_same && j < i) continue;
                    lscanned++;
                    if (!pair_ok(x, y)) continue;
                    lkept++;
                    int8_t need[32]; for (int k = 0; k < H; k++) need[k] = (int8_t)(-(x->paf[k] + y->paf[k]));
                    uint64_t key = mix(need);
                    if ((int)((key >> 40) % (uint64_t)shards) != sh) continue;
                    long b = key >> shift;
                    for (long q = start[b]; q < start[b + 1]; q++) {
                        if (pairs[q].key != key) continue;
                        long ri = pairs[q].c, si = pairs[q].d;
                        if (qr_same && j > ri) continue;
                        const Seq *r = &R->v[ri], *s = &S->v[si]; int same = 1;
                        for (int k = 0; k < H && same; k++) if (r->paf[k] + s->paf[k] != need[k]) same = 0;
                        if (same) { const Seq *q4[4]; q4[pos[0]] = x; q4[pos[1]] = y; q4[pos[2]] = r; q4[pos[3]] = s; emit(q4); }
                    }
                }
            }
        }
        free(pairs); free(start);
        stat_scanned += lscanned; stat_kept += lkept;
        t_left += omp_get_wtime() - tt;
    }
    free(SL.t); free(SR.t); free(compR); free(offR); free(compL); free(offL);
}

/* cache of lifted, indexed lists keyed by (sum, compression) */
#define CACHE 24
static List cache[CACHE]; static long clock_;
static List *get_list(int s, const int *c) {
    for (int i = 0; i < CACHE; i++) if (cache[i].v && cache[i].s == s && !memcmp(cache[i].c, c, sizeof(int) * HM)) { cache[i].used = ++clock_; return &cache[i]; }
    int slot = 0;
    for (int i = 0; i < CACHE; i++) { if (!cache[i].v) { slot = i; break; } if (cache[i].used < cache[slot].used) slot = i; }
    List *L = &cache[slot];
    if (L->v) { free(L->v); free(L->gs); free(L->gk); free(L->t0); free(L->t1); }
    memset(L, 0, sizeof *L); L->s = s; memcpy(L->c, c, sizeof(int) * HM); L->used = ++clock_;
    uint64_t assign = 0;
    if (HM <= TASK_COSETS) lift_rec(0, 0, 0, c, s, &assign, L);
    else {
        ntasks = 0;
        lift_rec(0, 0, 0, c, s, &assign, NULL);
#pragma omp parallel
        {
            List mine; memset(&mine, 0, sizeof mine);
#pragma omp for schedule(dynamic, 1)
            for (long t = 0; t < ntasks; t++) { uint64_t a = tasks[t].assign; lift_rec(TASK_COSETS, 0, 0, c, s, &a, &mine); }
#pragma omp critical
            { for (long i = 0; i < mine.n; i++) push(L, &mine.v[i]); }
            free(mine.v);
        }
    }
    if (!L->v) L->v = malloc(sizeof(Seq));   /* empty list still owns a buffer for the cache test */
    index_list(L);
    return L;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: wc_lift5 n d instances [threads] [out] [first] [last]\n"); return 1; }
    N = atoi(argv[1]); D = atoi(argv[2]); M = N / D; H = (N - 1) / 2; HM = (M + 1) / 2; HD = (D - 1) / 2; MASK = (((u128)1) << N) - 1;
    if (HD > 16 || D > 64 || H > 32) { fprintf(stderr, "unsupported n, d\n"); return 1; }
    omp_set_num_threads(argc > 4 ? atoi(argv[4]) : 16);
    out = argc > 5 ? fopen(argv[5], "w") : stdout;
    long first = argc > 6 ? atol(argv[6]) : 0, last = argc > 7 ? atol(argv[7]) : (1L << 60);
    const char *pb = getenv("WIL_PAIR_BYTES"); if (pb && *pb) PAIR_BYTES = strtoull(pb, NULL, 10);
    cs = malloc(sizeof(double) * (H + 1) * (H + 1));
    for (int k = 1; k <= H; k++) for (int s = 1; s <= H; s++) cs[k * (H + 1) + s] = 2.0 * cos(2.0 * M_PI * s * k / N);
    cosets();
    CAPV = _mm256_set1_epi16((short)(64 * N));
    FILE *in = fopen(argv[3], "r"); if (!in) { perror(argv[3]); return 1; }
    char line[4096]; long ln = -1, done = 0; double t0 = omp_get_wtime();
    while (fgets(line, sizeof line, in)) {
        ln++; if (ln < first || ln > last) continue;
        int s4[4], c[4][32]; char *p = line;
        for (int i = 0; i < 4; i++) s4[i] = (int)strtol(p, &p, 10);
        for (int i = 0; i < 4; i++) for (int j = 0; j < HM; j++) c[i][j] = (int)strtol(p, &p, 10);
        double tl = omp_get_wtime();
        List *L[4];
        for (int i = 0; i < 4; i++) L[i] = get_list(s4[i], c[i]);
        t_lift += omp_get_wtime() - tl;
        int same01 = L[0] == L[1], same23 = L[2] == L[3], same12 = L[1] == L[2];
        int distinct = L[0] != L[1] && L[0] != L[2] && L[0] != L[3] && L[1] != L[2] && L[1] != L[3] && L[2] != L[3];
        long before = emitted;
        stat_scanned = stat_kept = stat_gk = stat_gt = 0;
        if (!distinct) {
            static const int id[4] = {0, 1, 2, 3};
            solve(L[0], L[1], L[2], L[3], same01, same23, same12, id);
        } else {
            /* pick the split with the least group-compatible pair work */
            static const int splits[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
            int best = 0; double bw = 1e300;
            for (int t = 0; t < 3; t++) {
                List *P = L[splits[t][0]], *Q = L[splits[t][1]], *R = L[splits[t][2]], *S = L[splits[t][3]];
                Set SL = pair_sums(P, Q), SR = pair_sums(R, S); double wr, wl; int *o1, *o2;
                int *c1 = compat_groups(R, S, &SL, &o1, &wr), *c2 = compat_groups(P, Q, &SR, &o2, &wl);
                free(SL.t); free(SR.t); free(c1); free(o1); free(c2); free(o2);
                if (wr + wl < bw) { bw = wr + wl; best = t; }
            }
            const int *sp = splits[best]; int pos[4]; for (int i = 0; i < 4; i++) pos[i] = sp[i];
            solve(L[sp[0]], L[sp[1]], L[sp[2]], L[sp[3]], 0, 0, 0, pos);
        }
        fprintf(stderr, "instance %ld (%d,%d,%d,%d): lifts %ld %ld %ld %ld, groups %d %d %d %d, group pairs %ld of %ld, pairs scanned %ld kept %ld, %ld quadruples, %.1f s\n",
                ln, s4[0], s4[1], s4[2], s4[3], L[0]->n, L[1]->n, L[2]->n, L[3]->n, L[0]->ng, L[1]->ng, L[2]->ng, L[3]->ng,
                stat_gk, stat_gt, stat_scanned, stat_kept, emitted - before, omp_get_wtime() - t0);
        done++;
    }
    if (out != stdout) fclose(out);
    fprintf(stderr, "%ld instances, %ld quadruples, %.1f s (lift %.1f count %.1f right %.1f index %.1f left %.1f)\n", done, emitted, omp_get_wtime() - t0, t_lift, t_count, t_right, t_index, t_left);
    return 0;
}
