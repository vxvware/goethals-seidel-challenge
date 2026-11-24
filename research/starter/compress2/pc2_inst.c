/* pc2_inst.c -- compressed quadruples ("instances") of a circulant
 * Goethals-Seidel family of odd order n at one compression length m = n / d,
 * enumerated directly at length m, without the 2^H row enumeration of
 * research/starter/compress/pc_compress.c.
 *
 * The pattern names the block kinds, skew blocks first: ssss (Williamson), ksss
 * (good), kkss (G-matrices), kkks (best).  A skew row has a_0 = 1 and
 * a_{n-j} = -a_j; a symmetric row has a_{n-j} = a_j and is sign-normalised to a
 * positive row sum.  Compression to length m (entry j is the sum of the row over
 * the coset j + k m) preserves the PAF and PSD conditions.
 *
 * Every compressed row is listed that some row of length n could compress to:
 * a skew compression has c_0 = 1, c_{m-j} = -c_j and c_j odd in [-d, d]; a
 * symmetric one has c_{m-j} = c_j and every c_j odd in [-d, d] (coset 0 holds
 * a_0 and (d-1)/2 equal pairs, so c_0 takes every odd value in [-d, d]).  It is
 * kept if its row sum is one of the row sums a decomposition of 4n allows and
 * its PSD at every nonzero frequency of Z_m is at most 4n minus the number of
 * other skew blocks (a skew row has PSD at least 1 at every frequency, since
 * its transform is 1 plus a purely imaginary number).  This is a superset of
 * the realized compressions pc_compress.c lists, so the instance set here is
 * a superset of the one pc_compress.c writes for the same n and d.
 *
 * The match is the one of pc_compress.c: per decomposition of 4n into squared
 * row sums, pairs of the last two blocks that pass the pair PSD ceiling are
 * hashed on minus their PAF sum, pairs of the first two blocks probe, every hit
 * is compared exactly, and with reduction one instance per orbit of the units
 * of Z_m is kept (the one whose sorted form is least).  With the argument full,
 * every instance is written, blocks of equal kind and row sum still in list
 * order (first index not above the second).
 *
 * Output line: k0 k1 k2 k3 A'_0..A'_h B'_0..B'_h C'_0..C'_h D'_0..D'_h with
 * h = (m-1)/2 and k = -1 for a skew block, else its row sum (the format of
 * pc_compress.c, so research/starter/compress/pc_lift.c lifts these lines).
 *
 *   build: gcc -O3 -march=native -fopenmp -o pc2_inst pc2_inst.c -lm
 *   run:   pc2_inst PATTERN n d threads OUT [full] [shard shards]
 *   env:   PC_MEM_MB  pair-table budget of the match (default 2048)
 *          PC_LISTS_ONLY=1  count the compressed rows without storing them, print
 *                           the list sizes and the number of pair scans the
 *                           match would make, and stop (for pricing)
 * The output is written as OUT.part and renamed when complete; statistics go
 * to standard output.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define NLMAX 32
#define HMX 64
static int N, D, M, H, HM, KIND[4], NK, NS, FULL, COUNT_ONLY;
static char PAT[5];

static void die(const char *m) { fprintf(stderr, "pc2_inst: %s\n", m); exit(1); }
static inline uint64_t mix64(uint64_t h) { h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33; return h; }

static void setup(const char *pat, int n, int d) {
    if (strlen(pat) != 4) die("pattern is four letters k/s, skew first");
    memcpy(PAT, pat, 5); NK = 0;
    for (int i = 0; i < 4; i++) {
        if (pat[i] != 'k' && pat[i] != 's') die("pattern letters are k and s");
        KIND[i] = pat[i] == 'k'; NK += KIND[i];
        if (i && KIND[i] > KIND[i - 1]) die("skew blocks first");
    }
    if (NK == 4) die("pattern kkkk has no symmetric block");
    N = n; D = d;
    if (N < 9 || N > 255 || N % 2 == 0 || D < 3 || D % 2 == 0 || N % D || N / D < 3) die("need odd n = d m with d, m >= 3 and n <= 255");
    M = N / D; H = (N - 1) / 2; HM = (M + 1) / 2;
    if (HM > HMX) die("m too large");
    int smax = (int)floor(sqrt(4.0 * N - NK)); if (smax % 2 == 0) smax--;
    NS = (smax + 1) / 2;
    if (NS > NLMAX) die("too many row sums");
}

/* ---- lists ---- */
typedef struct {
    int n, cap;
    int8_t *c;       /* HM entries per row */
    int16_t *paf;    /* HM-1 lags */
    int32_t *psd;    /* HM-1 frequencies, floor(16 PSD - 0.01) */
    uint8_t *omin;   /* least in its orbit under the units of Z_m */
} List;
static List LS[NLMAX + 1];   /* 0 skew, i >= 1 symmetric with row sum 2i-1 */
static int NEED[NLMAX + 1];

static void lpush(List *L, const int *c) {
    if (COUNT_ONLY) { L->n++; return; }
    if (L->n == L->cap) {
        L->cap = L->cap ? 2 * L->cap : 1024;
        L->c = realloc(L->c, (size_t)L->cap * HM);
        if (!L->c) die("out of memory (list)");
    }
    for (int j = 0; j < HM; j++) L->c[(size_t)L->n * HM + j] = (int8_t)c[j];
    L->n++;
}
static int cmp_rows(const int8_t *a, const int8_t *b) { for (int j = 0; j < HM; j++) if (a[j] != b[j]) return a[j] < b[j] ? -1 : 1; return 0; }
static int cmp_rows_q(const void *a, const void *b) { return cmp_rows(a, b); }

/* enumeration: depth-first over c_1..c_{HM-1} keeping the transform at every
   frequency; symmetric c_0 is fixed by the row sum at the leaf */
static double COSM[HMX][HMX], SINM[HMX][HMX];
typedef struct { int skew, s; double cap; List *out; } Job;
static void leaf(const Job *jb, const int *c, const double *acc) {
    int cc[HMX]; memcpy(cc, c, sizeof(int) * HM);
    if (!jb->skew) {
        int t = 0; for (int j = 1; j < HM; j++) t += 2 * c[j];
        int c0 = jb->s - t;
        if (c0 > D || c0 < -D) return;
        cc[0] = c0;
        for (int k = 1; k < HM; k++) { double v = c0 + acc[k]; if (v * v > jb->cap) return; }
    } else {
        cc[0] = 1;
        for (int k = 1; k < HM; k++) { double v = 1.0 + acc[k] * acc[k]; if (v > jb->cap) return; }
    }
    lpush(jb->out, cc);
}
static void dfs(const Job *jb, int j, int *c, double *acc, int partial) {
    if (j == HM) { leaf(jb, c, acc); return; }
    /* symmetric: remaining c_j..c_{HM-1} contribute 2 sum in [-2 D r, 2 D r], c_0 in [-D, D] */
    if (!jb->skew) {
        int r = HM - j, lo = jb->s - partial - 2 * D * r, hi = jb->s - partial + 2 * D * r;
        if (hi < -D || lo > D) return;
    }
    double nacc[HMX];
    for (int v = -D; v <= D; v += 2) {
        c[j] = v;
        if (jb->skew) for (int k = 1; k < HM; k++) nacc[k] = acc[k] + 2.0 * v * SINM[j][k];
        else for (int k = 1; k < HM; k++) nacc[k] = acc[k] + 2.0 * v * COSM[j][k];
        dfs(jb, j + 1, c, nacc, partial + 2 * v);
    }
}
static void build_list(int skew, int s, List *out) {
    Job jb = {skew, s, 4.0 * N - (NK - skew) + 1e-6, out};
    /* split on c_1 and c_2 */
    int nv = D + 1, top = HM >= 3 ? nv * nv : (HM == 2 ? nv : 1);
    List part[1024]; if (top > 1024) die("split too wide");
    memset(part, 0, sizeof(List) * top);
#pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < top; t++) {
        int c[HMX]; double acc[HMX]; memset(acc, 0, sizeof acc); c[0] = 0;
        Job my = jb; my.out = &part[t];
        int depth = HM >= 3 ? 3 : HM, partial = 0, tt = t;
        for (int j = 1; j < depth; j++) {
            int v = -D + 2 * (tt % nv); tt /= nv; c[j] = v; partial += 2 * v;
            for (int k = 1; k < HM; k++) acc[k] += 2.0 * v * (skew ? SINM[j][k] : COSM[j][k]);
        }
        dfs(&my, depth, c, acc, partial);
    }
    if (COUNT_ONLY) { for (int t = 0; t < top; t++) out->n += part[t].n; return; }
    for (int t = 0; t < top; t++) {
        for (int r = 0; r < part[t].n; r++) {
            int c[HMX]; for (int j = 0; j < HM; j++) c[j] = part[t].c[(size_t)r * HM + j];
            lpush(out, c);
        }
        free(part[t].c);
    }
    qsort(out->c, out->n, HM, cmp_rows_q);
}

/* ---- orbits ---- */
static int is_unit(int u) { int g = u, h = M; while (h) { int r = g % h; g = h; h = r; } return g == 1; }
static void image(const int *c, int skew, int u, int *img) {
    for (int j = 0; j < HM; j++) {
        int k = (int)(((long)u * j) % M);
        img[j] = k > HM - 1 ? (skew ? -c[M - k] : c[M - k]) : c[k];
    }
}
static int cmp_c(const int *a, const int *b) { for (int j = 0; j < HM; j++) if (a[j] != b[j]) return a[j] < b[j] ? -1 : 1; return 0; }
static void sorted_form(const int s4[4], int c[4][HMX], int outc[4][HMX]) {
    int ord[4] = {0, 1, 2, 3};
    for (int i = 1; i < 4; i++) for (int j = i; j > 0; j--) {
        int x = ord[j - 1], y = ord[j];
        if (s4[x] == s4[y] && cmp_c(c[x], c[y]) > 0) { ord[j - 1] = y; ord[j] = x; } else break;
    }
    for (int i = 0; i < 4; i++) memcpy(outc[i], c[ord[i]], sizeof(int) * HMX);
}
static int canonical(const int s4[4], int c[4][HMX]) {
    int self[4][HMX]; sorted_form(s4, c, self);
    for (int u = 2; u < M; u++) {
        if (!is_unit(u)) continue;
        int img[4][HMX], srt[4][HMX];
        for (int i = 0; i < 4; i++) image(c[i], KIND[i], u, img[i]);
        sorted_form(s4, img, srt);
        for (int i = 0; i < 4; i++) { int r = cmp_c(srt[i], self[i]); if (r < 0) return 0; if (r > 0) break; }
    }
    return 1;
}

static void finish_list(List *L, int skew) {
    int K = HM - 1;
    L->paf = malloc(sizeof(int16_t) * ((size_t)L->n * K + 1));
    L->psd = malloc(sizeof(int32_t) * ((size_t)L->n * K + 1));
    L->omin = malloc((size_t)L->n + 1);
    if (!L->paf || !L->psd || !L->omin) die("out of memory (list data)");
#pragma omp parallel for schedule(static)
    for (int r = 0; r < L->n; r++) {
        int x[2 * HMX], c[HMX];
        for (int j = 0; j < HM; j++) c[j] = L->c[(size_t)r * HM + j];
        for (int j = 0; j < M; j++) x[j] = j < HM ? c[j] : (skew ? -c[M - j] : c[M - j]);
        for (int s = 1; s < HM; s++) {
            int p = 0; for (int j = 0; j < M; j++) p += x[j] * x[(j + s) % M];
            L->paf[(size_t)r * K + s - 1] = (int16_t)p;
            double re = 0, im = 0;
            for (int j = 0; j < M; j++) { re += x[j] * cos(2 * M_PI * j * s / M); im += x[j] * sin(2 * M_PI * j * s / M); }
            L->psd[(size_t)r * K + s - 1] = (int32_t)floor(16 * (re * re + im * im) - 0.01);
        }
        int om = 1;
        for (int u = 2; u < M && om; u++) { int img[HMX]; if (is_unit(u)) { image(c, skew, u, img); if (cmp_c(img, c) < 0) om = 0; } }
        L->omin[r] = (uint8_t)om;
    }
}

/* ---- match ---- */
typedef struct { uint64_t h; uint32_t i, j; } P;
static inline int psd_ok(const List *X, long i, const List *Y, long j) {
    int K = HM - 1; const int32_t *a = X->psd + (size_t)i * K, *b = Y->psd + (size_t)j * K;
    for (int k = 0; k < K; k++) if (a[k] + b[k] > 16 * 4 * N) return 0;
    return 1;
}
static inline uint64_t vhash(const int *v) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int k = 0; k < HM - 1; k++) { h ^= (uint64_t)(uint32_t)v[k]; h *= 0xff51afd7ed558ccdULL; h ^= h >> 29; }
    return mix64(h);
}
static inline uint64_t hroute(uint64_t h) { return h & 0xffffffffULL; }

static long match(FILE *out, int shard, int shards, double mem_bytes) {
    long inst_total = 0;
    int ns = 4 - NK, sv[4] = {0, 0, 0, 0}, K = HM - 1;
    long ndec = 1; for (int i = 0; i < ns; i++) ndec *= NS;
    for (long code = 0; code < ndec; code++) {
        long c2 = code; int ok = 1, sq = NK;
        for (int i = 0; i < ns; i++) { sv[i] = 2 * (int)(c2 % NS) + 1; c2 /= NS; }
        for (int i = 1; i < ns; i++) if (sv[i] > sv[i - 1]) ok = 0;
        for (int i = 0; i < ns; i++) sq += sv[i] * sv[i];
        if (!ok || sq != 4 * N) continue;
        int key4[4]; List *LL[4];
        for (int i = 0; i < 4; i++) {
            if (KIND[i]) { key4[i] = -1; LL[i] = &LS[0]; }
            else { key4[i] = sv[i - NK]; LL[i] = &LS[(key4[i] + 1) / 2]; }
        }
        long nn[4]; for (int i = 0; i < 4; i++) nn[i] = LL[i]->n;
        int same01 = LL[0] == LL[1], same12 = LL[1] == LL[2], same23 = LL[2] == LL[3];
        double t0 = omp_get_wtime();
        long rcount = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : rcount)
        for (long i = 0; i < nn[2]; i++) for (long j = (same23 ? i : 0); j < nn[3]; j++) {
            if (!psd_ok(LL[2], i, LL[3], j)) continue;
            int v[HMX]; for (int k = 0; k < K; k++) v[k] = -(LL[2]->paf[i * K + k] + LL[3]->paf[j * K + k]);
            if ((long)(hroute(vhash(v)) % (uint64_t)shards) != shard) continue;
            rcount++;
        }
        int passes = (int)ceil((double)rcount * sizeof(P) * 2.2 / mem_bytes); if (passes < 1) passes = 1;
        long inst = 0, lscan = 0;
        for (int pass = 0; pass < passes; pass++) {
            long np = 0, rcap = rcount / passes + 1024; P *R = malloc(sizeof(P) * rcap);
            if (!R) die("out of memory (pairs)");
#pragma omp parallel
            {
                long lcap = 1 << 12, ln = 0; P *loc = malloc(sizeof(P) * lcap);
#pragma omp for schedule(dynamic, 16) nowait
                for (long i = 0; i < nn[2]; i++) for (long j = (same23 ? i : 0); j < nn[3]; j++) {
                    if (!psd_ok(LL[2], i, LL[3], j)) continue;
                    int v[HMX]; for (int k = 0; k < K; k++) v[k] = -(LL[2]->paf[i * K + k] + LL[3]->paf[j * K + k]);
                    uint64_t h = vhash(v), r = hroute(h);
                    if ((long)(r % (uint64_t)shards) != shard || (int)((r / (uint64_t)shards) % (uint64_t)passes) != pass) continue;
                    if (ln == lcap) { lcap *= 2; loc = realloc(loc, sizeof(P) * lcap); if (!loc) die("out of memory"); }
                    loc[ln].h = h; loc[ln].i = (uint32_t)i; loc[ln].j = (uint32_t)j; ln++;
                    if (ln == 1 << 16) {
#pragma omp critical(rappend)
                        { if (np + ln > rcap) { rcap = 2 * (np + ln); R = realloc(R, sizeof(P) * rcap); if (!R) die("out of memory"); } memcpy(R + np, loc, sizeof(P) * ln); np += ln; }
                        ln = 0;
                    }
                }
#pragma omp critical(rappend)
                { if (np + ln > rcap) { rcap = 2 * (np + ln); R = realloc(R, sizeof(P) * rcap); if (!R) die("out of memory"); } memcpy(R + np, loc, sizeof(P) * ln); np += ln; }
                free(loc);
            }
            int bb = 1; while ((1L << bb) < np) bb++;
            long nb = 1L << bb; int shift = 64 - bb;
            long *start = calloc(nb + 1, sizeof(long));
            for (long p = 0; p < np; p++) start[(R[p].h >> shift) + 1]++;
            for (long b = 0; b < nb; b++) start[b + 1] += start[b];
            P *S = malloc(sizeof(P) * (np ? np : 1)); long *fill = malloc(sizeof(long) * nb); memcpy(fill, start, sizeof(long) * nb);
            if (!S || !fill) die("out of memory (index)");
            for (long p = 0; p < np; p++) S[fill[R[p].h >> shift]++] = R[p];
            free(fill); free(R);
#pragma omp parallel
            {
                long bcap = 1 << 16, bn = 0; char *buf = malloc(bcap);
#pragma omp for schedule(dynamic, 4) reduction(+ : inst, lscan)
                for (long i = 0; i < nn[0]; i++) for (long j = (same01 ? i : 0); j < nn[1] && (FULL || LL[0]->omin[i]); j++) {
                    int v[HMX]; for (int k = 0; k < K; k++) v[k] = LL[0]->paf[i * K + k] + LL[1]->paf[j * K + k];
                    uint64_t h = vhash(v), r = hroute(h);
                    if ((long)(r % (uint64_t)shards) != shard || (int)((r / (uint64_t)shards) % (uint64_t)passes) != pass) continue;
                    if (!psd_ok(LL[0], i, LL[1], j)) continue;
                    lscan++;
                    long b = (long)(h >> shift);
                    for (long p = start[b]; p < start[b + 1]; p++) {
                        if (S[p].h != h) continue;
                        if (same12 && j > (long)S[p].i) continue;
                        long ci = S[p].i, di = S[p].j; int same = 1;
                        for (int k = 0; k < K && same; k++) if (v[k] + LL[2]->paf[ci * K + k] + LL[3]->paf[di * K + k] != 0) same = 0;
                        if (!same) continue;
                        long idx[4] = {i, j, ci, di};
                        int cc[4][HMX]; memset(cc, 0, sizeof cc);
                        for (int t = 0; t < 4; t++) for (int k = 0; k < HM; k++) cc[t][k] = LL[t]->c[(size_t)idx[t] * HM + k];
                        if (!FULL && !canonical(key4, cc)) continue;
                        inst++;
                        if (bn + 4096 > bcap) {
#pragma omp critical(outw)
                            fwrite(buf, 1, bn, out);
                            bn = 0;
                        }
                        bn += sprintf(buf + bn, "%d %d %d %d", key4[0], key4[1], key4[2], key4[3]);
                        for (int t = 0; t < 4; t++) for (int k = 0; k < HM; k++) bn += sprintf(buf + bn, " %d", cc[t][k]);
                        buf[bn++] = '\n';
                    }
                }
#pragma omp critical(outw)
                fwrite(buf, 1, bn, out);
                free(buf);
            }
            free(S); free(start);
        }
        printf("(%d,%d,%d,%d): lists %ld %ld %ld %ld, shard %d/%d, %d passes, right pairs %ld, left pairs %ld, %ld instances, %.1f s\n",
               key4[0], key4[1], key4[2], key4[3], nn[0], nn[1], nn[2], nn[3], shard, shards, passes, rcount, lscan, inst, omp_get_wtime() - t0);
        inst_total += inst;
    }
    return inst_total;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 6) { fprintf(stderr, "usage: pc2_inst PATTERN n d threads OUT [full] [shard shards]\n"); return 1; }
    setup(argv[1], atoi(argv[2]), atoi(argv[3]));
    omp_set_num_threads(atoi(argv[4]));
    int a = 6, shard = 0, shards = 1;
    if (argc > a && !strcmp(argv[a], "full")) { FULL = 1; a++; }
    if (argc == a + 2) { shard = atoi(argv[a]); shards = atoi(argv[a + 1]); if (shards < 1 || shard < 0 || shard >= shards) die("bad shard"); }
    else if (argc != a) die("bad arguments");
    const char *e = getenv("PC_MEM_MB"); double mb = e && *e ? atof(e) : 2048;
    if (!(mb > 0)) die("PC_MEM_MB must be positive");
    double t0 = omp_get_wtime();
    for (int j = 0; j < HM; j++) for (int k = 0; k < HM; k++) { COSM[j][k] = cos(2 * M_PI * j * k / M); SINM[j][k] = sin(2 * M_PI * j * k / M); }
    /* which lists the decompositions use */
    int ns = 4 - NK; long ndec = 1; for (int i = 0; i < ns; i++) ndec *= NS;
    for (long code = 0; code < ndec; code++) {
        long c2 = code; int sv[4], ok = 1, sq = NK;
        for (int i = 0; i < ns; i++) { sv[i] = 2 * (int)(c2 % NS) + 1; c2 /= NS; }
        for (int i = 1; i < ns; i++) if (sv[i] > sv[i - 1]) ok = 0;
        for (int i = 0; i < ns; i++) sq += sv[i] * sv[i];
        if (!ok || sq != 4 * N) continue;
        for (int i = 0; i < ns; i++) NEED[(sv[i] + 1) / 2] = 1;
        if (NK) NEED[0] = 1;
    }
    long tot = 0;
    e = getenv("PC_LISTS_ONLY"); COUNT_ONLY = e && *e && strcmp(e, "0");
    for (int i = 0; i <= NS; i++) if (NEED[i]) {
        build_list(i == 0, i == 0 ? 1 : 2 * i - 1, &LS[i]);
        if (!COUNT_ONLY) finish_list(&LS[i], i == 0);
        tot += LS[i].n;
        if (i == 0) printf("  skew: %d compressed rows\n", LS[i].n);
        else printf("  sum %2d: %d compressed rows\n", 2 * i - 1, LS[i].n);
    }
    printf("pc2_inst %s n=%d d=%d m=%d: %ld compressed rows in %.1f s\n", PAT, N, D, M, tot, omp_get_wtime() - t0);
    if (COUNT_ONLY) {   /* the pair scans the match would make, without making them */
        double scans = 0; long ndec2 = 1; for (int i = 0; i < ns; i++) ndec2 *= NS;
        for (long code = 0; code < ndec2; code++) {
            long c2 = code; int sv[4], ok = 1, sq = NK;
            for (int i = 0; i < ns; i++) { sv[i] = 2 * (int)(c2 % NS) + 1; c2 /= NS; }
            for (int i = 1; i < ns; i++) if (sv[i] > sv[i - 1]) ok = 0;
            for (int i = 0; i < ns; i++) sq += sv[i] * sv[i];
            if (!ok || sq != 4 * N) continue;
            double nn[4];
            for (int i = 0; i < 4; i++) nn[i] = KIND[i] ? LS[0].n : LS[(sv[i - NK] + 1) / 2].n;
            scans += nn[0] * nn[1] + nn[2] * nn[3];
        }
        printf("lists only: %.4g pair scans in the match\n", scans);
        return 0;
    }
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.part", argv[5]);
    FILE *out = fopen(tmp, "w"); if (!out) die("cannot write instance file");
    long inst = match(out, shard, shards, mb * 1048576.0);
    if (fflush(out) || ferror(out) || fclose(out)) die("write error on instance file");
    if (rename(tmp, argv[5])) die("cannot rename instance file");
    printf("total: %ld instances%s (%.1f s)\n", inst, FULL ? " (full)" : "", omp_get_wtime() - t0);
    return 0;
}
