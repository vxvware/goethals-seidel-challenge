/* Exact three-way join for the fixed-skew-block pool join (see PROVENANCE.md).
 * usage: pool_join plan hitsB hitsC hitsD      build: gcc -O3 -march=native -fopenmp -o pool_join pool_join.c
 * Each hits file holds one +/- row per line (output of pool_gen.c). Every row is rechecked against its windows
 * (rows outside them are counted and skipped), expanded over the multiplier group H of the plan, and packed into
 * a key of 2 bits per lag, lags ordered by fewest admissible value triples first. Pools are sorted and
 * deduplicated; the pool with most rows keeps one key per H-orbit (the least), so each orbit of solutions is
 * found once. A trie join over the three sorted pools runs in OpenMP tasks, switching to a whole-key test when
 * few candidates remain (env THR, default 64). Env PART=i/N runs one of N disjoint parts of the search; RED=b
 * picks the orbit-reduced pool. Every leaf is rechecked from the rows (all four autocorrelations, all lags)
 * before it is printed as "solution A,B,C,D". Windows may hold at most four values. */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NMAX 128
typedef unsigned __int128 u128;
typedef struct __attribute__((packed)) { u128 key; uint32_t id; } ent;   /* id = hit * nh + image index */
static int n, m, nh = 1, Hm[NMAX], row[3], lo[3][NMAX], hi[3][NMAX], PA[NMAX], ord[NMAX], need[NMAX], wd[3][NMAX];
static int8_t a[NMAX]; static ent *pool[3]; static long cnt[3]; static uint64_t *hits[3]; static long nhits[3]; static int red;   /* hits: 2 words of sign bits per row */
static long sols; static double nodes; static u128 ME, MO, TE, TO; static long THR = 64;   /* even/odd lag masks and packed targets: 4 bits of room per field */
static int cmp(const void *p, const void *q) { u128 x = ((const ent *)p)->key, y = ((const ent *)q)->key; return x < y ? -1 : x > y; }
static inline int dig(u128 k, int s) { return (int)(k >> (2 * (m - 1 - s))) & 3; }
static long lb(int b, long lo_, long hi_, int s, int v) { const ent *p = pool[b]; while (lo_ < hi_) { long md = (lo_ + hi_) >> 1; if (dig(p[md].key, s) < v) lo_ = md + 1; else hi_ = md; } return lo_; }
static void seq_of(int b, uint32_t id, int8_t *y) { const uint64_t *x = hits[b] + 2 * (long)(id / nh); int t = Hm[id % nh]; for (int j = 0; j < n; j++) { int q = (int)((long)t * j % n); y[j] = (x[q >> 6] >> (q & 63)) & 1 ? 1 : -1; } }
static void leaf(long i, long j, long k) {
    int8_t q[4][NMAX]; memcpy(q[0], a, n); seq_of(0, pool[0][i].id, q[1]); seq_of(1, pool[1][j].id, q[2]); seq_of(2, pool[2][k].id, q[3]);
    for (int s = 1; s <= m; s++) { int t = 0; for (int b = 0; b < 4; b++) for (int x = 0; x < n; x++) t += q[b][x] * q[b][(x + s) % n]; if (t) { fprintf(stderr, "recheck failed at a leaf\n"); exit(1); } }
    #pragma omp critical
    { sols++; printf("solution "); for (int b = 0; b < 4; b++) { for (int x = 0; x < n; x++) putchar(q[b][x] > 0 ? '+' : '-'); putchar(b < 3 ? ',' : '\n'); } fflush(stdout); }
}
static int PARTI, PARTN = 1; static long done4; static double tstart;   /* PART=i/N: only the depth-4 subtrees with path index = i mod N */
static void tj(int s, long b0, long b1, long c0, long c1, long d0, long d1, double *nd, long path) {
    (*nd)++;
    if (s == m) { for (long i = b0; i < b1; i++) for (long j = c0; j < c1; j++) for (long k = d0; k < d1; k++) leaf(i, j, k); return; }
    if ((b1 - b0) * (c1 - c0) <= THR && (b1 - b0) * (c1 - c0) * (d1 - d0) <= THR) {      /* few candidates: test whole keys, even and odd lags separately so digit sums cannot carry */
        for (long i = b0; i < b1; i++) { u128 kb = pool[0][i].key, be = kb & ME, bo = kb & MO;
            for (long j = c0; j < c1; j++) { u128 kc = pool[1][j].key, pe = TE - be - (kc & ME), po = TO - bo - (kc & MO);
                for (long k = d0; k < d1; k++) { u128 kd = pool[2][k].key; if ((kd & ME) == pe && (kd & MO) == po) leaf(i, j, k); } } }
        return;
    }
    long bb[6], cc[6], dd[6];
    for (int v = 0; v <= wd[0][s] + 1; v++) bb[v] = lb(0, b0, b1, s, v);
    for (int v = 0; v <= wd[1][s] + 1; v++) cc[v] = lb(1, c0, c1, s, v);
    for (int v = 0; v <= wd[2][s] + 1; v++) dd[v] = lb(2, d0, d1, s, v);
    for (int vb = 0; vb <= wd[0][s]; vb++) { if (bb[vb] == bb[vb + 1]) continue;
        for (int vc = 0; vc <= wd[1][s]; vc++) { if (cc[vc] == cc[vc + 1]) continue;
            int vd = need[s] - vb - vc; if (vd < 0 || vd > wd[2][s] || dd[vd] == dd[vd + 1]) continue;
            long np = path * 16 + vb * 4 + vc;
            if (s == 3 && np % PARTN != PARTI) continue;
            if (s < 4) {
                #pragma omp task firstprivate(s, vb, vc, vd, bb, cc, dd, np)
                { double loc = 0; tj(s + 1, bb[vb], bb[vb + 1], cc[vc], cc[vc + 1], dd[vd], dd[vd + 1], &loc, np);
                  #pragma omp atomic
                  nodes += loc;
                  if (s == 3) {
                      #pragma omp critical(prog)
                      { if ((++done4 & 31) == 0) fprintf(stderr, "progress: %ld depth-4 subtrees done, %.3g nodes, %.0f s\n", done4, nodes, omp_get_wtime() - tstart); } } }
            } else tj(s + 1, bb[vb], bb[vb + 1], cc[vc], cc[vc + 1], dd[vd], dd[vd + 1], nd, np); } }
}
int main(int argc, char **argv) {
    if (argc < 5) return 2;
    FILE *f = fopen(argv[1], "r"); char abuf[NMAX + 2]; if (!f || fscanf(f, "%d %s", &n, abuf) != 2) return 2; m = (n - 1) / 2;
    for (int b = 0; b < 3; b++) { if (fscanf(f, "%d", &row[b]) != 1) return 2; for (int s = 1; s <= m; s++) if (fscanf(f, "%d %d", &lo[b][s], &hi[b][s]) != 2) return 2; }
    { int hh; if (fscanf(f, "%d", &hh) == 1) { nh = hh; for (int i = 0; i < hh; i++) if (fscanf(f, "%d", &Hm[i]) != 1) return 2; } else Hm[0] = 1; }
    for (int j = 0; j < n; j++) a[j] = abuf[j] == '+' ? 1 : -1;
    for (int s = 1; s <= m; s++) { int t = 0; for (int j = 0; j < n; j++) t += a[j] * a[(j + s) % n]; PA[s] = t; }
    int key[NMAX], nd0[NMAX];
    for (int s = 0; s < m; s++) { nd0[s] = (-PA[s + 1] - lo[0][s + 1] - lo[1][s + 1] - lo[2][s + 1]) / 4; int c = 0, w0 = (hi[0][s + 1] - lo[0][s + 1]) / 4, w1 = (hi[1][s + 1] - lo[1][s + 1]) / 4, w2 = (hi[2][s + 1] - lo[2][s + 1]) / 4;
        for (int vb = 0; vb <= w0; vb++) for (int vc = 0; vc <= w1; vc++) { int r = nd0[s] - vb - vc; if (r >= 0 && r <= w2) c++; } key[s] = c; ord[s] = s; }
    for (int i = 1; i < m; i++) for (int j = i; j > 0 && key[ord[j]] < key[ord[j - 1]]; j--) { int t = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = t; }
    for (int s = 0; s < m; s++) { need[s] = nd0[ord[s]]; for (int b = 0; b < 3; b++) wd[b][s] = (hi[b][ord[s] + 1] - lo[b][ord[s] + 1]) / 4; }
    for (int s = 0; s < m; s++) { u128 f = (u128)3 << (2 * (m - 1 - s)), t = (u128)need[s] << (2 * (m - 1 - s)); if (s & 1) { MO |= f; TO += t; } else { ME |= f; TE += t; } }
    if (getenv("THR")) THR = atol(getenv("THR"));
    if (getenv("PART") && sscanf(getenv("PART"), "%d/%d", &PARTI, &PARTN) != 2) return 2;
    int red_env = getenv("RED") ? atoi(getenv("RED")) : -1;
    double t0 = omp_get_wtime();
    long nls[3]; for (int b = 0; b < 3; b++) { FILE *g = fopen(argv[2 + b], "r"); if (!g) return 2; fseek(g, 0, SEEK_END); nls[b] = ftell(g) / (n + 1); fclose(g); if (nls[b] > nls[red]) red = b; }
    if (red_env >= 0 && red_env < 3) red = red_env;   /* default: orbit-reduce the pool with most hits (least memory); RED picks another (least join work: the smallest) */
    for (int b = 0; b < 3; b++) {
        FILE *g = fopen(argv[2 + b], "r"); if (!g) return 2; long nl = nls[b];
        hits[b] = malloc(16 * (size_t)(nl + 1)); pool[b] = malloc(sizeof(ent) * (size_t)(b == red ? nl + nh : nl * nh + nh)); if (!hits[b] || !pool[b]) { fprintf(stderr, "out of memory\n"); return 1; } char line[NMAX + 8]; long h = 0, skipped = 0;
        while (fgets(line, sizeof line, g) && h < nl) {
            if ((int)strlen(line) < n) continue;
            int8_t x[NMAX]; uint64_t *hw = hits[b] + 2 * h; hw[0] = hw[1] = 0; int P[NMAX], sum = 0; for (int j = 0; j < n; j++) { x[j] = (int8_t)(line[j] == '+' ? 1 : -1); sum += x[j]; if (x[j] > 0) hw[j >> 6] |= 1ull << (j & 63); }
            int ok = sum == row[b];
            for (int s = 1; s <= m; s++) { int t = 0; for (int j = 0; j < n; j++) t += x[j] * x[(j + s) % n]; P[s] = t; }
            u128 kmin = 0; long first = cnt[b];
            for (int i = 0; i < nh && ok; i++) {
                u128 k = 0;
                for (int s = 0; s < m; s++) { int lag = ord[s] + 1, q = (int)((long)Hm[i] * lag % n); if (q > m) q = n - q; int pv = P[q]; if (pv < lo[b][lag] || pv > hi[b][lag]) { ok = 0; break; } k = (k << 2) | (u128)((pv - lo[b][lag]) / 4); }
                if (!ok) break;
                if (i == 0 || k < kmin) kmin = k;
                pool[b][cnt[b]].key = k; pool[b][cnt[b]++].id = (uint32_t)(h * nh + i);
            }
            if (!ok) { cnt[b] = first; skipped++; continue; }   /* rows from a wider plan are filtered here */
            if (b == red) { long w = first; for (long i = first; i < cnt[b]; i++) if (pool[b][i].key == kmin) { pool[b][w++] = pool[b][i]; break; } cnt[b] = w; }
            h++;
        }
        fclose(g); nhits[b] = h;
        qsort(pool[b], (size_t)cnt[b], sizeof(ent), cmp);
        long w = 0; for (long i = 0; i < cnt[b]; i++) if (w == 0 || pool[b][i].key != pool[b][w - 1].key) pool[b][w++] = pool[b][i];
        cnt[b] = w;
        printf("pool %d: row %d, %ld hits (%ld rows outside the windows skipped), %ld distinct keys%s\n", b, row[b], h, skipped, w, b == red ? " (one per orbit)" : ""); fflush(stdout);
    }
    #pragma omp parallel
    #pragma omp single
    { double loc = 0; tstart = omp_get_wtime(); tj(0, 0, cnt[0], 0, cnt[1], 0, cnt[2], &loc, 0);
      #pragma omp atomic
      nodes += loc; }
    printf("n=%d: %ld solution orbits (of %d), %.3g nodes, %.1f s on %d threads\n", n, sols, nh, nodes, omp_get_wtime() - t0, omp_get_max_threads());
    return 0;
}
