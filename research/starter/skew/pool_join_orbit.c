/* Exact three-way join for the orbit-restricted pool join (see PROVENANCE.md): pool_join.c for pools whose rows are
 * constant on the orbits of the plan's multiplier group H, as written by pool_gen_orbit.c.
 * usage: pool_join_orbit plan hitsB hitsC hitsD      build: gcc -O3 -march=native -fopenmp -o pool_join_orbit pool_join_orbit.c
 * The autocorrelation of such a row is constant on the classes of lags under <H,-1>, so the key takes one lag per
 * class, 4 bits each (windows of up to 16 values, at most 24 classes), and rows are not expanded over H. Rows that
 * are not orbit-constant, have the wrong row sum or leave their windows are counted and skipped. Pools are sorted
 * and deduplicated, and a trie join over the three sorted pools runs in OpenMP tasks, switching to a whole-key
 * test when few candidates remain (env THR, default 64); env PART=i/N runs one of N disjoint parts. Every leaf is
 * rechecked from the rows (all four autocorrelations, all lags) before it is printed as "solution A,B,C,D".
 * Lengths up to 255. */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NMAX 256
typedef unsigned __int128 u128;
typedef u128 ent;   /* key << 32 | id, id = hit * nh + image index; needs 2 m + 32 <= 128 */
#define KEY(e) ((e) & ~(u128)0xffffffffu)   /* key left in place: all key arithmetic is done 32 bits up */
#define ID(e) ((uint32_t)(e))
static int n, m, mf, nx = 1, orbit, sel[NMAX], nh = 1, Hm[NMAX], row[3], lo[3][NMAX], hi[3][NMAX], PA[NMAX], ord[NMAX], need[NMAX], wd[3][NMAX];
static int8_t a[NMAX]; static ent *pool[3]; static long cnt[3]; static uint64_t *hits[3]; static long nhits[3]; static int red;   /* hits: 2 words of sign bits per row */
static long sols; static double nodes; static u128 ME, MO, TE, TO; static long THR = 64;   /* even/odd lag masks and packed targets: 4 bits of room per field */
static int cmp(const void *p, const void *q) { u128 x = *(const ent *)p, y = *(const ent *)q; return x < y ? -1 : x > y; }
#define DB 4   /* bits per key digit: windows of up to 16 values */
static inline int dig(u128 k, int s) { return (int)(k >> (32 + DB * (m - 1 - s))) & ((1 << DB) - 1); }
static long lb(int b, long lo_, long hi_, int s, int v) { const ent *p = pool[b]; while (lo_ < hi_) { long md = (lo_ + hi_) >> 1; if (dig(p[md], s) < v) lo_ = md + 1; else hi_ = md; } return lo_; }
static void seq_of(int b, uint32_t id, int8_t *y) { const uint64_t *x = hits[b] + 4 * (long)(id / nx); int t = Hm[id % nx]; for (int j = 0; j < n; j++) { int q = (int)((long)t * j % n); y[j] = (x[q >> 6] >> (q & 63)) & 1 ? 1 : -1; } }
static void leaf(long i, long j, long k) {
    int8_t q[4][NMAX]; memcpy(q[0], a, n); seq_of(0, ID(pool[0][i]), q[1]); seq_of(1, ID(pool[1][j]), q[2]); seq_of(2, ID(pool[2][k]), q[3]);
    for (int s = 1; s <= mf; s++) { int t = 0; for (int b = 0; b < 4; b++) for (int x = 0; x < n; x++) t += q[b][x] * q[b][(x + s) % n]; if (t) { fprintf(stderr, "recheck failed at a leaf\n"); exit(1); } }
    #pragma omp critical
    { sols++; printf("solution "); for (int b = 0; b < 4; b++) { for (int x = 0; x < n; x++) putchar(q[b][x] > 0 ? '+' : '-'); putchar(b < 3 ? ',' : '\n'); } fflush(stdout); }
}
static int PARTI, PARTN = 1; static long done4; static double tstart;   /* PART=i/N: only the depth-4 subtrees with path index = i mod N */
static void tj(int s, long b0, long b1, long c0, long c1, long d0, long d1, double *nd, long path) {
    (*nd)++;
    if (s == m) { for (long i = b0; i < b1; i++) for (long j = c0; j < c1; j++) for (long k = d0; k < d1; k++) leaf(i, j, k); return; }
    if ((b1 - b0) * (c1 - c0) <= THR && (b1 - b0) * (c1 - c0) * (d1 - d0) <= THR) {      /* few candidates: test whole keys, even and odd lags separately so digit sums cannot carry */
        for (long i = b0; i < b1; i++) { u128 kb = KEY(pool[0][i]), be = kb & ME, bo = kb & MO;
            for (long j = c0; j < c1; j++) { u128 kc = KEY(pool[1][j]), pe = TE - be - (kc & ME), po = TO - bo - (kc & MO);
                for (long k = d0; k < d1; k++) { u128 kd = KEY(pool[2][k]); if ((kd & ME) == pe && (kd & MO) == po) leaf(i, j, k); } } }
        return;
    }
    long bb[18], cc[18], dd[18];
    for (int v = 0; v <= wd[0][s] + 1; v++) bb[v] = lb(0, b0, b1, s, v);
    for (int v = 0; v <= wd[1][s] + 1; v++) cc[v] = lb(1, c0, c1, s, v);
    for (int v = 0; v <= wd[2][s] + 1; v++) dd[v] = lb(2, d0, d1, s, v);
    for (int vb = 0; vb <= wd[0][s]; vb++) { if (bb[vb] == bb[vb + 1]) continue;
        for (int vc = 0; vc <= wd[1][s]; vc++) { if (cc[vc] == cc[vc + 1]) continue;
            int vd = need[s] - vb - vc; if (vd < 0 || vd > wd[2][s] || dd[vd] == dd[vd + 1]) continue;
            long np = path * 256 + vb * 16 + vc;
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
    FILE *f = fopen(argv[1], "r"); char abuf[NMAX + 2]; if (!f || fscanf(f, "%d %s", &n, abuf) != 2) return 2; mf = (n - 1) / 2; if (n > 255) { fprintf(stderr, "n too large\n"); return 2; }
    for (int b = 0; b < 3; b++) { if (fscanf(f, "%d", &row[b]) != 1) return 2; for (int s = 1; s <= mf; s++) if (fscanf(f, "%d %d", &lo[b][s], &hi[b][s]) != 2) return 2; }
    { int hh; if (fscanf(f, "%d", &hh) == 1) { nh = hh; for (int i = 0; i < hh; i++) if (fscanf(f, "%d", &Hm[i]) != 1) return 2; } else Hm[0] = 1; }
    for (int j = 0; j < n; j++) a[j] = abuf[j] == '+' ? 1 : -1;
    for (int s = 1; s <= mf; s++) { int t = 0; for (int j = 0; j < n; j++) t += a[j] * a[(j + s) % n]; PA[s] = t; }
    /* one lag per class under <H,-1> */
    orbit = 1; nx = 1; m = 0;
    { char seen[NMAX] = {0}; for (int s = 1; s <= mf; s++) { if (orbit && seen[s]) continue; sel[m++] = s; for (int i = 0; i < nh; i++) { int q = (int)((long)Hm[i] * s % n); if (q > mf) q = n - q; seen[q] = 1; } } }
    if (DB * m + 32 > 128) { fprintf(stderr, "too many lags for the packed key\n"); return 2; }
    int key[NMAX], nd0[NMAX];
    for (int s = 0; s < m; s++) { int L = sel[s]; nd0[s] = (-PA[L] - lo[0][L] - lo[1][L] - lo[2][L]) / 4; int c = 0, w0 = (hi[0][L] - lo[0][L]) / 4, w1 = (hi[1][L] - lo[1][L]) / 4, w2 = (hi[2][L] - lo[2][L]) / 4;
        for (int vb = 0; vb <= w0; vb++) for (int vc = 0; vc <= w1; vc++) { int r = nd0[s] - vb - vc; if (r >= 0 && r <= w2) c++; } key[s] = c; ord[s] = s; }
    for (int i = 1; i < m; i++) for (int j = i; j > 0 && key[ord[j]] < key[ord[j - 1]]; j--) { int t = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = t; }
    for (int s = 0; s < m; s++) { need[s] = nd0[ord[s]]; for (int b = 0; b < 3; b++) wd[b][s] = (hi[b][sel[ord[s]]] - lo[b][sel[ord[s]]]) / 4; }
    for (int s = 0; s < m; s++) { u128 f = (u128)((1 << DB) - 1) << (32 + DB * (m - 1 - s)), t = (u128)need[s] << (32 + DB * (m - 1 - s)); if (s & 1) { MO |= f; TO += t; } else { ME |= f; TE += t; } }
    if (getenv("THR")) THR = atol(getenv("THR"));
    if (getenv("PART") && sscanf(getenv("PART"), "%d/%d", &PARTI, &PARTN) != 2) return 2;
    int red_env = getenv("RED") ? atoi(getenv("RED")) : -1;
    double t0 = omp_get_wtime();
    long nls[3]; for (int b = 0; b < 3; b++) { FILE *g = fopen(argv[2 + b], "r"); if (!g) return 2; fseek(g, 0, SEEK_END); nls[b] = ftell(g) / (n + 1); fclose(g); if (nls[b] > nls[red]) red = b; }
    if (red_env >= 0 && red_env < 3) red = red_env;   /* default: orbit-reduce the pool with most hits (least memory); RED picks another (least join work: the smallest) */
    for (int b = 0; b < 3; b++) {
        FILE *g = fopen(argv[2 + b], "r"); if (!g) return 2; long nl = nls[b];
        hits[b] = malloc(32 * (size_t)(nl + 1)); pool[b] = malloc(sizeof(ent) * (size_t)(b == red ? nl + nx : nl * nx + nx)); if (!hits[b] || !pool[b]) { fprintf(stderr, "out of memory\n"); return 1; } char line[NMAX + 8]; long h = 0, skipped = 0;
        while (fgets(line, sizeof line, g) && h < nl) {
            if ((int)strlen(line) < n) continue;
            int8_t x[NMAX]; uint64_t *hw = hits[b] + 4 * h; hw[0] = hw[1] = hw[2] = hw[3] = 0; int P[NMAX], sum = 0; for (int j = 0; j < n; j++) { x[j] = (int8_t)(line[j] == '+' ? 1 : -1); sum += x[j]; if (x[j] > 0) hw[j >> 6] |= 1ull << (j & 63); }
            int ok = sum == row[b];
            if (orbit) for (int i = 1; i < nh && ok; i++) for (int j = 0; j < n; j++) if (x[(int)((long)Hm[i] * j % n)] != x[j]) { ok = 0; break; }
            for (int s = 1; s <= mf; s++) { int t = 0; for (int j = 0; j < n; j++) t += x[j] * x[(j + s) % n]; P[s] = t; }
            u128 kmin = 0; long first = cnt[b];
            for (int i = 0; i < nx && ok; i++) {
                u128 k = 0;
                for (int s = 0; s < m; s++) { int lag = sel[ord[s]], q = (int)((long)Hm[i] * lag % n); if (q > mf) q = n - q; int pv = P[q]; if (pv < lo[b][lag] || pv > hi[b][lag]) { ok = 0; break; } k = (k << DB) | (u128)((pv - lo[b][lag]) / 4); }
                if (!ok) break;
                if (i == 0 || k < kmin) kmin = k;
                pool[b][cnt[b]++] = (k << 32) | (uint32_t)(h * nx + i);
            }
            if (!ok) { cnt[b] = first; skipped++; continue; }   /* rows from a wider plan are filtered here */
            if (b == red) { long w = first; for (long i = first; i < cnt[b]; i++) if (KEY(pool[b][i]) == kmin << 32) { pool[b][w++] = pool[b][i]; break; } cnt[b] = w; }
            h++;
        }
        fclose(g); nhits[b] = h;
        qsort(pool[b], (size_t)cnt[b], sizeof(ent), cmp);
        long w = 0; for (long i = 0; i < cnt[b]; i++) if (w == 0 || KEY(pool[b][i]) != KEY(pool[b][w - 1])) pool[b][w++] = pool[b][i];
        cnt[b] = w;
        printf("pool %d: row %d, %ld hits (%ld rows outside the windows skipped), %ld distinct keys\n", b, row[b], h, skipped, w); fflush(stdout);
    }
    #pragma omp parallel
    #pragma omp single
    { double loc = 0; tstart = omp_get_wtime(); tj(0, 0, cnt[0], 0, cnt[1], 0, cnt[2], &loc, 0);
      #pragma omp atomic
      nodes += loc; }
    printf("n=%d: %ld solutions, %.3g nodes, %.1f s on %d threads\n", n, sols, nodes, omp_get_wtime() - t0, omp_get_max_threads());
    return 0;
}
