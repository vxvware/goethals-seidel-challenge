/* Pool generator for the fixed-skew-block pool join, restricted to blocks that are constant on the orbits of the
 * plan's multiplier group H (see PROVENANCE.md). Same plan and output format as pool_gen.c, so pool_join.c joins
 * its pools unchanged. The plan's row sums must be attainable by such blocks (orbit_rows.py lists the triples
 * that are; pass one to pool_plan.py as ROWS). Hits are not deduplicated.
 * usage: pool_gen_orbit plan b out temp seed hits_wanted      env KR = restart length (default 200000)
 * build: gcc -O3 -march=native -fopenmp -o pool_gen_orbit pool_gen_orbit.c -lm
 * A move flips one orbit, or the entry at 0. Energy: sum over lag classes of (distance to window / 4)^2 plus
 * ((|row sum| - r) / 2)^2 for the plan's row sum r. Every hit is rechecked from the row before it is written.
 * The state is one sign e_i per orbit O_i (the entry at 0 is an orbit of its own). For a lag class with
 * representative s the autocorrelation is sum_{i,j} e_i e_j M[i][j] with M[i][j] = #{u in O_i : u + s in O_j}, so
 * flipping orbit a changes it by -2 e_a h[a], h[a] = sum_{j != a} e_j (M[a][j] + M[j][a]). The fields h are
 * kept up to date: a move costs one pass over the classes and an accepted move one pass over orbits x classes,
 * both on 16-bit lanes. */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NMAX 512
int main(int argc, char **argv) {
    if (argc < 7) return 2;
    FILE *f = fopen(argv[1], "r"); int blk = atoi(argv[2]); double temp = atof(argv[4]); uint64_t seed = strtoull(argv[5], 0, 10); long want = atol(argv[6]);
    int n, m, row[3], nh, H[NMAX]; static int lo3[3][NMAX], hi3[3][NMAX]; char abuf[NMAX + 2];
    if (!f || fscanf(f, "%d %s", &n, abuf) != 2) return 2; m = (n - 1) / 2;
    for (int b = 0; b < 3; b++) { if (fscanf(f, "%d", &row[b]) != 1) return 2; for (int s = 1; s <= m; s++) if (fscanf(f, "%d %d", &lo3[b][s], &hi3[b][s]) != 2) return 2; }
    if (fscanf(f, "%d", &nh) != 1) return 2; for (int i = 0; i < nh; i++) if (fscanf(f, "%d", &H[i]) != 1) return 2;
    static int orb[NMAX], osz[NMAX], crep[NMAX]; int K = 0, nc = 0;
    for (int x = 0; x < n; x++) orb[x] = -1;
    for (int x = 1; x < n; x++) if (orb[x] < 0) { osz[K] = 0; for (int i = 0; i < nh; i++) { int y = (int)((long)x * H[i] % n); if (orb[y] < 0) { orb[y] = K; osz[K]++; } } K++; }
    orb[0] = K; osz[K] = 1; int V = K + 1;
    static int16_t lo[NMAX] __attribute__((aligned(64))), hi[NMAX] __attribute__((aligned(64))); static int diag[NMAX];
    { static char seen[NMAX]; for (int s = 1; s <= m; s++) if (!seen[s]) { for (int i = 0; i < nh; i++) { int y = (int)((long)s * H[i] % n); if (y > m) y = n - y; seen[y] = 1; if (lo3[blk][y] != lo3[blk][s] || hi3[blk][y] != hi3[blk][s]) { fprintf(stderr, "windows are not H-invariant\n"); return 2; } } lo[nc] = (int16_t)lo3[blk][s]; hi[nc] = (int16_t)hi3[blk][s]; crep[nc++] = s; } }
    int C = (nc + 31) & ~31;                      /* classes padded to a multiple of 32; padding lanes stay at zero */
    /* S2[a][j][c] = 4 (M_c[a][j] + M_c[j][a]) for j != a, 0 on the diagonal; the fields are kept as Hh = 2 h */
    int16_t *S2 = aligned_alloc(64, (size_t)V * V * C * 2); memset(S2, 0, (size_t)V * V * C * 2);
    for (int c = 0; c < nc; c++) for (int u = 0; u < n; u++) { int i = orb[u], j = orb[(u + crep[c]) % n]; if (i == j) diag[c]++; else { S2[((size_t)i * V + j) * C + c] += 4; S2[((size_t)j * V + i) * C + c] += 4; } }
    FILE *out = fopen(argv[3], "a"); if (!out) return 2;
    long hits = 0; double evals = 0, t0 = omp_get_wtime(); long KR = getenv("KR") ? atol(getenv("KR")) : 200000; int R = row[blk];
    #pragma omp parallel reduction(+:evals)
    {
        uint64_t rs = (seed + 1000003ull * omp_get_thread_num()) * 0x9E3779B97F4A7C15ull + 1;
        #define RND() (rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17, rs)
        int e[NMAX]; int16_t *Hh = aligned_alloc(64, (size_t)V * C * 2);
        int16_t P[NMAX] __attribute__((aligned(64))), D[NMAX] __attribute__((aligned(64)));
        double et[4096]; for (int i = 0; i < 4096; i++) et[i] = exp(-i / temp);
        #define RPEN(r) ({ int a_ = ((r) < 0 ? -(r) : (r)) - R; (long)(a_ / 2) * (a_ / 2); })
        while (1) {
            long hnow;
            #pragma omp atomic read
            hnow = hits;
            if (hnow >= want) break;
            int r = 0; e[K] = (RND() >> 20) & 1 ? 1 : -1;
            for (int i = 0; i < K; i++) e[i] = (RND() >> 20) & 1 ? 1 : -1;
            for (int i = 0; i < V; i++) r += e[i] * osz[i];
            memset(Hh, 0, (size_t)V * C * 2);
            for (int a = 0; a < V; a++) { int16_t *h = Hh + (size_t)a * C; for (int j = 0; j < V; j++) { const int16_t *s2 = S2 + ((size_t)a * V + j) * C; if (e[j] > 0) for (int c = 0; c < C; c++) h[c] += s2[c]; else for (int c = 0; c < C; c++) h[c] -= s2[c]; } }
            for (int q = 0; q < V * C; q++) Hh[q] /= 2;
            long E = RPEN(r);
            for (int c = 0; c < C; c++) P[c] = 0;
            for (int c = 0; c < nc; c++) { int t = 0; for (int a = 0; a < V; a++) t += e[a] * Hh[(size_t)a * C + c]; t = t / 4 + diag[c]; P[c] = (int16_t)t; int v = t < lo[c] ? lo[c] - t : t > hi[c] ? t - hi[c] : 0; E += (long)(v / 4) * (v / 4); }
            long k;
            for (k = 0; k < KR && E > 0; k++) {
                int a = (int)(RND() % (uint64_t)V); a = a == 0 ? K : a - 1;   /* same order of choices as pool_gen_orbit.c */
                int ea = e[a], dr = -2 * ea * osz[a];
                const int16_t *h = Hh + (size_t)a * C; int32_t acc = 0;
                #pragma omp simd reduction(+:acc) aligned(h, P, D, lo, hi : 64)
                for (int c = 0; c < C; c++) {
                    int16_t d = (int16_t)(ea > 0 ? -h[c] : h[c]); D[c] = d;
                    int16_t t = (int16_t)(P[c] + d), u1 = (int16_t)(lo[c] - t), u2 = (int16_t)(t - hi[c]);
                    int16_t v = u1 > u2 ? u1 : u2; v = v > 0 ? v : 0; v = (int16_t)(v >> 2);
                    acc += (int32_t)v * v;
                }
                long E2 = RPEN(r + dr) + acc, dE = E2 - E;
                if (dE <= 0 || (dE < 4096 && (RND() >> 11) * (1.0 / 9007199254740992.0) < et[dE])) {
                    for (int c = 0; c < C; c++) P[c] += D[c];
                    const int16_t *s2 = S2 + (size_t)a * V * C; int tot = V * C;
                    if (ea > 0) { for (int q = 0; q < tot; q++) Hh[q] -= s2[q]; } else { for (int q = 0; q < tot; q++) Hh[q] += s2[q]; }
                    e[a] = -ea; E = E2; r += dr;
                }
            }
            evals += (double)k;
            if (E) continue;
            int8_t x[NMAX]; for (int u = 0; u < n; u++) x[u] = (int8_t)(r < 0 ? -e[orb[u]] : e[orb[u]]);
            int ok = 1, sum = 0; for (int u = 0; u < n; u++) sum += x[u];
            for (int s = 1; s <= m; s++) { int t = 0; for (int u = 0; u < n; u++) t += x[u] * x[(u + s) % n]; ok &= t >= lo3[blk][s] && t <= hi3[blk][s]; }
            if (!ok || sum != R) { fprintf(stderr, "recheck failed\n"); exit(1); }
            char line[NMAX + 2]; for (int u = 0; u < n; u++) line[u] = x[u] > 0 ? '+' : '-'; line[n] = '\n';
            #pragma omp critical
            { fwrite(line, 1, n + 1, out); hits++; if ((hits & 15) == 0) fflush(out); }
        }
        free(Hh);
    }
    fclose(out);
    printf("block %d row %d: %ld hits, %.3g evals, %.3g per hit, %.3g evals/s on %d threads\n", blk, R, hits, evals, evals / (hits ? hits : 1), evals / (omp_get_wtime() - t0), omp_get_max_threads());
    return 0;
}
