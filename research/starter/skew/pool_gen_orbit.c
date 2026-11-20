/* Pool generator for the fixed-skew-block pool join, restricted to blocks that are constant on the orbits of the
 * plan's multiplier group H (see PROVENANCE.md). Same plan and output format as pool_gen.c, so pool_join.c joins
 * its pools unchanged. The plan's row sums must be attainable by such blocks (orbit_rows.py lists the triples
 * that are; pass one to pool_plan.py as ROWS). Hits are not deduplicated.
 * usage: pool_gen_orbit plan b out temp seed hits_wanted      env KR = restart length (default 200000)
 * build: gcc -O3 -march=native -fopenmp -o pool_gen_orbit pool_gen_orbit.c -lm
 * A move flips one orbit, or the entry at 0. Energy: sum over lag classes of (distance to window / 4)^2 plus
 * ((|row sum| - r) / 2)^2 for the plan's row sum r. Every hit is rechecked from the row before it is written. */
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
    static int orb[NMAX], osz[NMAX], *omem[NMAX], crep[NMAX], lo[NMAX], hi[NMAX]; int K = 0, nc = 0;
    for (int x = 0; x < n; x++) orb[x] = -1;
    for (int x = 1; x < n; x++) if (orb[x] < 0) { omem[K] = malloc(sizeof(int) * nh); osz[K] = 0; for (int i = 0; i < nh; i++) { int y = (int)((long)x * H[i] % n); if (orb[y] < 0) { orb[y] = K; omem[K][osz[K]++] = y; } } K++; }
    { static char seen[NMAX]; for (int s = 1; s <= m; s++) if (!seen[s]) { for (int i = 0; i < nh; i++) { int y = (int)((long)s * H[i] % n); if (y > m) y = n - y; seen[y] = 1; if (lo3[blk][y] != lo3[blk][s] || hi3[blk][y] != hi3[blk][s]) { fprintf(stderr, "windows are not H-invariant\n"); return 2; } } lo[nc] = lo3[blk][s]; hi[nc] = hi3[blk][s]; crep[nc++] = s; } }
    FILE *out = fopen(argv[3], "a"); if (!out) return 2;
    long hits = 0; double evals = 0, t0 = omp_get_wtime(); long KR = getenv("KR") ? atol(getenv("KR")) : 200000; int R = row[blk];
    #pragma omp parallel reduction(+:evals)
    {
        uint64_t rs = (seed + 1000003ull * omp_get_thread_num()) * 0x9E3779B97F4A7C15ull + 1;
        #define RND() (rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17, rs)
        int8_t x[NMAX]; int P[NMAX], D[NMAX], F[NMAX]; char inF[NMAX] = {0};
        double et[4096]; for (int i = 0; i < 4096; i++) et[i] = exp(-i / temp);
        #define PEN(t, c) ({ int v_ = (t) < lo[c] ? lo[c] - (t) : (t) > hi[c] ? (t) - hi[c] : 0; (long)(v_ / 4) * (v_ / 4); })
        #define RPEN(r) ({ int a_ = ((r) < 0 ? -(r) : (r)) - R; (long)(a_ / 2) * (a_ / 2); })
        while (1) {
            long hnow;
            #pragma omp atomic read
            hnow = hits;
            if (hnow >= want) break;
            int r = 0; x[0] = (RND() >> 20) & 1 ? 1 : -1; r = x[0];
            for (int i = 0; i < K; i++) { int e = (RND() >> 20) & 1 ? 1 : -1; for (int j = 0; j < osz[i]; j++) x[omem[i][j]] = (int8_t)e; r += e * osz[i]; }
            long E = RPEN(r);
            for (int c = 0; c < nc; c++) { int s = crep[c], t = 0; for (int u = 0; u < n; u++) t += x[u] * x[(u + s) % n]; P[c] = t; E += PEN(t, c); }
            long k;
            for (k = 0; k < KR && E > 0; k++) {
                int a = (int)(RND() % (uint64_t)(K + 1)) - 1, nfl = 0, dr;
                if (a < 0) { F[nfl++] = 0; dr = -2 * x[0]; } else { for (int j = 0; j < osz[a]; j++) F[nfl++] = omem[a][j]; dr = -2 * x[omem[a][0]] * osz[a]; }
                for (int j = 0; j < nfl; j++) inF[F[j]] = 1;
                long E2 = RPEN(r + dr);
                for (int c = 0; c < nc; c++) {
                    int s = crep[c], d = 0;
                    for (int j = 0; j < nfl; j++) { int u = F[j], p = u + s; if (p >= n) p -= n; int q = u - s; if (q < 0) q += n; if (!inF[p]) d += x[u] * x[p]; if (!inF[q]) d += x[u] * x[q]; }
                    D[c] = -2 * d; E2 += PEN(P[c] + D[c], c);
                }
                for (int j = 0; j < nfl; j++) inF[F[j]] = 0;
                long dE = E2 - E;
                if (dE <= 0 || (dE < 4096 && (RND() >> 11) * (1.0 / 9007199254740992.0) < et[dE])) {
                    for (int j = 0; j < nfl; j++) x[F[j]] = (int8_t)-x[F[j]];
                    for (int c = 0; c < nc; c++) P[c] += D[c];
                    E = E2; r += dr;
                }
            }
            evals += (double)k;
            if (E) continue;
            if (r < 0) for (int u = 0; u < n; u++) x[u] = (int8_t)-x[u];
            int ok = 1, sum = 0; for (int u = 0; u < n; u++) sum += x[u];
            for (int s = 1; s <= m; s++) { int t = 0; for (int u = 0; u < n; u++) t += x[u] * x[(u + s) % n]; ok &= t >= lo3[blk][s] && t <= hi3[blk][s]; }
            if (!ok || sum != R) { fprintf(stderr, "recheck failed\n"); exit(1); }
            char line[NMAX + 2]; for (int u = 0; u < n; u++) line[u] = x[u] > 0 ? '+' : '-'; line[n] = '\n';
            #pragma omp critical
            { fwrite(line, 1, n + 1, out); hits++; if ((hits & 15) == 0) fflush(out); }
        }
    }
    fclose(out);
    printf("block %d row %d: %ld hits, %.3g evals, %.3g per hit, %.3g evals/s on %d threads\n", blk, R, hits, evals, evals / (hits ? hits : 1), evals / (omp_get_wtime() - t0), omp_get_max_threads());
    return 0;
}
