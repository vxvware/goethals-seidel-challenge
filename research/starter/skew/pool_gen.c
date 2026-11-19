/* Pool generator for the fixed-skew-block pool join (see PROVENANCE.md). Reads a plan written by pool_plan.py and
 * a block index b (0..2), and appends to the output file every sequence it finds with the block's row sum whose
 * periodic autocorrelation lies in the block's window at every lag: one line of +/- per hit, not deduplicated
 * and not expanded over the plan's multipliers. One annealer per OpenMP thread; needs AVX2.
 * usage: pool_gen plan b out temp seed hits_wanted      env KR = restart length (default 300000)
 * build: gcc -O3 -march=native -fopenmp -o pool_gen pool_gen.c -lm
 * Annealing with swap moves at fixed row sum; energy = sum over lags of (distance to window / 4)^2.
 * A swap of positions j (sign +) and j2 (sign -) changes PAF(s) by -2 (x[j+s] + x[j-s] - x[j2+s] - x[j2-s]),
 * minus 4 at the lag joining j and j2. Every hit is rechecked from the row before it is written. */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NMAX 128
#define M 64
int main(int argc, char **argv) {
    if (argc < 7) return 2;
    FILE *f = fopen(argv[1], "r"); int blk = atoi(argv[2]); double temp = atof(argv[4]); uint64_t seed = strtoull(argv[5], 0, 10); long want = atol(argv[6]);
    int n, m, row[3]; static int lo3[3][NMAX], hi3[3][NMAX]; char abuf[NMAX + 2];
    if (!f || fscanf(f, "%d %s", &n, abuf) != 2) return 2; m = (n - 1) / 2; if (m > M) return 2;
    for (int b = 0; b < 3; b++) { if (fscanf(f, "%d", &row[b]) != 1) return 2; for (int s = 1; s <= m; s++) if (fscanf(f, "%d %d", &lo3[b][s], &hi3[b][s]) != 2) return 2; }
    FILE *out = fopen(argv[3], "a"); if (!out) return 2;
    long hits = 0; double evals = 0, t0 = omp_get_wtime(); long KR = getenv("KR") ? atol(getenv("KR")) : 300000;
    static int8_t corr[NMAX][M] __attribute__((aligned(32)));      /* corr[dl]: -4 at lag dl */
    for (int dl = 1; dl <= m; dl++) corr[dl][dl - 1] = -4;
    int nv = (m + 15) / 16;
    #pragma omp parallel reduction(+:evals)
    {
        uint64_t rs = (seed + 1000003ull * omp_get_thread_num()) * 0x9E3779B97F4A7C15ull + 1;
        #define RND() (rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17, rs)
        int8_t x[3 * NMAX + M] = {0}, xr[3 * NMAX + M] = {0};      /* xr[k] = x[-k]; three copies each */
        int16_t P[M] __attribute__((aligned(32))) = {0}, PN[M] __attribute__((aligned(32))), lo[M] __attribute__((aligned(32))), hi[M] __attribute__((aligned(32))), msk[M] __attribute__((aligned(32)));
        for (int s = 0; s < M; s++) { lo[s] = -30000; hi[s] = 30000; msk[s] = s < m ? -1 : 0; }
        for (int s = 1; s <= m; s++) { lo[s - 1] = (int16_t)lo3[blk][s]; hi[s - 1] = (int16_t)hi3[blk][s]; }
        double et[64]; for (int i = 0; i < 64; i++) et[i] = exp(-i / temp);
        int pl[NMAX], mn[NMAX], np = (n + row[blk]) / 2, nm = n - np;
        #define SETV(j, v) do { int j_ = (j), k_ = (n - j_) % n; int8_t v_ = (int8_t)(v); x[j_] = x[j_ + n] = x[j_ + 2 * n] = v_; xr[k_] = xr[k_ + n] = xr[k_ + 2 * n] = v_; } while (0)
        const __m256i zero = _mm256_setzero_si256();
        while (1) {
            long hnow;
            #pragma omp atomic read
            hnow = hits;
            if (hnow >= want) break;
            int idx[NMAX];
            for (int j = 0; j < n; j++) idx[j] = j;
            for (int j = n - 1; j > 0; j--) { int k = (int)(RND() % (uint64_t)(j + 1)), t = idx[j]; idx[j] = idx[k]; idx[k] = t; }
            for (int j = 0; j < n; j++) { SETV(idx[j], j < np ? 1 : -1); if (j < np) pl[j] = idx[j]; else mn[j - np] = idx[j]; }
            int E = 0;
            for (int s = 1; s <= m; s++) { int t = 0; for (int j = 0; j < n; j++) t += x[j] * x[j + s]; P[s - 1] = (int16_t)t; int v = t < lo[s - 1] ? lo[s - 1] - t : t > hi[s - 1] ? t - hi[s - 1] : 0; E += (v / 4) * (v / 4); }
            long k;
            for (k = 0; k < KR && E > 0; k++) {
                uint64_t r = RND();
                int pi = (int)(((r >> 32) * (uint64_t)np) >> 32), mi = (int)(((r & 0xffffffffu) * (uint64_t)nm) >> 32);
                int j = pl[pi], j2 = mn[mi];
                const int8_t *p1 = x + j + n + 1, *q1 = xr + ((n - j) % n) + n + 1, *p2 = x + j2 + n + 1, *q2 = xr + ((n - j2) % n) + n + 1;
                int dl = j2 > j ? j2 - j : j - j2; if (dl > m) dl = n - dl;
                __m256i acc = zero;
                for (int h = 0; h < 2; h++) {
                    if (32 * h >= m) break;
                    __m256i t = _mm256_sub_epi8(_mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(p1 + 32 * h)), _mm256_loadu_si256((const __m256i *)(q1 + 32 * h))),
                                                _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(p2 + 32 * h)), _mm256_loadu_si256((const __m256i *)(q2 + 32 * h))));
                    t = _mm256_sub_epi8(_mm256_load_si256((const __m256i *)(corr[dl] + 32 * h)), _mm256_add_epi8(t, t));     /* x[j] = +1: delta = -2 t - 4 at dl */
                    for (int g = 0; g < 2; g++) {
                        int v16 = 2 * h + g; if (v16 >= nv) break;
                        __m256i d = _mm256_and_si256(_mm256_cvtepi8_epi16(g ? _mm256_extracti128_si256(t, 1) : _mm256_castsi256_si128(t)), _mm256_load_si256((const __m256i *)(msk + 16 * v16)));
                        __m256i pn = _mm256_add_epi16(_mm256_load_si256((const __m256i *)(P + 16 * v16)), d);
                        _mm256_store_si256((__m256i *)(PN + 16 * v16), pn);
                        __m256i v = _mm256_max_epi16(_mm256_sub_epi16(_mm256_load_si256((const __m256i *)(lo + 16 * v16)), pn), _mm256_sub_epi16(pn, _mm256_load_si256((const __m256i *)(hi + 16 * v16))));
                        v = _mm256_srai_epi16(_mm256_max_epi16(v, zero), 2);
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(v, v));
                    }
                }
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e)); s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xb1));
                int dE = _mm_cvtsi128_si32(s4) - E;
                if (dE <= 0 || (dE < 64 && (RND() >> 11) * (1.0 / 9007199254740992.0) < et[dE])) {
                    SETV(j, -1); SETV(j2, 1); pl[pi] = j2; mn[mi] = j;
                    for (int v16 = 0; v16 < nv; v16++) _mm256_store_si256((__m256i *)(P + 16 * v16), _mm256_load_si256((const __m256i *)(PN + 16 * v16)));
                    E += dE;
                }
            }
            evals += (double)k;
            if (E) continue;
            /* recheck from the row before writing */
            int ok = 1, sum = 0; for (int j = 0; j < n; j++) sum += x[j];
            for (int s = 1; s <= m; s++) { int t = 0; for (int j = 0; j < n; j++) t += x[j] * x[(j + s) % n]; ok &= t >= lo3[blk][s] && t <= hi3[blk][s]; }
            if (!ok || sum != row[blk]) { fprintf(stderr, "recheck failed\n"); exit(1); }
            char line[NMAX + 2]; for (int j = 0; j < n; j++) line[j] = x[j] > 0 ? '+' : '-'; line[n] = '\n';
            #pragma omp critical
            { fwrite(line, 1, n + 1, out); hits++; if ((hits & 15) == 0) fflush(out); }
        }
    }
    fclose(out);
    printf("block %d row %d: %ld hits, %.3g evals, %.3g per hit, %.3g evals/s on %d threads\n", blk, row[blk], hits, evals, evals / hits, evals / (omp_get_wtime() - t0), omp_get_max_threads());
    return 0;
}
