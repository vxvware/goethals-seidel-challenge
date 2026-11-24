/* Pool generator for the fixed-A pool join, AVX2 on bytes (pg2.c widened to 16 bits; same arguments, same output format).
 * env EP=1: energy = sum of distances / 4 instead of the sum of squares. env COOL=T0:T1: geometric cooling from T0 to T1 in 16 stages over each restart (the temp argument is then ignored).
 * usage: pg2 plan b out temp seed hits_wanted      env KR = restart length (default 300000)
 * Annealing with swap moves at fixed row sum; energy = sum over lags of (distance to window / 4)^2.
 * A swap of positions j (sign c) and j2 (sign -c) changes PAF(s) by -2c (x[j+s] + x[j-s] - x[j2+s] - x[j2-s]),
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
    int nv = (m + 31) / 32, EP = getenv("EP") ? atoi(getenv("EP")) : 2; double T0 = temp, T1 = temp; int cool = getenv("COOL") && sscanf(getenv("COOL"), "%lf:%lf", &T0, &T1) == 2;
    #pragma omp parallel reduction(+:evals)
    {
        uint64_t rs = (seed + 1000003ull * omp_get_thread_num()) * 0x9E3779B97F4A7C15ull + 1;
        #define RND() (rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17, rs)
        int8_t x[3 * NMAX + M] = {0}, xr[3 * NMAX + M] = {0};      /* xr[k] = x[-k]; three copies each */
        int8_t P[M] __attribute__((aligned(32))) = {0}, PN[M] __attribute__((aligned(32))), lo[M] __attribute__((aligned(32))), hi[M] __attribute__((aligned(32)));
        for (int s = 0; s < M; s++) { lo[s] = -128; hi[s] = 127; }      /* unused lags: saturating distances are never positive */
        for (int s = 1; s <= m; s++) { lo[s - 1] = (int8_t)lo3[blk][s]; hi[s - 1] = (int8_t)hi3[blk][s]; }
        double et[16][64]; for (int g = 0; g < 16; g++) for (int i = 0; i < 64; i++) et[g][i] = exp(-i / (cool ? T0 * pow(T1 / T0, g / 15.0) : temp));
        const __m256i ones = _mm256_set1_epi16(1);
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
            for (int s = 1; s <= m; s++) { int t = 0; for (int j = 0; j < n; j++) t += x[j] * x[j + s]; P[s - 1] = (int8_t)t; int v = t < lo[s - 1] ? lo[s - 1] - t : t > hi[s - 1] ? t - hi[s - 1] : 0; E += EP == 1 ? v / 4 : (v / 4) * (v / 4); }
            long stage = KR / 16 + 1;
            long k;
            for (k = 0; k < KR && E > 0; k++) {
                uint64_t r = RND();
                int pi = (int)(((r >> 32) * (uint64_t)np) >> 32), mi = (int)(((r & 0xffffffffu) * (uint64_t)nm) >> 32);
                int j = pl[pi], j2 = mn[mi];
                const int8_t *p1 = x + j + n + 1, *q1 = xr + ((n - j) % n) + n + 1, *p2 = x + j2 + n + 1, *q2 = xr + ((n - j2) % n) + n + 1;
                int dl = j2 > j ? j2 - j : j - j2; if (dl > m) dl = n - dl;
                __m256i acc = zero;
                for (int h = 0; h < nv; h++) {
                    __m256i t = _mm256_sub_epi8(_mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(p1 + 32 * h)), _mm256_loadu_si256((const __m256i *)(q1 + 32 * h))),
                                                _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(p2 + 32 * h)), _mm256_loadu_si256((const __m256i *)(q2 + 32 * h))));
                    t = _mm256_sub_epi8(_mm256_load_si256((const __m256i *)(corr[dl] + 32 * h)), _mm256_add_epi8(t, t));     /* x[j] = +1: delta = -2 t - 4 at dl */
                    __m256i pn = _mm256_add_epi8(_mm256_load_si256((const __m256i *)(P + 32 * h)), t);
                    _mm256_store_si256((__m256i *)(PN + 32 * h), pn);
                    __m256i v = _mm256_max_epi8(_mm256_subs_epi8(_mm256_load_si256((const __m256i *)(lo + 32 * h)), pn), _mm256_subs_epi8(pn, _mm256_load_si256((const __m256i *)(hi + 32 * h))));
                    v = _mm256_max_epi8(v, zero);
                    acc = _mm256_add_epi32(acc, EP == 1 ? _mm256_sad_epu8(v, zero) : _mm256_madd_epi16(_mm256_maddubs_epi16(v, v), ones));
                }
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e)); s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xb1));
                int En = EP == 1 ? _mm_cvtsi128_si32(s4) >> 2 : _mm_cvtsi128_si32(s4) >> 4;
                int dE = En - E;
                if (dE <= 0 || (dE < 64 && (RND() >> 11) * (1.0 / 9007199254740992.0) < et[cool ? k / stage : 0][dE])) {
                    SETV(j, -1); SETV(j2, 1); pl[pi] = j2; mn[mi] = j;
                    for (int h = 0; h < nv; h++) _mm256_store_si256((__m256i *)(P + 32 * h), _mm256_load_si256((const __m256i *)(PN + 32 * h)));
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
