/* Pool generator for the fixed-A pool join: two-stage tabu search or self-avoiding walk, AVX2 byte kernel, position-indexed.
 * Drop-in for pg2/pg4: same arguments, same output format (one row of n '+'/'-' characters per line, appended), per-thread
 * seeds derived from the seed argument as in pg2.
 * usage: pgu plan b out temp seed hits_wanted      (temp is ignored; it keeps the argument shape of pg2)
 * env: R     plus positions kept by the first stage (default 4, at most 16)
 *      MODE  1 = self-avoiding walk (default): a swap is refused if it leads to a row already visited since the last
 *                restart; no tabu list unless TLO/THI are given.   0 = tabu list only.
 *      TLO, THI  tabu tenure in steps, drawn uniformly from [TLO, THI] (default 1, 3 with MODE=0; 0, 0 with MODE=1)
 *      KR    steps per restart (default 1000; at most 60000 with MODE=1)
 *      ET    energy: 1 = sum over lags of (distance to window)/4 (default), 0 = number of lags outside their window
 *      TEST  1 = check the codes, both stages, P and the tracked energy against brute force at every step (slow)
 *
 * Move rule (that of litsurvey/sequences/proto/tabu2.c): rank the +1 positions by the energy left after flipping each
 * alone, keep the best R (ties at random), score every swap of those with a -1 position, take the best admissible swap
 * (ties at random). With a tabu list a swap is inadmissible if one of its positions is tabu, unless it reaches a new
 * best energy of this restart; the two swapped positions become tabu for a random number of steps.
 *
 * Kernel. a_k(s) = -2 x_k (x_{k+s} + x_{k-s}) in {-4, 0, 4} is the change of PAF(s) when x_k alone is flipped. Its code
 * c_k(s) = x_k (x_{k+s} + x_{k-s}) + 2 in {0, 2, 4} is recomputed for all positions at every lag from three unaligned
 * loads of the tripled row (no incremental bookkeeping). With f_s the penalty of lag s and
 * H_s[d] = f_s(P[s] + d) - f_s(P[s]), a static table LT[s][P[s]][c_p] holds, as a 16-byte shuffle table indexed by c_q,
 * the value H_s[a_p + a_q]; entry c_q + 2 holds H_s[a_p + a_q - 4], used at the lag joining p and q, where the swap
 * leaves the product x_p x_q unchanged (a static vector with 2 at positions p + s and p - s is added to the codes).
 *   stage 1: e1[k] = sum_s LT[s][P[s]][2][c_k(s)], one byte shuffle per lag over all positions;
 *   stage 2: T[p][q] = sum_s LT[s][P[s]][c_p(s)][c_q(s) + bump], one byte shuffle per (kept p, lag) over all positions.
 * Every hit is rechecked from the row before it is written. */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NMAX 96
#define M 47
#define W 96
#define RMAX 16
static int n, m, np, nm, blk, row[3], lo3[3][128], hi3[3][128], ET = 1, poff;
static int8_t F[M + 1][256];
static int8_t LT[M + 1][64][4][16] __attribute__((aligned(64)));
static uint8_t B3[M + 1][3 * NMAX + 32];
static uint8_t lagtab[NMAX][NMAX];
static uint64_t Z[NMAX];
#ifdef PROF
#include <x86intrin.h>
static uint64_t tsc[8], tprev;
#define TK(i) do { uint64_t t_ = __rdtsc(); tsc[i] += t_ - tprev; tprev = t_; } while (0)
#else
#define TK(i)
#endif
typedef struct {
    int8_t x[3 * NMAX + 64] __attribute__((aligned(32)));
    uint8_t C[M + 1][W] __attribute__((aligned(32)));
    int8_t T[RMAX][W] __attribute__((aligned(32)));
    int8_t raw[RMAX][W] __attribute__((aligned(32)));
    int8_t e1[W] __attribute__((aligned(32)));
    uint8_t notminus[W] __attribute__((aligned(32)));
    uint8_t notplus[W] __attribute__((aligned(32)));
    int8_t e1m[W] __attribute__((aligned(32)));
    uint8_t tl[W] __attribute__((aligned(32)));
    const int8_t *ltp[M + 1];
    int P[M + 1], E;
    uint64_t rs, h;
} St;
#define RND(st) ((st)->rs ^= (st)->rs << 13, (st)->rs ^= (st)->rs >> 7, (st)->rs ^= (st)->rs << 17, (st)->rs)
static inline void setx(St *st, int k, int v) { st->x[k] = st->x[k + n] = st->x[k + 2 * n] = (int8_t)v; }
static int energy_of(const int *P) { int e = 0; for (int s = 1; s <= m; s++) e += F[s][P[s] + 128]; return e; }
static void paf_scratch(const int8_t *x, int *P) { for (int s = 1; s <= m; s++) { int t = 0; for (int j = 0; j < n; j++) t += x[j] * x[(j + s) % n]; P[s] = t; } }
static void init_random(St *st) {
    int idx[NMAX];
    for (int j = 0; j < n; j++) idx[j] = j;
    for (int j = n - 1; j > 0; j--) { int k = (int)(RND(st) % (uint64_t)(j + 1)), t = idx[j]; idx[j] = idx[k]; idx[k] = t; }
    st->h = 0; memset(st->notminus, 0xff, W); memset(st->notplus, 0xff, W); memset(st->tl, 0, W);
    for (int j = 0; j < n; j++) {
        if (j < np) { setx(st, idx[j], 1); st->notplus[idx[j]] = 0; st->h ^= Z[idx[j]]; }
        else { setx(st, idx[j], -1); st->notminus[idx[j]] = 0; }
    }
    paf_scratch(st->x, st->P); st->E = energy_of(st->P);
    for (int s = 1; s <= m; s++) st->ltp[s] = LT[s][(st->P[s] + 128) >> 2][0];
}
/* codes of every position at every lag, and the stage-1 score of every position */
static inline void stage1(St *st) {
    const __m256i two = _mm256_set1_epi8(2);
    const int8_t *x = st->x + n;
    const __m256i x0 = _mm256_loadu_si256((const __m256i *)x), x1 = _mm256_loadu_si256((const __m256i *)(x + 32)), x2 = _mm256_loadu_si256((const __m256i *)(x + 64));
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0;
    for (int s = 1; s <= m; s++) {
        const int8_t *xp = x + s, *xm = x - s;
        __m256i L = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(st->ltp[s] + 16)));
        __m256i c0 = _mm256_add_epi8(_mm256_sign_epi8(_mm256_add_epi8(_mm256_loadu_si256((const __m256i *)xp), _mm256_loadu_si256((const __m256i *)xm)), x0), two);
        __m256i c1 = _mm256_add_epi8(_mm256_sign_epi8(_mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(xp + 32)), _mm256_loadu_si256((const __m256i *)(xm + 32))), x1), two);
        __m256i c2 = _mm256_add_epi8(_mm256_sign_epi8(_mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(xp + 64)), _mm256_loadu_si256((const __m256i *)(xm + 64))), x2), two);
        _mm256_store_si256((__m256i *)st->C[s], c0); _mm256_store_si256((__m256i *)(st->C[s] + 32), c1); _mm256_store_si256((__m256i *)(st->C[s] + 64), c2);
        a0 = _mm256_add_epi8(a0, _mm256_shuffle_epi8(L, c0)); a1 = _mm256_add_epi8(a1, _mm256_shuffle_epi8(L, c1)); a2 = _mm256_add_epi8(a2, _mm256_shuffle_epi8(L, c2));
    }
    _mm256_store_si256((__m256i *)st->e1, a0); _mm256_store_si256((__m256i *)(st->e1 + 32), a1); _mm256_store_si256((__m256i *)(st->e1 + 64), a2);
}
/* stage 2: the swap scores of K kept plus positions against every position, into st->raw[r0 ..] */
#define BUILD(NAME, K) \
static inline __attribute__((always_inline)) void NAME(St *st, const int *ps, int r0) { \
    __m256i acc[K][3]; int bo[K]; \
    for (int r = 0; r < K; r++) { acc[r][0] = acc[r][1] = acc[r][2] = _mm256_setzero_si256(); bo[r] = n - ps[r]; } \
    for (int s = 1; s <= m; s++) { \
        const __m256i c0 = _mm256_load_si256((const __m256i *)st->C[s]), c1 = _mm256_load_si256((const __m256i *)(st->C[s] + 32)), c2 = _mm256_load_si256((const __m256i *)(st->C[s] + 64)); \
        for (int r = 0; r < K; r++) { \
            const uint8_t *b = B3[s] + bo[r]; \
            __m256i L = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(st->ltp[s] + 8 * st->C[s][ps[r]]))); \
            acc[r][0] = _mm256_add_epi8(acc[r][0], _mm256_shuffle_epi8(L, _mm256_add_epi8(c0, _mm256_loadu_si256((const __m256i *)b)))); \
            acc[r][1] = _mm256_add_epi8(acc[r][1], _mm256_shuffle_epi8(L, _mm256_add_epi8(c1, _mm256_loadu_si256((const __m256i *)(b + 32))))); \
            acc[r][2] = _mm256_add_epi8(acc[r][2], _mm256_shuffle_epi8(L, _mm256_add_epi8(c2, _mm256_loadu_si256((const __m256i *)(b + 64))))); \
        } \
    } \
    for (int r = 0; r < K; r++) for (int h = 0; h < 3; h++) _mm256_store_si256((__m256i *)(st->raw[r0 + r] + 32 * h), acc[r][h]); \
}
BUILD(build1, 1)
BUILD(build2, 2)
BUILD(build4, 4)
static void test_step(St *st, const int *cand, int nc) {
    int P2[M + 1], Pc[M + 1]; paf_scratch(st->x, Pc);
    for (int s = 1; s <= m; s++) if (Pc[s] != st->P[s]) { fprintf(stderr, "TEST: P mismatch at lag %d\n", s); exit(1); }
    if (energy_of(Pc) != st->E) { fprintf(stderr, "TEST: E mismatch\n"); exit(1); }
    for (int k = 0; k < n; k++) {
        if ((st->x[k] < 0) != (st->notminus[k] == 0)) { fprintf(stderr, "TEST: mask mismatch\n"); exit(1); }
        if (st->x[k] < 0) continue;
        int8_t y[NMAX]; memcpy(y, st->x, n); y[k] = -1; paf_scratch(y, P2);
        if (energy_of(P2) - st->E != st->e1[k]) { fprintf(stderr, "TEST: stage-1 mismatch at %d\n", k); exit(1); }
    }
    for (int r = 0; r < nc; r++) for (int q = 0; q < n; q++) {
        if (st->x[q] > 0) continue;
        int8_t y[NMAX]; memcpy(y, st->x, n); y[cand[r]] = -1; y[q] = 1; paf_scratch(y, P2);
        if (energy_of(P2) - st->E != st->raw[r][q]) { fprintf(stderr, "TEST: table mismatch at (%d,%d): %d against %d\n", cand[r], q, st->raw[r][q], energy_of(P2) - st->E); exit(1); }
    }
}
int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: pgu plan b out temp seed hits_wanted\n"); return 2; }
    FILE *f = fopen(argv[1], "r"); blk = atoi(argv[2]); uint64_t seed = strtoull(argv[5], 0, 10); long want = atol(argv[6]);
    char abuf[256];
    if (!f || fscanf(f, "%d %255s", &n, abuf) != 2) return 2; m = (n - 1) / 2; if (n > NMAX - 7 || n < 7 || m > M || blk < 0 || blk > 2) return 2;
    for (int b = 0; b < 3; b++) { if (fscanf(f, "%d", &row[b]) != 1) return 2; for (int s = 1; s <= m; s++) if (fscanf(f, "%d %d", &lo3[b][s], &hi3[b][s]) != 2) return 2; }
    fclose(f);
    FILE *out = fopen(argv[3], "a"); if (!out) return 2;
    #define ENVI(name, def) (getenv(name) ? atoi(getenv(name)) : (def))
    int MODE = ENVI("MODE", 1), R = ENVI("R", 4), TEST = ENVI("TEST", 0); ET = ENVI("ET", 1);
    int TLO = ENVI("TLO", MODE == 0 ? 1 : 0), THI = ENVI("THI", MODE == 0 ? 3 : TLO); long KR = getenv("KR") ? atol(getenv("KR")) : 1000;
    if (ET < 0 || ET > 1 || R < 1 || R > RMAX || TLO < 0 || THI < TLO || THI > 120 || KR < 1 || MODE < 0 || MODE > 1 || (MODE == 1 && KR > 60000)) return 2;
    np = (n + row[blk]) / 2; nm = n - np; if (nm < 1 || np < 1 || ((n + row[blk]) & 1)) return 2;
    if (R > np) R = np;
    poff = n & 3;
    for (int s = 1; s <= m; s++) for (int v = -128; v < 128; v++) {
        int d = v < lo3[blk][s] ? lo3[blk][s] - v : v > hi3[blk][s] ? v - hi3[blk][s] : 0; d = (d + 3) / 4;
        F[s][v + 128] = (int8_t)(ET == 1 ? d : d > 0);
    }
    for (int s = 1; s <= m; s++) for (int pv = 0; pv < 64; pv++) {
        int Pv = 4 * pv - 128 + poff; if (Pv - 12 < -128 || Pv + 8 > 127) continue;
        for (int c = 0; c < 3; c++) for (int i = 0; i < 4; i++) {            /* a_p = 4 - 4c; code 2i: a_q = 4 - 4i */
            int d = F[s][Pv + (4 - 4 * c) + (4 - 4 * i) + 128] - F[s][Pv + 128];
            LT[s][pv][c][2 * i] = (int8_t)d;
        }
    }
    for (int s = 1; s <= m; s++) for (int k = 0; k < 3 * NMAX + 32; k++) B3[s][k] = (k % n == s || k % n == n - s) ? 2 : 0;
    for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) { int d = a > b ? a - b : b - a; if (d > m) d = n - d; lagtab[a][b] = (uint8_t)d; }
    { uint64_t z = 0x243F6A8885A308D3ull; for (int k = 0; k < n; k++) { z ^= z << 13; z ^= z >> 7; z ^= z << 17; Z[k] = z; } }
    int HB = 12; while (((long)1 << HB) < 8 * KR) HB++;
    long hits = 0; double steps = 0, walks = 0, cpu = 0, t0 = omp_get_wtime();
    #pragma omp parallel reduction(+:steps, walks, cpu)
    {
        St *st = aligned_alloc(64, (sizeof(St) + 63) / 64 * 64); memset(st, 0, sizeof *st);
        st->rs = (seed + 1000003ull * omp_get_thread_num()) * 0x9E3779B97F4A7C15ull + 1;
        uint64_t *hs = MODE == 1 ? calloc((size_t)1 << HB, sizeof(uint64_t)) : 0; int *hlist = MODE == 1 ? malloc(sizeof(int) * (size_t)(KR + 2)) : 0, hn = 0;
        const uint64_t HM = ((uint64_t)1 << HB) - 1;
        const __m256i zero = _mm256_setzero_si256(), v127 = _mm256_set1_epi8(127), onev = _mm256_set1_epi8(1), flip = _mm256_set1_epi8((char)0x80);
        struct timespec c0, c1; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0);
        while (1) {
            long hnow;
            #pragma omp atomic read
            hnow = hits;
            if (hnow >= want) break;
            init_random(st); walks += 1;
            if (MODE == 1) { for (int k = 0; k < hn; k++) hs[hlist[k]] = 0; hn = 0; uint64_t k = st->h >> (64 - HB); hs[k] = st->h | 1; hlist[hn++] = (int)k; }
            int Ebest = st->E; long it;
            for (it = 1; it <= KR && st->E > 0; it++) {
                TK(0); stage1(st); TK(1);
                int cand[RMAX], nc = R;
                {   /* the R plus positions of least stage-1 score, ties at random */
                    __m256i k0 = _mm256_blendv_epi8(_mm256_load_si256((const __m256i *)st->e1), v127, _mm256_load_si256((const __m256i *)st->notplus));
                    __m256i k1 = _mm256_blendv_epi8(_mm256_load_si256((const __m256i *)(st->e1 + 32)), v127, _mm256_load_si256((const __m256i *)(st->notplus + 32)));
                    __m256i k2 = _mm256_blendv_epi8(_mm256_load_si256((const __m256i *)(st->e1 + 64)), v127, _mm256_load_si256((const __m256i *)(st->notplus + 64)));
                    _mm256_store_si256((__m256i *)st->e1m, k0); _mm256_store_si256((__m256i *)(st->e1m + 32), k1); _mm256_store_si256((__m256i *)(st->e1m + 64), k2);
                    uint64_t rr = RND(st); int r = 0;
                    while (r < nc) {
                        __m256i mv = _mm256_min_epi8(_mm256_min_epi8(k0, k1), k2);
                        __m128i u = _mm_xor_si128(_mm_min_epi8(_mm256_castsi256_si128(mv), _mm256_extracti128_si256(mv, 1)), _mm256_castsi256_si128(flip));
                        u = _mm_min_epu8(u, _mm_srli_epi16(u, 8));
                        const __m256i bvv = _mm256_set1_epi8((char)((_mm_extract_epi16(_mm_minpos_epu16(u), 0) & 0xff) ^ 0x80));
                        uint32_t mk[3] = { (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(k0, bvv)), (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(k1, bvv)), (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(k2, bvv)) };
                        int cnt = __builtin_popcount(mk[0]) + __builtin_popcount(mk[1]) + __builtin_popcount(mk[2]);
                        if (cnt <= nc - r) {      /* all of this level are kept */
                            for (int h = 0; h < 3; h++) for (uint32_t w = mk[h]; w; w &= w - 1) cand[r++] = 32 * h + __builtin_ctz(w);
                        } else {                   /* a random subset of this level fills the rest */
                            int pos[W], c = 0;
                            for (int h = 0; h < 3; h++) for (uint32_t w = mk[h]; w; w &= w - 1) pos[c++] = 32 * h + __builtin_ctz(w);
                            while (r < nc) { rr = rr * 6364136223846793005ull + 1442695040888963407ull; int k = (int)(((rr >> 32) * (uint64_t)c) >> 32); cand[r++] = pos[k]; pos[k] = pos[--c]; }
                            break;
                        }
                        k0 = _mm256_blendv_epi8(k0, v127, _mm256_cmpeq_epi8(k0, bvv)); k1 = _mm256_blendv_epi8(k1, v127, _mm256_cmpeq_epi8(k1, bvv)); k2 = _mm256_blendv_epi8(k2, v127, _mm256_cmpeq_epi8(k2, bvv));
                    }
                }
                TK(2);
                { int r = 0; for (; r + 4 <= nc; r += 4) build4(st, cand + r, r); for (; r + 2 <= nc; r += 2) build2(st, cand + r, r); if (r < nc) build1(st, cand + r, r); }
                TK(3);
                if (TEST) test_step(st, cand, nc);
                /* admissible scores */
                int thr = Ebest - st->E; if (thr < -127) thr = -127;
                const __m256i thrv = _mm256_set1_epi8((char)thr);
                __m256i mnv = v127;
                for (int h = 0; h < 3; h++) {
                    const __m256i nmk = _mm256_load_si256((const __m256i *)(st->notminus + 32 * h));
                    const __m256i tq = _mm256_cmpgt_epi8(_mm256_load_si256((const __m256i *)(st->tl + 32 * h)), zero);
                    for (int r = 0; r < nc; r++) {
                        __m256i v = _mm256_load_si256((const __m256i *)(st->raw[r] + 32 * h));
                        __m256i tb = st->tl[cand[r]] ? _mm256_set1_epi8(-1) : tq;
                        __m256i ex = _mm256_or_si256(nmk, _mm256_andnot_si256(_mm256_cmpgt_epi8(thrv, v), tb));
                        v = _mm256_blendv_epi8(v, v127, ex);
                        _mm256_store_si256((__m256i *)(st->T[r] + 32 * h), v);
                        mnv = _mm256_min_epi8(mnv, v);
                    }
                }
                int bv, br = -1, bq = -1;
                while (1) {
                    __m128i u = _mm_xor_si128(_mm_min_epi8(_mm256_castsi256_si128(mnv), _mm256_extracti128_si256(mnv, 1)), _mm256_castsi256_si128(flip));
                    u = _mm_min_epu8(u, _mm_srli_epi16(u, 8));
                    bv = (int)(int8_t)((_mm_extract_epi16(_mm_minpos_epu16(u), 0) & 0xff) ^ 0x80);
                    if (bv == 127) { br = -1; break; }
                    const __m256i bvv = _mm256_set1_epi8((char)bv);
                    uint32_t mk[RMAX][3]; int cnt = 0;
                    for (int r = 0; r < nc; r++) for (int h = 0; h < 3; h++) { mk[r][h] = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((const __m256i *)(st->T[r] + 32 * h)), bvv)); cnt += __builtin_popcount(mk[r][h]); }
                    int pick = cnt > 1 ? (int)(((RND(st) >> 32) * (uint64_t)cnt) >> 32) : 0;
                    for (int r = 0; r < nc && br < 0; r++) for (int h = 0; h < 3; h++) { int c = __builtin_popcount(mk[r][h]);
                        if (pick < c) { uint32_t w = mk[r][h]; while (pick--) w &= w - 1; br = r; bq = 32 * h + __builtin_ctz(w); break; }
                        pick -= c; }
                    if (MODE == 0) break;
                    uint64_t h2 = (st->h ^ Z[cand[br]] ^ Z[bq]) | 1, k = h2 >> (64 - HB); int seen = 0;
                    while (hs[k]) { if (hs[k] == h2) { seen = 1; break; } k = (k + 1) & HM; }
                    if (!seen) { hs[k] = h2; hlist[hn++] = (int)k; break; }
                    st->T[br][bq] = 127; br = -1;
                    mnv = v127;
                    for (int r = 0; r < nc; r++) for (int h = 0; h < 3; h++) mnv = _mm256_min_epi8(mnv, _mm256_load_si256((const __m256i *)(st->T[r] + 32 * h)));
                }
                TK(4);
                if (THI > 0) for (int h = 0; h < 3; h++) _mm256_store_si256((__m256i *)(st->tl + 32 * h), _mm256_subs_epu8(_mm256_load_si256((const __m256i *)(st->tl + 32 * h)), onev));
                if (br < 0) continue;      /* no admissible swap: the step is spent, new ties next step */
                int p = cand[br], q = bq, sj = lagtab[p][q];
                for (int s = 1; s <= m; s++) st->P[s] += 8 - 2 * (st->C[s][p] + st->C[s][q]);
                st->P[sj] -= 4;
                for (int s = 1; s <= m; s++) st->ltp[s] = LT[s][(st->P[s] + 128) >> 2][0];
                setx(st, p, -1); setx(st, q, 1); st->notminus[p] = 0; st->notminus[q] = 0xff; st->notplus[p] = 0xff; st->notplus[q] = 0; st->h ^= Z[p] ^ Z[q];
                if (THI > 0) { int ten = TLO + (int)(RND(st) % (uint64_t)(THI - TLO + 1)); st->tl[p] = st->tl[q] = (uint8_t)ten; }
                st->E += bv;
                if (st->E < Ebest) Ebest = st->E;
                TK(5);
            }
            steps += (double)(it - 1);
            if (st->E) continue;
            int ok = 1, sum = 0, Pc[M + 1]; for (int j = 0; j < n; j++) sum += st->x[j];
            paf_scratch(st->x, Pc);
            for (int s = 1; s <= m; s++) ok &= Pc[s] >= lo3[blk][s] && Pc[s] <= hi3[blk][s];
            if (!ok || sum != row[blk]) { fprintf(stderr, "recheck failed\n"); exit(1); }
            char line[NMAX + 2]; for (int j = 0; j < n; j++) line[j] = st->x[j] > 0 ? '+' : '-'; line[n] = '\n';
            #pragma omp critical
            { if (hits < want) { fwrite(line, 1, n + 1, out); hits++; if ((hits & 15) == 0) fflush(out); } }
        }
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1); cpu += (double)(c1.tv_sec - c0.tv_sec) + 1e-9 * (double)(c1.tv_nsec - c0.tv_nsec);
        free(st); free(hs); free(hlist);
    }
    fclose(out);
#ifdef PROF
    { double tot = 0; for (int i = 0; i < 8; i++) tot += (double)tsc[i]; fprintf(stderr, "cycles/step: other %.0f stage1 %.0f rank %.0f stage2 %.0f pick %.0f apply %.0f total %.0f\n", tsc[0] / steps, tsc[1] / steps, tsc[2] / steps, tsc[3] / steps, tsc[4] / steps, tsc[5] / steps, tot / steps); }
#endif
    double ev = steps * (n + np + (double)R * nm);      /* the evaluation count of tabu2.c, for comparison */
    printf("block %d row %d: %ld hits, %.3g evals, %.3g per hit, %.3g evals/s on %d threads; R=%d T=%d..%d KR=%ld MODE=%d ET=%d, %.4g steps/hit, %.4g restarts/hit, %.4g steps/thread-s, %.5g thread-s/hit\n",
           blk, row[blk], hits, ev, ev / hits, ev / (omp_get_wtime() - t0), omp_get_max_threads(), R, TLO, THI, KR, MODE, ET, steps / hits, walks / hits, steps / cpu, cpu / hits);
    return 0;
}
