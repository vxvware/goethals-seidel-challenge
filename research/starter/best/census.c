/*
 * Exact census of circulant best matrix quadruples.
 *
 * C port of census.py with the same constraint set and the same answers,
 * built for speed: AVX2 over the spectral checks, OpenMP over the symmetric
 * sequences, int8 autocorrelations and int16 scaled spectra, and an open
 * addressed hash table instead of binary search.
 *
 * Constraints, in the order they fire:
 *   row sum      sigma_d^2 = 4n-3, positive sign only (negation is a symmetry)
 *   ceiling      max_k PSD_d(k) <= 4n-3, from PSD >= 1 for each skew sequence
 *   decimation   only d minimal in its orbit under j -> mj
 *   single       PSD_a(k) <= 4n-2-PSD_d(k) for every k, given d
 *   mask         a partner sharing a frequency where both exceed half the
 *                cap provably fails, so one AND rejects it
 *   pair         PSD_a(k)+PSD_b(k) <= 4n-1-PSD_d(k) for every k
 *   range        the implied autocorrelation of c must lie in [-(n-2), n-4]
 *   ordering     a <= b <= c absorbs the S_3 symmetry
 *
 * Spectra are stored as floor(PSD * 16). Flooring is safe in the only
 * direction that matters: floor(x)+floor(y) <= floor(x+y), so a pair that
 * truly satisfies a cap is never rejected.
 *
 * Build: gcc -O3 -march=native -fopenmp -o census census.c -lm
 * Run:   ./census 43 [threads] [out.txt]
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define SCALE 16

static int N, H, HP;
static long long SCANNED = 0;
#pragma omp threadprivate(SCANNED)              /* order, (n-1)/2, H padded to 16 lanes */
static size_t NSKEW;              /* 1 << H */
static int8_t *PAFT;              /* [NSKEW][H]  autocorrelations */
static int16_t *PSDT;             /* [NSKEW][HP] scaled spectra, padded */
static double *COS;               /* [H][H] cos(2 pi s k / n) */

static double wall(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static void build_tables(void) {
    PAFT = aligned_alloc(64, NSKEW * (size_t)H);
    PSDT = aligned_alloc(64, NSKEW * (size_t)HP * sizeof(int16_t));
    COS = malloc((size_t)H * H * sizeof(double));
    if (!PAFT || !PSDT || !COS) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int k = 1; k <= H; k++)
        for (int s = 1; s <= H; s++)
            COS[(size_t)(k - 1) * H + (s - 1)] = cos(2.0 * M_PI * s * k / N);
#pragma omp parallel
    {
        int8_t *a = malloc(N);
#pragma omp for schedule(static)
        for (long long code = 0; code < (long long)NSKEW; code++) {
            a[0] = 1;
            for (int j = 1; j <= H; j++) {
                int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
                a[j] = v;
                a[N - j] = -v;
            }
            int8_t *paf = PAFT + (size_t)code * H;
            for (int s = 1; s <= H; s++) {
                int acc = 0;
                for (int j = 0; j < N; j++) {
                    int t = j + s;
                    if (t >= N) t -= N;
                    acc += a[j] * a[t];
                }
                paf[s - 1] = (int8_t)acc;
            }
            int16_t *psd = PSDT + (size_t)code * HP;
            for (int k = 0; k < H; k++) {
                double acc = N;
                const double *row = COS + (size_t)k * H;
                for (int s = 0; s < H; s++) acc += 2.0 * paf[s] * row[s];
                double v = floor((acc - 1e-7) * SCALE);
                if (v > 32767) v = 32767;
                if (v < -32768) v = -32768;
                psd[k] = (int16_t)v;
            }
            for (int k = H; k < HP; k++) psd[k] = -32768;   /* padding */
        }
        free(a);
    }
}

/* ---------- symmetric sequences: row sum, ceiling, decimation ---------- */

static int popcount_ll(unsigned long long x) { return __builtin_popcountll(x); }

static void sym_seq(long long code, int8_t *d) {
    d[0] = (code & 1) ? 1 : -1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> j) & 1 ? 1 : -1;
        d[j] = v;
        d[N - j] = v;
    }
}

static void sym_psd(const int8_t *d, double *out) {
    for (int k = 1; k <= H; k++) {
        double re = 0, im = 0;
        for (int j = 0; j < N; j++) {
            double ang = 2.0 * M_PI * j * k / N;
            re += d[j] * cos(ang);
            im += d[j] * sin(ang);
        }
        out[k - 1] = re * re + im * im;
    }
}

static long long *canonical_ds(int *count, int *n_admissible) {
    int root = (int)(sqrt((double)(4 * N - 3)) + 0.5);
    int units[256], nu = 0;
    for (int m = 1; m < N; m++) {
        int a = m, b = N;
        while (b) { int t = a % b; a = b; b = t; }
        if (a == 1) units[nu++] = m;
    }
    /* bit j of a code is position j; decimation maps j -> min(mj, n-mj) */
    int *perm = malloc((size_t)nu * (H + 1) * sizeof(int));
    for (int u = 0; u < nu; u++)
        for (int j = 1; j <= H; j++) {
            int t = (units[u] * j) % N;
            if (t > N / 2) t = N - t;
            perm[(size_t)u * (H + 1) + j] = t;
        }
    long long cap = 1LL << (H + 1);
    long long *keep = malloc((size_t)(1 << 20) * sizeof(long long));
    int nk = 0, nadm = 0, kept_cap = 1 << 20;
    int8_t *d = malloc(N);
    double *psd = malloc((size_t)H * sizeof(double));
    for (long long code = 0; code < cap; code++) {
        int d0 = (code & 1) ? 1 : -1;
        int s = d0 + 2 * (2 * popcount_ll((unsigned long long)(code >> 1)) - H);
        if (s != root) continue;
        nadm++;
        sym_seq(code, d);
        sym_psd(d, psd);
        double peak = 0;
        for (int k = 0; k < H; k++) if (psd[k] > peak) peak = psd[k];
        if (peak > 4.0 * N - 3.0 + 1e-6) continue;          /* ceiling */
        long long best = code;                              /* decimation */
        for (int u = 0; u < nu; u++) {
            long long c2 = code & 1;
            const int *p = perm + (size_t)u * (H + 1);
            for (int j = 1; j <= H; j++)
                if ((code >> p[j]) & 1) c2 |= 1LL << j;
            if (c2 < best) best = c2;
        }
        if (best != code) continue;
        if (nk == kept_cap) {
            kept_cap *= 2;
            keep = realloc(keep, (size_t)kept_cap * sizeof(long long));
        }
        keep[nk++] = code;
    }
    free(perm); free(d); free(psd);
    *count = nk;
    *n_admissible = nadm;
    return keep;
}

/* ---------------------------- the per-d search ---------------------------- */

typedef struct { uint64_t key; int32_t idx; } Slot;

#define BLK 64

typedef struct { const uint64_t *w0, *w1; } SortCtx;
static int cmp_words(const void *A, const void *B, void *arg) {
    const SortCtx *c = arg;
    int a = *(const int32_t *)A, b = *(const int32_t *)B;
    if (c->w0[a] != c->w0[b]) return c->w0[a] < c->w0[b] ? -1 : 1;
    if (c->w1[a] != c->w1[b]) return c->w1[a] < c->w1[b] ? -1 : 1;
    return 0;
}

static inline uint64_t hash_bytes(const int8_t *v, int len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)(uint8_t)(v[i] + 128);
        h *= 0x100000001b3ULL;
    }
    return h ? h : 1;
}

typedef struct {
    int32_t *idx;        /* codes of the filtered list, sorted */
    int16_t *psd;        /* [m][HP] permuted so the tightest cap is lane 0 */
    int8_t *paf;         /* [m][H] unpermuted */
    uint64_t *mask, *mask2, *bmin, *bmin2;
    Slot *tab;
    int tabmask;
    int cap_m;
    int32_t *ord, *idx2, *hist;
    int32_t bstart[4098];
    uint64_t *mask2_tmp, *mask3_tmp;
    int16_t *psd2;
    int8_t *paf2;
    int cap_sort;
} Scratch;

static void scratch_reserve(Scratch *s, int m) {
    if (m <= s->cap_m) return;
    s->cap_m = m * 2;
    s->idx = realloc(s->idx, (size_t)s->cap_m * sizeof(int32_t));
    s->psd = realloc(s->psd, (size_t)s->cap_m * HP * sizeof(int16_t));
    s->paf = realloc(s->paf, (size_t)s->cap_m * H);
    s->mask = realloc(s->mask, (size_t)s->cap_m * sizeof(uint64_t));
    s->mask2 = realloc(s->mask2, (size_t)s->cap_m * sizeof(uint64_t));
    s->mask2_tmp = realloc(s->mask2_tmp, (size_t)s->cap_m * sizeof(uint64_t));
    s->mask3_tmp = realloc(s->mask3_tmp, (size_t)s->cap_m * sizeof(uint64_t));
    s->bmin = realloc(s->bmin, (size_t)(s->cap_m / BLK + 4) * sizeof(uint64_t));
    s->bmin2 = realloc(s->bmin2, (size_t)(s->cap_m / BLK + 4) * sizeof(uint64_t));
}

/* one candidate pair: full spectral check, range check, hash lookup */
static inline void try_pair(Scratch *sc, int ia, int ib, const int8_t *aa,
                            const int8_t *target, const int32_t *cap2p,
                            const int16_t *cap16, int lo_c, int hi_c,
                            int32_t *out, int *nout, int max_out,
                            long long *found) {
    const int16_t *pa = sc->psd + (size_t)ia * HP;
    const int16_t *pb = sc->psd + (size_t)ib * HP;
#ifdef __AVX2__
    __m256i S0 = _mm256_adds_epi16(_mm256_loadu_si256((const __m256i *)pa),
                                   _mm256_loadu_si256((const __m256i *)pb));
    __m256i S1 = _mm256_adds_epi16(_mm256_loadu_si256((const __m256i *)(pa + 16)),
                                   _mm256_loadu_si256((const __m256i *)(pb + 16)));
    __m256i g = _mm256_or_si256(
        _mm256_cmpgt_epi16(S0, _mm256_loadu_si256((const __m256i *)cap16)),
        _mm256_cmpgt_epi16(S1, _mm256_loadu_si256((const __m256i *)(cap16 + 16))));
    if (!_mm256_testz_si256(g, g)) return;
#else
    for (int k = 0; k < H; k++)
        if ((int32_t)pa[k] + (int32_t)pb[k] > cap2p[k]) return;
#endif
    const int8_t *bb = sc->paf + (size_t)ib * H;
    int8_t want[64];
    for (int k = 0; k < H; k++) {
        int v = (int)target[k] - aa[k] - bb[k];
        if (v < lo_c || v > hi_c) return;
        want[k] = (int8_t)v;
    }
    uint64_t h = hash_bytes(want, H);
    int p = (int)(h & sc->tabmask);
    while (sc->tab[p].key) {
        if (sc->tab[p].key == h) {
            int ic = sc->tab[p].idx;
            if (ic >= ib && memcmp(sc->paf + (size_t)ic * H, want, H) == 0) {
                if (*nout >= max_out) { fprintf(stderr, "local solution buffer exhausted\n"); exit(2); }
                if (*nout < max_out) {
                    int32_t *w = out + (size_t)(*nout) * 3;
                    w[0] = sc->idx[ia]; w[1] = sc->idx[ib]; w[2] = sc->idx[ic];
                }
                (*nout)++;
                (*found)++;
            }
        }
        p = (p + 1) & sc->tabmask;
    }
}

static long long solve_d(long long code, Scratch *sc, int32_t *out, int *nout,
                         int max_out, long long *tested_out, double *phase) {
    double _t0 = wall(), _t1;
    int8_t d[512];
    double psd_d[64];
    sym_seq(code, d);
    sym_psd(d, psd_d);
    int32_t cap1[64], cap2[64], order[64];
    for (int k = 0; k < H; k++) {
        cap1[k] = (int32_t)floor((4.0 * N - 2.0 - psd_d[k] + 1e-7) * SCALE);
        cap2[k] = (int32_t)floor((4.0 * N - 1.0 - psd_d[k] + 1e-7) * SCALE);
        order[k] = k;
    }
    for (int i = 1; i < H; i++) {                    /* tightest cap first */
        int32_t key = order[i]; int j = i - 1;
        while (j >= 0 && cap2[order[j]] > cap2[key]) { order[j + 1] = order[j]; j--; }
        order[j + 1] = key;
    }
    int32_t cap1p[64], cap2p[64];
    int16_t cap16[64];

    /* The single filter uses PSD_b, PSD_c >= 1. But b and c are drawn from
       the filtered list, so their spectra are at least the list's own minimum
       at each frequency, which is usually well above 1. Tightening the caps
       with that minimum shrinks the list, which lowers the minimum again, so
       iterate to a fixed point. Each round is a pass over a list that only
       shrinks, and the pair loop cost falls with the square of its length. */
    int16_t cap1nat[64], cap2nat[64];
    int32_t base[64];
    for (int k = 0; k < H; k++) base[k] = (int32_t)floor((4.0 * N - psd_d[k] + 1e-7) * SCALE);
    for (int k = 0; k < H; k++) {
        cap1nat[k] = (int16_t)(base[k] - 2 * SCALE > 32767 ? 32767 : base[k] - 2 * SCALE);
        cap2nat[k] = (int16_t)(base[k] - SCALE > 32767 ? 32767 : base[k] - SCALE);
    }
    for (int k = H; k < HP; k++) { cap1nat[k] = 32767; cap2nat[k] = 32767; }
    int m = 0;
    scratch_reserve(sc, 4096);
    for (int round = 0; round < 8; round++) {
        int prev = m;
        m = 0;
#ifdef __AVX2__
        __m256i F0 = _mm256_loadu_si256((const __m256i *)cap1nat);
        __m256i F1 = _mm256_loadu_si256((const __m256i *)(cap1nat + 16));
        if (round == 0) {
            for (size_t c = 0; c < NSKEW; c++) {
                const int16_t *p = PSDT + c * HP;
                __m256i g = _mm256_or_si256(
                    _mm256_cmpgt_epi16(_mm256_load_si256((const __m256i *)p), F0),
                    _mm256_cmpgt_epi16(_mm256_load_si256((const __m256i *)(p + 16)), F1));
                if (!_mm256_testz_si256(g, g)) continue;
                if (m + 1 >= sc->cap_m) scratch_reserve(sc, m + 2);
                sc->idx[m++] = (int32_t)c;
            }
        } else {
            for (int i = 0; i < prev; i++) {
                const int16_t *p = PSDT + (size_t)sc->idx[i] * HP;
                __m256i g = _mm256_or_si256(
                    _mm256_cmpgt_epi16(_mm256_load_si256((const __m256i *)p), F0),
                    _mm256_cmpgt_epi16(_mm256_load_si256((const __m256i *)(p + 16)), F1));
                if (!_mm256_testz_si256(g, g)) continue;
                sc->idx[m++] = sc->idx[i];
            }
        }
#else
        size_t lim = round == 0 ? NSKEW : (size_t)prev;
        for (size_t t = 0; t < lim; t++) {
            size_t c = round == 0 ? t : (size_t)sc->idx[t];
            const int16_t *p = PSDT + c * HP;
            int ok = 1;
            for (int k = 0; k < H; k++) if (p[k] > cap1nat[k]) { ok = 0; break; }
            if (!ok) continue;
            if (m + 1 >= sc->cap_m) scratch_reserve(sc, m + 2);
            sc->idx[m++] = (int32_t)c;
        }
#endif
        if (m < 1) break;
        int32_t mn[64];
        for (int k = 0; k < H; k++) mn[k] = 32767;
        for (int i = 0; i < m; i++) {
            const int16_t *p = PSDT + (size_t)sc->idx[i] * HP;
            for (int k = 0; k < H; k++) if (p[k] < mn[k]) mn[k] = p[k];
        }
        int changed = 0;
        for (int k = 0; k < H; k++) {
            int32_t c1 = base[k] - 2 * mn[k];
            int32_t c2 = base[k] - mn[k];
            if (c1 > 32767) c1 = 32767;
            if (c2 > 32767) c2 = 32767;
            if (c1 < cap1nat[k]) { cap1nat[k] = (int16_t)c1; changed = 1; }
            if (c2 < cap2nat[k]) { cap2nat[k] = (int16_t)c2; }
        }
        if (!changed && round > 0) break;
        if (round > 0 && m == prev && !changed) break;
    }
    for (int k = 0; k < H; k++) { cap1p[k] = cap1nat[order[k]]; cap2p[k] = cap2nat[order[k]]; }
    for (int k = H; k < HP; k++) { cap1p[k] = 32767; cap2p[k] = 32767; }
    for (int k = 0; k < HP; k++) cap16[k] = (int16_t)(cap2p[k] > 32767 ? 32767 : cap2p[k]);
    *nout = 0;
    *tested_out = 0;
    if (m < 1) return 0;   /* a, b, c may repeat, so one sequence can do */

    for (int i = 0; i < m; i++) {
        const int16_t *p = PSDT + (size_t)sc->idx[i] * HP;
        int16_t *q = sc->psd + (size_t)i * HP;
        for (int k = 0; k < H; k++) q[k] = p[order[k]];
        for (int k = H; k < HP; k++) q[k] = -32768;
        memcpy(sc->paf + (size_t)i * H, PAFT + (size_t)sc->idx[i] * H, H);
    }
    /* Quantise each spectrum into levels 0..3 with level = floor(4*psd/cap),
       packed into 3-bit fields. Then level_a + level_b >= 4 implies
       psd_a + psd_b > cap, so one add and one mask reject the pair: the guard
       bit of any field carries exactly that condition. Fields hold sums up to
       6, so no carry crosses a field boundary. Strictly stronger than testing
       whether both exceed half the cap, at the same cost. */
    for (int i = 0; i < m; i++) {
        uint64_t w0 = 0, w1 = 0;
        const int16_t *p = sc->psd + (size_t)i * HP;
        for (int k = 0; k < H; k++) {
            int32_t c = cap2p[k];
            int lv = c > 0 ? (int)((16LL * (int64_t)p[k]) / c) : 15;
            if (lv < 0) lv = 0;
            if (lv > 15) lv = 15;
            if (k < 12) w0 |= (uint64_t)lv << (5 * (11 - k));
            else w1 |= (uint64_t)lv << (5 * (20 - k));
        }
        sc->mask[i] = w0;
        sc->mask2[i] = w1;
    }
    /* elementwise minimum level per block of BLK: if a candidate already
       trips a guard bit against the block minimum, every member of the block
       trips it too, so the whole block is skipped with one add */
    {
        int nblk = (m + BLK - 1) / BLK;
        for (int b = 0; b < nblk; b++) {
            int end = (b + 1) * BLK; if (end > m) end = m;
            uint64_t lo0 = 0, lo1 = 0;
            for (int k = 0; k < H; k++) {
                int mn = 15;
                for (int i = b * BLK; i < end; i++) {
                    int lv = (k < 12)
                        ? (int)((sc->mask[i] >> (5 * k)) & 31ULL)
                        : (int)((sc->mask2[i] >> (5 * (k - 12))) & 31ULL);
                    if (lv < mn) mn = lv;
                }
                if (k < 12) lo0 |= (uint64_t)mn << (5 * k);
                else lo1 |= (uint64_t)mn << (5 * (k - 12));
            }
            sc->bmin[b] = lo0;
            sc->bmin2[b] = lo1;
        }
    }
    {   /* lexicographic sort by the packed level words: primary key is the
           level at the tightest frequency, so the prefix bound still applies,
           and neighbours now agree on every frequency, which is what makes
           block minima tight */
        if (m > sc->cap_sort) {
            sc->cap_sort = m * 2;
            sc->ord = realloc(sc->ord, (size_t)sc->cap_sort * sizeof(int32_t));
            sc->psd2 = realloc(sc->psd2, (size_t)sc->cap_sort * HP * sizeof(int16_t));
            sc->paf2 = realloc(sc->paf2, (size_t)sc->cap_sort * H);
            sc->idx2 = realloc(sc->idx2, (size_t)sc->cap_sort * sizeof(int32_t));
        }
        for (int i = 0; i < m; i++) sc->ord[i] = i;
        SortCtx ctx = {sc->mask, sc->mask2};
        qsort_r(sc->ord, (size_t)m, sizeof(int32_t), cmp_words, &ctx);
        for (int i = 0; i < m; i++) {
            int src = sc->ord[i];
            memcpy(sc->psd2 + (size_t)i * HP, sc->psd + (size_t)src * HP,
                   HP * sizeof(int16_t));
            memcpy(sc->paf2 + (size_t)i * H, sc->paf + (size_t)src * H, H);
            sc->idx2[i] = sc->idx[src];
        }
        memcpy(sc->psd, sc->psd2, (size_t)m * HP * sizeof(int16_t));
        memcpy(sc->paf, sc->paf2, (size_t)m * H);
        memcpy(sc->idx, sc->idx2, (size_t)m * sizeof(int32_t));
        for (int i = 0; i < m; i++) {
            int src = sc->ord[i];
            sc->mask2_tmp[i] = sc->mask[src];
            sc->mask3_tmp[i] = sc->mask2[src];
        }
        memcpy(sc->mask, sc->mask2_tmp, (size_t)m * sizeof(uint64_t));
        memcpy(sc->mask2, sc->mask3_tmp, (size_t)m * sizeof(uint64_t));
    }

    /* 2D index over the two tightest frequencies: the list is sorted by
       (level0, level1, ...), so each level0 group holds a contiguous run per
       level1. Scanning only the runs with level1 <= 15-level1_a skips half
       the prefix without testing it. */
    {   /* index over the three tightest frequencies: the list is sorted by
           (level0, level1, level2, ...), so each (level0, level1) pair holds a
           contiguous run per level2 */
        for (int t = 0; t <= 257; t++) sc->bstart[t] = m;
        int prevb = -1;
        for (int i = 0; i < m; i++) {
            int l0 = (int)((sc->mask[i] >> 55) & 31ULL);
            int l1 = (int)((sc->mask[i] >> 50) & 31ULL);
            int l2 = (int)((sc->mask[i] >> 45) & 31ULL);
            if (l0 > 15) l0 = 15;
            if (l1 > 15) l1 = 15;
            if (l2 > 15) l2 = 15;
            int b = l0 * 16 + l1;
            if (b != prevb) {
                for (int t = prevb + 1; t <= b; t++) sc->bstart[t] = i;
                prevb = b;
            }
        }
        for (int t = prevb + 1; t <= 257; t++) sc->bstart[t] = m;
    }
    _t1 = wall(); phase[2] += _t1 - _t0; _t0 = _t1;
    int bits = 1;
    while ((1 << bits) < 2 * m) bits++;
    int tsize = 1 << bits;
    if (!sc->tab || sc->tabmask + 1 < tsize) { free(sc->tab);
        sc->tab = malloc((size_t)tsize * sizeof(Slot)); }
    sc->tabmask = tsize - 1;
    memset(sc->tab, 0, (size_t)tsize * sizeof(Slot));
    for (int i = 0; i < m; i++) {
        uint64_t h = hash_bytes(sc->paf + (size_t)i * H, H);
        int p = (int)(h & sc->tabmask);
        while (sc->tab[p].key) p = (p + 1) & sc->tabmask;
        sc->tab[p].key = h; sc->tab[p].idx = i;
    }
    int8_t target[64];
    for (int s = 1; s <= H; s++) {
        int acc = 0;
        for (int j = 0; j < N; j++) { int t = j + s; if (t >= N) t -= N; acc += d[j] * d[t]; }
        target[s - 1] = (int8_t)(-acc);
    }
    _t1 = wall(); phase[4] += _t1 - _t0; _t0 = _t1;
    const int lo_c = -(N - 2), hi_c = N - 4;
    long long found = 0, tested = 0, scanned = 0;
    int32_t tightcap = cap2p[0];
    for (int ia = 0; ia < m; ia++) {
        int la0 = (int)((sc->mask[ia] >> 55) & 31ULL);
        int la1 = (int)((sc->mask[ia] >> 50) & 31ULL);
        if (la0 > 15) la0 = 15;
        if (la1 > 15) la1 = 15;
        int R0 = 15 - la0, R1 = 15 - la1;
        if (R0 < 0 || R1 < 0) continue;
        uint64_t ma = sc->mask[ia], ma2 = sc->mask2[ia];
        uint64_t guard = 0, guard2 = 0;
        for (int k = 0; k < 12 && k < H; k++) guard |= 16ULL << (5 * (11 - k));
        for (int k = 12; k < H; k++) guard2 |= 16ULL << (5 * (20 - k));
        const int8_t *aa = sc->paf + (size_t)ia * H;
#ifdef __AVX2__
        __m256i MA = _mm256_set1_epi64x((long long)ma);
        __m256i MA2 = _mm256_set1_epi64x((long long)ma2);
        __m256i GUARD = _mm256_set1_epi64x((long long)guard);
        __m256i GUARD2 = _mm256_set1_epi64x((long long)guard2);
        __m256i ZERO = _mm256_setzero_si256();
        for (int l0 = 0; l0 <= R0; l0++) {
            int rs = sc->bstart[l0 * 16];
            int bend = sc->bstart[l0 * 16 + R1 + 1];
            int ib = rs > ia ? rs : ia;
            if (ib >= bend) continue;
            for (; ib + 4 <= bend; ib += 4) {
                scanned += 4;
                __m256i MB = _mm256_loadu_si256((const __m256i *)(sc->mask + ib));
                __m256i MB2 = _mm256_loadu_si256((const __m256i *)(sc->mask2 + ib));
                __m256i bad = _mm256_or_si256(
                    _mm256_and_si256(_mm256_add_epi64(MA, MB), GUARD),
                    _mm256_and_si256(_mm256_add_epi64(MA2, MB2), GUARD2));
                int bits4 = _mm256_movemask_pd(_mm256_castsi256_pd(
                    _mm256_cmpeq_epi64(bad, ZERO)));
                if (!bits4) continue;
                for (int t = 0; t < 4; t++)
                    if (bits4 & (1 << t)) {
                        tested++;
                        try_pair(sc, ia, ib + t, aa, target, cap2p, cap16,
                                 lo_c, hi_c, out, nout, max_out, &found);
                    }
            }
            for (; ib < bend; ib++) {
                scanned++;
                if (((ma + sc->mask[ib]) & guard) |
                    ((ma2 + sc->mask2[ib]) & guard2)) continue;
                tested++;
                try_pair(sc, ia, ib, aa, target, cap2p, cap16, lo_c, hi_c,
                         out, nout, max_out, &found);
            }
        }
#else
        for (int l0 = 0; l0 <= R0; l0++) {
            int rs = sc->bstart[l0 * 16];
            int bend = sc->bstart[l0 * 16 + R1 + 1];
            for (int ib = rs > ia ? rs : ia; ib < bend; ib++) {
                scanned++;
                if (((ma + sc->mask[ib]) & guard) |
                    ((ma2 + sc->mask2[ib]) & guard2)) continue;
                tested++;
                try_pair(sc, ia, ib, aa, target, cap2p, cap16, lo_c, hi_c,
                         out, nout, max_out, &found);
            }
        }
#endif
    }
    _t1 = wall(); phase[3] += _t1 - _t0;
    *tested_out = tested; SCANNED += scanned;
    return found;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: census <n> [threads] [out]\n"); return 1; }
    N = atoi(argv[1]);
    if (!(N == 3 || N == 7 || N == 13 || N == 21 || N == 31 || N == 43)) {
        fprintf(stderr, "supported census orders: 3,7,13,21,31,43\n"); return 2;
    }
    H = (N - 1) / 2;
    HP = ((H + 15) / 16) * 16;
    if (HP < 32) HP = 32;
    /* A skew sequence and its complement (all free bits flipped) have
       identical autocorrelations, since complementing is decimation by -1 and
       PAF(-s) = PAF(s). No code is self-complementary, so exactly half the
       skew sequences are redundant: enumerate one of each pair and expand the
       results at the end. */
    NSKEW = (size_t)1 << (H - 1);
    int threads = argc > 2 ? atoi(argv[2]) : 4;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const char *outpath = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    int dlimit = argc > 4 ? atoi(argv[4]) : 0;
    double t0 = wall();
    int nd, nadm;
    long long *ds = canonical_ds(&nd, &nadm);
    if (dlimit && dlimit < nd) nd = dlimit;
    printf("n=%d: %d admissible d -> %d after ceiling and decimation (%.0fx)\n",
           N, nadm, nd, (double)nadm / (nd ? nd : 1));
    fflush(stdout);
    build_tables();
    printf("   tables built: %.0f MB autocorrelation (int8) + %.0f MB spectra "
           "(int16), half the skew space  [%.1fs]\n", NSKEW * (double)H / 1e6,
           NSKEW * (double)HP * 2 / 1e6, wall() - t0);
    fflush(stdout);

    int max_out = 1 << 16;
    int32_t *sols = malloc((size_t)max_out * 3 * sizeof(int32_t));
    long long *sold = malloc((size_t)max_out * sizeof(long long));
    int total_out = 0;
    long long tested_all = 0;
    int done = 0;
    double PHASE[5] = {0, 0, 0, 0, 0};
    long long SCANNED_TOTAL = 0;

#pragma omp parallel
    {
        Scratch sc; memset(&sc, 0, sizeof(sc));
        double ph[5] = {0, 0, 0, 0, 0};
        int32_t local[3 * 256];
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < nd; i++) {
            int nout = 0;
            long long tested = 0;
            solve_d(ds[i], &sc, local, &nout, 256, &tested, ph);
#pragma omp critical
            {
                tested_all += tested;
                if (total_out + nout > max_out) { fprintf(stderr, "global solution buffer exhausted\n"); exit(2); }
                for (int j = 0; j < nout; j++) {
                    sols[(size_t)total_out * 3 + 0] = local[j * 3 + 0];
                    sols[(size_t)total_out * 3 + 1] = local[j * 3 + 1];
                    sols[(size_t)total_out * 3 + 2] = local[j * 3 + 2];
                    sold[total_out] = ds[i];
                    total_out++;
                }
                for (int q = 0; q < 5; q++) PHASE[q] += ph[q];
                SCANNED_TOTAL += SCANNED; SCANNED = 0;
                for (int q = 0; q < 5; q++) ph[q] = 0;
                done++;
                if (done % (nd / 20 + 1) == 0 || done == nd)
                    printf("   d %d/%d | solutions %d | pairs tested %lld [%.0fs]\n",
                           done, nd, total_out, tested_all, wall() - t0);
                fflush(stdout);
            }
        }
        free(sc.idx); free(sc.psd); free(sc.paf); free(sc.mask); free(sc.mask2); free(sc.mask2_tmp); free(sc.mask3_tmp); free(sc.bmin); free(sc.bmin2); free(sc.tab);
        free(sc.ord); free(sc.idx2); free(sc.hist); free(sc.psd2); free(sc.paf2);

    }
    double dt = wall() - t0;
    printf("\nn=%d: %d canonical solutions\n", N, total_out);
    printf("   pairs actually tested: %lld\n", tested_all);
    printf("   wall clock: %.1fs on %d threads\n", dt, threads);
    printf("   thread-seconds by phase: filter %.0f | gather+sort %.0f | "
           "levels %.0f | hash %.0f | pair loop %.0f\n", PHASE[0], PHASE[1], PHASE[2], PHASE[4], PHASE[3]);
    printf("   positions scanned: %lld (%.1f per surviving candidate)\n",
           SCANNED_TOTAL, (double)SCANNED_TOTAL / (tested_all ? tested_all : 1));
    if (outpath) {
        FILE *f = fopen(outpath, "w");
        if (!f) { perror("census output"); return 2; }
        int8_t *a = malloc(N), *b = malloc(N), *c = malloc(N), *d = malloc(N);
        long long fullmask = (1LL << H) - 1;
        for (int i = 0; i < total_out * 8; i++) {
            int32_t *w = sols + (size_t)(i / 8) * 3;
            int flip = i % 8;
            int8_t *seqs[3] = {a, b, c};
            for (int t = 0; t < 3; t++) {
                long long code = w[t];
                if (flip & (1 << t)) code ^= fullmask;
                seqs[t][0] = 1;
                for (int j = 1; j <= H; j++) {
                    int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
                    seqs[t][j] = v; seqs[t][N - j] = -v;
                }
            }
            sym_seq(sold[i / 8], d);
            int8_t *all[4] = {a, b, c, d};
            for (int t = 0; t < 4; t++) {
                for (int j = 0; j < N; j++) fputc(all[t][j] > 0 ? '+' : '-', f);
                fputc(t == 3 ? '\n' : ' ', f);
            }
        }
        fclose(f);
        printf("   solutions -> %s\n", outpath);
    }
    return 0;
}
