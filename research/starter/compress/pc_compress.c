/* pc_compress.c -- first stage of the compression census of a circulant
 * Goethals-Seidel family of odd order n = d m: it writes the compressed
 * quadruples ("instances") that pc_lift.c lifts back to order n.
 *
 * The pattern argument names the block kinds in order, k for skew and s for
 * symmetric, skew blocks first: ssss (Williamson), ksss (good), kkss
 * (G-matrices), kkks (best).  A skew row has a_0 = 1 and a_{n-j} = -a_j, so its
 * row sum is 1; a symmetric row has a_{n-j} = a_j and is sign-normalised to a
 * positive row sum.  A quadruple has PAF_A + PAF_B + PAF_C + PAF_D = 0 off zero,
 * so the squared row sums add up to 4n and every PSD is at most 4n.
 *
 * Compression to length m (entry j is the sum of the row over the coset
 * j + k m) preserves the PAF and PSD conditions.  A compressed skew row has
 * A'_0 = 1 and A'_{m-j} = -A'_j; a compressed symmetric row is symmetric.
 *
 * Stage 1 (enum): every skew row (2^H of them, H = (n-1)/2) and every
 * symmetric row with x_0 = +1 whose squared row sum fits is tested against the
 * PSD ceiling 4n at all H frequencies; each survivor is compressed and counted
 * in one list per block kind and row sum.  The code range can be split into
 * pieces whose list files are summed by the merge mode; merge checks that the
 * pieces tile the whole range.
 *
 * Stage 2 (match): per decomposition of 4n into the squared row sums
 * (symmetric sums nonincreasing), the realized compressions are matched into
 * instances on the length-m PAF condition by a meet in the middle: pairs of
 * the last two blocks that pass the pointwise PSD ceiling are hashed on minus
 * their PAF sum into a bucket table, and pairs of the first two blocks probe it;
 * every hit is compared exactly.  The pair hash also partitions the work: a run
 * can take one of K external shards (for distribution and restart), and inside
 * a shard the pair table is split into as many passes as the memory budget
 * PC_MEM_MB needs.  The units of Z_m act on instances and lift to units of
 * Z_n, so one instance per orbit is kept: the one whose sorted form (blocks
 * with equal row-sum keys sorted lexicographically, then compared block by
 * block) is least.  The lists are in the same lexicographic order and pairs
 * from one list are taken with the first index not above the second, so the
 * first block of an enumerated instance is first in its sorted form; such an
 * instance can be least only if that block is least in its own orbit, and
 * left pairs whose first block is not are skipped before the probe.
 * PC_NOREDUCE=1 writes every instance.
 *
 * Nothing here drops a quadruple: the PSD tests are necessary conditions with a
 * margin of 1e-6 against rounding, the hash only routes pairs, and every match
 * is checked exactly.  Rows are 128-bit words and codes 63-bit, so n <= 125
 * (127 is prime); the 2^H enumeration is the practical limit.
 *
 * Output line: k0 k1 k2 k3 A'_0..A'_h B'_0..B'_h C'_0..C'_h D'_0..D'_h with
 * h = (m-1)/2 and k = -1 for a skew block, else its row sum.
 *
 *   build: gcc -O3 -march=native -fopenmp -o pc_compress pc_compress.c -lm
 *   run:   pc_compress all   PATTERN n d threads OUT
 *          pc_compress enum  PATTERN n d threads LISTS [piece pieces]
 *          pc_compress merge LISTS_OUT LISTS_IN...
 *          pc_compress match LISTS threads OUT [shard shards]
 *   env:   PC_MEM_MB    pair-table budget of the match (default 4096)
 *          PC_HASH_LOG2 slots per list of the sparse count table (default 22)
 *          PC_SPARSE=1  use the sparse table even when the dense one fits
 *          PC_NOREDUCE=1 keep every instance, not one per multiplier orbit
 * Every output file is written as OUT.part and renamed when complete.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef unsigned __int128 u128;
#define NLMAX 16
static int N, D, M, H, HM, KIND[4], NK, NS;
static char PAT[5];
static u128 MASK;
static uint64_t K, BASE;

static void die(const char *m) { fprintf(stderr, "pc_compress: %s\n", m); exit(1); }
static inline u128 rot(u128 x, int s) { return ((x << s) | (x >> (N - s))) & MASK; }
static inline int pc(u128 x) { return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64)); }
static uint64_t ckey(const int *c) { uint64_t k = 0; for (int j = 0; j < HM; j++) k = k * BASE + (uint64_t)((c[j] + D) / 2); return k; }
static void unkey(uint64_t k, int *c) { for (int j = HM - 1; j >= 0; j--) { c[j] = 2 * (int)(k % BASE) - D; k /= BASE; } }
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
    if (N < 9 || N > 125 || N % 2 == 0 || D < 3 || D % 2 == 0 || N % D || N / D < 3) die("need odd n = d m with d, m >= 3 and n <= 125");
    M = N / D; H = (N - 1) / 2; HM = (M + 1) / 2;
    if (HM > 32) die("m too large");
    BASE = (uint64_t)D + 1;
    double kd = pow((double)BASE, HM);
    if (kd > 9.0e18) die("compression keys exceed 64 bits; use a larger d");
    K = 1; for (int j = 0; j < HM; j++) K *= BASE;
    MASK = (((u128)1) << N) - 1;
    int smax = (int)floor(sqrt(4.0 * N - NK)); if (smax % 2 == 0) smax--;
    NS = (smax + 1) / 2;
    if (NS + 1 > NLMAX) die("too many row sums");
}

/* ---- realized compressions: sorted (key, count) lists; list 0 skew, list i>0 row sum 2i-1 ---- */
typedef struct { uint64_t key, cnt; } KC;
typedef struct { KC *v; long n, cap; } Acc;
static Acc ACC[NLMAX];
static long STAT_SA, STAT_CS, STAT_SS;   /* skew survivors, symmetric candidates, symmetric survivors */
static long RANGE_LO, RANGE_HI;

static void acc_merge(Acc *a, const KC *b, long nb) {   /* b sorted by key */
    KC *o = malloc(sizeof(KC) * (a->n + nb + 1)); long i = 0, j = 0, k = 0;
    while (i < a->n && j < nb) {
        if (a->v[i].key < b[j].key) o[k++] = a->v[i++];
        else if (a->v[i].key > b[j].key) o[k++] = b[j++];
        else { o[k] = a->v[i++]; o[k++].cnt += b[j++].cnt; }
    }
    while (i < a->n) o[k++] = a->v[i++];
    while (j < nb) o[k++] = b[j++];
    free(a->v); a->v = o; a->n = k; a->cap = a->n + nb + 1;
}
static int cmpKC(const void *x, const void *y) { uint64_t p = ((const KC *)x)->key, q = ((const KC *)y)->key; return p < q ? -1 : p > q; }

/* count table of one chunk: dense array, or open addressing on key+1 */
typedef struct { int used, dense; uint32_t *cnt; uint64_t *hk; uint32_t *hc; uint64_t mask; long fill; } Tab;
static int SPARSE; static int HLOG2 = 22;
static void tab_init(Tab *t, int used) {
    memset(t, 0, sizeof *t);
    if (!(t->used = used)) return;
    t->dense = !SPARSE && K <= (1ULL << 25);
    if (t->dense) { t->cnt = calloc(K, 4); if (!t->cnt) die("out of memory (dense table)"); }
    else { uint64_t cap = 1ULL << HLOG2; t->hk = calloc(cap, 8); t->hc = calloc(cap, 4); t->mask = cap - 1; if (!t->hk || !t->hc) die("out of memory (sparse table)"); }
}
static inline void tab_add(Tab *t, uint64_t key) {
    if (t->dense) { __atomic_add_fetch(&t->cnt[key], 1, __ATOMIC_RELAXED); return; }
    uint64_t k1 = key + 1, i = mix64(k1) & t->mask;
    for (;;) {
        uint64_t cur = __atomic_load_n(&t->hk[i], __ATOMIC_ACQUIRE);
        if (cur == 0) {
            if (__atomic_compare_exchange_n(&t->hk[i], &cur, k1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                long f = __atomic_add_fetch(&t->fill, 1, __ATOMIC_RELAXED);
                if ((uint64_t)f * 10 > (t->mask + 1) * 7) die("sparse table over 70% full; raise PC_HASH_LOG2");
                cur = k1;
            }
        }
        if (cur == k1) { __atomic_add_fetch(&t->hc[i], 1, __ATOMIC_RELAXED); return; }
        i = (i + 1) & t->mask;
    }
}
/* fold a chunk table into the accumulator and clear it */
static void tab_fold(Tab *t, Acc *a) {
    long r = 0; KC *b;
    if (!t->used) return;
    if (t->dense) {
        for (uint64_t k = 0; k < K; k++) if (t->cnt[k]) r++;
        b = malloc(sizeof(KC) * (r + 1)); r = 0;
        for (uint64_t k = 0; k < K; k++) if (t->cnt[k]) { b[r].key = k; b[r++].cnt = t->cnt[k]; }
        memset(t->cnt, 0, K * 4);
    } else {
        b = malloc(sizeof(KC) * (t->fill + 1));
        for (uint64_t i = 0; i <= t->mask; i++) if (t->hk[i]) { b[r].key = t->hk[i] - 1; b[r++].cnt = t->hc[i]; }
        qsort(b, r, sizeof(KC), cmpKC);
        memset(t->hk, 0, (t->mask + 1) * 8); memset(t->hc, 0, (t->mask + 1) * 4); t->fill = 0;
    }
    acc_merge(a, b, r); free(b);
}
static void tab_free(Tab *t) { free(t->cnt); free(t->hk); free(t->hc); }

/* ---- stage 1 ---- */
static void enumerate(long lo, long hi) {
    int KP = (H + 7) / 8 * 8;
    double *cs = calloc((size_t)(H + 1) * KP, sizeof(double));   /* cs[s][k-1] = 2 cos(2 pi s k / n) */
    for (int s = 1; s <= H; s++) for (int k = 1; k <= H; k++) cs[(size_t)s * KP + k - 1] = 2.0 * cos(2.0 * M_PI * s * k / N);
    const double cap = 4.0 * N + 1e-6;
    /* XT[skew][byte][v]: the bits of the row set by byte number byte of the code */
    int NBY = (H + 7) / 8;
    u128 (*XT)[8][256] = calloc(2, sizeof *XT);
    for (int sk = 0; sk < 2; sk++) for (int by = 0; by < NBY; by++) for (int v = 0; v < 256; v++) {
        u128 x = 0;
        for (int t = 0; t < 8; t++) {
            int j = 8 * by + t + 1; if (j > H) break;
            int b = (v >> t) & 1;
            if (sk) x |= ((u128)1) << (b ? j : N - j);
            else if (b) { x |= ((u128)1) << j; x |= ((u128)1) << (N - j); }
        }
        XT[sk][by][v] = x;
    }
    int need_skew = NK > 0, need_sym = NK < 4;
    Tab T[NLMAX]; for (int i = 0; i <= NS; i++) tab_init(&T[i], i ? need_sym : need_skew);
    /* per chunk a key is counted at most once per code, so 2^31 codes keep counts below 2^32 */
    const long CHUNK = 1L << 31;
    long sa = 0, cs_ = 0, ss = 0;
    for (long c0 = lo; c0 < hi; c0 += CHUNK) {
        long c1 = hi - c0 > CHUNK ? c0 + CHUNK : hi;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+ : sa, cs_, ss)
        for (long code = c0; code < c1; code++) {
            for (int skew = 0; skew < 2; skew++) {
                if (skew ? !need_skew : !need_sym) continue;
                int sum = 1;
                if (!skew) {
                    int p = __builtin_popcountll((uint64_t)code);
                    sum = 1 + 2 * (2 * p - H);
                    if (sum * sum > 4 * N - NK) continue;
                    cs_++;
                }
                u128 x = 1;
                for (int by = 0; by < NBY; by++) x |= XT[skew][by][(code >> (8 * by)) & 255];
                double paf[64];
                for (int s = 1; s <= H; s++) paf[s] = N - 2 * pc(x ^ rot(x, s));
                int ok = 1;
                for (int kb = 0; kb < KP && ok; kb += 8) {
                    double acc[8];
                    for (int j = 0; j < 8; j++) acc[j] = N;
                    for (int s = 1; s <= H; s++) {
                        const double *row = cs + (size_t)s * KP + kb; double p = paf[s];
                        for (int j = 0; j < 8; j++) acc[j] += p * row[j];
                    }
                    for (int j = 0; j < 8; j++) if (acc[j] > cap) ok = 0;
                }
                if (!ok) continue;
                int sign = (!skew && sum < 0) ? -1 : 1, c[32];
                for (int j = 0; j < HM; j++) { int v = 0; for (int k = 0; k < D; k++) v += ((x >> ((j + M * k) % N)) & 1) ? 1 : -1; c[j] = sign * v; }
                if (skew) { sa++; tab_add(&T[0], ckey(c)); }
                else { ss++; tab_add(&T[(sign * sum + 1) / 2], ckey(c)); }
            }
        }
        for (int i = 0; i <= NS; i++) tab_fold(&T[i], &ACC[i]);
    }
    for (int i = 0; i <= NS; i++) tab_free(&T[i]);
    free(cs); free(XT);
    STAT_SA += sa; STAT_CS += cs_; STAT_SS += ss;
}

/* ---- list files ---- */
static void write_lists(const char *path) {
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.part", path);
    FILE *f = fopen(tmp, "w"); if (!f) die("cannot write list file");
    fprintf(f, "pc-lists 1 %s %d %d %ld %ld\n", PAT, N, D, RANGE_LO, RANGE_HI);
    fprintf(f, "stat %ld %ld %ld\n", STAT_SA, STAT_CS, STAT_SS);
    for (int i = 0; i <= NS; i++) for (long r = 0; r < ACC[i].n; r++) fprintf(f, "L %d %llu %llu\n", i, (unsigned long long)ACC[i].v[r].key, (unsigned long long)ACC[i].v[r].cnt);
    fprintf(f, "end\n");
    if (fflush(f) || ferror(f) || fclose(f)) die("write error on list file");
    if (rename(tmp, path)) die("cannot rename list file");
}
/* read one list file; the first sets the order, later ones must agree; lists are summed */
static void read_lists(const char *path, int first, long *lo, long *hi) {
    FILE *f = fopen(path, "r"); if (!f) die("cannot open list file");
    char pat[8]; int n, d; long a, b, sa, sc, ss;
    if (fscanf(f, "pc-lists 1 %4s %d %d %ld %ld\n", pat, &n, &d, &a, &b) != 5) die("bad list file header");
    if (first) setup(pat, n, d);
    else if (strcmp(pat, PAT) || n != N || d != D) die("list files are of different censuses");
    if (fscanf(f, "stat %ld %ld %ld\n", &sa, &sc, &ss) != 3) die("bad list file stat line");
    STAT_SA += sa; STAT_CS += sc; STAT_SS += ss;
    KC *buf[NLMAX]; long nb[NLMAX], cb[NLMAX];
    for (int i = 0; i < NLMAX; i++) { nb[i] = 0; cb[i] = 1024; buf[i] = malloc(sizeof(KC) * cb[i]); }
    char line[256]; int ended = 0;
    while (fgets(line, sizeof line, f)) {
        if (!strcmp(line, "end\n")) { ended = 1; break; }
        int i; unsigned long long k, c;
        if (sscanf(line, "L %d %llu %llu", &i, &k, &c) != 3 || i < 0 || i > NS || k >= K || c == 0) die("bad list file line");
        if (nb[i] && buf[i][nb[i] - 1].key >= k) die("list file keys not increasing");
        if (nb[i] == cb[i]) { cb[i] *= 2; buf[i] = realloc(buf[i], sizeof(KC) * cb[i]); }
        buf[i][nb[i]].key = k; buf[i][nb[i]++].cnt = c;
    }
    if (!ended) die("list file is truncated");
    fclose(f);
    for (int i = 0; i <= NS; i++) acc_merge(&ACC[i], buf[i], nb[i]);
    for (int i = 0; i < NLMAX; i++) free(buf[i]);
    *lo = a; *hi = b;
}

/* ---- stage 2 ---- */
typedef struct { int paf[32]; int psd[32]; uint64_t lifts; int c[32]; int omin; } C;
typedef struct { uint64_t h; uint32_t i, j; } P;

/* lexicographic order on compressed rows; it is also the order of the list keys */
static int cmp_c(const int *a, const int *b) { for (int j = 0; j < HM; j++) if (a[j] != b[j]) return a[j] < b[j] ? -1 : 1; return 0; }
static void sorted_form(const int s4[4], int c[4][32], int outc[4][32]) {
    int ord[4] = {0, 1, 2, 3};
    for (int i = 1; i < 4; i++) for (int j = i; j > 0; j--) {
        int x = ord[j - 1], y = ord[j];
        if (s4[x] == s4[y] && cmp_c(c[x], c[y]) > 0) { ord[j - 1] = y; ord[j] = x; } else break;
    }
    for (int i = 0; i < 4; i++) memcpy(outc[i], c[ord[i]], sizeof(int) * 32);
}
/* image of one compressed row under the unit u of Z_m */
static void image(const int *c, int skew, int u, int *img) {
    for (int j = 0; j < HM; j++) {
        int k = (u * j) % M;
        img[j] = k > HM - 1 ? (skew ? -c[M - k] : c[M - k]) : c[k];
    }
}
static int is_unit(int u) { int g = u, h = M; while (h) { int r = g % h; g = h; h = r; } return g == 1; }
/* keep an instance only if its sorted form is least in its orbit under the units of Z_m */
static int canonical(const int s4[4], int c[4][32]) {
    int self[4][32]; sorted_form(s4, c, self);
    for (int u = 2; u < M; u++) {
        if (!is_unit(u)) continue;
        int img[4][32], srt[4][32];
        for (int i = 0; i < 4; i++) image(c[i], KIND[i], u, img[i]);
        sorted_form(s4, img, srt);
        for (int i = 0; i < 4; i++) { int r = cmp_c(srt[i], self[i]); if (r < 0) return 0; if (r > 0) break; }
    }
    return 1;
}

static C *make_list(const Acc *a, int skew, long *nout) {
    C *L = malloc(sizeof(C) * (a->n ? a->n : 1));
    for (long r = 0; r < a->n; r++) {
        C *q = &L[r]; int x[64];
        memset(q, 0, sizeof *q);
        unkey(a->v[r].key, q->c); q->lifts = a->v[r].cnt;
        for (int j = 0; j < M; j++) x[j] = j < HM ? q->c[j] : (skew ? -q->c[M - j] : q->c[M - j]);
        for (int s = 1; s < HM; s++) {
            int p = 0; for (int j = 0; j < M; j++) p += x[j] * x[(j + s) % M];
            q->paf[s - 1] = p;
            double re = 0, im = 0;
            for (int j = 0; j < M; j++) { re += x[j] * cos(2 * M_PI * j * s / M); im += x[j] * sin(2 * M_PI * j * s / M); }
            q->psd[s - 1] = (int)floor(16 * (re * re + im * im) - 0.01);
        }
        q->omin = 1;   /* least in its orbit under the units of Z_m */
        for (int u = 2; u < M && q->omin; u++) { int img[32]; if (is_unit(u)) { image(q->c, skew, u, img); if (cmp_c(img, q->c) < 0) q->omin = 0; } }
    }
    *nout = a->n; return L;
}
static inline int psd_ok(const C *x, const C *y) {
    for (int k = 0; k < HM - 1; k++) if (x->psd[k] + y->psd[k] > 16 * 4 * N) return 0;
    return 1;
}
static inline uint64_t vhash(const int *v) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int k = 0; k < HM - 1; k++) { h ^= (uint64_t)(uint32_t)v[k]; h *= 0xff51afd7ed558ccdULL; h ^= h >> 29; }
    return mix64(h);
}
/* route of a pair: shard = low word mod shards, pass = (low word / shards) mod passes, bucket = high bits */
static inline uint64_t hroute(uint64_t h) { return h & 0xffffffffULL; }

static long match(const char *outp, int shard, int shards, double mem_bytes) {
    long nA = 0; C *LA = make_list(&ACC[0], 1, &nA);
    C *L[NLMAX]; long nL[NLMAX];
    for (int si = 0; si < NS; si++) L[si] = make_list(&ACC[si + 1], 0, &nL[si]);
    printf("n=%d d=%d m=%d %s: skew %ld survive; symmetric %ld candidates by row sum, %ld survive\n", N, D, M, PAT, STAT_SA, STAT_CS, STAT_SS);
    if (NK) printf("  skew: %ld realized compressions\n", nA);
    for (int si = 0; si < NS; si++) printf("  sum %2d: %ld realized compressions\n", 2 * si + 1, nL[si]);
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.part", outp);
    FILE *out = fopen(tmp, "w"); if (!out) die("cannot write instance file");
    int noreduce = getenv("PC_NOREDUCE") && *getenv("PC_NOREDUCE") && strcmp(getenv("PC_NOREDUCE"), "0");
    long inst_total = 0; double mitm_total = 0;
    int ns = 4 - NK, sv[4] = {0, 0, 0, 0};
    long ndec = 1; for (int i = 0; i < ns; i++) ndec *= NS;
    for (long code = 0; code < ndec; code++) {
        long c2 = code; int ok = 1, sq = NK;
        for (int i = 0; i < ns; i++) { sv[i] = 2 * (int)(c2 % NS) + 1; c2 /= NS; }
        for (int i = 1; i < ns; i++) if (sv[i] > sv[i - 1]) ok = 0;
        for (int i = 0; i < ns; i++) sq += sv[i] * sv[i];
        if (!ok || sq != 4 * N) continue;
        int key4[4]; C *LL[4]; long nn[4];
        for (int i = 0; i < 4; i++) {
            if (KIND[i]) { key4[i] = -1; LL[i] = LA; nn[i] = nA; }
            else { key4[i] = sv[i - NK]; LL[i] = L[(key4[i] - 1) / 2]; nn[i] = nL[(key4[i] - 1) / 2]; }
        }
        if (nn[0] > 0xffffffffL || nn[2] > 0xffffffffL || nn[3] > 0xffffffffL) die("list too long");
        int same01 = key4[0] == key4[1], same12 = key4[1] == key4[2], same23 = key4[2] == key4[3];
        double t0 = omp_get_wtime();
        /* right pairs of this shard, counted first to size the passes */
        long rcount = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : rcount)
        for (long i = 0; i < nn[2]; i++) for (long j = (same23 ? i : 0); j < nn[3]; j++) {
            const C *Cq = &LL[2][i], *Dq = &LL[3][j];
            if (!psd_ok(Cq, Dq)) continue;
            int v[32]; for (int k = 0; k < HM - 1; k++) v[k] = -(Cq->paf[k] + Dq->paf[k]);
            if ((long)(hroute(vhash(v)) % (uint64_t)shards) != shard) continue;
            rcount++;
        }
        int passes = (int)ceil((double)rcount * sizeof(P) * 2.2 / mem_bytes); if (passes < 1) passes = 1;
        long inst = 0, lscan = 0; double mitm = 0;
        for (int pass = 0; pass < passes; pass++) {
            long np = 0; P *R = malloc(sizeof(P) * (rcount / passes + 1024)); long rcap = rcount / passes + 1024;
#pragma omp parallel
            {
                long lcap = 1 << 12, ln = 0; P *loc = malloc(sizeof(P) * lcap);
#pragma omp for schedule(dynamic, 16) nowait
                for (long i = 0; i < nn[2]; i++) for (long j = (same23 ? i : 0); j < nn[3]; j++) {
                    const C *Cq = &LL[2][i], *Dq = &LL[3][j];
                    if (!psd_ok(Cq, Dq)) continue;
                    int v[32]; for (int k = 0; k < HM - 1; k++) v[k] = -(Cq->paf[k] + Dq->paf[k]);
                    uint64_t h = vhash(v), r = hroute(h);
                    if ((long)(r % (uint64_t)shards) != shard || (int)((r / (uint64_t)shards) % (uint64_t)passes) != pass) continue;
                    if (ln == lcap) { lcap *= 2; loc = realloc(loc, sizeof(P) * lcap); }
                    loc[ln].h = h; loc[ln].i = (uint32_t)i; loc[ln].j = (uint32_t)j; ln++;
                    if (ln == 1 << 16) {
#pragma omp critical(rappend)
                        { if (np + ln > rcap) { rcap = 2 * (np + ln); R = realloc(R, sizeof(P) * rcap); } memcpy(R + np, loc, sizeof(P) * ln); np += ln; }
                        ln = 0;
                    }
                }
#pragma omp critical(rappend)
                { if (np + ln > rcap) { rcap = 2 * (np + ln); R = realloc(R, sizeof(P) * rcap); } memcpy(R + np, loc, sizeof(P) * ln); np += ln; }
                free(loc);
            }
            /* bucket index on the high bits of the hash (parallel counting sort) */
            int bb = 1; while ((1L << bb) < np) bb++;
            long nb = 1L << bb; int shift = 64 - bb;
            long *start = calloc(nb + 1, sizeof(long));
#pragma omp parallel for schedule(static)
            for (long p = 0; p < np; p++) __atomic_fetch_add(&start[(R[p].h >> shift) + 1], 1, __ATOMIC_RELAXED);
            for (long b = 0; b < nb; b++) start[b + 1] += start[b];
            P *S = malloc(sizeof(P) * (np ? np : 1)); long *fill = malloc(sizeof(long) * nb); memcpy(fill, start, sizeof(long) * nb);
#pragma omp parallel for schedule(static)
            for (long p = 0; p < np; p++) S[__atomic_fetch_add(&fill[R[p].h >> shift], 1, __ATOMIC_RELAXED)] = R[p];
            free(fill); free(R);
#pragma omp parallel
            {
                long bcap = 1 << 16, bn = 0; char *buf = malloc(bcap);
#pragma omp for schedule(dynamic, 4) reduction(+ : inst, mitm, lscan)
                for (long i = 0; i < nn[0]; i++) for (long j = (same01 ? i : 0); j < nn[1] && (noreduce || LL[0][i].omin); j++) {
                    const C *A = &LL[0][i], *B = &LL[1][j];
                    int v[32]; for (int k = 0; k < HM - 1; k++) v[k] = A->paf[k] + B->paf[k];
                    uint64_t h = vhash(v), r = hroute(h);
                    if ((long)(r % (uint64_t)shards) != shard || (int)((r / (uint64_t)shards) % (uint64_t)passes) != pass) continue;
                    if (!psd_ok(A, B)) continue;
                    lscan++;
                    long b = (long)(h >> shift);
                    for (long p = start[b]; p < start[b + 1]; p++) {
                        if (S[p].h != h) continue;
                        if (same12 && j > (long)S[p].i) continue;
                        const C *Cq = &LL[2][S[p].i], *Dq = &LL[3][S[p].j];
                        int same = 1;
                        for (int k = 0; k < HM - 1 && same; k++) if (v[k] + Cq->paf[k] + Dq->paf[k] != 0) same = 0;
                        if (!same) continue;
                        const C *q4[4] = {A, B, Cq, Dq};
                        if (!noreduce) {
                            int cc[4][32];
                            memset(cc, 0, sizeof cc);
                            for (int t = 0; t < 4; t++) memcpy(cc[t], q4[t]->c, sizeof(int) * HM);
                            if (!canonical(key4, cc)) continue;
                        }
                        inst++;
                        mitm += (double)A->lifts * B->lifts + (double)Cq->lifts * Dq->lifts;
                        if (bn + 2048 > bcap) {
#pragma omp critical(outw)
                            fwrite(buf, 1, bn, out);
                            bn = 0;
                        }
                        bn += sprintf(buf + bn, "%d %d %d %d", key4[0], key4[1], key4[2], key4[3]);
                        for (int t = 0; t < 4; t++) for (int k = 0; k < HM; k++) bn += sprintf(buf + bn, " %d", q4[t]->c[k]);
                        buf[bn++] = '\n';
                    }
                }
#pragma omp critical(outw)
                fwrite(buf, 1, bn, out);
                free(buf);
            }
            free(S); free(start);
        }
        printf("(%d,%d,%d,%d): lists %ld %ld %ld %ld, shard %d/%d, %d passes, right pairs %ld, left pairs %ld, %ld instances, %.3g lifting pairs, %.1f s\n",
               key4[0], key4[1], key4[2], key4[3], nn[0], nn[1], nn[2], nn[3], shard, shards, passes, rcount, lscan, inst, mitm, omp_get_wtime() - t0);
        inst_total += inst; mitm_total += mitm;
    }
    if (fflush(out) || ferror(out) || fclose(out)) die("write error on instance file");
    if (rename(tmp, outp)) die("cannot rename instance file");
    printf("total: %ld instances, %.3g lifting pairs\n", inst_total, mitm_total);
    free(LA); for (int si = 0; si < NS; si++) free(L[si]);
    return inst_total;
}

static void env_setup(void) {
    const char *e = getenv("PC_SPARSE"); SPARSE = e && *e && strcmp(e, "0");
    e = getenv("PC_HASH_LOG2"); if (e && *e) HLOG2 = atoi(e);
    if (HLOG2 < 10 || HLOG2 > 34) die("PC_HASH_LOG2 out of range");
}
static double mem_budget(void) {
    const char *e = getenv("PC_MEM_MB"); double mb = e && *e ? atof(e) : 4096;
    if (!(mb > 0)) die("PC_MEM_MB must be positive");
    return mb * 1048576.0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    env_setup();
    const char *usage = "usage: pc_compress all PATTERN n d threads OUT\n"
                        "       pc_compress enum PATTERN n d threads LISTS [piece pieces]\n"
                        "       pc_compress merge LISTS_OUT LISTS_IN...\n"
                        "       pc_compress match LISTS threads OUT [shard shards]\n";
    if (argc < 2) { fputs(usage, stderr); return 1; }
    double t0 = omp_get_wtime();
    if (!strcmp(argv[1], "all") || !strcmp(argv[1], "enum")) {
        int is_all = !strcmp(argv[1], "all");
        if (argc != 7 && !(argc == 9 && !is_all)) { fputs(usage, stderr); return 1; }
        setup(argv[2], atoi(argv[3]), atoi(argv[4]));
        omp_set_num_threads(atoi(argv[5]));
        long total = 1L << H, piece = 0, pieces = 1;
        if (argc == 9) { piece = atol(argv[7]); pieces = atol(argv[8]); if (pieces < 1 || piece < 0 || piece >= pieces) die("bad piece"); }
        RANGE_LO = (long)((__int128)total * piece / pieces); RANGE_HI = (long)((__int128)total * (piece + 1) / pieces);
        enumerate(RANGE_LO, RANGE_HI);
        printf("enum %s n=%d d=%d codes [%ld, %ld) of %ld: skew %ld survive, symmetric %ld candidates %ld survive (%.1f s)\n",
               PAT, N, D, RANGE_LO, RANGE_HI, total, STAT_SA, STAT_CS, STAT_SS, omp_get_wtime() - t0);
        if (is_all) { match(argv[6], 0, 1, mem_budget()); printf("(%.1f s)\n", omp_get_wtime() - t0); }
        else write_lists(argv[6]);
        return 0;
    }
    if (!strcmp(argv[1], "merge")) {
        if (argc < 4) { fputs(usage, stderr); return 1; }
        int np = argc - 3; long *lo = malloc(sizeof(long) * np), *hi = malloc(sizeof(long) * np);
        for (int i = 0; i < np; i++) read_lists(argv[3 + i], i == 0, &lo[i], &hi[i]);
        /* the pieces must tile [0, 2^H) exactly */
        for (int i = 0; i < np; i++) for (int j = i + 1; j < np; j++) if (lo[j] < lo[i]) { long t = lo[i]; lo[i] = lo[j]; lo[j] = t; t = hi[i]; hi[i] = hi[j]; hi[j] = t; }
        long at = 0;
        for (int i = 0; i < np; i++) { if (lo[i] != at || hi[i] <= lo[i]) die("list pieces do not tile the code range"); at = hi[i]; }
        if (at != (1L << H)) die("list pieces do not cover the code range");
        RANGE_LO = 0; RANGE_HI = at;
        write_lists(argv[2]);
        printf("merged %d pieces: skew %ld survive, symmetric %ld candidates %ld survive\n", np, STAT_SA, STAT_CS, STAT_SS);
        return 0;
    }
    if (!strcmp(argv[1], "match")) {
        if (argc != 5 && argc != 7) { fputs(usage, stderr); return 1; }
        long lo, hi;
        read_lists(argv[2], 1, &lo, &hi);
        if (lo != 0 || hi != (1L << H)) die("list file does not cover the whole code range; merge the pieces first");
        omp_set_num_threads(atoi(argv[3]));
        int shard = 0, shards = 1;
        if (argc == 7) { shard = atoi(argv[5]); shards = atoi(argv[6]); if (shards < 1 || shard < 0 || shard >= shards) die("bad shard"); }
        match(argv[4], shard, shards, mem_budget());
        printf("(%.1f s)\n", omp_get_wtime() - t0);
        return 0;
    }
    fputs(usage, stderr);
    return 1;
}
