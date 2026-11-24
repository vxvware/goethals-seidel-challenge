/* pc2_join.c -- lift stage of a two-divisor compression census of a circulant
 * Goethals-Seidel family of odd order n: every quadruple of the family is
 * found from its compressions to two lengths m1 = n / d1 and m2 = n / d2
 * (gcd(d1, d2) = 1) by enumerating the rows with both compressions fixed and
 * joining them on the exact length-n PAF.
 *
 * The pattern names the block kinds, skew blocks first (ssss, ksss, kkss,
 * kkks).  A skew row has a_0 = 1 and a_{n-j} = -a_j; a symmetric row has
 * a_{n-j} = a_j and a positive row sum.  INST1 is the instance file of
 * pc2_inst.c at length m1 (one instance per orbit of the units of Z_m1), put
 * in signature order by the sort mode below.  The length-m2 side is built
 * here: every compressed row of length m2 that a row can compress to, of a
 * needed row sum, with PSD at most 4n minus the number of other skew blocks
 * at every frequency of Z_m2 (as pc2_inst.c lists them).
 *
 * Fibres.  Since gcd(d1, d2) = 1, lcm(m1, m2) = n and t -> (t mod m1, t mod m2)
 * is injective on Z_n.  With g = gcd(m1, m2) = n / (d1 d2), the positions
 * t = r (mod g) form a d2 x d1 grid (rows: residues mod m1, columns: residues
 * mod m2, both = r mod g), and the two compressions are its row and column
 * sums.  The map t -> -t takes fibre r to fibre -r.  A row of either kind is
 * fixed by its values at t = 1..(n-1)/2 (and t = 0 for a symmetric row); these
 * positions fall into the orbits {r, -r} of fibres, which meet disjoint rows
 * and columns, so the rows with both compressions given are the products of
 * independent orbit solutions, each a +/-1 assignment with the given margins.
 *
 * Per instance line I1 = (c1_A, c1_B, c1_C, c1_D) of INST1:
 *  - signature: the kinds, row sums and length-g compressions of the four
 *    blocks.  For each new signature, every ordered length-m2 quadruple with
 *    that signature, PAF sum zero at every lag of Z_m2 and pair PSD sums
 *    between 4n minus the largest PSD sum of the other pair and 4n is found by
 *    a meet in the middle (table on the side with fewer pairs, in passes by
 *    hash when it exceeds PC_MEM_MB; pairs whose first block is not least in
 *    its own orbit are skipped), and only the least quadruple of each orbit under
 *    the multipliers u = 1 (mod m1) is kept: these fix I1, so a class with a
 *    lift of I1 has a lift whose length-m2 quadruple is kept.  Lines of one
 *    signature are consecutive in a sorted INST1, so this runs once per
 *    signature (and once more at the start of each line range);
 *  - quadruples whose blocks fail the Gale-Ryser condition with c1 on some
 *    fibre (no 0/1 matrix with those margins) are dropped;
 *  - for each block and each length-m2 compression c2 still used, the rows
 *    with compressions (c1, c2) and PSD at most 4n minus the number of other
 *    skew blocks at every frequency are enumerated: orbit solutions by
 *    depth-first search with row and column capacity pruning (cached by the
 *    orbit's margins), combined by depth-first search on the partial
 *    transform, cut when the interval the remaining positions can add leaves
 *    some frequency outside the ceiling.  An orbit with more than
 *    PC_ORBIT_CAP solutions is not stored but searched under each combination
 *    of the others.  A quadruple with an empty row set in a block is dropped;
 *  - the blocks are split into two pairs (the split with the least pair work
 *    of the three), each side takes the row pairs over the distinct length-m2
 *    compression pairs of the surviving quadruples, pairs with a PSD sum above
 *    4n at some frequency are dropped, and the sides meet on a hash of the PAF
 *    sum (in key shards when the table would exceed PC_PAIR_MB), every hit
 *    compared exactly at all lags.
 * Every quadruple written is rechecked from its rows at every lag.  Nothing is
 * lost: each class has a member whose length-m1 compression is a line of
 * INST1 (equal-key blocks in sorted order; the units of Z_m1 lift to
 * multipliers of Z_n) and, among those, one whose length-m2 compression is a
 * kept quadruple; the row PSD is decided from the transform with a margin of
 * 0.01 and from the exact PAF within that margin, the cuts keep a slack of
 * 1e-6 n, and the PSD values the pair tests use are rounded down.
 *
 * Output line: four +/- rows, blocks in pattern order; a quadruple can appear
 * more than once.  Statistics per line and a summary go to standard error.
 *
 *   build: gcc -O3 -march=native -fopenmp -o pc2_join pc2_join.c -lm
 *   run:   pc2_join sort PATTERN n d1 d2 INST1 SORTED   (also writes SORTED.sigs,
 *                                                       the first line of each signature)
 *          pc2_join PATTERN n d1 d2 SORTED threads OUT [first last]
 *   env:   PC_PAIR_MB    pair-table budget in MiB (default 1024)
 *          PC_CACHE_MB   row-list cache in MiB (default 512)
 *          PC_ORBIT_MB   orbit-solution cache in MiB (default 256)
 *          PC_MEM_MB     table budget of the length-m2 match in MiB (default 512)
 *          PC_ORBIT_CAP  stored solutions per orbit (default 200000)
 *          PC_PROGRESS=file  append each finished line number
 *          PC_STOP=file      stop cleanly (exit 3) before the next line when
 *                            the file exists
 *          PC_ESTIMATE=k     cost estimate only: build k row lists per block
 *                            and line, print the extrapolated row count, row
 *                            CPU and pair visits, and join nothing
 *          PC_DEBUG=1        per-list and per-signature detail
 * Rows are 128-bit words, so n <= 127.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include <immintrin.h>
#include <time.h>

typedef unsigned __int128 u128;
#define VL 64   /* lanes of the PAF and PSD vectors */
#define HMX 64
static int N, D1, D2, M1, M2, G, H, HM1, HM2, KIND[4], NK;
static u128 MASK;
static char PAT[5];
static double PAIR_BYTES = 1024.0 * 1048576, CACHE_BYTES = 512.0 * 1048576;
static long ORBIT_CAP = 200000;
static int DEBUG, ESTIMATE;
static double COSK[VL + 1][VL + 1];   /* 2 cos(2 pi s k / n) */

static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void die(const char *m) { fprintf(stderr, "pc2_join: %s\n", m); exit(1); }
static inline u128 rot(u128 x, int s) { return ((x << s) | (x >> (N - s))) & MASK; }
static inline int pc(u128 x) { return __builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64)); }

/* ---------------- fibre orbits ---------------- */
/* a free position t sets a_t and a_{n-t}; its coefficients on the row sums
   (residues mod m1) and column sums (residues mod m2) are merged when both
   cells share a row or a column */
typedef struct { int t, nr, nc, row[2], rco[2], col[2], cco[2]; double w[VL]; } Var;
typedef struct { int nv; Var *v; double *rem; double tot[VL]; int nr, nc, rows[256], cols[256]; } Orbit;   /* rem[i H + k] = sum over vars >= i of |w_k| */
static Orbit ORB[2][HMX];   /* [skew][orbit] */
static int NORB;
static int CAPR0[2][256], CAPC0[2][256];

static int cmp_var(const void *a, const void *b) {
    const Var *p = a, *q = b;
    if (p->col[0] != q->col[0]) return p->col[0] - q->col[0];
    return p->row[0] - q->row[0];
}
static void add_coef(int *idx, int *co, int *n, int i, int c) {
    for (int k = 0; k < *n; k++) if (idx[k] == i) { co[k] += c; return; }
    idx[*n] = i; co[*n] = c; (*n)++;
}
static void build_orbits(void) {
    NORB = (G + 1) / 2;
    for (int skew = 0; skew < 2; skew++) {
        for (int o = 0; o < NORB; o++) { ORB[skew][o].nv = 0; ORB[skew][o].v = calloc(H + 1, sizeof(Var)); }
        for (int t = skew ? 1 : 0; t <= H; t++) {
            int r = t % G, o = r <= G / 2 ? r : G - r;
            Var *v = &ORB[skew][o].v[ORB[skew][o].nv++];
            memset(v, 0, sizeof *v); v->t = t;
            int sg = skew ? -1 : 1;
            add_coef(v->row, v->rco, &v->nr, t % M1, 1);
            add_coef(v->col, v->cco, &v->nc, t % M2, 1);
            if (t) { add_coef(v->row, v->rco, &v->nr, (N - t) % M1, sg); add_coef(v->col, v->cco, &v->nc, (N - t) % M2, sg); }
            /* transform: symmetric X(k) = a_0 + 2 sum a_t cos, skew X(k) = 1 + 2 i sum a_t sin */
            for (int k = 1; k <= H; k++)
                v->w[k - 1] = skew ? 2.0 * sin(2 * M_PI * (double)t * k / N) : (t ? 2.0 * cos(2 * M_PI * (double)t * k / N) : 1.0);
            for (int k = 0; k < v->nr; k++) CAPR0[skew][v->row[k]] += abs(v->rco[k]);
            for (int k = 0; k < v->nc; k++) CAPC0[skew][v->col[k]] += abs(v->cco[k]);
        }
        for (int o = 0; o < NORB; o++) {
            Orbit *ob = &ORB[skew][o];
            qsort(ob->v, ob->nv, sizeof(Var), cmp_var);
            ob->rem = calloc((size_t)(ob->nv + 1) * H, sizeof(double));
            for (int i = ob->nv - 1; i >= 0; i--) for (int k = 0; k < H; k++) ob->rem[i * H + k] = ob->rem[(i + 1) * H + k] + fabs(ob->v[i].w[k]);
            for (int k = 0; k < H; k++) ob->tot[k] = ob->rem[k];
            /* the rows and columns this orbit meets (no other orbit meets them) */
            ob->nr = ob->nc = 0;
            for (int i = 0; i < ob->nv; i++) {
                for (int k = 0; k < ob->v[i].nr; k++) { int u = ob->v[i].row[k], f = 0; for (int q = 0; q < ob->nr; q++) if (ob->rows[q] == u) f = 1; if (!f) ob->rows[ob->nr++] = u; }
                for (int k = 0; k < ob->v[i].nc; k++) { int u = ob->v[i].col[k], f = 0; for (int q = 0; q < ob->nc; q++) if (ob->cols[q] == u) f = 1; if (!f) ob->cols[ob->nc++] = u; }
            }
        }
    }
}

/* ---------------- rows with both compressions fixed ---------------- */
typedef struct { u128 bits; int16_t paf[VL]; int16_t psd[VL]; } Row;
typedef struct { long n, cap; Row *v; } RowList;
typedef struct { long n, cap; u128 *bits; double *w; double lo[VL], hi[VL]; int stream; } Sol;
typedef struct {
    int skew; double cap, r;   /* r: largest |X(k)| allowed, with slack */
    int R[256], C[256], capR[256], capC[256];
    Sol sol[HMX]; int ord[HMX]; double lo[HMX + 1][VL], hi[HMX + 1][VL];   /* bounds of the transform from levels l.. */
    RowList *out, scratch; long nodes;
} Ctx;

/* a row whose transform acc passed the cut: decide the ceiling (exactly near it) and store */
static void row_push(Ctx *c, u128 bits, const double *acc) {
    RowList *L = c->out;
    if (L->n == L->cap) { L->cap = L->cap ? 2 * L->cap : 1024; L->v = realloc(L->v, sizeof(Row) * L->cap); if (!L->v) die("out of memory (rows)"); }
    Row q; q.bits = bits;
    for (int k = H; k < VL; k++) q.paf[k] = q.psd[k] = 0;
    int paf[VL + 1];
    for (int s = 1; s <= H; s++) { paf[s] = N - 2 * pc(bits ^ rot(bits, s)); q.paf[s - 1] = (int16_t)paf[s]; }
    double sk = c->skew ? 1.0 : 0.0;
    for (int k = 1; k <= H; k++) {
        double v = acc[k - 1] * acc[k - 1] + sk;
        if (v > c->cap + 0.01) return;
        if (v > c->cap - 0.01) {   /* near the ceiling: the exact PSD from the PAF */
            double e = N; for (int s = 1; s <= H; s++) e += paf[s] * COSK[s][k];
            if (e > c->cap + 1e-6) return;
            v = e;
        }
        double f = floor(16.0 * v - 0.05);   /* at most 16 PSD: the pair test stays necessary */
        q.psd[k - 1] = (int16_t)(f < 0 ? 0 : f);
    }
    L->v[L->n++] = q;
}

/* 1 when, at some frequency, the transform stays outside [-r, r] for every completion:
   acc + [lo, hi] (the rest of this orbit: +-rem; later levels: their bounds) */
static inline int cut(const Ctx *c, const double *acc, const double *rem, const double *lo, const double *hi) {
    for (int k = 0; k < H; k++) {
        double a = acc[k] - rem[k] + lo[k], b = acc[k] + rem[k] + hi[k];
        if (a > c->r || b < -c->r) return 1;
    }
    return 0;
}

static const double ZERO[VL];
static void comb(Ctx *c, int level, u128 bits, const double *acc);
static int fdfs(Ctx *c, int level, const Orbit *o, int i, u128 bits, const double *acc, int collect) {
    c->nodes++;
    if (i == o->nv) {
        if (collect) {
            Sol *s = &c->sol[level];
            if (s->n >= ORBIT_CAP) return 1;
            if (s->n == s->cap) {
                s->cap = s->cap ? 2 * s->cap : 64;
                s->bits = realloc(s->bits, sizeof(u128) * s->cap); s->w = realloc(s->w, sizeof(double) * H * s->cap);
                if (!s->bits || !s->w) die("out of memory (orbit solutions)");
            }
            s->bits[s->n] = bits;
            for (int k = 0; k < H; k++) {
                s->w[s->n * H + k] = acc[k];
                if (!s->n || acc[k] < s->lo[k]) s->lo[k] = acc[k];
                if (!s->n || acc[k] > s->hi[k]) s->hi[k] = acc[k];
            }
            s->n++;
        } else comb(c, level + 1, bits, acc);
        return 0;
    }
    if (!collect && cut(c, acc, o->rem + (size_t)i * H, c->lo[level + 1], c->hi[level + 1])) return 0;
    const Var *v = &o->v[i];
    int over = 0;
    for (int k = 0; k < v->nr; k++) c->capR[v->row[k]] -= abs(v->rco[k]);
    for (int k = 0; k < v->nc; k++) c->capC[v->col[k]] -= abs(v->cco[k]);
    for (int a = 1; a >= -1 && !over; a -= 2) {
        int ok = 1;
        for (int k = 0; k < v->nr; k++) { c->R[v->row[k]] -= a * v->rco[k]; if (abs(c->R[v->row[k]]) > c->capR[v->row[k]]) ok = 0; }
        for (int k = 0; k < v->nc; k++) { c->C[v->col[k]] -= a * v->cco[k]; if (abs(c->C[v->col[k]]) > c->capC[v->col[k]]) ok = 0; }
        if (ok) {
            double nacc[VL];
            for (int k = 0; k < H; k++) nacc[k] = acc[k] + a * v->w[k];
            u128 nb = bits;
            if (a > 0) { nb |= ((u128)1) << v->t; if (v->t && !c->skew) nb |= ((u128)1) << (N - v->t); }
            else if (c->skew) nb |= ((u128)1) << (N - v->t);
            over = fdfs(c, level, o, i + 1, nb, nacc, collect);
        }
        for (int k = 0; k < v->nr; k++) c->R[v->row[k]] += a * v->rco[k];
        for (int k = 0; k < v->nc; k++) c->C[v->col[k]] += a * v->cco[k];
    }
    for (int k = 0; k < v->nr; k++) c->capR[v->row[k]] += abs(v->rco[k]);
    for (int k = 0; k < v->nc; k++) c->capC[v->col[k]] += abs(v->cco[k]);
    return over;
}

static void comb(Ctx *c, int level, u128 bits, const double *acc) {
    if (level == NORB) { row_push(c, bits, acc); return; }
    const Orbit *o = &ORB[c->skew][c->ord[level]];
    Sol *s = &c->sol[level];
    if (s->stream) { fdfs(c, level, o, 0, bits, acc, 0); return; }
    const double *lo = c->lo[level + 1], *hi = c->hi[level + 1];
    for (long q = 0; q < s->n; q++) {
        double nacc[VL]; const double *w = s->w + q * H;
        for (int k = 0; k < H; k++) nacc[k] = acc[k] + w[k];
        if (cut(c, nacc, ZERO, lo, hi)) continue;
        comb(c, level + 1, bits | s->bits[q], nacc);
    }
}

/* cache of orbit solutions keyed by (kind, orbit, residual margins on the orbit's rows and columns);
   an entry with stream = 1 had more than PC_ORBIT_CAP solutions */
typedef struct { int used, len; int8_t key[2 + 512]; Sol s; } OEnt;
static OEnt *OC; static long OCAP, ON; static double OBYTES, OC_BYTES = 256.0 * 1048576;
static long OC_HIT, OC_MISS;
static uint64_t okey_hash(const int8_t *k, int len) { uint64_t h = 1469598103934665603ULL; for (int i = 0; i < len; i++) { h ^= (uint64_t)(uint8_t)k[i]; h *= 1099511628211ULL; } return h ^ (h >> 29); }
static void oc_clear(void) {
    for (long i = 0; i < OCAP; i++) if (OC[i].used) { free(OC[i].s.bits); free(OC[i].s.w); }
    memset(OC, 0, sizeof(OEnt) * OCAP); ON = 0; OBYTES = 0;
}
/* room for more entries; called between lines only, since growing moves the entries */
static void oc_reserve(long more) {
    while ((ON + more + 1) * 2 > OCAP) {
        long oc = OCAP; OEnt *old = OC;
        OCAP = OCAP ? 2 * OCAP : 1 << 14; OC = calloc(OCAP, sizeof(OEnt)); if (!OC) die("out of memory (orbit cache)");
        for (long i = 0; i < oc; i++) if (old[i].used) { uint64_t p = okey_hash(old[i].key, old[i].len) & (OCAP - 1); while (OC[p].used) p = (p + 1) & (OCAP - 1); OC[p] = old[i]; }
        free(old);
    }
}
/* the entry of key, or NULL; with insert, an absent key gets a new entry and *fresh = 1 */
static OEnt *oc_find(const int8_t *k, int len, int insert, int *fresh) {
    uint64_t p = okey_hash(k, len) & (OCAP - 1);
    *fresh = 0;
    while (OC[p].used) { if (OC[p].len == len && !memcmp(OC[p].key, k, len)) return &OC[p]; p = (p + 1) & (OCAP - 1); }
    if (!insert) return NULL;
    if ((ON + 1) * 2 > OCAP) die("internal error: orbit cache full");
    OEnt *e = &OC[p]; memset(e, 0, sizeof *e); e->used = 1; e->len = len; memcpy(e->key, k, len); ON++;
    *fresh = 1; return e;
}

/* every row of kind skew with compressions c1 (length m1, full) and c2 (length m2, full) under the PSD ceiling */
static void build_rows_(Ctx *c, int skew, const int *c1f, const int *c2f);
/* the rows go to the context's scratch list first, then to an exact-size copy */
static void build_rows(Ctx *c, int skew, const int *c1f, const int *c2f, RowList *out) {
    c->scratch.n = 0; c->out = &c->scratch;
    build_rows_(c, skew, c1f, c2f);
    out->n = out->cap = c->scratch.n; out->v = NULL;
    if (out->n) { out->v = malloc(sizeof(Row) * out->n); if (!out->v) die("out of memory (rows)"); memcpy(out->v, c->scratch.v, sizeof(Row) * out->n); }
}
static void build_rows_(Ctx *c, int skew, const int *c1f, const int *c2f) {
    c->skew = skew; c->nodes = 0;
    c->cap = 4.0 * N - (NK - skew);
    c->r = sqrt(c->cap - (skew ? 1.0 : 0.0)) + 1e-6 * N;
    for (int u = 0; u < M1; u++) { c->R[u] = c1f[u]; c->capR[u] = CAPR0[skew][u]; }
    for (int v = 0; v < M2; v++) { c->C[v] = c2f[v]; c->capC[v] = CAPC0[skew][v]; }
    if (skew) { c->R[0] -= 1; c->C[0] -= 1; }
    for (int u = 0; u < M1; u++) if (abs(c->R[u]) > c->capR[u] || (c->R[u] - c->capR[u]) % 2) return;
    for (int v = 0; v < M2; v++) if (abs(c->C[v]) > c->capC[v] || (c->C[v] - c->capC[v]) % 2) return;
    /* collect each orbit's solutions; an orbit with too many is searched in the combination instead */
    int nst = 0, nsr = 0, st[HMX], sr[HMX];
    Sol tmp[HMX];
    for (int o = 0; o < NORB; o++) {
        const Orbit *ob = &ORB[skew][o];
        int8_t key[2 + 512]; int len = 0;
        key[len++] = (int8_t)skew; key[len++] = (int8_t)o;
        for (int q = 0; q < ob->nr; q++) key[len++] = (int8_t)c->R[ob->rows[q]];
        for (int q = 0; q < ob->nc; q++) key[len++] = (int8_t)c->C[ob->cols[q]];
        /* an entry is inserted and filled in one critical section, so a found entry is complete */
        int fresh; OEnt *e;
#pragma omp critical(ocache)
        e = oc_find(key, len, 0, &fresh);
        if (!e) {
            Sol s; memset(&s, 0, sizeof s);
            c->sol[0] = s;
            double acc[VL] = {0};
            int over = fdfs(c, 0, ob, 0, 0, acc, 1);
            s = c->sol[0];
            if (over) { free(s.bits); free(s.w); memset(&s, 0, sizeof s); s.stream = 1; }
#pragma omp critical(ocache)
            {
                e = oc_find(key, len, 1, &fresh);
                if (fresh) { e->s = s; OBYTES += (double)s.n * (sizeof(u128) + sizeof(double) * H) + sizeof(OEnt); OC_MISS++; }
                else { free(s.bits); free(s.w); }
            }
        } else {
#pragma omp atomic
            OC_HIT++;
        }
        tmp[o] = e->s;
        if (tmp[o].stream) sr[nsr++] = o;
        else if (tmp[o].n == 0) return;
        else st[nst++] = o;
    }
    /* stored orbits first, fewest solutions first (the cut then works on the larger lists); searched orbits last */
    for (int i = 1; i < nst; i++) for (int j = i; j > 0 && tmp[st[j]].n < tmp[st[j - 1]].n; j--) { int x = st[j]; st[j] = st[j - 1]; st[j - 1] = x; }
    int L = 0;
    for (int i = 0; i < nst; i++) { c->ord[L] = st[i]; c->sol[L] = tmp[st[i]]; L++; }
    for (int i = 0; i < nsr; i++) { c->ord[L] = sr[i]; c->sol[L] = tmp[sr[i]]; L++; }
    for (int k = 0; k < H; k++) c->lo[NORB][k] = c->hi[NORB][k] = 0;
    for (int l = NORB - 1; l >= 0; l--) for (int k = 0; k < H; k++) {
        double lo, hi;
        if (c->sol[l].stream) { hi = ORB[skew][c->ord[l]].tot[k]; lo = -hi; } else { lo = c->sol[l].lo[k]; hi = c->sol[l].hi[k]; }
        c->lo[l][k] = c->lo[l + 1][k] + lo; c->hi[l][k] = c->hi[l + 1][k] + hi;
    }
    double acc[VL] = {0};
    comb(c, 0, skew ? (u128)1 : 0, acc);
}

/* ---------------- row-list cache ---------------- */
typedef struct { int used; int skew; int8_t c1[HMX], c2[HMX]; RowList L; } CEnt;
static CEnt *CACHE; static long CCAP, CN; static double CBYTES;
static uint64_t ckey(int skew, const int8_t *c1, const int8_t *c2) {
    uint64_t h = 1469598103934665603ULL ^ (uint64_t)skew;
    for (int j = 0; j < HM1; j++) { h ^= (uint64_t)(uint8_t)c1[j]; h *= 1099511628211ULL; }
    for (int j = 0; j < HM2; j++) { h ^= (uint64_t)(uint8_t)c2[j] + 7; h *= 1099511628211ULL; }
    return h ^ (h >> 29);
}
static void cache_clear(void) {
    for (long i = 0; i < CCAP; i++) if (CACHE[i].used) { free(CACHE[i].L.v); }
    memset(CACHE, 0, sizeof(CEnt) * CCAP); CN = 0; CBYTES = 0;
}
static void cache_grow(void) {
    long oc = CCAP; CEnt *old = CACHE;
    CCAP = CCAP ? 2 * CCAP : 1 << 12; CACHE = calloc(CCAP, sizeof(CEnt)); if (!CACHE) die("out of memory (cache)");
    for (long i = 0; i < oc; i++) if (old[i].used) {
        uint64_t p = ckey(old[i].skew, old[i].c1, old[i].c2) & (CCAP - 1);
        while (CACHE[p].used) p = (p + 1) & (CCAP - 1);
        CACHE[p] = old[i];
    }
    free(old);
}
/* the slot of (skew, c1, c2), inserted empty (used = 2) when absent */
static CEnt *cache_slot(int skew, const int8_t *c1, const int8_t *c2, int *fresh) {
    if ((CN + 1) * 2 > CCAP) die("internal error: cache full");
    uint64_t p = ckey(skew, c1, c2) & (CCAP - 1);
    while (CACHE[p].used) {
        CEnt *e = &CACHE[p];
        if (e->skew == skew && !memcmp(e->c1, c1, HM1) && !memcmp(e->c2, c2, HM2)) { *fresh = 0; return e; }
        p = (p + 1) & (CCAP - 1);
    }
    CEnt *e = &CACHE[p]; memset(e, 0, sizeof *e);
    e->used = 2; e->skew = skew; memcpy(e->c1, c1, HM1); memcpy(e->c2, c2, HM2); CN++;
    *fresh = 1; return e;
}

/* ---------------- length-m2 compressed rows ---------------- */
/* every compressed row of length m2 that a row of length n can compress to
   (entries odd in [-d2, d2], c_0 = 1 for skew, c_{m-j} = c_j or -c_j) with a
   needed row sum and PSD at most 4n minus the number of other skew blocks at
   every frequency of Z_m2, as in pc2_inst.c; grouped by the compression to
   length g */
#define NLMAX 32
typedef struct { long n, cap; int8_t *c; int16_t *paf; int32_t *psd; int *gs; long nb; long *bstart; uint64_t *bhash; long bcap; long *bslot; } CL;
static CL L2[NLMAX + 1];   /* 0 skew, i >= 1 symmetric with row sum 2i - 1 */
static int NS;
static double C2COS[HMX][HMX], C2SIN[HMX][HMX];
static void full_row(const int *c, int hm, int m, int skew, int *f) { for (int j = 0; j < m; j++) f[j] = j < hm ? c[j] : (skew ? -c[m - j] : c[m - j]); }
static void gcomp(const int *f, int m, int *gc) { for (int r = 0; r < G; r++) gc[r] = 0; for (int j = 0; j < m; j++) gc[j % G] += f[j]; }
static uint64_t ghash(const int *gc) { uint64_t h = 1469598103934665603ULL; for (int r = 0; r < G; r++) { h ^= (uint64_t)(gc[r] + 100000); h *= 1099511628211ULL; } return h ^ (h >> 31); }

static void cl_push(CL *L, const int *c) {
    if (L->n == L->cap) { L->cap = L->cap ? 2 * L->cap : 1024; L->c = realloc(L->c, (size_t)L->cap * HM2); if (!L->c) die("out of memory (m2 list)"); }
    for (int j = 0; j < HM2; j++) L->c[(size_t)L->n * HM2 + j] = (int8_t)c[j];
    L->n++;
}
typedef struct { int skew, s; double cap; CL *out; } J2;
static void dfs2(const J2 *jb, int j, int *c, const double *acc, int partial) {
    if (j == HM2) {
        int cc[HMX]; memcpy(cc, c, sizeof(int) * HM2);
        if (!jb->skew) {
            int c0 = jb->s - partial;
            if (c0 > D2 || c0 < -D2) return;
            cc[0] = c0;
            for (int k = 1; k < HM2; k++) { double v = c0 + acc[k]; if (v * v > jb->cap) return; }
        } else {
            cc[0] = 1;
            for (int k = 1; k < HM2; k++) { double v = 1.0 + acc[k] * acc[k]; if (v > jb->cap) return; }
        }
        cl_push(jb->out, cc);
        return;
    }
    if (!jb->skew) {
        int r = HM2 - j, lo = jb->s - partial - 2 * D2 * r, hi = jb->s - partial + 2 * D2 * r;
        if (hi < -D2 || lo > D2) return;
    }
    double nacc[HMX];
    for (int v = -D2; v <= D2; v += 2) {
        c[j] = v;
        for (int k = 1; k < HM2; k++) nacc[k] = acc[k] + 2.0 * v * (jb->skew ? C2SIN[j][k] : C2COS[j][k]);
        dfs2(jb, j + 1, c, nacc, partial + 2 * v);
    }
}
static CL *SORTL;
static int cmp_gs(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    const int *gx = SORTL->gs + x * G, *gy = SORTL->gs + y * G;
    for (int r = 0; r < G; r++) if (gx[r] != gy[r]) return gx[r] < gy[r] ? -1 : 1;
    return memcmp(SORTL->c + x * HM2, SORTL->c + y * HM2, HM2);
}
static void build_l2(int skew, int s, CL *L) {
    for (int j = 0; j < HM2; j++) for (int k = 0; k < HM2; k++) { C2COS[j][k] = cos(2 * M_PI * j * k / M2); C2SIN[j][k] = sin(2 * M_PI * j * k / M2); }
    int nv = D2 + 1, top = HM2 >= 3 ? nv * nv : nv;
    CL *part = calloc(top, sizeof(CL));
    J2 jb = {skew, s, 4.0 * N - (NK - skew) + 1e-6, NULL};
#pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < top; t++) {
        int c[HMX]; double acc[HMX]; memset(acc, 0, sizeof acc); c[0] = 0;
        J2 my = jb; my.out = &part[t];
        int depth = HM2 >= 3 ? 3 : 2, partial = 0, tt = t;
        for (int j = 1; j < depth; j++) {
            int v = -D2 + 2 * (tt % nv); tt /= nv; c[j] = v; partial += 2 * v;
            for (int k = 1; k < HM2; k++) acc[k] += 2.0 * v * (skew ? C2SIN[j][k] : C2COS[j][k]);
        }
        dfs2(&my, depth, c, acc, partial);
    }
    CL raw; memset(&raw, 0, sizeof raw);
    for (int t = 0; t < top; t++) { for (long r = 0; r < part[t].n; r++) { int c[HMX]; for (int j = 0; j < HM2; j++) c[j] = part[t].c[r * HM2 + j]; cl_push(&raw, c); } free(part[t].c); }
    free(part);
    /* signatures, then sort by signature and bucket */
    raw.gs = malloc(sizeof(int) * (raw.n * G + 1));
    for (long r = 0; r < raw.n; r++) { int c[HMX], f[256]; for (int j = 0; j < HM2; j++) c[j] = raw.c[r * HM2 + j]; full_row(c, HM2, M2, skew, f); gcomp(f, M2, raw.gs + r * G); }
    long *ord = malloc(sizeof(long) * (raw.n + 1)); for (long r = 0; r < raw.n; r++) ord[r] = r;
    SORTL = &raw; qsort(ord, raw.n, sizeof(long), cmp_gs);
    memset(L, 0, sizeof *L);
    L->n = raw.n; L->c = malloc((size_t)raw.n * HM2 + 1); L->gs = malloc(sizeof(int) * (raw.n * G + 1));
    for (long r = 0; r < raw.n; r++) { memcpy(L->c + r * HM2, raw.c + ord[r] * HM2, HM2); memcpy(L->gs + r * G, raw.gs + ord[r] * G, sizeof(int) * G); }
    free(raw.c); free(raw.gs); free(ord);
    int K = HM2 - 1;
    double TC[HMX * 2], TS[HMX * 2];
    for (int t = 0; t < M2; t++) { TC[t] = cos(2 * M_PI * t / M2); TS[t] = sin(2 * M_PI * t / M2); }
    L->paf = malloc(sizeof(int16_t) * (L->n * K + 1)); L->psd = malloc(sizeof(int32_t) * (L->n * K + 1));
#pragma omp parallel for schedule(static)
    for (long r = 0; r < L->n; r++) {
        int x[256], c[HMX];
        for (int j = 0; j < HM2; j++) c[j] = L->c[r * HM2 + j];
        full_row(c, HM2, M2, skew, x);
        for (int sh = 1; sh < HM2; sh++) {
            int p = 0; for (int j = 0; j < M2; j++) p += x[j] * x[(j + sh) % M2];
            L->paf[r * K + sh - 1] = (int16_t)p;
            double re = 0, im = 0;
            for (int j = 0; j < M2; j++) { int t = (j * sh) % M2; re += x[j] * TC[t]; im += x[j] * TS[t]; }
            L->psd[r * K + sh - 1] = (int32_t)floor(16 * (re * re + im * im) - 0.01);
        }
    }
    /* buckets: bstart[b], b in [0, nb); hash table on the signature -> bucket */
    L->bstart = malloc(sizeof(long) * (L->n + 2)); L->nb = 0;
    for (long r = 0; r < L->n; r++) if (!r || memcmp(L->gs + r * G, L->gs + (r - 1) * G, sizeof(int) * G)) L->bstart[L->nb++] = r;
    L->bstart[L->nb] = L->n;
    L->bcap = 16; while (L->bcap < 2 * L->nb + 16) L->bcap <<= 1;
    L->bslot = malloc(sizeof(long) * L->bcap); for (long i = 0; i < L->bcap; i++) L->bslot[i] = -1;
    for (long b = 0; b < L->nb; b++) { uint64_t p = ghash(L->gs + L->bstart[b] * G) & (L->bcap - 1); while (L->bslot[p] >= 0) p = (p + 1) & (L->bcap - 1); L->bslot[p] = b; }
}
/* the rows of L whose length-g compression is gc: [*lo, *hi) */
static void cl_bucket(const CL *L, const int *gc, long *lo, long *hi) {
    *lo = *hi = 0;
    if (!L->bslot) return;
    uint64_t p = ghash(gc) & (L->bcap - 1);
    while (L->bslot[p] >= 0) {
        long b = L->bslot[p];
        if (!memcmp(L->gs + L->bstart[b] * G, gc, sizeof(int) * G)) { *lo = L->bstart[b]; *hi = L->bstart[b + 1]; return; }
        p = (p + 1) & (L->bcap - 1);
    }
}
/* Gale-Ryser on every fibre: a 0/1 matrix with these margins exists (necessary for a row) */
static int fibres_feasible(const int *c1f, const int *c2f) {
    for (int r = 0; r < G; r++) {
        int p[256], q[256], np = 0, nq = 0;
        for (int u = r; u < M1; u += G) p[np++] = (c1f[u] + D1) / 2;   /* ones in a fibre row of length d1 */
        for (int v = r; v < M2; v += G) q[nq++] = (c2f[v] + D2) / 2;   /* ones in a fibre column of length d2 */
        for (int i = 1; i < nq; i++) for (int j = i; j > 0 && q[j] > q[j - 1]; j--) { int t = q[j]; q[j] = q[j - 1]; q[j - 1] = t; }
        int sq = 0;
        for (int k = 1; k <= nq; k++) {
            sq += q[k - 1];
            int sp = 0; for (int i = 0; i < np; i++) sp += p[i] < k ? p[i] : k;
            if (sq > sp) return 0;
        }
    }
    return 1;
}
static int parse_line(const char *line, int hm, int s4[4], int c[4][HMX], int dd) {
    const char *p = line; char *q;
    for (int i = 0; i < 4; i++) { s4[i] = (int)strtol(p, &q, 10); if (q == p) return 0; p = q; if (s4[i] != -1 && (s4[i] < 1 || s4[i] % 2 == 0)) return 0; }
    for (int i = 0; i < 4; i++) for (int j = 0; j < hm; j++) { c[i][j] = (int)strtol(p, &q, 10); if (q == p) return 0; p = q; if (abs(c[i][j]) > dd || (c[i][j] + dd) % 2) return 0; }
    while (*p == ' ' || *p == '\t' || *p == '\r') p++;
    return *p == '\n' || *p == 0;
}

/* every ordered (c2_A, c2_B, c2_C, c2_D), block i from rows [lo_i, hi_i) of
   its list, with PAF sum zero at every lag of Z_m2 and pair PSD sums at most
   4n: a meet in the middle over the split of least pair work; writes list
   indices, four per instance */
#define L2W 32
typedef struct { int16_t psd[L2W]; int16_t paf[L2W]; } C2;
typedef struct { uint64_t h; uint32_t c, d; } P2;
static int NW2;   /* 256-bit words of a C2 vector */
static __m256i CAP2;
/* pair PSD sum at most 4n, and at least 4n minus the largest PSD sum the other two
   blocks reach (lb, in the same 16 PSD units, lowered by 4 for the floors) */
static inline int c2_ok(const C2 *a, const C2 *b, const __m256i *lb) {
    __m256i m = _mm256_setzero_si256();
    for (int w = 0; w < NW2; w++) {
        __m256i s = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)a->psd + w), _mm256_loadu_si256((const __m256i *)b->psd + w));
        m = _mm256_or_si256(m, _mm256_or_si256(_mm256_cmpgt_epi16(s, CAP2), _mm256_cmpgt_epi16(lb[w], s)));
    }
    return _mm256_testz_si256(m, m);
}
static inline uint64_t c2_key(const C2 *a, const C2 *b, int neg) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int w = 0; w < NW2; w++) {
        __m256i s = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)a->paf + w), _mm256_loadu_si256((const __m256i *)b->paf + w));
        if (neg) s = _mm256_sub_epi16(_mm256_setzero_si256(), s);
        uint64_t x0 = (uint64_t)_mm256_extract_epi64(s, 0), x1 = (uint64_t)_mm256_extract_epi64(s, 1), x2 = (uint64_t)_mm256_extract_epi64(s, 2), x3 = (uint64_t)_mm256_extract_epi64(s, 3);
        h = (h ^ (x0 * 0xff51afd7ed558ccdULL)) * 0xc4ceb9fe1a85ec53ULL; h ^= h >> 31;
        h = (h ^ (x1 * 0x9E3779B97F4A7C15ULL)) * 0xff51afd7ed558ccdULL; h ^= h >> 29;
        h = (h ^ (x2 * 0xc4ceb9fe1a85ec53ULL)) * 0x9E3779B97F4A7C15ULL; h ^= h >> 31;
        h = (h ^ (x3 * 0xff51afd7ed558ccdULL)) * 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    }
    return h;
}
static C2 *c2_gather(const CL *L, long lo, long hi) {
    int K = HM2 - 1; C2 *g = calloc((size_t)(hi - lo + 1), sizeof(C2));
    if (!g) die("out of memory (m2 candidates)");
    for (long r = lo; r < hi; r++) for (int k = 0; k < K; k++) { g[r - lo].psd[k] = (int16_t)L->psd[r * K + k]; g[r - lo].paf[k] = L->paf[r * K + k]; }
    return g;
}
static int KU[256], NKU;   /* defined with k_units below */
/* 1 when the compressed row c is least among its images under KU: the first block of a kept
   quadruple must be (an image with a smaller first block makes a smaller quadruple) */
static int row_kmin(const int8_t *c, int skew) {
    for (int q = 0; q < NKU; q++) {
        int u = KU[q];
        for (int j = 0; j < HM2; j++) {
            int k = (int)(((long)u * j) % M2);
            int v = k > HM2 - 1 ? (skew ? -c[M2 - k] : c[M2 - k]) : c[k];
            if (v != c[j]) { if (v < c[j]) return 0; break; }
        }
    }
    return 1;
}
static double M2_BYTES = 512.0 * 1048576;   /* PC_MEM_MB: table budget of the length-m2 match */
static long match2(CL *LL[4], const long lo[4], const long hi[4], long **outp, double *work) {
    C2 *GV[4]; long nx[4];
    for (int i = 0; i < 4; i++) { nx[i] = hi[i] - lo[i]; GV[i] = c2_gather(LL[i], lo[i], hi[i]); }
    uint8_t *kmin0 = malloc(nx[0] + 1);
    for (long a = 0; a < nx[0]; a++) kmin0[a] = (uint8_t)row_kmin(LL[0]->c + (lo[0] + a) * HM2, KIND[0]);
    static const int SPL[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    int best = 0; double bw = -1;
    for (int t = 0; t < 3; t++) { double w = (double)nx[SPL[t][0]] * nx[SPL[t][1]] + (double)nx[SPL[t][2]] * nx[SPL[t][3]]; if (bw < 0 || w < bw) { bw = w; best = t; } }
    int P = SPL[best][0], Q = SPL[best][1], R = SPL[best][2], S = SPL[best][3];
    *work = bw;
    /* largest PSD of each block's candidates, per frequency */
    int16_t mx[4][L2W]; int K = HM2 - 1;
    for (int i = 0; i < 4; i++) for (int k = 0; k < L2W; k++) { int16_t v = 0; if (k < K) for (long a = 0; a < nx[i]; a++) if (GV[i][a].psd[k] > v) v = GV[i][a].psd[k]; mx[i][k] = v; }
    __m256i lbl[2], lbr[2];
    {
        int16_t l[L2W], r[L2W];
        for (int k = 0; k < L2W; k++) {
            l[k] = (int16_t)(k < K ? 64 * N - mx[R][k] - mx[S][k] - 4 : -32000);
            r[k] = (int16_t)(k < K ? 64 * N - mx[P][k] - mx[Q][k] - 4 : -32000);
        }
        for (int w = 0; w < 2; w++) { lbl[w] = _mm256_loadu_si256((const __m256i *)l + w); lbr[w] = _mm256_loadu_si256((const __m256i *)r + w); }
    }
    /* kept pairs on each side; the table goes on the side with fewer, in as many passes as the budget needs */
    long kl = 0, kr = 0;
    for (long a = 0; a < nx[P]; a++) for (long b = 0; b < nx[Q] && (P || kmin0[a]); b++) kl += c2_ok(&GV[P][a], &GV[Q][b], lbl);
    for (long a = 0; a < nx[R]; a++) for (long b = 0; b < nx[S] && (R || kmin0[a]); b++) kr += c2_ok(&GV[R][a], &GV[S][b], lbr);
    if (kl < kr) {
        int t = P; P = R; R = t; t = Q; Q = S; S = t; long x = kl; kl = kr; kr = x;
        __m256i y = lbl[0]; lbl[0] = lbr[0]; lbr[0] = y; y = lbl[1]; lbl[1] = lbr[1]; lbr[1] = y;
    }
    int passes = (int)ceil((double)kr * sizeof(P2) * 2.2 / M2_BYTES); if (passes < 1) passes = 1;
    long ocap = 256, no = 0, npmax = 0; long *o = malloc(sizeof(long) * 4 * ocap);
    for (int pass = 0; pass < passes; pass++) {
        long cap = kr / passes + 1024, np = 0; P2 *T = malloc(sizeof(P2) * cap);
        if (!T) die("out of memory (m2 pairs)");
        for (long a = 0; a < nx[R]; a++) for (long b = 0; b < nx[S] && (R || kmin0[a]); b++) {
            if (!c2_ok(&GV[R][a], &GV[S][b], lbr)) continue;
            uint64_t h = c2_key(&GV[R][a], &GV[S][b], 1);
            if (passes > 1 && (int)((h & 0xffffffffULL) % (uint64_t)passes) != pass) continue;
            if (np == cap) { cap *= 2; T = realloc(T, sizeof(P2) * cap); if (!T) die("out of memory (m2 pairs)"); }
            T[np].h = h; T[np].c = (uint32_t)a; T[np].d = (uint32_t)b; np++;
        }
        if (np > npmax) npmax = np;
        int bb = 1; while ((1L << bb) < np) bb++;
        long nb = 1L << bb; int shift = 64 - bb;
        long *start = calloc(nb + 1, sizeof(long));
        for (long p = 0; p < np; p++) start[(T[p].h >> shift) + 1]++;
        for (long b = 0; b < nb; b++) start[b + 1] += start[b];
        P2 *SRT = malloc(sizeof(P2) * (np + 1)); long *fill = malloc(sizeof(long) * nb); memcpy(fill, start, sizeof(long) * nb);
        if (!start || !SRT || !fill) die("out of memory (m2 index)");
        for (long p = 0; p < np; p++) SRT[fill[T[p].h >> shift]++] = T[p];
        free(fill); free(T);
        for (long a = 0; a < nx[P]; a++) for (long b = 0; b < nx[Q] && (P || kmin0[a]); b++) {
            if (!c2_ok(&GV[P][a], &GV[Q][b], lbl)) continue;
            uint64_t h = c2_key(&GV[P][a], &GV[Q][b], 0);
            if (passes > 1 && (int)((h & 0xffffffffULL) % (uint64_t)passes) != pass) continue;
            long bk = (long)(h >> shift);
            for (long p = start[bk]; p < start[bk + 1]; p++) {
                if (SRT[p].h != h) continue;
                const C2 *x = &GV[P][a], *y = &GV[Q][b], *z = &GV[R][SRT[p].c], *u = &GV[S][SRT[p].d]; int same = 1;
                for (int k = 0; k < K && same; k++) if (x->paf[k] + y->paf[k] + z->paf[k] + u->paf[k] != 0) same = 0;
                if (!same) continue;
                if (no == ocap) { ocap *= 2; o = realloc(o, sizeof(long) * 4 * ocap); if (!o) die("out of memory (m2 instances)"); }
                o[no * 4 + P] = lo[P] + a; o[no * 4 + Q] = lo[Q] + b; o[no * 4 + R] = lo[R] + SRT[p].c; o[no * 4 + S] = lo[S] + SRT[p].d; no++;
            }
        }
        free(SRT); free(start);
    }
    free(kmin0); for (int i = 0; i < 4; i++) free(GV[i]);
    if (DEBUG) fprintf(stderr, "  m2 match: lists %ld %ld %ld %ld, kept pairs %ld and %ld, %d passes, %ld instances\n", nx[0], nx[1], nx[2], nx[3], kl, kr, passes, no);
    (void)npmax;
    *outp = o; return no;
}

/* ---------------- join ---------------- */
typedef struct { __m256i w[4]; } PV;
static __m256i CAPV;
static inline int pair_ok(const Row *p, const Row *q) {
    __m256i m = _mm256_setzero_si256();
    for (int i = 0; i < 4; i++) {
        __m256i a = _mm256_loadu_si256((const __m256i *)p->psd + i), b = _mm256_loadu_si256((const __m256i *)q->psd + i);
        m = _mm256_or_si256(m, _mm256_cmpgt_epi16(_mm256_add_epi16(a, b), CAPV));
    }
    return _mm256_testz_si256(m, m);
}
static inline PV paf_sum(const Row *x, const Row *y, int neg) {
    PV r;
    for (int i = 0; i < 4; i++) {
        __m256i s = _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)x->paf + i), _mm256_loadu_si256((const __m256i *)y->paf + i));
        r.w[i] = neg ? _mm256_sub_epi16(_mm256_setzero_si256(), s) : s;
    }
    return r;
}
static inline uint64_t vkey(const PV *p) {
    uint64_t h = 0x243F6A8885A308D3ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t a = (uint64_t)_mm256_extract_epi64(p->w[i], 0), b = (uint64_t)_mm256_extract_epi64(p->w[i], 1);
        uint64_t c = (uint64_t)_mm256_extract_epi64(p->w[i], 2), d = (uint64_t)_mm256_extract_epi64(p->w[i], 3);
        h ^= a * 0x9E3779B97F4A7C15ULL; h = (h << 23) | (h >> 41);
        h ^= b * 0xC2B2AE3D27D4EB4FULL; h = (h << 23) | (h >> 41);
        h ^= c * 0x165667B19E3779F9ULL; h = (h << 23) | (h >> 41);
        h ^= d * 0xD6E8FEB86659FD93ULL; h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32;
    }
    return h;
}
typedef struct { uint64_t key; uint32_t gp, x, y; } Pair;
typedef struct { RowList *P, *Q; } GPair;
typedef struct { int gp; long x0, x1; } Task;

static FILE *OUTF; static long EMITTED;
static void emit(const Row *q[4]) {
    for (int sh = 1; sh <= H; sh++) {
        int t = 0; for (int i = 0; i < 4; i++) t += N - 2 * pc(q[i]->bits ^ rot(q[i]->bits, sh));
        if (t) die("internal error: false match");
    }
#pragma omp critical(emit)
    {
        for (int i = 0; i < 4; i++) { for (int j = 0; j < N; j++) fputc(((q[i]->bits >> j) & 1) ? '+' : '-', OUTF); fputc(i == 3 ? '\n' : ' ', OUTF); }
        EMITTED++;
        if (fflush(OUTF)) die("write error on output");
    }
}

static Task *make_tasks(const GPair *gp, long ng, long *nt) {
    long cap = 1024, n = 0; Task *t = malloc(sizeof(Task) * cap);
    for (long g = 0; g < ng; g++) {
        long nx = gp[g].P->n, ny = gp[g].Q->n; if (!nx || !ny) continue;
        long step = 4096 / (ny + 1) + 1; if (step > 256) step = 256;
        for (long x = 0; x < nx; x += step) {
            if (n == cap) { cap *= 2; t = realloc(t, sizeof(Task) * cap); }
            t[n].gp = (int)g; t[n].x0 = x; t[n].x1 = x + step < nx ? x + step : nx; n++;
        }
    }
    *nt = n; return t;
}

static double ST_VIS, ST_KEPT;
/* left side LG (table), right side RG (probe); pos maps (left P, left Q, right P, right Q) to blocks */
static void join(const GPair *LG, long nl, const GPair *RG, long nr, const int pos[4]) {
    long ntl, ntr; Task *TL = make_tasks(LG, nl, &ntl), *TR = make_tasks(RG, nr, &ntr);
    double lvis = 0; for (long g = 0; g < nl; g++) lvis += (double)LG[g].P->n * LG[g].Q->n;
    int shards = 1;
    if (lvis * sizeof(Pair) > PAIR_BYTES) {
        long cnt = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : cnt)
        for (long t = 0; t < ntl; t++) {
            const GPair *g = &LG[TL[t].gp];
            for (long x = TL[t].x0; x < TL[t].x1; x++) for (long y = 0; y < g->Q->n; y++) if (pair_ok(&g->P->v[x], &g->Q->v[y])) cnt++;
        }
        ST_VIS += lvis;
        shards = (int)ceil((double)cnt * sizeof(Pair) * 1.05 / PAIR_BYTES); if (shards < 1) shards = 1;
    }
    for (int sh = 0; sh < shards; sh++) {
        long cap = 1 << 16, np = 0; Pair *pairs = malloc(sizeof(Pair) * cap);
        double kept = 0;
#pragma omp parallel reduction(+ : kept)
        {
            long lcap = 1 << 12, ln = 0; Pair *loc = malloc(sizeof(Pair) * lcap);
#pragma omp for schedule(dynamic, 1)
            for (long t = 0; t < ntl; t++) {
                const GPair *g = &LG[TL[t].gp];
                for (long x = TL[t].x0; x < TL[t].x1; x++) for (long y = 0; y < g->Q->n; y++) {
                    const Row *a = &g->P->v[x], *b = &g->Q->v[y];
                    if (!pair_ok(a, b)) continue;
                    kept++;
                    PV s = paf_sum(a, b, 0); uint64_t key = vkey(&s);
                    if (shards > 1 && (int)((key >> 40) % (uint64_t)shards) != sh) continue;
                    if (ln == lcap) { lcap *= 2; loc = realloc(loc, sizeof(Pair) * lcap); if (!loc) die("out of memory (pairs)"); }
                    loc[ln].key = key; loc[ln].gp = (uint32_t)TL[t].gp; loc[ln].x = (uint32_t)x; loc[ln].y = (uint32_t)y; ln++;
                }
            }
#pragma omp critical(pappend)
            { while (np + ln > cap) { cap *= 2; pairs = realloc(pairs, sizeof(Pair) * cap); if (!pairs) die("out of memory (pairs)"); } memcpy(pairs + np, loc, sizeof(Pair) * ln); np += ln; }
            free(loc);
        }
        ST_VIS += lvis; ST_KEPT += kept;
        if (np == 0) { free(pairs); continue; }
        int bb = 1; while ((1L << bb) < np) bb++; long nb = 1L << bb; int shift = 64 - bb;
        long *start = calloc(nb + 1, sizeof(long));
        for (long p = 0; p < np; p++) start[(pairs[p].key >> shift) + 1]++;
        for (long b = 0; b < nb; b++) start[b + 1] += start[b];
        Pair *sorted = malloc(sizeof(Pair) * np); long *fill = malloc(sizeof(long) * nb); memcpy(fill, start, sizeof(long) * nb);
        if (!sorted || !fill || !start) die("out of memory (pair index)");
        for (long p = 0; p < np; p++) sorted[fill[pairs[p].key >> shift]++] = pairs[p];
        free(fill); free(pairs);
        double rvis = 0, rkept = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : rvis, rkept)
        for (long t = 0; t < ntr; t++) {
            const GPair *g = &RG[TR[t].gp];
            for (long x = TR[t].x0; x < TR[t].x1; x++) for (long y = 0; y < g->Q->n; y++) {
                const Row *c = &g->P->v[x], *d = &g->Q->v[y];
                rvis++;
                if (!pair_ok(c, d)) continue;
                rkept++;
                PV s = paf_sum(c, d, 1); uint64_t key = vkey(&s);
                if (shards > 1 && (int)((key >> 40) % (uint64_t)shards) != sh) continue;
                int16_t need[VL] __attribute__((aligned(32)));
                for (int w = 0; w < 4; w++) _mm256_store_si256((__m256i *)need + w, s.w[w]);
                long b = (long)(key >> shift);
                for (long q = start[b]; q < start[b + 1]; q++) {
                    if (sorted[q].key != key) continue;
                    const GPair *lg = &LG[sorted[q].gp];
                    const Row *a = &lg->P->v[sorted[q].x], *bb2 = &lg->Q->v[sorted[q].y];
                    int same = 1;
                    for (int k = 0; k < H && same; k++) if (a->paf[k] + bb2->paf[k] != need[k]) same = 0;
                    if (!same) continue;
                    const Row *q4[4]; q4[pos[0]] = a; q4[pos[1]] = bb2; q4[pos[2]] = c; q4[pos[3]] = d;
                    emit(q4);
                }
            }
        }
        ST_VIS += rvis; ST_KEPT += rkept;
        free(sorted); free(start);
    }
    free(TL); free(TR);
}

/* ---------------- driver ---------------- */
typedef struct { int8_t c2[HMX]; RowList *L; } Grp;
/* sort mode: INST1 lines in the order of their length-g signature (stable) */
typedef struct { char *line; int sig[4 * (1 + HMX)]; long idx; } SL;
static int NSIG;
static int cmp_sl(const void *a, const void *b) {
    const SL *x = a, *y = b;
    for (int i = 0; i < NSIG; i++) if (x->sig[i] != y->sig[i]) return x->sig[i] < y->sig[i] ? -1 : 1;
    return x->idx < y->idx ? -1 : x->idx > y->idx;
}
static void sort_mode(const char *inp, const char *outp) {
    FILE *f = fopen(inp, "r"); if (!f) die("cannot open INST1");
    long n = 0, cap = 1024; SL *v = malloc(sizeof(SL) * cap); char line[16384];
    NSIG = 4 * (1 + G);
    while (fgets(line, sizeof line, f)) {
        int s4[4], c1[4][HMX];
        if (!parse_line(line, HM1, s4, c1, D1)) die("malformed INST1 line (wrong n or d1?)");
        if (n == cap) { cap *= 2; v = realloc(v, sizeof(SL) * cap); if (!v) die("out of memory (sort)"); }
        SL *e = &v[n]; e->line = strdup(line); e->idx = n; int k = 0;
        for (int i = 0; i < 4; i++) {
            int f1[256], gc[HMX]; full_row(c1[i], HM1, M1, KIND[i], f1); gcomp(f1, M1, gc);
            e->sig[k++] = s4[i]; for (int r = 0; r < G; r++) e->sig[k++] = gc[r];
        }
        n++;
    }
    fclose(f);
    qsort(v, n, sizeof(SL), cmp_sl);
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.part", outp);
    FILE *o = fopen(tmp, "w"); if (!o) die("cannot write sorted file");
    char stmp[4096]; snprintf(stmp, sizeof stmp, "%s.sigs.part", outp);
    FILE *so = fopen(stmp, "w"); if (!so) die("cannot write signature file");
    long ns = 0;
    for (long i = 0; i < n; i++) {
        if (!i || memcmp(v[i].sig, v[i - 1].sig, sizeof(int) * NSIG)) { ns++; fprintf(so, "%ld\n", i); }
        fputs(v[i].line, o); free(v[i].line);
    }
    if (fflush(o) || ferror(o) || fclose(o)) die("write error on sorted file");
    if (fflush(so) || ferror(so) || fclose(so)) die("write error on signature file");
    char sp[4096]; snprintf(sp, sizeof sp, "%s.sigs", outp);
    if (rename(stmp, sp) || rename(tmp, outp)) die("cannot rename sorted file");
    fprintf(stderr, "sorted %ld lines, %ld signatures (first line of each in %s.sigs)\n", n, ns, outp);
    free(v);
}
/* multipliers u of Z_n with u = 1 (mod m1) fix every length-m1 compression; their residues
   mod m2 other than 1 (KU, NKU of them) act on the length-m2 compressions of an I1's lifts,
   so only the least length-m2 quadruple of each orbit under them is needed */
static void k_units(void) {
    NKU = 0;
    for (int u = 2; u < N; u++) {
        int g = u, h = N; while (h) { int r = g % h; g = h; h = r; }
        if (g != 1 || u % M1 != 1 || u % M2 == 1) continue;
        int f = 0; for (int i = 0; i < NKU; i++) if (KU[i] == u % M2) f = 1;
        if (!f) KU[NKU++] = u % M2;
    }
}
/* 1 when the quadruple of list rows ent[0..3] is least among its images under KU */
static int k_least(CL *LL[4], const long *ent) {
    for (int q = 0; q < NKU; q++) {
        int u = KU[q], r = 0;
        for (int i = 0; i < 4 && !r; i++) {
            const int8_t *c = LL[i]->c + ent[i] * HM2;
            for (int j = 0; j < HM2 && !r; j++) {
                int k = (int)(((long)u * j) % M2);
                int v = k > HM2 - 1 ? (KIND[i] ? -c[M2 - k] : c[M2 - k]) : c[k];
                if (v != c[j]) r = v < c[j] ? -1 : 1;
            }
        }
        if (r < 0) return 0;
    }
    return 1;
}
static int cmp_long(const void *a, const void *b) { long x = *(const long *)a, y = *(const long *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    int sorting = argc == 8 && !strcmp(argv[1], "sort");
    if (sorting) { for (int i = 1; i < 7; i++) argv[i] = argv[i + 1]; argc = 7; }
    if (!sorting && argc != 8 && argc != 10) { fprintf(stderr, "usage: pc2_join sort PATTERN n d1 d2 INST1 SORTED\n       pc2_join PATTERN n d1 d2 SORTED threads OUT [first last]\n"); return 1; }
    const char *pat = argv[1];
    if (strlen(pat) != 4) die("pattern is four letters k/s, skew first");
    memcpy(PAT, pat, 5); NK = 0;
    for (int i = 0; i < 4; i++) {
        if (pat[i] != 'k' && pat[i] != 's') die("pattern letters are k and s");
        KIND[i] = pat[i] == 'k'; NK += KIND[i];
        if (i && KIND[i] > KIND[i - 1]) die("skew blocks first");
    }
    if (NK == 4) die("pattern kkkk has no symmetric block");
    N = atoi(argv[2]); D1 = atoi(argv[3]); D2 = atoi(argv[4]);
    if (N < 9 || N > 127 || N % 2 == 0) die("need odd n <= 127");
    if (D1 < 3 || D2 < 3 || D1 % 2 == 0 || D2 % 2 == 0 || N % D1 || N % D2 || N % (D1 * D2)) die("need odd d1, d2 >= 3 with d1 d2 dividing n");
    { int a = D1, b = D2; while (b) { int r = a % b; a = b; b = r; } if (a != 1) die("d1 and d2 must be coprime"); }
    M1 = N / D1; M2 = N / D2; G = N / (D1 * D2); H = (N - 1) / 2; HM1 = (M1 + 1) / 2; HM2 = (M2 + 1) / 2;
    if (M1 < 3 || M2 < 3) die("compression lengths must be at least 3");
    if (H > VL || HM1 > HMX || HM2 > HMX) die("n too large");
    MASK = (((u128)1) << N) - 1;
    if (sorting) { sort_mode(argv[5], argv[6]); return 0; }
    omp_set_num_threads(atoi(argv[6]));
    long first = argc == 10 ? atol(argv[8]) : 0, last = argc == 10 ? atol(argv[9]) : (1L << 60);
    const char *e;
    if ((e = getenv("PC_PAIR_MB")) && *e) PAIR_BYTES = atof(e) * 1048576;
    if ((e = getenv("PC_CACHE_MB")) && *e) CACHE_BYTES = atof(e) * 1048576;
    if ((e = getenv("PC_ORBIT_CAP")) && *e) ORBIT_CAP = atol(e);
    if (PAIR_BYTES < 65536) PAIR_BYTES = 65536;
    if ((e = getenv("PC_DEBUG")) && *e) DEBUG = atoi(e);
    if ((e = getenv("PC_ESTIMATE")) && *e) ESTIMATE = atoi(e);
    if ((e = getenv("PC_ORBIT_MB")) && *e) OC_BYTES = atof(e) * 1048576;
    if ((e = getenv("PC_MEM_MB")) && *e) M2_BYTES = atof(e) * 1048576;
    if (M2_BYTES < 65536) M2_BYTES = 65536;
    if (ORBIT_CAP < 1) ORBIT_CAP = 1;
    for (int s = 1; s <= H; s++) for (int k = 1; k <= H; k++) COSK[s][k] = 2.0 * cos(2 * M_PI * (double)s * k / N);
    CAPV = _mm256_set1_epi16((short)(64 * N));
    CAP2 = _mm256_set1_epi16((short)(64 * N));
    NW2 = (HM2 - 1 + 15) / 16;
    if (HM2 - 1 > L2W) die("m2 too large");
    build_orbits();
    k_units();
    {   /* length-m2 lists for every row sum a decomposition of 4n uses */
        double tb = omp_get_wtime();
        int smax = (int)floor(sqrt(4.0 * N - NK)); if (smax % 2 == 0) smax--;
        NS = (smax + 1) / 2; if (NS > NLMAX) die("too many row sums");
        int need[NLMAX + 1] = {0}, ns = 4 - NK; long ndec = 1; for (int i = 0; i < ns; i++) ndec *= NS;
        for (long code = 0; code < ndec; code++) {
            long c2 = code; int sv[4], ok = 1, sq = NK;
            for (int i = 0; i < ns; i++) { sv[i] = 2 * (int)(c2 % NS) + 1; c2 /= NS; }
            for (int i = 1; i < ns; i++) if (sv[i] > sv[i - 1]) ok = 0;
            for (int i = 0; i < ns; i++) sq += sv[i] * sv[i];
            if (!ok || sq != 4 * N) continue;
            for (int i = 0; i < ns; i++) need[(sv[i] + 1) / 2] = 1;
            if (NK) need[0] = 1;
        }
        long tot = 0, nb = 0;
        for (int i = 0; i <= NS; i++) if (need[i]) { build_l2(i == 0, i == 0 ? 1 : 2 * i - 1, &L2[i]); tot += L2[i].n; nb += L2[i].nb; }
        fprintf(stderr, "length-%d lists: %ld compressed rows in %ld length-%d classes, %.1f s\n", M2, tot, nb, G, omp_get_wtime() - tb);
    }
    const char *stopf = getenv("PC_STOP"), *progf = getenv("PC_PROGRESS");
    FILE *prog = progf && *progf ? fopen(progf, "a") : NULL;
    if (progf && *progf && !prog) die("cannot open progress file");
    FILE *in = fopen(argv[5], "r"); if (!in) die("cannot open INST1");
    OUTF = fopen(argv[7], "a"); if (!OUTF) die("cannot open output");
    int nth = atoi(argv[6]); if (nth < 1) nth = 1;
    Ctx *ctx = calloc((size_t)(nth + 1), sizeof(Ctx)); if (!ctx) die("out of memory (contexts)");
    fprintf(stderr, "pc2_join %s n=%d d1=%d d2=%d: m1=%d m2=%d g=%d, %d fibre orbits, %d multipliers fix Z_m1, lines %ld to %ld\n", PAT, N, D1, D2, M1, M2, G, NORB, NKU + 1, first, last);
    char line[16384]; long ln = -1, done = 0; double t0 = omp_get_wtime();
    double tot_vis = 0, tot_kept = 0, tot_rows = 0, t_rows = 0, t_join = 0, cpu_m2 = 0, cpu_rows = 0, cpu_join = 0;
    double est_rows_cpu = 0, est_visits = 0, est_rows = 0;
    int have_sig = 0, sig_s4[4] = {0, 0, 0, 0}, sig_gc[4][HMX]; long sig_n = 0, *sig_ent = NULL, nsig = 0;
    while (fgets(line, sizeof line, in)) {
        ln++; if (ln < first || ln > last) continue;
        if (stopf && *stopf && access(stopf, F_OK) == 0) { fprintf(stderr, "stopped by %s before line %ld\n", stopf, ln); fclose(OUTF); if (prog) fclose(prog); return 3; }
        double tl = omp_get_wtime(), c0 = cpu();
        int s4[4], c1[4][HMX];
        if (!parse_line(line, HM1, s4, c1, D1)) die("malformed INST1 line (wrong n or d1?)");
        for (int i = 0; i < 4; i++) if ((s4[i] == -1) != KIND[i]) die("INST1 line does not fit the pattern");
        int c1f[4][256], gc[4][HMX]; int8_t c1b[4][HMX];
        for (int i = 0; i < 4; i++) { full_row(c1[i], HM1, M1, KIND[i], c1f[i]); gcomp(c1f[i], M1, gc[i]); for (int j = 0; j < HM1; j++) c1b[i][j] = (int8_t)c1[i][j]; }
        /* length-m2 instances with the length-g signature of this line (kinds, row sums and the
           length-g compression of each block), computed once per run of lines with one signature */
        CL *LL[4]; long blo[4], bhi[4], nx[4];
        for (int i = 0; i < 4; i++) { LL[i] = &L2[KIND[i] ? 0 : (s4[i] + 1) / 2]; cl_bucket(LL[i], gc[i], &blo[i], &bhi[i]); nx[i] = bhi[i] - blo[i]; }
        int newsig = !have_sig;
        for (int i = 0; i < 4 && !newsig; i++) { if (s4[i] != sig_s4[i]) newsig = 1; for (int r = 0; r < G && !newsig; r++) if (gc[i][r] != sig_gc[i][r]) newsig = 1; }
        if (newsig) {
            free(sig_ent); double w2;
            sig_n = match2(LL, blo, bhi, &sig_ent, &w2);
            long kept = 0;
            for (long q = 0; q < sig_n; q++) if (k_least(LL, sig_ent + q * 4)) { for (int i = 0; i < 4; i++) sig_ent[kept * 4 + i] = sig_ent[q * 4 + i]; kept++; }
            if (DEBUG) fprintf(stderr, "  signature: %ld length-m2 quadruples, %ld least under the %d multipliers fixing Z_m1\n", sig_n, kept, NKU + 1);
            sig_n = kept;
            have_sig = 1; nsig++;
            for (int i = 0; i < 4; i++) { sig_s4[i] = s4[i]; for (int r = 0; r < G; r++) sig_gc[i][r] = gc[i][r]; }
        }
        /* the fibre margins of (c1, c2) must be feasible (Gale-Ryser on every fibre) */
        int8_t *feas[4];
        for (int i = 0; i < 4; i++) { feas[i] = malloc(nx[i] + 1); memset(feas[i], 2, nx[i] + 1); }
        long nE = 0; long *ent = malloc(sizeof(long) * 4 * (sig_n + 1));
        for (long q = 0; q < sig_n; q++) {
            int ok = 1;
            for (int i = 0; i < 4 && ok; i++) {
                long r = sig_ent[q * 4 + i]; int8_t *fz = &feas[i][r - blo[i]];
                if (*fz == 2) { int c2[HMX], f[256]; for (int j = 0; j < HM2; j++) c2[j] = LL[i]->c[r * HM2 + j]; full_row(c2, HM2, M2, KIND[i], f); *fz = (int8_t)fibres_feasible(c1f[i], f); }
                ok = *fz;
            }
            if (ok) { for (int i = 0; i < 4; i++) ent[nE * 4 + i] = sig_ent[q * 4 + i]; nE++; }
        }
        double c1t = cpu();
        /* distinct c2 per block: groups by list index */
        int ng[4] = {0, 0, 0, 0}; Grp *grp[4]; int *gid[4];
        int *eg = malloc(sizeof(int) * 4 * (nE + 1));
        for (int i = 0; i < 4; i++) {
            gid[i] = malloc(sizeof(int) * (nx[i] + 1)); for (long a = 0; a < nx[i]; a++) gid[i][a] = -1;
            for (long q = 0; q < nE; q++) { long a = ent[q * 4 + i] - blo[i]; if (gid[i][a] < 0) gid[i][a] = ng[i]++; }
            grp[i] = malloc(sizeof(Grp) * (ng[i] + 1));
            for (long a = 0; a < nx[i]; a++) if (gid[i][a] >= 0) { Grp *g = &grp[i][gid[i][a]]; memset(g, 0, sizeof *g); memcpy(g->c2, LL[i]->c + (blo[i] + a) * HM2, HM2); }
            for (long q = 0; q < nE; q++) eg[q * 4 + i] = gid[i][ent[q * 4 + i] - blo[i]];
            free(gid[i]); free(feas[i]);
        }
        free(ent);
        if (DEBUG) fprintf(stderr, "line %ld: I2 %ld, groups %d %d %d %d\n", ln, nE, ng[0], ng[1], ng[2], ng[3]);
        if (ESTIMATE > 0) {
            if (OBYTES > OC_BYTES) oc_clear();
            oc_reserve(4L * ESTIMATE * NORB);
            /* cost estimate only: row lists of ESTIMATE groups per block (spread evenly), extrapolated */
            double arows[4], acpu[4], tcpu = 0, trows = 0;
            for (int i = 0; i < 4; i++) {
                int k = ng[i] < ESTIMATE ? ng[i] : ESTIMATE; double r = 0, t = 0;
                for (int j = 0; j < k; j++) {
                    int g = (int)(((long)j * ng[i]) / k);
                    int c2[HMX], f[256]; for (int jj = 0; jj < HM2; jj++) c2[jj] = grp[i][g].c2[jj];
                    full_row(c2, HM2, M2, KIND[i], f);
                    RowList L = {0, 0, NULL}; double ct = cpu();
                    build_rows(&ctx[0], KIND[i], c1f[i], f, &L);
                    t += cpu() - ct; r += L.n; free(L.v);
                }
                arows[i] = k ? r / k : 0; acpu[i] = k ? t / k : 0;
                tcpu += acpu[i] * ng[i]; trows += arows[i] * ng[i];
            }
            static const int SPE[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
            double bestv = -1;
            for (int sp = 0; sp < 3 && nE; sp++) {
                double v = 0;
                for (int h = 0; h < 2; h++) {
                    int P = SPE[sp][2 * h], Q = SPE[sp][2 * h + 1];
                    long *key = malloc(sizeof(long) * nE);
                    for (long q = 0; q < nE; q++) key[q] = (long)eg[q * 4 + P] * (ng[Q] + 1) + eg[q * 4 + Q];
                    qsort(key, nE, sizeof(long), cmp_long);
                    long nd = 0; for (long a = 0; a < nE; a++) if (!a || key[a] != key[a - 1]) nd++;
                    free(key);
                    v += nd * arows[P] * arows[Q];
                }
                if (bestv < 0 || v < bestv) bestv = v;
            }
            if (bestv < 0) bestv = 0;
            double c3t = cpu(); cpu_m2 += c1t - c0; est_rows_cpu += tcpu; est_visits += bestv; est_rows += trows;
            fprintf(stderr, "estimate line %ld (%d,%d,%d,%d): I2 %ld, groups %d %d %d %d, rows per list %.3g %.3g %.3g %.3g, cpu per list %.3g %.3g %.3g %.3g, rows %.4g, row cpu %.4g s, pair visits %.4g, m2 cpu %.3f, sample cpu %.2f\n",
                    ln, s4[0], s4[1], s4[2], s4[3], nE, ng[0], ng[1], ng[2], ng[3], arows[0], arows[1], arows[2], arows[3], acpu[0], acpu[1], acpu[2], acpu[3], trows, tcpu, bestv, c1t - c0, c3t - c1t);
            for (int i = 0; i < 4; i++) free(grp[i]);
            free(eg); done++;
            continue;
        }
        /* row lists, from the cache or built in parallel */
        if (CBYTES > CACHE_BYTES) cache_clear();
        if (OBYTES > OC_BYTES) oc_clear();
        oc_reserve((long)(ng[0] + ng[1] + ng[2] + ng[3]) * NORB);
        while ((CN + ng[0] + ng[1] + ng[2] + ng[3] + 1) * 2 > CCAP) cache_grow();
        long nneed = 0; CEnt **need = malloc(sizeof(CEnt *) * (ng[0] + ng[1] + ng[2] + ng[3] + 1)); int *nkind = malloc(sizeof(int) * (ng[0] + ng[1] + ng[2] + ng[3] + 1));
        int (*nc2f)[256] = malloc(sizeof(int[256]) * (ng[0] + ng[1] + ng[2] + ng[3] + 1)); int *nblk = malloc(sizeof(int) * (ng[0] + ng[1] + ng[2] + ng[3] + 1));
        for (int i = 0; i < 4; i++) for (int k = 0; k < ng[i]; k++) {
            int fresh; CEnt *ce = cache_slot(KIND[i], c1b[i], grp[i][k].c2, &fresh);
            if (fresh) { int c2[HMX]; for (int j = 0; j < HM2; j++) c2[j] = grp[i][k].c2[j]; full_row(c2, HM2, M2, KIND[i], nc2f[nneed]); nblk[nneed] = i; nkind[nneed] = KIND[i]; need[nneed++] = ce; }
        }
        /* cache_slot may move entries when it grows, so the lists are attached after all inserts */
#pragma omp parallel for schedule(dynamic, 1)
        for (long q = 0; q < nneed; q++) {
            Ctx *c = &ctx[omp_get_thread_num()];
            RowList L = {0, 0, NULL};
            double tb = omp_get_wtime();
            build_rows(c, nkind[q], c1f[nblk[q]], nc2f[q], &L);
            if (DEBUG) fprintf(stderr, "  list %ld/%ld block %d: %ld rows, %ld nodes, %.3f s\n", q, nneed, nblk[q], L.n, c->nodes, omp_get_wtime() - tb);
            need[q]->L = L; need[q]->used = 1;
        }
        for (long q = 0; q < nneed; q++) CBYTES += (double)need[q]->L.n * sizeof(Row) + 256;
        for (int i = 0; i < 4; i++) for (int k = 0; k < ng[i]; k++) { int fresh; CEnt *ce = cache_slot(KIND[i], c1b[i], grp[i][k].c2, &fresh); if (fresh) die("internal error: cache miss"); grp[i][k].L = &ce->L; }
        free(need); free(nkind); free(nc2f); free(nblk);
        double rows = 0; for (int i = 0; i < 4; i++) for (int k = 0; k < ng[i]; k++) rows += grp[i][k].L->n;
        double tr = omp_get_wtime(), c2t = cpu(); t_rows += tr - tl;
        /* surviving entries and the split with the least pair work */
        long nS = 0; for (long q = 0; q < nE; q++) { int ok = 1; for (int i = 0; i < 4; i++) if (!grp[i][eg[q * 4 + i]].L->n) ok = 0; if (ok) { for (int i = 0; i < 4; i++) eg[nS * 4 + i] = eg[q * 4 + i]; nS++; } }
        static const int SPL[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
        int best = 0; double bw = -1; GPair *bl = NULL, *br = NULL; long bnl = 0, bnr = 0;
        for (int s = 0; s < 3 && nS; s++) {
            const int *sp = SPL[s];
            GPair *side[2]; long nside[2]; double w = 0;
            for (int h = 0; h < 2; h++) {
                int P = sp[2 * h], Q = sp[2 * h + 1];
                long *key = malloc(sizeof(long) * nS); long nk = 0;
                for (long q = 0; q < nS; q++) key[nk++] = (long)eg[q * 4 + P] * (ng[Q] + 1) + eg[q * 4 + Q];
                /* distinct keys */
                qsort(key, nk, sizeof(long), cmp_long);
                long nd = 0; for (long a = 0; a < nk; a++) if (!a || key[a] != key[a - 1]) key[nd++] = key[a];
                side[h] = malloc(sizeof(GPair) * (nd + 1)); nside[h] = nd;
                for (long a = 0; a < nd; a++) { side[h][a].P = grp[P][key[a] / (ng[Q] + 1)].L; side[h][a].Q = grp[Q][key[a] % (ng[Q] + 1)].L; w += (double)side[h][a].P->n * side[h][a].Q->n; }
                free(key);
            }
            if (bw < 0 || w < bw) { free(bl); free(br); bw = w; best = s; bl = side[0]; br = side[1]; bnl = nside[0]; bnr = nside[1]; }
            else { free(side[0]); free(side[1]); }
        }
        long before = EMITTED; ST_VIS = 0; ST_KEPT = 0;
        if (nS) {
            const int *sp = SPL[best];
            double wl = 0, wr = 0; for (long a = 0; a < bnl; a++) wl += (double)bl[a].P->n * bl[a].Q->n; for (long a = 0; a < bnr; a++) wr += (double)br[a].P->n * br[a].Q->n;
            int pos[4];
            if (wl <= wr) { pos[0] = sp[0]; pos[1] = sp[1]; pos[2] = sp[2]; pos[3] = sp[3]; join(bl, bnl, br, bnr, pos); }
            else { pos[0] = sp[2]; pos[1] = sp[3]; pos[2] = sp[0]; pos[3] = sp[1]; join(br, bnr, bl, bnl, pos); }
        }
        free(bl); free(br);
        t_join += omp_get_wtime() - tr;
        tot_vis += ST_VIS; tot_kept += ST_KEPT; tot_rows += rows;
        double c3t = cpu(); cpu_m2 += c1t - c0; cpu_rows += c2t - c1t; cpu_join += c3t - c2t;
        fprintf(stderr, "line %ld (%d,%d,%d,%d): signature rows %ld %ld %ld %ld, I2 %ld kept %ld, groups %d %d %d %d, rows %.0f, pair visits %.3g kept %.3g, %ld quadruples, %.3f s, cpu m2 %.3f rows %.3f join %.3f\n",
                ln, s4[0], s4[1], s4[2], s4[3], nx[0], nx[1], nx[2], nx[3], nE, nS, ng[0], ng[1], ng[2], ng[3], rows, ST_VIS, ST_KEPT, EMITTED - before, omp_get_wtime() - tl, c1t - c0, c2t - c1t, c3t - c2t);
        for (int i = 0; i < 4; i++) free(grp[i]);
        free(eg);
        if (prog) { fprintf(prog, "%ld\n", ln); if (fflush(prog)) die("write error on progress file"); }
        done++;
    }
    if (fclose(OUTF)) die("write error on output");
    if (prog) fclose(prog);
    fprintf(stderr, "signatures %ld, orbit cache hits %ld misses %ld\n", nsig, OC_HIT, OC_MISS);
    if (ESTIMATE > 0) fprintf(stderr, "estimate: %ld lines, m2 cpu %.1f s, row cpu %.4g s, rows %.4g, pair visits %.4g\n", done, cpu_m2, est_rows_cpu, est_rows, est_visits);
    {
        FILE *st = fopen("/proc/self/status", "r"); char b[256];
        if (st) { while (fgets(b, sizeof b, st)) if (!strncmp(b, "VmHWM:", 6)) fprintf(stderr, "peak memory %s", b + 6); fclose(st); }
    }
    fprintf(stderr, "join complete: %ld lines, %ld quadruples, rows %.4g, pair visits %.4g kept %.4g, %.1f s (rows %.1f join %.1f), cpu s: m2 match %.1f rows %.1f join %.1f\n",
            done, EMITTED, tot_rows, tot_vis, tot_kept, omp_get_wtime() - t0, t_rows, t_join, cpu_m2, cpu_rows, cpu_join);
    return 0;
}
