/*
 * gencensus.c -- exhaustive census of the two unconstrained tracks of the
 * Goethals-Seidel board (general and skew-sds) at small odd order n.
 *
 * A quadruple is four {-1,+1} sequences A, B, C, D of odd length n with
 *
 *     PAF_A(s) + PAF_B(s) + PAF_C(s) + PAF_D(s) = 0     for s = 1 .. n-1,
 *
 * where PAF_X(s) = sum_j X[j] X[(j+s) mod n].  Every row is normalised to
 * X[0] = +1.  The general track is every such quadruple; the skew-sds track
 * is every quadruple with at least one skew row (X[j] = -X[n-j] for j != 0).
 *
 * Equivalence is the board's equiv-v1 (verify/equivalence.py): independent
 * row negation, independent row reversal j -> -j, permutation of the four
 * rows, and a simultaneous multiplier j -> t j for a unit t of Z_n.  Cyclic
 * shifts are not identified.  Skewness of a row is invariant under all of
 * these generators, so the skew-sds stratum is a class invariant and the
 * track is decided per quadruple exactly as memberships() in gs_verify.py
 * decides it: some row is literally skew.
 *
 * Method.  PAF is invariant under negation, reversal and shift of a single
 * row, and a multiplier t permutes the PAF vector (PAF_{tX}(s) = PAF_X(t s)).
 * The engine
 *
 *   1. enumerates all 2^(n-1) sequences with X[0] = +1, keeps one of each
 *      pair {X, reverse(X)}, and groups the survivors by PAF vector.  Group g
 *      holds the list S(g) of its surviving sequences; its row sum squared
 *      is a^2 = n + 2 sum_{s=1}^{(n-1)/2} PAF(s).
 *   2. builds the multiplier action on groups and marks the groups that are
 *      minimal in their multiplier orbit.
 *   3. meets in the middle on PAF vectors: for each value m of a^2 + b^2 it
 *      sorts all pairs (i, j) of groups with i <= j, i orbit-minimal and
 *      a_i^2 + a_j^2 = m by the vector PAF_i + PAF_j, then for every pair
 *      (k, l) with k <= l and a_k^2 + a_l^2 = 4n - m looks up -(PAF_k + PAF_l)
 *      and keeps the index 4-tuples (i, j, k, l) with j <= k that are
 *      lexicographically minimal among their images under all multipliers
 *      (images are sorted before comparison).  The row-sum identity
 *      a^2 + b^2 + c^2 + d^2 = 4n is implied by the PAF identity, so the m
 *      partition loses nothing and only bounds the table size.  A sorted
 *      tuple is found exactly once, at m = a_i^2 + a_j^2.
 *   4. emits, for every accepted 4-tuple, the full product
 *      S(i) x S(j) x S(k) x S(l), one quadruple per line as four +/- words
 *      (GEN_SKEW build: only the members with at least one skew row).
 *
 * Exhaustiveness.  Let (A, B, C, D) be any solution.  Negate rows so that
 * every X[0] = +1 (PAF unchanged).  Apply a multiplier t and a row
 * permutation that carry the 4-tuple of PAF-group indices to the sorted,
 * multiplier-minimal 4-tuple of its orbit (one exists, and its first index
 * is orbit-minimal because the sorted image's first entry is the minimum of
 * the four images, so step 3 does not skip it).  Replace each row by the
 * kept member of {X, reverse(X)}; reversal commutes with multipliers and
 * preserves X[0].  The result is an equiv-v1 image of the original with
 * every row in the corresponding S(.) list, so it lies in the emitted
 * product.  Skewness is preserved by every step, so the GEN_SKEW filter
 * keeps it when the original had a skew row.  Hence every equiv-v1 class of
 * the track has a member in the output; the board re-derives the classes
 * with verify.equivalence.canonical, and duplicates in the output (members
 * of a product identified by the stabiliser of the PAF 4-tuple) are
 * harmless.  Every emitted line is an exact solution: the PAF sum is zero
 * by construction of the matched 4-tuple.
 *
 * Builds:
 *   general track:   gcc -O3 -march=native -fopenmp -o gencensus gencensus.c
 *   skew-sds track:  add -DGEN_SKEW
 *   shift-class mode (the board's equiv-v2 census of both tracks): add
 *     -DGEN_SHIFTCLASS to keep one sequence per shift/reversal/negation orbit,
 *     the least skew member when the orbit has one and the least member
 *     otherwise.  The output then covers every class of the coarser quotient
 *     equiv-v2, which also identifies independent cyclic shifts, and every
 *     class with a skew member is emitted with a literally skew row; the same
 *     exhaustiveness argument applies because shifts commute with multipliers
 *     up to a shift.  The board pins this mode as general-census-c-v1 and,
 *     with -DGEN_SKEW, as skew-sds-census-c-v1.
 *   count-only mode: add -DGEN_COUNT_ONLY to skip step 4 and leave the output
 *     file empty; the class totals below are still computed.
 * Run:  ./gencensus <n> <threads> <out.txt>       (3 <= n <= 31, n odd)
 * Environment: GEN_VERBOSE=1 prints progress to stderr; nothing in the
 * environment changes the output.
 *
 * Besides the lines, the engine reports on stderr the exact number of
 * classes in the emitted set (classes_all) and in its skew-sds subset
 * (classes_skew), computed by Burnside's lemma over the stabiliser of each
 * accepted PAF 4-tuple (see orbit_count).  These totals equal what the board
 * module derives from the lines; they were checked against it at every order
 * where the lines could be written, and they are what the count-only build
 * reports when the product is too large to write.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAXN 31
#define MAXH 15

static int N, H;
static uint32_t MASK;
static int verbose;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void die(const char *msg) {
    fprintf(stderr, "gencensus: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) die("out of memory");
    return p;
}

/* A sequence is a bitmask: bit j set iff X[j] = -1; bit 0 is clear. */
static inline uint32_t rotl(uint32_t b, int s) {
    return s ? (((b << s) | (b >> (N - s))) & MASK) : b;
}

static inline int paf_at(uint32_t b, int s) {
    return N - 2 * __builtin_popcount(b ^ rotl(b, s));
}

static uint32_t rev(uint32_t b) {
    uint32_t r = 0;
    for (int j = 1; j < N; j++)
        if (b >> j & 1) r |= 1u << (N - j);
    return r;
}

/* Negate so that X[0] = +1. */
static inline uint32_t norm(uint32_t b) { return (b & 1) ? (b ^ MASK) : b; }

static int is_skew(uint32_t b) { return (rev(b) ^ b) == (MASK ^ 1u); }

/* Multiplier t on a bitmask: image[j] = X[t j mod n], as the board module. */
static uint32_t mul_code(uint32_t b, int t) {
    uint32_t r = 0;
    for (int j = 1; j < N; j++)
        if (b >> ((int)((long)t * j % N)) & 1) r |= 1u << j;
    return r;
}

#ifdef GEN_SHIFTCLASS
/* Does the shift/reversal/negation orbit of b contain a skew sequence? */
static int row_skew(uint32_t b) {
    uint32_t rb = rev(b);
    for (int s = 0; s < N; s++)
        if (is_skew(norm(rotl(b, s))) || is_skew(norm(rotl(rb, s)))) return 1;
    return 0;
}
/* The representative of the shift/reversal/negation orbit of b: the least
   skew member when the orbit has one (so a skew row is emitted literally
   skew), else the least member.  Either rule is constant on orbits, which is
   all the enumeration and the Burnside count need. */
static uint32_t row_min(uint32_t b) {
    uint32_t best = b, best_skew = 0, rb = rev(b);
    int have_skew = 0;
    for (int s = 0; s < N; s++) {
        uint32_t u = norm(rotl(b, s)), v = norm(rotl(rb, s));
        if (u < best) best = u;
        if (v < best) best = v;
        if (is_skew(u) && (!have_skew || u < best_skew)) { best_skew = u; have_skew = 1; }
        if (is_skew(v) && (!have_skew || v < best_skew)) { best_skew = v; have_skew = 1; }
    }
    return have_skew ? best_skew : best;
}
#else
/* The kept member of {X, reverse(X)}. */
static uint32_t row_min(uint32_t b) {
    uint32_t rb = rev(b);
    return rb < b ? rb : b;
}
static int row_skew(uint32_t b) { return is_skew(b); }
#endif

typedef struct {
    int8_t pv[MAXH];
    uint8_t skew;
    uint32_t code;
} seq_t;

static int cmp_seq(const void *x, const void *y) {
    const seq_t *a = x, *b = y;
    int c = memcmp(a->pv, b->pv, MAXH);
    if (c) return c;
    return (a->code > b->code) - (a->code < b->code);
}

static int cmp_skew_first(const void *x, const void *y) {
    const seq_t *a = x, *b = y;
    if (a->skew != b->skew) return (int)b->skew - (int)a->skew;
    return (a->code > b->code) - (a->code < b->code);
}

typedef struct {
    int8_t pv[MAXH];
    int a2;          /* row sum squared */
    uint32_t start;  /* range in the sorted sequence table */
    uint32_t count;
    uint32_t nskew;  /* skew members are sorted to the front of the range */
    uint8_t rep;     /* minimal in its multiplier orbit */
} group_t;

static seq_t *seqs;
static uint32_t nseq;
static group_t *groups;
static uint32_t ngroups;
static uint32_t *mult;   /* mult[u * ngroups + g] */
static int nunits;
static int units[MAXN], unit_index[MAXN];

static uint32_t find_group(const int8_t *pv) {
    uint32_t lo = 0, hi = ngroups;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        int c = memcmp(groups[mid].pv, pv, MAXH);
        if (c < 0) lo = mid + 1;
        else if (c > 0) hi = mid;
        else return mid;
    }
    die("PAF image not found: the sequence set is closed under multipliers");
    return 0;
}

static void build_tables(void) {
    uint32_t total = 1u << (N - 1);
    seqs = xmalloc((size_t)total * sizeof(seq_t));
    nseq = 0;
    for (uint32_t k = 0; k < total; k++) {
        uint32_t b = k << 1;
        if (row_min(b) != b) continue;
        seq_t *s = &seqs[nseq++];
        memset(s->pv, 0, MAXH);
        for (int t = 1; t <= H; t++) s->pv[t - 1] = (int8_t)paf_at(b, t);
        s->skew = (uint8_t)row_skew(b);
        s->code = b;
    }
    qsort(seqs, nseq, sizeof(seq_t), cmp_seq);
    ngroups = 0;
    for (uint32_t i = 0; i < nseq; i++)
        if (i == 0 || memcmp(seqs[i].pv, seqs[i - 1].pv, MAXH)) ngroups++;
    groups = xmalloc((size_t)ngroups * sizeof(group_t));
    uint32_t g = 0;
    for (uint32_t i = 0; i < nseq; i++) {
        if (i == 0 || memcmp(seqs[i].pv, seqs[i - 1].pv, MAXH)) {
            group_t *G = &groups[g++];
            memcpy(G->pv, seqs[i].pv, MAXH);
            G->start = i;
            G->count = 0;
            int sum = 0;
            for (int t = 0; t < H; t++) sum += G->pv[t];
            G->a2 = N + 2 * sum;
        }
        groups[g - 1].count++;
    }
    for (g = 0; g < ngroups; g++) {
        group_t *G = &groups[g];
        qsort(seqs + G->start, G->count, sizeof(seq_t), cmp_skew_first);
        G->nskew = 0;
        for (uint32_t i = 0; i < G->count; i++) G->nskew += seqs[G->start + i].skew;
    }
    /* Multiplier action: image of pv under t is pv'(s) = pv(fold(t s)). */
    nunits = 0;
    for (int t = 1; t < N; t++) {
        int a = t, b = N;
        while (b) { int r = a % b; a = b; b = r; }
        if (a == 1) { unit_index[t] = nunits; units[nunits++] = t; }
    }
    mult = xmalloc((size_t)nunits * ngroups * sizeof(uint32_t));
    for (int u = 0; u < nunits; u++) {
        int t = units[u];
        for (g = 0; g < ngroups; g++) {
            int8_t pv[MAXH];
            memset(pv, 0, MAXH);
            for (int s = 1; s <= H; s++) {
                int w = (int)((long)t * s % N);
                if (w > H) w = N - w;
                pv[s - 1] = groups[g].pv[w - 1];
            }
            mult[(size_t)u * ngroups + g] = find_group(pv);
        }
    }
    for (g = 0; g < ngroups; g++) {
        groups[g].rep = 1;
        for (int u = 0; u < nunits; u++)
            if (mult[(size_t)u * ngroups + g] < g) { groups[g].rep = 0; break; }
    }
}

/* Pair table for one value of a^2 + b^2. */
typedef struct {
    uint8_t key[16];
    uint32_t i, j;
} pair_t;

static int cmp_pair(const void *x, const void *y) {
    const pair_t *a = x, *b = y;
    int c = memcmp(a->key, b->key, 16);
    if (c) return c;
    if (a->i != b->i) return (a->i > b->i) - (a->i < b->i);
    return (a->j > b->j) - (a->j < b->j);
}

static inline void sum_key(const group_t *a, const group_t *b, int sign, uint8_t *key) {
    memset(key, 0, 16);
    for (int t = 0; t < H; t++) key[t] = (uint8_t)(int8_t)(sign * (a->pv[t] + b->pv[t]));
}

typedef struct { uint32_t g[4]; } ptuple_t;

typedef struct {
    ptuple_t *v;
    size_t n, cap;
} quadvec_t;

static void push_quad(quadvec_t *q, const uint32_t *g) {
    if (q->n == q->cap) {
        q->cap = q->cap ? 2 * q->cap : 1024;
        q->v = realloc(q->v, q->cap * sizeof(ptuple_t));
        if (!q->v) die("out of memory");
    }
    memcpy(q->v[q->n++].g, g, 4 * sizeof(uint32_t));
}

static int cmp_quad(const void *x, const void *y) {
    const uint32_t *a = ((const ptuple_t *)x)->g, *b = ((const ptuple_t *)y)->g;
    for (int r = 0; r < 4; r++)
        if (a[r] != b[r]) return (a[r] > b[r]) - (a[r] < b[r]);
    return 0;
}

static void sort4(uint32_t *x) {
    for (int a = 1; a < 4; a++)
        for (int b = a; b > 0 && x[b - 1] > x[b]; b--) {
            uint32_t t = x[b]; x[b] = x[b - 1]; x[b - 1] = t;
        }
}

/* Is the sorted tuple x minimal among the sorted images under all units? */
static int orbit_minimal(const uint32_t *x) {
    for (int u = 0; u < nunits; u++) {
        uint32_t y[4];
        for (int r = 0; r < 4; r++) y[r] = mult[(size_t)u * ngroups + x[r]];
        sort4(y);
        for (int r = 0; r < 4; r++) {
            if (y[r] < x[r]) return 0;
            if (y[r] > x[r]) break;
        }
    }
    return 1;
}

/* Groups bucketed by a^2. */
static uint32_t *by_a2;
static int *a2_vals;
static uint32_t *a2_start;
static int nvals;

static void bucket_by_a2(void) {
    int maxa2 = N * N;
    uint32_t *cnt = xmalloc((size_t)(maxa2 + 1) * sizeof(uint32_t));
    memset(cnt, 0, (size_t)(maxa2 + 1) * sizeof(uint32_t));
    for (uint32_t g = 0; g < ngroups; g++) cnt[groups[g].a2]++;
    uint32_t acc = 0;
    for (int v = 0; v <= maxa2; v++) { uint32_t c = cnt[v]; cnt[v] = acc; acc += c; }
    by_a2 = xmalloc(ngroups * sizeof(uint32_t));
    for (uint32_t g = 0; g < ngroups; g++) by_a2[cnt[groups[g].a2]++] = g;
    free(cnt);
    a2_vals = xmalloc(ngroups * sizeof(int));
    a2_start = xmalloc((ngroups + 1) * sizeof(uint32_t));
    nvals = 0;
    for (uint32_t p = 0; p < ngroups; p++)
        if (p == 0 || groups[by_a2[p]].a2 != groups[by_a2[p - 1]].a2) {
            a2_vals[nvals] = groups[by_a2[p]].a2;
            a2_start[nvals++] = p;
        }
    a2_start[nvals] = ngroups;
}

/* Collect pairs (i, j), i <= j, a_i^2 + a_j^2 = m, optionally i orbit-minimal.
   Two passes: count, then fill. */
static size_t collect_pairs(int m, int reps_only, uint32_t **pi, uint32_t **pj) {
    size_t c = 0;
    for (int pass = 0; pass < 2; pass++) {
        c = 0;
        for (int xa = 0; xa < nvals; xa++)
            for (int xb = 0; xb < nvals; xb++) {
                if (a2_vals[xa] + a2_vals[xb] != m) continue;
                for (uint32_t p = a2_start[xa]; p < a2_start[xa + 1]; p++) {
                    uint32_t i = by_a2[p];
                    if (reps_only && !groups[i].rep) continue;
                    for (uint32_t q = a2_start[xb]; q < a2_start[xb + 1]; q++) {
                        uint32_t j = by_a2[q];
                        if (j < i) continue;
                        if (pass) { (*pi)[c] = i; (*pj)[c] = j; }
                        c++;
                    }
                }
            }
        if (!pass) {
            *pi = xmalloc((c ? c : 1) * sizeof(uint32_t));
            *pj = xmalloc((c ? c : 1) * sizeof(uint32_t));
        }
    }
    return c;
}

/* Enumerate the accepted PAF 4-tuples into result. */
static void match(quadvec_t *result, int threads) {
    bucket_by_a2();
    quadvec_t *local = xmalloc((size_t)threads * sizeof(quadvec_t));
    memset(local, 0, (size_t)threads * sizeof(quadvec_t));
    size_t npairs_total = 0, nqueries_total = 0, maxtable = 0;

    /* Distinct m = a^2 + b^2 with 4n - m also of that form. */
    int maxm = 4 * N;
    uint8_t *has = xmalloc((size_t)maxm + 1);
    memset(has, 0, (size_t)maxm + 1);
    for (int va = 0; va < nvals; va++)
        for (int vb = va; vb < nvals; vb++) {
            int m = a2_vals[va] + a2_vals[vb];
            if (m <= maxm) has[m] = 1;
        }
    for (int m = 0; m <= maxm; m++) {
        if (!has[m] || !has[maxm - m]) continue;
        uint32_t *ti, *tj;
        size_t npairs = collect_pairs(m, 1, &ti, &tj);
        pair_t *table = xmalloc((npairs ? npairs : 1) * sizeof(pair_t));
        for (size_t w = 0; w < npairs; w++) {
            sum_key(&groups[ti[w]], &groups[tj[w]], 1, table[w].key);
            table[w].i = ti[w];
            table[w].j = tj[w];
        }
        free(ti);
        free(tj);
        qsort(table, npairs, sizeof(pair_t), cmp_pair);
        npairs_total += npairs;
        if (npairs > maxtable) maxtable = npairs;

        uint32_t *qk, *ql;
        size_t nq = collect_pairs(maxm - m, 0, &qk, &ql);
        nqueries_total += nq;
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads)
        for (size_t qi = 0; qi < nq; qi++) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            uint32_t k = qk[qi], l = ql[qi];
            uint8_t key[16];
            sum_key(&groups[k], &groups[l], -1, key);
            size_t lo = 0, hi = npairs;
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (memcmp(table[mid].key, key, 16) < 0) lo = mid + 1;
                else hi = mid;
            }
            for (size_t r = lo; r < npairs && !memcmp(table[r].key, key, 16); r++) {
                uint32_t i = table[r].i, j = table[r].j;
                if (j > k) continue;
                uint32_t x[4] = {i, j, k, l};
                if (orbit_minimal(x)) push_quad(&local[tid], x);
            }
        }
        free(qk);
        free(ql);
        free(table);
        if (verbose) {
            size_t acc = 0;
            for (int t = 0; t < threads; t++) acc += local[t].n;
            fprintf(stderr, "  m=%d: table %zu, queries %zu, accepted so far %zu\n",
                    m, npairs, nq, acc);
        }
    }
    free(has);
    for (int t = 0; t < threads; t++) {
        for (size_t r = 0; r < local[t].n; r++) push_quad(result, local[t].v[r].g);
        free(local[t].v);
    }
    free(local);
    qsort(result->v, result->n, sizeof(ptuple_t), cmp_quad);
    if (verbose)
        fprintf(stderr, "pairs tabled %zu (largest table %zu), queries %zu, accepted PAF 4-tuples %zu\n",
                npairs_total, maxtable, nqueries_total, result->n);
}

/*
 * Exact class count by Burnside, independent of the emitted lines.  Two
 * quadruples of an emitted product are equivalent iff some (pi, t) in the
 * stabiliser of the PAF 4-tuple (t p_r = p_{pi(r)} for all r) carries one to
 * the other after the per-row normalisation row_min (reversal in the board
 * modes, the full dihedral orbit in GEN_SHIFTCLASS).  Products of distinct
 * accepted 4-tuples never share a class.  A quadruple is fixed by (pi, t)
 * iff along every cycle of pi of length k starting at row r the sequence
 * X_r satisfies row_min(t^k X_r) = X_r, so |Fix| is a product over cycles
 * of per-group counts.  Rows with a skew member form an invariant subset, so
 * the skew-sds count is the count over all minus the count over non-skew
 * rows only.
 */
static int32_t *fixcnt, *fixcnt_ns;   /* [u * ngroups + g] */
static int perms[24][4];

static void prepare_burnside(int threads) {
    int c = 0;
    for (int a = 0; a < 4; a++)
        for (int b = 0; b < 4; b++)
            for (int d = 0; d < 4; d++)
                for (int e = 0; e < 4; e++)
                    if (a != b && a != d && a != e && b != d && b != e && d != e) {
                        perms[c][0] = a; perms[c][1] = b; perms[c][2] = d; perms[c][3] = e;
                        c++;
                    }
    fixcnt = xmalloc((size_t)nunits * ngroups * sizeof(int32_t));
    fixcnt_ns = xmalloc((size_t)nunits * ngroups * sizeof(int32_t));
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
    for (uint32_t g = 0; g < ngroups; g++)
        for (int u = 0; u < nunits; u++) {
            size_t idx = (size_t)u * ngroups + g;
            fixcnt[idx] = fixcnt_ns[idx] = 0;
            if (mult[idx] != g) continue;
            const group_t *G = &groups[g];
            for (uint32_t i = 0; i < G->count; i++) {
                uint32_t code = seqs[G->start + i].code;
                if (row_min(mul_code(code, units[u])) == code) {
                    fixcnt[idx]++;
                    if (!seqs[G->start + i].skew) fixcnt_ns[idx]++;
                }
            }
        }
}

static void orbit_count(const uint32_t *x, uint64_t *all, uint64_t *skew) {
    unsigned __int128 sum_all = 0, sum_ns = 0;
    int stab = 0;
    for (int p = 0; p < 24; p++)
        for (int u = 0; u < nunits; u++) {
            int ok = 1;
            for (int r = 0; r < 4 && ok; r++)
                if (mult[(size_t)u * ngroups + x[r]] != x[perms[p][r]]) ok = 0;
            if (!ok) continue;
            stab++;
            unsigned __int128 fa = 1, fns = 1;
            int seen[4] = {0, 0, 0, 0};
            for (int r = 0; r < 4; r++) {
                if (seen[r]) continue;
                int q = r;
                long tk = 1;
                do { seen[q] = 1; q = perms[p][q]; tk = tk * units[u] % N; } while (q != r);
                size_t idx = (size_t)unit_index[tk] * ngroups + x[r];
                fa *= (unsigned)fixcnt[idx];
                fns *= (unsigned)fixcnt_ns[idx];
            }
            sum_all += fa;
            sum_ns += fns;
        }
    if (sum_all % stab || sum_ns % stab) die("Burnside sum not divisible by the stabiliser order");
    *all = (uint64_t)(sum_all / stab);
    *skew = (uint64_t)(sum_all / stab - sum_ns / stab);
}

/* Output. */
typedef struct {
    char *buf;
    size_t len, cap;
} outbuf_t;

static FILE *outfile;
static size_t emitted;

static void flush_buf(outbuf_t *o) {
#pragma omp critical(output)
    {
        if (o->len && fwrite(o->buf, 1, o->len, outfile) != o->len) die("write failed");
    }
    o->len = 0;
}

static inline void emit(outbuf_t *o, const uint32_t *codes) {
    size_t need = 4 * (size_t)N + 4;
    if (o->len + need > o->cap) flush_buf(o);
    char *p = o->buf + o->len;
    for (int r = 0; r < 4; r++) {
        uint32_t c = codes[r];
        for (int j = 0; j < N; j++) *p++ = (c >> j & 1) ? '-' : '+';
        *p++ = r == 3 ? '\n' : ' ';
    }
    o->len = (size_t)(p - o->buf);
}

/* Emit the product of the four groups (GEN_SKEW: members with a skew row,
   partitioned by the first skew position so each member appears once). */
static size_t emit_product(outbuf_t *o, const uint32_t *g) {
    const group_t *G[4];
    for (int r = 0; r < 4; r++) G[r] = &groups[g[r]];
    size_t count = 0;
    uint32_t codes[4];
#ifdef GEN_SKEW
    for (int first = 0; first < 4; first++) {
        if (!G[first]->nskew) continue;
        /* rows before `first` are non-skew, row `first` is skew, rest are any */
        uint32_t lo[4], hi[4];
        for (int r = 0; r < 4; r++) {
            lo[r] = r < first ? G[r]->nskew : 0;
            hi[r] = r == first ? G[r]->nskew : G[r]->count;
            if (r < first && lo[r] >= hi[r]) goto next_first;
        }
        for (uint32_t a = lo[0]; a < hi[0]; a++) {
            codes[0] = seqs[G[0]->start + a].code;
            for (uint32_t b = lo[1]; b < hi[1]; b++) {
                codes[1] = seqs[G[1]->start + b].code;
                for (uint32_t c = lo[2]; c < hi[2]; c++) {
                    codes[2] = seqs[G[2]->start + c].code;
                    for (uint32_t d = lo[3]; d < hi[3]; d++) {
                        codes[3] = seqs[G[3]->start + d].code;
                        emit(o, codes);
                        count++;
                    }
                }
            }
        }
    next_first:;
    }
#else
    for (uint32_t a = 0; a < G[0]->count; a++) {
        codes[0] = seqs[G[0]->start + a].code;
        for (uint32_t b = 0; b < G[1]->count; b++) {
            codes[1] = seqs[G[1]->start + b].code;
            for (uint32_t c = 0; c < G[2]->count; c++) {
                codes[2] = seqs[G[2]->start + c].code;
                for (uint32_t d = 0; d < G[3]->count; d++) {
                    codes[3] = seqs[G[3]->start + d].code;
                    emit(o, codes);
                    count++;
                }
            }
        }
    }
#endif
    return count;
}

static long peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <n> <threads> <out.txt>\n", argv[0]);
        return 2;
    }
    N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    if (N < 3 || N > MAXN || N % 2 == 0) die("n must be odd, 3 <= n <= 31");
    if (threads < 1) threads = 1;
    H = (N - 1) / 2;
    MASK = (N == 32) ? 0xffffffffu : ((1u << N) - 1);
    verbose = getenv("GEN_VERBOSE") != NULL;
    double t0 = now();

    build_tables();
    double t1 = now();
    if (verbose) {
        uint32_t nskew = 0;
        for (uint32_t g = 0; g < ngroups; g++) nskew += groups[g].nskew;
        fprintf(stderr, "n=%d: %u sequences kept, %u PAF groups, %u skew sequences, %d units, %.2fs\n",
                N, nseq, ngroups, nskew, nunits, t1 - t0);
    }

    quadvec_t found = {0, 0, 0};
    match(&found, threads);
    double t2 = now();

    prepare_burnside(threads);
    uint64_t classes_all = 0, classes_skew = 0;
    outfile = fopen(argv[3], "w");
    if (!outfile) die("cannot open output");
#pragma omp parallel num_threads(threads) reduction(+ : emitted, classes_all, classes_skew)
    {
        outbuf_t o;
        o.cap = 1 << 20;
        o.buf = xmalloc(o.cap);
        o.len = 0;
#pragma omp for schedule(dynamic, 16)
        for (size_t r = 0; r < found.n; r++) {
            uint64_t a, s;
            orbit_count(found.v[r].g, &a, &s);
            classes_all += a;
            classes_skew += s;
#ifndef GEN_COUNT_ONLY
            emitted += emit_product(&o, found.v[r].g);
#endif
        }
        flush_buf(&o);
        free(o.buf);
    }
    if (fclose(outfile)) die("close failed");
    double t3 = now();
    fprintf(stderr,
            "n=%d groups=%u paf4tuples=%zu quadruples=%zu classes_all=%llu classes_skew=%llu "
            "tables=%.2fs match=%.2fs emit=%.2fs total=%.2fs peak_rss=%ldkB\n",
            N, ngroups, found.n, emitted, (unsigned long long)classes_all,
            (unsigned long long)classes_skew, t1 - t0, t2 - t1, t3 - t2, t3 - t0, peak_rss_kb());
    free(found.v);
    return 0;
}
