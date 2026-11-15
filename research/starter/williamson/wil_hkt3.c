/* wil_hkt3.c -- exhaustive census of Williamson matrices of odd order n by the
 * mod-4 linearisation of Holzmann, Kharaghani and Tayfeh-Rezaie (2008).
 *
 * A Williamson quadruple is four symmetric (-1,1) sequences A, B, C, D of
 * length n (x_j = x_{n-j}) with PAF_A + PAF_B + PAF_C + PAF_D = 0 at every
 * shift 1..(n-1)/2.  The row sums satisfy a^2 + b^2 + c^2 + d^2 = 4n with all
 * four odd.  Every row is normalised to x_0 = +1, which fixes the sign of its
 * row sum (s = n mod 4), so the positive decomposition a >= b >= c >= d maps
 * to one signed decomposition and one popcount per row.
 *
 * Enumeration, unchanged from wil_hkt.c: A runs over the symmetric codes of
 * popcount p_a that are least in their orbit under the phi(n)/2 folded
 * decimations and pass the PSD ceiling; B runs over a table of the symmetric
 * codes of popcount p_b that pass the ceiling, filtered per A by the pointwise
 * bound PSD_A + PSD_B <= 4n through an index of H bitmaps; each surviving
 * pair goes to the two-stage GF(2) solve of hkt_solve64.h for C and D, and
 * each candidate is checked exactly.  For a = b the table is entered at
 * code(A); for c = d codeC <= codeD keeps one of each swapped pair.
 *
 * Revision of 2026-11-14 (wil_hkt3).  wil_hkt2.c on the widened solver
 * hkt_solve64.h, so the order limit moves from n <= 63 to n <= 127, with the
 * B row shrunk for the orders that are now reachable.  Nothing in the method
 * changed; what changed is the word width and the row layout:
 *
 *   H-bit words   codes, patterns t, rhs1, X, the stage-2 columns, bb, the
 *                 pivot masks and the candidates are hw_t: uint32_t in the
 *                 default build (H <= 32), uint64_t under -DHKT_WIDE
 *                 (H <= 64).  The decimation byte tables, the t map, the B
 *                 records and the solution quadruples follow the type.
 *   N-bit words   the unfolded pattern word uw, the row masks of sym_mask /
 *                 paf_mask and of the exact check are nw_t: uint64_t by
 *                 default (n <= 63), unsigned __int128 under -DHKT_WIDE
 *                 (n <= 127).  The four-pattern AVX2 column build and the mc
 *                 bits hold uw as two 64-bit lanes (lo, hi) per pattern and
 *                 read 64-bit windows of it at uniform offsets; the scalar
 *                 forms use the 128-bit type directly.  Both are checked
 *                 against the reference row build at startup as before.
 *   lanes         the batched elimination has HKT_LANES lanes, 8 narrow and
 *                 4 wide; the flush queue, ct, the pivot block and the tail
 *                 are sized from it.
 *   B row         the table no longer stores the PAF of each B row (H bytes
 *                 per row, read only by the exact check of a candidate); the
 *                 exact check forms PAF_B(k) = n - 2 popcount(w ^ rot(w, k))
 *                 from the code, the same form it already uses for C and D,
 *                 so one more popcount per lag tested.  The record of a row
 *                 is its three solver words with no padding: 12 bytes narrow,
 *                 24 wide.  At n = 67 this takes the largest B table from
 *                 7.2 GB to 3.3 GB.
 *
 * The default build is wil_hkt2.c with the types spelled through typedefs:
 * same instructions in the hot loops, same class sets.  The wide build is
 * what runs at n >= 65 (n = 65 already has N = 65 > 63, with H = 32) and is
 * tested by forcing it at small orders, where its class sets must agree with
 * the narrow build.
 *
 * The 2^H consistency table of wil_hkt2 is unchanged and stays off above
 * H = 30 (it would be 16 GB at n = 65 and does not fit beyond); without it
 * every survivor of the pair filter goes to the elimination, as at n = 61 in
 * wil_hkt2.
 *
 * Per pair, the work (wil_hkt2.c, revision of 2026-11-14):
 *
 *   consistency test   the stage-2 matrix depends only on the sign pattern t,
 *                      and so does its left null space.  The system for a
 *                      pair is consistent exactly when its right-hand side bb
 *                      is orthogonal to every left null vector.  A table of
 *                      2^H entries holds those vectors (K words each); a pair
 *                      that fails a parity against them is dropped before any
 *                      elimination.  At the prime orders 47..61 every pattern
 *                      has corank exactly 1 (sampled), so K = 1 and half the
 *                      pairs are dropped; composite orders have corank 3 and
 *                      drop about 87%.
 *   folded right side  bb = G(t) XOR X with G a function of t alone and
 *                      X = WA1 ^ WB1 ^ (TA0 & ~D0) from two words of the A
 *                      row, one word of the B row and rhs1 = RA ^ RB (the
 *                      derivation is at nw_fold below).
 *   direct columns     the stage-2 matrix of a pair that passes is built by
 *                      columns in closed form straight into the transposed
 *                      layout of the lane-vectorised elimination (nw_cols4).
 *   elimination        wil_elim is hkt_elim8v with the columns already
 *                      pivoted in every lane skipped.
 *   tail               the particular solution and the null vector are read
 *                      off the reduced system for all lanes at once
 *                      (wil_tail); only a lane with two or more free columns
 *                      falls back to the scalar hkt_tail_m.
 *   AND chain          the pair filter intersects four index words per AVX2
 *                      register.
 *   B record           the three words a pair reads from its B row share one
 *                      record (16 bytes narrow, 32 wide), prefetched a chunk
 *                      ahead; the table word is prefetched a chunk ahead too.
 *
 * Memory at the large orders is the B table and its index, not the solver:
 * per kept B row, sizeof(hw_t) + sizeof(BRec) bytes (32 wide), plus the
 * index of H x NLEV bitmaps over the rows (H NLEV n_B / 8 bytes), plus one
 * int per row per thread for the survivor list, which is virtual: only the
 * entries a row's survivors touch become resident.  WIL_ENUM_ONLY=1 builds
 * nothing but the counts: canonical A rows and kept B rows per group, with
 * the index size at several NLEV values, and exits.
 *
 * The stage-2 nullity cap of the solver (nullity > 6) is counted rather than
 * assumed: the run prints how many solve calls were dropped by it, and the
 * census is exhaustive only when that count is zero.
 *
 *   build: gcc -O3 -march=native -fopenmp -DHKT_WIDE -o wil_hkt3 wil_hkt3.c -lm
 *          (without -DHKT_WIDE the build is the 32-bit path, n <= 63)
 *   run:   ./wil_hkt3 n [threads] [out]
 *   env:   WIL_LIST=1    list the (a, b) groups and exit
 *          WIL_ENUM_ONLY=1  count canonical A and kept B per group, print
 *                        the index size at several NLEV values, and exit
 *          WIL_DONLY=i   run group i only
 *          WIL_CCK=k     words per consistency-table entry (0 = no table;
 *                        default chosen from a sample of patterns; off
 *                        above H = 30 whatever the setting); WIL_NOCACHE=1
 *                        is the older spelling of WIL_CCK=0
 *          WIL_GTAB=0/1  store G(t) in the table (default on up to H = 26)
 *          WIL_AB=ij     which two of the sorted sums are enumerated
 *                        (default 01, the two largest)
 *          WIL_NLEV=L    PSD quantisation levels of the B index (default 256;
 *                        coarser is looser, never lossy)
 *          WIL_SAMPLE=k  solve every k-th chunk of A codes only and report
 *                        the sampled pair count and solve time per pair
 *
 * Output: one line per distinct quadruple found, four words of n '+'/'-'
 * characters (A B C D), the format wil_hkt.c writes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(__AVX2__)
#error "wil_hkt3.c needs AVX2 (build with -march=native on an AVX2 host)"
#endif

static int N, H;
#define SCALE 16
#define CEIL16 (4 * N * SCALE)

#include "hkt_solve64.h"

#define LANES HKT_LANES

static double wnow(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double tcpun(void) {
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ------------------------------------------------- per-row prefix halves

   PAF(k) is congruent to n mod 4 for every lag of a (-1,1) row of odd length,
   so PAF_B(k) = n - 4 u_k with u_k >= 0.  With A_k = -PAF_A(k) - n - 2 FIXED_[k]
   (even, depends only on A) the solver's working value is v = A_k + 4 u_k, its
   stage-1 right-hand side bit is bit 1 of (v >> 1) - 2 M1CONST_[k], which is
   bit2(A_k) ^ bit0(M1CONST_[k]) ^ bit0(u_k), so rhs1 = RA ^ RB with one word
   per row, and the stage-1 solution is t = T(rhs1) = T(RA) ^ T(RB) since T is
   linear.  The stage-2 target is ta_k + u_k with ta_k = A_k >> 2. */
static inline hw_t nw_tmap(hw_t r) {
    hw_t t = 0;
    for (int b = 0; b < TNB_; b++)
        t ^= TBYTE_[(size_t)b * 256 + ((r >> (8 * b)) & 0xffu)];
    return t;
}
static inline void nw_arow(const int8_t *pafA, int *ta, hw_t *raraw) {
    hw_t r = 0;
    for (int k = 1; k <= H; k++) {
        int ak = -(int)pafA[k - 1] - N - 2 * FIXED_[k];
        ta[k] = ak >> 2;
        if ((((ak >> 1) - 2 * M1CONST_[k]) >> 1) & 1) r |= HW_ONE << (k - 1);
    }
    *raraw = r;
}
/* Reference right-hand side of stage 2 (wil_hkt.c's nw_bvec): bit 1 of
   mc_k - ta_k - u_k at each lag. */
static inline hw_t nw_bvec_ref(const int16_t *mc, const int *ta, const int8_t *su) {
    hw_t bvec = 0;
    for (int kb = 1; kb <= H; kb++)
        if (((mc[kb - 1] - ta[kb] - su[kb - 1]) >> 1) & 1) bvec |= HW_ONE << (kb - 1);
    return bvec;
}

/* ------------------------------------------------- lane-vectorised elimination

   hkt_elim8v's construction (columns transposed, ct[c*LANES + lane], one-hot
   pivot masks, dead lanes carry zero) with one change: a column that has
   taken its pivot in every lane is a unit vector e_q with q in p, no later
   pivot bit can match it, and it never changes again, so the inner loop runs
   over the columns not yet reached plus the few columns some lane left
   without a pivot.  At the prime orders the corank is 1 and the free column
   is the last one in half the patterns and the second-to-last in a quarter,
   so that list is almost always empty and the update count is about H^2 / 2
   rather than H^2.  Output identical to hkt_elim8v. */
static inline void wil_elim(hw_t *c, hw_t *bb, hw_t *pivb, hw_t *pm) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i p = zero;
    __m256i vb = _mm256_loadu_si256((const __m256i *)bb);
    int pend[64], npend = 0;
    for (int cc = 0; cc < H; cc++) {
        __m256i colv = _mm256_loadu_si256((const __m256i *)(c + (size_t)cc * LANES));
        __m256i m    = _mm256_andnot_si256(p, colv);
        __m256i bit  = _mm256_and_si256(m, HV_SUB(zero, m));
        __m256i R    = _mm256_xor_si256(colv, bit);
        p = _mm256_or_si256(p, bit);
        _mm256_storeu_si256((__m256i *)(pivb + (size_t)cc * LANES), bit);
        for (int k = cc; k < H; k++) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * LANES));
            __m256i ns = HV_CMPEQ(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * LANES), v);
        }
        for (int i = 0; i < npend; i++) {
            int k = pend[i];
            __m256i v = _mm256_loadu_si256((const __m256i *)(c + (size_t)k * LANES));
            __m256i ns = HV_CMPEQ(_mm256_and_si256(v, bit), zero);
            v = _mm256_xor_si256(v, _mm256_andnot_si256(ns, R));
            _mm256_storeu_si256((__m256i *)(c + (size_t)k * LANES), v);
        }
        __m256i nb = HV_CMPEQ(_mm256_and_si256(vb, bit), zero);
        vb = _mm256_xor_si256(vb, _mm256_andnot_si256(nb, R));
        if (HV_MOVEMASK(HV_CMPEQ(bit, zero)))
            pend[npend++] = cc;              /* some lane has no pivot here */
    }
    _mm256_storeu_si256((__m256i *)bb, vb);
    _mm256_storeu_si256((__m256i *)pm, p);
}

/* Lane-vectorised tail for the reduced system.  For every lane: part has
   bit c when the pivot row of column c carries the right-hand side, fm has
   bit c when column c has no pivot, and for the first free column f basis0
   is the null vector e_f + sum of e_c over the pivot columns c whose pivot
   row appears in the reduced column f.  Lanes with two or more free columns
   are left to the scalar hkt_tail_m by the caller (nf[L] >= 2). */
static inline void wil_tail(const hw_t *ct, const hw_t *pivb, const hw_t *bbv,
                            hw_t *part, hw_t *fm, hw_t *basis0, int *nf) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i vb = _mm256_loadu_si256((const __m256i *)bbv);
    __m256i vpart = zero, vfm = zero;
    for (int cc = 0; cc < H; cc++) {
        __m256i bit = _mm256_loadu_si256((const __m256i *)(pivb + (size_t)cc * LANES));
        __m256i e   = HV_SET1(HW_ONE << cc);
        __m256i nz  = HV_CMPEQ(_mm256_and_si256(vb, bit), zero);
        vpart = _mm256_or_si256(vpart, _mm256_andnot_si256(nz, e));
        vfm   = _mm256_or_si256(vfm, _mm256_and_si256(HV_CMPEQ(bit, zero), e));
    }
    _mm256_storeu_si256((__m256i *)part, vpart);
    _mm256_storeu_si256((__m256i *)fm, vfm);
    hw_t colf[LANES];
    for (int L = 0; L < LANES; L++) {
        nf[L] = hw_popcount(fm[L]);
        int f = fm[L] ? hw_ctz(fm[L]) : 0;
        colf[L] = fm[L] ? ct[(size_t)f * LANES + L] : (hw_t)0;
    }
    __m256i vc = _mm256_loadu_si256((const __m256i *)colf);
    __m256i vw = _mm256_and_si256(vfm, HV_SUB(zero, vfm));   /* e_f */
    for (int cc = 0; cc < H; cc++) {
        __m256i bit = _mm256_loadu_si256((const __m256i *)(pivb + (size_t)cc * LANES));
        __m256i e   = HV_SET1(HW_ONE << cc);
        __m256i nz  = HV_CMPEQ(_mm256_and_si256(vc, bit), zero);
        vw = _mm256_or_si256(vw, _mm256_andnot_si256(nz, e));
    }
    _mm256_storeu_si256((__m256i *)basis0, vw);
}

/* ------------------------------------------------- the folded right side

   Write d_k = ta_k + u_k.  Bit 1 of mc_k - d_k is
       bit1(mc_k) ^ bit1(d_k) ^ (bit0(d_k) & ~bit0(mc_k))
   (the last term is the borrow out of bit 0), and
       bit1(d_k) = bit1(ta_k) ^ bit1(u_k) ^ (bit0(ta_k) & bit0(u_k)).
   The stage-1 right side is rhs1_k = bit2(A_k) ^ m_k ^ bit0(u_k) with
   m_k = bit0(M1CONST_[k]) and bit2(A_k) = bit0(ta_k), so
       bit0(d_k) = bit0(ta_k) ^ bit0(u_k) = rhs1_k ^ m_k =: D0_k,
   a function of t alone (rhs1 = T^{-1} t), and
       bit0(ta_k) & bit0(u_k) = bit0(ta_k) & ~D0_k.
   Hence, as H-bit words,
       bb = [MC1(t) ^ (D0(t) & ~MC0(t))] ^ [WA1 ^ WB1 ^ (TA0 & ~D0)]
   with MC0, MC1 the bit-0 and bit-1 words of mc(t), WA1 = bit1(ta),
   TA0 = bit0(ta), WB1 = bit1(u).  The first bracket is G(t); the second is X,
   which a pair forms from two words of A, one word of B and rhs1 = RA ^ RB.
   For a left null vector w of the stage-2 matrix of t, parity(bb & w) =
   parity(G(t) & w) ^ parity(X & w), and the table stores the first term. */
static hw_t M0W;                  /* bit0(M1CONST_[k]) as a word over lags */
static hw_t *M1FWD;               /* forward stage-1 matrix: rhs1_k = parity(M1FWD[k] & t) */

static inline hw_t gf2_parity_mul(const hw_t *rows, hw_t v) {
    hw_t r = 0;
    for (int k = 0; k < H; k++) r |= (hw_t)hw_parity(rows[k] & v) << k;
    return r;
}
/* Invert the map t = T(rhs1) by elimination on its byte-table columns. */
static int fwd_init(void) {
    hw_t inv[64], aug[64];
    for (int k = 0; k < H; k++) { inv[k] = 0; }
    /* rows of T: t_c = parity(M1INVCOL_[c] & rhs1), so row c of T is M1INVCOL_[c] */
    for (int k = 0; k < H; k++) { aug[k] = M1INVCOL_[k]; inv[k] = HW_ONE << k; }
    int rank = 0;
    for (int c = 0; c < H; c++) {
        int pr = -1;
        for (int r2 = rank; r2 < H; r2++) if ((aug[r2] >> c) & 1) { pr = r2; break; }
        if (pr < 0) continue;
        hw_t tt = aug[pr]; aug[pr] = aug[rank]; aug[rank] = tt;
        tt = inv[pr]; inv[pr] = inv[rank]; inv[rank] = tt;
        for (int r2 = 0; r2 < H; r2++)
            if (r2 != rank && ((aug[r2] >> c) & 1)) { aug[r2] ^= aug[rank]; inv[r2] ^= inv[rank]; }
        rank++;
    }
    if (rank != H) return 0;
    M1FWD = calloc(H, sizeof(hw_t));
    for (int k = 0; k < H; k++) M1FWD[k] = inv[k];
    M0W = 0;
    for (int k = 1; k <= H; k++) if (M1CONST_[k] & 1) M0W |= HW_ONE << (k - 1);
    /* check: T(M1FWD t) = t and M1FWD T(r) = r on random words */
    hw_t hm = hkt_hmask();
    uint64_t seed = 0x1234567ULL ^ (uint64_t)N;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        hw_t r = (hw_t)(seed & hm);
        if (gf2_parity_mul(M1FWD, nw_tmap(r)) != r) return 0;
    }
    return 1;
}

/* ------------------------------------------------- stage-2 reference build */
static inline nw_t nw_uw(hw_t t) {
    return NW_ONE | ((nw_t)t << 1) | ((nw_t)hkt_revh(t) << (H + 1));
}
static void nw_rows_ref(hw_t t, int16_t *mc, hw_t *col) {
    const nw_t maskn = hkt_nmask();
    const hw_t hmask = hkt_hmask();
    nw_t uw = nw_uw(t);
    hw_t a[64];
    for (int i = 0; i < 64; i++) a[i] = 0;
    for (int k = 1; k <= H; k++) {
        nw_t kept = (~(uw ^ hkt_rot(uw, N - k))) & maskn;
        nw_t G    = kept & ~CONSTALL_[k] & maskn;
        nw_t oddu = (G ^ hkt_rot(G, k)) & maskn;
        a[k - 1] = (hw_t)(oddu >> 1) & hmask;
        mc[k - 1] = (int16_t)(nw_popcount(kept & NFIXM_[k]) / 2);
    }
    for (int k = H; k < 64; k++) mc[k] = 0;
    hkt_tr(a, H, col);
}
/* Left null space of the stage-2 matrix of pattern t, as masks over the lags. */
static int nw_leftnull(hw_t t, hw_t *out, int maxout) {
    int16_t mc[64]; hw_t col[64];
    nw_rows_ref(t, mc, col);
    hw_t bt[64];
    for (int j = 0; j < H; j++) bt[j] = HW_ONE << j;
    hw_t pv = 0;
    for (int c = 0; c < H; c++) {
        hw_t m = col[c] & ~pv;
        if (!m) continue;
        int q = hw_ctz(m);
        pv |= HW_ONE << q;
        hw_t R = col[c] & ~(HW_ONE << q);
        for (int c2 = 0; c2 < H; c2++)
            col[c2] ^= R & ((hw_t)0 - (hw_t)((col[c2] >> q) & 1u));
        for (int j = 0; j < H; j++)
            bt[j] ^= R & ((hw_t)0 - (hw_t)((bt[j] >> q) & 1u));
    }
    int nc = 0;
    for (int q = 0; q < H; q++) {
        if ((pv >> q) & 1u) continue;
        if (nc < maxout) {
            hw_t colq = 0;
            for (int j = 0; j < H; j++) if ((bt[j] >> q) & 1u) colq |= HW_ONE << j;
            out[nc] = colq;
        }
        nc++;
    }
    return nc;
}

/* ------------------------------------------------- direct columns and mc
   (gmat_c_v2.c).  Writing j = c + 1, the row build gives
       M[k][c] = ([u_j = u_{j+k}] & ~CA_k[j]) ^ ([u_j = u_{j-k}] & ~CA_k[j-k])
   with CA_k = CONSTALL_[k]; over k at fixed j the first bracket reads bits of
   uw >> j and the second bits of rot(uw, j), so
       col_j = ((P & NT1) ^ (Q & NT2) ^ (u_j ? 0 : NT1 ^ NT2)) >> 1.
   And kept_k always contains the one fixed point of j -> -j-k, so
       mc_k = H - popcount(uw ^ rot(uw, k)) / 2.
   NT1, NT2 and CJ are words over the lags k = 1..H, so they are 64-bit in
   both builds; only uw changes width. */
static uint64_t NT1_[64], NT2_[64], CJ_[64];
static __m256i *NT1V_, *NT2V_, *CJV_;
static void cols_init(void) {
    uint64_t hb = 0;
    for (int k = 1; k <= H; k++) hb |= 1ULL << k;
    for (int j = 1; j <= H; j++) {
        uint64_t t1 = 0, t2 = 0;
        for (int k = 1; k <= H; k++) {
            if ((CONSTALL_[k] >> j) & 1) t1 |= 1ULL << k;
            int jj = ((j - k) % N + N) % N;
            if ((CONSTALL_[k] >> jj) & 1) t2 |= 1ULL << k;
        }
        NT1_[j] = ~t1 & hb; NT2_[j] = ~t2 & hb; CJ_[j] = NT1_[j] ^ NT2_[j];
    }
    NT1V_ = aligned_alloc(32, 64 * sizeof(__m256i));
    NT2V_ = aligned_alloc(32, 64 * sizeof(__m256i));
    CJV_  = aligned_alloc(32, 64 * sizeof(__m256i));
    for (int j = 0; j < 64; j++) {
        NT1V_[j] = _mm256_set1_epi64x((long long)NT1_[j]);
        NT2V_[j] = _mm256_set1_epi64x((long long)NT2_[j]);
        CJV_[j]  = _mm256_set1_epi64x((long long)CJ_[j]);
    }
}
static inline void nw_mc_from_t(hw_t t, int16_t *mc) {
    const nw_t maskn = hkt_nmask();
    nw_t uw = nw_uw(t);
    for (int k = 1; k <= H; k++) {
        nw_t r = ((uw >> k) | (uw << (N - k))) & maskn;
        mc[k - 1] = (int16_t)(H - nw_popcount(uw ^ r) / 2);
    }
    for (int k = H; k < 64; k++) mc[k] = 0;
}
/* MC0 and MC1 words (bits 0 and 1 of mc_k over the lags) for one pattern. */
static inline void nw_mcbits(hw_t t, hw_t *mc0, hw_t *mc1) {
    const nw_t maskn = hkt_nmask();
    nw_t uw = nw_uw(t);
    hw_t a0 = 0, a1 = 0;
    for (int k = 1; k <= H; k++) {
        nw_t r = ((uw >> k) | (uw << (N - k))) & maskn;
        int m = H - nw_popcount(uw ^ r) / 2;
        a0 |= (hw_t)(m & 1) << (k - 1);
        a1 |= (hw_t)((m >> 1) & 1) << (k - 1);
    }
    *mc0 = a0; *mc1 = a1;
}
static inline void nw_cols_scalar(hw_t t, hw_t *col) {
    const hw_t hmask = hkt_hmask();
    nw_t uw = nw_uw(t);
    for (int j = 1; j <= H; j++) {
        nw_t P = uw >> j;
        nw_t Q = (uw << j) | (uw >> (N - j));
        nw_t s = (nw_t)0 - ((uw >> j) & NW_ONE);
        nw_t X = (P & NT1_[j]) ^ (Q & NT2_[j]) ^ (CJ_[j] & ~s);
        col[j - 1] = (hw_t)(X >> 1) & hmask;
    }
}

#ifndef HKT_WIDE
/* MC0 and MC1 words for four patterns at once: the bits 0 and 1 of mc_k are
   masked out of the 64-bit lane and shifted into place, so no value leaves
   the vector unit until the end. */
static inline void nw_mcbits4(const hw_t *t4, hw_t *mc0, hw_t *mc1) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i nib = _mm256_set1_epi8(0x0f), zero = _mm256_setzero_si256();
    const __m256i maskn = _mm256_set1_epi64x((long long)hkt_nmask());
    const __m256i hh = _mm256_set1_epi64x(H), one = _mm256_set1_epi64x(1);
    __m256i uw = _mm256_setr_epi64x((long long)nw_uw(t4[0]), (long long)nw_uw(t4[1]),
                                    (long long)nw_uw(t4[2]), (long long)nw_uw(t4[3]));
    __m256i a0 = zero, a1 = zero;
    for (int k = 1; k <= H; k++) {
        __m128i ck = _mm_cvtsi32_si128(k), cn = _mm_cvtsi32_si128(N - k), sh = _mm_cvtsi32_si128(k - 1);
        __m256i r = _mm256_and_si256(_mm256_or_si256(_mm256_srl_epi64(uw, ck),
                                                     _mm256_sll_epi64(uw, cn)), maskn);
        __m256i x = _mm256_xor_si256(uw, r);
        __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(x, nib));
        __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), nib));
        __m256i pc = _mm256_sad_epu8(_mm256_add_epi8(lo, hi), zero);
        __m256i m  = _mm256_sub_epi64(hh, _mm256_srli_epi64(pc, 1));
        a0 = _mm256_or_si256(a0, _mm256_sll_epi64(_mm256_and_si256(m, one), sh));
        a1 = _mm256_or_si256(a1, _mm256_sll_epi64(_mm256_and_si256(_mm256_srli_epi64(m, 1), one), sh));
    }
    uint64_t o0[4], o1[4];
    _mm256_storeu_si256((__m256i *)o0, a0);
    _mm256_storeu_si256((__m256i *)o1, a1);
    for (int u = 0; u < 4; u++) { mc0[u] = (hw_t)o0[u]; mc1[u] = (hw_t)o1[u]; }
}
/* Four patterns at once into the transposed layout ct[c*8 + lane]. */
static inline void nw_cols4(const hw_t *t4, hw_t *ct, int L0) {
    const __m128i hmask4 = _mm_set1_epi32((int)hkt_hmask());
    const __m256i one = _mm256_set1_epi64x(1), zero = _mm256_setzero_si256();
    const __m256i pick = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    __m256i uw = _mm256_setr_epi64x((long long)nw_uw(t4[0]), (long long)nw_uw(t4[1]),
                                    (long long)nw_uw(t4[2]), (long long)nw_uw(t4[3]));
    for (int j = 1; j <= H; j++) {
        __m128i cj = _mm_cvtsi32_si128(j), cn = _mm_cvtsi32_si128(N - j);
        __m256i P = _mm256_srl_epi64(uw, cj);
        __m256i Q = _mm256_or_si256(_mm256_sll_epi64(uw, cj), _mm256_srl_epi64(uw, cn));
        __m256i s = _mm256_sub_epi64(zero, _mm256_and_si256(P, one));
        __m256i X = _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(P, NT1V_[j]),
                                                      _mm256_and_si256(Q, NT2V_[j])),
                                     _mm256_andnot_si256(s, CJV_[j]));
        __m256i Y = _mm256_permutevar8x32_epi32(_mm256_srli_epi64(X, 1), pick);
        __m128i lo = _mm_and_si128(_mm256_castsi256_si128(Y), hmask4);
        _mm_storeu_si128((__m128i *)(ct + (size_t)(j - 1) * 8 + L0), lo);
    }
}
#else
/* Wide build.  uw is an N-bit word with N up to 127, held as two 64-bit
   lanes (lo, hi) per pattern, four patterns per register pair.  Every shift
   count is uniform across the lanes (it depends on j or k, never on the
   pattern), so a shift of the 128-bit word is a few 64-bit shifts and ORs
   with the count in an XMM register, and a window of 64 bits at offset s is
   the usual two-shift merge.  Bits at or above N are zero in uw, and are
   zero in every window read above N, which the column build relies on: the
   rotate rot(uw, j) read at bits 1..H is (lo << j) | window(uw, N - j),
   because bit k of the second term is bit N - j + k of uw and that is zero
   once k >= j. */
static inline __m256i v_win(__m256i lo, __m256i hi, int s) {
    if (s == 0) return lo;
    if (s < 64) return _mm256_or_si256(_mm256_srl_epi64(lo, _mm_cvtsi32_si128(s)),
                                       _mm256_sll_epi64(hi, _mm_cvtsi32_si128(64 - s)));
    if (s == 64) return hi;
    return _mm256_srl_epi64(hi, _mm_cvtsi32_si128(s - 64));
}
static inline void v_uw4(const hw_t *t4, __m256i *lo, __m256i *hi) {
    nw_t u0 = nw_uw(t4[0]), u1 = nw_uw(t4[1]), u2 = nw_uw(t4[2]), u3 = nw_uw(t4[3]);
    *lo = _mm256_setr_epi64x((long long)(uint64_t)u0, (long long)(uint64_t)u1,
                             (long long)(uint64_t)u2, (long long)(uint64_t)u3);
    *hi = _mm256_setr_epi64x((long long)(uint64_t)(u0 >> 64), (long long)(uint64_t)(u1 >> 64),
                             (long long)(uint64_t)(u2 >> 64), (long long)(uint64_t)(u3 >> 64));
}
static inline void nw_mcbits4(const hw_t *t4, hw_t *mc0, hw_t *mc1) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                         0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i nib = _mm256_set1_epi8(0x0f), zero = _mm256_setzero_si256();
    const nw_t maskn = hkt_nmask();
    const __m256i mlo = _mm256_set1_epi64x((long long)(uint64_t)maskn);
    const __m256i mhi = _mm256_set1_epi64x((long long)(uint64_t)(maskn >> 64));
    const __m256i hh = _mm256_set1_epi64x(H), one = _mm256_set1_epi64x(1);
    __m256i lo, hi;
    v_uw4(t4, &lo, &hi);
    __m256i a0 = zero, a1 = zero;
    for (int k = 1; k <= H; k++) {
        int m = N - k;                                   /* H+1 <= m <= N-1 */
        __m128i ck = _mm_cvtsi32_si128(k), sh = _mm_cvtsi32_si128(k - 1);
        /* r = rot right by k = (uw >> k) | (uw << m), masked to N bits */
        __m256i rlo = v_win(lo, hi, k);
        __m256i rhi = _mm256_srl_epi64(hi, ck);
        if (m < 64) {
            __m128i cm = _mm_cvtsi32_si128(m), cm2 = _mm_cvtsi32_si128(64 - m);
            rlo = _mm256_or_si256(rlo, _mm256_sll_epi64(lo, cm));
            rhi = _mm256_or_si256(rhi, _mm256_or_si256(_mm256_sll_epi64(hi, cm),
                                                       _mm256_srl_epi64(lo, cm2)));
        } else {
            rhi = _mm256_or_si256(rhi, _mm256_sll_epi64(lo, _mm_cvtsi32_si128(m - 64)));
        }
        rlo = _mm256_and_si256(rlo, mlo);
        rhi = _mm256_and_si256(rhi, mhi);
        __m256i xlo = _mm256_xor_si256(lo, rlo), xhi = _mm256_xor_si256(hi, rhi);
        __m256i plo = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(xlo, nib)),
                                      _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(xlo, 4), nib)));
        __m256i phi = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(xhi, nib)),
                                      _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(xhi, 4), nib)));
        __m256i pc = _mm256_add_epi64(_mm256_sad_epu8(plo, zero), _mm256_sad_epu8(phi, zero));
        __m256i mm = _mm256_sub_epi64(hh, _mm256_srli_epi64(pc, 1));
        a0 = _mm256_or_si256(a0, _mm256_sll_epi64(_mm256_and_si256(mm, one), sh));
        a1 = _mm256_or_si256(a1, _mm256_sll_epi64(_mm256_and_si256(_mm256_srli_epi64(mm, 1), one), sh));
    }
    _mm256_storeu_si256((__m256i *)mc0, a0);
    _mm256_storeu_si256((__m256i *)mc1, a1);
}
/* Four patterns at once into the transposed layout ct[c*4 + lane]; L0 is 0. */
static inline void nw_cols4(const hw_t *t4, hw_t *ct, int L0) {
    const __m256i hmaskv = _mm256_set1_epi64x((long long)hkt_hmask());
    const __m256i one = _mm256_set1_epi64x(1), zero = _mm256_setzero_si256();
    __m256i lo, hi;
    v_uw4(t4, &lo, &hi);
    for (int j = 1; j <= H; j++) {
        __m256i P = v_win(lo, hi, j);
        __m256i Q = _mm256_or_si256(_mm256_sll_epi64(lo, _mm_cvtsi32_si128(j)), v_win(lo, hi, N - j));
        __m256i s = _mm256_sub_epi64(zero, _mm256_and_si256(P, one));
        __m256i X = _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(P, NT1V_[j]),
                                                      _mm256_and_si256(Q, NT2V_[j])),
                                     _mm256_andnot_si256(s, CJV_[j]));
        __m256i Y = _mm256_and_si256(_mm256_srli_epi64(X, 1), hmaskv);
        _mm256_storeu_si256((__m256i *)(ct + (size_t)(j - 1) * LANES + L0), Y);
    }
}
#endif
static void cols_selfcheck(void) {
    hw_t hm = hkt_hmask();
    uint64_t seed = 0x2545F4914F6CDD1DULL ^ (uint64_t)N;
    for (int i = 0; i < 2048; i++) {
        hw_t t4[4];
        for (int u = 0; u < 4; u++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            t4[u] = (hw_t)(seed & hm);
        }
        hw_t ct[64 * LANES]; hw_t q0[4], q1[4];
        nw_cols4(t4, ct, 0);
        nw_mcbits4(t4, q0, q1);
        for (int u = 0; u < 4; u++) {
            int16_t m1[64], m2[64]; hw_t c1[64], c2[64], b0, b1;
            nw_rows_ref(t4[u], m1, c1);
            nw_mc_from_t(t4[u], m2);
            nw_cols_scalar(t4[u], c2);
            nw_mcbits(t4[u], &b0, &b1);
            for (int k = 0; k < H; k++) {
                int bad = (m1[k] != m2[k]) || (c1[k] != c2[k]);
                bad |= (ct[(size_t)k * LANES + u] != c1[k]);
                bad |= (((b0 >> k) & 1) != (hw_t)(m1[k] & 1)) || (((b1 >> k) & 1) != (hw_t)((m1[k] >> 1) & 1));
                bad |= (q0[u] != b0) || (q1[u] != b1);
                if (bad) {
                    fprintf(stderr, "n=%d: direct column build disagrees with the "
                            "row build at t=%llu lag %d\n", N, (unsigned long long)t4[u], k + 1);
                    exit(1);
                }
            }
        }
    }
}

/* ------------------------------------------------- consistency table (CC)

   K uint32 words per pattern t:
     bits 0..H-1   a left null vector w of the stage-2 matrix; zero ends the
                   list
     bit 31        parity(G(t) & w), the t-only half of the test
     bit 30        on the last stored word: the null space has more vectors
                   than fit, so a pair that passes the stored ones still runs
                   the elimination (which rechecks consistency)
     0xffffffff    in word 0: nullity above S2_MAXFREE; the pattern cannot be
                   completed under the cap and the pair is counted as capped
   Needs H <= 30; the entries stay 32-bit in both builds. */
#define S2_MAXFREE 6
#define CC_REJECT 0xffffffffu
#define CC_MORE   0x40000000u
#define CC_G      0x80000000u
static uint32_t *CC = NULL;
static int CCK = 0;              /* null-vector words per pattern; 0 = no table */
static int GTAB = 0;             /* 1: the entry also stores G(t) after the K words */
static int CCS = 0;              /* entry stride in words, CCK + GTAB */
static long long CC_REJECTED = 0;

static inline hw_t nw_G(hw_t t) {
    hw_t mc0, mc1;
    nw_mcbits(t, &mc0, &mc1);
    hw_t d0 = gf2_parity_mul(M1FWD, t) ^ M0W;
    return mc1 ^ (d0 & ~mc0);
}
static int cc_choose_k(void) {
    long hist[40] = {0}; int ns = 4096;
    hw_t hm = hkt_hmask();
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    hw_t tmp[64];
    for (int i = 0; i < ns; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        int c = nw_leftnull((hw_t)(seed & hm), tmp, 64);
        hist[c > S2_MAXFREE ? S2_MAXFREE : c]++;
    }
    long acc = 0;
    for (int k = 0; k <= S2_MAXFREE; k++) {
        acc += hist[k];
        if (acc * 100 >= ns * 99) return k > 0 ? k : 1;
    }
    return S2_MAXFREE;
}
static void cc_build(void) {
    size_t nt = (size_t)1 << H;
    CCS = CCK + GTAB;
    size_t bytes = nt * (size_t)CCS * sizeof(uint32_t);
    bytes = (bytes + (1u << 21) - 1) & ~(size_t)((1u << 21) - 1);
    CC = aligned_alloc(1u << 21, bytes);
    if (!CC) {
        fprintf(stderr, "out of memory for the consistency table (%.2f GB)\n",
                nt * (double)CCS * 4 / 1073741824.0);
        exit(1);
    }
    madvise(CC, bytes, MADV_HUGEPAGE);
    hw_t hm = hkt_hmask();
    long long rej = 0;
#pragma omp parallel for schedule(dynamic, 4096) reduction(+:rej)
    for (long long ti = 0; ti < (long long)nt; ti++) {
        hw_t t = (hw_t)ti;
        hw_t nv[S2_MAXFREE + 1];
        uint32_t *e = CC + (size_t)t * CCS;
        int c = nw_leftnull(t, nv, S2_MAXFREE + 1);
        if (c > S2_MAXFREE) {
            e[0] = CC_REJECT; for (int i = 1; i < CCS; i++) e[i] = 0; rej++; continue;
        }
        hw_t G = nw_G(t);
        if (GTAB) e[CCK] = (uint32_t)G;
        for (int i = 0; i < CCK; i++) {
            if (i < c) {
                uint32_t w = (uint32_t)(nv[i] & hm);
                e[i] = w | (hw_parity(w & G) ? CC_G : 0u);
            } else e[i] = 0;
        }
        if (c > CCK) {
            uint32_t w = e[CCK - 1] | CC_MORE;
            /* a stored vector that would read as CC_REJECT is dropped; the
               flag alone still sends the pair to the full solve */
            e[CCK - 1] = (w == CC_REJECT) ? CC_MORE : w;
        }
    }
    CC_REJECTED = rej;
}

/* -------------------------------------------------------------- row filter

   ALLOWD[pc] is the set of popcounts pd such that (pc, pd) are the popcounts
   of some (C, D) completing the current (a, b).  When C and D have the same
   popcount the candidate set contains both (C, D) and (D, C), so codeC <=
   codeD keeps one of them. */
static hw_t ALLOWD[64];
/* Note on a pruning that was measured and dropped.  codeC = ~x and
   codeD = t ^ x give popcount(codeC) + popcount(codeD) = H + popcount(t) mod 2,
   so a row-sum match needs popcount(t) = a + b + H mod 2 for an allowed
   (a, b), which is a per-B-row bit that could join the pair filter.  Every
   pair that reaches the solve already satisfies it (the same pair counts
   with and without the filter at 13..21 and 47): the stage-1 system is the
   PAF equations, whose sum carries the row sums, and the table's popcount
   fixes them.  There is nothing to gain from it. */
static inline int wil_keep(hw_t v, hw_t s0, hw_t maskH) {
    int a = H - hw_popcount(v & maskH);        /* popcount of codeC */
    hw_t am = ALLOWD[a];
    if (!am) return 0;
    hw_t cD = (s0 ^ v) & maskH;
    int b = hw_popcount(cD);
    if (!((am >> b) & 1u)) return 0;
    return a != b || ((~v) & maskH) <= cD;
}
static inline int wil_rowfilter(hw_t *cand, int nc, hw_t s, hw_t maskH) {
    int keep = 0;
    for (int i = 0; i < nc; i++)
        if (wil_keep(cand[i], s, maskH)) cand[keep++] = cand[i];
    return keep;
}

/* ------------------------------------------------------------------ rows */
static void sym_row(hw_t code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v; x[N - j] = v;
    }
}
static inline nw_t sym_mask(hw_t code) {
    nw_t m = NW_ONE;
    for (int j = 1; j <= H; j++)
        if ((code >> (j - 1)) & 1) { m |= NW_ONE << j; m |= NW_ONE << (N - j); }
    return m;
}
static inline void paf_mask(nw_t m, int8_t *paf) {
    nw_t full = hkt_nmask();
    for (int s = 1; s <= H; s++) {
        nw_t r = ((m >> s) | (m << (N - s))) & full;
        paf[s - 1] = (int8_t)(N - 2 * nw_popcount(m ^ r));
    }
}
static double *COSTAB;                 /* [H+1][H] */
static void cos_init(void) {
    COSTAB = malloc(sizeof(double) * (size_t)(H + 1) * H);
    for (int k = 0; k <= H; k++)
        for (int s = 1; s <= H; s++)
            COSTAB[(size_t)k * H + s - 1] = 2.0 * cos(2.0 * M_PI * s * k / N);
}
static inline double psd_at(const int8_t *paf, int k) {
    double acc = N;
    const double *row = COSTAB + (size_t)k * H;
    for (int s = 0; s < H; s++) acc += paf[s] * row[s];
    return acc;
}
static double psd_all(const int8_t *paf, float *out) {
    double m = -1e18;
    for (int k = 1; k <= H; k++) {
        double v = psd_at(paf, k);
        out[k - 1] = (float)v;
        if (v > m) m = v;
    }
    return m;
}

/* Startup check of the folded right-hand side against nw_bvec_ref on random
   symmetric row pairs. */
static void fold_selfcheck(void) {
    hw_t hm = hkt_hmask();
    uint64_t seed = 0xD1B54A32D192ED03ULL ^ (uint64_t)N;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        hw_t ca = (hw_t)(seed & hm);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        hw_t cb = (hw_t)(seed & hm);
        int8_t pa[64], pb[64], su[64]; int ta[64];
        paf_mask(sym_mask(ca), pa); paf_mask(sym_mask(cb), pb);
        hw_t raraw; nw_arow(pa, ta, &raraw);
        hw_t wa1 = 0, ta0 = 0, rbraw = 0, wb1 = 0;
        for (int k = 1; k <= H; k++) {
            if (ta[k] & 1) ta0 |= HW_ONE << (k - 1);
            if (ta[k] & 2) wa1 |= HW_ONE << (k - 1);
            int u = (N - (int)pb[k - 1]) >> 2;
            su[k - 1] = (int8_t)u;
            if (u & 1) rbraw |= HW_ONE << (k - 1);
            if (u & 2) wb1 |= HW_ONE << (k - 1);
        }
        hw_t rhs1 = raraw ^ rbraw;
        hw_t t = nw_tmap(rhs1);
        int16_t mc[64]; nw_mc_from_t(t, mc);
        hw_t ref = nw_bvec_ref(mc, ta, su);
        hw_t d0 = rhs1 ^ M0W;
        hw_t X = wa1 ^ wb1 ^ (ta0 & ~d0);
        hw_t got = (nw_G(t) ^ X) & hm;
        if (got != (ref & hm) || gf2_parity_mul(M1FWD, t) != rhs1) {
            fprintf(stderr, "n=%d: folded right-hand side disagrees with nw_bvec "
                    "(A=%llu B=%llu ref=%llx got=%llx)\n", N, (unsigned long long)ca,
                    (unsigned long long)cb, (unsigned long long)ref, (unsigned long long)got);
            exit(1);
        }
    }
}


/* Known-answer check of the whole per-pair path at the order in hand.  No
   stored census exists above n = 63, and the forced-wide runs below it never
   set a bit above 63 in an unfolded word, so the branches that only run at
   n >= 65 (the high lane of uw, shift counts of 64 or more) would otherwise
   go untested until a real census.  The solve reads nothing of A but
   PAF_A + PAF_B, so for random symmetric rows C, D and B the synthetic
   PAF_A = -(PAF_C + PAF_D) - PAF_B is congruent to n mod 4 at every lag like
   a real row's, and the candidates for that pair must contain (C, D).  The
   check runs the hot path as the census does: nw_arow, the t map, the
   folded X, nw_cols4, nw_mcbits4, wil_elim, wil_tail or hkt_tail_m, the
   candidate decoding of WIL_EMIT and its exact PAF check through
   hkt_symword and hkt_rotk; paf_mask is checked against paf_of on the way.
   Instances the nullity cap would drop are skipped and counted. */
static void solve_selfcheck(void) {
    const hw_t hm = hkt_hmask();
    uint64_t seed = 0x6A09E667F3BCC909ULL ^ (uint64_t)N;
    int checked = 0, capped = 0;
    for (int it = 0; it < 512; it++) {
        hw_t cC[LANES], cD[LANES], cB[LANES], t4[LANES], X4[LANES], d04[LANES];
        int8_t pa[LANES][64], pb[LANES][64];
        for (int L = 0; L < LANES; L++) {
            int8_t pc[64], pd[64], x[128], ref[64];
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; cC[L] = (hw_t)(seed & hm);
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; cD[L] = (hw_t)(seed & hm);
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; cB[L] = (hw_t)(seed & hm);
            paf_mask(sym_mask(cC[L]), pc); paf_mask(sym_mask(cD[L]), pd); paf_mask(sym_mask(cB[L]), pb[L]);
            sym_row(cC[L], x); paf_of(x, ref);
            if (memcmp(ref, pc, (size_t)H)) {
                fprintf(stderr, "n=%d: paf_mask disagrees with paf_of (code %llu)\n", N, (unsigned long long)cC[L]);
                exit(1);
            }
            /* PAF_A = -(PAF_C + PAF_D) - PAF_B is n mod 4 at every lag; it is
               int8 unless three PAFs of random rows all exceed 40 or so at
               one lag, which does not happen, but is checked rather than
               clamped (clamping would break the congruence). */
            int fits = 1;
            for (int k = 0; k < H; k++) {
                int v = -((int)pc[k] + (int)pd[k]) - (int)pb[L][k];
                if (v < -127 || v > 127) fits = 0;
                pa[L][k] = (int8_t)v;
            }
            if (!fits) { L--; continue; }
            int ta[64]; hw_t raraw; nw_arow(pa[L], ta, &raraw);
            hw_t rat = nw_tmap(raraw), wa1 = 0, ta0 = 0, rbraw = 0, wb1 = 0;
            for (int k = 1; k <= H; k++) {
                if (ta[k] & 1) ta0 |= HW_ONE << (k - 1);
                if (ta[k] & 2) wa1 |= HW_ONE << (k - 1);
                int u = (N - (int)pb[L][k - 1]) >> 2;
                if (u & 1) rbraw |= HW_ONE << (k - 1);
                if (u & 2) wb1 |= HW_ONE << (k - 1);
            }
            hw_t rbt = nw_tmap(rbraw);
            t4[L] = rat ^ rbt;
            d04[L] = (raraw ^ M0W) ^ rbraw;
            X4[L] = wa1 ^ wb1 ^ (ta0 & ~d04[L]);
        }
        hw_t ct[64 * LANES], pb8[64 * LANES], pm8[LANES], bbv[LANES], q0[LANES], q1[LANES];
        nw_cols4(t4, ct, 0);
        if (LANES > 4) nw_cols4(t4 + 4, ct, 4);
        nw_mcbits4(t4, q0, q1);
        if (LANES > 4) nw_mcbits4(t4 + 4, q0 + 4, q1 + 4);
        for (int L = 0; L < LANES; L++) bbv[L] = (q1[L] ^ (d04[L] & ~q0[L]) ^ X4[L]) & hm;
        wil_elim(ct, bbv, pb8, pm8);
        hw_t part[LANES], fm[LANES], bas0[LANES]; int nf[LANES];
        wil_tail(ct, pb8, bbv, part, fm, bas0, nf);
        for (int L = 0; L < LANES; L++) {
            if (bbv[L] & ~pm8[L]) {
                fprintf(stderr, "n=%d: known-answer solve inconsistent (C=%llu D=%llu B=%llu)\n", N,
                        (unsigned long long)cC[L], (unsigned long long)cD[L], (unsigned long long)cB[L]);
                exit(1);
            }
            if (nf[L] > S2_MAXFREE) { capped++; continue; }
            hw_t cand[HKT_MAXC]; int nc;
            if (nf[L] <= 1) {
                cand[0] = part[L]; nc = 1;
                if (nf[L]) { cand[1] = part[L] ^ bas0[L]; nc = 2; }
            } else {
                hw_t colL[64], pivL[64];
                for (int k = 0; k < H; k++) { colL[k] = ct[(size_t)k * LANES + L]; pivL[k] = pb8[(size_t)k * LANES + L]; }
                nc = hkt_tail_m(colL, bbv[L], pm8[L], pivL, cand);
            }
            int hit = 0;
            for (int i = 0; i < nc; i++) {
                hw_t codeC = (~cand[i]) & hm, codeD = (t4[L] ^ cand[i]) & hm;
                if (codeC != cC[L] || codeD != cD[L]) continue;
                nw_t wb = hkt_symword(cB[L]), wc = hkt_symword(codeC), wd = hkt_symword(codeD);
                for (int k = 1; k <= H; k++) {
                    int t2 = (int)pa[L][k - 1] + 3 * N
                           - 2 * (nw_popcount(wb ^ hkt_rotk(wb, k)) + nw_popcount(wc ^ hkt_rotk(wc, k))
                                + nw_popcount(wd ^ hkt_rotk(wd, k)));
                    if (t2 != 0) {
                        fprintf(stderr, "n=%d: exact check fails on the known answer at lag %d\n", N, k);
                        exit(1);
                    }
                }
                hit = 1;
            }
            if (!hit) {
                fprintf(stderr, "n=%d: known-answer (C, D) missing from the %d candidates (C=%llu D=%llu B=%llu t=%llu)\n",
                        N, nc, (unsigned long long)cC[L], (unsigned long long)cD[L],
                        (unsigned long long)cB[L], (unsigned long long)t4[L]);
                exit(1);
            }
            checked++;
        }
    }
    printf("   known-answer check: %d synthetic (C, D, B) triples recovered, %d skipped by the nullity cap\n",
           checked, capped);
}

/* -------------------------------------------------- decimation on codes */
static int NDEC;
static int *DECMAP;              /* [NDEC][H] */
static hw_t *DBYTE;              /* [NDEC][DNB][256] */
static int DNB;
static int gcd_int(int a, int b) { while (b) { int q = a % b; a = b; b = q; } return a; }
static void dec_init(void) {
    int cnt = 0;
    for (int s = 1; s <= H; s++) if (gcd_int(s, N) == 1) cnt++;
    NDEC = cnt;
    DECMAP = malloc(sizeof(int) * (size_t)cnt * H);
    int i = 0;
    for (int s = 1; s <= H; s++) {
        if (gcd_int(s, N) != 1) continue;
        for (int j = 1; j <= H; j++) {
            int m = (int)(((long long)s * j) % N);
            if (m > H) m = N - m;
            DECMAP[(size_t)i * H + j - 1] = m - 1;
        }
        i++;
    }
    DNB = (H + 7) / 8;
    DBYTE = calloc((size_t)NDEC * DNB * 256, sizeof(hw_t));
    if (!DBYTE) { fprintf(stderr, "out of memory for the decimation tables\n"); exit(1); }
    for (int d = 0; d < NDEC; d++)
        for (int b = 0; b < DNB; b++)
            for (int v = 0; v < 256; v++) {
                hw_t acc = 0;
                for (int bit = 0; bit < 8; bit++) {
                    int pos = b * 8 + bit;
                    if (pos < H && ((v >> bit) & 1))
                        acc |= HW_ONE << DECMAP[(size_t)d * H + pos];
                }
                DBYTE[(((size_t)d * DNB) + b) * 256 + v] = acc;
            }
}
static inline hw_t dec_code(hw_t code, int d) {
    const hw_t *t = DBYTE + (size_t)d * DNB * 256;
    hw_t r = 0;
    for (int b = 0; b < DNB; b++) r |= t[(size_t)b * 256 + ((code >> (8 * b)) & 0xffu)];
    return r;
}
static inline int a_is_canonical(hw_t code) {
    for (int d = 1; d < NDEC; d++)
        if (dec_code(code, d) < code) return 0;
    return 1;
}

/* --------------------------------------------------------- combinations */
static long long BINOM[64][64];
static void binom_init(void) {
    for (int i = 0; i < 64; i++) {
        BINOM[i][0] = 1;
        for (int j = 1; j < 64; j++)
            BINOM[i][j] = (j > i) ? 0 : BINOM[i-1][j-1] + BINOM[i-1][j];
    }
}
static hw_t unrank_colex(long long rank, int k) {
    hw_t m = 0;
    for (int i = k; i >= 1; i--) {
        int c = i - 1;
        while (BINOM[c + 1][i] <= rank) c++;
        m |= HW_ONE << c;
        rank -= BINOM[c][i];
    }
    return m;
}
static inline hw_t gosper_next(hw_t v) {
    hw_t t = v | (v - 1);
    return (t + 1) | (((~t & (t + 1)) - 1) >> (hw_ctz(v) + 1));
}

/* ------------------------------------------------------------ row sums */
static int signed_sum(int s) {
    return (((s - N) % 4) + 4) % 4 == 0 ? s : -s;
}
static int popcount_for(int s) {
    int num = s - 1 + 2 * H;
    if (num < 0 || (num & 3)) return -1;
    int p = num / 4;
    return (p >= 0 && p <= H) ? p : -1;
}
typedef struct { int s[4]; int p[4]; } Decomp;
static int decomps(Decomp *out, int maxout) {
    int nd = 0, fourn = 4 * N;
    int lim = (int)(sqrt((double)fourn) + 1e-9);
    for (int a = 1; a <= lim; a += 2)
        for (int b = 1; b <= a; b += 2)
            for (int c = 1; c <= b; c += 2)
                for (int d = 1; d <= c; d += 2) {
                    if (a*a + b*b + c*c + d*d != fourn) continue;
                    if (nd >= maxout) continue;
                    Decomp *D = &out[nd];
                    D->s[0] = a; D->s[1] = b; D->s[2] = c; D->s[3] = d;
                    int ok = 1;
                    for (int u = 0; u < 4; u++) {
                        D->p[u] = popcount_for(signed_sum(D->s[u]));
                        if (D->p[u] < 0) ok = 0;
                    }
                    if (ok) nd++;
                }
    return nd;
}

/* --------------------------------------------------------- the B table */
static int NLEV = 256;
static int SAMPLE = 1;
static int ENUM_ONLY = 0;
#ifndef ACHUNK
#define ACHUNK 4096
#endif
#ifndef NW_CHUNK
#define NW_CHUNK 32
#endif
static int lev_of(double x, int fourn) {
    int v = (int)(x * NLEV / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > NLEV - 1) v = NLEV - 1;
    return v;
}
static int lev256(double x, int fourn) {
    int v = (int)(x * 256 / (double)(fourn + 1));
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return v;
}
typedef struct { hw_t rbt, rbraw, wb1; } BRec;
typedef struct {
    int n;
    hw_t     *code;        /* ascending */
    BRec     *rec;         /* [n] */
    int cwords;
    uint64_t *bmp;         /* [H][NLEV][cwords], prefix-ored over levels */
    uint8_t  *lv8;         /* [n][H], sampling runs only */
    long long ncomb;
    double table_mb;       /* code + rec */
} Table;

/* Index size in bytes for a table of n rows at nlev levels. */
static double index_bytes(long long n, int nlev) {
    size_t cw = (size_t)((n + 63) >> 6);
    cw = (cw + 3) & ~(size_t)3;
    return (double)H * nlev * cw * 8.0;
}

/* count_only: enumerate and count the kept codes, build nothing. */
static void table_build(Table *T, int pb, int fourn, int count_only) {
    long long ncomb = BINOM[H][pb];
    memset(T, 0, sizeof *T);
    T->ncomb = ncomb;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    hw_t **lc = calloc(nth, sizeof(hw_t *));
    long long *ln = calloc(nth, sizeof(long long));
#pragma omp parallel num_threads(nth)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        long long per = (ncomb + nth - 1) / nth;
        long long lo = (long long)t * per, hi = lo + per;
        if (hi > ncomb) hi = ncomb;
        long long cap = 1024, cnt = 0;
        hw_t *buf = count_only ? NULL : malloc(sizeof(hw_t) * (size_t)cap);
        int8_t paf[64]; float psd[64];
        if (lo < hi) {
            hw_t c = unrank_colex(lo, pb);
            for (long long ci = lo; ci < hi; ci++) {
                paf_mask(sym_mask(c), paf);
                if (!(psd_all(paf, psd) > fourn + 1e-6)) {
                    if (!count_only) {
                        if (cnt == cap) { cap *= 2; buf = realloc(buf, sizeof(hw_t) * (size_t)cap); }
                        buf[cnt] = c;
                    }
                    cnt++;
                }
                if (ci + 1 < hi) c = gosper_next(c);
            }
        }
        lc[t] = buf; ln[t] = cnt;
    }
    long long total = 0;
    for (int t = 0; t < nth; t++) total += ln[t];
    if (total > 0x7fffffffLL) { fprintf(stderr, "B table too large for int indices\n"); exit(1); }
    T->n = (int)total;
    T->table_mb = (double)total * (sizeof(hw_t) + sizeof(BRec)) / 1048576.0;
    if (count_only) { free(lc); free(ln); return; }
    T->code = malloc(sizeof(hw_t) * (size_t)(total + 1));
    {   long long off = 0;
        for (int t = 0; t < nth; t++) {
            memcpy(T->code + off, lc[t], sizeof(hw_t) * (size_t)ln[t]);
            off += ln[t]; free(lc[t]);
        }
    }
    free(lc); free(ln);
    T->rec = aligned_alloc(64, ((size_t)(total + 4) * sizeof(BRec) + 63) & ~(size_t)63);
    T->cwords = (int)((total + 63) >> 6);
    /* the AND chain reads four words at a time; pad to a multiple of four */
    size_t cw4 = ((size_t)T->cwords + 3) & ~(size_t)3;
    size_t bmpn = (size_t)H * NLEV * cw4;
    T->bmp = calloc(bmpn + 4, sizeof(uint64_t));
    T->lv8 = (SAMPLE > 1) ? malloc((size_t)(total + 1) * H) : NULL;
    if (!T->rec || !T->bmp || (SAMPLE > 1 && !T->lv8)) {
        fprintf(stderr, "out of memory for the B table (%lld rows, index %.1f GB)\n",
                total, bmpn * 8.0 / (1024.0 * 1024.0 * 1024.0));
        exit(1);
    }
    T->cwords = (int)cw4;
#pragma omp parallel for schedule(dynamic, 4)
    for (int w = 0; w < T->cwords; w++) {
        int8_t paf[64]; float psd[64];
        int hi = (w + 1) * 64; if (hi > T->n) hi = T->n;
        for (int bi = w * 64; bi < hi; bi++) {
            paf_mask(sym_mask(T->code[bi]), paf);
            psd_all(paf, psd);
            hw_t rb = 0, wb1 = 0;
            for (int k = 1; k <= H; k++) {
                int u = (N - (int)paf[k - 1]) >> 2;
                if (u & 1) rb |= HW_ONE << (k - 1);
                if (u & 2) wb1 |= HW_ONE << (k - 1);
            }
            T->rec[bi].rbt = nw_tmap(rb);
            T->rec[bi].rbraw = rb;
            T->rec[bi].wb1 = wb1;
            for (int k = 0; k < H; k++) {
                int lv = lev_of(psd[k], fourn);
                T->bmp[(((size_t)k * NLEV) + lv) * T->cwords + w] |= 1ULL << (bi & 63);
                if (T->lv8) T->lv8[(size_t)bi * H + k] = (uint8_t)lev256(psd[k], fourn);
            }
        }
    }
#pragma omp parallel for schedule(static)
    for (int k = 0; k < H; k++)
        for (int t = 1; t < NLEV; t++) {
            uint64_t *dst = T->bmp + (((size_t)k * NLEV) + t) * T->cwords;
            const uint64_t *src = dst - T->cwords;
            for (int w = 0; w < T->cwords; w++) dst[w] |= src[w];
        }
}
static void table_free(Table *T) {
    free(T->code); free(T->rec); free(T->bmp); free(T->lv8);
    memset(T, 0, sizeof *T);
}
static int lower_bound_hw(const hw_t *a, int n, hw_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}

/* ------------------------------------------------------------- solutions */
typedef struct { hw_t c[4]; } Quad;
static Quad *SOL; static long long NSOL, SOLCAP;
static void sol_push(const hw_t *c) {
    if (NSOL == SOLCAP) {
        SOLCAP = SOLCAP ? 2 * SOLCAP : 1024;
        SOL = realloc(SOL, sizeof(Quad) * (size_t)SOLCAP);
    }
    memcpy(SOL[NSOL++].c, c, sizeof(hw_t) * 4);
}
static void sort4(hw_t *r) {
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3 - a; b++)
            if (r[b] > r[b + 1]) { hw_t t = r[b]; r[b] = r[b + 1]; r[b + 1] = t; }
}
static int cmp_quad(const void *x, const void *y) {
    return memcmp(((const Quad *)x)->c, ((const Quad *)y)->c, sizeof(hw_t) * 4);
}
static void canon_class(const hw_t *q, hw_t *out) {
    hw_t best[4]; int have = 0;
    for (int d = 0; d < NDEC; d++) {
        hw_t r[4];
        for (int u = 0; u < 4; u++) r[u] = dec_code(q[u], d);
        sort4(r);
        if (!have || memcmp(r, best, sizeof best) < 0) { memcpy(best, r, sizeof best); have = 1; }
    }
    memcpy(out, best, sizeof best);
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: wil_hkt3 <n> [threads] [out]\n"); return 1; }
    N = atoi(argv[1]);
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    const char *outpath = (argc > 3 && argv[3][0] != '-') ? argv[3] : NULL;
    if (N < 3 || (N % 2) == 0) { fprintf(stderr, "n must be odd and >= 3\n"); return 1; }
    H = (N - 1) / 2;
    if (N > HKT_NBITS - 1 || H > HKT_HBITS) {
        fprintf(stderr, "n = %d needs wider words than this build (%d-bit codes, %d-bit rows); %s\n",
                N, HKT_HBITS, HKT_NBITS,
                HKT_HBITS == 32 ? "rebuild with -DHKT_WIDE" : "n <= 127 is the limit");
        return 1;
    }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    binom_init();
    cos_init();
    if (!hkt_build()) { fprintf(stderr, "hkt_build failed\n"); return 1; }
    {   hw_t hm = hkt_hmask();
        uint64_t seed = 12345u;
        for (int trial = 0; trial < 4096; trial++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; hw_t x = (hw_t)(seed >> 8) & hm;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; hw_t y = (hw_t)(seed >> 8) & hm;
            if (nw_tmap(x ^ y) != (nw_tmap(x) ^ nw_tmap(y))) {
                fprintf(stderr, "n=%d: the t map is not linear\n", N);
                return 1;
            }
        }
    }
    for (int k = 1; k <= H; k++)
        if (((FIXED_[k] + N) & 1) != 0) {
            fprintf(stderr, "n=%d: FIXED_[%d]=%d breaks the mod-4 invariant\n", N, k, FIXED_[k]);
            return 1;
        }
    if (!fwd_init()) { fprintf(stderr, "n=%d: stage-1 forward matrix failed\n", N); return 1; }
    cols_init();
    cols_selfcheck();
    fold_selfcheck();
    solve_selfcheck();
    dec_init();
    int fourn = 4 * N;
    printf("n=%d: 4n = %d, H = %d, %d folded decimations; %d-bit codes, %d-bit rows, %d lanes\n",
           N, fourn, H, NDEC, HKT_HBITS, HKT_NBITS, LANES);

    Decomp dec[256];
    int nd = decomps(dec, 256);
    int i0 = 0, i1 = 1;
    {   const char *ab = getenv("WIL_AB");
        if (ab && strlen(ab) == 2 && ab[0] >= '0' && ab[0] <= '3' && ab[1] >= '0' && ab[1] <= '3' && ab[0] < ab[1]) {
            i0 = ab[0] - '0'; i1 = ab[1] - '0';
        }
    }
    int o0 = -1, o1 = -1;
    for (int u = 0; u < 4; u++) if (u != i0 && u != i1) { if (o0 < 0) o0 = u; else o1 = u; }
    typedef struct { int sa, sb, pa, pb; int nmem; int mem[16]; } Group;
    Group grp[256]; int ng = 0;
    for (int i = 0; i < nd; i++) {
        int g;
        for (g = 0; g < ng; g++)
            if (grp[g].sa == dec[i].s[i0] && grp[g].sb == dec[i].s[i1]) break;
        if (g == ng) {
            grp[g].sa = dec[i].s[i0]; grp[g].sb = dec[i].s[i1];
            grp[g].pa = dec[i].p[i0]; grp[g].pb = dec[i].p[i1];
            grp[g].nmem = 0; ng++;
        }
        if (grp[g].nmem < 16) grp[g].mem[grp[g].nmem++] = i;
    }
    printf("   %d decompositions a >= b >= c >= d, %d (a, b) groups (enumerating positions %d and %d)\n",
           nd, ng, i0, i1);
    for (int g = 0; g < ng; g++) {
        printf("   group %d: (%d, %d) with", g, grp[g].sa, grp[g].sb);
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            printf(" (%d, %d)", D->s[o0], D->s[o1]);
        }
        printf("   popcounts %d %d\n", grp[g].pa, grp[g].pb);
    }
    if (getenv("WIL_LIST")) return 0;
    int donly = -1;
    { const char *dv = getenv("WIL_DONLY"); if (dv) donly = atoi(dv); }
    { const char *v = getenv("WIL_NLEV"); if (v) { NLEV = atoi(v); if (NLEV < 2) NLEV = 2; if (NLEV > 256) NLEV = 256; } }
    { const char *v = getenv("WIL_SAMPLE"); if (v) { SAMPLE = atoi(v); if (SAMPLE < 1) SAMPLE = 1; } }
    { const char *v = getenv("WIL_ENUM_ONLY"); if (v && atoi(v)) ENUM_ONLY = 1; }
    if (SAMPLE > 1) printf("   sampling: pair filter and solve on every %d-th chunk of %d A codes\n", SAMPLE, ACHUNK);
    if (NLEV != 256) printf("   B index quantised to %d levels\n", NLEV);
    if (ENUM_ONLY) printf("   enumeration only: canonical A and kept B per group, no index, no solve\n");

    double t0 = wnow();
    {
        int kk = -1;
        const char *v = getenv("WIL_CCK"); if (v) kk = atoi(v);
        const char *nc = getenv("WIL_NOCACHE"); if (nc && atoi(nc)) kk = 0;
        if (ENUM_ONLY) kk = 0;
        if (H > 30) { if (kk > 0) fprintf(stderr, "H > 30: the consistency table is off\n"); kk = 0; }
        if (kk < 0) kk = cc_choose_k();
        if (kk > S2_MAXFREE) kk = S2_MAXFREE;
        CCK = kk;
        /* G(t) in the table saves the mc bits of every pair that passes the
           test, at four more bytes per pattern.  Default on up to H = 26
           (512 MB); WIL_GTAB=0/1 overrides. */
        GTAB = (CCK > 0 && H <= 26) ? 1 : 0;
        const char *gv = getenv("WIL_GTAB"); if (gv && CCK > 0) GTAB = atoi(gv) ? 1 : 0;
    }
    if (CCK) {
        double tb = wnow();
        cc_build();
        printf("   consistency table: K=%d null-vector words%s per pattern, %.1f MB, %lld patterns nullity-capped, built in %.2fs\n",
               CCK, GTAB ? " + G(t)" : "", (double)CCS * 4.0 * (double)((size_t)1 << H) / 1048576.0, CC_REJECTED, wnow() - tb);
    } else {
        printf("   no consistency table: every survivor goes to the elimination\n");
    }
    fflush(stdout);

    const hw_t maskH = hkt_hmask();
    long long tot_pairs = 0, tot_pass = 0, tot_raw = 0, tot_filt = 0, tot_found = 0, tot_capped = 0, tot_A = 0;
    double max_index_mb = 0, max_table_mb = 0;

    for (int g = 0; g < ng; g++) {
        if (donly >= 0 && g != donly) continue;
        double tg = wnow();
        int pa = grp[g].pa, pb = grp[g].pb;
        int equal_ab = (grp[g].sa == grp[g].sb);
        memset(ALLOWD, 0, sizeof ALLOWD);
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            ALLOWD[D->p[o0]] |= HW_ONE << D->p[o1];
            ALLOWD[D->p[o1]] |= HW_ONE << D->p[o0];
        }
        Table T;
        double ttb = wnow();
        table_build(&T, pb, fourn, ENUM_ONLY);
        double idx_mb = index_bytes(T.n, NLEV) / 1048576.0;
        if (idx_mb > max_index_mb) max_index_mb = idx_mb;
        if (T.table_mb > max_table_mb) max_table_mb = T.table_mb;
        long long ncombA = BINOM[H][pa];
        printf("   group %d (%d, %d): B table %d of %lld codes (%.1f%%), %.0f MB of rows, index %.0f MB at %d levels, built in %.1fs; A over %lld codes%s\n",
               g, grp[g].sa, grp[g].sb, T.n, T.ncomb, 100.0 * T.n / (double)T.ncomb, T.table_mb, idx_mb, NLEV,
               wnow() - ttb, ncombA, equal_ab ? ", B >= A" : "");
        if (ENUM_ONLY) {
            printf("   group %d: index MB at NLEV 256/128/64/32/16/8: %.0f / %.0f / %.0f / %.0f / %.0f / %.0f; %zu bytes per row; survivor list %.0f MB per thread (virtual)\n",
                   g, index_bytes(T.n, 256) / 1048576.0, index_bytes(T.n, 128) / 1048576.0,
                   index_bytes(T.n, 64) / 1048576.0, index_bytes(T.n, 32) / 1048576.0,
                   index_bytes(T.n, 16) / 1048576.0, index_bytes(T.n, 8) / 1048576.0,
                   sizeof(hw_t) + sizeof(BRec), (double)T.n * sizeof(int) / 1048576.0);
        }
        fflush(stdout);
        if (!T.n && !ENUM_ONLY) { table_free(&T); continue; }

        long long nA = 0, pairs = 0, pass = 0, raw = 0, filt = 0, found = 0, capped = 0;
        long long nAs = 0, pairs256 = 0; double tsolve = 0, tscan = 0, tfilt = 0;
#pragma omp parallel
        {
            int8_t pafa[64]; float psda[64];
            hw_t tb[2 * NW_CHUNK + 8]; int ta[64];
            hw_t pend_t[LANES], pend_X[LANES], pend_d0[LANES], pend_G[LANES]; int pend_b[LANES]; int npend = 0;
            int *surv = malloc(sizeof(int) * (size_t)(T.n + 1));
            long long lA = 0, lpairs = 0, lpass = 0, lraw = 0, lfilt = 0, lfound = 0, lcapped = 0;
            long long lAs = 0, lpairs256 = 0; double lsolve = 0, lscan = 0, lfilt_t = 0;
#pragma omp for schedule(dynamic, 1)
            for (long long c0 = 0; c0 < ncombA; c0 += ACHUNK) {
              long long chi = c0 + ACHUNK; if (chi > ncombA) chi = ncombA;
              int solve_this = !ENUM_ONLY && ((c0 / ACHUNK) % SAMPLE) == 0;
              hw_t code = unrank_colex(c0, pa);
              double tc0 = tcpun();
              for (long long ci = c0; ci < chi; ci++) {
                hw_t am = code;
                if (ci + 1 < chi) code = gosper_next(code);
                if (!a_is_canonical(am)) continue;
                paf_mask(sym_mask(am), pafa);
                if (psd_all(pafa, psda) > fourn + 1e-6) continue;
                lA++;
                if (!solve_this) continue;
                lAs++;
                double ts0 = tcpun(); lscan += ts0 - tc0;
                hw_t raraw; nw_arow(pafa, ta, &raraw);
                hw_t rat = nw_tmap(raraw);
                hw_t wa1 = 0, ta0 = 0;
                for (int k = 1; k <= H; k++) {
                    if (ta[k] & 1) ta0 |= HW_ONE << (k - 1);
                    if (ta[k] & 2) wa1 |= HW_ONE << (k - 1);
                }
                const hw_t xa = wa1, da = raraw ^ M0W;   /* X = xa ^ wb1 ^ (ta0 & ~(da ^ rbraw)) */
                /* pair filter: rows whose level at every frequency is at most
                   the level of 4n - PSD_A; for a = b only rows at or above A */
                int nsurv = 0;
                {
                    const uint64_t *bp[64]; int nbp = 0, dead = 0;
                    for (int k = 0; k < H && !dead; k++) {
                        double cap = (double)fourn - psda[k];
                        if (cap < 0) { dead = 1; break; }
                        int t = lev_of(cap, fourn);
                        if (t >= NLEV - 1) continue;
                        bp[nbp++] = T.bmp + (((size_t)k * NLEV) + t) * T.cwords;
                    }
                    int lo = equal_ab ? lower_bound_hw(T.code, T.n, am) : 0;
                    int w0 = lo >> 6;
                    if (!dead) {
                        uint64_t lomask = (lo & 63) ? (~0ULL << (lo & 63)) : ~0ULL;
                        int wlast = (T.n - 1) >> 6;
                        uint64_t himask = (T.n & 63) ? ((1ULL << (T.n & 63)) - 1) : ~0ULL;
                        int w = w0 & ~3;
                        for (; w < T.cwords; w += 4) {
                            __m256i fv = _mm256_set1_epi64x(-1);
                            for (int i = 0; i < nbp; i++)
                                fv = _mm256_and_si256(fv, _mm256_loadu_si256((const __m256i *)(bp[i] + w)));
                            if (_mm256_testz_si256(fv, fv)) continue;
                            uint64_t v4[4];
                            _mm256_storeu_si256((__m256i *)v4, fv);
                            for (int q = 0; q < 4; q++) {
                                int ww = w + q;
                                if (ww < w0 || ww > wlast) continue;
                                uint64_t v = v4[q];
                                if (ww == w0) v &= lomask;
                                if (ww == wlast) v &= himask;
                                while (v) {
                                    int b = __builtin_ctzll(v);
                                    v &= v - 1;
                                    surv[nsurv++] = (ww << 6) + b;
                                }
                            }
                        }
                    }
                }
                lpairs += nsurv;
                double tf1 = tcpun(); lfilt_t += tf1 - ts0;
                if (T.lv8) {
                    uint8_t capl[64];
                    for (int k = 0; k < H; k++) capl[k] = (uint8_t)lev256((double)fourn - psda[k], fourn);
                    for (int i = 0; i < nsurv; i++) {
                        const uint8_t *l = T.lv8 + (size_t)surv[i] * H;
                        int ok = 1;
                        for (int k = 0; k < H && ok; k++) if (l[k] > capl[k]) ok = 0;
                        lpairs256 += ok;
                    }
                }

#define WIL_EMIT(NCD, CANDS, SBITS, BIDX)                                      \
                do {                                                           \
                    nw_t wb = hkt_symword(T.code[BIDX]);                       \
                    for (int cc = 0; cc < (NCD); cc++) {                       \
                        hw_t vv = (CANDS)[cc];                                 \
                        hw_t codeC = (~vv) & maskH;                            \
                        hw_t codeD = ((SBITS) ^ vv) & maskH;                   \
                        nw_t wc = hkt_symword(codeC), wd = hkt_symword(codeD); \
                        int good = 1;                                          \
                        for (int k = 1; k <= H && good; k++) {                 \
                            int t2 = (int)pafa[k - 1] + 3 * N                  \
                                   - 2 * (nw_popcount(wb ^ hkt_rotk(wb, k))    \
                                        + nw_popcount(wc ^ hkt_rotk(wc, k))    \
                                        + nw_popcount(wd ^ hkt_rotk(wd, k)));  \
                            if (t2 != 0) good = 0;                             \
                        }                                                      \
                        lfilt++;                                               \
                        if (!good) continue;                                   \
                        lfound++;                                              \
                        hw_t q[4] = { am, T.code[BIDX], codeC, codeD };        \
                        _Pragma("omp critical")                                \
                        sol_push(q);                                           \
                    }                                                          \
                } while (0)

/* LANES queued pairs through one elimination: columns straight into the
   transposed layout, bb = G(t) ^ X per lane, dead lanes zero. */
#define WIL_FLUSH()                                                            \
                do {                                                           \
                    hw_t ct[64 * LANES], pb8[64 * LANES], pm8[LANES], bbv[LANES]; \
                    hw_t q0[LANES] = {0}, q1[LANES] = {0};                      \
                    for (int L = npend; L < LANES; L++) { pend_t[L] = 0; pend_X[L] = 0; pend_d0[L] = 0; pend_G[L] = 0; } \
                    for (int k = 0; k < H; k++)                                 \
                        for (int L = npend; L < LANES; L++) ct[(size_t)k * LANES + L] = 0; \
                    nw_cols4(pend_t, ct, 0);                                    \
                    if (LANES > 4 && npend > 4) nw_cols4(pend_t + 4, ct, 4);    \
                    if (GTAB) {                                                 \
                        for (int L = 0; L < LANES; L++)                         \
                            bbv[L] = L < npend ? (pend_G[L] ^ pend_X[L]) & maskH : (hw_t)0; \
                    } else {                                                    \
                        nw_mcbits4(pend_t, q0, q1);                             \
                        if (LANES > 4 && npend > 4) nw_mcbits4(pend_t + 4, q0 + 4, q1 + 4); \
                        for (int L = 0; L < LANES; L++)                         \
                            bbv[L] = L < npend ? (q1[L] ^ (pend_d0[L] & ~q0[L]) ^ pend_X[L]) & maskH : (hw_t)0; \
                    }                                                           \
                    wil_elim(ct, bbv, pb8, pm8);                                \
                    hw_t part[LANES], fm[LANES], bas0[LANES]; int nf[LANES];    \
                    wil_tail(ct, pb8, bbv, part, fm, bas0, nf);                 \
                    for (int L = 0; L < npend; L++) {                           \
                        if (bbv[L] & ~pm8[L]) continue;                         \
                        if (nf[L] > S2_MAXFREE) { lcapped++; continue; }        \
                        hw_t cand[HKT_MAXC]; int nc;                            \
                        if (nf[L] <= 1) {                                       \
                            cand[0] = part[L]; nc = 1;                          \
                            if (nf[L]) { cand[1] = part[L] ^ bas0[L]; nc = 2; } \
                        } else {                                                \
                            hw_t colL[64], pivL[64];                            \
                            for (int k = 0; k < H; k++) {                       \
                                colL[k] = ct[(size_t)k * LANES + L];            \
                                pivL[k] = pb8[(size_t)k * LANES + L];           \
                            }                                                   \
                            nc = hkt_tail_m(colL, bbv[L], pm8[L], pivL, cand);  \
                        }                                                       \
                        lraw += nc;                                             \
                        int nk = wil_rowfilter(cand, nc, pend_t[L], maskH);     \
                        WIL_EMIT(nk, cand, pend_t[L], pend_b[L]);               \
                    }                                                           \
                    npend = 0;                                                  \
                } while (0)

                /* Two passes per chunk of survivors: the first forms each
                   pattern and prefetches its table word, with the B records
                   of the next chunk prefetched alongside; the second does the
                   test.  The next chunk's patterns are formed while this one
                   is tested, so every table line has a chunk of work to
                   arrive in. */
                hw_t *tcur = tb, *tnext = tb + NW_CHUNK;
                int mch = nsurv < NW_CHUNK ? nsurv : NW_CHUNK;
                for (int u = 0; u < mch; u++) {
                    hw_t t = rat ^ T.rec[surv[u]].rbt;
                    tcur[u] = t;
                    if (CCK) __builtin_prefetch(CC + (size_t)t * CCS);
                    if (NW_CHUNK + u < nsurv) __builtin_prefetch(&T.rec[surv[NW_CHUNK + u]]);
                }
                for (int s0i = 0; s0i < nsurv; s0i += NW_CHUNK) {
                    int s1i = s0i + NW_CHUNK;
                    int mnext = nsurv - s1i; if (mnext > NW_CHUNK) mnext = NW_CHUNK;
                    int s2i = s1i + NW_CHUNK;
                    int m2 = nsurv - s2i; if (m2 > NW_CHUNK) m2 = NW_CHUNK;
                    for (int u = 0; u < mch; u++) {
                        if (u < m2) __builtin_prefetch(&T.rec[surv[s2i + u]]);
                        if (u < mnext) {
                            hw_t t = rat ^ T.rec[surv[s1i + u]].rbt;
                            tnext[u] = t;
                            if (CCK) __builtin_prefetch(CC + (size_t)t * CCS);
                        }
                        hw_t t = tcur[u];
                        int b0i = surv[s0i + u];
                        const BRec *r = &T.rec[b0i];
                        hw_t d0 = da ^ r->rbraw;
                        hw_t X = xa ^ r->wb1 ^ (ta0 & ~d0);
                        hw_t gt = 0;
                        if (CCK) {
                            const uint32_t *e = CC + (size_t)t * CCS;
                            uint32_t e0 = e[0];
                            if (e0 == CC_REJECT) { lcapped++; continue; }
                            int bad = 0;
                            for (int i = 0; i < CCK; i++) {
                                uint32_t ei = i ? e[i] : e0;
                                uint32_t w = ei & (uint32_t)maskH;
                                if (!w) break;
                                if ((__builtin_parity(w & (uint32_t)X) ^ (ei >> 31)) & 1u) { bad = 1; break; }
                            }
                            if (bad) continue;
                            if (GTAB) gt = e[CCK];
                        }
                        lpass++;
                        pend_t[npend] = t; pend_X[npend] = X; pend_d0[npend] = d0; pend_b[npend] = b0i;
                        pend_G[npend] = gt;
                        if (++npend == LANES) WIL_FLUSH();
                    }
                    { hw_t *sw = tcur; tcur = tnext; tnext = sw; }
                    mch = mnext;
                }
                if (npend) WIL_FLUSH();
#undef WIL_EMIT
#undef WIL_FLUSH
                tc0 = tcpun(); lsolve += tc0 - tf1;
              }
              lscan += tcpun() - tc0;
            }
            free(surv);
#pragma omp atomic
            nAs += lAs;
#pragma omp atomic
            pairs256 += lpairs256;
#pragma omp atomic
            tsolve += lsolve;
#pragma omp atomic
            tfilt += lfilt_t;
#pragma omp atomic
            tscan += lscan;
#pragma omp atomic
            nA += lA;
#pragma omp atomic
            pairs += lpairs;
#pragma omp atomic
            pass += lpass;
#pragma omp atomic
            raw += lraw;
#pragma omp atomic
            filt += lfilt;
#pragma omp atomic
            found += lfound;
#pragma omp atomic
            capped += lcapped;
        }
        table_free(&T);
        if (ENUM_ONLY) {
            printf("   group %d (%d, %d): %lld canonical A past the ceiling (scan %.1f thread-s), %.1fs\n",
                   g, grp[g].sa, grp[g].sb, nA, tscan, wnow() - tg);
            fflush(stdout);
            tot_A += nA;
            continue;
        }
        printf("   group %d (%d, %d): %lld canonical A, %lld pairs, %lld (%.1f%%) past the consistency test, %lld lifts, %lld row-sum matches, %lld quadruples, %lld capped, %.1fs\n",
               g, grp[g].sa, grp[g].sb, nA, pairs, pass, pairs ? 100.0 * pass / (double)pairs : 0.0,
               raw, filt, found, capped, wnow() - tg);
        printf("   group %d: thread-seconds scan %.1f, filter %.1f, solve %.1f; %.0f ns per pair\n",
               g, tscan, tfilt, tsolve, pairs ? 1e9 * (tsolve + tfilt) / (double)pairs : 0.0);
        if (SAMPLE > 1)
            printf("   group %d: sampled %lld of %lld canonical A: %lld pairs (%lld at 256 levels), %.1f pairs per A; projected %.3g pairs, %.3g at 256 levels, filter+solve %.0f thread-s\n",
                   g, nAs, nA, pairs, pairs256, nAs ? (double)pairs / (double)nAs : 0.0,
                   nAs ? (double)pairs * (double)nA / (double)nAs : 0.0,
                   nAs ? (double)pairs256 * (double)nA / (double)nAs : 0.0,
                   nAs ? (tsolve + tfilt) * (double)nA / (double)nAs : 0.0);
        fflush(stdout);
        tot_A += nA; tot_pairs += pairs; tot_pass += pass; tot_raw += raw; tot_filt += filt;
        tot_found += found; tot_capped += capped;
    }

    double tsearch = wnow() - t0;
    if (ENUM_ONLY) {
        printf("   canonical A rows passing the ceiling: %lld\n", tot_A);
        printf("   largest B table %.0f MB of rows, largest index %.0f MB at %d levels\n",
               max_table_mb, max_index_mb, NLEV);
        printf("n=%d: enumeration only, %.2fs\n", N, tsearch);
        return 0;
    }
    Quad *keyed = malloc(sizeof(Quad) * (size_t)(NSOL + 1));
    Quad *orig  = malloc(sizeof(Quad) * (size_t)(NSOL + 1));
    long long ndistinct = 0;
    {
        typedef struct { Quad key, o; } KQ;
        KQ *kq = malloc(sizeof(KQ) * (size_t)(NSOL + 1));
        for (long long i = 0; i < NSOL; i++) {
            kq[i].o = SOL[i]; kq[i].key = SOL[i]; sort4(kq[i].key.c);
        }
        qsort(kq, (size_t)NSOL, sizeof(KQ), cmp_quad);
        for (long long i = 0; i < NSOL; i++) {
            if (i && !memcmp(kq[i].key.c, kq[i - 1].key.c, sizeof(hw_t) * 4)) continue;
            keyed[ndistinct] = kq[i].key; orig[ndistinct] = kq[i].o; ndistinct++;
        }
        free(kq);
    }
    long long nclasses = 0;
    {
        Quad *cl = malloc(sizeof(Quad) * (size_t)(ndistinct + 1));
        for (long long i = 0; i < ndistinct; i++) canon_class(keyed[i].c, cl[i].c);
        qsort(cl, (size_t)ndistinct, sizeof(Quad), cmp_quad);
        for (long long i = 0; i < ndistinct; i++)
            if (!i || memcmp(cl[i].c, cl[i - 1].c, sizeof(Quad))) nclasses++;
        free(cl);
    }
    printf("   canonical A rows passing the ceiling: %lld\n", tot_A);
    printf("   pairs %lld, past the consistency test %lld, lifts %lld, row-sum matches %lld, nullity-capped calls %lld\n",
           tot_pairs, tot_pass, tot_raw, tot_filt, tot_capped);
    if (tot_capped)
        printf("   WARNING: %lld solve calls were dropped by the nullity cap; the census is not exhaustive\n",
               tot_capped);
    printf("   largest B table %.0f MB of rows, largest index %.0f MB\n", max_table_mb, max_index_mb);
    {
        FILE *st = fopen("/proc/self/status", "r");
        char line[256];
        if (st) {
            while (fgets(line, sizeof line, st))
                if (!strncmp(line, "VmHWM:", 6)) { printf("   peak RSS %s", line + 6); break; }
            fclose(st);
        }
    }
    printf("n=%d: %lld quadruples found (%lld distinct), %lld inequivalent classes, %.2fs\n",
           N, tot_found, ndistinct, nclasses, tsearch);
    if (outpath) {
        FILE *f = fopen(outpath, "w");
        if (!f) { perror(outpath); return 1; }
        int8_t x[128];
        for (long long i = 0; i < ndistinct; i++) {
            for (int u = 0; u < 4; u++) {
                if (u) fputc(' ', f);
                sym_row(orig[i].c[u], x);
                for (int j = 0; j < N; j++) fputc(x[j] > 0 ? '+' : '-', f);
            }
            fputc('\n', f);
        }
        fclose(f);
        printf("   solutions -> %s\n", outpath);
    }
    return 0;
}
