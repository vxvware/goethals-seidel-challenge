/* hkt_census.c -- the shared HKT census engine for the Williamson and good
 * families, written 2026-11-23 from research/starter/williamson/wil_hkt3.c
 * and the 2026-11-20 slice-cache revision wil_hkt4.c, so that the orders 65
 * to 127 can run in pieces on a machine with tens of GB.
 *
 *   family wil    A, B, C, D symmetric, a^2 + b^2 + c^2 + d^2 = 4n; A runs
 *                 over the symmetric codes least under the phi(n)/2 folded
 *                 multipliers, B over a table of symmetric codes (B >= A when
 *                 a = b).  This is wil_hkt4.c.
 *   family good   A skew (a_0 = 1, a_j = -a_{n-j}), B, C, D symmetric,
 *                 1 + b^2 + c^2 + d^2 = 4n; A runs over all 2^H skew codes
 *                 least under the phi(n) multipliers (t and -t differ on a
 *                 skew row), B over the symmetric codes of the largest sum b,
 *                 and every (c, d) with c^2 + d^2 = 4n - 1 - b^2 is accepted
 *                 in one pass.  This is good_hkt.c's search on the wide words
 *                 and the slice cache of wil_hkt4.c.  The solver reads A only
 *                 through PAF_A, which is n mod 4 at every lag for a skew row
 *                 as for a symmetric one, so the solve and the exact check are
 *                 the Williamson ones.
 *
 * Both families share the 64-bit codes and 128-bit rows (n <= 127), the
 * slice cache, the tiled pair filter and the exact check of every candidate.
 *
 * Units.  A group (the row sums of A and B) is cut into units: B piece i of
 * HKT_BPIECES (a range of the colex ranks of the B codes) against A part j of
 * HKT_APARTS (the chunks of 4096 A codes whose index is j mod HKT_APARTS),
 * over slice range l of HKT_SLICES (the pairs whose pattern t has its top
 * bits in that range).  The units of a group partition its pairs, so their
 * outputs together are the output of the whole run.  A unit holds only its B
 * piece and its A part in memory, which bounds memory; slice ranges bound
 * the time of a unit at almost no cost, since every pattern belongs to one
 * slice and is eliminated once whichever unit meets it.  HKT_PLAN=1
 * estimates, from random samples of A and B codes, the kept rows and pairs
 * of every group and the split each needs under HKT_MEM_MB, and prints one
 * PLAN line per group; run_census.sh runs the units of a plan with restart
 * markers.  HKT_STOP=file: a unit that finds the file (checked every 64
 * slices) exits with status 3 and writes nothing.
 *
 * The stage-2 nullity cap (nullity > 6) is counted: a unit prints how many
 * solves it dropped, and a census is exhaustive only when every unit's count
 * is 0.  At composite orders the corank is higher; check the count.
 *
 *   build: gcc -O3 -march=native -fopenmp -DHKT_WIDE -I ../kkss -o hkt_census hkt_census.c -lm
 *   run:   ./hkt_census wil|good n [threads] [out]
 *   env:   HKT_DONLY=g     group g only (HKT_LIST=1 lists the groups and exits)
 *          HKT_BPIECES=p HKT_BPIECE=i HKT_APARTS=j HKT_APART=k
 *          HKT_SLICES=s HKT_SLICE=l   one unit
 *          HKT_PLAN=1 HKT_MEM_MB=m [HKT_PLAN_SAMPLES=s HKT_UNIT_PAIRS=q]   plan only
 *          HKT_STOP=file   stop request
 *          HKT_AB=ij (wil only), HKT_NLEV, HKT_NLEVC, HKT_TBITS, HKT_KNUL,
 *          HKT_PRE, HKT_EXACTPSD, HKT_ACHUNK tune speed and memory only;
 *          HKT_SAMPLE, HKT_ASAMPLE, HKT_REPEAT and HKT_ENUM_ONLY are
 *          measurements, not censuses (described below under their WIL_
 *          names).
 *
 * Output: one line per distinct quadruple, four words of n '+'/'-' in key
 * order (wil: the four rows by code; good: the skew row, then the three
 * symmetric rows by code), so a quadruple has one spelling whichever unit
 * found it; then a line "UNIT ... pairs= distinct= capped= seconds=" on
 * stdout, which a driver reads as the mark of a finished unit.
 *
 * The headers of wil_hkt4.c and wil_hkt3.c follow for the method; their
 * environment variables are spelled HKT_ here.
 */
/* wil_hkt4.c -- revision of wil_hkt3.c written 2026-11-20 as a performance
 * experiment.  The search space, the pair filter's pair set and the
 * candidates per pair are those of wil_hkt3.c; what changed is how often the
 * stage-2 elimination runs and how the pair filter reads its index.
 *
 *   slice cache    The stage-2 matrix of a pair depends only on its sign
 *                  pattern t, one of 2^H values, and the census visits many
 *                  more pairs than that (about 500 per pattern at n = 67).
 *                  The pairs are regrouped by the top bits of t into slices;
 *                  within a slice one elimination per pattern is cached with
 *                  its left null vectors and its solution map, and every
 *                  later pair with that pattern costs a parity test and, when
 *                  consistent, one application of the stored map.  See the
 *                  comment at "slice cache" below.
 *   B order        the B rows are ordered by class (top bits of T(RB)) so a
 *                  slice pairs each A with one block of B rows.
 *   pair filter    the bitmap index is stored per tile of 256 rows with a
 *                  few coarse levels, and each row it passes is compared
 *                  against the exact one-byte levels, so the pairs kept are
 *                  those of the NLEV-level filter of wil_hkt3.c at a
 *                  fraction of its index size.
 *   nullity 1      when the right null vector is ~t the two candidates are
 *                  (C, D) and (D, C); one exact check serves both.
 *
 * Needs -DHKT_WIDE (the 64-bit words carry the packed table entries).
 *
 *   build: gcc -O3 -march=native -fopenmp -DHKT_WIDE -o wil_hkt4 wil_hkt4.c -lm
 *   run:   ./wil_hkt4 n [threads] [out]
 *   env (besides WIL_LIST, WIL_ENUM_ONLY, WIL_DONLY, WIL_AB, WIL_NLEV of
 *   wil_hkt3.c; WIL_CCK and WIL_GTAB are gone):
 *          WIL_TBITS=b    low bits of t that index the slice table (default
 *                         18; 2^(H-b) slices).  Larger b: fewer passes over
 *                         the B index, a larger shared table.
 *          WIL_NLEVC=c    coarse levels of the bitmap index (default 16)
 *          WIL_KNUL=k     null vectors stored per pattern (default: the
 *                         largest nullity in a sample of 4096 patterns)
 *          WIL_PRE=1      nullity-1 orders: also store the two row-sum
 *                         parity functionals and drop a consistent pair
 *                         whose candidates cannot have an allowed row-sum
 *                         pair (acts only in groups with c = d mod 8)
 *          WIL_EXACTPSD=1 keep a pair only when PSD_A + PSD_B <= 4n + 0.01
 *                         on the stored float PSD rows; this changes the
 *                         pair count (about 4% fewer at 53), never the
 *                         quadruples
 *          WIL_SAMPLE=k   every k-th slice only; WIL_ASAMPLE=k every k-th A
 *          WIL_REPEAT=r   measurement only: each slice is processed r times
 *                         against a warm table, to time the per-pair path at
 *                         the pairs-per-pattern ratio of a larger order
 *          WIL_ACHUNK=a   A rows per work item inside a slice
 *
 * The header of wil_hkt3.c follows unchanged for the parts it still
 * describes (rows, the folded right side, the direct columns, the exact
 * check). */
#ifndef HKT_WIDE
#error "hkt_census.c is the wide build only: compile with -DHKT_WIDE"
#endif
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
#include <unistd.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(__AVX2__)
#error "hkt_census.c needs AVX2 (build with -march=native on an AVX2 host)"
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
/* skew row from a code: x_0 = 1, bit j-1 set means x_j = +1 and x_{n-j} = -1 */
static void skew_row(hw_t code, int8_t *x) {
    x[0] = 1;
    for (int j = 1; j <= H; j++) {
        int8_t v = (code >> (j - 1)) & 1 ? 1 : -1;
        x[j] = v; x[N - j] = (int8_t)-v;
    }
}
static inline nw_t skew_mask(hw_t code) {
    nw_t m = NW_ONE;
    for (int j = 1; j <= H; j++) {
        if ((code >> (j - 1)) & 1) m |= NW_ONE << j;
        else                       m |= NW_ONE << (N - j);
    }
    return m;
}
static int FAMILY = 0;            /* 0: Williamson (A symmetric), 1: good (A skew) */
static inline nw_t a_mask(hw_t code) { return FAMILY ? skew_mask(code) : sym_mask(code); }
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

/* -------------------------------------------------- decimation on codes

   x'_{tj} = x_j for a unit t of Z_n.  The source position j lands on
   q = tj mod n; when q <= H the bit moves to q - 1, and when q > H the stored
   position is n - q, where a symmetric row repeats the value and a skew row
   negates it.  So the image of a code is a bit permutation (the same for both
   kinds) followed, for a skew row, by an XOR with the mask of the wrapped
   positions (DFLIP).  On symmetric rows t and -t act alike, so the Williamson
   family keeps the phi(n)/2 units 1..H; the good family keeps all phi(n),
   index 0 being t = 1, because t and -t differ on its skew row. */
static int NDEC;
static int *DECMAP;              /* [NDEC][H] */
static hw_t *DBYTE;              /* [NDEC][DNB][256] */
static hw_t *DFLIP;              /* [NDEC] */
static int DNB;
static int gcd_int(int a, int b) { while (b) { int q = a % b; a = b; b = q; } return a; }
static void dec_init(void) {
    int tmax = FAMILY ? N - 1 : H;
    int cnt = 0;
    for (int s = 1; s <= tmax; s++) if (gcd_int(s, N) == 1) cnt++;
    NDEC = cnt;
    DECMAP = malloc(sizeof(int) * (size_t)cnt * H);
    DFLIP = calloc((size_t)cnt, sizeof(hw_t));
    if (!DECMAP || !DFLIP) { fprintf(stderr, "out of memory for the decimation maps\n"); exit(1); }
    int i = 0;
    for (int s = 1; s <= tmax; s++) {
        if (gcd_int(s, N) != 1) continue;
        for (int j = 1; j <= H; j++) {
            int m = (int)(((long long)s * j) % N);
            if (m > H) { m = N - m; DFLIP[i] |= HW_ONE << (m - 1); }
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
static inline hw_t dec_a(hw_t code, int d) { return FAMILY ? dec_code(code, d) ^ DFLIP[d] : dec_code(code, d); }
/* Canonical A: the least code in its decimation orbit. */
static inline int a_is_canonical(hw_t code) {
    for (int d = 1; d < NDEC; d++)
        if (dec_a(code, d) < code) return 0;
    return 1;
}
/* the decimation tables against the definition, on random codes of both kinds */
static void dec_selfcheck(void) {
    const hw_t hm = hkt_hmask();
    uint64_t seed = 0x243F6A8885A308D3ULL ^ (uint64_t)N;
    int8_t x[128], y[128];
    for (int trial = 0; trial < 256; trial++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        hw_t c = (hw_t)(seed & hm);
        int ti = 0;
        for (int t = 1; t <= (FAMILY ? N - 1 : H); t++) {
            if (gcd_int(t, N) != 1) continue;
            for (int kind = 0; kind <= FAMILY; kind++) {
                if (kind) skew_row(c, x); else sym_row(c, x);
                for (int j = 0; j < N; j++) y[(int)(((long long)t * j) % N)] = x[j];
                hw_t img = kind ? dec_code(c, ti) ^ DFLIP[ti] : dec_code(c, ti);
                for (int j = 1; j <= H; j++) {
                    int bit = (int)((img >> (j - 1)) & 1u);
                    if ((y[j] > 0) != bit || y[0] != 1 || y[N - j] != (kind ? -y[j] : y[j])) {
                        fprintf(stderr, "n=%d: decimation table disagrees with the definition at t=%d\n", N, t);
                        exit(1);
                    }
                }
            }
            ti++;
        }
    }
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
/* Williamson: a^2 + b^2 + c^2 + d^2 = 4n, a >= b >= c >= d, all four rows
   symmetric.  Good: the skew row A sums to 1, so b^2 + c^2 + d^2 = 4n - 1
   with b >= c >= d over the symmetric rows; s[0] = 1 and p[0] = -1 stand for
   A, whose codes are all 2^H skew codes whatever their popcount. */
typedef struct { int s[4]; int p[4]; } Decomp;
static int decomps(Decomp *out, int maxout) {
    int nd = 0, fourn = 4 * N;
    int lim = (int)(sqrt((double)fourn) + 1e-9);
    if (FAMILY) {
        for (int b = 1; b <= lim; b += 2)
            for (int c = 1; c <= b; c += 2)
                for (int d = 1; d <= c; d += 2) {
                    if (b*b + c*c + d*d != fourn - 1 || nd >= maxout) continue;
                    Decomp *D = &out[nd];
                    D->s[0] = 1; D->s[1] = b; D->s[2] = c; D->s[3] = d; D->p[0] = -1;
                    int ok = 1;
                    for (int u = 1; u < 4; u++) {
                        D->p[u] = popcount_for(signed_sum(D->s[u]));
                        if (D->p[u] < 0) ok = 0;
                    }
                    if (ok) nd++;
                }
        return nd;
    }
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
static const char *STOPFILE = NULL;   /* HKT_STOP: checked every 64 slices */
static volatile int STOPPED = 0;
static int SAMPLE = 1;
static int ENUM_ONLY = 0;
static int ASAMPLE = 1, PRETEST = 0, REPEAT = 1;
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
/* ------------------------------------------------- slice cache (wil_hkt4)

   The stage-2 matrix, its left null space and its reduced form depend on the
   pattern t alone, and t = T(RA) ^ T(RB) takes only 2^H values while the
   census visits far more pairs than that (about 500 pairs per pattern at
   n = 67).  The pairs are therefore regrouped so that one elimination serves
   every pair with the same t.  The top MBITS bits of t split the pairs into
   2^MBITS slices; within a slice the low TBITS = H - MBITS bits index a
   table that is filled lazily, one elimination per pattern met, and holds

     hot   meta, G(t), and KNUL left null vectors w_i (zero-padded)
     cold  the solution map as columns: part = XOR of pcol[j] over the set
           bits j of bb, and the right null basis

   A pair reads the hot entry, forms bb = G(t) ^ X and is dropped when some
   parity(bb & w_i) is odd; a consistent pair reads the cold entry and gets
   the same candidates the per-pair elimination produces.  The B table is
   ordered by class (top MBITS bits of T(RB)), so slice phi pairs each A with
   the one block of class class(A) ^ phi.  A pattern whose nullity exceeds
   KNUL is solved on the spot for each of its pairs (the nullity cap of the
   solver, S2_MAXFREE, still applies and is still counted). */
static int MBITS = 0, TBITS = 0, KNUL = 1, HS = 4, CS = 40, GPACK = 1, WOFF = 1;
static int CPW = 0, CNHI = 0, CBAS = 0;   /* cold layout in words: packed columns, high rows, basis offset */
#define ENT_OVER 0x100u
#define ENT_BUSY 0x200u
typedef struct { hw_t G; int nul; hw_t w[64], pcol[68], basis[64]; } BigEnt;
static __m256i BSEL_[17];
static __m256i BSEL32_[4];
static void bsel_init(void) {
    for (int i = 0; i < 4; i++)
        BSEL32_[i] = _mm256_setr_epi32(1 << (8 * i), 1 << (8 * i + 1), 1 << (8 * i + 2), 1 << (8 * i + 3), 1 << (8 * i + 4), 1 << (8 * i + 5), 1 << (8 * i + 6), (int)(1u << (8 * i + 7)));
    for (int i = 0; i < 16; i++)
        BSEL_[i] = _mm256_setr_epi64x((long long)(1ULL << (4 * i)), (long long)(1ULL << (4 * i + 1)),
                                      (long long)(1ULL << (4 * i + 2)), (long long)(1ULL << (4 * i + 3)));
}
/* One elimination with the row transform carried along.  col[c] is a mask
   over rows (lags), bt[j] is column j of the accumulated row transform. */
static void ent_build(hw_t t, hw_t d0, BigEnt *E) {
    const hw_t hm = hkt_hmask();
    hw_t col[64], bt[64], q0, q1;
    int qc[64], fr[64], nfr = 0;
    nw_cols_scalar(t, col);
    nw_mcbits(t, &q0, &q1);
    E->G = (q1 ^ (d0 & ~q0)) & hm;
    for (int j = 0; j < H; j++) bt[j] = HW_ONE << j;
    hw_t pv = 0;
    for (int c = 0; c < H; c++) {
        hw_t m = col[c] & ~pv;
        if (!m) { qc[c] = -1; fr[nfr++] = c; continue; }
        int q = hw_ctz(m);
        pv |= HW_ONE << q; qc[c] = q;
        hw_t R = col[c] & ~(HW_ONE << q);
        col[c] = HW_ONE << q;
        for (int c2 = c + 1; c2 < H; c2++)
            col[c2] ^= R & ((hw_t)0 - ((col[c2] >> q) & 1u));
        for (int i = 0; i < nfr; i++) {
            int c2 = fr[i];
            col[c2] ^= R & ((hw_t)0 - ((col[c2] >> q) & 1u));
        }
        for (int j = 0; j < H; j++)
            bt[j] ^= R & ((hw_t)0 - ((bt[j] >> q) & 1u));
    }
    E->nul = nfr;
    int nw = 0;
    for (int q = 0; q < H; q++) {
        if ((pv >> q) & 1u) continue;
        hw_t w = 0;
        for (int j = 0; j < H; j++) w |= ((bt[j] >> q) & 1u) << j;
        E->w[nw++] = w;
    }
    for (int j = 0; j < 68; j++) E->pcol[j] = 0;
    for (int c = 0; c < H; c++) {
        int q = qc[c];
        if (q < 0) continue;
        for (int j = 0; j < H; j++) E->pcol[j] |= ((bt[j] >> q) & 1u) << c;
    }
    for (int i = 0; i < nfr; i++) {
        int f = fr[i];
        hw_t v = HW_ONE << f;
        for (int c = 0; c < H; c++)
            if (qc[c] >= 0 && ((col[f] >> qc[c]) & 1u)) v |= HW_ONE << c;
        E->basis[i] = v;
    }
}
static inline hw_t ent_part(hw_t bb, const hw_t *pcol) {
    const __m256i vb = _mm256_set1_epi64x((long long)bb);
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < H; j += 4) {
        __m256i s = BSEL_[j >> 2];
        __m256i m = _mm256_cmpeq_epi64(_mm256_and_si256(vb, s), s);
        acc = _mm256_xor_si256(acc, _mm256_and_si256(m, _mm256_loadu_si256((const __m256i *)(pcol + j))));
    }
    __m128i x = _mm_xor_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    return (hw_t)(_mm_cvtsi128_si64(x) ^ _mm_extract_epi64(x, 1));
}
/* Packed form of the solution map for the table.  The low 32 bits of every
   column are a uint32 (H of them, CPW words); a solution bit c >= 32 is kept
   as a row, a mask over the bits of bb (CNHI words); then the null basis. */
static void ent_pack(const BigEnt *E, hw_t *ce) {
    uint32_t *p32 = (uint32_t *)ce;
    for (int j = 0; j < 2 * CPW; j++) p32[j] = j < H ? (uint32_t)E->pcol[j] : 0u;
    for (int i = 0; i < CNHI; i++) {
        hw_t r = 0;
        for (int j = 0; j < H; j++) r |= ((E->pcol[j] >> (32 + i)) & 1u) << j;
        ce[CPW + i] = r;
    }
    for (int i = 0; i < KNUL; i++) ce[CBAS + i] = i < E->nul ? E->basis[i] : (hw_t)0;
}
static inline hw_t ent_part_pk(hw_t bb, const hw_t *ce) {
    const uint32_t *p32 = (const uint32_t *)ce;
    const __m256i vlo = _mm256_set1_epi32((int)(uint32_t)bb), vhi = _mm256_set1_epi32((int)(uint32_t)(bb >> 32));
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < H; j += 8) {
        __m256i s = BSEL32_[(j & 31) >> 3];
        __m256i m = _mm256_cmpeq_epi32(_mm256_and_si256(j < 32 ? vlo : vhi, s), s);
        acc = _mm256_xor_si256(acc, _mm256_and_si256(m, _mm256_loadu_si256((const __m256i *)(p32 + j))));
    }
    __m128i x = _mm_xor_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    uint64_t f = (uint64_t)_mm_cvtsi128_si64(x) ^ (uint64_t)_mm_extract_epi64(x, 1);
    hw_t part = (hw_t)(uint32_t)(f ^ (f >> 32));
    for (int i = 0; i < CNHI; i++) part |= (hw_t)hw_parity(bb & ce[CPW + i]) << (32 + i);
    return part;
}
static inline int ent_cands(hw_t part, const hw_t *basis, int nul, hw_t *cand) {
    int nc = 1 << nul;
    cand[0] = part;
    for (int i = 0; i < nul; i++) {
        int h = 1 << i;
        for (int k = 0; k < h; k++) cand[h + k] = cand[k] ^ basis[i];
    }
    return nc;
}
static int ent_choose_k(void) {
    hw_t hm = hkt_hmask();
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    int mx = 1;
    BigEnt E;
    for (int i = 0; i < 4096; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        ent_build((hw_t)(seed & hm), 0, &E);
        if (E.nul > mx) mx = E.nul;
    }
    return mx > S2_MAXFREE ? S2_MAXFREE : mx;
}
static double TSC_NS = 1.0;
static void tsc_calibrate(void) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    uint64_t t0 = __rdtsc();
    do { clock_gettime(CLOCK_MONOTONIC, &b); }
    while ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec) < 5e7);
    uint64_t t1 = __rdtsc();
    TSC_NS = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / (double)(t1 - t0);
}

typedef struct { hw_t rbt, rbraw, wb1; } BRec;
typedef struct {
    int n;                 /* real rows */
    int np;                /* rows with every class block padded to 256 */
    hw_t     *code;        /* [np], ascending inside a block, ~0 on pad rows */
    BRec     *rec;         /* [np] */
    int      *bs, *bc;     /* [2^MBITS] block start and real row count */
    uint64_t *bmp;         /* [np/256]{[H][NLEVC][4], [4]}: per tile of 256 rows, prefix-ored over the coarse levels, then the mask of real rows */
    uint8_t  *lv8;         /* [np][H] the level of every row at every frequency */
    float    *psdf;        /* [np][HF] the PSD itself, only with WIL_EXACTPSD=1 */
    long long ncomb;
    double table_mb;       /* code + rec */
} Table;

/* Two-stage pair filter.  The baseline keeps one bitmap per (frequency,
   level) over all rows and reads H of them end to end for every A.  Here the
   rows are in class blocks and an A reads one block per slice, so the index
   is stored tile by tile (256 rows): the H x NLEVC bitmaps of a tile are
   contiguous and a block's index stays in cache while the A rows of one class
   pass over it.  The bitmaps are coarse (NLEVC levels); every row they let
   through is then compared bytewise against the exact levels, so the
   surviving pairs are exactly those of the NLEV-level bitmaps. */
static int NLEVC = 16, EXACTPSD = 0, HF = 0;
static uint8_t CLEVT[256];
static inline int clev(int l) { return CLEVT[l]; }
static double index_bytes(long long n, int nlevc) {
    return (double)H * nlevc * (double)n / 8.0 + (double)H * (double)n;
}
/* Exact level test of one row: every byte of the row at most the cap.  The
   first load reads 32 bytes at the row; bytes past H there belong to the
   next row and are compared against 255.  For H > 32 a second load reads the
   last 32 bytes of the row and only its last H - 32 bytes carry a cap. */
static inline void lev_caps(const uint8_t *lev, __m256i *c0, __m256i *c1) {
    uint8_t capb[64];
    memset(capb, 255, 64);
    memcpy(capb, lev, (size_t)(H < 32 ? H : 32));
    if (H > 32) memcpy(capb + 64 - (H - 32), lev + 32, (size_t)(H - 32));
    *c0 = _mm256_loadu_si256((const __m256i *)capb);
    *c1 = _mm256_loadu_si256((const __m256i *)(capb + 32));
}
static inline int lev_ok(const uint8_t *lr, __m256i c0v, __m256i c1v) {
    __m256i e = _mm256_cmpeq_epi8(_mm256_max_epu8(_mm256_loadu_si256((const __m256i *)lr), c0v), c0v);
    if (H > 32)
        e = _mm256_and_si256(e, _mm256_cmpeq_epi8(_mm256_max_epu8(_mm256_loadu_si256((const __m256i *)(lr + H - 32)), c1v), c1v));
    return _mm256_movemask_epi8(e) == -1;
}
static void lev_selfcheck(void) {
    uint64_t seed = 0x3C6EF372FE94F82BULL ^ (uint64_t)N;
    uint8_t buf[3 * 64 + 64], cap[64];
    int npass = 0;
    for (int it = 0; it < 200000; it++) {
        for (int i = 0; i < (int)sizeof buf; i++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            buf[i] = (uint8_t)(seed >> 24);
        }
        const uint8_t *row = buf + H;               /* a row between two others */
        int nviol = it % 3;                         /* 0: passes; else that many violations */
        for (int k = 0; k < H; k++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            unsigned r = (unsigned)(seed >> 40) % 3u;
            cap[k] = (uint8_t)(row[k] > 252 ? 255 : row[k] + r);
        }
        for (int v = 0; v < nviol; v++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            int k = (int)((seed >> 33) % (uint64_t)H);
            if (row[k] > 0) cap[k] = (uint8_t)(row[k] - 1);
        }
        int ref = 1;
        for (int k = 0; k < H; k++) if (row[k] > cap[k]) ref = 0;
        __m256i c0, c1; lev_caps(cap, &c0, &c1);
        if (lev_ok(row, c0, c1) != ref) { fprintf(stderr, "n=%d: vector level test disagrees with the scalar one\n", N); exit(1); }
        npass += ref;
    }
    printf("   level-test check: 200000 random rows agree with the scalar comparison (%d pass)\n", npass);
}
#define B_PAD (~(hw_t)0)

/* count_only: enumerate and count the kept codes, build nothing. */
static void table_build(Table *T, int pb, long long rlo, long long rhi, int fourn, int count_only) {
    long long ncomb = rhi - rlo;
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
        lo += rlo; hi += rlo;
        if (lo > rhi) lo = rhi;
        if (hi > rhi) hi = rhi;
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
    if (total > 0x7fffffffLL - ((long long)256 << MBITS)) { fprintf(stderr, "B table too large for int indices\n"); exit(1); }
    T->n = (int)total;
    T->table_mb = (double)total * (sizeof(hw_t) + sizeof(BRec)) / 1048576.0;
    if (count_only) { free(lc); free(ln); return; }
    hw_t *code0 = malloc(sizeof(hw_t) * (size_t)(total + 1));
    {   long long off = 0;
        for (int t = 0; t < nth; t++) {
            memcpy(code0 + off, lc[t], sizeof(hw_t) * (size_t)ln[t]);
            off += ln[t]; free(lc[t]);
        }
    }
    free(lc); free(ln);
    /* the solver words of every row, then the class order */
    BRec *rec0 = malloc(sizeof(BRec) * (size_t)(total + 1));
#pragma omp parallel for schedule(dynamic, 1024)
    for (long long bi = 0; bi < total; bi++) {
        int8_t paf[64];
        paf_mask(sym_mask(code0[bi]), paf);
        hw_t rb = 0, wb1 = 0;
        for (int k = 1; k <= H; k++) {
            int u = (N - (int)paf[k - 1]) >> 2;
            if (u & 1) rb |= HW_ONE << (k - 1);
            if (u & 2) wb1 |= HW_ONE << (k - 1);
        }
        rec0[bi].rbt = nw_tmap(rb); rec0[bi].rbraw = rb; rec0[bi].wb1 = wb1;
    }
    int ncl = 1 << MBITS;
    T->bs = calloc((size_t)ncl + 1, sizeof(int)); T->bc = calloc((size_t)ncl + 1, sizeof(int));
    for (long long bi = 0; bi < total; bi++) T->bc[MBITS ? (int)(rec0[bi].rbt >> TBITS) : 0]++;
    long long np = 0;
    for (int c = 0; c < ncl; c++) { T->bs[c] = (int)np; np += (T->bc[c] + 255) & ~255; }
    T->np = (int)np;
    T->code = malloc(sizeof(hw_t) * (size_t)(np + 4));
    T->rec = aligned_alloc(64, ((size_t)(np + 4) * sizeof(BRec) + 63) & ~(size_t)63);
    size_t ntile = (size_t)(np >> 8);
    const size_t tstride = (size_t)H * NLEVC * 4 + 4;   /* the last four words mark the real rows */
    size_t bmpn = ntile * tstride;
    T->bmp = calloc(bmpn + 4, sizeof(uint64_t));
    T->lv8 = calloc((size_t)np * H + 64, 1);
    if (EXACTPSD) { T->psdf = calloc((size_t)np * HF + 8, sizeof(float)); if (!T->psdf) { fprintf(stderr, "out of memory for the exact PSD rows\n"); exit(1); } }
    if (!T->code || !T->rec || !T->bmp || !T->lv8) {
        fprintf(stderr, "out of memory for the B table (%lld rows, index %.1f GB)\n",
                total, bmpn * 8.0 / (1024.0 * 1024.0 * 1024.0));
        exit(1);
    }
    for (long long i = 0; i < np; i++) T->code[i] = B_PAD;
    memset(T->rec, 0, (size_t)np * sizeof(BRec));
    {   int *fill = calloc((size_t)ncl + 1, sizeof(int));
        for (long long bi = 0; bi < total; bi++) {
            int c = MBITS ? (int)(rec0[bi].rbt >> TBITS) : 0;
            int pos = T->bs[c] + fill[c]++;
            T->code[pos] = code0[bi]; T->rec[pos] = rec0[bi];
        }
        free(fill);
    }
    free(code0); free(rec0);
#pragma omp parallel for schedule(dynamic, 4)
    for (long long tile = 0; tile < (long long)ntile; tile++) {
        int8_t paf[64]; float psd[64];
        uint64_t *tb = T->bmp + (size_t)tile * tstride;
        for (long long bi = tile * 256; bi < (tile + 1) * 256; bi++) {
            if (T->code[bi] == B_PAD) continue;
            paf_mask(sym_mask(T->code[bi]), paf);
            psd_all(paf, psd);
            int r = (int)(bi & 255);
            tb[(size_t)H * NLEVC * 4 + (r >> 6)] |= 1ULL << (r & 63);
            for (int k = 0; k < H; k++) {
                int lv = lev_of(psd[k], fourn);
                T->lv8[(size_t)bi * H + k] = (uint8_t)lv;
                if (T->psdf) T->psdf[(size_t)bi * HF + k] = psd[k];
                tb[((size_t)k * NLEVC + clev(lv)) * 4 + (r >> 6)] |= 1ULL << (r & 63);
            }
        }
        for (int k = 0; k < H; k++)
            for (int c = 1; c < NLEVC; c++)
                for (int q = 0; q < 4; q++)
                    tb[((size_t)k * NLEVC + c) * 4 + q] |= tb[((size_t)k * NLEVC + c - 1) * 4 + q];
    }
}
static void table_free(Table *T) {
    free(T->code); free(T->rec); free(T->bmp); free(T->lv8); free(T->psdf); free(T->bs); free(T->bc);
    memset(T, 0, sizeof *T);
}
static int lower_bound_hw(const hw_t *a, int n, hw_t key) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < key) lo = mid + 1; else hi = mid; }
    return lo;
}

/* Known-answer check of the cached path: the entry built for the pattern of
   a synthetic (C, D, B) triple must call the pair consistent and list (C, D)
   among its candidates. */
static void ent_selfcheck(void) {
    const hw_t hm = hkt_hmask();
    uint64_t seed = 0xBB67AE8584CAA73BULL ^ (uint64_t)N;
    int checked = 0, capped = 0;
    BigEnt *E = malloc(sizeof(BigEnt));
    for (int it = 0; it < 2048; it++) {
        int8_t pc[64], pd[64], pbv[64], pa[64];
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; hw_t cC = (hw_t)(seed & hm);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; hw_t cD = (hw_t)(seed & hm);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; hw_t cB = (hw_t)(seed & hm);
        paf_mask(sym_mask(cC), pc); paf_mask(sym_mask(cD), pd); paf_mask(sym_mask(cB), pbv);
        int fits = 1;
        for (int k = 0; k < H; k++) {
            int v = -((int)pc[k] + (int)pd[k]) - (int)pbv[k];
            if (v < -127 || v > 127) fits = 0;
            pa[k] = (int8_t)v;
        }
        if (!fits) continue;
        int ta[64]; hw_t raraw; nw_arow(pa, ta, &raraw);
        hw_t rat = nw_tmap(raraw), wa1 = 0, ta0 = 0, rbraw = 0, wb1 = 0;
        for (int k = 1; k <= H; k++) {
            if (ta[k] & 1) ta0 |= HW_ONE << (k - 1);
            if (ta[k] & 2) wa1 |= HW_ONE << (k - 1);
            int u = (N - (int)pbv[k - 1]) >> 2;
            if (u & 1) rbraw |= HW_ONE << (k - 1);
            if (u & 2) wb1 |= HW_ONE << (k - 1);
        }
        hw_t t = rat ^ nw_tmap(rbraw);
        hw_t d0 = (raraw ^ M0W) ^ rbraw;
        hw_t X = wa1 ^ wb1 ^ (ta0 & ~d0);
        ent_build(t, d0, E);
        if (E->G != (nw_G(t) & hm)) { fprintf(stderr, "n=%d: cached G(t) disagrees with nw_G\n", N); exit(1); }
        hw_t bb = (E->G ^ X) & hm;
        for (int i = 0; i < E->nul; i++)
            if (hw_parity(bb & E->w[i])) {
                fprintf(stderr, "n=%d: cached path calls a known answer inconsistent (t=%llu)\n", N, (unsigned long long)t);
                exit(1);
            }
        if (E->nul > S2_MAXFREE) { capped++; continue; }
        hw_t cand[HKT_MAXC], pk[80];
        hw_t part = ent_part(bb, E->pcol);
        if (E->nul <= KNUL) {
            ent_pack(E, pk);
            if (ent_part_pk(bb, pk) != part) { fprintf(stderr, "n=%d: packed solution map disagrees with the column form\n", N); exit(1); }
        }
        int nc = ent_cands(part, E->basis, E->nul, cand), hit = 0;
        for (int i = 0; i < nc; i++)
            if (((~cand[i]) & hm) == cC && ((t ^ cand[i]) & hm) == cD) hit = 1;
        if (!hit) {
            fprintf(stderr, "n=%d: cached path misses the known answer (C=%llu D=%llu B=%llu t=%llu, %d candidates)\n",
                    N, (unsigned long long)cC, (unsigned long long)cD, (unsigned long long)cB, (unsigned long long)t, nc);
            exit(1);
        }
        checked++;
    }
    free(E);
    printf("   cached-path known-answer check: %d synthetic (C, D, B) triples recovered, %d skipped by the nullity cap\n",
           checked, capped);
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
/* The key of a quadruple: the rows that may trade places sorted by code.
   Williamson: all four rows; good: the three symmetric rows, the skew row
   stays in front.  The output is written in key order, so a quadruple has one
   spelling whichever run found it. */
static void sort_key(hw_t *r) {
    if (!FAMILY) { sort4(r); return; }
    for (int a = 1; a < 3; a++)
        for (int b = 1; b < 4 - a; b++)
            if (r[b] > r[b + 1]) { hw_t t = r[b]; r[b] = r[b + 1]; r[b + 1] = t; }
}
static int cmp_quad(const void *x, const void *y) {
    return memcmp(((const Quad *)x)->c, ((const Quad *)y)->c, sizeof(hw_t) * 4);
}
static void canon_class(const hw_t *q, hw_t *out) {
    hw_t best[4]; int have = 0;
    for (int d = 0; d < NDEC; d++) {
        hw_t r[4];
        r[0] = dec_a(q[0], d);
        for (int u = 1; u < 4; u++) r[u] = dec_code(q[u], d);
        sort_key(r);
        if (!have || memcmp(r, best, sizeof best) < 0) { memcpy(best, r, sizeof best); have = 1; }
    }
    memcpy(out, best, sizeof best);
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    if (argc < 3 || (strcmp(argv[1], "wil") && strcmp(argv[1], "good"))) {
        fprintf(stderr, "usage: hkt_census wil|good <n> [threads] [out]\n");
        return 1;
    }
    FAMILY = !strcmp(argv[1], "good");
    argc--; argv++;
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
    dec_selfcheck();
    int fourn = 4 * N;
    printf("%s n=%d: 4n = %d, H = %d, %d %sdecimations; %d-bit codes, %d-bit rows, %d lanes\n",
           FAMILY ? "good" : "williamson", N, fourn, H, NDEC, FAMILY ? "" : "folded ",
           HKT_HBITS, HKT_NBITS, LANES);

    Decomp dec[256];
    int nd = decomps(dec, 256);
    int i0 = 0, i1 = 1;
    {   const char *ab = getenv("HKT_AB");
        if (ab && !FAMILY && strlen(ab) == 2 && ab[0] >= '0' && ab[0] <= '3' && ab[1] >= '0' && ab[1] <= '3' && ab[0] < ab[1]) {
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
    if (getenv("HKT_LIST")) return 0;
    int donly = -1;
    { const char *dv = getenv("HKT_DONLY"); if (dv) donly = atoi(dv); }
    STOPFILE = getenv("HKT_STOP");
    /* one unit of work: B piece i of P (a range of the colex ranks of the B
       codes) against A part j of J (every J-th chunk of the A codes) in one
       group, over slice range k of S (the pairs whose pattern has its top
       MBITS bits in that range); the units of a group partition its pairs.
       Splitting by slices adds no stage-2 eliminations (each pattern belongs
       to one slice), only a rebuild of the B piece and a rescan of the A part
       per unit, so it is the cheap way to make units short. */
    long long BPIECES = 1, BPIECE = 0, APARTS = 1, APART = 0, SLICES = 1, SLICE = 0;
    { const char *v = getenv("HKT_SLICES"); if (v) SLICES = atoll(v); }
    { const char *v = getenv("HKT_SLICE"); if (v) SLICE = atoll(v); }
    if (SLICES < 1 || SLICE < 0 || SLICE >= SLICES) {
        fprintf(stderr, "bad unit: need 0 <= HKT_SLICE < HKT_SLICES\n");
        return 1;
    }
    { const char *v = getenv("HKT_BPIECES"); if (v) BPIECES = atoll(v); }
    { const char *v = getenv("HKT_BPIECE"); if (v) BPIECE = atoll(v); }
    { const char *v = getenv("HKT_APARTS"); if (v) APARTS = atoll(v); }
    { const char *v = getenv("HKT_APART"); if (v) APART = atoll(v); }
    if (BPIECES < 1 || BPIECE < 0 || BPIECE >= BPIECES || APARTS < 1 || APART < 0 || APART >= APARTS) {
        fprintf(stderr, "bad unit: need 0 <= HKT_BPIECE < HKT_BPIECES and 0 <= HKT_APART < HKT_APARTS\n");
        return 1;
    }
    if (BPIECES > 1 || APARTS > 1 || SLICES > 1)
        printf("   unit: B piece %lld of %lld, A part %lld of %lld, slice range %lld of %lld%s\n", BPIECE, BPIECES,
               APART, APARTS, SLICE, SLICES, donly >= 0 ? "" : " in every group");
    { const char *v = getenv("HKT_NLEV"); if (v) { NLEV = atoi(v); if (NLEV < 2) NLEV = 2; if (NLEV > 256) NLEV = 256; } }
    { const char *v = getenv("HKT_SAMPLE"); if (v) { SAMPLE = atoi(v); if (SAMPLE < 1) SAMPLE = 1; } }
    { const char *v = getenv("HKT_ENUM_ONLY"); if (v && atoi(v)) ENUM_ONLY = 1; }
    if (SAMPLE > 1) printf("   sampling: pair filter and solve on every %d-th chunk of %d A codes\n", SAMPLE, ACHUNK);
    if (NLEV != 256) printf("   B index quantised to %d levels\n", NLEV);
    if (ENUM_ONLY) printf("   enumeration only: canonical A and kept B per group, no index, no solve\n");

    double t0 = wnow();
    tsc_calibrate();
    bsel_init();
    {
        TBITS = H < 18 ? H : 18;
        const char *v = getenv("HKT_TBITS"); if (v) TBITS = atoi(v);
        if (TBITS < 4) TBITS = 4;
        if (TBITS > H) TBITS = H;
        if (H - TBITS > 20) TBITS = H - 20;
        MBITS = H - TBITS;
        KNUL = ent_choose_k();
        v = getenv("HKT_KNUL"); if (v) { KNUL = atoi(v); if (KNUL < 1) KNUL = 1; if (KNUL > S2_MAXFREE) KNUL = S2_MAXFREE; }
        { const char *a = getenv("HKT_PRE"); if (a) PRETEST = atoi(a) ? 1 : 0; }
        GPACK = (H <= 40); WOFF = GPACK ? 1 : 2;
        if (KNUL != 1 || !GPACK) PRETEST = 0;      /* the row-sum parities are stored for nullity 1 only */
        HS = 2; while (HS < WOFF + KNUL + (PRETEST ? 2 : 0)) HS *= 2;
        CPW = (H + 1) / 2; CNHI = H > 32 ? H - 32 : 0; CBAS = CPW + CNHI;
        CS = (CBAS + KNUL + 3) & ~3;
        { const char *a = getenv("HKT_ASAMPLE"); if (a) { ASAMPLE = atoi(a); if (ASAMPLE < 1) ASAMPLE = 1; } }
        { const char *a = getenv("HKT_REPEAT"); if (a) { REPEAT = atoi(a); if (REPEAT < 1) REPEAT = 1; } }
        if (REPEAT > 1) printf("   MEASUREMENT ONLY: every slice is processed %d times, so every count below is %d times the true one\n", REPEAT, REPEAT);
        { const char *a = getenv("HKT_NLEVC"); if (a) NLEVC = atoi(a); }
        { const char *a = getenv("HKT_EXACTPSD"); if (a) EXACTPSD = atoi(a) ? 1 : 0; }
        HF = (H + 7) & ~7;
        if (EXACTPSD) printf("   exact pair ceiling: a pair is kept only when PSD_A + PSD_B <= 4n + 0.01 at every frequency (float rows, %d bytes per B row)\n", HF * 4);
        if (NLEVC > NLEV) NLEVC = NLEV;
        if (NLEVC < 2) NLEVC = 2;
        for (int l = 0; l < 256; l++) CLEVT[l] = (uint8_t)(l < NLEV ? (l * NLEVC) / NLEV : NLEVC - 1);
        ent_selfcheck();
        lev_selfcheck();
        if (getenv("HKT_CHECK_ONLY")) { printf("n=%d: startup checks passed\n", N); return 0; }
        printf("   slice cache: %d slices of 2^%d patterns, %d null vectors per entry%s, %d + %d bytes per pattern, %.1f MB shared by the threads\n",
               1 << MBITS, TBITS, KNUL, PRETEST ? " + row-sum parities" : "", HS * 8, CS * 8, (double)(HS + CS) * 8.0 * (double)((size_t)1 << TBITS) / 1048576.0);
        if (SAMPLE > 1 || ASAMPLE > 1)
            printf("   sampling: every %d-th slice, every %d-th canonical A\n", SAMPLE, ASAMPLE);
    }
    fflush(stdout);

    /* HKT_PLAN=1: size the units of every group under the memory budget
       HKT_MEM_MB and estimate the pair count, from random samples of A and B
       codes (HKT_PLAN_SAMPLES of each, default 1000000, fixed seed), and cuts
       every (A part, B piece) into slice ranges of at most HKT_UNIT_PAIRS
       estimated pairs (default 4e11, a few hours on 16 threads).  Nothing
       is searched; the PLAN lines are what run_census.sh reads.  The memory
       model is the peak of a unit: per kept B row its code, its record, its
       level bytes and its share of the tile index, with the build's temporary
       copies, plus the padding of 2^MBITS class blocks to 256 rows; per kept
       A row its record, levels and PAF, twice while they are gathered; the
       slice cache; and one survivor list per thread. */
    if (getenv("HKT_PLAN")) {
        double mem_mb = 20000;
        long long ns = 1000000;
        { const char *v = getenv("HKT_MEM_MB"); if (v) mem_mb = atof(v); }
        { const char *v = getenv("HKT_PLAN_SAMPLES"); if (v) ns = atoll(v); }
        double unit_pairs = 4e11;
        { const char *v = getenv("HKT_UNIT_PAIRS"); if (v) unit_pairs = atof(v); }
        const double brow = 2.0 * sizeof(hw_t) + sizeof(BRec) + 8 + H + H * NLEVC / 8.0 + 4.0 * omp_get_max_threads();
        const double arow = 2.0 * (5 * sizeof(hw_t) + 2.0 * H) + 5 * sizeof(hw_t) + 2.0 * H + 4;
        const double fixed = (double)(HS + CS) * 8.0 * (double)((size_t)1 << TBITS) + 64e6;
        const double pad = 128.0 * (double)((size_t)1 << MBITS);
        printf("   plan: budget %.0f MB, %lld samples per row kind; %.0f bytes per B row, %.0f per A row, %.0f MB fixed, %.0f padding rows per piece\n",
               mem_mb, ns, brow, arow, fixed / 1048576.0, pad);
        double tot_est_pairs = 0;
        for (int g = 0; g < ng; g++) {
            if (donly >= 0 && g != donly) continue;
            int pa = grp[g].pa, pb = grp[g].pb;
            int equal_ab = !FAMILY && (grp[g].sa == grp[g].sb);
            long long ncombA = FAMILY ? (1LL << H) : BINOM[H][pa], ncombB = BINOM[H][pb];
            uint64_t seed = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)N << 32) ^ (uint64_t)g;
            const int KS = 2000;            /* rows of each kind kept for the pair sample */
            uint8_t *bl = malloc((size_t)KS * H), *al = malloc((size_t)KS * H);
            hw_t *bc = malloc(sizeof(hw_t) * KS), *ac = malloc(sizeof(hw_t) * KS);
            long long keptB = 0, keptA = 0, nbl = 0, nal = 0;
            int8_t paf[64]; float psd[64];
            for (long long i = 0; i < ns; i++) {
                /* a uniform code of popcount pb by Floyd's sampling */
                hw_t c = 0;
                for (int j = H - pb; j < H; j++) {
                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                    int r = (int)(seed % (uint64_t)(j + 1));
                    c |= ((c >> r) & 1u) ? (HW_ONE << j) : (HW_ONE << r);
                }
                paf_mask(sym_mask(c), paf);
                if (psd_all(paf, psd) > fourn + 1e-6) continue;
                keptB++;
                if (nbl < KS) { for (int k = 0; k < H; k++) bl[nbl * H + k] = (uint8_t)lev_of(psd[k], fourn); bc[nbl++] = c; }
            }
            for (long long i = 0; i < ns; i++) {
                hw_t c = 0;
                if (FAMILY) { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; c = (hw_t)seed & hkt_hmask(); }
                else for (int j = H - pa; j < H; j++) {
                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                    int r = (int)(seed % (uint64_t)(j + 1));
                    c |= ((c >> r) & 1u) ? (HW_ONE << j) : (HW_ONE << r);
                }
                if (!a_is_canonical(c)) continue;
                paf_mask(a_mask(c), paf);
                if (psd_all(paf, psd) > fourn + 1e-6) continue;
                keptA++;
                if (nal < KS) { for (int k = 0; k < H; k++) al[nal * H + k] = (uint8_t)lev_of((double)fourn - psd[k], fourn); ac[nal++] = c; }
            }
            double estB = (double)keptB / ns * ncombB, estA = (double)keptA / ns * ncombA;
            long long pp = 0, pt = 0;
            for (long long i = 0; i < nal; i++)
                for (long long j = 0; j < nbl; j++) {
                    pt++;
                    int ok = 1;
                    for (int k = 0; k < H && ok; k++) if (bl[j * H + k] > al[i * H + k]) ok = 0;
                    if (ok && equal_ab && bc[j] < ac[i]) ok = 0;
                    pp += ok;
                }
            double fpair = pt ? (double)pp / pt : 0.0;
            double estP = fpair * estA * estB;
            tot_est_pairs += estP;
            /* A takes at most a third of the budget, B the rest */
            double avail = mem_mb * 1048576.0 - fixed;
            long long J = 1, P = 1;
            if (avail <= 0) { fprintf(stderr, "HKT_MEM_MB below the fixed cost\n"); return 1; }
            while (estA * arow / J > avail / 3) J++;
            double bavail = avail - estA * arow / J;
            while ((estB / P + pad) * brow > bavail || estB / P + pad > 2.0e9) {
                P++;
                if (pad * brow > bavail) { fprintf(stderr, "HKT_MEM_MB too small for the class-block padding\n"); return 1; }
            }
            double peak = (fixed + estA * arow / J + (estB / P + pad) * brow) / 1048576.0;
            /* slice ranges per (A part, B piece): at most unit_pairs pairs each */
            long long S = (long long)ceil(estP / (double)(J * P) / unit_pairs);
            if (S < 1) S = 1;
            if (S > ((long long)1 << MBITS)) S = (long long)1 << MBITS;
            printf("PLAN group=%d s=(%d,%d) estA=%.4g estB=%.4g pairfrac=%.4g (%lld of %lld sampled) estpairs=%.4g aparts=%lld bpieces=%lld slices=%lld units=%lld peak_mb=%.0f\n",
                   g, grp[g].sa, grp[g].sb, estA, estB, fpair, pp, pt, estP, J, P, S, J * P * S, peak);
            free(bl); free(al); free(bc); free(ac);
        }
        printf("PLAN total estpairs=%.4g\n", tot_est_pairs);
        return 0;
    }

    const hw_t maskH = hkt_hmask();
    long long tot_pairs = 0, tot_pass = 0, tot_raw = 0, tot_filt = 0, tot_found = 0, tot_capped = 0, tot_A = 0;
    double max_index_mb = 0, max_table_mb = 0;

    for (int g = 0; g < ng; g++) {
        if (donly >= 0 && g != donly) continue;
        double tg = wnow();
        int pa = grp[g].pa, pb = grp[g].pb;
        int equal_ab = !FAMILY && (grp[g].sa == grp[g].sb);
        memset(ALLOWD, 0, sizeof ALLOWD);
        for (int m = 0; m < grp[g].nmem; m++) {
            const Decomp *D = &dec[grp[g].mem[m]];
            ALLOWD[D->p[o0]] |= HW_ONE << D->p[o1];
            ALLOWD[D->p[o1]] |= HW_ONE << D->p[o0];
        }
        Table T;
        double ttb = wnow();
        long long ncombB = BINOM[H][pb];
        long long brlo = (long long)((__int128)ncombB * BPIECE / BPIECES);
        long long brhi = (long long)((__int128)ncombB * (BPIECE + 1) / BPIECES);
        table_build(&T, pb, brlo, brhi, fourn, ENUM_ONLY);
        double idx_mb = index_bytes(T.np, NLEVC) / 1048576.0;
        if (idx_mb > max_index_mb) max_index_mb = idx_mb;
        if (T.table_mb > max_table_mb) max_table_mb = T.table_mb;
        long long ncombA = FAMILY ? (1LL << H) : BINOM[H][pa];
        /* A part j takes the chunks of ACHUNK codes whose index is j mod J:
           canonical codes crowd the low ranks (a least code in its orbit
           rarely has its top bit set), so contiguous ranges would leave one
           part with nearly every row, while interleaved chunks split them
           evenly and still scan each code once */
        const long long nchA = (ncombA + ACHUNK - 1) / ACHUNK;
        printf("   group %d (%d, %d): B table %d of %lld codes (%.1f%%), %.0f MB of rows, index %.0f MB (%d coarse levels and the row levels), built in %.1fs; A over %lld codes%s\n",
               g, grp[g].sa, grp[g].sb, T.n, T.ncomb, 100.0 * T.n / (double)T.ncomb, T.table_mb, idx_mb, NLEVC,
               wnow() - ttb, ncombA, equal_ab ? ", B >= A" : "");
        if (ENUM_ONLY) {
            printf("   group %d: index MB at 256/128/64/32/16/8 coarse levels: %.0f / %.0f / %.0f / %.0f / %.0f / %.0f; %zu bytes per row; survivor list %.0f MB per thread (virtual)\n",
                   g, index_bytes(T.n, 256) / 1048576.0, index_bytes(T.n, 128) / 1048576.0,
                   index_bytes(T.n, 64) / 1048576.0, index_bytes(T.n, 32) / 1048576.0,
                   index_bytes(T.n, 16) / 1048576.0, index_bytes(T.n, 8) / 1048576.0,
                   sizeof(hw_t) + sizeof(BRec), (double)T.n * sizeof(int) / 1048576.0);
        }
        fflush(stdout);
        if (!T.n && !ENUM_ONLY) { table_free(&T); continue; }

        /* ---- the canonical A rows of the group, kept with their solver words */
        long long nA = 0, pairs = 0, pass = 0, raw = 0, filt = 0, found = 0, capped = 0;
        long long nbuilt = 0, nover = 0, npre = 0, nvisit = 0, nbusy = 0, ncoarse = 0, nlevel = 0;
        double tsolve = 0, tscan = 0, tfilt = 0, tbuild = 0;
        typedef struct { hw_t code, rat, da, xa, ta0; } ARec;
        ARec *AR = NULL; uint8_t *ALEV = NULL; int8_t *APAF = NULL; int *ACLS = NULL; float *APSD = NULL;
        {
            int nth = omp_get_max_threads();
            ARec **la = calloc(nth, sizeof(ARec *)); uint8_t **ll = calloc(nth, sizeof(uint8_t *));
            int8_t **lp = calloc(nth, sizeof(int8_t *));
            float **lf = calloc(nth, sizeof(float *));
            long long *ln = calloc(nth, sizeof(long long)), *lcap = calloc(nth, sizeof(long long));
#pragma omp parallel
            {
                int th = omp_get_thread_num();
                int8_t pafa[64]; float psda[64]; int ta[64];
                long long lA = 0; double lscan = 0;
#pragma omp for schedule(dynamic, 1)
                for (long long ck = APART; ck < nchA; ck += APARTS) {
                    long long c0 = ck * ACHUNK, chi = c0 + ACHUNK; if (chi > ncombA) chi = ncombA;
                    hw_t code = FAMILY ? (hw_t)c0 : unrank_colex(c0, pa);
                    double tc0 = tcpun();
                    for (long long ci = c0; ci < chi; ci++) {
                        hw_t am = code;
                        if (ci + 1 < chi) code = FAMILY ? code + 1 : gosper_next(code);
                        if (!a_is_canonical(am)) continue;
                        paf_mask(a_mask(am), pafa);
                        if (psd_all(pafa, psda) > fourn + 1e-6) continue;
                        lA++;
                        if (ENUM_ONLY) continue;
                        int dead = 0; uint8_t lev[64];
                        for (int k = 0; k < H; k++) {
                            double cap = (double)fourn - psda[k];
                            if (cap < 0) { dead = 1; break; }
                            lev[k] = (uint8_t)lev_of(cap, fourn);
                        }
                        if (dead) continue;
                        if (ln[th] == lcap[th]) {
                            lcap[th] = lcap[th] ? 2 * lcap[th] : 1024;
                            la[th] = realloc(la[th], sizeof(ARec) * (size_t)lcap[th]);
                            ll[th] = realloc(ll[th], (size_t)lcap[th] * H);
                            lp[th] = realloc(lp[th], (size_t)lcap[th] * H);
                            if (EXACTPSD) lf[th] = realloc(lf[th], (size_t)lcap[th] * HF * sizeof(float));
                        }
                        hw_t raraw; nw_arow(pafa, ta, &raraw);
                        hw_t wa1 = 0, ta0 = 0;
                        for (int k = 1; k <= H; k++) {
                            if (ta[k] & 1) ta0 |= HW_ONE << (k - 1);
                            if (ta[k] & 2) wa1 |= HW_ONE << (k - 1);
                        }
                        ARec *r = &la[th][ln[th]];
                        r->code = am; r->rat = nw_tmap(raraw); r->da = raraw ^ M0W; r->xa = wa1; r->ta0 = ta0;
                        memcpy(ll[th] + (size_t)ln[th] * H, lev, (size_t)H);
                        memcpy(lp[th] + (size_t)ln[th] * H, pafa, (size_t)H);
                        if (EXACTPSD) {
                            float *pf = lf[th] + (size_t)ln[th] * HF;
                            for (int k = 0; k < HF; k++) pf[k] = k < H ? psda[k] : 0.0f;
                        }
                        ln[th]++;
                    }
                    lscan += tcpun() - tc0;
                }
#pragma omp atomic
                nA += lA;
#pragma omp atomic
                tscan += lscan;
            }
            long long nlive = 0;
            for (int t = 0; t < nth; t++) nlive += ln[t];
            int ncl = 1 << MBITS;
            long long *cs = calloc((size_t)ncl + 1, sizeof(long long));
            for (int t = 0; t < nth; t++)
                for (long long i = 0; i < ln[t]; i++) cs[(MBITS ? (int)(la[t][i].rat >> TBITS) : 0) + 1]++;
            for (int c = 0; c < ncl; c++) cs[c + 1] += cs[c];
            AR = malloc(sizeof(ARec) * (size_t)(nlive + 1));
            ALEV = malloc((size_t)(nlive + 1) * H); APAF = malloc((size_t)(nlive + 1) * H);
            ACLS = malloc(sizeof(int) * (size_t)(nlive + 1));
            if (EXACTPSD) APSD = malloc((size_t)(nlive + 1) * HF * sizeof(float));
            for (int t = 0; t < nth; t++) {
                for (long long i = 0; i < ln[t]; i++) {
                    int c = MBITS ? (int)(la[t][i].rat >> TBITS) : 0;
                    long long pos = cs[c]++;
                    AR[pos] = la[t][i]; ACLS[pos] = c;
                    memcpy(ALEV + (size_t)pos * H, ll[t] + (size_t)i * H, (size_t)H);
                    memcpy(APAF + (size_t)pos * H, lp[t] + (size_t)i * H, (size_t)H);
                    if (EXACTPSD) memcpy(APSD + (size_t)pos * HF, lf[t] + (size_t)i * HF, (size_t)HF * sizeof(float));
                }
                free(la[t]); free(ll[t]); free(lp[t]); free(lf[t]);
            }
            free(la); free(ll); free(lp); free(lf); free(ln); free(lcap); free(cs);
            long long nAlive = nlive;
            int maxblk = 0;
            if (!ENUM_ONLY) for (int c = 0; c < ncl; c++) { int b = (T.bc[c] + 255) & ~255; if (b > maxblk) maxblk = b; }

            /* the pre-test table: for a pattern with popcount s, which
               (parity of popcount(x & t), parity of popcount(x & ~t)) can
               belong to an allowed pair of row sums; bit (p1 + 2 p0) */
            uint8_t PREOK[65];
            for (int s = 0; s <= H; s++) {
                uint8_t ok = 0;
                for (int pc = 0; pc <= H; pc++)
                    for (int pd = 0; pd <= H; pd++) {
                        if (!((ALLOWD[pc] >> pd) & 1u)) continue;
                        int num = H + s - pc - pd;
                        if (num < 0 || (num & 1)) continue;
                        int n1 = num / 2, n0 = H - pc - n1;
                        if (n1 > s || n0 < 0 || n0 > H - s) continue;
                        ok |= (uint8_t)(1u << ((n1 & 1) + 2 * (n0 & 1)));
                    }
                PREOK[s] = ok;
            }

            size_t nent = (size_t)1 << TBITS;
            hw_t *hot = NULL, *cold = NULL;
            if (!ENUM_ONLY) {
                hot = aligned_alloc(1u << 21, ((nent * HS * 8) + (1u << 21) - 1) & ~(size_t)((1u << 21) - 1));
                cold = aligned_alloc(1u << 21, ((nent * CS * 8 + 256) + (1u << 21) - 1) & ~(size_t)((1u << 21) - 1));
                if (!hot || !cold) { fprintf(stderr, "out of memory for the slice cache\n"); exit(1); }
                madvise(hot, nent * HS * 8, MADV_HUGEPAGE);
                madvise(cold, nent * CS * 8, MADV_HUGEPAGE);
            }
            const long long zchunk = 1 << 12;      /* table entries zeroed per work item */
            int achunk = (int)(nAlive >> (MBITS + 6)); if (achunk < 4) achunk = 4; if (achunk > 256) achunk = 256;
            { const char *v = getenv("HKT_ACHUNK"); if (v && atoi(v) > 0) achunk = atoi(v); }
#pragma omp parallel if(!ENUM_ONLY)
            {
                long long lpairs = 0, lpass = 0, lraw = 0, lfilt = 0, lfound = 0, lcapped = 0;
                long long lbuilt = 0, lover = 0, lpre = 0, lvisit = 0, lbusy = 0, lcoarse = 0, llevel = 0;
                double dfilt = 0, dsolve = 0, dbuild = 0;
                int *surv = ENUM_ONLY ? NULL : malloc(sizeof(int) * (size_t)(maxblk + 8));
                BigEnt *EB = malloc(sizeof(BigEnt));
                const hw_t tmask = (TBITS >= 64) ? ~(hw_t)0 : ((HW_ONE << TBITS) - 1);
#define RQ 64
#ifdef PROF
#define PROF_T(x) uint64_t x = __rdtsc()
#else
#define PROF_T(x) uint64_t x = 0
#endif
                uint64_t prof1 = 0, prof2 = 0;
#define RB 32
                struct { hw_t t, X, d0; int b, a; } q1[RQ];
                struct { hw_t t, bb; int b, a; } q2[2 * RQ];
                unsigned h1 = 0, n1q = 0, h2 = 0, n2q = 0;
                hw_t sink[8] = {0};

/* exact check of one (C, D) for the pair (AIDX, BIDX) */
#define WIL_EXACT(GOOD, CODEC, CODED, BIDX, AIDX)                              \
                do {                                                           \
                    nw_t wb = hkt_symword(T.code[BIDX]);                       \
                    const int8_t *pafa = APAF + (size_t)(AIDX) * H;            \
                    nw_t wc = hkt_symword(CODEC), wd = hkt_symword(CODED);     \
                    GOOD = 1;                                                  \
                    for (int k = 1; k <= H && GOOD; k++) {                     \
                        int t2 = (int)pafa[k - 1] + 3 * N                      \
                               - 2 * (nw_popcount(wb ^ hkt_rotk(wb, k))        \
                                    + nw_popcount(wc ^ hkt_rotk(wc, k))        \
                                    + nw_popcount(wd ^ hkt_rotk(wd, k)));      \
                        if (t2 != 0) GOOD = 0;                                 \
                    }                                                          \
                } while (0)
#define WIL_EMIT(NCD, CANDS, SBITS, BIDX, AIDX)                                \
                do {                                                           \
                    for (int cc = 0; cc < (NCD); cc++) {                       \
                        hw_t vv = (CANDS)[cc];                                 \
                        hw_t codeC = (~vv) & maskH;                            \
                        hw_t codeD = ((SBITS) ^ vv) & maskH;                   \
                        int good;                                              \
                        WIL_EXACT(good, codeC, codeD, BIDX, AIDX);             \
                        lfilt++;                                               \
                        if (!good) continue;                                   \
                        lfound++;                                              \
                        hw_t q[4] = { AR[AIDX].code, T.code[BIDX], codeC, codeD }; \
                        _Pragma("omp critical")                                \
                        sol_push(q);                                           \
                    }                                                          \
                } while (0)

/* second stage: a consistent pair reads the solution map.  With nullity 1
   and the null vector ~t the two candidates are (C, D) and (D, C): one pair
   of popcounts decides both and one exact check serves both. */
#define WIL_STAGE2()                                                           \
                do {                                                           \
                    hw_t t_ = q2[h2].t, bb_ = q2[h2].bb; int b_ = q2[h2].b, a_ = q2[h2].a; \
                    h2 = (h2 + 1) & (2 * RQ - 1); n2q--;                       \
                    const hw_t *ce = cold + (size_t)(t_ & tmask) * CS;         \
                    hw_t m_ = hot[(size_t)(t_ & tmask) * HS];                  \
                    int nul_ = (int)(m_ & 0xffu) - 1;                          \
                    hw_t part_ = ent_part_pk(bb_, ce);                            \
                    if ((m_ >> 21) & 1u) {                                     \
                        lraw += 2;                                             \
                        hw_t codeC = (~part_) & maskH, codeD = (t_ ^ part_) & maskH; \
                        int pc_ = hw_popcount(codeC), pd_ = hw_popcount(codeD); \
                        if (!((ALLOWD[pc_] >> pd_) & 1u)) break;               \
                        int wt_ = (pc_ != pd_ || codeC == codeD) ? 2 : 1;      \
                        int good;                                              \
                        WIL_EXACT(good, codeC, codeD, b_, a_);                 \
                        lfilt += wt_;                                          \
                        if (!good) break;                                      \
                        lfound += wt_;                                         \
                        hw_t q[4] = { AR[a_].code, T.code[b_], codeC, codeD }; \
                        _Pragma("omp critical")                                \
                        { sol_push(q); if (wt_ == 2) { hw_t q2_[4] = { q[0], q[1], codeD, codeC }; sol_push(q2_); } } \
                        break;                                                 \
                    }                                                          \
                    hw_t cand[HKT_MAXC];                                       \
                    int nc = ent_cands(part_, ce + CBAS, nul_, cand);           \
                    lraw += nc;                                                \
                    int nk = wil_rowfilter(cand, nc, t_, maskH);               \
                    WIL_EMIT(nk, cand, t_, b_, a_);                            \
                } while (0)

/* first stage: the hot entry, built on first use, and the consistency test.
   The consistent pairs are appended without a branch on the test. */
#define WIL_STAGE1()                                                           \
                do {                                                           \
                    hw_t t_ = q1[h1].t, X_ = q1[h1].X, d0_ = q1[h1].d0; int b_ = q1[h1].b, a_ = q1[h1].a; \
                    h1 = (h1 + 1) & (RQ - 1); n1q--;                           \
                    hw_t *e = hot + (size_t)(t_ & tmask) * HS;                 \
                    hw_t meta = __atomic_load_n(&e[0], __ATOMIC_ACQUIRE);      \
                    hw_t zero_ = 0;                                            \
                    /* an empty entry is claimed by one thread, which alone    \
                       writes it and publishes meta last; a thread that meets  \
                       a claimed entry solves its pair without the table */    \
                    if (__builtin_expect(!meta, 0) &&                          \
                        !__atomic_compare_exchange_n(&e[0], &zero_, (hw_t)ENT_BUSY, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) \
                        meta = zero_;                                          \
                    if (__builtin_expect(!meta, 0)) {                          \
                        uint64_t c0_ = __rdtsc();                              \
                        ent_build(t_, d0_, EB); lbuilt++;                      \
                        if (EB->nul <= KNUL) {                                 \
                            hw_t *ce = cold + (size_t)(t_ & tmask) * CS;       \
                            if (!GPACK) e[1] = EB->G;                          \
                            for (int i = 0; i < KNUL; i++) e[WOFF + i] = i < EB->nul ? EB->w[i] : (hw_t)0; \
                            ent_pack(EB, ce);                                  \
                            meta = (hw_t)(EB->nul + 1);                        \
                            if (EB->nul == 1 && EB->basis[0] == ((~t_) & maskH)) meta |= (hw_t)1 << 21; \
                            if (PRETEST) {                                     \
                                /* parity(part & t) = parity(bb & y), parity(part & ~t) = parity(bb & z) */ \
                                hw_t y = 0, z = 0;                             \
                                if (EB->nul == 1) {                            \
                                    for (int j = 0; j < H; j++) {              \
                                        y |= (hw_t)hw_parity(EB->pcol[j] & t_) << j; \
                                        z |= (hw_t)hw_parity(EB->pcol[j] & ~t_ & maskH) << j; \
                                    }                                          \
                                    unsigned ok = PREOK[hw_popcount(t_ & maskH)]; \
                                    unsigned f_ = (unsigned)hw_parity(EB->basis[0] & t_) \
                                                + 2u * (unsigned)hw_parity(EB->basis[0] & ~t_ & maskH); \
                                    unsigned okb = 0;                          \
                                    for (unsigned p = 0; p < 4; p++)           \
                                        if (((ok >> p) & 1u) || ((ok >> (p ^ f_)) & 1u)) okb |= 1u << p; \
                                    meta |= (hw_t)okb << 16;                   \
                                } else meta |= (hw_t)0xf << 16;                \
                                e[2] = y; e[3] = z;                            \
                            }                                                  \
                            if (GPACK) meta |= EB->G << 24;                    \
                        } else { meta = ENT_OVER; lover++; }                   \
                        __atomic_store_n(&e[0], meta, __ATOMIC_RELEASE);       \
                        cbuild += __rdtsc() - c0_;                             \
                    }                                                          \
                    if (__builtin_expect((meta & 0xff00u) != 0, 0)) {          \
                        if ((meta & 0xffffu) == ENT_BUSY) lbusy++;             \
                        ent_build(t_, d0_, EB);                                \
                        hw_t bb_ = (EB->G ^ X_) & maskH; int bad_ = 0;         \
                        for (int i = 0; i < EB->nul; i++) bad_ |= hw_parity(bb_ & EB->w[i]); \
                        if (bad_) break;                                       \
                        lpass++;                                               \
                        if (EB->nul > S2_MAXFREE) { lcapped++; break; }        \
                        hw_t cand[HKT_MAXC];                                   \
                        int nc = ent_cands(ent_part(bb_, EB->pcol), EB->basis, EB->nul, cand); \
                        lraw += nc;                                            \
                        int nk = wil_rowfilter(cand, nc, t_, maskH);           \
                        WIL_EMIT(nk, cand, t_, b_, a_);                        \
                        break;                                                 \
                    }                                                          \
                    hw_t bb_ = ((GPACK ? (meta >> 24) : e[1]) ^ X_) & maskH;   \
                    unsigned ok_ = 1u ^ (unsigned)hw_parity(bb_ & e[WOFF]);    \
                    for (int i = 1; i < KNUL; i++) ok_ &= 1u ^ (unsigned)hw_parity(bb_ & e[WOFF + i]); \
                    lpass += ok_;                                              \
                    if (PRETEST) {                                             \
                        unsigned p_ = (unsigned)hw_parity(bb_ & e[2]) + 2u * (unsigned)hw_parity(bb_ & e[3]); \
                        unsigned k_ = (unsigned)(meta >> (16 + p_)) & 1u;      \
                        lpre += ok_ & (k_ ^ 1u); lraw += 2 * (ok_ & (k_ ^ 1u)); \
                        ok_ &= k_;                                             \
                    }                                                          \
                    {   const hw_t *ce = ok_ ? cold + (size_t)(t_ & tmask) * CS : sink; \
                        __builtin_prefetch(ce); __builtin_prefetch(ce + 8);    \
                        if (CS > 16) __builtin_prefetch(ce + 16);              \
                        if (CS > 24) __builtin_prefetch(ce + CS - 1); }        \
                    unsigned s2_ = (h2 + n2q) & (2 * RQ - 1);                  \
                    q2[s2_].t = t_; q2[s2_].bb = bb_; q2[s2_].b = b_; q2[s2_].a = a_; n2q += ok_; \
                } while (0)
#define WIL_BATCH1()                                                           \
                do {                                                           \
                    PROF_T(pa_);                                               \
                    for (int i_ = 0; i_ < RB; i_++) WIL_STAGE1();              \
                    PROF_T(pb_); prof1 += pb_ - pa_;                           \
                    if (n2q > RQ) { for (int i_ = 0; i_ < RB; i_++) WIL_STAGE2(); \
                        PROF_T(pc_); prof2 += pc_ - pb_; }                      \
                } while (0)

                /* every thread walks the slices in step: the table is zeroed
                   in shares, the A rows are handed out in chunks, and each
                   thread drains its own queues before the barrier */
                const long long nsl = (long long)1 << MBITS;
                const long long sl0 = (long long)((__int128)nsl * SLICE / SLICES);
                const long long sl1 = (long long)((__int128)nsl * (SLICE + 1) / SLICES);
                for (long long phi = sl0; phi < sl1; phi++) {
                    if (STOPFILE && (phi & 63) == 0) {
#pragma omp single
                        STOPPED = access(STOPFILE, F_OK) == 0;
                        if (STOPPED) break;
                    }
                    if (ENUM_ONLY || (phi % SAMPLE) != 0) continue;
#pragma omp for schedule(static)
                    for (long long zc = 0; zc < (long long)nent; zc += zchunk) {
                        long long ze = zc + zchunk; if (ze > (long long)nent) ze = (long long)nent;
                        memset(hot + (size_t)zc * HS, 0, (size_t)(ze - zc) * HS * 8);
                    }
                    /* TSC intervals split the slice between filter, solve and
                       build; the thread's CPU time over the slice scales them,
                       so time spent descheduled is not counted */
                    double cpu0 = tcpun();
                    uint64_t tsc0 = __rdtsc(), cfilt = 0, csolve = 0, cbuild = 0;
                    uint64_t cs0 = tsc0;
                    for (int rep = 0; rep < REPEAT; rep++)
#pragma omp for schedule(dynamic, achunk) nowait
                    for (long long ai = 0; ai < nAlive; ai++) {
                        if (ASAMPLE > 1 && (ai % ASAMPLE) != 0) continue;
                        int beta = ACLS[ai] ^ (int)phi;
                        int cnt = T.bc[beta];
                        if (!cnt) continue;
                        lvisit++;
                        uint64_t cf0 = __rdtsc();
                        csolve += cf0 - cs0;
                        const ARec *ar = &AR[ai];
                        const uint8_t *lev = ALEV + (size_t)ai * H;
                        const int bs = T.bs[beta];
                        int lo = bs;
                        if (equal_ab) lo = bs + lower_bound_hw(T.code + bs, cnt, ar->code);
                        int nsurv = 0;
                        if (lo < bs + cnt) {
                            int off[64], nbp = 0;
                            for (int k = 0; k < H; k++) {
                                int cl = clev(lev[k]);
                                if (cl >= NLEVC - 1) continue;
                                off[nbp++] = (k * NLEVC + cl) * 4;
                            }
                            const size_t tstride = (size_t)H * NLEVC * 4 + 4;
                            int w0 = lo >> 6, wend = (bs + ((cnt + 255) & ~255)) >> 6;
                            uint64_t lomask = (lo & 63) ? (~0ULL << (lo & 63)) : ~0ULL;
                            for (int w = w0 & ~3; w < wend; w += 4) {
                                const uint64_t *tb = T.bmp + (size_t)(w >> 2) * tstride;
                                __m256i fv = _mm256_loadu_si256((const __m256i *)(tb + tstride - 4));   /* real rows */
                                for (int i = 0; i < nbp; i++)
                                    fv = _mm256_and_si256(fv, _mm256_loadu_si256((const __m256i *)(tb + off[i])));
                                if (_mm256_testz_si256(fv, fv)) continue;
                                uint64_t v4[4];
                                _mm256_storeu_si256((__m256i *)v4, fv);
                                for (int q = 0; q < 4; q++) {
                                    int ww = w + q;
                                    if (ww < w0) continue;
                                    uint64_t v = v4[q];
                                    if (ww == w0) v &= lomask;
                                    /* four indices are written whatever the word
                                       holds (tzcnt of zero is 64, and the list has
                                       slack), so a word with at most four survivors
                                       costs no data-dependent branch */
                                    const int base = ww << 6, cntv = (int)_mm_popcnt_u64(v);
                                    surv[nsurv]     = base + (int)_tzcnt_u64(v); v = _blsr_u64(v);
                                    surv[nsurv + 1] = base + (int)_tzcnt_u64(v); v = _blsr_u64(v);
                                    surv[nsurv + 2] = base + (int)_tzcnt_u64(v); v = _blsr_u64(v);
                                    surv[nsurv + 3] = base + (int)_tzcnt_u64(v); v = _blsr_u64(v);
                                    if (__builtin_expect(cntv > 4, 0)) {
                                        nsurv += 4;
                                        while (v) { surv[nsurv++] = base + (int)_tzcnt_u64(v); v = _blsr_u64(v); }
                                    } else nsurv += cntv;
                                }
                            }
                            /* exact levels: keep a row when its level is at most
                               the cap at every frequency.  Bytes past H in the
                               first load, and the bytes the second load repeats,
                               are compared against 255. */
                            lcoarse += nsurv;
                            __m256i c0v, c1v;
                            lev_caps(lev, &c0v, &c1v);
                            int ns2 = 0;
                            for (int u = 0; u < nsurv; u++) {
                                int b = surv[u];
                                surv[ns2] = b; ns2 += lev_ok(T.lv8 + (size_t)b * H, c0v, c1v);
                            }
                            nsurv = ns2;
                            if (EXACTPSD) {
                                /* the ceiling itself on the stored PSD values; the
                                   0.01 margin is far above float rounding at these
                                   magnitudes, so no pair under the true ceiling goes */
                                const float *pa_ = APSD + (size_t)ai * HF;
                                const __m256 lim = _mm256_set1_ps((float)fourn + 0.01f);
                                llevel += nsurv;
                                ns2 = 0;
                                for (int u = 0; u < nsurv; u++) {
                                    int b = surv[u];
                                    const float *pb_ = T.psdf + (size_t)b * HF;
                                    __m256 bad = _mm256_setzero_ps();
                                    for (int k = 0; k < HF; k += 8)
                                        bad = _mm256_or_ps(bad, _mm256_cmp_ps(_mm256_add_ps(_mm256_loadu_ps(pa_ + k), _mm256_loadu_ps(pb_ + k)), lim, _CMP_GT_OQ));
                                    surv[ns2] = b; ns2 += (_mm256_movemask_ps(bad) == 0);
                                }
                                nsurv = ns2;
                            }
                        }
                        lpairs += nsurv;
                        cs0 = __rdtsc();
                        cfilt += cs0 - cf0;
                        const hw_t rat = ar->rat, da = ar->da, xa = ar->xa, ta0 = ar->ta0;
                        for (int u = 0; u < nsurv; u++) {
                            int b0i = surv[u];
                            const BRec *r = &T.rec[b0i];
                            hw_t t = rat ^ r->rbt;
                            hw_t d0 = da ^ r->rbraw;
                            hw_t X = xa ^ r->wb1 ^ (ta0 & ~d0);
                            __builtin_prefetch(hot + (size_t)(t & tmask) * HS);
                            if (n1q == RQ) WIL_BATCH1();
                            unsigned s1_ = (h1 + n1q) & (RQ - 1);
                            q1[s1_].t = t; q1[s1_].X = X; q1[s1_].d0 = d0; q1[s1_].b = b0i; q1[s1_].a = (int)ai; n1q++;
                        }
                    }
                    while (n1q) WIL_STAGE1();
                    while (n2q) WIL_STAGE2();
                    uint64_t tsc1 = __rdtsc();
                    csolve += tsc1 - cs0;
                    double sc = (tcpun() - cpu0) / (double)(tsc1 - tsc0 ? tsc1 - tsc0 : 1);
                    dfilt += cfilt * sc; dsolve += (csolve - cbuild) * sc; dbuild += cbuild * sc;
#pragma omp barrier
                }
#undef WIL_EMIT
#undef WIL_STAGE1
#undef WIL_STAGE2
#undef WIL_BATCH1
#undef WIL_EXACT
                free(surv); free(EB);
#pragma omp critical
                {
                    pairs += lpairs; pass += lpass; raw += lraw; filt += lfilt; found += lfound; capped += lcapped;
                    nbuilt += lbuilt; nover += lover; npre += lpre; nvisit += lvisit; nbusy += lbusy; ncoarse += lcoarse; nlevel += llevel;
                    tfilt += dfilt; tsolve += dsolve; tbuild += dbuild;
#ifdef PROF
                    printf("   prof: stage-1 batches %.1f s, stage-2 batches %.1f s (TSC, includes builds and descheduled time)\n",
                           prof1 * TSC_NS * 1e-9, prof2 * TSC_NS * 1e-9);
#endif
                }
            }
            free(hot); free(cold);
            free(AR); free(ALEV); free(APAF); free(ACLS); free(APSD);
        }
        table_free(&T);
        if (STOPPED) { printf("STOPPED: %s exists; the unit wrote nothing and runs again from the start\n", STOPFILE); return 3; }
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
        printf("   group %d: thread-seconds scan %.1f, filter %.1f, solve %.1f, build %.1f; %.1f ns per pair (filter %.1f, solve %.1f, build %.1f)\n",
               g, tscan, tfilt, tsolve, tbuild, pairs ? 1e9 * (tsolve + tfilt + tbuild) / (double)pairs : 0.0,
               pairs ? 1e9 * tfilt / (double)pairs : 0.0, pairs ? 1e9 * tsolve / (double)pairs : 0.0,
               pairs ? 1e9 * tbuild / (double)pairs : 0.0);
        printf("   group %d: %lld patterns eliminated (%.0f ns each, %.1f pairs per pattern), %lld over the stored nullity, %lld pairs solved off-table during another thread's build, %lld A-block visits (%.1f pairs each), %lld consistent pairs dropped by the row-sum parities\n",
               g, nbuilt, nbuilt ? 1e9 * tbuild / (double)nbuilt : 0.0, nbuilt ? (double)pairs / (double)nbuilt : 0.0,
               nover, nbusy, nvisit, nvisit ? (double)pairs / (double)nvisit : 0.0, npre);
        printf("   group %d: coarse filter (%d levels) passed %lld rows, %.2f per exact survivor\n",
               g, NLEVC, ncoarse, pairs ? (double)ncoarse / (double)pairs : 0.0);
        if (EXACTPSD)
            printf("   group %d: %lld pairs pass the %d-level filter, %lld (%.1f%%) pass the exact ceiling\n",
                   g, nlevel, NLEV, pairs, nlevel ? 100.0 * pairs / (double)nlevel : 0.0);
        if (SAMPLE > 1 || ASAMPLE > 1)
            printf("   group %d: sampled 1/%d of the slices and 1/%d of the A rows; projected %.4g pairs\n",
                   g, SAMPLE, ASAMPLE, (double)pairs * SAMPLE * ASAMPLE);
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
            kq[i].o = SOL[i]; kq[i].key = SOL[i]; sort_key(kq[i].key.c);
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
    /* the line a driver reads: a unit is complete only when it is present */
    printf("UNIT family=%s n=%d group=%d bpiece=%lld/%lld apart=%lld/%lld slice=%lld/%lld pairs=%lld distinct=%lld capped=%lld seconds=%.1f\n",
           FAMILY ? "good" : "wil", N, donly, BPIECE, BPIECES, APART, APARTS, SLICE, SLICES, tot_pairs, ndistinct,
           tot_capped, tsearch);
    if (outpath) {
        /* key order, one spelling per quadruple: the outputs of the units of
           a census concatenate and sort to the output of the whole run */
        FILE *f = fopen(outpath, "w");
        if (!f) { perror(outpath); return 1; }
        int8_t x[128];
        for (long long i = 0; i < ndistinct; i++) {
            for (int u = 0; u < 4; u++) {
                if (u) fputc(' ', f);
                if (u == 0 && FAMILY) skew_row(keyed[i].c[u], x); else sym_row(keyed[i].c[u], x);
                for (int j = 0; j < N; j++) fputc(x[j] > 0 ? '+' : '-', f);
            }
            fputc('\n', f);
        }
        if (fclose(f)) { perror(outpath); return 1; }
        printf("   solutions -> %s\n", outpath);
    }
    (void)orig;
    return 0;
}

