/* best_plant.c -- planted-solution test of every class of the best-matrix
 * census (best_hkt2.c with BEST_QB=q, built -DWIDE above n = 63).
 *
 *   build: gcc -O3 -march=native -fopenmp -DWIDE best_plant.c -lm -o best_plant
 *          (best_hkt2.c is included unmodified from the same directory)
 *   run:   ./best_plant <n> <q> <threads> <first class> <last class> <plants per class>
 *
 * For a class b, random skew rows A, B, C are drawn with A under the spectrum
 * ceiling (PSD_A <= 4n - 2), code bit 0 of A clear (R5), t = code(B) xor
 * code(C) with t_1 = 0 (R7) and pattern class cls(t) = b.  The values
 * uD_k = n - uA_k - uB_k - uC_k define a virtual symmetric row D that (A, B,
 * C) completes exactly at every lag; its D record is filled the way main()
 * of best_hkt2.c fills one (u values, tD, class label, gD of the two
 * completions).  The record carries the code of the all-plus row, whose PSD
 * is 0 at every k >= 1, and R4 is switched off (NOMAX_), since a virtual D has
 * no spectrum; everything after that is the production code: the consistency
 * table of class b is built by cc_build, an A table holding A and 63 random
 * decoys by atable_build, and process_d runs the index scan, the fine test,
 * the stage-1 patterns, the consistency test, the batched stage-2 solve, the
 * lag tests and the exact check.  The plant is recovered when process_d emits
 * the R8/R9 representative of (B, C) with this A.  A plant can be lost only by
 * a defect in that path, including a wrong entry in the consistency table of
 * class b or a wrong class label, or by the nullity cap, which is counted.
 *
 * The output also prints the class forms psi_i and rho_i of class_init, to be
 * compared with the forms best_cover.c derives independently, and one line
 * per class.  The draws depend only on n, q, the class and the plant index.
 */
#define main best_hkt2_main
#include "best_hkt2.c"
#undef main

static void skew_pm(cw y, int *x) {
    x[0] = 1;
    for (int u = 1; u <= H; u++) { int v = ((y >> (u - 1)) & 1) ? -1 : 1; x[u] = v; x[N - u] = -v; }
}
static void uref(const int *x, int *u) {
    for (int k = 1; k <= H; k++) {
        int p = 0;
        for (int j = 0; j < N; j++) p += x[j] * x[(j + k) % N];
        if ((N - p) % 4) die("PAF not congruent to n mod 4");
        u[k] = (N - p) / 4;
    }
}
static uint64_t pr_state;
static uint64_t pr(void) { pr_state ^= pr_state << 13; pr_state ^= pr_state >> 7; pr_state ^= pr_state << 17; return pr_state; }

static void make_drec(DRec *r, const int *uD) {
    cw ud0 = 0, ud1 = 0;
    for (int k = 1; k <= H; k++) { ud0 |= (cw)(uD[k] & 1) << (k - 1); ud1 |= (cw)((uD[k] >> 1) & 1) << (k - 1); }
    r->code = 0;
    r->ud0 = ud0;
    r->tD = tmap((cw)(~ud0 & MASKH));
    r->cls = cls_r((cw)(~ud0 & MASKH));
    for (int e = 0; e < 2; e++) {
        cw de = ((int)(r->tD & 1) == e) ? r->tD : (r->tD ^ NBSTAR_);
        r->gD[e] = s2_G(symh_word(de)) ^ G0_ ^ ud1;
    }
    r->uD[0] = 0;
    for (int k = 1; k <= H; k++) r->uD[k] = (uint8_t)uD[k];
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: best_plant <n> <q> <threads> <first class> <last class> <plants per class>\n"); return 2; }
    N = atoi(argv[1]);
    int q = atoi(argv[2]), threads = atoi(argv[3]), b0 = atoi(argv[4]), b1 = atoi(argv[5]), ntr = atoi(argv[6]);
    if (N < 7 || !(N & 1) || N > NMAX) die("unsupported order");
    H = (N - 1) / 2;
    FULL = (N >= (int)(8 * sizeof(uw))) ? ~(uw)0 : (((uw)1 << N) - 1);
    MASKH = (H >= CWB) ? ~(cw)0 : (((cw)1 << H) - 1);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    stage_init(); fast_init(); solve_selfcheck(); trig_init(); dec_init();
    CCK_ = 2; class_init(q); arec_init(); cc_index_init();
    if (b0 < 0 || b1 >= (1 << QB_) || b0 > b1) die("class range out of bounds");
    printf("n %d H %d q %d corank %d patterns per pair %d\n", N, H, QB_, NNULL_, NCOMB_);
    for (int i = 0; i < QB_; i++) printf("form %d psi %016llx rho %016llx\n", i, (unsigned long long)PSI_[i], (unsigned long long)RHO_[i]);
    fflush(stdout);
    int *surv = malloc(sizeof(int) * BLKW * 64);
    Batch *B = aligned_alloc(64, (sizeof(Batch) + 63) & ~(size_t)63);
    memset(B, 0, sizeof(Batch));
    NOMAX_ = 1;
    long long all_planted = 0, all_recovered = 0;
    for (int beta = b0; beta <= b1; beta++) {
        double t0 = wnow();
        cc_build(beta);
        long long planted = 0, recovered = 0, capped = 0, tzero = 0, emitted = 0;
        for (int tr = 0; tr < ntr; tr++) {
            pr_state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)N << 48) ^ ((uint64_t)beta << 20) ^ (uint64_t)tr;
            for (int w = 0; w < 4; w++) pr();
            cw ya, yb, yc, t;
            int xa[NMAX], xb[NMAX], xc[NMAX], ua[HMAX + 1], ub[HMAX + 1], uc[HMAX + 1], uD[HMAX + 1];
            double p[HMAX];
            for (;;) {
                ya = (cw)pr() & MASKH & ~(cw)1;
                yb = (cw)pr() & MASKH;
                int kind = (int)(pr() % 4);   /* C equal to B, B with one bit changed, or random */
                yc = kind == 0 ? yb : kind == 1 ? (yb ^ ((cw)1 << (1 + pr() % (H - 1)))) : (cw)pr() & MASKH;
                t = yb ^ yc;
                if (t & 1) { yc = ~yc & MASKH; t = yb ^ yc; }      /* R7 */
                if (cls_t(t) != beta) continue;
                if (psd_skew(ya, p) > 4.0 * N - 2.0) continue;
                skew_pm(ya, xa); skew_pm(yb, xb); skew_pm(yc, xc);
                uref(xa, ua); uref(xb, ub); uref(xc, uc);
                int ok = 1;
                for (int k = 1; k <= H; k++) { uD[k] = N - ua[k] - ub[k] - uc[k]; if (uD[k] < 0 || uD[k] > 255) ok = 0; }
                if (ok) break;
            }
            if (t == 0) tzero++;
            DRec dr;
            make_drec(&dr, uD);
            if ((sig_of(dr.ud0) ^ sig_of(MASKH)) != ASIG_) die("a planted virtual D fails R6");
            cw list[64]; int nl = 0;
            list[nl++] = ya;
            while (nl < 64) list[nl++] = (cw)pr() & MASKH & ~(cw)1;
            ATable T = {0};
            atable_build(&T, 0, 0, list, nl, 1000);
            Stat st = {0};
            long long before = NSOL_;
            process_d(&dr, beta, &T, &st, surv, B);
            atable_free(&T);
            cw eb = yb, ec = yc;                                   /* R8, then R9 */
            if (eb & 1) { eb = ~eb & MASKH; ec = ~ec & MASKH; }
            if ((eb ^ t) < eb) { cw z = eb; eb = ec; ec = z; }
            int hit = 0;
            for (long long i = before; i < NSOL_; i++) if (SOL_[i].a == ya && SOL_[i].b == eb && SOL_[i].c == ec) hit = 1;
            emitted += NSOL_ - before;
            capped += st.capped;
            planted++;
            recovered += hit;
            if (!hit) printf("MISS class %d plant %d a=%llx b=%llx c=%llx t=%llx capped %lld\n", beta, tr, (unsigned long long)ya, (unsigned long long)yb, (unsigned long long)yc, (unsigned long long)t, st.capped);
        }
        all_planted += planted; all_recovered += recovered;
        printf("class %d planted %lld recovered %lld emitted %lld capped %lld t0 %lld seconds %.1f\n", beta, planted, recovered, emitted, capped, tzero, wnow() - t0);
        fflush(stdout);
    }
    printf("total planted %lld recovered %lld\n", all_planted, all_recovered);
    return all_planted == all_recovered ? 0 : 1;
}
