/* dedup.c: keep one row per PAF class. Reads rows (+/- strings of length n) on stdin, drops rows that are not hits of block blk
 * of the plan (row sum, windows), and prints the first row of every class, a class being a PAF vector modulo the plan's
 * multiplier group (the least of the nh digit keys, the same class tj7 reduces to). Windows of at most 4 values per lag.
 * usage: dedup plan blk max_classes [rejects] < rows > unique_rows      (hash table of 2^k >= 1.6 max_classes slots, 17 bytes each;
 * once max_classes classes are kept, later hits of new classes are dropped and counted in a warning)
 * rejects: if given, every row that is not a hit of the plan is written there unchanged (used to split a pool into core and shell).
 * stderr: rows read, non-hits, distinct classes, and the number of classes seen exactly j times (j = 1, 2, ..., 15+). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef unsigned __int128 u128;
int main(int argc, char **argv) {
    if (argc < 4) return 2;
    FILE *f = fopen(argv[1], "r"); int blk = atoi(argv[2]); long maxc = atol(argv[3]); int n, m, row[3], lo[3][128], hi[3][128], nh, H[128]; char a[256];
    if (!f || fscanf(f, "%d %255s", &n, a) != 2 || n > 127) return 2; m = (n - 1) / 2; if (2 * m > 126) return 2;
    for (int b = 0; b < 3; b++) { if (fscanf(f, "%d", &row[b]) != 1) return 2; for (int s = 1; s <= m; s++) { if (fscanf(f, "%d %d", &lo[b][s], &hi[b][s]) != 2) return 2; if (b == blk && hi[b][s] - lo[b][s] > 12) { fprintf(stderr, "window wider than 4 values\n"); return 2; } } }
    if (fscanf(f, "%d", &nh) != 1) return 2; for (int i = 0; i < nh; i++) if (fscanf(f, "%d", &H[i]) != 1) return 2;
    static int QT[128][128]; for (int i = 0; i < nh; i++) for (int s = 1; s <= m; s++) { int q = (int)((long)s * H[i] % n); if (q > m) q = n - q; QT[i][s] = q; }
    size_t cap = 1; while (cap < (size_t)(1.6 * (double)maxc) + 16) cap <<= 1;
    u128 *tab = malloc(sizeof(u128) * cap); uint8_t *cnt = calloc(cap, 1); if (!tab || !cnt) { fprintf(stderr, "out of memory\n"); return 1; }
    FILE *rej = argc > 4 ? fopen(argv[4], "w") : NULL; if (argc > 4 && !rej) return 2;
    char line[512]; long tot = 0, bad = 0, distinct = 0, dropped = 0; u128 MSK = (((u128)1) << n) - 1; static char obuf[1 << 20]; setvbuf(stdout, obuf, _IOFBF, sizeof obuf);
    while (fgets(line, sizeof line, stdin)) {
        if ((int)strlen(line) < n) continue;
        u128 X = 0; int sum = 0, P[128], ok = 1; for (int j = 0; j < n; j++) if (line[j] == '+') { X |= ((u128)1) << j; sum++; } else sum--;
        tot++;
        for (int s = 1; s <= m; s++) { u128 D = X ^ (((X << s) | (X >> (n - s))) & MSK); P[s] = n - 2 * (__builtin_popcountll((uint64_t)D) + __builtin_popcountll((uint64_t)(D >> 64))); ok &= P[s] >= lo[blk][s] && P[s] <= hi[blk][s]; }
        if (!ok || sum != row[blk]) { bad++; if (rej) { line[n] = '\n'; fwrite(line, 1, (size_t)n + 1, rej); } continue; }
        u128 best = 0;
        for (int i = 0; i < nh; i++) { u128 k = 0; for (int s = 1; s <= m; s++) { int q = QT[i][s]; k = (k << 2) | (u128)((P[q] - lo[blk][s]) / 4); } if (i == 0 || k < best) best = k; }
        uint64_t hsh = (uint64_t)best * 0x9E3779B97F4A7C15ull ^ (uint64_t)(best >> 64) * 0xC2B2AE3D27D4EB4Full; hsh ^= hsh >> 29; size_t p = (size_t)(hsh * 0xBF58476D1CE4E5B9ull >> 20) & (cap - 1);
        while (cnt[p] && tab[p] != best) p = (p + 1) & (cap - 1);
        if (!cnt[p]) { if (distinct >= maxc) { dropped++; continue; } tab[p] = best; cnt[p] = 1; distinct++; line[n] = '\n'; fwrite(line, 1, (size_t)n + 1, stdout); }
        else if (cnt[p] < 255) cnt[p]++;
    }
    long hist[16] = { 0 }; for (size_t p = 0; p < cap; p++) if (cnt[p]) hist[cnt[p] < 15 ? cnt[p] : 15]++;
    fprintf(stderr, "%ld rows, %ld not hits, %ld classes; multiplicities:", tot, bad, distinct); for (int j = 1; j < 16; j++) if (hist[j]) fprintf(stderr, " %ldx%d%s", hist[j], j, j == 15 ? "+" : ""); fprintf(stderr, "\n");
    if (dropped) fprintf(stderr, "WARNING: %ld hits arrived after max_classes classes were kept and were dropped without a duplicate check\n", dropped);
    if (rej && fclose(rej)) return 1;
    return fflush(stdout) ? 1 : 0;
}
