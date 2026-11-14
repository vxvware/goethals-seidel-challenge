# Starter provenance

`best/` originates from vxvware/saw-best-matrix, commit
8cdf530f5461eacabebb0fb3294e12b829115031 (origin/exact-census).
The C census has local hardening: supported-order bounds, fail-closed solution
buffer limits, output-file error checks, and conservative spectral margins.
It is replayed as a computation, not advertised as a formal proof. The SAW
files are retained as research kernels and excluded from the trusted verifier.

`good/good_c.c` is the exact census of circulant good matrices from vxvware/saw-best-matrix, in the build that reran the n=65 and n=67 censuses after the wide-word defects recorded as finding 522 of that repository's optimization log; it is the source that runs on bos-game and archdesktop in October 2026. It is replayed as a computation with the wide build flags it was validated under, and only at odd orders 3 to 43, where a replay takes under a minute on two threads. Before it was pinned, its class sets under equiv-v1 were checked equal to the Python enumerator at orders 3 to 15, to the stored saw-best-matrix censuses at orders 17 to 45, and to the published counts of Bright, Đoković, Kotsireas and Ganesh (arXiv:1811.05094) at every order divisible by 3 up to 45. Its environment variables only tune performance; the verifier strips them anyway.

`williamson/williamson_c.c` is a new exact census of Williamson quadruples written for this board in October 2026: a meet in the middle over the symmetric sequences that survive the spectral ceiling, bucketed by the four-square decomposition of 4n, with an exact autocorrelation match and a recheck of every emitted quadruple. It is replayed at odd orders 3 to 37. Its class sets under equiv-v1 equal the classification of Holzmann, Kharaghani and Tayfeh-Rezaie (2008) at every order from 3 to 35, including the empty order 35.

`williamson/wc_compress.c` and `williamson/wc_lift.c` are the Williamson compression pipeline (engine id williamson-compress-v1), written 2026-11-12 for the orders the direct census cannot reach and registered offline through baselines/census_replays.json. For odd n = d m, wc_compress enumerates the 2^((n-1)/2) symmetric sequences with x_0 = +1, keeps those whose row sum fits a decomposition of 4n into four odd squares and whose PSD never exceeds 4n, compresses each survivor to length m (entry j is the sum over the coset j + k m), and matches the realized compressions into quadruples per decomposition on the length-m PAF condition with the PSD pair filter; one instance per orbit of the multipliers of Z_m is written, which loses nothing because every unit of Z_m lifts to a unit of Z_n. wc_lift enumerates, for each instance, every full sequence with that compression (coset by coset, pruned by the remaining capacity), keeps those under the PSD ceiling, and matches the four lists on the exact PAF sum with the sharded pair table of williamson_c.c; every emitted quadruple is rechecked from its rows. The pipeline reproduces the direct census exactly at 25, 35 and 45, with and without the multiplier reduction, and the published classifications at 49, 51, 55 and 57 (Holzmann, Kharaghani and Tayfeh-Rezaie) and 63 (Bright, Kotsireas and Ganesh).

`williamson/wc_lift5.c` (engine id williamson-compress-v2, with the same wc_compress and the original lifter) is the faster lifter written 2026-11-13: it also compresses every lifted sequence by m, to length d, whose PAF the four sequences must still cancel, groups each list by that length-d PAF vector and enumerates only the pairs of groups that some pair of groups on the other side can meet; chooses, when the four lists are distinct, the meet-in-the-middle split with the least group-pair work; tests the PSD pair ceiling on 16-lane vectors, eight sequences per tile; indexes the pair table by key buckets instead of sorting it; lifts the cosets of one compressed sequence in parallel; and caches lifted lists across instances. It wrote the same quadruple sets as wc_lift at 45 (3,501 instances), 57 (58,408) and 63 (480,403), and is about sixteen times faster per instance at 65. An order-65 replay may split its instance lines between the two lifters; the report's command records the split.

`good_gpu/` is the Gaussian v4 snapshot archived with the n=67 campaign in
vxvware/good-matrix-campaign, commit
018ef2d7754017eca1f9f22c8ddd84b9b6cf5581. This is important: the n=67 run used
Gaussian v4, since the v5 CRT implementation could not represent its degree-33
field. Historical good-65 passes used multiple versions. Replaying with this
pinned engine is a new check, not proof of the exact original binary.

`cyclotomic_sweep.py` is a new exact-PAF join implementing the same precisely
stated same-subgroup search scope. The archived p=73 candidates are independently
verified on import; an old sweep result is never used as an exhaustive census.

Sage imports contain numerical rows extracted from a pinned literal table,
not executed Sage code. The source SHA and original Hall/London/Szekeres
references are retained on each imported entry.
