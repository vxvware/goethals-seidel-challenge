# Starter provenance

`best/` originates from vxvware/saw-best-matrix, commit
8cdf530f5461eacabebb0fb3294e12b829115031 (origin/exact-census).
The C census has local hardening: supported-order bounds, fail-closed solution
buffer limits, output-file error checks, and conservative spectral margins.
It is replayed as a computation, not advertised as a formal proof. The SAW
files are retained as research kernels and excluded from the trusted verifier.

`good/good_c.c` is the exact census of circulant good matrices from vxvware/saw-best-matrix, in the build that reran the n=65 and n=67 censuses after the wide-word defects recorded as finding 522 of that repository's optimization log; it is the source that runs on bos-game and archdesktop in October 2026. It is replayed as a computation with the wide build flags it was validated under, and only at odd orders 3 to 43, where a replay takes under a minute on two threads. Before it was pinned, its class sets under equiv-v1 were checked equal to the Python enumerator at orders 3 to 15, to the stored saw-best-matrix censuses at orders 17 to 45, and to the published counts of Bright, Đoković, Kotsireas and Ganesh (arXiv:1811.05094) at every order divisible by 3 up to 45. Its environment variables only tune performance; the verifier strips them anyway.

`williamson/williamson_c.c` is a new exact census of Williamson quadruples written for this board in October 2026: a meet in the middle over the symmetric sequences that survive the spectral ceiling, bucketed by the four-square decomposition of 4n, with an exact autocorrelation match and a recheck of every emitted quadruple. It is replayed at odd orders 3 to 37. Its class sets under equiv-v1 equal the classification of Holzmann, Kharaghani and Tayfeh-Rezaie (2008) at every order from 3 to 35, including the empty order 35.

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
