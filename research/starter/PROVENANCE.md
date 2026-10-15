# Starter provenance

`best/` originates from vxvware/saw-best-matrix, commit
8cdf530f5461eacabebb0fb3294e12b829115031 (origin/exact-census).
The C census has local hardening: supported-order bounds, fail-closed solution
buffer limits, output-file error checks, and conservative spectral margins.
It is replayed as a computation, not advertised as a formal proof. The SAW
files are retained as research kernels and excluded from the trusted verifier.

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
