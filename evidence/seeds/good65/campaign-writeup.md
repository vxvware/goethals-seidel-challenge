# No good matrices of order 65 exist — 2026-09-24

**Result.** There are no circulant good matrices of order n=65 (A skew, B,C,D
symmetric, AAᵀ+BBᵀ+CCᵀ+DDᵀ=260·I). This settles a previously open order and
extends the nonexistence run {41,47,49,51,59,63,69} to include 65. Consequently
the Goethals–Seidel/good-matrix route to a skew-Hadamard matrix of order 260 is
closed (order 260 is known by other constructions).

**Method.** GPU linear-completion engine (`benchmark/good_gpu_complete.py`):
per outer symmetric-B decimation class, a fused-MITM skew-A pool (float64
conservative budgets, a₁=+1 halving), then one thread per (A,B) solving the
corrected folded GF(2) completion for C (Gaussian v4, then the CRT field
solver v5 — X²+YX=R in F₂[t]/Ψ₆₅ split into F4 ⊕ 5·F64, proven nullity ≤ 6),
D = g·B·C, exact in-kernel integer PAF verification, host re-verification of
any hit. Every engine version gated by: n=13 full verified-solution set
equality GPU=CPU (52=52), n=15/21 recovery, n=41 nonexistence, fused-pool and
halved-pool set-equality, CRT affine-set equality (10,000 systems).

**Coverage (fail-closed receipts; order-digest-verified resume seams).**

| pass (outer B row-sum) | classes | covered by | receipt |
|---|---|---|---|
| σ_B = −15 | 1,282,051 | box 192.222.58.99 [0, 517,400) (v1/v3/v4) + box 192.222.50.112 [517,400, 1,282,051) (v5) — digest `0541f36e58c1f5bc` both sides | `data/box_192.222.58.99/progress_complete_n65_sB-15.json`, `data/box_192.222.50.112/coverage_n65_sB-15.json` |
| σ_B = +13 | 1,863,512 | box 192.222.50.112 [0, 1,863,512) (v5) — digest `650c57c75d7d1426` | `data/box_192.222.50.112/coverage_n65_sB13.json` |

These are the only two row-sum decompositions of 4·65−1 = 259 (up to the
C↔D relabeling and sign normalizations the engine covers by construction;
σ_A = 1 forced by skewness). Pending lists empty everywhere; 0 systems ever
exceeded the nullity guard (consistent with the CRT bound ≤ 6 ≤ 12).

**Scale.** ≈ 6.1 trillion completion systems settled (≈ 2.4T in pass 1,
3.72T in pass 2), ≈ 127 billion GF(2)/CRT-consistent systems individually
exact-verified. Wall-clock ≈ 42 GH200-hours across two boxes (≈ $96).

**Independent-check invitation.** Every class is independently recomputable:
`good_gpu_complete.py 65 --pass-sB <s> --start <i> --dmax <i+k>` rebuilds the
deterministic class order (verify the printed order-digest) and re-settles any
slice; the CPU oracle (`good_complete_search.py`) cross-checks small ranges.
