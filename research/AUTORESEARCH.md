# Research workflow

1. Read ../AGENTS.md and ../notes/mathematical-contract.md.
2. Inspect `./gs targets --stratum <track>`. The order-status baseline is dated;
   verify current literature before making a novelty claim. Import tasks are
   exposed separately by `./gs import-tasks`.
3. State n, cyclic group, symmetry restrictions, budget, and search method.
   Keep random seeds, source commits, parameters, and progress records.
4. Use the starter kernels or an appropriate external engine. Each output is
   only a candidate until `./gs verify` succeeds on a schema-valid submission.
5. Stage verified candidates with `./gs submit`. Equivalent existing entries
   are rejected, even if rows have been negated, reversed, permuted, or decimated.
6. Preserve null results with their actual scope and interval coverage. A
   heuristic failure is not nonexistence. Missing receipts stay explicit gaps.

## Starters

- `starter/best/saw_fast.py` and `saw_n.py`: the original SAW kernels from
  saw-best-matrix @ 8cdf530f5461eacabebb0fb3294e12b829115031. They need NumPy and
  Numba and may launch workers; inspect their arguments and choose a budget.
- `starter/cyclotomic_sweep.py`: a bounded coset-union search using exact PAF
  keys. It deliberately avoids the old rounded-PSD key, which is not a
  completeness-safe way to match residual sums. Output is JSONL candidates.
- `starter/good_gpu/`: the archived v4 Gaussian engine and its dependencies,
  pinned from good-matrix-campaign @ 018ef2d7754017eca1f9f22c8ddd84b9b6cf5581.
  It requires a CUDA GPU/CuPy and substantial memory at large orders.

GPU campaign sources:

- https://github.com/vxvware/best-matrix-campaign
- https://github.com/vxvware/good-matrix-campaign
- https://github.com/vxvware/skew356-campaign

The CPU-only normal workflow never invokes these GPU engines. Recheck reports
must be reviewed and registered through a separate maintainer change before
changing a board verification tier.
