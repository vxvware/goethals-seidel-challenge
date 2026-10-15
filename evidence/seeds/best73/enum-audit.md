# Enumeration audit and gap closure (n=73) — 2026-09-21

Prompted by the independent Codex review (`codex_plan_2026-09-20_best_matrix_audit.md`),
which proved the archived deck **omitted admissible classes**: `enum73.py`'s PSD
viability filter runs in **float32**, whose spacing near the bound `4n−3 = 289` is
~3×10⁻⁵, so a row with true max-PSD just below 289 can be pushed above it and dropped.
Codex exhibited D mask **1890925712** (product-compatible, all doubling-cycle products
−1, PSD max ≈ 288.99998) as absent from the deck.

## What we did (rigorous conservative re-enumeration)

`benchmark/enum73_f64.py` = `enum73.py` with the enumeration/FFT/filter in **float64**
(error ~10⁻¹³ ≪ the 1e-6 margin, so the filter is conservative: it never drops a truly
admissible row). `benchmark/reenum_f64.py` runs it and saves the full decimation-class
rep set. Ran on one GH200 in **339 s** (viable 402,420,258; classes 11,178,678).

**Diff of the float64 (ground-truth) rep set against the archived deck:**

| direction | count | meaning | action |
|---|---:|---|---|
| in float64, **not** in deck | **2** | float32 false-negatives (admissible, never scanned) | must settle |
| in deck, not in float64 | 14 | float32 false-positives (truly PSD>289, inadmissible) | harmless — were scanned empty, and can't hold a best matrix anyway |

Of the **2 omissions**, only **1 is product-compatible** (`1890925712`, max-PSD
288.99998); the other fails the Layer-A product sieve → provably empty.

## Settling the one omitted product-compatible class

Ran the **production pipeline** on a 1-mask deck (after the n=57 correctness gate PASSED
on the same box):
```
mask 1890925712 → 11,200 systems, 0 solutions, 0 unsettled  (ckpt: omit_ckpt.json)
```
This matches Codex's independent CPU completion (11,200 systems, 0 solutions) exactly.

## Conclusion — nonexistence restored rigorously

The float64 re-enumeration is a **superset of the true admissible set** (conservative
filter). Every product-compatible class in it is now settled empty:
- the 11,178,690 deck classes (original campaign; 69 checkpoints re-verified by Codex — 0
  sols, 0 unsettled), plus
- the 1 omitted product-compatible class (settled empty here), plus
- non-product-compatible classes are provably empty by the proven product theorem.

Hence **no best matrix of order 73 exists** (circulant Goethals–Seidel construction),
with the enumeration gap closed. Artifacts in `data/enum_audit_2026-09-21/`:
`reps_f64_n73.npy` (float64 rep set), `omit_pc.npy` (the omitted PC mask), `omit_ckpt.json`
(completion receipt), `enum73_f64.py` / `reenum_f64.py` (conservative enumerator).

## Remaining (lower-priority) audit items from the review
- **Provenance:** the legacy 84,500-class 10–20M prefix (`gpu_ckpt_1020.json`) and
  `chunk_32.log` lack modern receipts (identity/digest). Codex confirmed the 69 modern
  checkpoints pass the stronger identity audit; the legacy items are a provenance
  inconsistency, not evidence of missed work. Recover logs or re-run those 104,500 classes
  under the modern receipt format for a fully-identified coverage union.
- **Verifier hardening:** `verify_coverage.py` should also check checkpoint identity/digest,
  `done==count`, and require (not default) `unsettled_n`/`sols`.
- **Wording:** the completion guard marks systems of nullity > 16 as *unsettled*; it does
  not require an empty nullspace. `unsettled_n==0` means no such exception remained.
