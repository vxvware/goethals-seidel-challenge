"""FINAL COVERAGE VERIFIER for the n=73 nonexistence search — the teardown gate.

Proves that EVERY product-compatible decimation class (the only classes that can
contain a best matrix, by the proven product theorem) has been fully scanned and
settled (0 unsettled, 0 solutions), with no gap and no double-counting, across ALL
coverage sources:
  - the mod-16 GPU fleet manifest (fleet_manifest.json + local ckpt_<id>.json),
  - the main-box 10-20M partial run (gpu_ckpt_1020.json, covers deck [0,84500)).

Ground truth = the 1,400,390 reps of enum_n73.npz that pass the Layer-A sieve,
partitioned into mA = est*2^17 tiers. Coverage is checked at the level of each
tier's deck [0, decklen): the union of scanned [start,start+count) intervals must
equal [0, decklen) exactly. Every scanned chunk's checkpoint must show
cursor>=start+count, unsettled_n==0, sols==0.

Exit 0 + "COVERAGE COMPLETE" only if the whole space is provably covered and clean.
Run from _bigfiles_local/gh200_campaign/ (needs fleet_manifest.json + ckpt_*.json
+ gpu_ckpt_1020.json). Safe to run any time; before completion it reports the gaps.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_sieve import sieve_masks

N, H, D0 = 73, 36, 1
_HERE=os.path.dirname(os.path.abspath(__file__))
ENUM = os.path.join(_HERE,"..","data","enum_n73.npz")
STATE = os.path.join(_HERE,"..","data","campaign_state")
TIERS = [(0.0, 2e6), (2e6, 5e6), (5e6, 1e7), (1e7, 2e7), (2e7, 1.3e8), (1.3e8, float("inf"))]

def ground_truth():
    z = np.load(ENUM)
    reps, est = z["reps"].astype(np.uint64), z["est"]
    mA = est.astype(np.float64) * (1 << 17)
    ok = sieve_masks(reps, N, H, D0)
    pc = mA[ok]
    decklen = {}
    for lo, hi in TIERS:
        decklen[(lo, hi)] = int(((pc >= lo) & (pc < hi)).sum())
    return int(ok.sum()), decklen

def tier_key(lo_s, hi_s):
    lo = float(lo_s); hi = float(hi_s)
    for t in TIERS:
        if abs(t[0]-lo) < 1 and (t[1]==hi or abs(t[1]-hi) < 1):
            return t
    return (lo, hi)

def union_exact(intervals, lo, hi):
    """True iff sorted intervals tile [lo,hi) with no gap/overlap."""
    iv = sorted(intervals)
    cur = lo
    for a, b in iv:
        if a != cur:
            return False, cur, (a, b)
        cur = b
    return cur == hi, cur, None

def main():
    pc_total, decklen = ground_truth()
    print(f"ground truth: {pc_total:,} product-compatible classes")
    assert pc_total == 1_400_390, f"UNEXPECTED product-compatible total {pc_total}"
    assert decklen[(1.3e8, float("inf"))] == 0, "classes exist with mA>=1.3e8 (tier UNASSIGNED)"
    print("  per-tier decklen:", {f"[{a:.3g},{b:.3g})": n for (a, b), n in decklen.items() if n})

    m = json.load(open(os.path.join(STATE,"fleet_manifest.json")))
    problems, covered = [], {}     # covered[tier] = list of (start,end) intervals
    unsettled_total = sols_total = 0

    # not-done fleet chunks => incomplete
    notdone = [c["id"] for c in m["chunks"] if c["status"] != "done"]
    if notdone:
        problems.append(f"{len(notdone)} fleet chunks not done: {notdone[:12]}{'...' if len(notdone)>12 else ''}")

    for c in m["chunks"]:
        t = tier_key(c["lo"], c["hi"])
        s, cnt = c["start"], c["count"]
        if c["status"] != "done":
            continue
        ck = os.path.join(STATE,f"ckpt_{c['id']}.json")
        if not os.path.exists(ck):
            problems.append(f"chunk {c['id']}: DONE but no local ckpt"); continue
        d = json.load(open(ck))
        cur = d.get("cursor", -1); un = d.get("unsettled_n", 0); so = d.get("sols", 0)
        if isinstance(un, list): un = len(un)
        if isinstance(so, list): so = len(so)
        unsettled_total += un; sols_total += so
        if cur < s + cnt:
            problems.append(f"chunk {c['id']}: cursor {cur} < end {s+cnt} (not fully scanned)")
        if un != 0:
            problems.append(f"chunk {c['id']}: unsettled_n={un}")
        if so != 0:
            problems.append(f"chunk {c['id']}: sols={so}  <-- SOLUTION?!")
        covered.setdefault(t, []).append((s, s + cnt))

    # main-box 10-20M partial: covers deck [0, cursor)
    gp = os.path.join(STATE,"gpu_ckpt_1020.json")
    if os.path.exists(gp):
        d = json.load(open(gp))
        cur = d.get("cursor", 0); un = d.get("unsettled_n", 0); so = d.get("sols", 0)
        unsettled_total += un; sols_total += so
        if un: problems.append(f"main-box 10-20M: unsettled_n={un}")
        if so: problems.append(f"main-box 10-20M: sols={so}  <-- SOLUTION?!")
        covered.setdefault((1e7, 2e7), []).append((0, cur))
        print(f"  main-box 10-20M evidence: deck [0,{cur}) unsettled={un} sols={so}")
    else:
        problems.append("gpu_ckpt_1020.json (main-box 10-20M evidence) MISSING")

    # per-tier: union of covered intervals must equal [0, decklen)
    print("\ntier coverage:")
    all_ok = True
    for (lo, hi), dl in decklen.items():
        if dl == 0:
            print(f"  [{lo:.3g},{hi:.3g}): 0 classes (nothing to cover) OK"); continue
        iv = covered.get((lo, hi), [])
        ok, reached, gap = union_exact(iv, 0, dl)
        status = "OK" if ok else f"INCOMPLETE (covered up to {reached}/{dl}" + (f", gap before {gap}" if gap else "") + ")"
        if not ok: all_ok = False
        print(f"  [{lo:.3g},{hi:.3g}): {len(iv)} intervals -> [0,{dl}) {status}")

    print(f"\nunsettled across ALL sources: {unsettled_total} | solutions: {sols_total}")
    ok = all_ok and not problems and unsettled_total == 0 and sols_total == 0 and not notdone
    if problems:
        print("PROBLEMS:")
        for p in problems[:30]:
            print("  -", p)
    print("\n" + ("=== COVERAGE COMPLETE: all 1,400,390 product-compatible classes settled, 0 unsettled, 0 solutions -> SAFE TO TEARDOWN ==="
                  if ok else "=== COVERAGE INCOMPLETE / DIRTY -> DO NOT TEARDOWN ==="))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
