# Plan for the fixed-skew-block pool join at an odd order n (see PROVENANCE.md).
# usage: pool_plan.py n gens w [wide_values] > plan       env ROWS="b,c,d" fixes the row sums of blocks B, C, D
#   gens: comma-separated multipliers generating a subgroup H of Z_n^* of odd order ("1" for the trivial group)
# The skew block A is constant on the orbits of H, with opposite signs on an orbit and its negative; among those
# sequences the plan takes the best by a Gaussian model of the solution count (all of them when there are at
# most 2^16, otherwise the best found by hill climbing from SEED). For each lag and each of B, C, D the plan
# fixes a window of w consecutive admissible autocorrelation values (w + 1 on the lags where PAF_A takes one of
# the wide_values least favourable values), placed so that in-window triples can cancel PAF_A. PAF_A is
# invariant under H and -1, so the windows are too. Plan format: n; A as +/-; for each of B, C, D its row sum
# and then "lo hi" for lags 1..(n-1)/2; last line: |H| and the elements of H. The model's pool sizes and
# expected solution count (log2, in classes under shifts and negation) go to stderr.
import itertools
import math
import os
import random
import sys

n = int(sys.argv[1])
gens = [int(v) % n for v in sys.argv[2].split(",")]
w = int(sys.argv[3])
wide = int(sys.argv[4]) if len(sys.argv) > 4 else 0
m = (n - 1) // 2
base = n % 4
sig = math.sqrt(n)

H = {1}
frontier = [1]
while frontier:
    x = frontier.pop()
    for g in gens:
        if math.gcd(g, n) != 1:
            sys.exit("generator %d is not a unit mod %d" % (g, n))
        y = x * g % n
        if y not in H:
            H.add(y)
            frontier.append(y)
H = sorted(H)
if len(H) % 2 == 0:
    sys.exit("H has even order %d: no skew sequence is invariant under it" % len(H))

# orbits of H on the nonzero residues, paired with their negatives
seen = set()
pairs = []
for x in range(1, n):
    if x in seen:
        continue
    orb = sorted({x * t % n for t in H})
    neg = sorted({(n - y) % n for y in orb})
    seen.update(orb)
    seen.update(neg)
    pairs.append((orb, neg))


def sequence(eps):
    a = [0] * n
    a[0] = 1
    for e, (orb, neg) in zip(eps, pairs):
        for y in orb:
            a[y] = e
        for y in neg:
            a[y] = -e
    return a


def paf(a):
    return [sum(a[j] * a[(j + s) % n] for j in range(n)) for s in range(1, m + 1)]


pats = [(b, c, d) for b in range(1, 2 * n, 2) for c in range(b, 2 * n, 2) for d in range(c, 2 * n, 2)
        if 1 + b * b + c * c + d * d == 4 * n]
if os.environ.get("ROWS"):
    pats = [tuple(int(v) for v in os.environ["ROWS"].split(","))]


def setup(rows):
    mu = [(r * r - n) / (n - 1) for r in rows]

    def pm(i, v):
        return math.exp(-((v - mu[i]) ** 2) / (2 * sig * sig))

    Z = [sum(pm(i, base + 4 * k) for k in range(-40, 41)) for i in range(3)]
    cache = {}

    def opt(t, ww):
        if (t, ww) in cache:
            return cache[(t, ww)]
        bj = None
        for o in itertools.product(range(-7, 5), repeat=3):
            wins = [[base + 4 * (o[i] + k) for k in range(ww)] for i in range(3)]
            J = 0
            for vb in wins[0]:
                for vc in wins[1]:
                    vd = t - vb - vc
                    if vd in wins[2]:
                        J += pm(0, vb) / Z[0] * pm(1, vc) / Z[1] * pm(2, vd) / Z[2]
            if bj is None or J > bj[0]:
                bj = (J, wins, [sum(pm(i, v) for v in wins[i]) / Z[i] for i in range(3)])
        cache[(t, ww)] = bj
        return bj

    return opt


def rowbits(rows):
    return [math.log2(math.comb(n, (n + r) // 2)) - math.log2(2 * n) for r in rows]


best = None
rng = random.Random(int(os.environ.get("SEED", "1")))
for rows in pats:
    opt = setup(rows)
    rb = sum(rowbits(rows))

    def score(eps):
        js = [opt(-v, w)[0] for v in paf(sequence(eps))]
        # a PAF_A value that no in-range window triple can cancel rules the sequence out
        return rb + sum(math.log2(j) for j in js) if min(js) > 0 else -math.inf

    k = len(pairs)
    if k <= 16:
        cands = (e for e in itertools.product([1, -1], repeat=k) if e[0] > 0)
        for eps in cands:
            sc = score(eps)
            if best is None or sc > best[0]:
                best = (sc, rows, eps)
    else:
        for _ in range(int(os.environ.get("RESTARTS", "4"))):
            eps = [rng.choice([1, -1]) for _ in range(k)]
            sc = score(eps)
            improved = True
            while improved:
                improved = False
                for i in rng.sample(range(k), k):
                    eps[i] = -eps[i]
                    s2 = score(eps)
                    if s2 > sc:
                        sc = s2
                        improved = True
                    else:
                        eps[i] = -eps[i]
            if best is None or sc > best[0]:
                best = (sc, rows, tuple(eps))

sc, rows, eps = best
if sc == -math.inf:
    sys.exit("no sequence found whose autocorrelations the windows can cancel; try another SEED or more RESTARTS")
a = sequence(eps)
p = paf(a)
opt = setup(rows)
rb = rowbits(rows)
vals = sorted(set(p), key=lambda v: opt(-v, w)[0])
widev = set(vals[:wide])
res = [opt(-p[s], w + 1 if p[s] in widev else w) for s in range(m)]
pools = [rb[i] + sum(math.log2(r[2][i]) for r in res) for i in range(3)]
E = sum(rb) + sum(math.log2(r[0]) for r in res)
print("n=%d |H|=%d w=%d wide=%d rows %s PAF_A values %s: model log2 pools %s, log2 E %.1f"
      % (n, len(H), w, wide, list(rows), sorted(set(p)), [round(x, 1) for x in pools], E), file=sys.stderr)
print(n)
print("".join("+" if v > 0 else "-" for v in a))
for i in range(3):
    print(rows[i])
    print(" ".join("%d %d" % (r[1][i][0], r[1][i][-1]) for r in res))
print(len(H), " ".join(str(t) for t in H))
