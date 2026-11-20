# Row sums for the orbit-restricted pool join (see PROVENANCE.md).
# usage: orbit_rows.py n gens      gens: comma-separated multipliers generating H, as for pool_plan.py
# Prints every triple b,c,d of odd positive integers with 1 + b^2 + c^2 + d^2 = 4n in which each entry is the
# absolute row sum of some +/-1 sequence of length n that is constant on the orbits of H.
import sys

n = int(sys.argv[1])
gens = [int(v) % n for v in sys.argv[2].split(",")]

H = {1}
frontier = [1]
while frontier:
    x = frontier.pop()
    for g in gens:
        y = x * g % n
        if y not in H:
            H.add(y)
            frontier.append(y)

seen = set()
sizes = []
for x in range(1, n):
    if x not in seen:
        orbit = {x * t % n for t in H}
        seen |= orbit
        sizes.append(len(orbit))

reach = {1, -1}
for s in sizes:
    reach = {r + s for r in reach} | {r - s for r in reach}
attainable = {abs(r) for r in reach}

for b in range(1, 2 * n, 2):
    for c in range(b, 2 * n, 2):
        for d in range(c, 2 * n, 2):
            if 1 + b * b + c * c + d * d == 4 * n and {b, c, d} <= attainable:
                print("%d,%d,%d" % (b, c, d))
