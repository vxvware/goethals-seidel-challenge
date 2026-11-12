"""Generate Turyn's Williamson quadruples and import them as entries.

Turyn (1972) proved that Williamson matrices of order n = (q+1)/2 exist for
every prime power q with q = 1 (mod 4). The rows written here come from the
quadratic character chi of GF(q) applied to the coordinates of the cyclic
subgroup of order q+1 in GF(q^2)*: with theta primitive in GF(q^2),
omega = theta^(q-1), s = theta^((q+1)/2) (so s^q = -s and {1, s} is a
GF(q)-basis), write omega^j = a_j + b_j s. Then

    A = B = (chi(a_j))_j,   C = (chi(b_j))_j with C_0 = +1,   D = C with D_0 = -1,

where b_0 = 0 is the only zero. Every quadruple is checked exactly by the
board's verifier before it is written, and an order whose class is already on
the board keeps its existing entry (which is how the recipe is checked against
the published Turyn-type representatives at 61, 63 and 69).

    uv run python tools/import_turyn_williamson.py [--max 131] [--dry-run]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store  # noqa: E402
from verify.common import ROOT  # noqa: E402
from verify.equivalence import canonical, fingerprint  # noqa: E402
from verify.gs_verify import memberships, validate_rows  # noqa: E402
from verify.model import validate  # noqa: E402

SUBMITTER = "vprusso"
SOURCE = {
    "title": "Turyn, An infinite class of Williamson matrices, J. Combin. Theory Ser. A 12 (1972)",
    "url": "https://doi.org/10.1016/0097-3165(72)90095-7",
}


def prime_factors(m):
    out, p = [], 2
    while p * p <= m:
        while m % p == 0:
            out.append(p)
            m //= p
        p += 1
    if m > 1:
        out.append(m)
    return sorted(set(out))


def prime_power(q):
    """(p, e) with q = p^e, or None."""
    for p in prime_factors(q):
        e = 0
        m = q
        while m % p == 0:
            m //= p
            e += 1
        return (p, e) if m == 1 else None
    return None


class GF:
    """GF(p^m) as polynomials over GF(p) modulo an irreducible monic f of degree m."""

    def __init__(self, p, m):
        self.p, self.m = p, m
        self.one = self.el([1])
        self.zero = self.el([])
        self.minus_one = self.el([p - 1])
        self.f = self._irreducible()

    def el(self, coeffs):
        c = [x % self.p for x in coeffs] + [0] * (self.m - len(coeffs))
        return tuple(c[: self.m])

    def add(self, x, y):
        return tuple((a + b) % self.p for a, b in zip(x, y))

    def sub(self, x, y):
        return tuple((a - b) % self.p for a, b in zip(x, y))

    def mul(self, x, y):
        p, m = self.p, self.m
        prod = [0] * (2 * m - 1)
        for i, a in enumerate(x):
            if a:
                for j, b in enumerate(y):
                    prod[i + j] = (prod[i + j] + a * b) % p
        for k in range(2 * m - 2, m - 1, -1):  # reduce x^k using f monic
            c = prod[k]
            if c:
                for i in range(m):
                    prod[k - m + i] = (prod[k - m + i] - c * self.f[i]) % p
        return tuple(prod[:m])

    def pow(self, x, e):
        out = self.one
        while e:
            if e & 1:
                out = self.mul(out, x)
            x = self.mul(x, x)
            e >>= 1
        return out

    def inv(self, x):
        return self.pow(x, self.p**self.m - 2)

    def _poly_mod(self, a, f):
        a = list(a)
        while len(a) >= len(f):
            c = a[-1]
            if c:
                for i in range(len(f)):
                    a[len(a) - len(f) + i] = (a[len(a) - len(f) + i] - c * f[i]) % self.p
            a.pop()
        return a

    def _poly_gcd_is_one(self, a, f):
        a, b = list(f), self._poly_mod(a, f)
        while b and any(b):
            while b and b[-1] == 0:
                b.pop()
            if not b:
                break
            a, b = b, self._poly_mod(a, b)
        while a and a[-1] == 0:
            a.pop()
        return len(a) == 1

    def _irreducible(self):
        p, m = self.p, self.m
        rng = random.Random(p * 1000 + m)
        while True:
            f = [rng.randrange(p) for _ in range(m)] + [1]  # monic, coeffs of 1..x^m
            if f[0] == 0:
                continue
            self.f = f  # used by mul below
            x = self.el([0, 1])
            xp = x
            ok = True
            for k in range(1, m + 1):
                xp = self.pow(xp, p)  # x^(p^k) mod f
                if k < m and m % k == 0:
                    diff = list(self.sub(xp, x))
                    if not self._poly_gcd_is_one(diff, f):
                        ok = False
                        break
            if ok and xp == x:
                return f

    def primitive(self):
        order = self.p**self.m - 1
        rng = random.Random(order)
        while True:
            g = self.el([rng.randrange(self.p) for _ in range(self.m)])
            if g != self.zero and all(
                self.pow(g, order // r) != self.one for r in prime_factors(order)
            ):
                return g


def turyn_rows(n):
    q = 2 * n - 1
    pe = prime_power(q)
    if pe is None or q % 4 != 1:
        return None
    p, e = pe
    F = GF(p, 2 * e)
    theta = F.primitive()
    omega = F.pow(theta, q - 1)
    s = F.pow(theta, (q + 1) // 2)
    assert F.pow(s, q) == F.sub(F.zero, s)
    half = F.inv(F.el([2]))
    inv2s = F.inv(F.mul(F.el([2]), s))

    def chi(y):
        if y == F.zero:
            return 0
        v = F.pow(y, (q - 1) // 2)
        if v == F.one:
            return 1
        if v == F.minus_one:
            return -1
        raise ValueError("character value outside GF(q)")

    a_row, b_row = [], []
    x = F.one
    for _ in range(n):
        xq = F.pow(x, q)
        a = F.mul(F.add(x, xq), half)
        b = F.mul(F.sub(x, xq), inv2s)
        a_row.append(chi(a))
        b_row.append(chi(b))
        x = F.mul(x, omega)
    if 0 in a_row or b_row[0] != 0 or 0 in b_row[1:]:
        raise ValueError(f"unexpected zero pattern at n={n}")
    c = [1] + b_row[1:]
    d = [-1] + b_row[1:]
    return validate_rows([a_row, a_row, c, d], n)


def board_classes(n):
    found = {}
    for path in sorted((ROOT / "quadruples").glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("type") == "quadruple" and doc.get("n") == n:
            found[canonical(validate_rows([list(r) for r in doc["rows"]], n))] = doc["id"]
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=131)
    ap.add_argument("--date", default="2026-11-12")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for n in range(3, a.max + 1, 2):
        rows = turyn_rows(n)
        if rows is None:
            continue
        assert "williamson" in memberships(rows)
        rep = canonical(rows)
        existing = board_classes(n)
        if rep in existing:
            print(f"n={n} q={2 * n - 1}: Turyn class is already on the board as {existing[rep]}")
            continue
        quad_id = f"n{n}-{fingerprint(rep)[3:23]}"
        print(f"n={n} q={2 * n - 1}: new entry {quad_id}")
        if a.dry_run:
            continue
        doc = validate(
            {
                "schema_version": 1,
                "n": n,
                "group": {"kind": "cyclic", "invariant_factors": [n]},
                "submitted_by": SUBMITTER,
                "submitted_at": a.date,
                "discovered_by": [
                    "Turyn 1972, the infinite family of Williamson matrices of order (q+1)/2 "
                    "for prime powers q = 1 (mod 4); rows generated by tools/import_turyn_williamson.py"
                ],
                "sources": [SOURCE],
                "equivalence": "equiv-v1",
                "id": quad_id,
                "type": "quadruple",
                "rows": [list(r) for r in rep],
                "notes": (
                    f"q = {2 * n - 1}. With theta primitive in GF(q^2), omega = theta^(q-1) and "
                    "s = theta^((q+1)/2), write omega^j = a_j + b_j s over GF(q); A = B = chi(a_j), "
                    "and C, D = chi(b_j) with the zero at j = 0 filled by +1 and -1. The rows are "
                    "stored in canonical form under equiv-v1."
                ),
            }
        )
        store(ROOT / f"quadruples/{quad_id}.json", doc, rows_may_match=True)


if __name__ == "__main__":
    main()
