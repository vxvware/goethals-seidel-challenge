"""Generate Spence's G-matrix quadruples (kkss) and import them as entries.

Spence (1977) constructed skew-Hadamard matrices of order 2(q+1) of
Goethals-Seidel type for every prime power q = 5 (mod 8), with four circulant
blocks of order n = (q+1)/2: two skew and two symmetric, so the rows form a
G-matrix quadruple. The construction starts from the cyclic relative
difference set with parameters (q+1, 4, q, (q-1)/4) that the trace-1 set of
GF(q^2) induces in Z_{4(q+1)}. The steps below follow Spence's paper as
restated in SageMath's skew_hadamard_matrix_spence_construction; the GF(q^2)
arithmetic is shared with tools/import_turyn_williamson.py.

With theta primitive in GF(q^2) and N = 4(q+1) = 8n:

    R  = { i mod N : theta^i + theta^(qi) = 1 }          (|R| = q)
    R' = a translate of R fixed by multiplication by q and shifted by a
         multiple of q+1 so that it contains 0
    U  = R' union (R' + (q+1))                           (|U| = 2q)
    D_i = { x in Z_n : (i, x) in U }  for i = 0, 1, 2, 3,

where Z_N = Z_8 x Z_n by the Chinese remainder theorem. The four rows are the
±1 indicator sequences of D_0, D_1, D_2, D_3; D_1 gives one skew row, D_0 the
other, and D_2, D_3 the symmetric rows. The resulting equiv-v1 class does not
depend on the choice of theta (checked over every unit of Z_N at the orders on
the board). Every quadruple is checked exactly by the board's verifier before
it is written, and an order whose class is already on the board keeps its
existing entry; the recipe reproduces the board's classes at 3, 7, 15, 19, 27,
31, 51 and 55.

    uv run python tools/import_spence_kkss.py [--max 131] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_dashboard import store  # noqa: E402
from tools.import_turyn_williamson import GF, prime_power  # noqa: E402
from verify.common import ROOT  # noqa: E402
from verify.equivalence import canonical, fingerprint  # noqa: E402
from verify.gs_verify import memberships, validate_rows  # noqa: E402
from verify.model import validate  # noqa: E402

SUBMITTER = "vprusso"
SOURCE = {
    "title": "Spence, Skew-Hadamard matrices of order 2(q+1), Discrete Math. 18 (1977) 79-85",
    "url": "https://doi.org/10.1016/0012-365X(77)90009-0",
}


def trace_one_set(F, theta, q, modulus):
    """Residues i mod `modulus` with theta^i + theta^(qi) = 1 in GF(q^2)."""
    found = set()
    x = F.one
    for i in range(q * q - 1):
        if F.add(x, F.pow(x, q)) == F.one:
            found.add(i % modulus)
        x = F.mul(x, theta)
    return found


def spence_rows(n):
    q = 2 * n - 1
    pe = prime_power(q)
    if pe is None or q % 8 != 5:
        return None
    p, e = pe
    F = GF(p, 2 * e)
    theta = F.primitive()
    big = 8 * n  # 4(q+1)
    rds = trace_one_set(F, theta, q, big)
    if len(rds) != q:
        raise ValueError(f"relative difference set of size {len(rds)} at n={n}")
    fixed = next(
        (
            shifted
            for shifted in ({(t + r) % big for r in rds} for t in range(big))
            if all((q * s) % big in shifted for s in shifted)
        ),
        None,
    )
    if fixed is None:
        raise ValueError(f"no translate fixed by q at n={n}")
    fixed = next(
        shifted
        for shifted in ({(2 * n * i + s) % big for s in fixed} for i in range(big))
        if 0 in shifted
    )
    union = fixed | {(s + 2 * n) % big for s in fixed}
    inv8 = pow(8, -1, n)
    blocks = [{(el * inv8) % n for el in union if el % 8 == i} for i in range(4)]
    rows = [[1 if j in block else -1 for j in range(n)] for block in blocks]
    a = [-v for v in rows[1]]
    b, c, d = rows[0], rows[2], rows[3]
    return validate_rows([a, b, c, d], n)


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
    ap.add_argument("--date", default="2026-11-14")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for n in range(3, a.max + 1, 2):
        rows = spence_rows(n)
        if rows is None:
            continue
        assert "kkss" in memberships(rows)
        rep = canonical(rows)
        existing = board_classes(n)
        if rep in existing:
            print(f"n={n} q={2 * n - 1}: Spence class is already on the board as {existing[rep]}")
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
                    "Spence 1977, the infinite family of G-matrices of order (q+1)/2 for prime "
                    "powers q = 5 (mod 8); rows generated by tools/import_spence_kkss.py"
                ],
                "sources": [SOURCE],
                "equivalence": "equiv-v1",
                "id": quad_id,
                "type": "quadruple",
                "rows": [list(r) for r in rep],
                "notes": (
                    f"q = {2 * n - 1}. With theta primitive in GF(q^2) and N = 4(q+1), the set "
                    "R of residues i mod N with theta^i + theta^(qi) = 1 is a relative "
                    "difference set with parameters (q+1, 4, q, (q-1)/4); translate it to be "
                    "fixed by multiplication by q and to contain 0, take the union with its "
                    "shift by q+1, and split Z_N = Z_8 x Z_n: the elements with first "
                    "coordinate 0, 1, 2, 3 give the four rows (two skew, two symmetric) as "
                    "+1/-1 indicators on Z_n. The rows are stored in canonical form under "
                    "equiv-v1."
                ),
            }
        )
        store(ROOT / f"quadruples/{quad_id}.json", doc, rows_may_match=True)


if __name__ == "__main__":
    main()
