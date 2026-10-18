"""The Goethals–Seidel array shown on the site and drawn by picture.js is Hadamard."""

import glob
import json
from verify import gs_verify
from verify.common import read_json


def circulant(row):
    n = len(row)
    return [[row[(j - i) % n] for j in range(n)] for i in range(n)]


def transpose(m):
    return [list(col) for col in zip(*m)]


def times_r(m):
    """X R reverses the columns."""
    return [row[::-1] for row in m]


def neg(m):
    return [[-x for x in row] for row in m]


def array(rows):
    a, b, c, d = (circulant(r) for r in rows)
    at, bt, ct, dt = (transpose(x) for x in (a, b, c, d))
    blocks = [
        [a, times_r(b), times_r(c), times_r(d)],
        [neg(times_r(b)), a, neg(times_r(dt)), times_r(ct)],
        [neg(times_r(c)), times_r(dt), a, neg(times_r(bt))],
        [neg(times_r(d)), neg(times_r(ct)), times_r(bt), a],
    ]
    n = len(rows[0])
    return [[blocks[i // n][j // n][i % n][j % n] for j in range(4 * n)] for i in range(4 * n)]


def test_site_array_is_hadamard_and_skew_when_a_is_skew():
    baseline = read_json("baselines/skew-gaps.json")
    checked = 0
    for path in sorted(glob.glob("quadruples/n1[35]-*.json"))[:4]:
        doc = json.load(open(path))
        result = gs_verify.verify(doc, baseline)
        h = array(result["canonical_rows"])
        m = len(h)
        for i in range(m):
            for j in range(m):
                dot = sum(h[i][k] * h[j][k] for k in range(m))
                assert dot == (m if i == j else 0)
        if result["pattern"][0] == "k":
            assert all(
                h[i][j] + h[j][i] == (2 if i == j else 0) for i in range(m) for j in range(m)
            )
        checked += 1
    assert checked == 4
