"""The published constructions of Williamson quadruples at even n, checked by the verifier."""

import pytest

from tools.construct_even_williamson import (
    antipalindromic_nega,
    board_williamson,
    correspondence,
    double,
    doubling,
    has_unit_products,
    interleave,
    negadouble,
    power_of_two,
    product,
    recipe,
    with_unit_products,
)
from verify.common import ROOT
from verify.gs_verify import memberships, validate_rows


def rows(*words):
    return tuple(tuple(1 if c == "+" else -1 for c in word) for word in words)


def williamson(q):
    q = validate_rows([list(r) for r in q])
    return "williamson" in memberships(q)


@pytest.fixture(scope="module")
def board():
    return board_williamson()


@pytest.mark.parametrize("t", range(9))
def test_power_of_two_quadruples_are_williamson_with_unit_products(t):
    q = power_of_two(t)
    assert len(q[0]) == 1 << t and williamson(q) and has_unit_products(q)


def test_interleaving_reproduces_the_printed_example_of_length_sixteen():
    # bright2020new prints this quadruple for d(W) x n(V) with W four copies of ++-+ and V as below.
    w = rows("++-+", "++-+", "++-+", "++-+")
    v = rows("+-+-", "+-+-", "++--", "++--")
    built = tuple(interleave(double(x), negadouble(y)) for x, y in zip(w, v, strict=True))
    assert built == rows("+++--++-+-++--++", "+++--++-+-++--++", "++++--+-+-+--+++", "++++--+-+-+--+++")
    assert williamson(built)


@pytest.mark.parametrize("t", range(1, 7))
def test_nega_sequences_are_antipalindromic_and_negacomplementary(t):
    q = antipalindromic_nega(t)
    n = 1 << t
    assert all(len(x) == n and all(x[k] == -x[n - 1 - k] for k in range(n)) for x in q)
    for s in range(1, n):
        total = sum(sum(x[j] * (x[j + s] if j + s < n else -x[j + s - n]) for j in range(n)) for x in q)
        assert total == 0


def test_correspondence_matches_the_printed_example():
    table = correspondence()
    # (++-+) four times corresponds to the sequence -1, -1, 1, -1 over the quaternions.
    assert [table[e] for e in zip(*rows("++-+", "++-+", "++-+", "++-+"), strict=True)] == [
        (-2, 0, 0, 0),
        (-2, 0, 0, 0),
        (2, 0, 0, 0),
        (-2, 0, 0, 0),
    ]


def test_doubling_of_every_small_board_entry_is_williamson_with_unit_products(board):
    checked = 0
    for m in (3, 5, 7, 9, 11, 13):
        for _, q in board[m]:
            built = doubling(q)
            assert len(built[0]) == 2 * m and williamson(built) and has_unit_products(built)
            checked += 1
    assert checked >= 6
    with pytest.raises(ValueError, match="odd length"):
        doubling(power_of_two(2))


def test_product_of_lengths_ten_and_thirteen_is_a_williamson_quadruple_of_length_130(board):
    seed = next(unit for _, q in board[10] if (unit := with_unit_products(q)))
    built = product(seed, board[13][0][1])
    assert len(built[0]) == 130 and williamson(built)
    with pytest.raises(ValueError, match="coprime"):
        product(seed, board[5][0][1])
    without = next(q for _, q in board[12])
    assert with_unit_products(without) is None
    with pytest.raises(ValueError, match="unit products"):
        product(without, board[13][0][1])


def test_every_recipe_passes_the_exact_check(board):
    reached, missing = [], []
    for n in range(22, 301, 2):
        built = recipe(n, board)
        if built is None:
            missing.append(n)
            continue
        q, how = built
        assert len(q[0]) == n and williamson(q), (n, how)
        reached.append(n)
    # Every even n where a source the board cites gives existence is reached, and the product reaches more.
    assert {22, 64, 128, 130, 140, 256, 286, 300} <= set(reached)
    # 280 = 40 x 7 needs an even factor from a catalogue file, when the board has one; 162 = 2 x 81 has none.
    assert {94, 106, 118, 250} <= set(missing)
    catalogues = (ROOT / "evidence/catalogues/williamson-n40.txt.gz").is_file()
    assert (280 in reached) == catalogues and (280 in missing) != catalogues and 162 in missing
    assert recipe(128, board)[1]["kind"] == "power-of-two" and recipe(22, board)[1]["kind"] == "doubling"
    assert recipe(44, board)[1] | {"entry": ""} == {"kind": "product", "e": 4, "m": 11, "entry": "", "t": 2}
