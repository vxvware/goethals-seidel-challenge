from copy import deepcopy
from itertools import permutations
import random

import pytest

from verify.common import ROOT, read_json
from verify.gs_verify import paf, validate_rows, verify, memberships
from verify.equivalence import canonical, fingerprint, normalize, reverse
from verify.model import validate


@pytest.fixture
def example():
    return read_json(next((ROOT / "quadruples").glob("n13-*.json")))


def test_bit_paf_matches_naive():
    rng = random.Random(713)
    for n in (1, 3, 13, 65, 129, 511):
        row = tuple(rng.choice((-1, 1)) for _ in range(n))
        assert paf(row) == tuple(sum(row[j] * row[(j + s) % n] for j in range(n)) for s in range(n))


def test_seed_and_single_bit_mutation(example):
    validate(example)
    verify(example)
    q = deepcopy(example)
    # Flipping the origin of a skew row preserves its PAF (negated reversal).
    # A non-origin corruption must fail the defining identity.
    q["rows"][0][1] *= -1
    with pytest.raises(ValueError, match="PAF"):
        verify(q)


@pytest.mark.parametrize("bad", [True, 1.0, 0, 2, "1", None])
def test_reject_non_integer_domain(example, bad):
    example["rows"][0][0] = bad
    with pytest.raises(ValueError):
        verify(example)


def test_group_shape_and_indices(example):
    q = deepcopy(example)
    q["n"] += 2
    with pytest.raises(ValueError):
        verify(q)
    q = deepcopy(example)
    q["group"] = {"kind": "abelian", "invariant_factors": [13]}
    with pytest.raises(ValueError):
        verify(q)
    q = deepcopy(example)
    q["blocks"] = [[j for j, x in enumerate(r) if x < 0] for r in q.pop("rows")]
    assert verify(q)["fingerprint"] == verify(example)["fingerprint"]
    q["blocks"][0] += [q["blocks"][0][0]]
    with pytest.raises(ValueError):
        verify(q)


def test_generator_invariance_and_idempotence(example):
    q = tuple(map(tuple, example["rows"]))
    want = canonical(q)
    n = example["n"]
    assert canonical(want) == want
    for ordering in permutations(q):
        assert canonical(ordering) == want
    for i in range(4):
        changed = list(q)
        changed[i] = tuple(-x for x in changed[i])
        assert canonical(changed) == want
        changed = list(q)
        changed[i] = reverse(changed[i])
        assert canonical(changed) == want
    for t in (2, 3, 5):
        assert canonical(tuple(tuple(r[(t * j) % n] for j in range(n)) for r in q)) == want
    assert fingerprint(want) == fingerprint(q)


def test_shift_exclusion_worked_example():
    from verify.gs_verify import row_types

    assert row_types((1, 1, -1)) == {"k"}
    assert row_types(normalize((-1, 1, 1))) == {"s"}


def test_typed_baseline_flag():
    best = read_json(next((ROOT / "quadruples").glob("n3-*.json")))
    baseline = {"id": "synthetic", "date": "2025-01-01", "orders": [12]}
    cases = [read_json(p) for p in (ROOT / "quadruples").glob("n3-*.json")]
    for q in cases:
        result = verify(q, baseline)
        assert bool(result["flags"]) == ("skew-sds" in result["tracks"])
    assert verify(best, {"id": "x", "date": "2025-01-01", "orders": []})["flags"] == []


def test_n1_degeneracy():
    assert set(memberships(((1,),) * 4)) == {
        "general",
        "skew-sds",
        "williamson",
        "good",
        "best",
        "kkss",
    }


def test_schema_rejects_self_declared_verdict(example):
    example["tier"] = "proven"
    with pytest.raises(ValueError):
        validate(example)


def test_input_bounds():
    for n in (0, 2, 513):
        with pytest.raises(ValueError):
            validate_rows([[1] * n] * 4, n)
