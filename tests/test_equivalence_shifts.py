"""equiv-v2 (shift quotient) and the small-order censuses of the unconstrained tracks."""

from copy import deepcopy
from itertools import product

import pytest

from verify.census_engine import C_ENGINES, c_census, c_census_count, shift_class_totals, verify
from verify.equivalence import canonical, canonical_shifts
from verify.gs_verify import memberships, paf, row_types, validate_rows

GENERAL = {3: 1, 5: 1, 7: 4, 9: 13, 11: 48, 13: 588}
SKEW = {3: 1, 5: 1, 7: 4, 9: 9, 11: 24, 13: 247}


def test_shift_quotient_merges_the_contract_example():
    skew = ((1, 1, -1),) * 4
    symmetric = ((1, -1, -1),) * 4
    assert canonical(skew) != canonical(symmetric)
    assert canonical_shifts(skew) == canonical_shifts(symmetric)


def test_shift_quotient_is_invariant_under_the_generators():
    q = max(c_census("general-census-c-v1", 13))
    c = canonical_shifts(q)
    shifted = tuple(r[s:] + r[:s] for r, s in zip(q, (0, 3, 7, 12), strict=True))
    assert validate_rows(shifted) and canonical(shifted) != canonical(q)
    assert canonical_shifts(shifted) == c
    assert canonical_shifts(tuple(tuple(-x for x in r) for r in q)) == c
    assert canonical_shifts(q[::-1]) == c
    assert canonical_shifts(tuple(tuple(r[(5 * j) % 13] for j in range(13)) for r in q)) == c
    assert canonical_shifts(c) == c


@pytest.mark.parametrize("n", [3, 5])
def test_shift_censuses_match_brute_force(n):
    rows = [(1, *t) for t in product((1, -1), repeat=n - 1)]
    pafs = {r: paf(r) for r in rows}
    found, skew = set(), set()
    for q in product(rows, repeat=4):
        if all(sum(v) == 0 for v in list(zip(*(pafs[r] for r in q), strict=True))[1:]):
            found.add(canonical_shifts(q))
            if any("k" in row_types(r) for r in q):
                skew.add(canonical_shifts(q))
    assert found == c_census("general-census-c-v1", n)
    assert skew == c_census("skew-sds-census-c-v1", n)


@pytest.mark.parametrize("n", sorted(GENERAL))
def test_shift_census_counts(n):
    general = c_census("general-census-c-v1", n)
    skew = c_census("skew-sds-census-c-v1", n)
    assert (len(general), len(skew)) == (GENERAL[n], SKEW[n])
    assert skew <= general


def test_shift_engines_declare_equiv_v2():
    for engine in ("general-census-c-v1", "skew-sds-census-c-v1"):
        assert C_ENGINES[engine]["equivalence"] == "equiv-v2"


def test_census_equivalence_must_match_the_enumerator():
    rows = c_census("general-census-c-v1", 3).pop()
    quads = {"q": {"n": 3, "tracks": memberships(rows), "canonical_rows": canonical(rows)}}
    entry = {
        "id": "general-n3",
        "n": 3,
        "stratum": "general",
        "equivalence": "equiv-v2",
        "representatives": ["q"],
        "claimed_count": 1,
        "claim_kind": "computed-exact",
        "enumerator": "general-census-c-v1",
    }
    assert verify(entry, quads)["bound"] == "exact"
    wrong = deepcopy(entry)
    wrong["equivalence"] = "equiv-v1"
    with pytest.raises(ValueError, match="differs from the enumerator"):
        verify(wrong, quads)
    wrong = deepcopy(entry)
    wrong["enumerator"] = "small-exact-v1"
    with pytest.raises(ValueError, match="no trusted enumerator"):
        verify(wrong, quads)


@pytest.mark.parametrize("n", sorted(GENERAL))
def test_count_only_build_agrees_with_the_written_classes(n):
    assert shift_class_totals(n) == {"classes_all": GENERAL[n], "classes_skew": SKEW[n]}


def test_count_only_census():
    assert c_census_count("general-census-c-v1", 15) == 17870
    assert c_census_count("skew-sds-census-c-v1", 15) == 4522
    with pytest.raises(ValueError, match="no count-only replay"):
        c_census_count("general-census-c-v1", 13)
    entry = {
        "id": "skew-sds-n15",
        "n": 15,
        "stratum": "skew-sds",
        "equivalence": "equiv-v2",
        "representatives": [],
        "claimed_count": 4522,
        "claim_kind": "computed-exact",
        "enumerator": "skew-sds-census-c-v1",
        "count_only": True,
    }
    result = verify(entry, {})
    assert (result["bound"], result["count"], result["tier"]) == ("exact", 4522, "reproduced")
    assert result["checks"] == ["full-enumeration-count-replayed"] and result["limitations"]
    unreplayed = verify(entry, {}, reproduce=False)
    assert (unreplayed["bound"], unreplayed["tier"]) == ("claimed", None)
    for change, message in (
        ({"claimed_count": 4521}, "disagrees with the claimed count"),
        ({"representatives": ["q"]}, "lists no representatives"),
        ({"claim_kind": "published-exact"}, "lists no representatives"),
        ({"enumerator": "general-census-c-v1"}, "no trusted enumerator"),
        ({"equivalence": "equiv-v1"}, "differs from the enumerator"),
        ({"n": 13}, "no count-only replay"),
    ):
        with pytest.raises(ValueError, match=message):
            verify({**entry, **change}, {})
