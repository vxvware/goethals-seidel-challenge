"""Materialize individual, sourced status records; never infer absence from a missing import."""

import sys
from math import isqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, JSONObject, write_json

records: list[JSONObject] = []


def add(
    track: str, n: int, status: str, url: str, title: str, date: str | None = None, **extra: JSONObject | int
) -> None:
    # The grid was extended from n = 131 to n = 300 on 2026-11-25, with the same rules and sources.
    date = date or ("2026-10-14" if n <= 131 else "2026-11-25")
    records.append(
        {
            "n": n,
            "stratum": track,
            "group": "cyclic",
            "status": status,
            "source": {"title": title, "url": url},
            "baseline_date": date,
            **extra,
        }
    )


for n in range(1, 64, 2):
    if n == 61:
        # The classification at 61 is not in the paper cited for the other
        # orders, which attributes it to this one.
        add(
            "williamson",
            n,
            "exists",
            "https://doi.org/10.1007/s10623-011-9492-2",
            "Lang, Schneider, Turyn type Williamson matrices up to order 99, Des. Codes "
            "Cryptogr. 62 (2012): exactly one class at order 61, reported in the introduction",
        )
        continue
    add(
        "williamson",
        n,
        "empty" if n in (35, 47, 53, 59) else "exists",
        "https://cs.uwaterloo.ca/~cbright/reports/jsc-willsat.pdf",
        "Williamson classification and Turyn construction",
    )


def prime_power(q: int) -> bool:
    for p in range(2, isqrt(q) + 1):
        if q % p == 0:
            while q % p == 0:
                q //= p
            return q == 1
    return q >= 2


for n in range(65, 300, 2):
    if prime_power(2 * n - 1):
        add(
            "williamson",
            n,
            "exists",
            "https://doi.org/10.1016/0097-3165(72)90095-7",
            "Turyn, An infinite class of Williamson matrices, J. Combin. Theory Ser. A 12 (1972): "
            "Williamson matrices of order n whenever 2n-1 is a prime power congruent to 1 mod 4",
        )
for n in range(1, 50, 2):
    add(
        "good",
        n,
        "empty" if n in (41, 47, 49) else "exists",
        "https://arxiv.org/abs/1802.00556",
        "Goethals–Seidel difference families, Table 1",
    )
for n in (51, 57, 63, 69):
    add(
        "good",
        n,
        "exists" if n == 57 else "empty",
        "https://arxiv.org/abs/1811.05094",
        "Good matrices SAT+CAS, Table 1",
    )
add(
    "good",
    127,
    "exists",
    "https://arxiv.org/html/1802.00556",
    "Known cyclic good-matrix orders, Section 5",
)
for n in (47, 61):
    add(
        "skew-sds",
        n,
        "exists",
        "https://arxiv.org/html/0903.5053v2",
        "Cyclic skew SDS constructions, Section 5",
    )
for n in (1, 3, 7, 13, 21, 31, 43, 57):
    add("best", n, "exists", "https://arxiv.org/abs/1907.04987", "Best classification, Section 6")
# kkss is the classical G-matrix stratum. Only individually sourced existence
# records are imported here; no assumption is made about blank table cells.
for n in (1, 3, 5, 7, 9, 13, 15, 19, 21, 23, 25, 27, 31, 33, 37, 41, 43, 45):
    add("kkss", n, "exists", "https://arxiv.org/abs/1802.00556", "G-matrix classification, Table 1")
for n in range(1, 300, 2):
    target = 2 * n - 1
    if not any(isqrt(target - a * a) ** 2 == target - a * a for a in range(isqrt(target) + 1)):
        add(
            "kkss",
            n,
            "empty",
            "https://arxiv.org/html/1802.00556",
            "G-matrix nonexistence: Proposition 3 and Table 1",
        )
# Spence (1977) gives circulant G-matrices of order (q+1)/2 for every prime
# power q = 5 (mod 8); orders through 45 are already covered by the
# classification above.
for n in range(47, 300, 2):
    if (2 * n - 1) % 8 == 5 and prime_power(2 * n - 1):
        add(
            "kkss",
            n,
            "exists",
            "https://www.sciencedirect.com/science/article/pii/0012365X77900097",
            "Spence, Skew-Hadamard matrices of order 2(q+1): G-matrices of order (q+1)/2, "
            "q a prime power, q = 5 (mod 8)",
        )
# The published exhaustive search at v=49 reports no G-matrices. Exactly
# verified rows on the board contradict it, so the claim is kept as a
# contradicted record rather than dropped or silently treated as empty.
add(
    "kkss",
    49,
    "contradicted",
    "https://arxiv.org/html/1802.00556",
    "G-matrix nonexistence claim: Table 1, parameter set (49; 24,24,22,18; 39)",
)
add(
    "skew-sds",
    73,
    "exists",
    "https://arxiv.org/html/2411.18897v2",
    "Cati–Pasechnik, Section 5.1 (Đoković construction)",
)
# Williamson quadruples at even n, read on 2026-11-25. The first source enumerates every even order up to 70 and
# gives the number of classes in its Table 1. Its equivalence (Section 2.2, operations E1 to E5) is coarser than
# equiv-v1: besides block order, negation of a block and a multiplier applied to all four blocks, it identifies
# the cyclic shift of a single block by n/2 (E3) and the negation of every second entry of all four blocks (E5).
# The count is recorded as published, with that equivalence named; it is not an equiv-v1 count.
EVEN_WILLIAMSON = {
    2: 1, 4: 1, 6: 1, 8: 1, 10: 2, 12: 3, 14: 5, 16: 6, 18: 23, 20: 17, 22: 15, 24: 72, 26: 26, 28: 83,
    30: 150, 32: 152, 34: 91, 36: 477, 38: 50, 40: 1499, 42: 301, 44: 249, 46: 50, 48: 9800, 50: 275,
    52: 926, 54: 498, 56: 40315, 58: 73, 60: 4083, 62: 61, 64: 69960, 66: 262, 68: 1113, 70: 98,
}  # fmt: skip
EVEN_WILLIAMSON_EQUIVALENCE = (
    "the operations E1 to E5 of [@bright2020williamson]: reordering the blocks, negating a block, shifting a "
    "block cyclically by n/2, applying a multiplier to all four blocks, and negating every second entry of all "
    "four blocks"
)
for n, classes in EVEN_WILLIAMSON.items():
    add(
        "williamson",
        n,
        "exists",
        "https://doi.org/10.1016/j.jsc.2019.07.024",
        "Bright, Kotsireas, Ganesh, Applying computer algebra systems with SAT solvers to the Williamson "
        f"conjecture, J. Symbolic Comput. 100 (2020), Table 1: {classes} {'class' if classes == 1 else 'classes'} "
        f"at order {n} in the exhaustive enumeration of the even orders up to 70, under the equivalence of its "
        "Section 2.2",
        date="2026-11-25",
        published_classes={"count": classes, "equivalence": EVEN_WILLIAMSON_EQUIVALENCE},
    )
# Above 70 no enumeration is recorded. The second source proves that Williamson sequences of length 2^t m exist
# for every t whenever they exist at odd length m, so an even n from 72 to 300 has a record when this baseline has
# an existence record at its odd part, and no record otherwise.
odd_williamson = {r["n"] for r in records if r["stratum"] == "williamson" and r["status"] == "exists" and r["n"] % 2}
for n in range(72, 301, 2):
    m = n
    while m % 2 == 0:
        m //= 2
    if m in odd_williamson:
        add(
            "williamson",
            n,
            "exists",
            "https://doi.org/10.1109/TIT.2020.3016510",
            "Bright, Kotsireas, Ganesh, New infinite families of perfect quaternion sequences and Williamson "
            "sequences, IEEE Trans. Inform. Theory 66 (2020): Williamson sequences of length 2^t m exist for every "
            f"t when they exist at odd length m; here {n} = {n // m} x {m}",
            date="2026-11-25",
        )
write_json(
    ROOT / "baselines/order-status.json",
    {
        "id": "gs-literature-2026-10-16",
        "records": records,
        "notes": [
            "Good n=53, 55, and 61 and G-matrices at n=49 and 59 have no published existence source here: "
            "their existence on the board comes from imported, exactly verified vxvware/saw-best-matrix rows. "
            "G-matrices at 51, 55, and 63 exist by Spence's infinite family; the imported censuses add the class "
            "counts.",
            "A contradicted record is a published nonexistence claim that exactly verified rows on this board "
            "refute. It is neither an existence nor an emptiness source.",
            "Missing records mean unresolved within this sourced baseline, not a claim of exhaustive literature "
            "review.",
            "A Williamson record at even n up to 70 carries published_classes: the class count of the cited "
            "enumeration and the equivalence it was counted under, which is coarser than equiv-v1. No good, best "
            "or G-matrix record exists at even n, because a skew circulant block needs odd n.",
        ],
    },
)
