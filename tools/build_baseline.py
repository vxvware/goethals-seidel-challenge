"""Materialize individual, sourced status records; never infer absence from a missing import."""

from pathlib import Path
from math import isqrt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, write_json

records = []


def add(track, n, status, url, title):
    records.append(
        {
            "n": n,
            "stratum": track,
            "group": "cyclic",
            "status": status,
            "source": {"title": title, "url": url},
            "baseline_date": "2026-10-14",
        }
    )


for n in range(1, 64, 2):
    add(
        "williamson",
        n,
        "empty" if n in (35, 47, 53, 59) else "exists",
        "https://cs.uwaterloo.ca/~cbright/reports/jsc-willsat.pdf",
        "Williamson classification and Turyn construction",
    )


def prime_power(q):
    for p in range(2, isqrt(q) + 1):
        if q % p == 0:
            while q % p == 0:
                q //= p
            return q == 1
    return q >= 2


for n in range(65, 132, 2):
    if prime_power(2 * n - 1):
        add(
            "williamson",
            n,
            "exists",
            "https://rikkyo.repo.nii.ac.jp/record/10210/files/AA00610867_31-01_10.pdf",
            "Yamada, Williamson matrices of Turyn's type: 2n-1 a prime power",
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
for n in range(1, 132, 2):
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
for n in range(47, 132, 2):
    if (2 * n - 1) % 8 == 5 and prime_power(2 * n - 1):
        add(
            "kkss",
            n,
            "exists",
            "https://www.sciencedirect.com/science/article/pii/0012365X77900097",
            "Spence, Skew-Hadamard matrices of order 2(q+1): G-matrices of order (q+1)/2, q a prime power, q = 5 (mod 8)",
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
write_json(
    ROOT / "baselines/order-status.json",
    {
        "id": "gs-literature-2026-10-16",
        "records": records,
        "notes": [
            "Good n=53, 55, and 61 and G-matrices at n=49 and 59 have no published existence source here: their existence on the board comes from imported, exactly verified vxvware/saw-best-matrix rows. G-matrices at 51, 55, and 63 exist by Spence's infinite family; the imported censuses add the class counts.",
            "A contradicted record is a published nonexistence claim that exactly verified rows on this board refute. It is neither an existence nor an emptiness source.",
            "Missing records mean unresolved within this sourced baseline, not a claim of exhaustive literature review.",
        ],
    },
)
