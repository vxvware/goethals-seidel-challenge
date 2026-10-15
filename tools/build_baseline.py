"""Materialize individual, sourced status records; never infer absence from a missing import."""

from pathlib import Path
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
        "id": "gs-literature-2026-10-14",
        "records": records,
        "notes": [
            "Good n=53,55,61 are not marked known without a construction source.",
            "Missing records mean unresolved within this sourced baseline, not a claim of exhaustive literature review.",
        ],
    },
)
