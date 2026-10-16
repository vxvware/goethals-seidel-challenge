#!/usr/bin/env python3
"""Reproducibly import the explicit plan inventory; never trust source verdicts."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import ast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from verify.common import ROOT, digest, write_json
from verify.equivalence import canonical, fingerprint
from verify.gs_verify import validate_rows

AUTHOR = "trevor-vincent"
DATE = "2026-10-14"


def base(identifier, kind, n, sources, discoverers):
    return {
        "schema_version": 1,
        "id": identifier,
        "type": kind,
        "n": n,
        "group": {"kind": "cyclic", "invariant_factors": [n]},
        "submitted_by": AUTHOR,
        "submitted_at": DATE,
        "discovered_by": discoverers,
        "sources": sources,
    }


def copy_artifact(src, dest):
    dest = ROOT / "evidence/seeds" / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return {"path": dest.relative_to(ROOT).as_posix(), "sha256": digest(dest)}


def source(repo, path, title=None):
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "title": title or f"{repo.name}: {path}",
        "url": f"https://github.com/vxvware/{repo.name}/blob/{commit}/{path}",
        "commit": commit,
        "source_path": path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent)
    parser.add_argument("--sage-file", type=Path, required=True)
    parser.add_argument("--sage-source", type=Path, required=True)
    args = parser.parse_args()
    pool = {}
    inventory = []
    counts = {}
    (ROOT / "evidence/seeds").mkdir(parents=True, exist_ok=True)

    def ingest(label, rows, src, discoverers):
        rows = list(rows)
        archive = ROOT / f"evidence/seeds/{label}.jsonl.gz"
        ids = set()
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                for item in rows:
                    q = tuple(tuple(int(v) for v in r) for r in item)
                    # Conversion here is from trusted numeric archives. Every
                    # public JSON submission rejects non-integer inputs.
                    validate_rows(q)
                    rep = canonical(q)
                    fp = fingerprint(rep)
                    n = len(rep[0])
                    ident = f"n{n}-{fp[3:23]}"
                    if ident in pool and pool[ident]["rows"] != [list(r) for r in rep]:
                        raise ValueError("fingerprint prefix collision")
                    if ident not in pool:
                        pool[ident] = {
                            **base(ident, "quadruple", n, [], discoverers),
                            "equivalence": "equiv-v1",
                            "rows": [list(r) for r in rep],
                        }
                    if src not in pool[ident]["sources"]:
                        pool[ident]["sources"].append(src)
                    pool[ident]["discovered_by"] = sorted(
                        set(pool[ident]["discovered_by"] + discoverers)
                    )
                    ids.add(ident)
                    stream.write(
                        (
                            json.dumps(
                                {"n": n, "rows": q, "representative": ident}, separators=(",", ":")
                            )
                            + "\n"
                        ).encode()
                    )
        inventory.append(
            {
                "id": label,
                "source": src,
                "rows": len(rows),
                "representatives": sorted(ids),
                "archive": {
                    "path": archive.relative_to(ROOT).as_posix(),
                    "sha256": digest(archive),
                },
            }
        )
        print(f"{label}: {len(rows)} verified rows -> {len(ids)} classes", flush=True)
        return ids

    best = args.source_root / "best-matrix-campaign"
    good = args.source_root / "good-matrix-campaign"
    skew = args.source_root / "skew356-campaign"
    with np.load(best / "data/catalogue_solutions.npz", allow_pickle=False) as z:
        for key in z.files:
            n = int(key[1:])
            src = source(best, "data/catalogue_solutions.npz", f"Best catalogue n={n}")
            ids = ingest(
                f"best-n{n}", z[key], src, ["Best-matrix catalogue; see published sources"]
            )
            counts[(n, "best")] = (ids, len(z[key]))
    for n in (3, 5, 7, 9, 11, 13, 15):
        path = f"benchmark/good_matrices_n{n}.npz"
        with np.load(good / path, allow_pickle=False) as z:
            ids = ingest(
                f"good-n{n}", z["sols"], source(good, path), ["Good-matrix campaign catalogue"]
            )
            counts[(n, "good")] = (ids, len(z["sols"]))
    for n in (15, 21):
        path = f"benchmark/good_solution_n{n}.npy"
        ingest(
            f"russo-n{n}",
            [np.load(good / path, allow_pickle=False)],
            source(good, path),
            ["vxvware (campaign attribution)"],
        )
    for label, path in (
        ("good-all-cpu-n13", "benchmark/good_all_cpu_n13.npy"),
        ("good-all-gpu-n13", "data/box_192.222.50.112/good_all_n13.npy"),
    ):
        ingest(
            label,
            np.load(good / path, allow_pickle=False),
            source(good, path),
            ["Good-matrix campaign CPU/GPU cross-check"],
        )
    path = "data/sweep_p73_d9.json"
    families = json.loads((skew / path).read_text())
    rows = (
        [[-1 if j in block else 1 for j in range(73)] for block in fam["blocks"]]
        for fam in families
    )
    ingest(
        "skew-p73-d9", rows, source(skew, path), ["Skew356 campaign (independent reconstruction)"]
    )

    # Extract only literal numerical data from Sage, never execute fetched code.
    if not args.sage_file.exists() or not args.sage_source.exists():
        raise ValueError("pinned Sage source required for literature seeds")
    sha, url = args.sage_source.read_text().splitlines()[:2]
    tree = ast.parse(args.sage_file.read_text())
    fn = next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef) and x.name == "williamson_type_quadruples_smallcases"
    )
    dictionary = None
    for x in ast.walk(fn):
        if isinstance(x, ast.Dict):
            try:
                value = ast.literal_eval(x)
                if value and all(isinstance(k, int) for k in value):
                    dictionary = value
                    break
            except (ValueError, TypeError):
                pass
    if dictionary is None:
        raise ValueError("Sage literal table layout changed")
    for n, words in dictionary.items():
        if n % 2 == 0:
            continue
        q = [[1 if c == "+" else -1 for c in word] for word in words]
        src = {
            "title": f"Sage Williamson table n={n} (Hall/London sources)",
            "url": url,
            "commit": sha,
            "source_path": "src/sage/combinat/matrices/hadamard_matrix.py",
        }
        ingest(f"williamson-n{n}", [q], src, ["Hall / London; Sage table references"])

    fn = next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name == "skew_hadamard_matrix_from_good_matrices_smallcases"
    )
    e_table = ast.literal_eval(
        next(
            x.value
            for x in fn.body
            if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "E_sequences" for t in x.targets)
        )
    )
    for h, words in e_table.items():
        n = 2 * h + 1
        q = []
        for i, word in enumerate(words):
            half = [1 if c == "+" else -1 for c in word]
            q.append([1] + half + [(-x if i == 0 else x) for x in reversed(half)])
        src = {
            "title": f"Sage good-matrix table n={n} (Szekeres)",
            "url": url,
            "commit": sha,
            "source_path": "src/sage/combinat/matrices/hadamard_matrix.py",
        }
        ingest(f"sage-good-n{n}", [q], src, ["Szekeres; Sage table references"])
    H = {1, 2, 4, 8, 16, 32, 37, 55, 64}
    cosets = [{a * h % 73 for h in H} for a in (1, -1, 3, -3, 5, -5, 11, -11)]
    blocks = [
        set().union(*(cosets[j] for j in indices))
        for indices in ((1, 3, 4, 6), (3, 5, 6, 7), (1, 4, 5, 7), (0, 2, 7))
    ]
    blocks[3].add(0)
    q = [[-1 if j in b else 1 for j in range(73)] for b in blocks]
    ingest(
        "published-skew-n73",
        [q],
        {
            "title": "Cati–Pasechnik §5.1: Đoković's order-292 construction",
            "url": "https://arxiv.org/html/2411.18897v2",
        },
        ["Dragomir Ž. Đoković (Cati–Pasechnik §5.1)"],
    )

    for ident, doc in sorted(pool.items()):
        write_json(ROOT / f"quadruples/{ident}.json", doc)
    for (n, track), (ids, raw) in counts.items():
        srcs = [
            {
                "title": "Published best classification"
                if track == "best"
                else "Published good classification",
                "url": "https://arxiv.org/abs/1907.04987"
                if track == "best"
                else "https://arxiv.org/abs/1811.05094",
            }
        ]
        doc = {
            **base(
                f"{track}-n{n}",
                "census",
                n,
                srcs,
                ["Campaign catalogue and published classification"],
            ),
            "stratum": track,
            "equivalence": "equiv-v1",
            "representatives": sorted(ids),
            "claimed_count": len(ids),
            "claim_kind": "published-exact",
            "secondary": {"stored_rows": raw},
        }
        if track == "best" and n <= 43:
            doc["enumerator"] = "best-census-c-v1"
            doc["claim_kind"] = "computed-exact"
        elif track == "good" and n <= 15:
            doc["enumerator"] = "small-exact-v1"
            doc["claim_kind"] = "computed-exact"
        write_json(ROOT / f"census/{doc['id']}.json", doc)

    # Archive the complete best receipt chain and the omission closure.
    records = []
    for p in sorted((best / "data/campaign_state").glob("*.json")):
        records.append(copy_artifact(p, "best73/" + p.name))
    records.append(
        copy_artifact(best / "data/enum_audit_2026-09-21/omit_ckpt.json", "best73/omit_ckpt.json")
    )
    records.append(copy_artifact(best / "docs/enum_audit_2026-09-21.md", "best73/enum-audit.md"))
    engine_ref = copy_artifact(best / "benchmark/verify_coverage.py", "best73/verify_coverage.py")
    src = source(best, "docs/enum_audit_2026-09-21.md")
    doc = {
        **base("best-n73-campaign", "certificate", 73, [src], ["Best-matrix campaign"]),
        "stratum": "best",
        "claim": "empty",
        "scope": {"kind": "full"},
        "method": "best-coverage-v1",
        "records": records,
        "engine": {
            "id": "best-coverage-v1",
            "snapshot_commit": src["commit"],
            "binding": "archived-with-receipt",
            "artifact": engine_ref,
        },
        "gaps": [
            "104500 legacy classes lack modern identity/digest; retained as an archived claim."
        ],
    }
    write_json(ROOT / f"certificates/{doc['id']}.json", doc)
    for n, paths, claim in [
        (41, ["data/box_192.222.50.112/coverage_n41_sB9.json"], "empty"),
        (
            65,
            [
                "data/box_192.222.58.99/progress_complete_n65_sB-15.json",
                "data/box_192.222.50.112/coverage_n65_sB-15.json",
                "data/box_192.222.50.112/coverage_n65_sB13.json",
            ],
            "empty",
        ),
        (67, ["data/box_192.222.50.112/progress_complete_n67_sB-13.json"], "partial"),
    ]:
        refs = [
            copy_artifact(good / p, f"good{n}/{Path(p).parent.name}-{Path(p).name}") for p in paths
        ]
        engine_path = "data/box_192.222.50.112/good_gpu_complete.py"
        eng = copy_artifact(good / engine_path, f"good{n}/good_gpu_complete.py")
        src = source(good, paths[0])
        doc = {
            **base(f"good-n{n}-campaign", "certificate", n, [src], ["Good-matrix campaign"]),
            "stratum": "good",
            "claim": claim,
            "scope": {"kind": "full"},
            "method": "archived-good-coverage-v1" if n == 65 else "good-coverage-v1",
            "records": refs,
            "engine": {
                "id": "good-gpu-v4",
                "snapshot_commit": src["commit"],
                "binding": "archived-with-receipt",
                "artifact": eng,
            },
            "gaps": [
                "Execution commit not present in legacy receipt; pinned archive snapshot only.",
                "GPU replay has not run on this CPU-only importer.",
            ],
        }
        if n == 65:
            for filename in ("complete_n65.log", "complete_n65_v3.log", "complete_n65_v4.log"):
                path = f"data/box_192.222.58.99/{filename}"
                doc["sources"].append(
                    {
                        **source(good, path),
                        "artifact": copy_artifact(good / path, f"good65/{filename}.txt"),
                    }
                )
            path = "docs/n65_nonexistence.md"
            doc["sources"].append(
                {
                    **source(good, path),
                    "artifact": copy_artifact(good / path, "good65/campaign-writeup.md"),
                }
            )
            doc["gaps"].append(
                "Modern JSON receipts leave [0,454800) uncovered in pass -15; older logs do not repair its missing execution identity."
            )
        write_json(ROOT / f"certificates/{doc['id']}.json", doc)
    src = source(skew, "benchmark/cyclotomic_sweep.py")
    ref = copy_artifact(skew / "data/sweep_p89.json", "skew89/original-empty-array.json")
    doc = {
        **base(
            "skew-n89-subgroup11",
            "certificate",
            89,
            [src],
            ["Skew356 campaign; exact row-sum obstruction"],
        ),
        "stratum": "skew-sds",
        "claim": "empty",
        "scope": {
            "kind": "same-subgroup-coset-unions",
            "subgroup_order": 11,
            "include_zero": True,
            "skew_blocks": 1,
        },
        "method": "coset-row-sum-theorem",
        "records": [ref],
        "gaps": [],
        "notes": "Original [] retained as provenance. The verifier independently proves the precisely scoped null by integer row sums.",
    }
    write_json(ROOT / f"certificates/{doc['id']}.json", doc)
    for track, ns, url in [
        ("good", [41, 47, 49, 51, 59, 63, 69], "https://arxiv.org/abs/1811.05094"),
        (
            "williamson",
            [35, 47, 53, 59],
            "https://cs.uwaterloo.ca/~cbright/reports/jsc-willsat.pdf",
        ),
    ]:
        for n in ns:
            if track == "good" and n in (41, 47, 49):
                source_url = "https://arxiv.org/abs/1802.00556"
            else:
                source_url = url
            doc = {
                **base(
                    f"{track}-n{n}-literature",
                    "certificate",
                    n,
                    [{"title": f"Published nonexistence: {track} n={n}", "url": source_url}],
                    ["Authors of cited classification"],
                ),
                "stratum": track,
                "claim": "empty",
                "scope": {"kind": "full"},
                "method": "literature",
                "records": [],
                "gaps": [],
            }
            write_json(ROOT / f"certificates/{doc['id']}.json", doc)
    from math import isqrt

    for n in range(1, 132, 2):
        if isqrt(4 * n - 3) ** 2 == 4 * n - 3:
            continue
        doc = {
            **base(
                f"best-n{n}-row-sums",
                "certificate",
                n,
                [
                    {
                        "title": "Best row-sum necessary condition",
                        "url": "https://arxiv.org/abs/1907.04987",
                    }
                ],
                ["Best-matrix row-sum theorem"],
            ),
            "stratum": "best",
            "claim": "empty",
            "scope": {"kind": "full"},
            "method": "row-sum-theorem",
            "records": [],
            "gaps": [],
        }
        write_json(ROOT / f"certificates/{doc['id']}.json", doc)
    write_json(
        ROOT / "baselines/seed-manifest.json",
        {
            "version": 1,
            "artifacts": inventory,
            "raw_rows": sum(x["rows"] for x in inventory),
            "representative_count": len(pool),
        },
    )
    print(f"TOTAL {len(pool)} representatives; {sum(x['rows'] for x in inventory)} raw rows")


if __name__ == "__main__":
    main()
