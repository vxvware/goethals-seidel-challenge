#!/usr/bin/env python3
"""Import the pinned vxvware/saw-best-matrix census files, without executing source code.

The source files are sign rows under data/good and data/gmat in that
repository. Each is pinned to the commit that last changed it and to its
SHA-256. Orders with rows become quadruple entries plus a lower-bound census;
orders whose census is empty become unreplayed-census certificates. Orders the
source leaves empty for arithmetic reasons are not imported: the sourced
baseline already records them.
"""

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, digest, write_json
from verify.equivalence import canonical, fingerprint
from verify.gs_verify import memberships, validate_rows
from verify.model import validate

REPO = "https://github.com/vxvware/saw-best-matrix"
SUBMITTER = "vxvware"
SUBMITTED_AT = "2026-10-15"
DISCOVERED_BY = ["vxvware"]
EVIDENCE = "evidence/participants/saw-best-matrix"
DK2018 = "https://arxiv.org/abs/1802.00556"

# (family, n): commit, sha256, classes reported by the source, note on prior work.
SOURCES = {
    ("good", 53): (
        "020601f6e15db4e2ca023b602bc460204ce40236",
        "79deb6ae7f913732e86d74786d1e453b522ae1c999083bc3eb20fcf3a8be7f60",
        1,
        "No published search covers this order.",
    ),
    ("good", 55): (
        "020601f6e15db4e2ca023b602bc460204ce40236",
        "b7b7d33e460c7eea6e5d2d1ffdf1cec7bfbff7dd948be40b8cc73cf3e0c1dde7",
        1,
        "No published search covers this order.",
    ),
    ("good", 59): (
        "33477db3deee7d5e4b965af85943ee764289f82b",
        "156bedad9b02ec2896722e0fc6c3c6128c6a402792c700c688030c3511d526be",
        0,
        "No published search covers this order. The source ran the census twice "
        "to completion with identical triple counts.",
    ),
    ("good", 61): (
        "2c46a00ee0b450957ed7ac2281cf22bec0edd344",
        "4818e93d5dab46a6a8b5e085672fbbd96e1803bb73dbf5e47087f7c947f37aa1",
        1,
        "No published search covers this order.",
    ),
    ("kkss", 3): (
        "43aec6a329165d179eca48a900f0ac6d5f6a8e53",
        "b8ec4807e9123af2f79d6388cc431df85d59cba045bee74ee09f61834badb7b2",
        1,
        "",
    ),
    ("kkss", 5): (
        "43aec6a329165d179eca48a900f0ac6d5f6a8e53",
        "f78f92ee0b98118f14cee8394b71e40e9eefb12c9b641ea2ab6f3c095d57f5eb",
        1,
        "",
    ),
    ("kkss", 7): (
        "43aec6a329165d179eca48a900f0ac6d5f6a8e53",
        "1244529ae746c132106b545d2d2cea818af9dcb42a1a2eacee4e9b23db1508b9",
        1,
        "",
    ),
    ("kkss", 9): (
        "43aec6a329165d179eca48a900f0ac6d5f6a8e53",
        "3926c40d60e7d78f71b940b2b302e82bd72e898782ac8774f0031431b44eedd8",
        1,
        "",
    ),
    ("kkss", 13): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "f37aac040039bb79740680f2d14888eafaf8eab4ffde4d1cca0d057a3247e84f",
        6,
        "",
    ),
    ("kkss", 15): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "2efca1e13e50626ec57e87b89f61228a6842e319128839d9f0a18f6a1e2e3b37",
        18,
        "",
    ),
    ("kkss", 19): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "7471c7adc8207471a0779c3d754239d11dbc8bc2a2ed742b787eaab1b0b4b965",
        5,
        "",
    ),
    ("kkss", 21): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "0fbfcb8fe617fa131356df27456cd48cd67ba7f167ebf30c0496be07656176f4",
        12,
        "",
    ),
    ("kkss", 23): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "aaf318491affc86c03ef3367a43bcd13eb41073e78e5214d779b277c80683641",
        8,
        "",
    ),
    ("kkss", 25): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "ce77a963017b1669e2ed949a45bea763bc462b518545c166980f9d62986b6212",
        7,
        "",
    ),
    ("kkss", 27): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "96f70ab1f5465b5e0e2ef5571937b0a0980c3f7c1380ba7905ad001630161cb9",
        10,
        "",
    ),
    ("kkss", 31): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "f8e07fc150d644734942343f64dafd07459f7bb62b231bc2c5fc4ea11dd1e99d",
        4,
        "",
    ),
    ("kkss", 33): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "89adc598eed1bfaa0f96ca23de1188b3e12d80ffc25bf3e6cd49184ae27dcccb",
        20,
        "Agrees with the 20 classes of arXiv:1802.00556.",
    ),
    ("kkss", 37): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "38f13734a070c9040ce1d99d08d50c814feed8b1d33d204d922cabb5091b94e6",
        7,
        "Agrees with the 7 classes of arXiv:1802.00556.",
    ),
    ("kkss", 41): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "cd5d0bd27d0bbde6e9e80a43c516038ce47bef7ab5e52a4f0fedab3b83f991b2",
        3,
        "Agrees with the 3 classes of arXiv:1802.00556.",
    ),
    ("kkss", 43): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "fc2873dba77a043124ed92c618172b1569587b566d41ba4125953908bf9a7ad2",
        4,
        "Agrees with the 4 classes of arXiv:1802.00556.",
    ),
    ("kkss", 45): (
        "3b25375351a3eb9041a49b920fff40c6e1d891d3",
        "ba978df01b63dab726581dbaefd94a43907b020fbf8ed13edadd814dde0f1a30",
        1,
        "Agrees with the single class of arXiv:1802.00556.",
    ),
    ("kkss", 49): (
        "8aaf08f4352a0b0787e7ff6c670a9c326de2f23f",
        "e96ad52183f15d10b21da0ffcd99b5aa72456ddaed5eda7fa6eadeac9699570f",
        1,
        "Contradicts arXiv:1802.00556, whose exhaustive search reports no G-matrices at "
        "v=49 for the parameter set (49; 24,24,22,18; 39). The rows here have exactly that "
        "parameter set and verify exactly on this board.",
    ),
    ("kkss", 51): (
        "5348f02f19b9824aabbb77ed26febad092a54b92",
        "6d8f89d7d1291d7ae8eae4186695845cdf7208cdecb609eb3455eb53d9f6af36",
        4,
        "First order past the range classified in arXiv:1802.00556.",
    ),
    ("kkss", 55): (
        "a38ad7b0dc6b6108273c3a549181edab6b7f8a6e",
        "4b5bc4bca201e70a5c2fff071c36c3057af98b7463113f6da940b0cae665b60d",
        3,
        "One example was known from Spence (1977); the complete count is new.",
    ),
    ("kkss", 57): (
        "298ff0aff7c09a7f4de4adc5f5456490522affd3",
        "e91ea95154b2d135a35f38cbca46d5e77f657f77864afda12477560f37954b97",
        0,
        "The only admissible parameter set is (57; 28,28,28,21; 48); no published search "
        "covers it.",
    ),
    ("kkss", 59): (
        "aaa2c7952b52762b8811e3772ba8760b9eecdbac",
        "a1f351b3220afb4cb741fcb9142f9e79c03ba8fd4f2dfcd5464bb01a9a9846b5",
        2,
        "A single run, calibrated beforehand on the stored v=43 and v=45 censuses. The "
        "source counts two classes under its own convention; under equiv-v1 the eight "
        "rows fall into one class.",
    ),
    ("kkss", 63): (
        "a6cef295734124cbb8ed07fb935209a8ecb51a2f",
        "c50fed164652e958de9b1874792dd382c47a7034e42176fb76c37bcae4d7b111",
        1,
        "Not a census: an orbit-restricted search with the skew rows constant on the orbits "
        "of the subgroup generated by 4 in the unit group. It reproduces the known Spence "
        "example at parameters (63; 31,31,27,25; 51).",
    ),
}
FULL_CENSUS = {key for key in SOURCES if key != ("kkss", 63)}
FILES = {"good": "good", "kkss": "gmat"}


def source_record(family, n, raw):
    commit, _, _, _ = SOURCES[family, n]
    path = f"data/{FILES[family]}/{FILES[family]}-{n:02d}.txt"
    label = "good matrices" if family == "good" else "G-matrices (kkss)"
    return {
        "title": f"vxvware/saw-best-matrix: {label} at n={n}",
        "url": f"{REPO}/blob/{commit}/{path}",
        "commit": commit,
        "source_path": path,
        "artifact": {"path": raw.relative_to(ROOT).as_posix(), "sha256": digest(raw)},
    }


def read_rows(original, family, n):
    rows = []
    for line in original.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        words = line.split()
        if any(set(word) - set("+-") for word in words):
            raise ValueError("source rows must contain signs only")
        q = validate_rows([[1 if ch == "+" else -1 for ch in word] for word in words], n)
        if family not in memberships(q):
            raise ValueError(f"source row is not in the {family} stratum")
        rows.append(q)
    return rows


def store(path, doc, rows_may_match=False):
    if path.exists():
        previous = json.loads(path.read_text())
        same_rows = rows_may_match and previous.get("rows") == doc.get("rows")
        if previous != doc and not same_rows:
            raise ValueError(f"refusing to overwrite a different record: {path.name}")
    write_json(path, doc)


def census_notes(family, n, rows, ids):
    _, _, reported, note = SOURCES[family, n]
    kind = "full census" if (family, n) in FULL_CENSUS else "orbit-restricted search"
    text = (
        f"The source reports a {kind} with {reported} class"
        f"{'' if reported == 1 else 'es'} under its own equivalence convention and stores "
        f"{len(rows)} quadruples. This board verifies the {len(ids)} equiv-v1 representative"
        f"{'' if len(ids) == 1 else 's'} and records a lower bound; the search has not been "
        "replayed here."
    )
    return f"{text} {note}".strip()


def import_order(family, n, source_dir, folder):
    original = source_dir / FILES[family] / f"{FILES[family]}-{n:02d}.txt"
    if digest(original) != SOURCES[family, n][1]:
        raise ValueError(f"source differs from the pinned saw-best-matrix file: {original.name}")
    raw = folder / FILES[family] / original.name
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists() and raw.read_bytes() != original.read_bytes():
        raise ValueError("refusing to overwrite a different source archive")
    raw.write_bytes(original.read_bytes())
    source = source_record(family, n, raw)
    common = {
        "schema_version": 1,
        "n": n,
        "group": {"kind": "cyclic", "invariant_factors": [n]},
        "submitted_by": SUBMITTER,
        "submitted_at": SUBMITTED_AT,
        "discovered_by": DISCOVERED_BY,
        "sources": [source],
    }
    rows = read_rows(original, family, n)
    ident = f"vxvware-{family}-n{n}"
    if not rows:
        if (family, n) not in FULL_CENSUS or SOURCES[family, n][2] != 0:
            raise ValueError(f"empty source without an empty census claim: {original.name}")
        certificate = validate(
            {
                **common,
                "id": f"{ident}-census",
                "type": "certificate",
                "stratum": family,
                "claim": "empty",
                "scope": {"kind": "full"},
                "method": "unreplayed-census",
                "records": [],
                "gaps": ["Unpublished; no paper is recorded for this census."],
                "notes": (
                    "The archived source file is a deliberately empty census output whose "
                    "header records the exhaustive search. " + SOURCES[family, n][3]
                ).strip(),
            }
        )
        store(ROOT / f"certificates/{certificate['id']}.json", certificate)
        print(f"{family} n={n}: empty census -> {certificate['id']}")
        return None
    mappings = []
    ids = set()
    for q in rows:
        rep = canonical(q)
        quad_id = f"n{n}-{fingerprint(rep)[3:23]}"
        doc = validate(
            {
                **common,
                "equivalence": "equiv-v1",
                "id": quad_id,
                "type": "quadruple",
                "rows": [list(r) for r in rep],
            }
        )
        store(ROOT / f"quadruples/{quad_id}.json", doc, rows_may_match=True)
        ids.add(quad_id)
        mappings.append({"n": n, "rows": q, "representative": quad_id})
    archive = folder / FILES[family] / f"{FILES[family]}-{n:02d}.jsonl.gz"
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in mappings).encode()
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as stream:
        stream.write(payload)
    if archive.exists() and archive.read_bytes() != compressed.getvalue():
        raise ValueError("refusing to overwrite a different raw-row archive")
    archive.write_bytes(compressed.getvalue())
    census = validate(
        {
            **common,
            "equivalence": "equiv-v1",
            "id": ident,
            "type": "census",
            "stratum": family,
            "representatives": sorted(ids),
            "claimed_count": len(ids),
            "claim_kind": "lower-bound",
            "secondary": {"stored_rows": len(rows)},
            "notes": census_notes(family, n, rows, ids),
        }
    )
    store(ROOT / f"census/{ident}.json", census)
    print(f"{family} n={n}: {len(rows)} verified quadruples -> {len(ids)} classes")
    return {
        "id": ident,
        "source": source,
        "rows": len(rows),
        "representatives": sorted(ids),
        "archive": {"path": archive.relative_to(ROOT).as_posix(), "sha256": digest(archive)},
    }


def import_all(source_dir):
    folder = ROOT / EVIDENCE
    records = []
    all_ids = set()
    for family, n in sorted(SOURCES, key=lambda key: (key[0] != "good", key[1])):
        record = import_order(family, n, source_dir, folder)
        if record:
            records.append(record)
            all_ids.update(record["representatives"])
    write_json(
        ROOT / "baselines/participant-seed-manifest.json",
        {
            "version": 1,
            "artifacts": records,
            "raw_rows": sum(r["rows"] for r in records),
            "representative_count": len(all_ids),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="directory holding good/ and gmat/ sign-row files (a saw-best-matrix data/ "
        "checkout or this repository's bundled evidence folder)",
    )
    import_all(p.parse_args().source_dir)
