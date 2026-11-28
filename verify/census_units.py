"""Check the per-class records of a census that a pinned engine runs as units of work.

best_hkt2.c (engine best-census-hkt-v2) splits its pairs (A, D) into 2^q
classes, cls(A) xor cls(D) (comment C6 of the source), and a run with
BEST_QB=q BEST_CLASS=b solves exactly the pairs of class b. A units manifest
records one such run per class: the host, the wall time, the sha256 of its log
and the counters the log prints. The logs themselves are kept in one xz
archive beside the manifest, and every load reparses each log and requires its
class, its counters and its header (the D and A list sizes, the stage-1 corank,
the consistency-table layout) to equal the manifest.

Two independent records sit beside the logs. best_cover.c, which shares no
code with the engine, recounts the D and A rows by class label from their
definitions; its totals must equal the headers of the logs, and the number of
pairs it gives a class bounds the pairs that class's log reports past the
spectral index. best_plant.c plants solutions of the GF(2) systems in every
class and runs the engine's own solve on them; every plant must come back, and
the class forms it prints must equal the ones best_cover.c derives.

A manifest may be partial while the census runs. census_replays.apply_report
accepts a report that points at a manifest only when every class has a unit,
every class has passed the plant test, and the solutions of the units add up to
the report's raw rows.
"""

import hashlib
import io
import lzma
import math
import re
import tarfile
from pathlib import Path
from typing import TypedDict

from verify.common import ROOT, JSONObject, artifact, digest, read_json

UNIT_DIR = "evidence/census-units/"
ENGINES = {"best-census-hkt-v2": ("research/starter/best/best_hkt2.c",)}
INDEPENDENT = {
    "coverage": "research/starter/best/best_cover.c",
    "plant": "research/starter/best/best_plant.c",
}
COUNTERS = (
    "pairs_index",
    "pairs_fine",
    "patterns",
    "patterns_table",
    "systems_consistent",
    "candidates",
    "capped",
    "solutions",
)
MAX_ARCHIVE = 8 * 1024 * 1024
MAX_LOG = 256 * 1024

_UNIT = re.compile(r"^   UNIT of a census: class (\d+) of (\d+) \(0-based\)$", re.M)
_TABLE = re.compile(r"^   class (\d+) of (\d+): consistency table built in [0-9.]+s$", re.M)
_HEAD = re.compile(r"^(n=\d+: H=\d+, row sum of D \d+, stage-1 corank .*)$", re.M)
_DROWS = re.compile(
    r"^   D rows: (\d+) with the row sum, (\d+) under the ceiling, (\d+) kept as orbit minima "
    r"\((?:read from [^)]*|[0-9.]+s)\)$",
    re.M,
)
_DSIG = re.compile(r"^   D rows with the signature of a skew row \(R6\): (\d+) of (\d+)$", re.M)
_CORANK = re.compile(r"^   stage-2 corank over (\d+) random patterns: (.*)$", re.M)
_CC = re.compile(r"^   consistency table: (\d+) words per pattern, (\d+) classes of ([0-9.]+) MB$", re.M)
_PIECE = re.compile(r"^   A piece (\d+) of (\d+): (\d+) rows, (\d+) levels, [0-9.]+ MB, built in [0-9.]+s$", re.M)
_AROWS = re.compile(r"^   A rows under the ceiling: (\d+) of (\d+) \([0-9.]+\)$", re.M)
_PAIRS = re.compile(r"^   pairs \(A, D\) past the index: (\d+), past the 256-level test: (\d+) ", re.M)
_PATTERNS = re.compile(
    r"^   patterns (\d+), past the consistency table (\d+) \([0-9.]+%\), consistent systems (\d+) \([0-9.]+%\), "
    r"candidates verified (\d+)$",
    re.M,
)
_CAPPED = re.compile(r"^   nullity-capped systems: (\d+)$", re.M)
_WRITTEN = re.compile(r"^n=(\d+): (\d+) solutions written \(representatives, not classes\), ([0-9.]+)s$", re.M)


class ParsedLog(TypedDict):
    """What one unit's log states: its class, the header shared by all units, and its counters."""

    n: int
    unit: int
    classes: int
    header: JSONObject
    counters: dict[str, int]
    seconds: float


def _one(pattern: re.Pattern[str], text: str, what: str) -> re.Match[str]:
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise ValueError(f"census unit log: expected one {what} line, found {len(found)}")
    return found[0]


def parse_log(text: str) -> ParsedLog:
    """The class, header and counters of one best_hkt2 unit log; any missing or repeated line is an error."""
    unit = _one(_UNIT, text, "unit")
    b, classes = int(unit[1]), int(unit[2])
    table = _one(_TABLE, text, "consistency table")
    if int(table[1]) != b + 1 or int(table[2]) != classes:
        raise ValueError("census unit log: the consistency table is not the one of the unit's class")
    drows, dsig, arows = _one(_DROWS, text, "D rows"), _one(_DSIG, text, "R6"), _one(_AROWS, text, "A rows")
    cc, corank = _one(_CC, text, "consistency table layout"), _one(_CORANK, text, "stage-2 corank")
    pieces = [(int(m[1]), int(m[2]), int(m[3]), int(m[4])) for m in _PIECE.finditer(text)]
    if (
        not pieces
        or [p[0] for p in pieces] != list(range(1, len(pieces) + 1))
        or {p[1] for p in pieces} != {len(pieces)}
    ):
        raise ValueError("census unit log: the A pieces are not numbered 1..p")
    if sum(p[2] for p in pieces) != int(arows[1]):
        raise ValueError("census unit log: the A pieces do not add up to the A rows under the ceiling")
    piece_list = ",".join(f"{p[2]}:{p[3]}" for p in pieces)
    pairs, patterns = _one(_PAIRS, text, "pairs"), _one(_PATTERNS, text, "patterns")
    capped, written = _one(_CAPPED, text, "nullity cap"), _one(_WRITTEN, text, "solutions written")
    if int(cc[2]) != classes:
        raise ValueError("census unit log: the class count disagrees with the table layout")
    header: JSONObject = {
        "first_line": _one(_HEAD, text, "first").group(1),
        "d_row_sum": int(drows[1]),
        "d_ceiling": int(drows[2]),
        "d_minima": int(drows[3]),
        "d_signature": int(dsig[1]),
        "d_signature_of": int(dsig[2]),
        "stage2_corank_sample": f"{corank[1]}:{corank[2]}",
        "table_words": int(cc[1]),
        "classes": int(cc[2]),
        "table_mb": cc[3],
        "a_ceiling": int(arows[1]),
        "a_codes": int(arows[2]),
        "a_pieces": len(pieces),
        "a_pieces_sha256": hashlib.sha256(piece_list.encode()).hexdigest(),
    }
    if header["d_signature_of"] != header["d_minima"]:
        raise ValueError("census unit log: R6 was applied to a different D list")
    counters = {
        "pairs_index": int(pairs[1]),
        "pairs_fine": int(pairs[2]),
        "patterns": int(patterns[1]),
        "patterns_table": int(patterns[2]),
        "systems_consistent": int(patterns[3]),
        "candidates": int(patterns[4]),
        "capped": int(capped[1]),
        "solutions": int(written[2]),
    }
    return {
        "n": int(written[1]),
        "unit": b,
        "classes": classes,
        "header": header,
        "counters": counters,
        "seconds": float(written[3]),
    }


def read_archive(path: Path) -> dict[str, bytes]:
    """Members of a units log archive (regular files only, bounded sizes, unique names)."""
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("census unit log archive is too large")
    members: dict[str, bytes] = {}
    with lzma.open(path) as raw, tarfile.open(fileobj=io.BytesIO(raw.read(64 * MAX_ARCHIVE)), mode="r:") as tar:
        for info in tar.getmembers():
            if not info.isfile() or "/" in info.name.removeprefix("replays/") or info.size > MAX_LOG:
                raise ValueError(f"census unit log archive holds an unexpected member: {info.name}")
            if info.name in members:
                raise ValueError(f"census unit log archive repeats {info.name}")
            stream = tar.extractfile(info)
            if stream is None:
                raise ValueError(f"census unit log archive member is unreadable: {info.name}")
            members[info.name] = stream.read()
    return members


def parse_coverage(text: str) -> JSONObject:
    """The forms, totals and per-label counts printed by best_cover.c."""
    out: JSONObject = {"forms": [], "dlabel": {}, "alabel": {}, "anear": []}
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "n" and len(words) >= 6:
            out.update(n=int(words[1]), q=int(words[5]))
        elif words[0] == "form":
            out["forms"].append((words[3], words[5]))
        elif words[0] == "anear":
            out["anear"].append((int(words[1], 16), float(words[2])))
        elif words[0] in ("dlabel", "alabel"):
            out[words[0]][int(words[1])] = int(words[2])
        elif line.startswith(("D ", "A ")) and words[-1].isdigit():
            out[" ".join(words[:-1])] = int(words[-1])
    return out


def parse_plant(text: str) -> JSONObject:
    """The forms and per-class lines printed by best_plant.c (several runs may be concatenated)."""
    forms: set[tuple[tuple[str, str], ...]] = set()
    current: list[tuple[str, str]] = []
    classes: dict[int, dict[str, int]] = {}
    misses = 0
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "n":
            if current:
                forms.add(tuple(current))
            current = []
        elif words[0] == "form":
            current.append((words[3], words[5]))
        elif words[0] == "MISS":
            misses += 1
        elif words[0] == "class":
            b = int(words[1])
            if b in classes:
                raise ValueError(f"planted-solution record repeats class {b}")
            classes[b] = {words[i]: int(words[i + 1]) for i in range(2, 12, 2)}
    if current:
        forms.add(tuple(current))
    return {"forms": forms, "classes": classes, "misses": misses}


def check_manifest(manifest: JSONObject, root: Path = ROOT) -> JSONObject:
    """Validate a (possibly partial) units manifest against its archive and the pinned sources; summarize it."""
    if manifest.get("version") != 1:
        raise ValueError("unsupported census units manifest version")
    engine = manifest.get("engine_id")
    if engine not in ENGINES:
        raise ValueError("census units manifest names an unknown engine")
    if manifest["engine_sha256"] != {p: digest(root / p) for p in ENGINES[engine]}:
        raise ValueError("census units manifest engine differs from the pinned sources")
    for key, path in INDEPENDENT.items():
        if manifest[key]["source_sha256"] != digest(root / path):
            raise ValueError(f"census units {key} record was made by a different {path}")
    for ref in (manifest["logs"], manifest["coverage"], manifest["plant"]):
        if not ref["path"].startswith(UNIT_DIR):
            raise ValueError(f"census unit records must live under {UNIT_DIR}")
    n, q = manifest["n"], manifest["class_bits"]
    classes = 1 << q
    members = read_archive(artifact(root, manifest["logs"]))
    header = manifest["header"]
    seen: set[int] = set()
    used: set[str] = set()
    for unit in manifest["units"]:
        b = unit["class"]
        if type(b) is not int or not 0 <= b < classes or b in seen:
            raise ValueError(f"census unit class {b} is out of range or repeated")
        seen.add(b)
        if unit["host"] not in manifest["hosts"]:
            raise ValueError(f"census unit {b} ran on an undeclared host")
        for record in [unit, *unit.get("replays", [])]:
            name = record["log"]
            data = members.get(name)
            if data is None or hashlib.sha256(data).hexdigest() != record["log_sha256"]:
                raise ValueError(f"census unit {b}: log {name} is missing or differs from its sha256")
            used.add(name)
            parsed = parse_log(data.decode())
            if parsed["n"] != n or parsed["unit"] != b or parsed["classes"] != classes:
                raise ValueError(f"census unit {b}: log {name} is for another order or class")
            if parsed["header"] != header:
                raise ValueError(f"census unit {b}: log {name} has a different header")
            if parsed["counters"] != unit["counters"]:
                raise ValueError(f"census unit {b}: log {name} counters differ from the manifest")
        if unit["counters"]["capped"] != 0:
            raise ValueError(f"census unit {b} dropped systems at the nullity cap")
        if set(unit["counters"]) != set(COUNTERS):
            raise ValueError(f"census unit {b} lacks a counter")
    if used != set(members):
        raise ValueError("census unit log archive holds logs the manifest does not list")
    coverage = check_coverage(manifest, header, root)
    plant = check_plant(manifest, coverage, root)
    for unit in manifest["units"]:
        if unit["counters"]["pairs_index"] > coverage["pairs"][unit["class"]]:
            raise ValueError(f"census unit {unit['class']} reports more pairs than its class holds")
    done = sorted(seen)
    return {
        "census": manifest["census"],
        "n": n,
        "classes": classes,
        "done": done,
        "missing": sorted(set(range(classes)) - seen),
        "solutions": sum(u["counters"]["solutions"] for u in manifest["units"]),
        "coverage": coverage,
        "plant": plant,
        "complete": len(done) == classes and len(plant["passed"]) == classes,
    }


def check_coverage(manifest: JSONObject, header: JSONObject, root: Path = ROOT) -> JSONObject:
    """best_cover.c totals against the log headers, and the pairs of every class from the label counts."""
    cov = parse_coverage(artifact(root, manifest["coverage"]).read_text())
    n, q = manifest["n"], manifest["class_bits"]
    classes = 1 << q
    if cov.get("n") != n or cov.get("q") != q or len(cov["forms"]) != q:
        raise ValueError("coverage record is for another order or class split")
    if set(cov["dlabel"]) != set(range(classes)) or set(cov["alabel"]) != set(range(classes)):
        raise ValueError("coverage record lacks a label")
    checks = {
        "D with the row sum": header["d_row_sum"],
        "D under the ceiling": header["d_ceiling"],
        "D orbit minima": header["d_minima"],
        "D with the signature": header["d_signature"],
        "A codes": header["a_codes"],
        "A under the ceiling": header["a_ceiling"],
        "D near the ceiling": 0,
        "A near the ceiling": len(cov["anear"]),
    }
    for key, want in checks.items():
        if cov.get(key) != want:
            raise ValueError(f"coverage record: {key} is {cov.get(key)}, the engine logs say {want}")
    nd, na = cov["dlabel"], cov["alabel"]
    if sum(nd.values()) != header["d_signature"] or sum(na.values()) != header["a_ceiling"]:
        raise ValueError("coverage record: the label counts do not add up to the totals")
    pairs = [sum(nd[label] * na[label ^ b] for label in range(classes)) for b in range(classes)]
    if sum(pairs) != header["d_signature"] * header["a_ceiling"]:
        raise ValueError("coverage record: the classes do not partition the pairs")
    near = [near_row(n, code, psd) for code, psd in cov["anear"]]
    return {
        "forms": cov["forms"],
        "dlabel": nd,
        "alabel": na,
        "pairs": pairs,
        "total_pairs": sum(pairs),
        "a_near": len(near),
        "a_near_above": sum(1 for gap in near if gap > 0),
        "a_near_least_gap": min((abs(gap) for gap in near), default=None),
    }


def near_row(n: int, code: int, psd: float) -> float:
    """Largest PSD of a skew row minus 4n - 2, recomputed here; it must agree with best_cover.c's value."""
    h = (n - 1) // 2
    if code >> h or code & 1:
        raise ValueError("coverage record: a near row is not an A code")
    x = [-1 if (code >> (u - 1)) & 1 else 1 for u in range(1, h + 1)]
    top = max(
        1 + 4 * math.fsum(x[u - 1] * math.sin(2 * math.pi * u * k / n) for u in range(1, h + 1)) ** 2
        for k in range(1, h + 1)
    )
    if abs(top - psd) > 1e-9 or abs(top - (4 * n - 2)) > 1e-6:
        raise ValueError("coverage record: a near row's PSD does not recompute")
    return top - (4 * n - 2)


def check_plant(manifest: JSONObject, coverage: JSONObject, root: Path = ROOT) -> JSONObject:
    """Every planted solution recovered, nothing capped, and the engine's class forms equal best_cover.c's."""
    plant = parse_plant(artifact(root, manifest["plant"]).read_text())
    classes = 1 << manifest["class_bits"]
    if plant["forms"] != {tuple(coverage["forms"])}:
        raise ValueError("the engine's class forms differ from the independently derived ones")
    passed = []
    for b, rec in plant["classes"].items():
        if not 0 <= b < classes:
            raise ValueError(f"planted-solution record names class {b}")
        if rec["planted"] < 1 or rec["recovered"] != rec["planted"] or rec["capped"]:
            raise ValueError(f"planted-solution test failed in class {b}")
        passed.append(b)
    if plant["misses"]:
        raise ValueError("planted-solution record lists a missed plant")
    return {"passed": sorted(passed), "classes": plant["classes"]}


def load_manifest(ref: JSONObject, root: Path = ROOT) -> tuple[JSONObject, JSONObject]:
    """Read a registered manifest by its pinned reference and validate it."""
    if not ref["path"].startswith(UNIT_DIR):
        raise ValueError(f"census units manifest must live under {UNIT_DIR}")
    manifest = read_json(artifact(root, ref), max_bytes=4 * 1024 * 1024)
    return manifest, check_manifest(manifest, root)


def registered(root: Path = ROOT) -> list[tuple[JSONObject, JSONObject]]:
    """Every units manifest registered in baselines/census_replays.json, validated."""
    registry = read_json(root / "baselines/census_replays.json")
    return [load_manifest(ref, root) for ref in registry.get("units", [])]


def apply_units(report: JSONObject, entry: JSONObject, rows: int, root: Path = ROOT) -> JSONObject:
    """A replay report that rests on units: the manifest must be complete and agree with the report."""
    manifest, summary = load_manifest(report["units"], root)
    if manifest["census"] != entry["id"] or manifest["n"] != entry["n"]:
        raise ValueError("census units manifest is for another census")
    if manifest["engine_id"] != report["engine_id"]:
        raise ValueError("census units manifest names another engine than its report")
    if not summary["complete"]:
        raise ValueError(
            f"census units manifest is incomplete: {len(summary['missing'])} classes without a unit, "
            f"{summary['classes'] - len(summary['plant']['passed'])} without a plant test"
        )
    if summary["solutions"] != rows:
        raise ValueError("census units write a different number of rows than the report lists")
    return summary


def log_member(path: Path, name: str) -> str:
    """One log from an archive, for the site."""
    return read_archive(path)[name].decode()
