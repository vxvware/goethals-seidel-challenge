"""Replay one class of a census run as units of work and compare it with the registered log.

    uv run python tools/replay_census_unit.py best-n73 228 --threads 3

builds the pinned engine source (after checking its sha256 against the
manifest) in a work directory, runs the one class, and compares the header and
the counters of the new log with the registered ones. Class 228 of best-n73
takes about 70 minutes on three laptop threads and 2.5 GB of memory. The first
run also writes the D list (dlist.bin) into the work directory and reuses it
afterwards; its sha256 is compared with the manifest. With --log, an existing
log is compared and nothing is run.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import census_units
from verify.common import ROOT, JSONObject, digest, read_json


def manifest_of(census: str) -> JSONObject:
    for ref in read_json(ROOT / "baselines/census_replays.json").get("units", []):
        manifest, _ = census_units.load_manifest(ref)
        if manifest["census"] == census:
            return manifest
    raise SystemExit(f"no registered units manifest for {census}")


def compare(manifest: JSONObject, b: int, text: str) -> bool:
    parsed = census_units.parse_log(text)
    if parsed["unit"] != b or parsed["n"] != manifest["n"]:
        print(f"the log is for n={parsed['n']} class {parsed['unit']}, not class {b}")
        return False
    ok = parsed["header"] == manifest["header"]
    print(f"header (D and A list sizes, stage-1 corank, table layout, A pieces): {'equal' if ok else 'DIFFERENT'}")
    unit = next((u for u in manifest["units"] if u["class"] == b), None)
    if unit is None:
        print(f"class {b} has no registered unit yet; counters of this run:")
        for key, value in parsed["counters"].items():
            print(f"  {key} {value}")
        return ok
    for key in census_units.COUNTERS:
        mine, theirs = parsed["counters"][key], unit["counters"][key]
        print(f"  {key}: {mine} {'=' if mine == theirs else '!='} {theirs} ({unit['host']})")
        ok = ok and mine == theirs
    print("REPLAY MATCHES" if ok else "REPLAY DIFFERS")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("census")
    p.add_argument("unit", type=int)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--work", default="census-unit-replay")
    p.add_argument("--log", help="compare this existing log instead of running")
    a = p.parse_args()
    manifest = manifest_of(a.census)
    if not 0 <= a.unit < 1 << manifest["class_bits"]:
        raise SystemExit("class out of range")
    if a.log:
        raise SystemExit(0 if compare(manifest, a.unit, Path(a.log).read_text()) else 1)
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (source,) = manifest["engine_sha256"]
    if digest(ROOT / source) != manifest["engine_sha256"][source]:
        raise SystemExit(f"{source} differs from the pinned engine")
    shutil.copyfile(ROOT / source, work / "best_hkt2.c")
    binary = work / "best_hkt2_wide"
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-DWIDE", "best_hkt2.c", "-lm", "-o", binary.name]
    subprocess.run(cmd, cwd=work, check=True)
    env = dict(
        os.environ,
        BEST_QB=str(manifest["class_bits"]),
        BEST_CLASS=str(a.unit),
        BEST_DFILE="dlist.bin",
        OMP_WAIT_POLICY="PASSIVE",
    )
    log = work / f"c{a.unit}.log"
    print(f"running class {a.unit} of {a.census} on {a.threads} threads; log {log}", flush=True)
    with log.open("w") as out:
        subprocess.run(
            ["./" + binary.name, str(manifest["n"]), str(a.threads), f"c{a.unit}.txt"],
            cwd=work,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )
    dlist = digest(work / "dlist.bin")
    print(f"D list sha256 {dlist}: {'equal' if dlist == manifest['dlist_sha256'] else 'DIFFERENT'}")
    raise SystemExit(0 if compare(manifest, a.unit, log.read_text()) else 1)


if __name__ == "__main__":
    main()
