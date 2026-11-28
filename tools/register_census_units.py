"""Write and register the per-class records of a census run as units of work.

The best-matrix census at n=73 runs best_hkt2.c once per class of pairs
(BEST_QB=9 BEST_CLASS=b) on several hosts. Each host keeps, under its output
directory, c<b>.log for every unit, c<b>.done for every unit that finished and
summary.txt with one line per finished unit (wall time, threads, finish time).
Copy those directories, then

    uv run python tools/register_census_units.py best-n73 --n 73 --class-bits 9 \\
        --hosts hosts.json --coverage best_cover.out --plant plant.out

hosts.json maps each host name to its output directory and to what ran there
(binary sha256, CPU, compiler, threads, build and run commands), and may list
replays of single classes on other hosts as {"class", "host", "log",
"threads", "wall_seconds"}. Only finished units are recorded; rerun the tool
as more classes finish. It writes, under evidence/census-units/<census>/, the
manifest units.json, every log in logs.tar.xz (deterministic: sorted names,
zero times and owners), and the raw outputs of best_cover.c and best_plant.c;
then it validates the manifest exactly as the board does and registers it in
baselines/census_replays.json. All of these are trusted paths, so this belongs
in a maintenance PR.
"""

import argparse
import hashlib
import io
import lzma
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import census_units
from verify.common import ROOT, JSONObject, digest, read_json, write_json

SUMMARY = re.compile(r"^c(\d+) wall=(\d+)s threads=(\d+) rows=(\d+) nullity-capped systems: (\d+) (\S+)$")


def tar_xz(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tar.addfile(info, io.BytesIO(files[name]))
    return lzma.compress(buffer.getvalue(), preset=9 | lzma.PRESET_EXTREME)


def host_units(name: str, folder: Path, classes: int) -> list[tuple[JSONObject, bytes]]:
    summary: dict[int, re.Match[str]] = {}
    path = folder / "summary.txt"
    for line in path.read_text().splitlines() if path.exists() else []:
        m = SUMMARY.match(line.strip())
        if m:
            summary[int(m[1])] = m
    found = []
    for b in range(classes):
        if not (folder / f"c{b}.done").exists():
            continue
        data = (folder / f"c{b}.log").read_bytes()
        parsed = census_units.parse_log(data.decode())
        m = summary.get(b)
        if m is None:
            raise SystemExit(f"{name}: unit c{b} is done but has no summary line")
        found.append(
            (
                {
                    "class": b,
                    "host": name,
                    "threads": int(m[3]),
                    "wall_seconds": int(m[2]),
                    "finished": m[6],
                    "log": f"c{b}.log",
                    "log_sha256": hashlib.sha256(data).hexdigest(),
                    "counters": parsed["counters"],
                },
                data,
            )
        )
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("census")
    p.add_argument("--engine", default="best-census-hkt-v2", choices=sorted(census_units.ENGINES))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--class-bits", type=int, required=True)
    p.add_argument("--hosts", required=True, help="JSON: host -> {dir, binary_sha256, cpu, compiler, threads, ...}")
    p.add_argument("--coverage", required=True, help="stdout of best_cover <n> <q> <threads>")
    p.add_argument("--plant", required=True, nargs="+", help="stdout of the best_plant runs, in order")
    a = p.parse_args()
    spec = read_json(a.hosts)
    classes = 1 << a.class_bits
    folder = ROOT / census_units.UNIT_DIR / a.census
    rel = folder.relative_to(ROOT).as_posix()
    units: dict[int, JSONObject] = {}
    logs: dict[str, bytes] = {}
    for name, info in spec["hosts"].items():
        if "dir" not in info:
            continue
        for unit, data in host_units(name, Path(info["dir"]), classes):
            if unit["class"] in units:
                raise SystemExit(
                    f"class {unit['class']} finished on two hosts; keep one and list the other as a replay"
                )
            units[unit["class"]] = unit
            logs[unit["log"]] = data
    for replay in spec.get("replays", []):
        b = replay["class"]
        if b not in units:
            continue
        data = Path(replay["log"]).read_bytes()
        name = f"replays/c{b}-{replay['host']}.log"
        logs[name] = data
        units[b].setdefault("replays", []).append(
            {
                "host": replay["host"],
                "threads": replay["threads"],
                "wall_seconds": replay["wall_seconds"],
                "log": name,
                "log_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if not units:
        raise SystemExit("no finished units")
    first = next(iter(logs.values()))
    header = census_units.parse_log(first.decode())["header"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "logs.tar.xz").write_bytes(tar_xz(logs))
    (folder / "coverage.txt").write_text(Path(a.coverage).read_text())
    (folder / "plant.txt").write_text("".join(Path(f).read_text() for f in a.plant))
    hosts = {name: {k: v for k, v in info.items() if k != "dir"} for name, info in spec["hosts"].items()}
    manifest = {
        "version": 1,
        "census": a.census,
        "engine_id": a.engine,
        "engine_sha256": {path: digest(ROOT / path) for path in census_units.ENGINES[a.engine]},
        "n": a.n,
        "class_bits": a.class_bits,
        "dlist_sha256": spec["dlist_sha256"],
        "build": spec["build"],
        "run": spec["run"],
        "hosts": hosts,
        "header": header,
        "logs": {"path": f"{rel}/logs.tar.xz", "sha256": digest(folder / "logs.tar.xz")},
        "coverage": {
            "path": f"{rel}/coverage.txt",
            "sha256": digest(folder / "coverage.txt"),
            "source_sha256": digest(ROOT / census_units.INDEPENDENT["coverage"]),
            **spec["coverage"],
        },
        "plant": {
            "path": f"{rel}/plant.txt",
            "sha256": digest(folder / "plant.txt"),
            "source_sha256": digest(ROOT / census_units.INDEPENDENT["plant"]),
            **spec["plant"],
        },
        "units": [units[b] for b in sorted(units)],
    }
    summary = census_units.check_manifest(manifest)
    path = folder / "units.json"
    write_json(path, manifest)
    registry = read_json(ROOT / "baselines/census_replays.json")
    ref = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
    registry["units"] = sorted(
        [r for r in registry.get("units", []) if r["path"] != ref["path"]] + [ref], key=lambda r: r["path"]
    )
    write_json(ROOT / "baselines/census_replays.json", registry)
    print(
        f"{a.census}: {len(summary['done'])} of {classes} classes finished, "
        f"{len(summary['plant']['passed'])} planted, {summary['solutions']} solutions, registered at {ref['path']}"
    )


if __name__ == "__main__":
    main()
