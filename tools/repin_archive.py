#!/usr/bin/env python3
"""Move the commit that the links to archived source files are pinned to.

Source records of files copied from the private campaign repositories link the
copy in this repository at one commit, the ARCHIVE constant of the importers.
This rewrites that commit everywhere outside docs/, rebinds each registered
census replay whose census entry changed by nothing but the pin, and updates
the trusted manifest.

    uv run python tools/repin_archive.py <new commit>
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.check_manifest import snapshot
from verify.common import ROOT, digest, read_json, write_json

PREFIX = "https://github.com/vxvware/goethals-seidel-challenge/blob/"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("commit")
    new = p.parse_args().commit
    if not re.fullmatch(r"[0-9a-f]{40}", new):
        p.exit(1, "FAIL: the commit must be a full 40-character sha\n")
    text = (ROOT / "tools/import_seeds.py").read_text()
    pinned = re.search(rf'ARCHIVE = "{re.escape(PREFIX)}([0-9a-f]{{40}})"', text)
    if pinned is None:
        p.exit(1, "FAIL: tools/import_seeds.py has no ARCHIVE pin\n")
    old = pinned.group(1)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    changed = 0
    for name in tracked:
        path = ROOT / name
        if not name or name.startswith("docs/") or not path.is_file():
            continue
        try:
            body = path.read_text()
        except UnicodeDecodeError:
            continue
        if PREFIX + old in body:
            path.write_text(body.replace(PREFIX + old, PREFIX + new))
            changed += 1
    registry = read_json(ROOT / "baselines/census_replays.json")
    rebound = 0
    for ref in registry["reports"]:
        path = ROOT / ref["path"]
        report = read_json(path)
        entry = ROOT / f"census/{report['entry_id']}.json"
        if digest(entry) == report["entry_sha256"]:
            continue
        # Rebind only when undoing the pin change restores the bound bytes.
        before = entry.read_text().replace(PREFIX + new, PREFIX + old).encode()
        same = hashlib.sha256(before).hexdigest() == report["entry_sha256"]
        if not same or digest(path) != ref["sha256"]:
            p.exit(1, f"FAIL: {entry.name} differs from its bound bytes by more than the pin\n")
        path.write_text(path.read_text().replace(report["entry_sha256"], digest(entry)))
        ref["sha256"] = digest(path)
        rebound += 1
    write_json(ROOT / "baselines/census_replays.json", registry)
    write_json(ROOT / "verify/validator_manifest.json", {"version": 1, "sha256": snapshot()})
    print(f"{old} -> {new}: {changed} files rewritten, {rebound} census replays rebound")


if __name__ == "__main__":
    main()
