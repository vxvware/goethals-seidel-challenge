"""Submission PRs may add data/evidence, but cannot modify the trusted stack."""

from pathlib import PurePosixPath
import argparse
import subprocess


def check_changes(changes):
    entries = False
    for status, path in changes:
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe changed path")
        if p.parts[0] in ("quadruples", "certificates", "census"):
            entries = True
            if status != "A" or len(p.parts) != 2 or p.suffix != ".json":
                raise ValueError("submission entries must be new, flat JSON files")
        elif p.parts[0] in ("evidence", "notes"):
            if status != "A":
                raise ValueError("submission evidence/notes are append-only")
            if p.suffix not in (".json", ".jsonl", ".gz", ".md", ".txt", ".npy", ".npz"):
                raise ValueError("executable or unsupported submission evidence")
        else:
            raise ValueError(f"submission may not change trusted path: {path}")
    if not entries:
        raise ValueError("submission must add an entry")


def git_changes(base, head="HEAD"):
    output = subprocess.check_output(
        ["git", "diff", "--name-status", "--no-renames", f"{base}...{head}"], text=True
    )
    return [line.split("\t", 1) for line in output.splitlines()]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--head", default="HEAD")
    a = p.parse_args()
    try:
        check_changes(git_changes(a.base, a.head))
    except (ValueError, subprocess.CalledProcessError) as e:
        p.exit(1, f"FAIL: {e}\n")
