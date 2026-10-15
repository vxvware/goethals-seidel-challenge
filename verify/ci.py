#!/usr/bin/env python3
"""PR gate executed from the base checkout against candidate data."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, read_json
from verify.check_manifest import check
from verify.check_submission_scope import check_changes
from verify.check_authorship import check_author
from verify.check_prose import check as check_prose
from verify.board import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--author", required=True)
    p.add_argument("--association", default="NONE")
    a = p.parse_args()
    root = a.candidate.resolve()
    output = subprocess.check_output(
        ["git", "diff", "--name-status", "--no-renames", f"{a.base}...HEAD"], cwd=root, text=True
    )
    changes = [line.split("\t", 1) for line in output.splitlines()]
    entries = any(
        Path(path).parts[0] in ("quadruples", "certificates", "census") for _, path in changes
    )
    if entries:
        check_changes(changes)
        for status, path in changes:
            parts = Path(path).parts
            if parts[0] in ("quadruples", "certificates", "census"):
                check_author(read_json(root / path), a.author)
            elif parts[0] == "notes" and Path(path).suffix == ".md":
                check_prose(root / path, root)
        check(ROOT)
        load(root)
        print("Submission passes trusted-base validation.")
    else:
        if a.association not in ("OWNER", "MEMBER", "COLLABORATOR"):
            raise ValueError(
                "trusted-stack changes require maintainer review and a maintainer-authored PR"
            )
        # No candidate code runs in this read-only base gate. Maintenance code
        # is tested in the ordinary isolated PR test job, without secrets.
        print("Maintainer infrastructure change; candidate checks run in isolated validation job.")


if __name__ == "__main__":
    main()
