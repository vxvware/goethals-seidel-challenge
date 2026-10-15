"""Bind newly submitted entries to the PR author, not the CI rerun actor."""

import argparse
from pathlib import Path
from verify.common import ROOT, read_json
from verify.check_submission_scope import git_changes


def check_author(document, author):
    if document["submitted_by"].casefold() != author.casefold():
        raise ValueError(f"{document['id']}: submitted_by must match PR author {author}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--author", required=True)
    a = p.parse_args()
    for status, path in git_changes(a.base):
        if status == "A" and Path(path).parts[0] in ("quadruples", "certificates", "census"):
            check_author(read_json(ROOT / path), a.author)
