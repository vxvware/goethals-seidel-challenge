"""Reject broken local Markdown links and private workspace paths in submitted notes."""

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from verify.common import ROOT


def check(path, root=ROOT):
    path = Path(path)
    text = path.read_text()
    if re.search(r"(?<!\w)(?:~/|/home/|/Users/)", text):
        raise ValueError(f"{path.name}: private filesystem reference")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.strip("<>")
        if urlparse(target).scheme or target.startswith("#"):
            continue
        target = unquote(target.split("#")[0])
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            raise ValueError(f"{path.name}: unresolved repository link {target}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    for name in a.files:
        check(ROOT / name)
