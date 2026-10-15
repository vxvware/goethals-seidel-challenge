"""Strict JSON and bounded, repository-local artifact access."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 256 * 1024
MAX_N = 511
TRACKS = ("williamson", "good", "best", "kkss", "skew-sds", "general")
PARENTS = {
    "williamson": ("general",),
    "good": ("skew-sds", "general"),
    "best": ("skew-sds", "general"),
    "kkss": ("skew-sds", "general"),
    "skew-sds": ("general",),
    "general": (),
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path, max_bytes=MAX_FILE):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError(f"symlink or oversized JSON: {path.name}")
    return json.loads(
        path.read_text(),
        object_pairs_hook=unique_object,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(root, ref):
    relative = Path(ref["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be repository-relative")
    path = Path(root) / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(Path(root).resolve())
    ):
        raise ValueError(f"missing or unsafe artifact: {relative}")
    if digest(path) != ref["sha256"]:
        raise ValueError(f"artifact hash mismatch: {relative}")
    return path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
