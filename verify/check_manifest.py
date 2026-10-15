"""Pin trusted validator, schema, baseline, replay engine, and build inputs."""

import argparse
from verify.common import ROOT, digest, read_json, write_json


def trusted_files(root):
    paths = []
    for name in ("verify", "schema", "baselines", "site"):
        paths.extend(
            p
            for p in (root / name).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in (".py", ".json", ".css", ".js")
        )
    paths.extend(p for p in (root / "research/starter/best").glob("census.c"))
    paths.extend(p for p in (root / "research/starter/good_gpu").glob("*.py"))
    paths.extend((root / ".github/workflows").glob("*.yml"))
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock", "tools/gs", "gs"))
    return sorted(p for p in paths if p.name != "validator_manifest.json")


def snapshot(root=ROOT):
    return {p.relative_to(root).as_posix(): digest(p) for p in trusted_files(root)}


def check(root=ROOT):
    expected = read_json(root / "verify/validator_manifest.json", max_bytes=4 * 1024 * 1024)[
        "sha256"
    ]
    actual = snapshot(root)
    if actual != expected:
        changed = sorted(
            k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k)
        )
        raise ValueError("trusted manifest drift: " + ", ".join(changed))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--update", action="store_true")
    a = p.parse_args()
    if a.update:
        write_json(ROOT / "verify/validator_manifest.json", {"version": 1, "sha256": snapshot()})
    else:
        check()
        print("Trusted manifest matches.")
