"""Schema validation shared by CLI, CI, and importer."""

from functools import lru_cache

from jsonschema import Draft202012Validator, FormatChecker

from verify.common import ROOT, JSONObject, read_json


@lru_cache(maxsize=1)
def validator() -> Draft202012Validator:
    return Draft202012Validator(read_json(ROOT / "schema/entry.schema.json"), format_checker=FormatChecker())


def validate(entry: JSONObject) -> JSONObject:
    errors = sorted(validator().iter_errors(entry), key=lambda e: str(e.path))
    if errors:
        first = errors[0]
        details = first.context or [first]
        raise ValueError("schema: " + "; ".join(e.message for e in details[:6]))
    n = entry["n"]
    if type(n) is not int or n % 2 != 1:
        raise ValueError("n must be an odd integer")
    if entry["group"] != {"kind": "cyclic", "invariant_factors": [n]}:
        raise ValueError("noncyclic groups are reserved for v2")
    return entry
