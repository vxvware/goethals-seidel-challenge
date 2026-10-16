"""Generate the strict, self-contained submission schema."""

from pathlib import Path
import json

S = {"type": "string", "minLength": 1, "maxLength": 2000}
NONNEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
REF = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path", "sha256"],
    "properties": {
        "path": {"type": "string", "pattern": "^evidence/[A-Za-z0-9_./-]+$"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}
TRACK = {"enum": ["williamson", "good", "best", "kkss", "skew-sds", "general"]}
SOURCE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "url"],
    "properties": {
        "title": S,
        "url": {"type": "string", "pattern": "^https://"},
        "commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "source_path": S,
        "artifact": REF,
    },
}
COMMON = {
    "schema_version": {"const": 1},
    "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,100}$"},
    "n": {"type": "integer", "minimum": 1, "maximum": 511},
    "group": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "invariant_factors"],
        "properties": {
            "kind": {"enum": ["cyclic", "abelian"]},
            "invariant_factors": {
                "type": "array",
                "minItems": 1,
                "maxItems": 9,
                "items": {"type": "integer", "minimum": 1},
            },
        },
    },
    "submitted_by": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9-]{0,38}$"},
    "submitted_at": {"type": "string", "format": "date"},
    "discovered_by": {"type": "array", "minItems": 1, "maxItems": 30, "items": S},
    "sources": {"type": "array", "minItems": 1, "maxItems": 100, "items": SOURCE},
    "notes": {"type": "string", "maxLength": 16000},
}


def entry(kind, properties, required):
    props = {**COMMON, "type": {"const": kind}, **properties}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": [k for k in COMMON if k != "notes"] + ["type"] + required,
    }


quad = entry(
    "quadruple",
    {
        "equivalence": {"const": "equiv-v1"},
        "rows": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 511,
                "items": {"type": "integer", "enum": [-1, 1]},
            },
        },
        "blocks": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "array",
                "maxItems": 511,
                "uniqueItems": True,
                "items": NONNEGATIVE_INTEGER,
            },
        },
    },
    ["equivalence"],
)
quad["oneOf"] = [
    {"required": ["rows"], "not": {"required": ["blocks"]}},
    {"required": ["blocks"], "not": {"required": ["rows"]}},
]
scope = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind"],
            "properties": {"kind": {"const": "full"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "subgroup_order", "include_zero", "skew_blocks"],
            "properties": {
                "kind": {"const": "same-subgroup-coset-unions"},
                "subgroup_order": {"type": "integer", "minimum": 1},
                "include_zero": {"type": "boolean"},
                "skew_blocks": {"const": 1},
            },
        },
    ]
}
cert = entry(
    "certificate",
    {
        "stratum": TRACK,
        "claim": {"enum": ["empty", "partial"]},
        "scope": scope,
        "method": {
            "enum": [
                "literature",
                "unreplayed-census",
                "row-sum-theorem",
                "coset-row-sum-theorem",
                "good-coverage-v1",
                "archived-good-coverage-v1",
                "best-coverage-v1",
            ]
        },
        "records": {"type": "array", "maxItems": 200, "items": REF},
        "engine": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "snapshot_commit", "binding", "artifact"],
            "properties": {
                "id": S,
                "snapshot_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "binding": {"enum": ["archived-with-receipt", "replayed"]},
                "artifact": REF,
            },
        },
        "gaps": {"type": "array", "maxItems": 30, "items": S},
    },
    ["stratum", "claim", "scope", "method", "records", "gaps"],
)
census = entry(
    "census",
    {
        "stratum": TRACK,
        "equivalence": {"const": "equiv-v1"},
        "representatives": {
            "type": "array",
            "maxItems": 4096,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^[a-z0-9-]+$"},
        },
        "claimed_count": NONNEGATIVE_INTEGER,
        "claim_kind": {"enum": ["lower-bound", "published-exact", "computed-exact"]},
        "enumerator": {"enum": ["best-census-c-v1", "small-exact-v1"]},
        "secondary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "branch_orbits": NONNEGATIVE_INTEGER,
                "expanded_quadruples": NONNEGATIVE_INTEGER,
                "stored_rows": NONNEGATIVE_INTEGER,
            },
        },
    },
    ["stratum", "equivalence", "representatives", "claimed_count", "claim_kind"],
)
schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Goethals–Seidel submission v1",
    "oneOf": [quad, cert, census],
}
Path("schema/entry.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
