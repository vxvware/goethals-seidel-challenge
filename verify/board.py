"""Board aggregation and consistency, shared by the CLI and static site."""

from pathlib import Path

from verify import census_engine, contributions, evidence, gs_verify
from verify.common import BOARD_MAX_N, PARENTS, ROOT, TRACKS, ById, JSONObject, PathLike, artifact, read_json, tracks_at
from verify.model import validate


def load(root: PathLike = ROOT, reproduce: bool = True) -> tuple[ById, ById, list[JSONObject]]:
    root = Path(root)
    documents: ById = {}
    results: ById = {}
    fingerprints: dict[str, str] = {}
    files = sorted(
        p for directory in ("quadruples", "certificates", "census") for p in (root / directory).glob("*.json")
    )
    if len(files) > 4096 or sum(p.stat().st_size for p in files) > 64 * 1024 * 1024:
        raise ValueError("board entry resource limit exceeded")
    baseline = read_json(root / "baselines/skew-gaps.json")
    for path in files:
        doc = validate(read_json(path))
        ident = doc["id"]
        if ident != path.stem or ident in documents:
            raise ValueError(f"entry id/filename mismatch or duplicate: {path.name}")
        expected = {"quadruple": "quadruples", "certificate": "certificates", "census": "census"}[doc["type"]]
        if path.parent.name != expected:
            raise ValueError("entry in wrong directory")
        for source in doc["sources"]:
            if "artifact" in source:
                artifact(root, source["artifact"])
        documents[ident] = doc
        if doc["type"] == "quadruple":
            result = gs_verify.verify(doc, baseline)
            fp = result["fingerprint"]
            if fp in fingerprints:
                previous = results[fingerprints[fp]]
                if result["canonical_rows"] == previous["canonical_rows"]:
                    raise ValueError(f"equivalent duplicate: {ident} and {previous['id']}")
                raise ValueError("fingerprint collision: canonical rows differ")
            fingerprints[fp] = ident
            results[ident] = result
    quads = dict(results)
    for ident, doc in documents.items():
        if doc["type"] == "certificate":
            results[ident] = evidence.verify(doc, root)
        elif doc["type"] == "census":
            results[ident] = census_engine.verify(doc, quads, reproduce, root)
    records = read_json(root / "baselines/order-status.json")["records"]
    from verify.census_replays import apply_registered as apply_census_replays
    from verify.replay_registry import apply_registered

    apply_registered(documents, results, root)
    apply_census_replays(documents, results, root)
    check_consistency(documents, results, records)
    contributions.apply(documents, results, root)
    return documents, results, records


def check_consistency(documents: ById, results: ById, baseline: list[JSONObject]) -> None:
    for known in (r for r in baseline if r["status"] == "exists"):
        for absent in (r for r in baseline if r["status"] == "empty"):
            if known["n"] == absent["n"] and absent["stratum"] in (
                known["stratum"],
                *PARENTS[known["stratum"]],
            ):
                raise ValueError("contradictory sourced baseline records")
    for cert in (x for x in results.values() if x["type"] == "certificate" and x["claim"] == "empty"):
        for q in (x for x in results.values() if x["type"] == "quadruple"):
            if (
                q["n"] == cert["n"]
                and cert["stratum"] in q["tracks"]
                and evidence.in_scope(q["canonical_rows"], cert["scope"])
            ):
                raise ValueError(f"existence/nonexistence conflict: {q['id']} / {cert['id']}")
        if cert["scope"]["kind"] == "full":
            for rec in baseline:
                if (
                    rec["n"] == cert["n"]
                    and rec["status"] == "exists"
                    and cert["stratum"] in (rec["stratum"], *PARENTS[rec["stratum"]])
                ):
                    raise ValueError(f"certificate conflicts with sourced existence baseline: {cert['id']}")
    for rec in baseline:
        if rec["status"] not in ("empty", "contradicted"):
            continue
        refuted = [
            q
            for q in results.values()
            if q["n"] == rec["n"]
            and (
                (q["type"] == "quadruple" and rec["stratum"] in q["tracks"])
                # The rows of a catalogue are verified quadruples of its stratum and of every stratum containing it.
                or (
                    q["type"] == "census"
                    and "catalogue" in q
                    and rec["stratum"] in (q["stratum"], *PARENTS[q["stratum"]])
                )
            )
        ]
        if rec["status"] == "empty" and refuted:
            raise ValueError(f"existence conflicts with nonexistence baseline: {refuted[0]['id']}")
        if rec["status"] == "contradicted" and not refuted:
            raise ValueError(f"contradicted claim without verified rows: {rec['stratum']} n={rec['n']}")


def cells(documents: ById, results: ById, baseline: list[JSONObject], max_n: int = BOARD_MAX_N) -> list[JSONObject]:
    """One cell per order and stratum: six at odd n, and at even n the two strata that need no skew row."""
    output = []
    for n in range(1, max_n + 1):
        for track in tracks_at(n):
            relevant = []
            for rec in baseline:
                if rec["n"] != n:
                    continue
                if (
                    rec["status"] == "exists"
                    and track
                    in (
                        rec["stratum"],
                        *PARENTS[rec["stratum"]],
                    )
                ) or (rec["status"] == "empty" and rec["stratum"] in (track, *PARENTS[track])):
                    relevant.append(rec)
            exists = [r for r in relevant if r["status"] == "exists"]
            empty = [r for r in relevant if r["status"] == "empty"]
            contradicted = [
                rec for rec in baseline if rec["n"] == n and rec["status"] == "contradicted" and rec["stratum"] == track
            ]
            quads = [r for r in results.values() if r["type"] == "quadruple" and r["n"] == n and track in r["tracks"]]
            certs = [
                r
                for r in results.values()
                if r["type"] == "certificate" and r["n"] == n and r["stratum"] in (track, *PARENTS[track])
            ]
            whole = [
                c
                for c in certs
                if c["claim"] == "empty"
                and c["scope"]["kind"] == "full"
                and (c["tier"] is not None or c.get("provenance") in ("literature", "unreplayed-census"))
            ]
            censuses = [r for r in results.values() if r["type"] == "census" and r["n"] == n and r["stratum"] == track]
            # Catalogues whose rows belong to this cell: those of the track and of the tracks it contains.
            catalogues = [
                {"census": r["id"], "stratum": r["stratum"], **r["catalogue"]}
                for r in results.values()
                if r["type"] == "census"
                and r["n"] == n
                and "catalogue" in r
                and track in (r["stratum"], *PARENTS[r["stratum"]])
            ]
            # An exact census with no classes is a replayed proof of emptiness.
            exact_empty = [c for c in censuses if c["bound"] == "exact" and c["count"] == 0]
            held = bool(quads or catalogues)
            status = "exists" if held or exists else "empty" if empty or whole or exact_empty else "unresolved"
            availability = "verified-on-board" if held else "published-not-imported" if exists else "none"
            tiers = [c["tier"] for c in whole if c["tier"]] + [c["tier"] for c in exact_empty]
            # Scoped evidence can be relevant to narrower strata without being
            # the result that establishes their status. Credit its target cell.
            credits = contributions.collect([*quads, *(c for c in certs if c["stratum"] == track), *censuses])
            origins = {c["relation"] for c in credits}
            if relevant or any(documents[c["id"]]["method"] in ("literature", "row-sum-theorem") for c in certs):
                origins.add("published-import")
            rank = {"receipt-backed": 0, "spot-checked": 1, "reproduced": 2, "proven": 3}
            # Every graded tier is a key of rank, so rank.get never returns None here.
            tier = "reproduced" if held else max(tiers, key=rank.get) if tiers else None  # type: ignore[arg-type]
            basis = sorted({c["tier"] or c["provenance"] for c in whole} | {c["tier"] for c in exact_empty})
            output.append(
                {
                    "n": n,
                    "stratum": track,
                    "group": "cyclic",
                    "status": status,
                    "artifact_status": availability,
                    "tier": tier,
                    "empty_basis": basis,
                    "quadruples": [q["id"] for q in quads],
                    "evidence": [c["id"] for c in certs],
                    "censuses": [c["id"] for c in censuses],
                    "catalogues": catalogues,
                    "contributions": credits,
                    "origins": sorted(origins),
                    "sources": relevant,
                    "contradicted": contradicted,
                    "flags": [f for q in quads for f in q["flags"]]
                    + [{"kind": "contradicts-published-nonexistence", "source": rec["source"]} for rec in contradicted],
                    "note": "Requires n=r²+r+1." if track == "best" else "",
                }
            )
    return output


def summary(documents: ById, results: ById, baseline: list[JSONObject], max_n: int = BOARD_MAX_N) -> JSONObject:
    grid = cells(documents, results, baseline, max_n)
    return {
        "schema_version": 1,
        "equivalence": "equiv-v1",
        "group": "cyclic",
        "entry_count": len(documents),
        "quadruples": sum(r["type"] == "quadruple" for r in results.values()),
        "certificates": sum(r["type"] == "certificate" for r in results.values()),
        "censuses": sum(r["type"] == "census" for r in results.values()),
        "catalogue_rows": sum(r["catalogue"]["rows"] for r in results.values() if "catalogue" in r),
        "contributions": contributions.collect(results.values()),
        "cells": grid,
        "research_targets": [c for c in grid if c["status"] == "unresolved"],
        "import_tasks": [c for c in grid if c["artifact_status"] == "published-not-imported"],
        "largest_verified": {
            track: max(
                (r["n"] for r in results.values() if r["type"] == "quadruple" and track in r["tracks"]),
                default=None,
            )
            for track in TRACKS
        },
        "tier_counts": {
            tier: sum(r.get("tier") == tier for r in results.values())
            for tier in ("proven", "reproduced", "spot-checked", "receipt-backed")
        },
        "ungraded_claims": sum(r.get("tier") is None for r in results.values()),
    }
