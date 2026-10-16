"""Board aggregation and consistency, shared by the CLI and static site."""

from pathlib import Path
from verify.common import ROOT, TRACKS, PARENTS, read_json, artifact
from verify.model import validate
from verify import gs_verify, evidence, census_engine, contributions


def load(root=ROOT, reproduce=True):
    root = Path(root)
    documents = {}
    results = {}
    fingerprints = {}
    files = sorted(
        p
        for directory in ("quadruples", "certificates", "census")
        for p in (root / directory).glob("*.json")
    )
    if len(files) > 4096 or sum(p.stat().st_size for p in files) > 64 * 1024 * 1024:
        raise ValueError("board entry resource limit exceeded")
    baseline = read_json(root / "baselines/skew-gaps.json")
    for path in files:
        doc = validate(read_json(path))
        ident = doc["id"]
        if ident != path.stem or ident in documents:
            raise ValueError(f"entry id/filename mismatch or duplicate: {path.name}")
        expected = {"quadruple": "quadruples", "certificate": "certificates", "census": "census"}[
            doc["type"]
        ]
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
            results[ident] = census_engine.verify(doc, quads, reproduce)
    records = read_json(root / "baselines/order-status.json")["records"]
    from verify.replay_registry import apply_registered

    apply_registered(documents, results, root)
    check_consistency(documents, results, records)
    contributions.apply(documents, results, root)
    return documents, results, records


def check_consistency(documents, results, baseline):
    for known in (r for r in baseline if r["status"] == "exists"):
        for absent in (r for r in baseline if r["status"] == "empty"):
            if known["n"] == absent["n"] and absent["stratum"] in (
                known["stratum"],
                *PARENTS[known["stratum"]],
            ):
                raise ValueError("contradictory sourced baseline records")
    for cert in (
        x for x in results.values() if x["type"] == "certificate" and x["claim"] == "empty"
    ):
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
                    raise ValueError(
                        f"certificate conflicts with sourced existence baseline: {cert['id']}"
                    )
    for rec in baseline:
        if rec["status"] not in ("empty", "contradicted"):
            continue
        refuted = [
            q
            for q in results.values()
            if q["type"] == "quadruple" and q["n"] == rec["n"] and rec["stratum"] in q["tracks"]
        ]
        if rec["status"] == "empty" and refuted:
            raise ValueError(f"existence conflicts with nonexistence baseline: {refuted[0]['id']}")
        if rec["status"] == "contradicted" and not refuted:
            raise ValueError(
                f"contradicted claim without verified rows: {rec['stratum']} n={rec['n']}"
            )


def cells(documents, results, baseline, max_n=131):
    output = []
    for n in range(1, max_n + 1, 2):
        for track in TRACKS:
            relevant = []
            for rec in baseline:
                if rec["n"] != n:
                    continue
                if rec["status"] == "exists" and track in (
                    rec["stratum"],
                    *PARENTS[rec["stratum"]],
                ):
                    relevant.append(rec)
                elif rec["status"] == "empty" and rec["stratum"] in (track, *PARENTS[track]):
                    relevant.append(rec)
            exists = [r for r in relevant if r["status"] == "exists"]
            empty = [r for r in relevant if r["status"] == "empty"]
            contradicted = [
                rec
                for rec in baseline
                if rec["n"] == n and rec["status"] == "contradicted" and rec["stratum"] == track
            ]
            quads = [
                r
                for r in results.values()
                if r["type"] == "quadruple" and r["n"] == n and track in r["tracks"]
            ]
            certs = [
                r
                for r in results.values()
                if r["type"] == "certificate"
                and r["n"] == n
                and r["stratum"] in (track, *PARENTS[track])
            ]
            whole = [
                c
                for c in certs
                if c["claim"] == "empty"
                and c["scope"]["kind"] == "full"
                and (
                    c["tier"] is not None
                    or c.get("provenance") in ("literature", "unreplayed-census")
                )
            ]
            status = "exists" if quads or exists else "empty" if empty or whole else "unresolved"
            availability = (
                "verified-on-board" if quads else "published-not-imported" if exists else "none"
            )
            tiers = [c["tier"] for c in whole if c["tier"]]
            censuses = [
                r
                for r in results.values()
                if r["type"] == "census" and r["n"] == n and r["stratum"] == track
            ]
            # Scoped evidence can be relevant to narrower strata without being
            # the result that establishes their status. Credit its target cell.
            credits = contributions.collect(
                [*quads, *(c for c in certs if c["stratum"] == track), *censuses]
            )
            origins = {c["relation"] for c in credits}
            if relevant or any(
                documents[c["id"]]["method"] in ("literature", "row-sum-theorem") for c in certs
            ):
                origins.add("published-import")
            rank = {"receipt-backed": 0, "spot-checked": 1, "reproduced": 2, "proven": 3}
            tier = "reproduced" if quads else max(tiers, key=rank.get) if tiers else None
            basis = sorted({c["tier"] or c["provenance"] for c in whole})
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
                    "contributions": credits,
                    "origins": sorted(origins),
                    "sources": relevant,
                    "contradicted": contradicted,
                    "flags": [f for q in quads for f in q["flags"]]
                    + [
                        {"kind": "contradicts-published-nonexistence", "source": rec["source"]}
                        for rec in contradicted
                    ],
                    "note": "Requires n=r²+r+1." if track == "best" else "",
                }
            )
    return output


def summary(documents, results, baseline, max_n=131):
    grid = cells(documents, results, baseline, max_n)
    return {
        "schema_version": 1,
        "equivalence": "equiv-v1",
        "group": "cyclic",
        "entry_count": len(documents),
        "quadruples": sum(r["type"] == "quadruple" for r in results.values()),
        "certificates": sum(r["type"] == "certificate" for r in results.values()),
        "censuses": sum(r["type"] == "census" for r in results.values()),
        "contributions": contributions.collect(results.values()),
        "cells": grid,
        "research_targets": [c for c in grid if c["status"] == "unresolved"],
        "import_tasks": [c for c in grid if c["artifact_status"] == "published-not-imported"],
        "largest_verified": {
            track: max(
                (
                    r["n"]
                    for r in results.values()
                    if r["type"] == "quadruple" and track in r["tracks"]
                ),
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
