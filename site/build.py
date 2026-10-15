#!/usr/bin/env python3
"""Build a deterministic, accessible static board from validated entries."""

import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify.common import ROOT, TRACKS, write_json
from verify.board import load, summary
from verify.seed_inventory import inventory


ORIGIN_LABELS = {
    "participant-result": "Participant work",
    "independent-reproduction": "Independent reproduction",
    "published-import": "Literature / catalogue",
}


def esc(value):
    return html.escape(str(value), quote=True)


def badges(origins):
    return "".join(
        f'<span class="origin {esc(origin)}">{esc(ORIGIN_LABELS[origin])}</span>'
        for origin in ORIGIN_LABELS
        if origin in origins
    )


def people(group):
    return ", ".join(
        f'<a href="https://github.com/{esc(person["github"])}">{esc(person["name"])}</a>'
        if person.get("github")
        else esc(person["name"])
        for person in group["contributors"]
    )


def credit_block(group):
    publication = (
        "Unpublished participant work"
        if group["publication_status"] == "unpublished"
        else "Published result"
    )
    return (
        f'<div class="credit">{badges([group["relation"]])}'
        f'<span class="publication">{publication}</span>'
        f'<p class="byline">{people(group) or "Original authors credited in the sources"}</p>'
        f"<p>{esc(group['summary'])}</p></div>"
    )


def contribution_card(group, results):
    members = [results[ident] for ident in group["entry_ids"]]
    quads = [r for r in members if r["type"] == "quadruple"]
    if quads:
        verification = f"{len(quads)} verified {'class' if len(quads) == 1 else 'classes'} · exact construction check"
    else:
        verification = ", ".join(
            sorted({r.get("tier") or "archived claim · verification incomplete" for r in members})
        )
    links = " ".join(
        f'<a href="entries/{esc(ident)}.html">{esc(ident)}</a>' for ident in group["entry_ids"]
    )
    return (
        f'<article class="card participant-card"><h3>{esc(group["title"])}</h3>'
        f'{credit_block(group)}<p class="verification">{esc(verification)}</p>'
        f"<details><summary>View {len(members)} {'record' if len(members) == 1 else 'records'}</summary>"
        f'<div class="record-links">{links}</div></details></article>'
    )


def shell(title, body, prefix=""):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Goethals–Seidel Challenge</title><meta name="description" content="An exactly verified catalogue of circulant Goethals–Seidel quadruples, research targets, and computational evidence.">
<link rel="stylesheet" href="{prefix}style.css"></head><body>
<header><a class="brand" href="{prefix}index.html">GS / CHALLENGE</a><nav><a href="{prefix}index.html#participants">Participants</a><a href="{prefix}index.html#board">The board</a><a href="{prefix}index.html#evidence">Evidence</a><a href="https://github.com/vxvware/goethals-seidel-challenge">Contribute ↗</a></nav></header>
<main>{body}</main><footer><span>Exact objects. Explicit evidence. Original attribution.</span><span>Cyclic groups · equiv-v1 · <a href="{prefix}stats.json">Open data</a></span></footer></body></html>'''


def build(quick=False):
    d, r, b = load(reproduce=not quick)
    stats = summary(d, r, b)
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "entries").mkdir(exist_ok=True)
    for name in ("style.css", "app.js"):
        shutil.copyfile(ROOT / "site" / name, out / name)
    (out / ".nojekyll").touch()
    write_json(out / "stats.json", stats)
    write_json(out / "entries.json", {"entries": r})
    (out / "badges").mkdir(exist_ok=True)
    for key, label in (("quadruples", "verified quadruples"), ("censuses", "census records")):
        value = stats[key]
        badge = f'''<svg xmlns="http://www.w3.org/2000/svg" width="202" height="24" role="img" aria-label="{label}: {value}">
<title>{label}: {value}</title><rect width="202" height="24" rx="4" fill="#275c49"/>
<path d="M4 0h151v24H4Q0 24 0 20V4Q0 0 4 0" fill="#36413d"/>
<g fill="white" font-family="Verdana,sans-serif" font-size="11" text-anchor="middle">
<text x="78" y="16">{label}</text><text x="178" y="16">{value}</text></g></svg>'''
        (out / f"badges/{key}.svg").write_text(badge)
    manifest = inventory()
    participant_cards = "".join(
        contribution_card(group, r) for group in stats["contributions"] if group["featured"]
    )
    by_cell = {(x["n"], x["stratum"]): x for x in stats["cells"]}
    rows = []
    for n in range(1, 132, 2):
        cells = []
        for track in TRACKS:
            c = by_cell[n, track]
            label = (
                "Verified"
                if c["quadruples"]
                else "Published"
                if c["artifact_status"] == "published-not-imported"
                else "Empty"
                if c["status"] == "empty"
                else "Unresolved"
            )
            small = c["tier"] or (
                "literature"
                if c["status"] == "empty"
                else "claim / partial evidence"
                if c["evidence"]
                else ""
            )
            accessible = f"Order {n}, {track}: {label}; {small}; " + ", ".join(
                ORIGIN_LABELS[o] for o in c["origins"]
            )
            cells.append(
                f'<td class="{esc(c["status"])} {esc(c["artifact_status"])}" data-track="{track}" data-status="{c["status"]}" data-artifact="{c["artifact_status"]}" data-origins="{esc(" ".join(c["origins"]))}"><button data-cell="{n}:{track}" aria-label="{esc(accessible)}">{label}<span class="marker">{esc(small) or "&nbsp;"}</span>{badges(c["origins"])}</button></td>'
            )
        rows.append(f'<tr data-n="{n}"><td>{n}</td>' + "".join(cells) + "</tr>")
    census_rows = []
    for c in sorted(
        (x for x in r.values() if x["type"] == "census"), key=lambda x: (x["stratum"], x["n"])
    ):
        count = str(c["count"]) if c["bound"] == "exact" else f"≥ {c['count']}"
        credits = (
            "<br>".join(badges([g["relation"]]) + " " + people(g) for g in c["contributions"])
            or "Catalogue census"
        )
        census_rows.append(
            f'<tr><td><a href="entries/{c["id"]}.html">{esc(c["stratum"])} / {c["n"]}</a></td><td>{count}</td><td>{esc(c["tier"] or "verified representatives")}</td><td>{credits}</td></tr>'
        )
    evidence_cards = []
    for ident in (
        "best-n73-campaign",
        "good-n65-campaign",
        "good-n41-campaign",
        "good-n67-campaign",
        "skew-n89-subgroup11",
    ):
        result = r[ident]
        title = f"{result['stratum']} · n={result['n']}"
        scope = (
            "Full-stratum claim" if result["scope"]["kind"] == "full" else "Order-11 subgroup only"
        )
        evidence_cards.append(
            f'<article class="card"><span class="tag">{esc(result["tier"] or "archived claim")}</span><h3><a href="entries/{ident}.html">{esc(title)}</a></h3>{"".join(credit_block(g) for g in result["contributions"])}<p>{scope}. {"Partial coverage; no emptiness claim." if result["claim"] == "partial" else "See the evidence record for coverage and limitations."}</p></article>'
        )
    embedded_data = json.dumps({"cells": stats["cells"]}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    body = f"""<section class="hero"><div><div class="eyebrow">An open combinatorial challenge</div><h1>Four rows.<br>A verifiable discovery.</h1><p>A living catalogue of Goethals–Seidel quadruples. Every submitted matrix is checked with exact integers; every computational claim carries its scope and evidence.</p><p><code>AAᵀ + BBᵀ + CCᵀ + DDᵀ = 4nI</code></p></div><aside class="challenge"><div class="eyebrow">Flagship baseline gap</div><strong>356</strong><p>Skew-Hadamard order<br>Quadruple order n = 89</p><small>Cati–Pasechnik baseline · 30 Aug 2025.<br>A verified skew construction fills this historical gap. Present-day novelty requires review.</small></aside></section>
<div class="metrics"><div class="metric"><strong>{stats["quadruples"]}</strong><span>inequivalent verified quadruples</span></div><div class="metric"><strong>{manifest["raw_rows"]:,}</strong><span>raw imported rows checked</span></div><div class="metric"><strong>{stats["certificates"]}</strong><span>proofs & computational records</span></div><div class="metric"><strong>{stats["censuses"]}</strong><span>census records with explicit bounds</span></div></div>
<section id="participants"><div class="eyebrow">Work contributed to the challenge</div><h2>From challenge participants</h2><p>Constructions, searches, and computations produced by participants are credited separately from published results imported into the catalogue.</p><div class="columns">{participant_cards}</div><p class="muted">“Unpublished” means no paper is recorded for this work. Origin and verification are separate: a participant result can be an exactly verified construction, a partial search, or an archived claim. These labels do not certify global priority.</p></section>
<section id="board"><div class="grid-title"><h2>The completeness map</h2><p>Odd orders through 131 · select a cell for sources</p></div><div class="controls"><label>Family <select id="track"><option value="">All six strata</option>{"".join(f'<option value="{x}">{x}</option>' for x in TRACKS)}</select></label><label>Show <select id="status"><option value="">Every status</option><option value="unresolved">Research targets</option><option value="published-not-imported">Import tasks</option><option value="exists">Existence known</option><option value="empty">Nonexistence</option></select></label><label>Origin <select id="origin"><option value="">Every origin</option><option value="participant-result">Challenge participants</option><option value="independent-reproduction">Independent reproductions</option><option value="published-import">Literature / catalogue</option></select></label><label>Through n <select id="max-order"><option>31</option><option>63</option><option selected>131</option></select></label></div>
<div class="legend"><span class="known">Verified on board</span><span class="published">Published, awaiting import</span><span class="empty">Empty, with stated evidence</span><span class="unresolved">Unresolved in baseline</span><span class="participant">Purple label: participant work</span></div>
<div class="scroll"><table id="grid"><thead><tr><th scope="col">n</th>{"".join(f'<th scope="col" data-track="{x}">{x}</th>' for x in TRACKS)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><div id="cell-detail" class="detail" aria-live="polite" hidden></div><p class="muted">An unfilled artifact is not automatically an open problem. Scoped nulls and partial searches do not close a whole cell. “kkss” is the G-matrix stratum.</p></section>
<section id="evidence"><h2>Evidence, with its limits</h2><p>Existence is rechecked exactly. Proof rules, complete replays, sampled replays, and receipt checks have distinct labels. Historical records with missing coverage identity remain archived claims.</p><div class="columns">{"".join(evidence_cards)}</div><p class="note">No GPU spot checks ran during this import. The best-73 legacy identity gap and the missing early good-65 receipt chain are preserved explicitly. Literature citations are attributed results, not locally checked proofs.</p></section>
<section><h2>Count the classes</h2><p>The primary convention includes independent reversals. At n=43, the same 10,080 best quadruples form five primary classes and 20 legacy orbits.</p><div class="scroll"><table><thead><tr><th>Family / order</th><th>Classes</th><th>Verification</th><th>Origin / credit</th></tr></thead><tbody>{"".join(census_rows)}</tbody></table></div></section>
<section class="columns"><article class="card"><h3>Bring a construction</h3><p>Submit four defining rows or negative-index sets. The verifier derives the family, checks the PAF identity, and detects equivalent entries.</p><p><code>./gs verify candidate.json</code><br><code>./gs submit candidate.json</code></p></article><article class="card"><h3>Keep the attribution</h3><p>Original discoverers and submission authors are recorded separately. Importing a published matrix earns importer credit. Baseline gaps never grant discovery credit automatically.</p><a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/CONTRIBUTING.md">Contribution guide ↗</a></article></section>
<noscript><p>JavaScript enables filters and cell details. All data remain available in <a href="stats.json">stats.json</a> and <a href="entries.json">entries.json</a>.</p></noscript>
<script id="board-data" type="application/json">{embedded_data}</script><script src="app.js" defer></script>"""
    (out / "index.html").write_text(shell("The board", body))
    for ident, doc in d.items():
        result = r[ident]
        sources = "".join(
            f'<li><a href="{esc(s["url"])}">{esc(s["title"])}</a></li>' for s in doc["sources"]
        )
        credits = "".join(credit_block(group) for group in result["contributions"])
        if not credits:
            label = (
                "Published result"
                if doc.get("method") == "literature"
                else "Published theorem"
                if doc.get("method") == "row-sum-theorem"
                else "Origin not yet classified"
            )
            credits = f'<p class="origin published-import">{label}</p>'
        text = f"""<article class="entry"><div class="eyebrow">{esc(doc["type"])} · order {doc["n"]}</div><h1>{esc(ident)}</h1><span class="tag">{esc(result.get("tier") or "attributed / archived claim or lower bound")}</span>{credits}<p>Original source attribution: {esc("; ".join(doc["discovered_by"]))}<br>Submitted / imported by: {esc(doc["submitted_by"])} · {doc["submitted_at"]}</p><h2>Checks and result</h2><pre>{esc(json.dumps(result, indent=2, ensure_ascii=False))}</pre><h2>Sources</h2><ul>{sources}</ul><details><summary>Submission JSON</summary><pre>{esc(json.dumps(doc, indent=2, ensure_ascii=False))}</pre></details></article>"""
        (out / f"entries/{ident}.html").write_text(shell(ident, text, "../"))
    print(
        f"Built docs/: {stats['quadruples']} quadruples, {stats['certificates']} evidence records, {stats['censuses']} censuses."
    )
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    build(a.quick)
