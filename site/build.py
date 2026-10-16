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
    "participant-result": "Challenge result",
    "independent-reproduction": "Independent reproduction",
    "published-import": "Literature / catalogue",
}


TRACK_LABELS = {
    "williamson": "Williamson (ssss)",
    "good": "Good (ksss)",
    "best": "Best (kkks)",
    "kkss": "G-matrices (kkss)",
    "skew-sds": "Skew (any k)",
    "general": "General (any)",
}


PRIMARY = ("williamson", "good", "best", "kkss")


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
        "Not yet published" if group["publication_status"] == "unpublished" else "Published result"
    )
    return (
        f'<div class="credit">{badges([group["relation"]])}'
        f'<span class="publication">{publication}</span>'
        f'<p class="byline">{people(group) or "Original authors credited in the sources"}</p>'
        f"<p>{esc(group['summary'])}</p></div>"
    )


def shell(title, body, prefix=""):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Goethals–Seidel Challenge</title><meta name="description" content="An exactly verified catalogue of circulant Goethals–Seidel quadruples, research targets, and computational evidence.">
<link rel="icon" href="{prefix}favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="{prefix}style.css"></head><body>
<header><a class="brand" href="{prefix}index.html">GS / CHALLENGE</a><nav><a href="{prefix}index.html#findings">Findings</a><a href="{prefix}index.html#board">The board</a><a href="https://github.com/vxvware/goethals-seidel-challenge">Contribute ↗</a></nav></header>
<main>{body}</main><footer><span>Goethals–Seidel Challenge · vxvware</span><span>Cyclic groups · equiv-v1 · <a href="{prefix}stats.json">Open data</a></span></footer></body></html>'''


def cell_list(cells):
    """Links to cell pages, grouped by track: Good 53, 55 · G-matrices 51."""
    groups = []
    for track in PRIMARY:
        orders = sorted(c["n"] for c in cells if c["stratum"] == track)
        if orders:
            links = ", ".join(f'<a href="cells/n{n}-{track}.html">{n}</a>' for n in orders)
            groups.append(f"{esc(TRACK_LABELS[track].split(' (')[0])} {links}")
    return " · ".join(groups)


def findings(cells, results):
    """What the board has established, computed from the cells of the four types."""
    primary = [c for c in cells if c["stratum"] in PRIMARY]
    corrected = [c for c in primary if c["contradicted"]]
    new = [
        c
        for c in primary
        if c["status"] == "exists"
        and c["quadruples"]
        and not c["contradicted"]
        and "participant-result" in c["origins"]
        and "published-import" not in c["origins"]
        and not c["sources"]
    ]
    reported_empty = [c for c in primary if c["empty_basis"] == ["unreplayed-census"]]
    reproduced = [c for c in primary if "independent-reproduction" in c["origins"]]
    imported = [c for c in primary if c["quadruples"] and "published-import" in c["origins"]]
    imported_quads = {
        q
        for c in imported
        for q in c["quadruples"]
        if any(g["relation"] == "published-import" for g in results[q]["contributions"])
    }
    items = []
    for c in corrected:
        claim = c["contradicted"][0]["source"]
        count = len(c["quadruples"])
        items.append(
            f'<div class="finding"><strong>A published nonexistence corrected</strong>'
            f'<p><a href="cells/n{c["n"]}-{c["stratum"]}.html">{esc(TRACK_LABELS[c["stratum"]])} at n = {c["n"]}</a>: '
            f'<a href="{esc(claim["url"])}" rel="noopener">{esc(claim["title"])}</a> reports none; '
            f"the board holds {count} exactly verified {'class' if count == 1 else 'classes'}.</p></div>"
        )
    if new:
        items.append(
            f'<div class="finding"><strong>Existence at orders with no published result</strong>'
            f"<p>{len(new)} cells with exactly verified rows where the sourced baseline records no "
            f"published construction: {cell_list(new)}.</p></div>"
        )
    if reported_empty:
        items.append(
            f'<div class="finding"><strong>Nonexistence reported, not yet replayed</strong>'
            f"<p>Complete censuses run for the challenge found nothing at {cell_list(reported_empty)}. "
            f"They stay attributed claims until the board replays them.</p></div>"
        )
    if reproduced:
        items.append(
            f'<div class="finding"><strong>Published results reproduced by an independent program</strong>'
            f"<p>{len(reproduced)} cells recomputed with the challenge code and found to agree with the "
            f"published tables: {cell_list(reproduced)}.</p></div>"
        )
    campaigns = sorted(
        (
            x
            for x in results.values()
            if x["type"] == "certificate"
            and x.get("provenance") != "unreplayed-census"
            and any(g["relation"] != "published-import" for g in x["contributions"])
        ),
        key=lambda x: (x["stratum"], x["n"]),
    )
    if campaigns:
        parts = []
        for x in campaigns:
            limits = [x.get("tier") or "archived claim"]
            if x["claim"] == "partial":
                limits.append("partial coverage")
            if x["scope"]["kind"] != "full":
                limits.append("one subgroup family only")
            parts.append(
                f'<a href="entries/{esc(x["id"])}.html">{esc(TRACK_LABELS[x["stratum"]].split(" (")[0])} {x["n"]}</a> '
                f"({esc(', '.join(limits))})"
            )
        items.append(
            f'<div class="finding"><strong>Search campaigns with stated limits</strong>'
            f"<p>Nonexistence searches run for the challenge whose evidence is incomplete or scoped; "
            f"none closes a cell on its own: {' · '.join(parts)}.</p></div>"
        )
    if imported:
        items.append(
            f'<div class="finding"><strong>Published constructions rechecked exactly</strong>'
            f"<p>{len(imported_quads)} inequivalent quadruples imported from papers and catalogues, "
            f"each rechecked with exact integers, across {len(imported)} cells of the four types.</p></div>"
        )
    return "".join(items)


def quad_figure(doc, result, caption):
    """A figure that picture.js paints from the embedded canonical rows."""
    payload = json.dumps(
        {"n": doc["n"], "rows": result["canonical_rows"], "pattern": result["pattern"]},
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    return (
        '<figure class="quad"><canvas class="rows" aria-label="Four defining rows"></canvas>'
        '<canvas class="array" aria-label="Goethals–Seidel array"></canvas>'
        f'<script type="application/json">{payload}</script>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def cell_links(result, prefix):
    tracks = result.get("tracks", [result.get("stratum")])
    return ", ".join(
        f'<a href="{prefix}cells/n{result["n"]}-{esc(track)}.html">{esc(TRACK_LABELS[track])}</a>'
        for track in tracks
        if track in TRACK_LABELS
    )


def cell_page(c, d, r):
    """One page per order and track: status, credit, sources, and the matrices."""
    n, track = c["n"], c["stratum"]
    status = " · ".join(x for x in (c["status"], c["artifact_status"], c["tier"]) if x)
    parts = [
        f'<article class="entry"><div class="eyebrow">cell · order {n} · {esc(track)}</div>'
        f'<h1>n = {n} · {esc(TRACK_LABELS[track])}</h1><span class="tag">{esc(status)}</span>'
    ]
    if c["note"]:
        parts.append(f"<p>{esc(c['note'])}</p>")
    for claim in c["contradicted"]:
        parts.append(
            '<p class="contradiction">Exactly verified rows on this board contradict a published '
            f'nonexistence claim: <a href="{esc(claim["source"]["url"])}" rel="noopener">'
            f"{esc(claim['source']['title'])}</a></p>"
        )
    parts.extend(credit_block(group) for group in c["contributions"])
    if c["quadruples"]:
        count = len(c["quadruples"])
        parts.append(
            f"<h2>{count} verified {'class' if count == 1 else 'classes'}</h2>"
            "<p>Each figure shows the four canonical rows and the Goethals–Seidel array they "
            f"generate, a Hadamard matrix of order {4 * n}. Skew blocks are green and symmetric "
            "blocks blue; dark is +1 and light is −1. The drawing is computed from the rows in the "
            'browser; open an entry for the verifier result.</p><div class="gallery">'
        )
        for ident in c["quadruples"]:
            result = r[ident]
            legend = " ".join(
                f"{name}&thinsp;{kind}" for name, kind in zip("ABCD", result["pattern"])
            )
            caption = (
                f'<a href="../entries/{esc(ident)}.html">{esc(ident)}</a>'
                f"<br>{legend} · {esc(result.get('tier') or 'verified')}"
            )
            parts.append(quad_figure(d[ident], result, caption))
        parts.append("</div>")
    records = [*c["evidence"], *c["censuses"]]
    if records:
        items = []
        for ident in records:
            result = r[ident]
            detail = result["type"]
            if result["type"] == "census":
                bound = "" if result["bound"] == "exact" else "≥ "
                detail += (
                    f" · {bound}{result['count']} {'class' if result['count'] == 1 else 'classes'}"
                )
            if result.get("tier"):
                detail += f" · {result['tier']}"
            items.append(
                f'<li><a href="../entries/{esc(ident)}.html">{esc(ident)}</a> · {esc(detail)}</li>'
            )
        parts.append(f"<h2>Evidence and census records</h2><ul>{''.join(items)}</ul>")
    if c["sources"]:
        parts.append(
            "<h2>Sources</h2><ul>"
            + "".join(
                f'<li><a href="{esc(s["source"]["url"])}" rel="noopener">{esc(s["source"]["title"])}</a></li>'
                for s in c["sources"]
            )
            + "</ul>"
        )
    if c["status"] == "unresolved":
        parts.append(
            "<p>Unresolved in the sourced baseline. Review the literature before claiming novelty.</p>"
        )
    if c["evidence"]:
        parts.append(
            "<p>Open the evidence records for scope, coverage, gaps, and the checks that actually ran.</p>"
        )
    parts.append('<p><a href="../index.html#board">Back to the board</a></p></article>')
    if c["quadruples"]:
        parts.append('<script src="../picture.js" defer></script>')
    return "".join(parts)


def build(quick=False):
    d, r, b = load(reproduce=not quick)
    stats = summary(d, r, b)
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "entries").mkdir(exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    for name in ("style.css", "app.js", "picture.js", "favicon.svg"):
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
    by_cell = {(x["n"], x["stratum"]): x for x in stats["cells"]}
    rows = []
    for n in range(1, 132, 2):
        cells = []
        for track in TRACKS:
            c = by_cell[n, track]
            label = (
                "Corrects published"
                if c["contradicted"]
                else "Verified"
                if c["quadruples"]
                else "Published"
                if c["artifact_status"] == "published-not-imported"
                else "Empty"
                if c["status"] == "empty"
                else "Unresolved"
            )
            small = c["tier"] or (
                " / ".join(c["empty_basis"]) or "literature"
                if c["status"] == "empty"
                else "claim / partial evidence"
                if c["evidence"]
                else ""
            )
            accessible = f"Order {n}, {TRACK_LABELS[track]}: {label}; {small}; " + ", ".join(
                ORIGIN_LABELS[o] for o in c["origins"]
            )
            classes = f"{c['status']} {c['artifact_status']}{' contradicted' if c['contradicted'] else ''}"
            marker = (
                '<span class="participant" title="Challenge result"></span>'
                if "participant-result" in c["origins"]
                else ""
            )
            cells.append(
                f'<td class="{esc(classes)}" data-track="{track}" data-status="{c["status"]}" data-artifact="{c["artifact_status"]}" data-origins="{esc(" ".join(c["origins"]))}"><a class="cell" href="cells/n{n}-{track}.html" aria-label="{esc(accessible)}">{label}{marker}</a></td>'
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
    body = f"""<section class="hero"><div><div class="eyebrow">An open combinatorial challenge</div><h1>The Goethals–Seidel Challenge</h1><p>A verified catalogue of circulant Goethals–Seidel quadruples. Every submitted matrix is checked with exact integers, and every computational claim carries its scope and evidence.</p><p><code>AAᵀ + BBᵀ + CCᵀ + DDᵀ = 4nI</code></p></div><aside class="challenge"><div class="eyebrow">Flagship baseline gap</div><strong>356</strong><p>Skew-Hadamard order<br>Quadruple order n = 89</p><small>Cati–Pasechnik baseline · 30 Aug 2025.<br>A verified skew construction fills this historical gap. Present-day novelty requires review.</small></aside></section>
<div class="metrics"><div class="metric"><strong>{stats["quadruples"]}</strong><span>inequivalent verified quadruples</span></div><div class="metric"><strong>{manifest["raw_rows"]:,}</strong><span>raw imported rows checked</span></div><div class="metric"><strong>{stats["certificates"]}</strong><span>proofs & computational records</span></div><div class="metric"><strong>{stats["censuses"]}</strong><span>census records with explicit bounds</span></div></div>
<section id="findings"><h2>What the board has established</h2><div class="findings">{findings(stats["cells"], r)}</div><p class="muted">Computed from the cells below. Existence claims rest on exactly rechecked rows; counts and nonexistence rest on the evidence each cell cites.</p></section>
<section id="board"><div class="grid-title"><h2>The completeness map</h2><p>Odd orders through 131 · open a cell for its tier, sources, and matrices</p></div><p class="key">Each column is a symmetry type of the four circulant blocks A, B, C, D: <strong>s</strong> symmetric, <strong>k</strong> skew, in any order. Skew means at least one skew block, so it contains the good, best, and G-matrix columns; general is any verified quadruple. Skew types give skew-Hadamard matrices of order 4n.</p><div class="controls"><label>Family <select id="track"><option value="">All six strata</option>{"".join(f'<option value="{x}">{esc(TRACK_LABELS[x])}</option>' for x in TRACKS)}</select></label><label>Show <select id="status"><option value="">Every status</option><option value="unresolved">Research targets</option><option value="published-not-imported">Import tasks</option><option value="exists">Existence known</option><option value="empty">Nonexistence</option></select></label><label>Origin <select id="origin"><option value="">Every origin</option><option value="participant-result">Challenge results</option><option value="independent-reproduction">Independent reproductions</option><option value="published-import">Literature / catalogue</option></select></label><label>Through n <select id="max-order"><option>31</option><option>63</option><option selected>131</option></select></label></div>
<div class="legend"><span class="known">Verified on board</span><span class="published">Published, awaiting import</span><span class="empty">Empty, with stated evidence</span><span class="unresolved">Unresolved in baseline</span><span class="contradicted">Verified rows contradict a published nonexistence</span><span class="participant">Purple mark: challenge result</span></div>
<div class="scroll"><table id="grid"><thead><tr><th scope="col">n</th>{"".join(f'<th scope="col" data-track="{x}">{esc(TRACK_LABELS[x])}</th>' for x in TRACKS)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class="muted">An unfilled artifact is not automatically an open problem. Scoped nulls and partial searches do not close a whole cell. “kkss” is the G-matrix stratum.</p></section>
<section><h2>Count the classes</h2><p>The primary convention includes independent reversals. At n=43, the same 10,080 best quadruples form five primary classes and 20 legacy orbits.</p><div class="scroll"><table><thead><tr><th>Family / order</th><th>Classes</th><th>Verification</th><th>Origin / credit</th></tr></thead><tbody>{"".join(census_rows)}</tbody></table></div></section>
<section class="columns"><article class="card"><h3>Bring a construction</h3><p>Submit four defining rows or negative-index sets. The verifier derives the family, checks the PAF identity, and detects equivalent entries.</p><p><code>./gs verify candidate.json</code><br><code>./gs submit candidate.json</code></p></article><article class="card"><h3>Keep the attribution</h3><p>Original discoverers and submission authors are recorded separately. Importing a published matrix earns importer credit. Baseline gaps never grant discovery credit automatically.</p><a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/CONTRIBUTING.md">Contribution guide ↗</a></article></section>
<noscript><p>JavaScript enables the filters and the matrix drawings. All data remain available in <a href="stats.json">stats.json</a> and <a href="entries.json">entries.json</a>.</p></noscript>
<script src="app.js" defer></script>"""
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
        picture = ""
        if doc["type"] == "quadruple":
            legend = " · ".join(f"{name} {kind}" for name, kind in zip("ABCD", result["pattern"]))
            picture = (
                f'<section class="picture"><h2>Picture</h2>'
                f"<p>The four canonical rows, one row of cells per block ({esc(legend)}; k skew, s symmetric), "
                f"and the Goethals–Seidel array they generate: a Hadamard matrix of order {4 * doc['n']}. "
                f"Skew blocks are green and symmetric blocks blue; dark is +1 and light is −1. "
                f"The drawing is computed from the rows in the browser; "
                f"the verifier result below is the evidence.</p>"
                f"{quad_figure(doc, result, '')}"
                f'<script src="../picture.js" defer></script></section>'
            )
        text = f"""<article class="entry"><div class="eyebrow">{esc(doc["type"])} · order {doc["n"]}</div><h1>{esc(ident)}</h1><span class="tag">{esc(result.get("tier") or "attributed / archived claim or lower bound")}</span>{credits}<p>Original source attribution: {esc("; ".join(doc["discovered_by"]))}<br>Submitted / imported by: {esc(doc["submitted_by"])} · {doc["submitted_at"]}<br>Board cell: {cell_links(result, "../")}</p>{picture}<h2>Checks and result</h2><pre>{esc(json.dumps(result, indent=2, ensure_ascii=False))}</pre><h2>Sources</h2><ul>{sources}</ul><details><summary>Submission JSON</summary><pre>{esc(json.dumps(doc, indent=2, ensure_ascii=False))}</pre></details></article>"""
        (out / f"entries/{ident}.html").write_text(shell(ident, text, "../"))
    for c in stats["cells"]:
        title = f"n = {c['n']} · {TRACK_LABELS[c['stratum']]}"
        (out / f"cells/n{c['n']}-{c['stratum']}.html").write_text(
            shell(title, cell_page(c, d, r), "../")
        )
    print(
        f"Built docs/: {stats['quadruples']} quadruples, {stats['certificates']} evidence records, {stats['censuses']} censuses."
    )
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    build(a.quick)
