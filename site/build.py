#!/usr/bin/env python3
"""Build a deterministic, accessible static board from validated entries."""

import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify.common import ROOT, TRACKS, write_json
from verify.board import load, summary
from references import Bibliography

BIB = Bibliography()


# Lean theorems about specific entries, linked from the entry pages.
FORMAL = json.loads((ROOT / "baselines/formal.json").read_text())["theorems"]

ORIGIN_LABELS = {
    "participant-result": "Challenge result",
    "independent-reproduction": "Independent reproduction",
    "published-import": "Literature / catalog",
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

KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/"
MATH_HEAD = (
    f'<link rel="stylesheet" href="{KATEX}katex.min.css">'
    f'<script defer src="{KATEX}katex.min.js"></script>'
    f'<script defer src="{KATEX}contrib/auto-render.min.js" '
    "onload=\"renderMathInElement(document.body,{delimiters:[{left:'\\\\[',right:'\\\\]',display:true},"
    "{left:'\\\\(',right:'\\\\)',display:false}]})\"></script>"
)
# The array below is the one picture.js draws. It differs from the usual
# textbook layout only in the sign convention of the transposed blocks; both
# satisfy H H^T = 4n I, and both are verified numerically in the tests.
DEFINITION = (
    r"""<aside class="definition"><div class="eyebrow">Definition</div>
<p>Let \(A, B, C, D\) be circulant \(n \times n\) matrices with entries in \(\{1, -1\}\). Write \(X^{\mathsf T}\) for the transpose of \(X\) and \(I_n\) for the \(n \times n\) identity matrix. The four form a Goethals–Seidel quadruple when</p>
\[AA^{\mathsf T} + BB^{\mathsf T} + CC^{\mathsf T} + DD^{\mathsf T} = 4nI_n.\]
<p>A block \(X\) is symmetric when \(X^{\mathsf T} = X\) and skew when \(X^{\mathsf T} = 2I_n - X\). With \(R\) the \(n \times n\) back-diagonal permutation matrix, the Goethals–Seidel array</p>
\[H = \begin{pmatrix} A & BR & CR & DR \\ -BR & A & -D^{\mathsf T}R & C^{\mathsf T}R \\ -CR & D^{\mathsf T}R & A & -B^{\mathsf T}R \\ -DR & -C^{\mathsf T}R & B^{\mathsf T}R & A \end{pmatrix}\]
<p>satisfies \(HH^{\mathsf T} = 4nI_{4n}\), so it is a Hadamard matrix of order \(4n\), and a skew-Hadamard matrix whenever \(A\) is skew. The array is due to Goethals and Seidel """
    + BIB.mark("goethals1970skew")
    + ".</p></aside>"
)


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


def source_keys(sources):
    """Bibliography keys of the literature among a list of source objects."""
    return [key for s in sources if (key := BIB.resolve(s["url"]))]


def source_item(s, prefix):
    """One source as a list item: its link, then the number of its reference."""
    key = BIB.resolve(s["url"])
    mark = f" {BIB.mark(key, prefix)}" if key else ""
    return (
        f'<li><a href="{esc(BIB.link(s["url"]))}" rel="noopener">{esc(s["title"])}</a>{mark}</li>'
    )


def references_block(keys, prefix):
    """The full references a page cites, numbered as in the board's reference list."""
    listing = BIB.listing(keys, prefix)
    return f"<h2>References</h2>{listing}" if listing else ""


def credit_block(group, prefix=""):
    publication = (
        "Not yet published" if group["publication_status"] == "unpublished" else "Published result"
    )
    cited = list(dict.fromkeys(BIB.keys_in(group["summary"])))
    authors = "; ".join(f"{esc(BIB.short(key))} {BIB.mark(key, prefix)}" for key in cited)
    return (
        f'<div class="credit">{badges([group["relation"]])}'
        f'<span class="publication">{publication}</span>'
        f'<p class="byline">{people(group) or authors or "Original authors as cited below"}</p>'
        f"<p>{BIB.cite(group['summary'], prefix)}</p></div>"
    )


def shell(title, body, prefix="", head=""):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Goethals–Seidel Challenge</title><meta name="description" content="An exactly verified catalog of circulant Goethals–Seidel quadruples, research targets, and computational evidence.">
<link rel="icon" href="{prefix}favicon.png" type="image/png"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"><link rel="stylesheet" href="{prefix}style.css">{head}</head><body>
<header><a class="brand" href="{prefix}index.html">gs-challenge</a><nav><a href="{prefix}index.html#board">board</a><a href="{prefix}index.html#new-results">new results</a><a href="{prefix}references.html">references</a><a href="https://github.com/vxvware/goethals-seidel-challenge">contribute</a></nav></header>
<main>{body}</main><footer><a class="logo" href="https://vxvware.com" aria-label="vxvware"><img src="{prefix}vxvware-logo.png" alt="vxvware"></a></footer></body></html>'''


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


# The quadruples the front-page animation cycles through: the first entry, by
# id, of each (track, order) below. Small orders keep the array legible.
HERO_CELLS = (
    ("williamson", 13),
    ("good", 15),
    ("best", 13),
    ("kkss", 19),
    ("skew-sds", 11),
    ("general", 9),
    ("williamson", 25),
    ("good", 23),
)


def hero_figure(r):
    quads = []
    for track, n in HERO_CELLS:
        ident = min(
            k
            for k, x in r.items()
            if x["type"] == "quadruple" and x["n"] == n and x["tracks"][0] == track
        )
        x = r[ident]
        quads.append(
            {
                "id": ident,
                "n": n,
                "rows": x["canonical_rows"],
                "pattern": x["pattern"],
                "label": TRACK_LABELS[track].lower(),
            }
        )
    payload = json.dumps(quads, separators=(",", ":")).replace("<", "\\u003c")
    return (
        '<figure class="hero-anim"><canvas aria-label="A Goethals–Seidel array being filled in">'
        '</canvas><div><pre aria-hidden="true"></pre><a href="#board"></a>'
        '<p class="muted">In every matrix picture skew blocks are green, symmetric blocks blue '
        "and blocks with neither symmetry grey; light is +1 and dark is −1.</p></div>"
        f'<script type="application/json">{payload}</script></figure>'
    )


def cell_links(result, prefix):
    tracks = result.get("tracks", [result.get("stratum")])
    return ", ".join(
        f'<a href="{prefix}cells/n{result["n"]}-{esc(track)}.html">{esc(TRACK_LABELS[track])}</a>'
        for track in tracks
        if track in TRACK_LABELS
    )


def published_count(x):
    """A published complete classification whose every class is stored here.

    The verifier has not rerun the search, so the record's bound stays lower,
    but the page states the published count rather than a bound.
    """
    return (
        x["bound"] != "exact"
        and x["claim_kind"] == "published-exact"
        and x["count"] == x["claimed_count"]
        and x["count"] > 0
    )


def census_count(c, r):
    """The class count a cell's census record states, as '2 classes', or '' without one."""
    if not c["censuses"]:
        return ""
    x = r[c["censuses"][0]]
    if x["bound"] != "exact" and x["count"] == 0:
        return ""
    if x["bound"] != "exact" and not published_count(x):
        # The search behind this record was not exhaustive, so there is no
        # count to state: the grid shows existence and the cell page says how
        # many classes are known.
        return ""
    return f"{x['count']} {'class' if x['count'] == 1 else 'classes'}"


# Cells whose statement is not in the published literature. The list is curated
# because a cell's origins only say who supplied the rows: an exact class count
# at an order whose existence was already published still counts here. The
# wording is derived from the live board, so a statement cannot outrun its
# evidence.
NEW_RESULTS = (
    ("williamson", 61),
    ("williamson", 65),
    ("good", 53),
    ("good", 55),
    ("good", 59),
    ("good", 61),
    ("good", 67),
    ("kkss", 49),
    ("kkss", 51),
    ("kkss", 53),
    ("kkss", 55),
    ("kkss", 57),
    ("kkss", 59),
    ("kkss", 61),
)
# The grid marks these cells as new alongside the cells a participant supplied
# rows for. The shift-class counts at 3 to 13 are not listed: no literature
# check has established that they were unknown.
NEW_CELLS = set(NEW_RESULTS)
NEW_RESULT_NAMES = {"williamson": "Williamson ", "good": "Good ", "kkss": "G-"}
NEW_RESULT_GROUPS = (
    ("correction", "Correction to a published result"),
    ("classification", "Classifications"),
    ("nonexistence", "Nonexistence"),
)


def new_results(stats, r):
    """The curated new results, grouped, each worded from its cell's evidence."""
    by_cell = {(x["n"], x["stratum"]): x for x in stats["cells"]}
    groups = {key: [] for key, _ in NEW_RESULT_GROUPS}
    for track, n in NEW_RESULTS:
        c = by_cell[n, track]
        item = {"n": n, "stratum": track, "source": None, "qualifier": None}
        subject = f"{NEW_RESULT_NAMES[track]}matrices of order {n}"
        census = r[c["censuses"][0]] if c["censuses"] else None
        if c["status"] == "exists" and census and census["count"]:
            noun = "class" if census["count"] == 1 else "classes"
            exact = census["bound"] == "exact"
            text = f"{subject}: {'exactly' if exact else 'at least'} {census['count']} {noun}."
            kind = "classification"
        elif c["status"] == "empty":
            # The statement is the same whether or not the search has been
            # rerun here; a single unreplayed search is noted on the item.
            text = f"{subject}: none exist."
            if "reproduced" not in c["empty_basis"]:
                item["qualifier"] = "one exhaustive search, not yet rerun"
            kind = "nonexistence"
        else:
            raise ValueError(f"new result without a census or an empty cell: {track} n={n}")
        if c["contradicted"]:
            kind = "correction"
            item["source"] = c["contradicted"][0]["source"]
        groups[kind].append(item | {"text": text})
    return [(title, groups[key]) for key, title in NEW_RESULT_GROUPS if groups[key]]


def new_results_section(stats, r):
    parts = [
        '<section id="new-results"><h2>New results</h2><p class="key">Statements on this board '
        "that are not in the published literature. Counts are equivalence classes. Each "
        "statement comes from an exhaustive computer search, not from a proof.</p>"
    ]
    for title, items in new_results(stats, r):
        parts.append(f"<h3>{esc(title)}</h3><ul>")
        for x in items:
            link = f'<a href="cells/n{x["n"]}-{x["stratum"]}.html">{esc(x["text"])}</a>'
            if x["source"]:
                url = x["source"]["url"]
                key = BIB.resolve(url)
                if not key:
                    raise ValueError(f"contradicted source is not in references.bib: {url}")
                link += (
                    " A published exhaustive search reports none "
                    f'(<a href="{esc(url)}" rel="noopener">{esc(BIB.short(key))}</a> '
                    f"{BIB.mark(key)})."
                )
            if x["qualifier"]:
                link += f' <span class="muted">({esc(x["qualifier"])})</span>'
            parts.append(f"<li>{link}</li>")
        parts.append("</ul>")
    parts.append("</section>")
    return "".join(parts)


def cell_page(c, d, r):
    """One page per order and track: status, credit, sources, and the matrices."""
    n, track = c["n"], c["stratum"]
    status = " · ".join(x for x in (c["status"], c["artifact_status"], c["tier"]) if x)
    parts = [
        f'<article class="entry"><div class="eyebrow">cell · order {n} · {esc(track)}</div>'
        f'<h1>n = {n} · {esc(TRACK_LABELS[track])}</h1><span class="tag">{esc(status)}</span>'
    ]
    if c["censuses"]:
        x = r[c["censuses"][0]]
        link = f'<a href="../entries/{esc(x["id"])}.html">{esc(x["id"])}</a>'
        if x["bound"] != "exact" and x["count"] == 0:
            parts.append(
                f'<p class="census">published empty census · awaiting a full replay · {link}</p>'
            )
        else:
            kind = (
                "complete census"
                if x["bound"] == "exact"
                else "published classification, not replayed here"
                if published_count(x)
                else "the search was not exhaustive"
            )
            if x["equivalence"] == "equiv-v2":
                kind += " under equiv-v2, which also identifies independent cyclic shifts"
            if x.get("count_only"):
                kind += " · a count only, the classes are not stored"
            stated = census_count(c, r) or (
                f"{x['count']} {'class' if x['count'] == 1 else 'classes'} known"
            )
            parts.append(
                f'<p class="census"><strong>{esc(stated)}</strong> · {kind} · '
                f"{esc(x['tier'] or 'verified representatives')} · {link}</p>"
            )
    if c["note"]:
        parts.append(f"<p>{esc(c['note'])}</p>")
    for claim in c["contradicted"]:
        parts.append(
            '<p class="contradiction">Exactly verified rows on this board contradict a published '
            f'nonexistence claim: <a href="{esc(BIB.link(claim["source"]["url"]))}" rel="noopener">'
            f"{esc(claim['source']['title'])}</a>"
            + "".join(f" {BIB.mark(key, '../')}" for key in source_keys([claim["source"]]))
            + "</p>"
        )
    # The catalogue credit is the same paragraph for every imported cell; the
    # cell's own sources already name the original authors.
    parts.extend(
        credit_block(group, "../")
        for group in c["contributions"]
        if group["id"] != "published-catalogues"
    )
    if c["quadruples"]:
        count = len(c["quadruples"])
        heading = (
            "Matrices"
            if c["censuses"]
            else f"{count} verified {'class' if count == 1 else 'classes'}"
        )
        parts.append(f'<h2>{heading}</h2><div class="gallery">')
        for ident in c["quadruples"]:
            result = r[ident]
            caption = f'<a href="../entries/{esc(ident)}.html">{esc(ident)}</a>'
            parts.append(quad_figure(d[ident], result, caption))
        parts.append("</div>")
    records = [*c["evidence"], *c["censuses"]]
    if records:
        items = []
        for ident in records:
            result = r[ident]
            detail = result["type"]
            if result["type"] == "census" and result["bound"] != "exact" and result["count"] == 0:
                detail += " · published empty, awaiting a full replay"
            elif result["type"] == "census":
                known = "" if result["bound"] == "exact" or published_count(result) else " known"
                detail += (
                    f" · {result['count']} {'class' if result['count'] == 1 else 'classes'}{known}"
                )
                if result["equivalence"] == "equiv-v2":
                    detail += " · equiv-v2 (shifts identified)"
                if published_count(result):
                    detail += " · published count"
            if result.get("tier"):
                detail += f" · {result['tier']}"
            items.append(
                f'<li><a href="../entries/{esc(ident)}.html">{esc(ident)}</a> · {esc(detail)}</li>'
            )
        parts.append(f"<h2>Evidence and census records</h2><ul>{''.join(items)}</ul>")
    if c["sources"]:
        parts.append(
            "<h2>Sources</h2><ul>"
            + "".join(source_item(s["source"], "../") for s in c["sources"])
            + "</ul>"
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
    for name in (
        "style.css",
        "app.js",
        "picture.js",
        "hero.js",
        "favicon.png",
        "vxvware-logo.png",
    ):
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
            origins = list(c["origins"])
            if (track, n) in NEW_CELLS and "participant-result" not in origins:
                origins.insert(0, "participant-result")
            # The diamond means a new result for this cell. A cell can hold
            # rows or evidence from the challenge without that: a new G-matrix
            # is also a skew quadruple at an order where those were known.
            marker = (
                '<span class="participant" title="New result"></span>'
                if (track, n) in NEW_CELLS
                else ""
            )
            count = census_count(c, r)
            if count:
                accessible += f"; {count}"
            text = (
                count.replace(" classes", "").replace(" class", "")
                if count
                else {"Published": "pub", "Empty": "–", "Unresolved": "?"}.get(label, "✓")
            )
            cells.append(
                f'<td class="{esc(classes)}" data-track="{track}" data-status="{c["status"]}" data-artifact="{c["artifact_status"]}" data-origins="{esc(" ".join(origins))}"><a class="cell" href="cells/n{n}-{track}.html" aria-label="{esc(accessible)}">{esc(text)}{marker}</a></td>'
            )
        rows.append(f'<tr data-n="{n}"><td>{n}</td>' + "".join(cells) + "</tr>")
    body = f"""<section class="hero"><div><div class="eyebrow">An open combinatorial challenge</div><h1>The Goethals–Seidel Challenge</h1><p>Four circulant ±1 matrices of order n whose Gram matrices sum to 4nI give, through the Goethals–Seidel array, a Hadamard matrix of order 4n.</p><p>This board lists the known quadruples of odd order n ≤ 131 by symmetry type. Each entry is rechecked in exact arithmetic, and each search is recorded with what it covered.</p>{hero_figure(r)}</div>{DEFINITION}</section>
<section id="board"><h2>Status by order and type</h2><p class="key">Columns are the symmetry types of the blocks A, B, C, D: <strong>s</strong> symmetric, <strong>k</strong> skew. Skew needs one skew block, so it contains the good, best and G-matrix columns; general is any quadruple. Class counts treat reversals as equivalent; the skew and general columns also identify independent cyclic shifts (equiv-v2), without which every class at those orders splits into thousands.</p><div class="controls"><label>Family <select id="track"><option value="">All six strata</option>{"".join(f'<option value="{x}">{esc(TRACK_LABELS[x])}</option>' for x in TRACKS)}</select></label><label>Show <select id="status"><option value="">Every status</option><option value="unresolved">Research targets</option><option value="published-not-imported">Import tasks</option><option value="exists">Existence known</option><option value="empty">Nonexistence</option></select></label><label>Origin <select id="origin"><option value="">Every origin</option><option value="participant-result">Challenge results</option><option value="independent-reproduction">Independent reproductions</option><option value="published-import">Literature / catalog</option></select></label><label>Through n <select id="max-order"><option>31</option><option>63</option><option selected>131</option></select></label></div>
<div class="legend"><span class="known">Verified; a number is the class count, ✓ is existence without a count</span><span class="published">Published, not imported</span><span class="empty">Empty</span><span class="unresolved">Unresolved</span><span class="contradicted">Contradicts a published result</span><span class="participant">New result</span></div>
<div class="grid-wrap"><table id="grid"><thead><tr><th scope="col">n</th>{"".join(f'<th scope="col" data-track="{x}">{esc(TRACK_LABELS[x])}</th>' for x in TRACKS)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
{new_results_section(stats, r)}
<section class="columns"><article class="card"><h3>Bring a construction</h3><p>Submit four defining rows or negative-index sets. The verifier derives the family, checks the PAF identity, and detects equivalent entries.</p><p><code>./gs verify candidate.json</code><br><code>./gs submit candidate.json</code></p><a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/CONTRIBUTING.md">Contribution guide ↗</a></article></section>
<noscript><p>JavaScript enables the filters and the matrix drawings. All data remain available in <a href="stats.json">stats.json</a> and <a href="entries.json">entries.json</a>.</p></noscript>
<script src="app.js" defer></script><script src="picture.js" defer></script><script src="hero.js" defer></script>"""
    (out / "index.html").write_text(shell("The board", body, head=MATH_HEAD))
    for ident, doc in d.items():
        result = r[ident]
        sources = "".join(source_item(s, "../") for s in doc["sources"])
        credits = "".join(
            credit_block(group, "../")
            for group in result["contributions"]
            if group["id"] != "published-catalogues"
        )
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
            picture = (
                f'<section class="picture"><h2>Picture</h2>'
                f"{quad_figure(doc, result, '')}"
                f'<script src="../picture.js" defer></script></section>'
            )
        formal = "".join(
            '<p class="census">Machine-checked in Lean: '
            f"{esc(t['statement'])} "
            f'<a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/{esc(t["file"])}">'
            f"<code>{esc(t['theorem'])}</code></a></p>"
            for t in FORMAL
            if t["entry_id"] == ident
        )
        text = f"""<article class="entry"><div class="eyebrow">{esc(doc["type"])} · order {doc["n"]}</div><h1>{esc(ident)}</h1><span class="tag">{esc(result.get("tier") or "attributed / archived claim or lower bound")}</span>{credits}<p>Original source attribution: {esc("; ".join(doc["discovered_by"]))}<br>Submitted / imported by: {esc(doc["submitted_by"])} · {doc["submitted_at"]}<br>Board cell: {cell_links(result, "../")}</p>{formal}{picture}<h2>Checks and result</h2><pre>{esc(json.dumps(result, indent=2, ensure_ascii=False))}</pre><h2>Sources</h2><ul>{sources}</ul><details><summary>Submission JSON</summary><pre>{esc(json.dumps(doc, indent=2, ensure_ascii=False))}</pre></details></article>"""
        (out / f"entries/{ident}.html").write_text(shell(ident, text, "../"))
    refs = (
        '<article class="entry"><div class="eyebrow">bibliography</div><h1>References</h1>'
        "<p>Every paper, dataset and software source cited on the board. The numbers are the "
        "ones used in citations throughout; the list is built from "
        '<a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/baselines/references.bib">'
        f"baselines/references.bib</a>.</p>{BIB.listing()}"
        '<p><a href="index.html#board">Back to the board</a></p></article>'
    )
    (out / "references.html").write_text(shell("References", refs))
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
