#!/usr/bin/env python3
"""Build a deterministic, accessible static board from validated entries."""

import argparse
import html
import json
import math
import shutil
import sys
from collections.abc import Collection, Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import orders
import propus
import skew
from references import Bibliography
from verify.board import load, summary
from verify.common import ROOT, JSONObject, Row, tracks_at, write_json

type Results = dict[str, JSONObject]
"""Entry documents or verifier results, by entry id."""

BIB = Bibliography()
ORDERS = orders.validate(orders.load_data(), BIB)
# The stored skew-Hadamard matrices, by order.
STORED = orders.validate_matrices(orders.load_matrices(), ORDERS, BIB)


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
    "skew-sds": "With a skew block",
    "general": "Any",
}
# The grid has two groups of columns. The first is the order 4n, one column
# for Hadamard and one for skew-Hadamard matrices, coloured by
# baselines/hadamard-orders.json and the board, each linked to the board cell
# of its track and showing that cell's class count; the second is the four
# named families of cyclic Goethals–Seidel quadruples at n.
EXISTENCE_GROUP = "matrix of order 4n"
TRACK_GROUP = "cyclic Goethals–Seidel quadruples at n, by family"
# The track behind each order column, in words, for the column filter.
COLUMN_TRACKS = {"hadamard": "any quadruple", "skew-hadamard": "with a skew block"}


PRIMARY = ("williamson", "good", "best", "kkss")
# Tracks in which every quadruple has a skew block, so that its array is a
# skew-Hadamard matrix once that block is placed first.
SKEW_TRACKS = ("good", "best", "kkss", "skew-sds")
REPOSITORY = "https://github.com/vxvware/goethals-seidel-challenge"
FOLDERS = {"quadruple": "quadruples", "certificate": "certificates", "census": "census"}
DESCRIPTION = (
    "An exactly verified catalog of circulant Goethals–Seidel quadruples, research targets, and computational evidence."
)


def tier_label(tier: str | None, rows: bool = False) -> str | None:
    """The displayed name of a tier. The schema value stays `reproduced`; for a
    quadruple, or a cell that holds one, that tier means its rows were verified."""
    return "rows verified" if rows and tier == "reproduced" else tier


def scope_text(scope: JSONObject) -> str:
    """The family a scoped emptiness certificate covers, in words."""
    if scope["kind"] == "same-subgroup-coset-unions":
        zero = ", each with or without 0" if scope.get("include_zero") else ""
        return (
            "quadruples whose four blocks are unions of cosets of the multiplier subgroup "
            f"of order {scope['subgroup_order']}{zero}"
        )
    return scope["kind"]


def certificate_label(result: JSONObject) -> str | None:
    """What a certificate establishes, so that a proof about a restricted family
    is not read as a statement about the whole cell."""
    tier = result.get("tier")
    if tier == "proven" and result["scope"]["kind"] != "full":
        return "proven empty within a restricted family, not for the whole cell"
    if tier == "proven":
        return "proven empty"
    return tier


def display_rows(result: JSONObject) -> tuple[list[Row], str]:
    """The rows and symmetry pattern in the order the site draws and lists them.

    The canonical order sorts the rows, so a skew block need not come first, and
    the Goethals–Seidel array is skew only when its first block is. A quadruple
    with a skew block is shown with its first skew block moved to the front and
    the other rows in canonical order. The order of the blocks does not affect
    the autocorrelation identity, so the array is Hadamard either way.
    """
    rows, pattern = list(result["canonical_rows"]), result["pattern"]
    k = pattern.find("k")
    if k > 0:
        order = [k, *(i for i in range(4) if i != k)]
        rows = [rows[i] for i in order]
        pattern = "".join(pattern[i] for i in order)
    return rows, pattern


def signs(row: Iterable[int]) -> str:
    return "".join("+" if v > 0 else "-" for v in row)


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
<p>satisfies \(HH^{\mathsf T} = 4nI_{4n}\), so it is a Hadamard matrix of order \(4n\), and a skew-Hadamard matrix whenever \(A\) is skew. The array is due to Goethals and Seidel """  # noqa: E501
    + BIB.mark("goethals1970skew")
    + ".</p></aside>"
)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def badges(origins: Collection[str]) -> str:
    return "".join(
        f'<span class="origin {esc(origin)}">{esc(ORIGIN_LABELS[origin])}</span>'
        for origin in ORIGIN_LABELS
        if origin in origins
    )


def people(group: JSONObject) -> str:
    return ", ".join(
        f'<a href="https://github.com/{esc(person["github"])}">{esc(person["name"])}</a>'
        if person.get("github")
        else esc(person["name"])
        for person in group["contributors"]
    )


def source_keys(sources: Iterable[JSONObject]) -> list[str]:
    """Bibliography keys of the literature among a list of source objects."""
    return [key for s in sources if (key := BIB.resolve(s["url"]))]


def source_item(s: JSONObject, prefix: str) -> str:
    """One source as a list item: its link, then the number of its reference."""
    key = BIB.resolve(s["url"])
    mark = f" {BIB.mark(key, prefix)}" if key else ""
    url, extra = BIB.link(s["url"]), ""
    if "artifact" in s:
        # A source that only names the repository links the archived file
        # itself; a source with its own address keeps it and gains the file.
        path = s["artifact"]["path"]
        file = f"{REPOSITORY}/blob/main/{path}"
        if url.rstrip("/") == REPOSITORY:
            url = file
        else:
            extra = f' · <a href="{esc(file)}" rel="noopener">{esc(path)}</a>'
    return f'<li><a href="{esc(url)}" rel="noopener">{esc(s["title"])}</a>{mark}{extra}</li>'


def references_block(keys: Iterable[str], prefix: str) -> str:
    """The full references a page cites, numbered as in the board's reference list."""
    listing = BIB.listing(keys, prefix)
    return f"<h2>References</h2>{listing}" if listing else ""


def credit_block(group: JSONObject, prefix: str = "") -> str:
    publication = "Not yet published" if group["publication_status"] == "unpublished" else "Published result"
    cited = list(dict.fromkeys(BIB.keys_in(group["summary"])))
    authors = "; ".join(f"{esc(BIB.short(key))} {BIB.mark(key, prefix)}" for key in cited)
    return (
        f'<div class="credit">{badges([group["relation"]])}'
        f'<span class="publication">{publication}</span>'
        f'<p class="byline">{people(group) or authors or "Original authors as cited below"}</p>'
        f"<p>{BIB.cite(group['summary'], prefix)}</p></div>"
    )


def shell(title: str, body: str, prefix: str = "", head: str = "", description: str = DESCRIPTION) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · Hunting Hadamards</title><meta name="description" content="{esc(description)}"><meta property="og:title" content="{esc(title)} · Hunting Hadamards"><meta property="og:description" content="{esc(description)}">
<link rel="icon" href="{prefix}favicon.png" type="image/png"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap"><link rel="stylesheet" href="{prefix}style.css">{head}</head><body>
<header><a class="brand" href="{prefix}./">hunting-hadamards</a><nav><a href="{prefix}./#board">board</a><a href="{prefix}./#new-results">new results</a><a href="{prefix}references.html">references</a><a href="https://github.com/vxvware/goethals-seidel-challenge">contribute</a></nav></header>
<main>{body}</main><footer><a class="logo" href="https://vxvware.com" aria-label="vxvware"><img src="{prefix}vxvware-logo.png" alt="vxvware"></a><p class="credit"><a href="https://vprusso.github.io/">Vincent Russo</a> and <a href="https://trevorvincent.com/">Trevor Vincent</a></p></footer></body></html>"""  # noqa: E501


# The colour convention of picture.js, stated once on every page that draws one.
PICTURE_KEY = '<p class="muted">Skew blocks are green, symmetric blue, others grey; light is +1, dark is −1.</p>'


def quad_figure(doc: JSONObject, result: JSONObject, caption: str) -> str:
    """A figure that picture.js paints from the embedded rows, in display order."""
    rows, pattern = display_rows(result)
    payload = json.dumps(
        {"n": doc["n"], "rows": rows, "pattern": pattern},
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


def hero_figure(r: Results) -> str:
    quads: list[JSONObject] = []
    for track, n in HERO_CELLS:
        ident = min(k for k, x in r.items() if x["type"] == "quadruple" and x["n"] == n and x["tracks"][0] == track)
        rows, pattern = display_rows(r[ident])
        quads.append(
            {
                "id": ident,
                "n": n,
                "rows": rows,
                "pattern": pattern,
                "label": TRACK_LABELS[track].lower(),
            }
        )
    payload = json.dumps(quads, separators=(",", ":")).replace("<", "\\u003c")
    return (
        '<figure class="hero-anim"><canvas aria-label="A Goethals–Seidel array being filled in">'
        '</canvas><div><pre aria-hidden="true"></pre><a href="#board"></a></div>'
        f'<script type="application/json">{payload}</script></figure>'
    )


def cell_links(result: JSONObject, prefix: str) -> str:
    tracks = result.get("tracks", [result.get("stratum")])
    return ", ".join(
        f'<a href="{prefix}cells/n{result["n"]}-{esc(track)}.html">{esc(TRACK_LABELS[track])}</a>'
        for track in tracks
        if track in TRACK_LABELS
    )


def published_count(x: JSONObject) -> bool:
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


def census_count(c: JSONObject, r: Results) -> str:
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


# Cells whose statement the literature search of LITERATURE_SEARCH_DATE did not
# find in a source it could read. The list is curated because a cell's origins
# only say who supplied the rows: an exact class count at an order whose
# existence was already published still counts here. The wording is derived
# from the live board, so a statement cannot outrun its evidence.
#
# Not listed: Williamson 61, whose single class is reported in a 2012 paper
# (see the contribution williamson-61-reproduction); Williamson 65, which two
# earlier public records report (see the contribution
# williamson-65-confirmation); and G-matrices at 53, where no parameter set
# exists, so the empty cell is arithmetic and not a search result.
# G-matrices at 63 and 69 were checked on 2026-11-23.
LITERATURE_SEARCH_DATE = "2026-11-21"
NEW_RESULTS = (
    ("williamson", 67),
    ("good", 53),
    ("good", 55),
    ("good", 59),
    ("good", 61),
    ("good", 67),
    ("good", 75),
    ("kkss", 49),
    ("kkss", 51),
    ("kkss", 55),
    ("kkss", 57),
    ("kkss", 59),
    ("kkss", 61),
    ("kkss", 63),
    ("kkss", 69),
)
# The grid marks these cells as new alongside the cells a participant supplied
# rows for. The shift-class counts at 3 to 13 are not listed: no literature
# check has established that they were unknown.
NEW_CELLS = set(NEW_RESULTS)
NEW_RESULT_NAMES = {"williamson": "Williamson ", "good": "Good ", "kkss": "G-"}
# What a contradicted source says and what was read of it, by bibliography key.
# {cite} is the linked reference.
CONTRADICTION_NOTES = {
    "djokovic2018goethals": ("{cite} report a complete classification for odd v < 50 with none at this order."),
}
# Existence results, by contribution id, with their kind and the date of the
# search of the literature and public repositories that found no earlier one.
# A skew result is skew quadruples of one order, whose Goethals–Seidel array is
# a skew-Hadamard matrix. A propus result is one quadruple with a symmetric
# block and two equal blocks, whose propus array (site/propus.py) is a
# symmetric Hadamard matrix; neither the Hadamard nor the skew-Hadamard order
# is new there, so the grid does not mark it. An item is shown only while its
# contribution is marked featured in baselines/contributions.json, which is the
# maintainer's decision. A stored matrix of origin found
# (baselines/skew-matrices.json) is an existence result too: it is not a board
# entry, so no contribution carries it, and its record states the search of the
# literature. matrix_results lists those.
NEW_EXISTENCE = {
    "skew-pool-join-89": ("skew", "2026-11-22"),
    "propus-pool-join-65": ("propus", "2026-11-24"),
}
NEW_RESULT_GROUPS = (
    ("correction", "Contradicts a published result"),
    ("classification", "Classifications"),
    ("nonexistence", "Nonexistence"),
    ("existence", "Existence"),
)


def existence_results(stats: JSONObject, r: Results) -> list[JSONObject]:
    """The featured existence results, each worded from its verified entries."""
    groups = {g["id"]: g for g in stats["contributions"]}
    items = []
    for ident, (kind, _) in NEW_EXISTENCE.items():
        group = groups.get(ident)
        if not group or not group["featured"]:
            continue
        entries = [r[e] for e in group["entry_ids"]]
        # The section's own sentence carries the search qualifier.
        item: JSONObject = {"kind": kind, "source": None, "qualifier": None}
        if kind == "propus":
            if len(entries) != 1 or entries[0]["type"] != "quadruple" or not propus.shape(*display_rows(entries[0])):
                raise ValueError(f"existence result is not one propus quadruple: {ident}")
            n = entries[0]["n"]
            item |= {
                "n": n,
                "stratum": "general",
                "text": (f"A symmetric Hadamard matrix of order {4 * n}, from a propus family over Z_{n}."),
                # The entry page carries the propus array and the matrix file.
                "href": f"entries/{entries[0]['id']}.html",
            }
        else:
            orders = {x["n"] for x in entries}
            if len(orders) != 1 or any(x["type"] != "quadruple" or "skew-sds" not in x["tracks"] for x in entries):
                raise ValueError(f"existence result is not skew quadruples of one order: {ident}")
            n = orders.pop()
            # Doubling a skew-Hadamard matrix gives one of twice the order; name that order too when
            # the cited lists give it without a matrix.
            skew_orders = ORDERS["types"]["skew-hadamard"]
            double = 8 * n if 8 * n in skew_orders["listed_open"] and 8 * n not in skew_orders["open"] else None
            text = f"A skew-Hadamard matrix of order {4 * n}" + (
                f", and of order {double} by doubling." if double else "."
            )
            item |= {"n": n, "stratum": "skew-sds", "text": text, "double": double}
        items.append(item)
    return items + matrix_results()


def matrix_results() -> list[JSONObject]:
    """The stored skew-Hadamard matrices found for this board, each linked to the page that describes it."""
    return [
        {
            "kind": "matrix",
            "source": None,
            "qualifier": None,
            "n": order // 4,
            "stratum": orders.CELL_TRACKS[skew.NAME],
            "text": f"A skew-Hadamard matrix of order {order}.",
            "href": orders.order_href(order // 4, skew.NAME),
        }
        for order, record in sorted(STORED.items())
        if record["origin"] == "found"
    ]


def propus_entries(stats: JSONObject) -> set[str]:
    """The entries of the propus existence results, featured or not: their pages carry the propus array."""
    kinds = {ident: kind for ident, (kind, _) in NEW_EXISTENCE.items()}
    return {e for g in stats["contributions"] if kinds.get(g["id"]) == "propus" for e in g["entry_ids"]}


def new_cells(stats: JSONObject, r: Results) -> set[tuple[str, int]]:
    """The cells the grid marks as new: the curated list and the featured skew existence results."""
    return NEW_CELLS | {(x["stratum"], x["n"]) for x in existence_results(stats, r) if x["kind"] == "skew"}


def new_results(stats: JSONObject, r: Results) -> list[tuple[str, list[JSONObject]]]:
    """The curated new results, grouped, each worded from its cell's evidence."""
    by_cell = {(x["n"], x["stratum"]): x for x in stats["cells"]}
    groups: dict[str, list[JSONObject]] = {key: [] for key, _ in NEW_RESULT_GROUPS}
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
            # Whether the board has replayed the search is shown on the cell page,
            # not on the front page.
            text = f"{subject}: none exist."
            kind = "nonexistence"
        else:
            raise ValueError(f"new result without a census or an empty cell: {track} n={n}")
        if c["contradicted"]:
            kind = "correction"
            item["source"] = c["contradicted"][0]["source"]
        groups[kind].append(item | {"text": text})
    groups["existence"] = existence_results(stats, r)
    return [(title, groups[key]) for key, title in NEW_RESULT_GROUPS if groups[key]]


def new_results_section(stats: JSONObject, r: Results) -> str:
    parts = [
        '<section id="new-results"><h2>New results</h2><p class="key">Statements that a literature '
        "search did not find in the sources it read. This does not claim priority.</p>"
    ]
    for title, items in new_results(stats, r):
        parts.append(f"<h3>{esc(title)}</h3><ul>")
        for x in items:
            href = x.get("href") or f"cells/n{x['n']}-{x['stratum']}.html"
            link = f'<a href="{esc(href)}">{esc(x["text"])}</a>'
            if x["source"]:
                url = x["source"]["url"]
                key = BIB.resolve(url)
                if not key:
                    raise ValueError(f"contradicted source is not in references.bib: {url}")
                if key not in CONTRADICTION_NOTES:
                    raise ValueError(f"contradicted source has no note on what was read: {key}")
                cite = f'<a href="{esc(url)}" rel="noopener">{esc(BIB.short(key))}</a> {BIB.mark(key)}'
                link += " " + esc(CONTRADICTION_NOTES[key]).replace("{cite}", cite)
            if x["qualifier"]:
                link += f' <span class="muted">({esc(x["qualifier"])})</span>'
            parts.append(f"<li>{link}</li>")
        parts.append("</ul>")
    parts.append("</section>")
    return "".join(parts)


def cell_description(c: JSONObject) -> str:
    """The meta description of a cell page, for search results and link previews."""
    n, track = c["n"], c["stratum"]
    head = f"Order n = {n}, {TRACK_LABELS[track]}: "
    count = len(c["quadruples"])
    if count:
        noun = "quadruple" if count == 1 else "quadruples"
        kind = (
            f"with a skew block, each giving a skew-Hadamard matrix of order {4 * n}"
            if track in SKEW_TRACKS
            else f"each giving a Hadamard matrix of order {4 * n}"
        )
        return f"{head}{count} exactly verified Goethals–Seidel {noun} {kind}."
    return (
        head
        + {
            "exists": "existence is published and no rows are on the board yet.",
            "empty": "recorded as empty.",
            "unresolved": "unresolved, with no quadruple and no proof of emptiness on the board.",
        }[c["status"]]
    )


def entry_description(doc: JSONObject, result: JSONObject) -> str:
    """The meta description of an entry page."""
    ident, n = doc["id"], doc["n"]
    if doc["type"] == "quadruple":
        _, pattern = display_rows(result)
        kind = "skew-Hadamard" if "k" in pattern else "Hadamard"
        return (
            f"Goethals–Seidel quadruple {ident} of order n = {n}, exactly verified. Its array "
            f"is a {kind} matrix of order {4 * n}."
        )
    label = TRACK_LABELS[result["stratum"]]
    if doc["type"] == "census":
        return f"Census record {ident} at order n = {n}, {label}: {result['count']} recorded."
    what = certificate_label(result) or "attributed claim"
    return f"Evidence record {ident} at order n = {n}, {label}: {what}."


def rows_block(doc: JSONObject, result: JSONObject) -> str:
    """The four rows of a quadruple as sign strings, in the order the picture uses."""
    rows, pattern = display_rows(result)
    n = doc["n"]
    kind = "skew-Hadamard" if pattern[0] == "k" else "Hadamard"
    text = (
        "The four rows A, B, C, D of the canonical representative, with + for 1 and - for −1. "
        "The letter after each name is its symmetry: k skew, s symmetric, · neither. "
    )
    if pattern[0] == "k":
        text += "A skew row is listed first, so the "
    else:
        text += "The "
    text += f"Goethals–Seidel array of the rows in this order is a {kind} matrix of order {4 * n}."
    if pattern != result["pattern"]:
        text += (
            " The verifier result below lists the same rows as canonical_rows in sorted order, "
            "in which the skew row is not first."
        )
    lines = "\n".join(f"{name} {letter} {signs(row)}" for name, letter, row in zip("ABCD", pattern, rows, strict=True))
    return f'<section class="rows-text"><h2>Rows</h2><p>{esc(text)}</p><pre>{esc(lines)}</pre></section>'


def cell_page(c: JSONObject, d: Results, r: Results, matrix: str = "") -> str:
    """One page per order and track: status, credit, sources, and the matrices.

    matrix is the block about the skew-Hadamard matrix of order 4n, which the
    page of the track for quadruples with a skew block carries.
    """
    n, track = c["n"], c["stratum"]
    tier = tier_label(c["tier"], rows=bool(c["quadruples"]))
    status = " · ".join(x for x in (c["status"], c["artifact_status"], tier) if x)
    expanded = " (sds: supplementary difference sets)" if track == "skew-sds" else ""
    parts = [
        f'<article class="entry"><div class="eyebrow">cell · order {n} · {esc(track)}{expanded}'
        f'</div><h1>n = {n} · {esc(TRACK_LABELS[track])}</h1><span class="tag">{esc(status)}</span>'
    ]
    if c["censuses"]:
        x = r[c["censuses"][0]]
        link = f'<a href="../entries/{esc(x["id"])}.html">{esc(x["id"])}</a>'
        if x["bound"] != "exact" and x["count"] == 0:
            parts.append(f'<p class="census">published empty census · awaiting a full replay · {link}</p>')
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
            stated = census_count(c, r) or (f"{x['count']} {'class' if x['count'] == 1 else 'classes'} known")
            parts.append(
                f'<p class="census"><strong>{esc(stated)}</strong> · {kind} · '
                f"{esc(x['tier'] or 'verified representatives')} · {link}</p>"
            )
    if c["note"]:
        parts.append(f"<p>{esc(c['note'])}</p>")
    for record in c["sources"]:
        if published := record.get("published_classes"):
            # A count under another equivalence is stated beside the board's own, never in place of it.
            key = BIB.resolve(record["source"]["url"])
            noun = "class" if published["count"] == 1 else "classes"
            parts.append(
                f'<p class="census">Published count: <strong>{published["count"]} {noun}</strong> '
                f"{BIB.mark(key, '../') if key else ''}, counted under {BIB.cite(published['equivalence'], '../')}. "
                "The board's equiv-v1 has neither the shift by n/2 nor the alternating negation, so its classes "
                "are finer and its count at this order is at least the published one.</p>"
            )
    if c["status"] == "unresolved":
        parts.append(
            "<p>No quadruple of this type at this order is known to us in the sources the board cites, and the "
            "board holds no proof that none exists. This is not a statement that the literature has none.</p>"
        )
    for claim in c["contradicted"]:
        parts.append(
            '<p class="contradiction">Exactly verified rows on this board contradict a published '
            f'nonexistence claim: <a href="{esc(BIB.link(claim["source"]["url"]))}" rel="noopener">'
            f"{esc(claim['source']['title'])}</a>"
            + "".join(f" {BIB.mark(key, '../')}" for key in source_keys([claim["source"]]))
            + "</p>"
        )
    parts.append(matrix)
    # The catalogue credit is the same paragraph for every imported cell; the
    # cell's own sources already name the original authors. A contribution that
    # reports quadruples of the cell comes before one that only holds evidence,
    # so that a note about a restricted family follows the result it qualifies.
    parts.extend(
        credit_block(group, "../")
        for group in sorted(c["contributions"], key=lambda g: not set(g["entry_ids"]) & set(c["quadruples"]))
        if group["id"] != "published-catalogues"
    )
    if c["quadruples"]:
        count = len(c["quadruples"])
        heading = "Matrices" if c["censuses"] else f"{count} verified {'class' if count == 1 else 'classes'}"
        parts.append(f'<h2>{heading}</h2>{PICTURE_KEY}<div class="gallery">')
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
                detail += f" · {result['count']} {'class' if result['count'] == 1 else 'classes'}{known}"
                if result["equivalence"] == "equiv-v2":
                    detail += " · equiv-v2 (shifts identified)"
                if published_count(result):
                    detail += " · published count"
            if result["type"] == "certificate":
                if label := certificate_label(result):
                    detail += f" · {label}"
                if result["scope"]["kind"] != "full":
                    detail += f" · covers only {scope_text(result['scope'])}"
            elif result.get("tier"):
                detail += f" · {result['tier']}"
            items.append(f'<li><a href="../entries/{esc(ident)}.html">{esc(ident)}</a> · {esc(detail)}</li>')
        parts.append(f"<h2>Evidence and census records</h2><ul>{''.join(items)}</ul>")
    if c["sources"]:
        parts.append("<h2>Sources</h2><ul>" + "".join(source_item(s["source"], "../") for s in c["sources"]) + "</ul>")
    parts.append('<p><a href="../#board">Back to the board</a></p></article>')
    if c["quadruples"]:
        parts.append('<script src="../picture.js" defer></script>')
    return "".join(parts)


def two_odd_squares(m: int) -> bool:
    """Whether m is a sum of two odd squares."""
    return any(
        (rest := m - a * a) >= 0 and math.isqrt(rest) ** 2 == rest and math.isqrt(rest) % 2 == 1
        for a in range(1, math.isqrt(m) + 1, 2)
    )


def grid_cell(c: JSONObject, marked: Collection[tuple[str, int]], r: Results) -> JSONObject:
    """What the front-page grid shows for one board cell, and the words of its tooltip."""
    track, n = c["stratum"], c["n"]
    label = (
        "Contradicts published"
        if c["contradicted"]
        else "Verified"
        if c["quadruples"]
        else "Published"
        if c["artifact_status"] == "published-not-imported"
        else "Empty"
        if c["status"] == "empty"
        else "Unresolved"
    )
    small = tier_label(c["tier"], rows=bool(c["quadruples"])) or (
        " / ".join(c["empty_basis"]) or "literature"
        if c["status"] == "empty"
        else "claim / partial evidence"
        if c["evidence"]
        else ""
    )
    origin_words = ", ".join(ORIGIN_LABELS[o] for o in c["origins"])
    accessible = "; ".join(x for x in (f"Order {n}, {TRACK_LABELS[track]}: {label}", small, origin_words) if x)
    origins = list(c["origins"])
    new = (track, n) in marked
    if new and "participant-result" not in origins:
        origins.insert(0, "participant-result")
    count = census_count(c, r)
    if count:
        accessible += f"; {count}"
    elif label in ("Verified", "Contradicts published"):
        accessible += "; exists, no class count"
    elif label == "Published":
        accessible += "; exists by a cited construction or enumeration, no rows on the board"
    elif label == "Unresolved":
        accessible += "; no quadruple of this type is known to us in the cited sources"
    published = next((s["published_classes"] for s in c["sources"] if "published_classes" in s), None)
    if published:
        accessible += f"; published count {published['count']}, under a coarser equivalence than equiv-v1"
    text = (
        count.replace(" classes", "").replace(" class", "")
        if count
        else {"Empty": "0", "Unresolved": "?"}.get(label, "✓")
    )
    if label == "Empty" and track == "best" and math.isqrt(4 * n - 3) ** 2 != 4 * n - 3:
        # Best matrices need 4n - 3 to be a square, so this cell is empty by a theorem and not by a search.
        text = "×"
        accessible += "; cannot exist: 4n − 3 is not a square"
    if c["status"] == "empty" and track == "kkss" and n % 2 and not two_odd_squares(4 * n - 2):
        # Two skew blocks have row sum 1, so the other two row sums are odd with squares adding to 4n - 2.
        text = "×"
        accessible += "; cannot exist: 4n − 2 is not a sum of two odd squares"
    return {
        "track": track,
        "status": c["status"],
        "artifact": c["artifact_status"],
        "origins": origins,
        "classes": f"{c['status']} {c['artifact_status']}{' contradicted' if c['contradicted'] else ''}",
        "contradicted": c["contradicted"],
        "accessible": accessible,
        "count": count,
        "text": text,
        "new": new,
        # The diamond means a new result for this cell. A cell can hold
        # rows or evidence from the challenge without that: a new G-matrix
        # is also a skew quadruple at an order where those were known.
        "marker": '<span class="participant" title="New result"></span>' if new else "",
    }


def filter_data(x: JSONObject) -> dict[str, str]:
    """The data attributes of a grid cell that the filters in app.js read."""
    return {
        "track": x["track"],
        "status": x["status"],
        "artifact": x["artifact"],
        "origins": " ".join(x["origins"]),
    }


def filter_attributes(x: JSONObject) -> str:
    return "".join(f' data-{key}="{esc(value)}"' for key, value in filter_data(x).items())


def grid(stats: JSONObject, r: Results) -> tuple[str, str, str]:
    """The front-page grid: its header rows, its body rows and the options of its column filter."""
    by_cell = {(x["n"], x["stratum"]): x for x in stats["cells"]}
    marked = new_cells(stats, r)
    states = orders.table(ORDERS, r, STORED)
    skew_results = [x for x in existence_results(stats, r) if x["kind"] == "skew"]
    # The doubled order of a new skew-Hadamard order is marked as new with it.
    new_skew = {x["n"] for x in skew_results} | {x["double"] // 4 for x in skew_results if x["double"]}
    rows = []
    # A skew circulant block has x[n - j] = -x[j] for j != 0, which at even n fails at j = n / 2.
    impossible = "Cannot exist: a skew circulant block needs odd n"
    for n in range(1, ORDERS["range"]["max"] // 4 + 1):
        # Every track has a cell at odd n. At even n only the tracks without a skew row do, and the
        # three families with one are marked impossible.
        found = {track: grid_cell(by_cell[n, track], marked, r) for track in tracks_at(n)}
        cells = [
            f'<td class="{esc(x["classes"])}"{filter_attributes(x)}>'
            f'<a class="cell" href="cells/n{n}-{track}.html" aria-label="{esc(x["accessible"])}" '
            f'title="{esc(x["accessible"])}">{esc(x["text"])}{x["marker"]}</a></td>'
            if (x := found.get(track))
            else f'<td class="blank impossible" data-track="{track}" title="{impossible}">×</td>'
            for track in PRIMARY
        ]
        board = {
            name: {
                "count": found[track]["count"],
                "contradicted": found[track]["contradicted"],
                "new": found[track]["new"],
                "words": f"board cell: {found[track]['accessible']}",
                "data": filter_data(found[track]),
            }
            for name, track in orders.CELL_TRACKS.items()
            if track in found
        }
        rows.append(
            f'<tr data-n="{n}"><td>{n}</td>{orders.grid_cells(n, states, new_skew, board)}' + "".join(cells) + "</tr>"
        )
    head = (
        '<tr><th scope="col" rowspan="2">n</th>'
        f'<th scope="colgroup" colspan="2" class="group">{esc(EXISTENCE_GROUP)}</th>'
        f'<th scope="colgroup" colspan="{len(PRIMARY)}" class="group sep">{esc(TRACK_GROUP)}</th>'
        "</tr><tr>"
        + "".join(f'<th scope="col" data-track="{orders.CELL_TRACKS[name]}">{name}</th>' for name in orders.TYPES)
        + "".join(
            f'<th scope="col" data-track="{x}"{sep}>{esc(TRACK_LABELS[x])}</th>'
            for x, sep in zip(PRIMARY, (' class="sep"', *[""] * len(PRIMARY)), strict=False)
        )
        + "</tr>"
    )
    columns = [(orders.CELL_TRACKS[name], f"{name} ({COLUMN_TRACKS[name]})") for name in orders.TYPES] + [
        (x, TRACK_LABELS[x]) for x in PRIMARY
    ]
    options = "".join(f'<option value="{x}">{esc(label)}</option>' for x, label in columns)
    return head, "".join(rows), options


def skew_matrices(states: orders.OrderStates, r: Results) -> dict[int, tuple[str, str]]:
    """The entry id and matrix file text of every skew-Hadamard order the board holds.

    The matrix is the Goethals–Seidel array of the entry with the smallest id,
    with its rows in display order, so that a skew block comes first. Raises
    ValueError unless the array passes the exact skew-Hadamard check.
    """
    files = {}
    for order, ident in skew.chosen_entries(states).items():
        rows, _ = display_rows(r[ident])
        h = skew.gs_array(rows)
        if len(h) != order:
            raise ValueError(f"the array of {ident} does not have order {order}")
        try:
            skew.check_skew_hadamard(h)
        except ValueError as error:
            raise ValueError(f"the array of {ident} is not a skew-Hadamard matrix: {error}") from error
        files[order] = (ident, skew.matrix_text(h))
    return files


def write_skew_matrices(out: Path, states: orders.OrderStates, r: Results, quick: bool = False) -> dict[int, str]:
    """Write the matrix file of every skew-Hadamard order that is not open; return the chosen board entries.

    An order the board holds gets the array of an entry, and every other order
    its stored matrix. A stored file is decompressed, compared with its sha256
    and checked exactly before it is written; any failure raises ValueError and
    stops the build. A quick build compares the sha256 only and skips the exact
    check of the stored files, which the full build and the tests run.
    """
    files = skew_matrices(states, r)
    folder = out / "matrices"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    for order, (_, text) in files.items():
        (out / skew.file_name(order)).write_text(text)
    for order, s in states.items():
        if s[skew.NAME]["state"] == "verified":
            (out / skew.file_name(order)).write_text(skew.stored_text(STORED[order], check=not quick))
    return {order: ident for order, (ident, _) in files.items()}


def write_propus_matrices(out: Path, idents: Iterable[str], r: Results) -> None:
    """Write the propus array of each listed entry as a matrix file.

    Call it after write_skew_matrices, which empties the folder. The array is
    checked exactly to be a symmetric Hadamard matrix before it is written; a
    failure raises ValueError and stops the build.
    """
    for ident in sorted(idents):
        try:
            text = propus.matrix_file(*display_rows(r[ident]))
        except ValueError as error:
            raise ValueError(f"the propus array of {ident} is not a symmetric Hadamard matrix: {error}") from error
        (out / propus.file_name(ident)).write_text(text)


def write_order_pages(out: Path, states: orders.OrderStates, chosen: dict[int, str]) -> None:
    """Write the page of every skew-Hadamard order whose n has no board cell for a skew block: the even n."""
    folder = out / "orders"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    for order, s in states.items():
        if orders.has_cell(order // 4, orders.CELL_TRACKS[skew.NAME]):
            continue
        info = s[skew.NAME]
        (folder / f"skew-{order}.html").write_text(
            shell(
                f"Skew-Hadamard matrix of order {order}",
                skew.order_page(ORDERS, order, info, STORED, chosen, BIB),
                "../",
                description=f"Skew-Hadamard matrix of order {order}: {orders.STATE_WORDS[info['state']]}.",
            )
        )


def build(quick: bool = False) -> JSONObject:
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
        badge = f"""<svg xmlns="http://www.w3.org/2000/svg" width="202" height="24" role="img" aria-label="{label}: {value}">
<title>{label}: {value}</title><rect width="202" height="24" rx="4" fill="#275c49"/>
<path d="M4 0h151v24H4Q0 24 0 20V4Q0 0 4 0" fill="#36413d"/>
<g fill="white" font-family="Verdana,sans-serif" font-size="11" text-anchor="middle">
<text x="78" y="16">{label}</text><text x="178" y="16">{value}</text></g></svg>"""  # noqa: E501
        (out / f"badges/{key}.svg").write_text(badge)
    head, rows, options = grid(stats, r)
    body = f"""<section class="hero"><div><h1>Hunting Hadamards</h1><p>Hadamard matrices are believed to exist in every size divisible by 4, but some sizes have never been found. This board tracks the search.</p>{hero_figure(r)}</div>{DEFINITION}</section>
<section id="board"><h2>The board</h2><p class="key">{orders.intro(ORDERS)} (<a href="{REPOSITORY}/blob/main/TRACKS.md#reading-the-board">how to read the board</a>)</p><div class="controls"><label>Column <select id="track"><option value="">Every column</option>{options}</select></label><label>Show <select id="status"><option value="">Every status</option><option value="unresolved">Research targets</option><option value="published-not-imported">Import tasks</option><option value="exists">Existence known</option><option value="empty">Nonexistence</option></select></label><label>Origin <select id="origin"><option value="">Every origin</option><option value="participant-result">Challenge results</option><option value="independent-reproduction">Independent reproductions</option><option value="published-import">Literature / catalog</option></select></label><label>Through n <select id="max-order"><option>31</option><option>63</option><option>131</option><option selected value="{orders.BOARD_MAX_N}">{orders.BOARD_MAX_N}, every n</option></select></label></div>
<div class="legend"><span class="known">exists</span><span class="empty">empty</span><span class="unresolved">open</span><span class="contradicted">contradicts the literature</span><span class="impossible">cannot exist</span><span class="participant">new result</span></div>
<div class="grid-wrap"><table id="grid"><thead>{head}</thead><tbody>{rows}</tbody></table></div></section>
{new_results_section(stats, r)}
<section class="columns"><article class="card"><h3>Bring a construction</h3><p>Found four sequences that fit? The verifier checks them exactly, works out their family and rejects duplicates of entries already here.</p><p><code>./gs verify candidate.json</code><br><code>./gs submit candidate.json</code></p><a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/CONTRIBUTING.md">Contribution guide ↗</a></article></section>
<noscript><p>JavaScript enables the filters and the matrix drawings. All data remain available in <a href="stats.json">stats.json</a> and <a href="entries.json">entries.json</a>.</p></noscript>
<script src="app.js" defer></script><script src="picture.js" defer></script><script src="hero.js" defer></script>"""  # noqa: E501
    (out / "index.html").write_text(shell("The board", body, head=MATH_HEAD))
    with_propus = propus_entries(stats)
    for ident, doc in d.items():
        result = r[ident]
        sources = "".join(source_item(s, "../") for s in doc["sources"])
        credits = "".join(
            credit_block(group, "../") for group in result["contributions"] if group["id"] != "published-catalogues"
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
        tag = result.get("tier")
        if doc["type"] == "quadruple":
            tag = tier_label(tag, rows=True)
            symmetric = propus.block(ident, *display_rows(result), BIB)[0] if ident in with_propus else ""
            picture = (
                f"{rows_block(doc, result)}{symmetric}"
                f'<section class="picture"><h2>Picture</h2>{PICTURE_KEY}'
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
        if doc["type"] == "certificate":
            tag = certificate_label(result)
        file = f"{FOLDERS[doc['type']]}/{ident}.json"
        raw = f"https://raw.githubusercontent.com/vxvware/goethals-seidel-challenge/main/{file}"
        links = (
            f'<br>Entry file: <a href="{REPOSITORY}/blob/main/{file}">{esc(file)}</a> · <a href="{raw}">raw JSON</a>'
        )
        notes = f"<h2>Notes</h2><p>{esc(doc['notes'])}</p>" if doc.get("notes") else ""
        text = (
            f'<article class="entry"><div class="eyebrow">{esc(doc["type"])} · order {doc["n"]}</div>'
            f'<h1>{esc(ident)}</h1><span class="tag">{esc(tag or "attributed / archived claim or lower bound")}</span>'
            f"{credits}<p>Original source attribution: {esc('; '.join(doc['discovered_by']))}<br>"
            f"Submitted / imported by: {esc(doc['submitted_by'])} · {doc['submitted_at']}<br>"
            f"Board cell: {cell_links(result, '../')}{links}</p>{formal}{picture}{notes}"
            f"<h2>Checks and result</h2><pre>{esc(json.dumps(result, indent=2, ensure_ascii=False))}</pre>"
            f"<h2>Sources</h2><ul>{sources}</ul><details><summary>Submission JSON</summary>"
            f"<pre>{esc(json.dumps(doc, indent=2, ensure_ascii=False))}</pre></details></article>"
        )
        (out / f"entries/{ident}.html").write_text(
            shell(ident, text, "../", description=entry_description(doc, result))
        )
    refs = (
        '<article class="entry"><div class="eyebrow">bibliography</div><h1>References</h1>'
        "<p>Every paper, dataset and software source cited on the board. The numbers are the "
        "ones used in citations throughout; the list is built from "
        '<a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/baselines/references.bib">'
        f"baselines/references.bib</a>.</p>{BIB.listing()}"
        '<p><a href="./#board">Back to the board</a></p></article>'
    )
    (out / "references.html").write_text(shell("References", refs))
    states = orders.table(ORDERS, r, STORED)
    chosen = write_skew_matrices(out, states, r, quick)
    write_propus_matrices(out, with_propus, r)
    write_order_pages(out, states, chosen)
    # Files of the page this build no longer writes.
    for name in ("skew.html", "skew.js"):
        (out / name).unlink(missing_ok=True)
    for c in stats["cells"]:
        title = f"n = {c['n']} · {TRACK_LABELS[c['stratum']]}"
        matrix = ""
        if c["stratum"] == orders.CELL_TRACKS[skew.NAME]:
            matrix, _ = skew.matrix_block(ORDERS, 4 * c["n"], states[4 * c["n"]][skew.NAME], STORED, chosen, BIB)
        (out / f"cells/n{c['n']}-{c['stratum']}.html").write_text(
            shell(title, cell_page(c, d, r, matrix), "../", description=cell_description(c))
        )
    print(
        f"Built docs/: {stats['quadruples']} quadruples, {stats['certificates']} evidence records, "
        f"{stats['censuses']} censuses."
    )
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    build(a.quick)
