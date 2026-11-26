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
import twisted
from references import Bibliography
from verify import catalogue
from verify.board import load, summary
from verify.common import ROOT, JSONObject, Row, Rows, tracks_at, write_json
from verify.gs_verify import row_types

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
# What each entry type is called on its page.
ENTRY_KINDS = {"quadruple": "quadruple", "certificate": "evidence record", "census": "census record"}
DESCRIPTION = (
    "Exactly verified circulant Goethals–Seidel quadruples, the Hadamard and skew-Hadamard matrices they give, and the "
    "orders still open."
)


# What each track's quadruples are called in a sentence. The slugs of the schema (track names, tiers,
# equivalence versions, availability) stay in data attributes and JSON; visible text uses these words.
TRACK_NOUNS = {
    "williamson": ("Williamson quadruple of length {n}", "Williamson quadruples of length {n}"),
    "good": ("quadruple of good matrices of length {n}", "quadruples of good matrices of length {n}"),
    "best": ("quadruple of best matrices of length {n}", "quadruples of best matrices of length {n}"),
    "kkss": ("quadruple of G-matrices of length {n}", "quadruples of G-matrices of length {n}"),
    "skew-sds": (
        "Goethals–Seidel quadruple of length {n} with a skew block",
        "Goethals–Seidel quadruples of length {n} with a skew block",
    ),
    "general": ("Goethals–Seidel quadruple of length {n}", "Goethals–Seidel quadruples of length {n}"),
}
# A record without a tier, by its provenance.
RECORD_WORDS = {
    "literature": "published result",
    "unreplayed-census": "report of a complete search",
    "archived-computation": "archived search with incomplete records",
}
# How a result is supported, by verification tier or, without a tier, by the provenance of its record.
EVIDENCE_WORDS = {
    "proven": "proved by a rule the board checks",
    "reproduced": "complete search rerun here",
    "spot-checked": "archived search records checked and a sample rerun here",
    "receipt-backed": "archived search records checked, search not rerun here",
    "unreplayed-census": "complete search reported, not rerun here",
    "archived-computation": "archived search with incomplete records, not rerun here",
    "literature": "published result, not checked here",
}
SAME_CLASS = (
    "Two quadruples are in the same class when one comes from the other by reordering the blocks, negating a "
    "block, reversing a block, or applying one multiplier to the indices of all four blocks."
)
SHIFT_CLASS = "cyclic shifts of single blocks also count as equivalent"
# The two operations of the published equivalence at even n that the board's classes do not have.
PUBLISHED_MERGES = (
    "The published count merges quadruples that differ by a half-period shift of a block or by negating every "
    "second entry of all four blocks; the board counts those separately, so its count is at least the published one."
)
PUBLISHED_CLASSES = (
    "The published classes treat two quadruples as the same when one comes from the other by reordering the "
    "blocks, negating a block, applying one multiplier to the indices of all four blocks, shifting a block by "
    "half its length, or negating every second entry of all four blocks."
)


def nouns(track: str, n: int) -> tuple[str, str]:
    """The quadruples of a track at length n, singular and plural, for a sentence."""
    singular, many = TRACK_NOUNS[track]
    return singular.format(n=n), many.format(n=n)


def plural(count: int, singular: str, many: str | None = None) -> str:
    return f"{count:,} {singular if count == 1 else many or singular + 's'}"


def evidence_words(tier: str | None) -> str | None:
    """A tier or provenance value in words; an unknown value is an error, so no slug reaches a page."""
    return None if tier is None else EVIDENCE_WORDS[tier]


def tier_label(tier: str | None, rows: bool = False) -> str | None:
    """The displayed words of a tier. The schema value stays `reproduced`; for a
    quadruple, or a cell that holds one, that tier means its rows were verified."""
    return "rows verified" if rows and tier == "reproduced" else evidence_words(tier)


def scope_text(scope: JSONObject) -> str:
    """The family a scoped emptiness certificate covers, in words."""
    if scope["kind"] == "same-subgroup-coset-unions":
        zero = ", each with or without 0" if scope.get("include_zero") else ""
        return (
            "quadruples whose four blocks are unions of cosets of the multiplier subgroup "
            f"of order {scope['subgroup_order']}{zero}"
        )
    return "a restricted family" if scope["kind"] != "full" else "every quadruple of the cell"


def certificate_label(result: JSONObject) -> str | None:
    """What a certificate establishes, so that a proof about a restricted family
    is not read as a statement about the whole cell."""
    tier = result.get("tier")
    if tier == "proven" and result["scope"]["kind"] != "full":
        return "proved empty within a restricted family, not for the whole cell"
    if tier == "proven":
        return "proved empty by a rule the board checks"
    if result.get("claim") == "partial":
        return "partial search, does not settle the cell"
    return evidence_words(tier) or RECORD_WORDS.get(result.get("provenance", ""))


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
<header><a class="brand" href="{prefix}./">hunting-hadamards</a><nav><a href="{prefix}./#board">board</a><a href="{prefix}./#new-results">new results</a><a href="{prefix}references.html">references</a><a href="https://github.com/vxvware/goethals-seidel-challenge">contribute</a><a class="gh" href="https://github.com/vxvware/goethals-seidel-challenge" aria-label="Source code on GitHub" title="Source code on GitHub"><svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg></a></nav></header>
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
# exists, so the empty cell is arithmetic and not a search result; and good
# matrices at 75, whose nonexistence a note of 2026-11-16 reports (see the
# contribution good-75-empty).
# G-matrices at 63 and 69 were checked on 2026-11-23.
LITERATURE_SEARCH_DATE = "2026-11-21"
NEW_RESULTS = (
    ("williamson", 67),
    ("good", 53),
    ("good", 55),
    ("good", 59),
    ("good", 61),
    ("good", 67),
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
    "djokovic2018goethals": "A complete classification for odd v < 50 in {cite} lists none at this order.",
}
# Existence results, by contribution id, with their kind and the date of the
# search of the sources its record names that found no earlier one.
# A skew result is skew quadruples of one order, whose Goethals–Seidel array is
# a skew-Hadamard matrix. A propus result is one quadruple with a symmetric
# block and two equal blocks, whose propus array (site/propus.py) is a
# symmetric Hadamard matrix; neither the Hadamard nor the skew-Hadamard order
# is new there, so the grid does not mark it. A twisted result is one quadruple
# whose rows are tied by a multiplier s with s² = 1 other than 1 and −1, whose
# array (site/twisted.py) is a symmetric Hadamard matrix; the grid does not
# mark it either. An item is shown only while its
# contribution is marked featured in baselines/contributions.json, which is the
# maintainer's decision. A stored matrix of origin found
# (baselines/skew-matrices.json) is an existence result too: it is not a board
# entry, so no contribution carries it, and its record states the search of the
# literature. matrix_results lists those, with one shared line for the orders
# above the range of the grid.
NEW_EXISTENCE = {
    "skew-pool-join-89": ("skew", "2026-11-22"),
    "propus-pool-join-65": ("propus", "2026-11-24"),
    "symmetric-twisted-247": ("twisted", "2026-11-25"),
    "symmetric-twisted-259": ("twisted", "2026-11-25"),
    "symmetric-twisted-267": ("twisted", "2026-11-25"),
}
# The kinds whose result is a symmetric Hadamard matrix shown on the entry page of one quadruple.
SYMMETRIC_KINDS = ("propus", "twisted")
NEW_RESULT_GROUPS = (
    ("correction", "Contradicts a published result"),
    ("existence", "Existence"),
    ("nonexistence", "Nonexistence"),
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
                # Order 260 was known before by another construction: what is new is the propus family at n.
                "text": f"Propus matrices of order {n}: at least 1 class.",
                # The entry page carries the propus array and the matrix file.
                "href": f"entries/{entries[0]['id']}.html",
            }
        elif kind == "twisted":
            if len(entries) != 1 or entries[0]["type"] != "quadruple" or not twisted.shape(display_rows(entries[0])[0]):
                raise ValueError(f"existence result is not one quadruple tied by a multiplier: {ident}")
            n = entries[0]["n"]
            item |= {
                "n": n,
                "stratum": "general",
                "text": f"A symmetric Hadamard matrix of order {4 * n}.",
                # The entry page carries the array, the argument for it and the matrix file.
                "href": f"entries/{entries[0]['id']}.html",
            }
        else:
            orders = {x["n"] for x in entries}
            if len(orders) != 1 or any(x["type"] != "quadruple" or "skew-sds" not in x["tracks"] for x in entries):
                raise ValueError(f"existence result is not skew quadruples of one order: {ident}")
            n = orders.pop()
            # Doubling a skew-Hadamard matrix gives one of twice the order; name that order too when
            # none of the cited sources gives a matrix of it.
            skew_orders = ORDERS["types"]["skew-hadamard"]
            double = 8 * n if 8 * n in skew_orders["listed_open"] and 8 * n not in skew_orders["open"] else None
            # Every line names its own order only; the section says once that doubling applies to all of them.
            text = f"A skew-Hadamard matrix of order {4 * n}."
            item |= {"n": n, "stratum": "skew-sds", "text": text, "double": double}
        items.append(item)
    # The skew-Hadamard results stand together, in order of the matrix order; the symmetric ones follow, in
    # order of the matrix order.
    skew_items = [x for x in items if x["kind"] not in SYMMETRIC_KINDS] + matrix_results()
    # The propus family is a class count, so it follows the matrix lines and leads the counts.
    symmetric = sorted(
        (x for x in items if x["kind"] in SYMMETRIC_KINDS and x["kind"] != "propus"), key=lambda x: x["n"]
    )
    return skew_items + symmetric + [x for x in items if x["kind"] == "propus"]


def matrix_results() -> list[JSONObject]:
    """The stored skew-Hadamard matrices found for this board: one line each, linked to the page that describes it."""
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
    return symmetric_entries(stats, "propus")


def symmetric_entries(stats: JSONObject, wanted: str) -> set[str]:
    """The entries of the existence results of one symmetric kind, featured or not: their pages carry the array."""
    kinds = {ident: kind for ident, (kind, _) in NEW_EXISTENCE.items()}
    return {e for g in stats["contributions"] if kinds.get(g["id"]) == wanted for e in g["entry_ids"]}


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
            kind = "existence"
            item["kind"] = "count"
        elif c["status"] == "empty":
            # A line beside replayed results must not read as one: a search the
            # board has not rerun carries the wording of its cell.
            text = f"{subject}: none exist."
            if "reproduced" not in c["empty_basis"]:
                item["qualifier"] = "; ".join(EVIDENCE_WORDS[b] for b in c["empty_basis"])
            kind = "nonexistence"
        else:
            raise ValueError(f"new result without a census or an empty cell: {track} n={n}")
        if c["contradicted"]:
            kind = "correction"
            item["source"] = c["contradicted"][0]["source"]
        groups[kind].append(item | {"text": text})
    # Matrices first, then the counts of classes, which also say that the matrices exist.
    groups["existence"] = existence_results(stats, r) + groups["existence"]
    return [(title, groups[key]) for key, title in NEW_RESULT_GROUPS if groups[key]]


def new_results_section(stats: JSONObject, r: Results) -> str:
    parts = ['<section id="new-results"><h2>New results</h2>']
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
                link += " " + esc(CONTRADICTION_NOTES[key]).replace("{cite}", BIB.mark(key))
            parts.append(f"<li>{link}</li>")
        parts.append("</ul>")
    parts.append("</section>")
    return "".join(parts)


def cell_description(c: JSONObject) -> str:
    """The meta description of a cell page, for search results and link previews."""
    n, track = c["n"], c["stratum"]
    singular, many = nouns(track, n)
    count = max(len(c["quadruples"]), sum(x["rows"] for x in c.get("catalogues", [])))
    if count:
        kind = "skew-Hadamard" if track in SKEW_TRACKS else "Hadamard"
        return (
            f"{plural(count, 'exactly verified ' + singular, 'exactly verified ' + many)}, "
            f"each giving a {kind} matrix of order {4 * n}."
        )
    return {
        "exists": f"{many} exist by a published result; the board holds none yet.",
        "empty": f"No {singular} exists.",
        "unresolved": f"Open: no {singular} is known in the sources the board cites.",
    }[c["status"]]


def entry_description(doc: JSONObject, result: JSONObject) -> str:
    """The meta description of an entry page."""
    ident, n = doc["id"], doc["n"]
    if doc["type"] == "quadruple":
        _, pattern = display_rows(result)
        kind = "skew-Hadamard" if "k" in pattern else "Hadamard"
        return (
            f"Goethals–Seidel quadruple {ident} of length n = {n}, exactly verified. Its array "
            f"is a {kind} matrix of order {4 * n}."
        )
    label = TRACK_LABELS[result["stratum"]]
    if doc["type"] == "census":
        return f"Census record {ident} at n = {n}, {label}: {plural(result['count'], 'class', 'classes')}."
    what = certificate_label(result) or "attributed claim"
    return f"Evidence record {ident} at n = {n}, {label}: {what}."


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
            " The verifier output at the foot of this page lists the same rows in sorted order, "
            "in which the skew row is not first."
        )
    lines = "\n".join(f"{name} {letter} {signs(row)}" for name, letter, row in zip("ABCD", pattern, rows, strict=True))
    return f'<section class="rows-text"><h2>Rows</h2><p>{esc(text)}</p><pre>{esc(lines)}</pre></section>'


def two_odd_squares(m: int) -> bool:
    """Whether m is a sum of two odd squares."""
    return any(
        (rest := m - a * a) >= 0 and math.isqrt(rest) ** 2 == rest and math.isqrt(rest) % 2 == 1
        for a in range(1, math.isqrt(m) + 1, 2)
    )


def impossible(c: JSONObject) -> str:
    """Why a cell is empty by arithmetic, as a clause, or '' when no such rule applies."""
    n, track = c["n"], c["stratum"]
    if c["status"] != "empty":
        return ""
    if track == "best" and math.isqrt(4 * n - 3) ** 2 != 4 * n - 3:
        # Best matrices need 4n - 3 to be a square.
        return f"best matrices need 4n − 3 to be a square, and {4 * n - 3} is not one"
    if track == "kkss" and n % 2 and not two_odd_squares(4 * n - 2):
        # Two skew blocks have row sum 1, so the other two row sums are odd with squares adding to 4n - 2.
        return f"with two skew blocks, 4n − 2 must be a sum of two odd squares, and {4 * n - 2} is not one"
    return ""


def published_classes(c: JSONObject) -> JSONObject | None:
    """The cell's baseline record with a published class count for this very track, if any."""
    return next((s for s in c["sources"] if "published_classes" in s and s["stratum"] == c["stratum"]), None)


def source_marks(c: JSONObject, prefix: str) -> str:
    """The reference numbers of the cell's literature sources, each once, with a leading space."""
    keys = dict.fromkeys(source_keys(s["source"] for s in c["sources"]))
    return "".join(f" {BIB.mark(key, prefix)}" for key in keys)


# How many lines of a catalogue a cell page draws; the rest are in the text file.
CATALOGUE_SHOWN = 12


def cell_catalogue(c: JSONObject) -> JSONObject | None:
    """The catalogue of a cell, if it has one: the verifier's record of the file and its checked numbers."""
    return c["catalogues"][0] if c.get("catalogues") else None


def catalogue_lines(cat: JSONObject) -> list[str]:
    """The lines of a catalogue file. The board load has checked every one of them."""
    return catalogue.lines(ROOT / cat["path"])


def catalogue_file(cat: JSONObject) -> str:
    """The name of the text file the site serves for a catalogue, under catalogues/."""
    return f"{cat['census']}.txt"


def outside_catalogue(c: JSONObject, r: Results) -> list[str]:
    """The quadruple entries of a cell that are not lines of its catalogue, which add to its count."""
    cat = cell_catalogue(c)
    if not cat or not c["quadruples"]:
        return list(c["quadruples"])
    present = set(catalogue_lines(cat))
    return [i for i in c["quadruples"] if catalogue.format_line(r[i]["canonical_rows"]) not in present]


def held_count(c: JSONObject, r: Results) -> int:
    """The number of pairwise inequivalent verified quadruples the board holds for a cell."""
    cat = cell_catalogue(c)
    return (cat["rows"] if cat else 0) + len(outside_catalogue(c, r))


def line_figure(n: int, line: str, caption: str) -> str:
    """The picture of one catalogue line, drawn like a quadruple entry."""
    rows: Rows = catalogue.parse_line(line, n)
    pattern = "".join("k" if "k" in row_types(x) else "s" if "s" in row_types(x) else "·" for x in rows)
    return quad_figure({"n": n}, {"canonical_rows": rows, "pattern": pattern}, caption)


def own_pages(c: JSONObject, r: Results) -> str:
    """The end of the heading over the quadruple entries of a cell that also has a catalogue file."""
    extra, count = len(outside_catalogue(c, r)), len(c["quadruples"])
    where = ", also in the file" if extra == 0 else ", not in the file" if extra == count else ""
    return f" with {'a page of its own' if count == 1 else 'pages of their own'}{where}"


def cell_sources(c: JSONObject, d: Results) -> list[JSONObject]:
    """The sources a cell page lists: the cited records of the cell, then those of its catalogue.

    A cell with no cited record lists the sources of the quadruples it holds, so that a quadruple built by a
    published construction shows where the construction is from.
    """
    found = [s["source"] for s in c["sources"]]
    cat = cell_catalogue(c)
    if cat and cat["stratum"] == c["stratum"]:
        found += d[cat["census"]]["sources"]
    if not found:
        found = [s for ident in c["quadruples"][:20] for s in d[ident]["sources"]]
    unique = {(s["title"], s["url"]): s for s in found}
    return list(unique.values())


def cell_tag(c: JSONObject, r: Results) -> str:
    """The status chip of a cell page, in words."""
    held = held_count(c, r)
    if c["status"] == "exists":
        return (
            f"exists · {plural(held, 'quadruple')} here, rows verified" if held else "exists · published, none here yet"
        )
    if c["status"] == "unresolved":
        return "open"
    basis = "by arithmetic" if impossible(c) else evidence_words(c["tier"])
    if not basis:
        basis = "; ".join(EVIDENCE_WORDS[b] for b in c["empty_basis"]) or "published result, not checked here"
    return f"none exists · {basis}"


def cell_lead(c: JSONObject, r: Results, prefix: str = "../") -> str:
    """The opening sentences of a cell page: what is known and what the board holds, as HTML."""
    n, track = c["n"], c["stratum"]
    singular, many = nouns(track, n)
    held = len(c["quadruples"])
    marks = source_marks(c, prefix)
    census = r[c["censuses"][0]] if c["censuses"] else None
    if c["status"] == "unresolved":
        text = (
            f"Open: no {singular} is known to us in the sources the board cites, and the board "
            "holds no proof that none exists."
        )
        if c["evidence"]:
            text += " The records below cover part of the search and do not settle it."
        return text
    if c["status"] == "empty":
        if rule := impossible(c):
            return f"No {singular} exists: {rule}."
        if c["tier"] == "reproduced":
            also = f" The same is published{marks}." if marks else ""
            return f"No {singular} exists: a complete search rerun here found none.{also}"
        basis = "; ".join(EVIDENCE_WORDS[b] for b in c["empty_basis"]) or "published result, not checked here"
        return f"No {singular} exists{marks}: {basis}."
    if cat := cell_catalogue(c):
        return catalogue_lead(c, cat, r, prefix)
    if not held:
        published = published_classes(c)
        if published:
            count = published["published_classes"]["count"]
            return (
                f"{many[0].upper()}{many[1:]} exist: {plural(count, 'class', 'classes')} "
                f"{'is' if count == 1 else 'are'} published{marks}. The board holds none of them yet."
            )
        return f"{many[0].upper()}{many[1:]} exist by a published result{marks}. The board holds none yet."
    kind = "skew-Hadamard" if track in SKEW_TRACKS else "Hadamard"
    gives = f" {'It gives' if held == 1 else 'Each gives'} a {kind} matrix of order {4 * n}."
    verified = f"{plural(held, 'verified ' + singular, 'verified ' + many)}"
    if census is None:
        distinct = ", no two in the same class" if held > 1 else ""
        return f"The board holds {verified}{distinct}. No complete search is recorded here.{gives}"
    count, complete = census["count"], census["bound"] == "exact"
    classes = plural(count, "class", "classes")
    if census.get("count_only"):
        rerun = "A complete count rerun here gives" if complete else "A complete count, not rerun in this build, gives"
        text = f"The board holds {verified}. {rerun} {classes} when {SHIFT_CLASS}; the classes are not all stored."
    elif census["equivalence"] == "equiv-v2":
        end = (
            ", and a complete search rerun here found no other class."
            if complete
            else "; a complete search, not rerun in this build, found no other class."
            if census["claim_kind"] == "computed-exact"
            else "; the list is not known to be complete."
        )
        text = f"The board holds {verified}. They fall into {classes} when {SHIFT_CLASS}{end}"
    elif complete and count == held:
        text = (
            f"The board holds {plural(held, singular, many)}, "
            f"{'the only class' if held == 1 else 'one per class'}. A complete search rerun here found no others."
        )
    elif published_count(census) and count == held:
        text = (
            f"The board holds {plural(held, singular, many)}, "
            f"{'the only class' if held == 1 else 'one per class'} of the published classification{marks}. "
            "The search has not been rerun here."
        )
    elif census["claim_kind"] == "computed-exact" and count == held:
        text = (
            f"The board holds {plural(held, singular, many)}, "
            f"{'the only class' if held == 1 else 'one per class'} of a complete search that this build did not rerun."
        )
    else:
        distinct = ", no two in the same class" if held > 1 else ""
        text = f"The board holds {verified}{distinct}. The list is not known to be complete."
    return text + gives


def catalogue_lead(c: JSONObject, cat: JSONObject, r: Results, prefix: str) -> str:
    """The opening sentences of a cell whose quadruples are in a catalogue file."""
    n, track = c["n"], c["stratum"]
    singular, many = nouns(track, n)
    rows, classes = cat["rows"], cat.get("published_classes")
    extra = len(outside_catalogue(c, r))
    marks = source_marks(c, prefix)
    gives = f" Each gives a Hadamard matrix of order {4 * n}."
    if cat["stratum"] != track:
        other_singular, other_many = nouns(cat["stratum"], n)
        more = f", and {plural(extra, 'other verified ' + singular, 'other verified ' + many)}" if extra else ""
        return (
            f"The board holds {plural(rows, 'verified ' + other_singular, 'verified ' + other_many)}{more}; every one "
            f"is a {singular.split(' of length')[0]}. They are listed on the page of the "
            f'<a href="n{n}-{esc(cat["stratum"])}.html">{esc(TRACK_LABELS[cat["stratum"]])}</a> cell.{gives}'
        )
    also = (
        f" The board holds {plural(extra, 'more ' + singular, 'more ' + many)} with "
        f"{'a page of its own' if extra == 1 else 'pages of their own'}, listed below."
        if extra
        else ""
    )
    if classes and cat["split"]:
        return (
            f"The board holds {plural(rows, singular, many)}, no two in the same class: the "
            f"{plural(classes, 'class', 'classes')} of the published enumeration{marks}, split under the board's "
            "finer equivalence. Every one is checked exactly here. The list is complete if the published one is; "
            f"no search was rerun here.{also}{gives}"
        )
    if classes:
        return (
            f"The board holds {plural(rows, singular, many)}, one from each of the "
            f"{plural(classes, 'class', 'classes')} of the published enumeration{marks}. Every one is checked "
            f"exactly here; no search was rerun here.{also}{gives}"
        )
    return (
        f"The board holds {plural(rows, 'verified ' + singular, 'verified ' + many)}, no two in the same class. "
        f"The list is not known to be complete.{also}{gives}"
    )


def cell_rows_text(c: JSONObject, r: Results) -> str:
    """The quadruples of a cell as a text file: one per line, its four rows as sign strings, in page order."""
    return "".join(" ".join(signs(row) for row in display_rows(r[ident])[0]) + "\n" for ident in c["quadruples"])


def record_words(result: JSONObject) -> str:
    """One evidence or census record of a cell, in words."""
    if result["type"] == "certificate":
        detail = "nonexistence record"
        if label := certificate_label(result):
            detail += f" · {label}"
        if result["scope"]["kind"] != "full":
            detail += f" · covers only {scope_text(result['scope'])}"
        return detail
    if result["bound"] != "exact" and result["count"] == 0:
        return "class list · published as empty, not rerun here"
    if cat := result.get("catalogue"):
        detail = f"catalogue file · {plural(cat['rows'], 'quadruple')}, every line checked here"
        if "published_classes" in cat:
            how = "every class inside them under the board's equivalence" if cat["split"] else "one quadruple each"
            detail += f" · {plural(cat['published_classes'], 'published class', 'published classes')}, {how}"
        return detail
    known = "" if result["bound"] == "exact" or published_count(result) else " known"
    detail = f"class list · {plural(result['count'], 'class', 'classes')}{known}"
    if result["equivalence"] == "equiv-v2":
        detail += " · shifted blocks count as the same"
    if result.get("count_only"):
        detail += " · a count only"
    if published_count(result):
        detail += " · published count"
    if result.get("tier"):
        detail += f" · {EVIDENCE_WORDS[result['tier']]}"
    return detail


def cell_page(c: JSONObject, d: Results, r: Results, matrix: str = "") -> str:
    """One page per order and track: what is known in one sentence, the quadruples, credit and sources.

    matrix is the block about the skew-Hadamard matrix of order 4n, which the
    page of the track for quadruples with a skew block carries. Visible text
    uses no schema slug; the track and status values stay in data attributes.
    """
    n, track = c["n"], c["stratum"]
    parts = [
        f'<article class="entry" data-track="{esc(track)}" data-status="{esc(c["status"])}" '
        f'data-artifact="{esc(c["artifact_status"])}"><div class="eyebrow">board cell · length {n} · '
        f"matrix order {4 * n}</div><h1>n = {n} · {esc(TRACK_LABELS[track])}</h1>"
        f'<span class="tag">{esc(cell_tag(c, r))}</span><p class="census">{cell_lead(c, r)}</p>'
    ]
    published = published_classes(c)
    cat = cell_catalogue(c)
    own = cat if cat and cat["stratum"] == track else None
    if c["censuses"] or published:
        words = []
        if c["censuses"] and c["status"] == "exists":
            words.append(SAME_CLASS)
        if published and own and not own["split"]:
            words.append(
                f"{PUBLISHED_CLASSES} The last two are not operations of the board's equivalence, under which a "
                "published class splits into as many as 32 classes; those are not listed for this length."
            )
        elif published and (c["quadruples"] or own):
            count = published["published_classes"]["count"]
            key = BIB.resolve(published["source"]["url"])
            words.append(
                f"Published count: <strong>{plural(count, 'class', 'classes')}</strong>"
                f"{' ' + BIB.mark(key, '../') if key else ''}. {PUBLISHED_MERGES}"
            )
        elif published:
            words.append(PUBLISHED_CLASSES)
        if words:
            parts.append(f'<p class="muted">{" ".join(words)}</p>')
    if c["note"] and c["status"] != "empty":
        parts.append(f"<p>{esc(c['note'])}</p>")
    for claim in c["contradicted"]:
        parts.append(
            '<p class="contradiction">The verified rows contradict a published result: '
            f'<a href="{esc(BIB.link(claim["source"]["url"]))}" rel="noopener">'
            f"{esc(claim['source']['title'])}</a>"
            + "".join(f" {BIB.mark(key, '../')}" for key in source_keys([claim["source"]]))
            + "</p>"
        )
    parts.append(matrix)
    if own:
        shown = catalogue_lines(own)[:CATALOGUE_SHOWN]
        first = "It is drawn below." if own["rows"] == 1 else f"The first {len(shown)} are drawn below."
        parts.append(
            f"<h2>{plural(own['rows'], 'quadruple')} in one file</h2>"
            f'<p><a href="../catalogues/{esc(catalogue_file(own))}" download>Download all {own["rows"]:,} as text</a>: '
            "one quadruple per line, its four rows separated by spaces, + for 1 and - for −1, each line the "
            f"least member of its class. {first} A compressed copy is "
            f'<a href="{REPOSITORY}/blob/main/{esc(own["path"])}" rel="noopener">in the repository</a>. '
            f"The rows were imported from the sources listed at the foot of this page"
            f"{''.join(' ' + BIB.mark(key, '../') for key in dict.fromkeys(source_keys(d[own['census']]['sources'])))}"
            ".</p>"
            f'{PICTURE_KEY}<div class="gallery">'
        )
        parts.extend(line_figure(n, line, f"line {k}") for k, line in enumerate(shown, 1))
        parts.append("</div>")
    if c["quadruples"]:
        count = len(c["quadruples"])
        parts.append(
            f"<h2>{plural(count, 'quadruple')}{own_pages(c, r) if own else ''}</h2>"
            f'<p><a href="n{n}-{esc(track)}.txt" download>Download {"it" if count == 1 else f"all {count:,}"} '
            "as text</a>: "
            "one quadruple per line, its four rows separated by spaces, + for 1 and - for −1. "
            f"Each name below opens the page of that quadruple.</p>{PICTURE_KEY}"
            '<div class="gallery">'
        )
        for ident in c["quadruples"]:
            result = r[ident]
            caption = f'<a href="../entries/{esc(ident)}.html">{esc(ident)}</a>'
            parts.append(quad_figure(d[ident], result, caption))
        parts.append("</div>")
    # The catalogue credit is the same paragraph for every imported cell; the
    # cell's own sources already name the original authors. A contribution that
    # reports quadruples of the cell comes before one that only holds evidence,
    # so that a note about a restricted family follows the result it qualifies.
    parts.extend(
        credit_block(group, "../")
        for group in sorted(c["contributions"], key=lambda g: not set(g["entry_ids"]) & set(c["quadruples"]))
        if group["id"] != "published-catalogues"
    )
    records = [*c["evidence"], *c["censuses"]]
    if records:
        items = [
            f'<li><a href="../entries/{esc(ident)}.html">{esc(ident)}</a> · {esc(record_words(r[ident]))}</li>'
            for ident in records
        ]
        parts.append(f"<h2>Search and evidence records</h2><ul>{''.join(items)}</ul>")
    if sources := cell_sources(c, d):
        parts.append("<h2>Sources</h2><ul>" + "".join(source_item(s, "../") for s in sources) + "</ul>")
    parts.append('<p><a href="../#board">Back to the board</a></p></article>')
    if c["quadruples"] or own:
        parts.append('<script src="../picture.js" defer></script>')
    return "".join(parts)


def grid_cell(c: JSONObject, marked: Collection[tuple[str, int]], r: Results) -> JSONObject:
    """What the front-page grid shows for one board cell, and the words of its tooltip."""
    track, n = c["stratum"], c["n"]
    label = (
        "Contradicts published"
        if c["contradicted"]
        else "Verified"
        if c["quadruples"] or c.get("catalogues")
        else "Published"
        if c["artifact_status"] == "published-not-imported"
        else "Empty"
        if c["status"] == "empty"
        else "Unresolved"
    )
    spoken = {
        "Contradicts published": "exists, contradicting a published result",
        "Verified": "exists",
        "Published": "exists by a published result",
        "Empty": "none exists",
        "Unresolved": "open",
    }[label]
    small = tier_label(c["tier"], rows=bool(c["quadruples"] or c.get("catalogues"))) or (
        "; ".join(EVIDENCE_WORDS[b] for b in c["empty_basis"]) or "published result, not checked here"
        if c["status"] == "empty"
        else "partial search records"
        if c["evidence"]
        else ""
    )
    origin_words = ", ".join(ORIGIN_LABELS[o] for o in c["origins"])
    accessible = "; ".join(x for x in (f"n = {n}, {TRACK_LABELS[track]}: {spoken}", small, origin_words) if x)
    origins = list(c["origins"])
    new = (track, n) in marked
    if new and "participant-result" not in origins:
        origins.insert(0, "participant-result")
    count = census_count(c, r)
    if count:
        accessible += f"; {count}"
    elif cat := cell_catalogue(c):
        accessible += f"; {plural(cat['rows'], 'quadruple')} in a catalogue file, no complete class count"
    elif label in ("Verified", "Contradicts published"):
        accessible += "; no class count"
    elif label == "Published":
        accessible += "; the board holds no quadruple of this type yet"
    elif label == "Unresolved":
        accessible += "; no quadruple of this type is known to us in the cited sources"
    if published := published_classes(c):
        number = plural(published["published_classes"]["count"], "class", "classes")
        accessible += f"; {number} published, under an equivalence that merges more than the board's"
    text = (
        count.replace(" classes", "").replace(" class", "")
        if count
        else {"Empty": "0", "Unresolved": "?"}.get(label, "✓")
    )
    if not count and (held := held_count(c, r)):
        # The cell holds quadruples without a complete census: show how many. A plus sign marks a list that is
        # not known to be every class; a catalogue that splits every published class is complete if the source is.
        cat = cell_catalogue(c)
        whole = bool(cat and cat.get("split") and cat["stratum"] == track and not outside_catalogue(c, r))
        text = f"{held:,}" if whole else f"{held:,}+"
        accessible += (
            "; every class of the published enumeration, complete if that enumeration is"
            if whole
            else f"; the board holds {plural(held, 'quadruple')}, no two in the same class, and there may be more"
        )
    if rule := impossible(c):
        # Empty by a theorem and not by a search: the cell shows a cross, not a zero.
        text = "×"
        accessible += f"; cannot exist: {rule}"
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


def write_above_matrix(out: Path, record: JSONObject, quick: bool = False) -> None:
    """Check a found matrix of an order above the range from its record and write its file, or its rows.

    Up to skew.BUILD_REBUILD_MAX the matrix is rebuilt: its text must have the
    recorded sha256, in a quick build too, and a full build runs the exact
    check. Above that order the build checks in integers what the array asks of
    the rows and does not rebuild; the tests marked slow rebuild the matrix,
    compare the sha256 and run the exact check. A matrix whose text is above
    skew.DOWNLOAD_LIMIT is not written: the site offers the rows of its
    record, one per line.
    """
    order = record["order"]
    if not skew.rebuilt_in_build(order):
        skew.checked_rows(record)
        text = ""
    else:
        text = skew.stored_text(record, check=not quick)
    if skew.has_download(order):
        (out / skew.file_name(order)).write_text(text)
    else:
        (out / skew.rows_file_name(order)).write_text("".join(line + "\n" for line in record["rows"]))


def write_skew_matrices(
    out: Path, states: orders.OrderStates, r: Results, quick: bool = False, above: bool = True
) -> dict[int, str]:
    """Write the matrix file of every skew-Hadamard order that is not open; return the chosen board entries.

    An order the board holds gets the array of an entry, and every other order
    its stored matrix. A stored file is decompressed, compared with its sha256
    and checked exactly before it is written; any failure raises ValueError and
    stops the build. A quick build compares the sha256 only and skips the exact
    check of the stored files, which the full build and the tests run. A found
    matrix of an order above the range has no stored file: write_above_matrix
    rebuilds it from its record. With above unset those are left out.
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
    for order, record in STORED.items():
        if above and orders.beyond_range(ORDERS, order):
            write_above_matrix(out, record, quick)
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


def write_twisted_matrices(out: Path, idents: Iterable[str], r: Results) -> None:
    """Write the symmetric array of each listed entry, a quadruple tied by a multiplier, as a matrix file.

    Call it after write_skew_matrices, which empties the folder. The matrix is
    rebuilt from the rows and checked exactly to be a symmetric Hadamard matrix
    before it is written; a failure raises ValueError and stops the build.
    """
    for ident in sorted(idents):
        try:
            text = twisted.matrix_file(display_rows(r[ident])[0])
        except ValueError as error:
            raise ValueError(f"the array of {ident} is not a symmetric Hadamard matrix: {error}") from error
        (out / twisted.file_name(ident)).write_text(text)


def write_order_pages(out: Path, states: orders.OrderStates, chosen: dict[int, str]) -> None:
    """Write the page of every skew-Hadamard order without a board cell for a skew block.

    These are the orders 4n of the grid with n even, and the orders above the
    range of the grid that have a found matrix.
    """
    folder = out / "orders"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    pages = {order: s[skew.NAME] for order, s in states.items()}
    pages |= {order: orders.beyond_state() for order in STORED if orders.beyond_range(ORDERS, order)}
    for order, info in pages.items():
        if orders.has_cell(order // 4, orders.CELL_TRACKS[skew.NAME]):
            continue
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
<section id="board"><h2>The board</h2><p class="key">{orders.intro(ORDERS)} (<a href="{REPOSITORY}/blob/main/TRACKS.md#reading-the-board">how to read the board</a>)</p><div class="controls"><label>Column <select id="track"><option value="">Every column</option>{options}</select></label><label>Show <select id="status"><option value="">Every status</option><option value="unresolved">Open</option><option value="published-not-imported">Published, none on the board</option><option value="exists">Existence known</option><option value="empty">Nonexistence</option></select></label><label>Origin <select id="origin"><option value="">Every origin</option><option value="participant-result">Challenge results</option><option value="independent-reproduction">Independent reproductions</option><option value="published-import">Literature / catalog</option></select></label><label>Through n <select id="max-order"><option>31</option><option>63</option><option>131</option><option selected value="{orders.BOARD_MAX_N}">{orders.BOARD_MAX_N}, every n</option></select></label></div>
<div class="legend"><span class="known">exists</span><span class="empty">empty</span><span class="unresolved">open</span><span class="contradicted">contradicts the literature</span><span class="impossible">cannot exist</span><span class="participant">new result</span></div>
<div class="grid-wrap"><table id="grid"><thead>{head}</thead><tbody>{rows}</tbody></table></div></section>
{new_results_section(stats, r)}
<section class="columns"><article class="card"><h3>Submit a result</h3><p><code>./gs verify candidate.json</code><br><code>./gs submit candidate.json</code></p><a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/CONTRIBUTING.md">Contribution guide ↗</a></article></section>
<noscript><p>JavaScript enables the filters and the matrix drawings. All data remain available in <a href="stats.json">stats.json</a> and <a href="entries.json">entries.json</a>.</p></noscript>
<script src="app.js" defer></script><script src="picture.js" defer></script><script src="hero.js" defer></script>"""  # noqa: E501
    (out / "index.html").write_text(shell("The board", body, head=MATH_HEAD))
    with_propus = propus_entries(stats)
    with_twisted = symmetric_entries(stats, "twisted")
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
            if ident in with_twisted:
                symmetric = twisted.block(ident, display_rows(result)[0], BIB)[0]
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
        if doc["type"] == "census":
            tag = (
                EVIDENCE_WORDS["reproduced"]
                if result["bound"] == "exact"
                else "published count, search not rerun here"
                if published_count(result)
                else "classes found so far, not known to be every class"
            )
        file = f"{FOLDERS[doc['type']]}/{ident}.json"
        raw = f"https://raw.githubusercontent.com/vxvware/goethals-seidel-challenge/main/{file}"
        links = (
            f'<br>Entry file: <a href="{REPOSITORY}/blob/main/{file}">{esc(file)}</a> · <a href="{raw}">raw JSON</a>'
        )
        notes = f"<h2>Notes</h2><p>{esc(doc['notes'])}</p>" if doc.get("notes") else ""
        text = (
            f'<article class="entry"><div class="eyebrow">{esc(ENTRY_KINDS[doc["type"]])} · n = {doc["n"]}</div>'
            f'<h1>{esc(ident)}</h1><span class="tag">{esc(tag or "claim not checked here")}</span>'
            f"{credits}<p>Discovered by: {esc('; '.join(doc['discovered_by']))}<br>"
            f"Submitted by: {esc(doc['submitted_by'])} · {doc['submitted_at']}<br>"
            f"Board cell: {cell_links(result, '../')}{links}</p>{formal}{picture}{notes}"
            f"<h2>Sources</h2><ul>{sources}</ul><details><summary>Verifier output (JSON)</summary>"
            f"<pre>{esc(json.dumps(result, indent=2, ensure_ascii=False))}</pre></details>"
            "<details><summary>Submission JSON</summary>"
            f"<pre>{esc(json.dumps(doc, indent=2, ensure_ascii=False))}</pre></details></article>"
        )
        (out / f"entries/{ident}.html").write_text(
            shell(ident, text, "../", description=entry_description(doc, result))
        )
    refs = (
        '<article class="entry"><div class="eyebrow">bibliography</div><h1>References</h1>'
        "<p>Every paper, dataset and software source cited on the board, numbered as in the citations ("
        '<a href="https://github.com/vxvware/goethals-seidel-challenge/blob/main/baselines/references.bib">'
        f"BibTeX</a>).</p>{BIB.listing()}"
        '<p><a href="./#board">Back to the board</a></p></article>'
    )
    (out / "references.html").write_text(shell("References", refs))
    states = orders.table(ORDERS, r, STORED)
    chosen = write_skew_matrices(out, states, r, quick)
    write_propus_matrices(out, with_propus, r)
    write_twisted_matrices(out, with_twisted, r)
    write_order_pages(out, states, chosen)
    # Files of the page this build no longer writes.
    for name in ("skew.html", "skew.js"):
        (out / name).unlink(missing_ok=True)
    folder = out / "catalogues"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    for c in stats["cells"]:
        for cat in c["catalogues"]:
            if cat["stratum"] == c["stratum"]:
                (folder / catalogue_file(cat)).write_text("".join(line + "\n" for line in catalogue_lines(cat)))
    for c in stats["cells"]:
        title = f"n = {c['n']} · {TRACK_LABELS[c['stratum']]}"
        matrix = ""
        if c["stratum"] == orders.CELL_TRACKS[skew.NAME]:
            matrix, _ = skew.matrix_block(ORDERS, 4 * c["n"], states[4 * c["n"]][skew.NAME], STORED, chosen, BIB)
        (out / f"cells/n{c['n']}-{c['stratum']}.html").write_text(
            shell(title, cell_page(c, d, r, matrix), "../", description=cell_description(c))
        )
        listing = out / f"cells/n{c['n']}-{c['stratum']}.txt"
        if c["quadruples"]:
            listing.write_text(cell_rows_text(c, r))
        else:
            listing.unlink(missing_ok=True)
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
