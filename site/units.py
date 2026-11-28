"""Pages for a census run as units of work: one page for the census, one per class.

The data come from the units manifests registered in baselines/census_replays.json,
validated by verify.census_units on every build (each log reparsed from its
archive, the independent coverage counts and the planted-solution record
checked against the logs). The pages state what each record checks and what it
does not; while the census runs they show the finished classes and list the
others as pending.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from orders import esc
from verify import census_units
from verify.common import ROOT, JSONObject

if TYPE_CHECKING:
    from collections.abc import Callable

    from references import Bibliography

REPOSITORY = "https://github.com/vxvware/goethals-seidel-challenge"
FOLDER = "census"
# The census each manifest belongs to, in words, for titles and links from the board cell.
TITLES = {"best-n73": ("best", 73, "Best matrices at n = 73")}
COUNTER_WORDS = {
    "pairs_index": "pairs (A, D) past the spectral index",
    "pairs_fine": "pairs past the 256-level spectral test",
    "patterns": "patterns t solved for",
    "patterns_table": "patterns past the consistency table",
    "systems_consistent": "consistent stage-2 systems",
    "candidates": "candidates checked exactly",
    "capped": "systems dropped at the nullity cap",
    "solutions": "solutions written",
}


def num(value: int) -> str:
    return f"{value:,}"


def source_link(path: str) -> str:
    return f'<a href="{REPOSITORY}/blob/main/{esc(path)}"><code>{esc(path)}</code></a>'


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h"


def theorem_items(formal: list[JSONObject], census: str) -> str:
    items = [
        f'<li><a href="{REPOSITORY}/blob/main/{esc(t["file"])}"><code>{esc(t["theorem"])}</code></a>: '
        f"{esc(t['statement'])}</li>"
        for t in formal
        if t["entry_id"] == census
    ]
    return f"<ul>{''.join(items)}</ul>" if items else ""


def unit_by_class(manifest: JSONObject) -> dict[int, JSONObject]:
    return {u["class"]: u for u in manifest["units"]}


def index_page(manifest: JSONObject, summary: JSONObject, formal: list[JSONObject], bib: Bibliography) -> str:
    census, n, classes = manifest["census"], manifest["n"], summary["classes"]
    header, cov, plant = manifest["header"], summary["coverage"], summary["plant"]
    units = unit_by_class(manifest)
    done = len(summary["done"])
    engine = next(iter(manifest["engine_sha256"]))
    sha = manifest["engine_sha256"][engine]
    replays = [(u["class"], r) for u in manifest["units"] for r in u.get("replays", [])]
    wall = sum(u["wall_seconds"] * u["threads"] for u in manifest["units"])
    status = (
        f"All {classes} classes have finished with no solution."
        if summary["complete"] and not summary["solutions"]
        else f"{done} of {classes} classes have finished, with {num(summary['solutions'])} solutions so far; "
        f"{classes - done} are pending. The board cell stays open until every class has finished and the "
        "census is registered as a replayed census entry."
    )
    rows = []
    for b in range(classes):
        u = units.get(b)
        p = plant["classes"].get(b)
        planted = f"{p['recovered']}/{p['planted']}" if p else "pending"
        if u is None:
            rows.append(
                f'<tr><td><a href="c{b}.html">{b}</a></td><td>pending</td><td></td><td>{num(cov["pairs"][b])}</td>'
                f"<td></td><td></td><td></td><td></td><td>{planted}</td></tr>"
            )
            continue
        c = u["counters"]
        rows.append(
            f'<tr><td><a href="c{b}.html">{b}</a></td><td>{esc(u["host"])}</td><td>{hours(u["wall_seconds"])}</td>'
            f"<td>{num(cov['pairs'][b])}</td><td>{num(c['pairs_index'])}</td><td>{num(c['patterns'])}</td>"
            f"<td>{num(c['capped'])}</td><td>{num(c['solutions'])}</td><td>{planted}</td></tr>"
        )
    replay_text = "; ".join(
        f'class <a href="c{b}.html">{b}</a> on {esc(r["host"])} ({r["threads"]} threads, {hours(r["wall_seconds"])})'
        for b, r in replays
    )
    hosts = "; ".join(
        f"{esc(name)} ({esc(h['cpu'])}, {esc(h['compiler'])}, "
        f"binary sha256 <code>{esc(h['binary_sha256'][:16])}</code>…)"
        for name, h in manifest["hosts"].items()
    )
    theorems = theorem_items(formal, census)
    passed = len(plant["passed"])
    plant_words = (
        f"Every one of the {classes} classes has passed: every plant recovered, none capped."
        if passed == classes
        else f"{passed} of {classes} classes have been tested so far, every plant recovered and none capped; "
        "the rest are pending."
        if passed
        else "No class has been tested yet."
    )
    bright = bib.mark("bright2019best", "../../")
    return f"""<article class="entry"><div class="eyebrow">census by class · n = {n}</div>
<h1>{esc(TITLES[census][2])}: the census by class</h1>
<p class="census">{status}</p>
<p>Order 73 was left open in the published classification {bright}. The census runs the board's pinned engine {source_link(engine)} (sha256 <code>{esc(sha)}</code>, engine id <code>{esc(manifest["engine_id"])}</code>) once per class. It enumerates the symmetric row D and one skew row A and solves for the other two skew rows B and C through two GF(2) stages of the PAF equations; the source comment lists every reduction and why it loses no class. The pairs (A, D) are split into {classes} classes by a {manifest["class_bits"]}-bit label, cls(A) xor cls(D), where each label is a set of linear forms on the parity word of the row (comment C6 of the source); a run with <code>BEST_QB={manifest["class_bits"]} BEST_CLASS=b</code> solves exactly the pairs of class b. Each class has a page below, with its log once it has finished.</p>
<h2>What is checked</h2>
<p>Logs. Every build of the board reads each finished class's log from <a href="{REPOSITORY}/blob/main/{esc(manifest["logs"]["path"])}">the archive</a>, checks its sha256 and reparses it: the class it ran, the header every class shares (D list: {num(header["d_row_sum"])} rows with the row sum, {num(header["d_ceiling"])} under the spectrum ceiling, {num(header["d_minima"])} least in their multiplier orbit, {num(header["d_signature"])} with the stage-1 signature; A rows: {num(header["a_ceiling"])} of {num(header["a_codes"])} under the ceiling, in {header["a_pieces"]} pieces whose sizes must match), and the counters, including 0 systems dropped at the nullity cap. The logs are what the engine printed. Any class can be run again with one command (about 70 to 80 minutes on three laptop threads) and its counters compared; this was done for {replay_text}, and every counter matched. Hosts: {hosts}.</p>
<p>Coverage. {source_link(census_units.INDEPENDENT["coverage"])}, written from the definitions and sharing no code with the engine, enumerates the D rows and all {num(header["a_codes"])} A codes again and counts them by class label (<a href="{REPOSITORY}/blob/main/{esc(manifest["coverage"]["path"])}">output</a>). Its totals equal the engine's at every step of the D list and for the A rows. No D row has a PSD within 1e-6 of its bound; {num(cov["a_near"])} A rows have their largest PSD within 1e-6 of 4n − 2, {num(cov["a_near_above"])} of them above it, the closest at a distance of {cov["a_near_least_gap"]:.1e} (each recomputed by the board). Both programs keep every row up to the bound plus 1e-6, with rounding errors far below that, so no row under a bound is lost and a row kept above one only adds pairs. The number of pairs of class b is the sum over labels l of the D rows of label l times the A rows of label l xor b. These {classes} numbers add up to {num(header["d_signature"])} × {num(header["a_ceiling"])} = {num(cov["total_pairs"])}, every pair having exactly one class. What this does not check: that the engine meets every pair of its class. The engine drops pairs by the pairwise spectral bounds before solving, and here only the bound "pairs past the index ≤ pairs of the class" is tested; whether the pairwise filter keeps every pair that meets the bounds is not rechecked here.</p>
<p>Planted solutions. {source_link(census_units.INDEPENDENT["plant"])} includes the engine source unchanged and, for every class, plants {plant["classes"][min(plant["classes"])]["planted"] if plant["classes"] else 40} solutions of the GF(2) systems: random skew rows A, B, C with the pattern of B and C in the class, and a virtual D whose PAF values complete them exactly. The engine's own consistency table for the class, A table, index scan, stage-1 patterns, batched stage-2 solve and exact check must return each one (<a href="{REPOSITORY}/blob/main/{esc(manifest["plant"]["path"])}">output</a>). {plant_words} The class forms the engine uses equal the ones the coverage program derives. What this does not check: the virtual D is not a ±1 row, so the D enumeration and the spectral tests on real rows are not exercised here, and the rule that picks A among the three skew rows by the spectrum of D is switched off for the plants.</p>
<p>Lean. The mathematical steps the engine relies on are proved in Lean in <a href="{REPOSITORY}/tree/main/formal">formal/</a>: {theorems or "none linked"} The proofs cover the identities, not the C code that implements them, and not the floating-point evaluation of the spectral bounds.</p>
<h2>Classes</h2>
<p class="muted">Pairs in class: from the coverage counts. Past index, patterns, capped and solutions: from the class's log. Planted: plants recovered of plants made.</p>
<div class="grid-wrap"><table><thead><tr><th>class</th><th>host</th><th>wall</th><th>pairs in class</th><th>past index</th><th>patterns</th><th>capped</th><th>solutions</th><th>planted</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">Thread-hours of the finished classes: {wall / 3600:,.0f}.</p></article>"""  # noqa: E501


def class_page(manifest: JSONObject, summary: JSONObject, b: int) -> str:
    census, n, classes = manifest["census"], manifest["n"], summary["classes"]
    unit = unit_by_class(manifest).get(b)
    cov, plant = summary["coverage"], summary["plant"]["classes"].get(b)
    engine = next(iter(manifest["engine_sha256"]))
    run = manifest["run"].replace("<class>", str(b)).replace("<threads>", "3")
    parts = [
        f'<article class="entry"><div class="eyebrow">census by class · n = {n}</div>'
        f"<h1>{esc(TITLES[census][2])}: class {b} of {classes}</h1>"
    ]
    if unit is None:
        parts.append('<p class="census">This class has not finished yet.</p>')
    else:
        host = manifest["hosts"][unit["host"]]
        c = unit["counters"]
        parts.append(
            f'<p class="census">Finished on {esc(unit["host"])} at {esc(unit["finished"])} after '
            f"{hours(unit['wall_seconds'])} on {unit['threads']} threads: {num(c['solutions'])} solutions, "
            f"{num(c['capped'])} systems dropped at the nullity cap.</p>"
        )
        counters = "".join(
            f"<tr><td>{esc(COUNTER_WORDS[k])}</td><td>{num(c[k])}</td></tr>" for k in census_units.COUNTERS
        )
        parts.append(f"<h2>Counters from the log</h2><table><tbody>{counters}</tbody></table>")
        parts.append(
            f'<p>Log: <a href="c{b}.log">c{b}.log</a>, sha256 <code>{esc(unit["log_sha256"])}</code>. '
            f"Source: {source_link(engine)}, sha256 <code>{esc(manifest['engine_sha256'][engine])}</code>. "
            f"Binary on {esc(unit['host'])}: sha256 <code>{esc(host['binary_sha256'])}</code> "
            f"({esc(host['compiler'])}, {esc(host['cpu'])}).</p>"
        )
        for r in unit.get("replays", []):
            parts.append(
                f"<p>Run again on {esc(r['host'])} ({r['threads']} threads, {hours(r['wall_seconds'])}): every "
                f'counter equal. <a href="{esc(r["log"].removeprefix("replays/"))}">Log</a>, sha256 '
                f"<code>{esc(r['log_sha256'])}</code>.</p>"
            )
    nd, na = cov["dlabel"], cov["alabel"]
    parts.append(
        "<h2>Rows and pairs of the class</h2>"
        f"<p>Every D row with the signature ({num(sum(nd.values()))}) enters every class; a D row of label l meets "
        f"the A rows of label l xor {b}. The class holds {num(cov['pairs'][b])} pairs (A, D) before the pairwise "
        "spectral bounds, counted by the independent coverage program from the "
        f"{len(nd)} D and {len(na)} A label counts.</p>"
    )
    if plant:
        parts.append(
            f"<p>Planted solutions: {plant['planted']} planted, {plant['recovered']} recovered, "
            f"{plant['capped']} capped.</p>"
        )
    else:
        parts.append("<p>Planted solutions: not yet run for this class.</p>")
    parts.append(
        "<h2>Replay</h2><p>From a clone of the repository, with gcc and OpenMP:</p>"
        f"<pre>uv run python tools/replay_census_unit.py {esc(census)} {b} --threads 3</pre>"
        "<p>The tool checks the source against its sha256, builds it, runs the class (about 70 to 80 minutes on "
        "three laptop threads, 2.5 GB of memory; the first run also writes the D list) and compares the header and "
        f"the counters with this page. By hand, after <code>{esc(manifest['build'])}</code>:</p>"
        f"<pre>{esc(run)}</pre>"
        '<p><a href="./">All classes</a></p></article>'
    )
    return "".join(parts)


def write(
    out: Path, shell: Callable[..., str], formal: list[JSONObject], bib: Bibliography
) -> dict[tuple[str, int], str]:
    """Write the pages of every registered units manifest; return the link of each census by (track, n)."""
    links: dict[tuple[str, int], str] = {}
    for manifest, summary in census_units.registered(ROOT):
        census = manifest["census"]
        track, n, title = TITLES[census]
        folder = out / FOLDER / census
        folder.mkdir(parents=True, exist_ok=True)
        archive = census_units.read_archive(ROOT / manifest["logs"]["path"])
        for name, data in archive.items():
            (folder / name.removeprefix("replays/")).write_bytes(data)
        (folder / "index.html").write_text(
            shell(
                f"{title}: census by class",
                index_page(manifest, summary, formal, bib),
                "../../",
                description=f"{title}: the census by class, {len(summary['done'])} of {summary['classes']} finished.",
            )
        )
        for b in range(summary["classes"]):
            (folder / f"c{b}.html").write_text(
                shell(
                    f"{title}: class {b}",
                    class_page(manifest, summary, b),
                    "../../",
                    description=f"{title}: class {b} of {summary['classes']} of the census.",
                )
            )
        links[(track, n)] = f"{FOLDER}/{census}/"
    return links
