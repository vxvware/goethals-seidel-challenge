"""The board's bibliography: parse baselines/references.bib, cite it, render it.

Prose cites a work with the marker [@key]. A source URL in an entry is matched
to its bibliography entry through the doi, eprint, url and urlalias fields, so a
paper named anywhere on the board resolves to one full reference.
"""

import html
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

type Entry = dict[str, str]
"""One bibliography entry: its fields, plus its type and key."""

BIB = Path(__file__).resolve().parents[1] / "baselines/references.bib"
MARKER = re.compile(r"\[@([A-Za-z0-9_:-]+)\]")
FIELDS = {
    "author", "title", "journal", "booktitle", "volume", "number", "pages", "year",
    "doi", "eprint", "url", "urlalias", "howpublished",
}  # fmt: skip


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def parse(text: str) -> dict[str, Entry]:
    """Parse the subset of BibTeX the bibliography uses: braced values, no macros."""
    entries: dict[str, Entry] = {}
    pos = 0
    while (start := text.find("@", pos)) != -1:
        line_start = text.rfind("\n", 0, start) + 1
        if text[line_start:start].strip().startswith("%"):
            pos = start + 1
            continue
        head = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[start:])
        if not head:
            raise ValueError(f"references.bib: malformed entry near offset {start}")
        kind, key = head.group(1).lower(), head.group(2)
        i = start + head.end()
        fields: Entry = {}
        while True:
            m = re.compile(r"\s*(\w+)\s*=\s*\{").match(text, i)
            if not m:
                break
            depth, j = 1, m.end()
            while depth:
                if j >= len(text):
                    raise ValueError(f"references.bib: unbalanced braces in {key}")
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            name = m.group(1).lower()
            if name not in FIELDS:
                raise ValueError(f"references.bib: unknown field {name} in {key}")
            value = " ".join(text[m.end() : j - 1].split())
            fields[name] = value.replace("{", "").replace("}", "").replace("\\_", "_")
            separator = re.compile(r"\s*,?").match(text, j)
            assert separator is not None  # the pattern matches the empty string
            i = separator.end()
        end = re.compile(r"\s*\}").match(text, i)
        if not end:
            raise ValueError(f"references.bib: unterminated entry {key}")
        if key in entries:
            raise ValueError(f"references.bib: duplicate key {key}")
        for required in ("author", "title", "year"):
            if required not in fields:
                raise ValueError(f"references.bib: {key} lacks {required}")
        entries[key] = fields | {"type": kind, "key": key}
        pos = end.end()
    return entries


def authors(entry: Entry) -> list[tuple[str, str]]:
    """Authors as (given, family) pairs; a corporate author has an empty given name."""
    people = []
    for name in entry["author"].split(" and "):
        family, _, given = name.partition(",")
        people.append((given.strip(), family.strip()))
    return people


def collate(text: str) -> str:
    """A sort key that files Đ under D and ignores other diacritics."""
    text = text.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()


def join(names: list[str]) -> str:
    if len(names) < 3:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def normalize(url: str) -> str:
    """The identifier a URL names, in the form the bibliography indexes it by."""
    url = url.strip()
    if m := re.match(r"https?://arxiv\.org/(?:abs|html|pdf)/([^/#?]+?)(?:v\d+)?(?:\.pdf)?$", url):
        return "arxiv:" + m.group(1)
    if m := re.match(r"https?://(?:dx\.)?doi\.org/(.+)$", url):
        return "doi:" + m.group(1).lower()
    if m := re.match(r"https?://zenodo\.org/records?/(\d+)", url):
        return "doi:10.5281/zenodo." + m.group(1)
    return "url:" + url


class Bibliography:
    def __init__(self, path: str | Path = BIB) -> None:
        self.entries = parse(Path(path).read_text())
        ordered = sorted(
            self.entries.values(),
            key=lambda e: (collate(authors(e)[0][1]), int(e["year"]), collate(e["title"])),
        )
        self.number = {e["key"]: i for i, e in enumerate(ordered, 1)}
        self.index: dict[str, str] = {}
        for e in ordered:
            ids = []
            if "doi" in e:
                ids.append("doi:" + e["doi"].lower())
            if "eprint" in e:
                ids.append("arxiv:" + e["eprint"])
            ids.extend(normalize(u) for u in (e.get("url", "") + " " + e.get("urlalias", "")).split())
            for ident in ids:
                if self.index.setdefault(ident, e["key"]) != e["key"]:
                    raise ValueError(f"references.bib: {ident} names two entries")

    def resolve(self, url: str) -> str | None:
        """The key of the entry a source URL points at, or None for a non-literature URL."""
        return self.index.get(normalize(url))

    def link(self, url: str) -> str:
        """The URL to link a source by. A stored DOI that is only an alias of an
        entry is a wrong DOI kept in an append-only record: link the entry's own."""
        key = self.resolve(url)
        doi = self.entries[key].get("doi") if key else None
        if doi and normalize(url).startswith("doi:") and normalize(url) != "doi:" + doi.lower():
            return "https://doi.org/" + doi
        return url

    def keys_in(self, text: str) -> list[str]:
        keys = MARKER.findall(text)
        for key in keys:
            if key not in self.entries:
                raise ValueError(f"citation [@{key}] has no entry in references.bib")
        return keys

    def mark(self, key: str, prefix: str = "") -> str:
        """The bracketed number that cites an entry, linked to the reference list."""
        return f'<a class="cite" href="{prefix}references.html#ref-{esc(key)}">[{self.number[key]}]</a>'

    def cite(self, text: str, prefix: str = "") -> str:
        """Escape prose and replace each [@key] marker by its linked number."""
        self.keys_in(text)
        return MARKER.sub(lambda m: self.mark(m.group(1), prefix), esc(text))

    def short(self, key: str) -> str:
        """Family names and year, as in 'Holzmann, Kharaghani and Tayfeh-Rezaie (2008)'."""
        e = self.entries[key]
        return f"{join([family for _, family in authors(e)])} ({e['year']})"

    def full(self, key: str) -> str:
        """One complete reference as HTML."""
        e = self.entries[key]
        names = join([f"{given} {family}".strip() for given, family in authors(e)])
        parts = [f"{esc(names)}.", f"{esc(e['title'])}."]
        venue = e.get("journal") or e.get("booktitle")
        if venue:
            where = f"<em>{esc(venue)}</em>"
            if "volume" in e:
                where += f" {esc(e['volume'])}"
                if "number" in e:
                    where += f"({esc(e['number'])})"
            if "pages" in e:
                where += f", {esc(e['pages'].replace('--', '–'))}"
            parts.append(f"{where}, {esc(e['year'])}.")
        else:
            parts.append(f"{esc(e.get('howpublished', 'Unpublished'))}, {esc(e['year'])}.")
        links = []
        if "doi" in e:
            links.append(f'<a href="https://doi.org/{esc(e["doi"])}" rel="noopener">doi:{esc(e["doi"])}</a>')
        if "eprint" in e:
            links.append(
                f'<a href="https://arxiv.org/abs/{esc(e["eprint"])}" rel="noopener">arXiv:{esc(e["eprint"])}</a>'
            )
        if "url" in e:
            links.append(f'<a href="{esc(e["url"])}" rel="noopener">source</a>')
        return " ".join([*parts, " · ".join(links)]).strip()

    def listing(self, keys: Iterable[str] | None = None, prefix: str | None = None) -> str:
        """A numbered reference list: every entry, or the given keys in board order."""
        chosen = sorted(set(self.entries if keys is None else keys), key=self.number.__getitem__)
        if not chosen:
            return ""
        items = []
        for key in chosen:
            anchor = f' id="ref-{esc(key)}"' if prefix is None else ""
            more = "" if prefix is None else f' <a class="cite" href="{prefix}references.html#ref-{esc(key)}">↗</a>'
            items.append(f'<li value="{self.number[key]}"{anchor}>{self.full(key)}{more}</li>')
        return f'<ol class="refs">{"".join(items)}</ol>'
