"""Catalogue files: the quadruples of one board cell in one file, each checked in full.

A cell with thousands of classes would need thousands of entry files, and the
board is bounded at 4096. A census entry can instead name one catalogue under
evidence/catalogues/: a gzip-compressed text file with one quadruple per line,
its four rows separated by single spaces, + for 1 and - for -1. The verifier
reads every line on every load and requires that

- the four rows have the length of the cell and satisfy the autocorrelation
  identity exactly, by the same check as a quadruple entry;
- the quadruple belongs to the stratum of the census;
- the line is the canonical form of its class under equiv-v1; and
- the lines are strictly increasing, so no two are equivalent.

The count of a catalogue is the number of its lines. Nothing in the file is
trusted beyond these checks, and the file is bound to the entry by sha256.
"""

import zlib
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

from verify.common import JSONObject, PathLike, Rows, artifact
from verify.equivalence import canonical, canonical_half_shifts
from verify.gs_verify import memberships, validate_rows

MAX_COMPRESSED = 4 * 1024 * 1024
MAX_TEXT = 32 * 1024 * 1024
MAX_ROWS = 100_000
SIGN = {"+": 1, "-": -1}


def text(path: Path) -> str:
    """The decompressed catalogue, with both sizes bounded before anything is parsed."""
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError(f"oversized catalogue: {path.name}")
    unpacker = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    data = unpacker.decompress(path.read_bytes(), MAX_TEXT + 1)
    if len(data) > MAX_TEXT or unpacker.unconsumed_tail or not unpacker.eof or unpacker.unused_data:
        raise ValueError(f"catalogue is oversized or not one complete gzip stream: {path.name}")
    return data.decode("ascii")


def parse_line(line: str, n: int) -> Rows:
    words = line.split(" ")
    if len(words) != 4 or any(len(word) != n for word in words) or set(line) - {"+", "-", " "}:
        raise ValueError("a catalogue line is four sign strings of the cell's length separated by single spaces")
    return tuple(tuple(SIGN[c] for c in word) for word in words)


def format_line(rows: Iterable[Iterable[int]]) -> str:
    return " ".join("".join("+" if v > 0 else "-" for v in row) for row in rows)


def lines(path: Path) -> list[str]:
    body = text(path)
    if body and not body.endswith("\n"):
        raise ValueError(f"catalogue does not end with a newline: {path.name}")
    found = body.split("\n")[:-1]
    if len(found) > MAX_ROWS:
        raise ValueError(f"catalogue has more than {MAX_ROWS} lines: {path.name}")
    return found


def in_stratum(rows: Rows, stratum: str) -> bool:
    """Membership of a quadruple in a stratum; the two strata without a skew row are decided directly."""
    if stratum == "general":
        return True
    if stratum == "williamson":
        # The pattern ssss holds exactly when every row is symmetric.
        return all(row[1:] == row[:0:-1] for row in rows)
    return stratum in memberships(rows)


@lru_cache(maxsize=128)
def checked(path: str, sha256: str, n: int, stratum: str) -> int:
    """Check every line of a catalogue and return the number of lines.

    The result is cached for the process by path and digest; the caller has
    compared the file with the digest first, so a cached verdict is about
    these exact bytes.
    """
    previous: Rows | None = None
    count = 0
    for number, line in enumerate(lines(Path(path)), 1):
        try:
            rows = validate_rows(parse_line(line, n), n)
            if not in_stratum(rows, stratum):
                raise ValueError(f"the quadruple is not in the {stratum} stratum")
            if canonical(rows) != rows:
                raise ValueError("the rows are not the canonical form of their class")
            if previous is not None and rows <= previous:
                raise ValueError("the lines are not strictly increasing")
        except ValueError as error:
            raise ValueError(f"{Path(path).name} line {number}: {error}") from error
        previous = rows
        count += 1
    return count


def half_shift_images(rows: Rows) -> Iterator[Rows]:
    """The images of a quadruple of even length under the two further operations of the published relation.

    They are the cyclic shift of one row by half its length, for each row, and
    the negation of every second entry of all four rows. Each is returned in
    canonical form under equiv-v1.
    """
    h = len(rows[0]) // 2
    for i in range(4):
        yield canonical([row[h:] + row[:h] if j == i else row for j, row in enumerate(rows)])
    yield canonical([tuple(-x if j % 2 else x for j, x in enumerate(row)) for row in rows])


@lru_cache(maxsize=128)
def published_classes(path: str, sha256: str, n: int) -> tuple[int, bool]:
    """The classes of a checked catalogue under the published relation at even n, and whether it is closed.

    Returns the number of classes of the lines under canonical_half_shifts,
    and whether every image of a line under the two further operations is
    again a line, which makes the catalogue a union of whole published classes.
    """
    found = lines(Path(path))
    present = set(found)
    classes = set()
    closed = True
    for line in found:
        rows = parse_line(line, n)
        classes.add(canonical_half_shifts(rows))
        closed = closed and all(format_line(image) in present for image in half_shift_images(rows))
    return len(classes), closed


def contains(path: Path, rows: Rows) -> bool:
    """Whether a quadruple in canonical form is a line of the catalogue."""
    return format_line(rows) in lines(path)


def verify(entry: JSONObject, quadruples: dict[str, JSONObject], root: PathLike, reproduce: bool) -> JSONObject:
    """Verify a census whose classes are the lines of a catalogue file.

    Such a census is a lower bound: it states how many pairwise inequivalent
    quadruples the file holds, each verified here, and no search is replayed.
    With published_classes it also states how the lines sit in a published
    classification at even n under the coarser relation of that
    classification: their number of classes there, and with split that the
    file holds every equiv-v1 class inside them. A full verification
    recomputes both; a quick one checks every line and leaves them unchecked.
    """
    ref, n, stratum = entry["catalogue"], entry["n"], entry["stratum"]
    if entry["representatives"] or entry["claim_kind"] != "lower-bound" or entry["equivalence"] != "equiv-v1":
        raise ValueError("a catalogue census is a lower bound under equiv-v1 and lists no representatives")
    if entry.get("enumerator") or entry.get("count_only"):
        raise ValueError("a catalogue census has no enumerator and is not count-only")
    path = artifact(root, ref)
    count = checked(str(path), ref["sha256"], n, stratum)
    if count == 0 or count != entry["claimed_count"]:
        raise ValueError("catalogue is empty or its number of lines differs from the claimed count")
    result: JSONObject = {
        "id": entry["id"],
        "type": "census",
        "n": n,
        "stratum": stratum,
        "equivalence": entry["equivalence"],
        "count": count,
        "bound": "lower",
        "tier": None,
        "claimed_count": entry["claimed_count"],
        "claim_kind": entry["claim_kind"],
        "checks": ["catalogue-rows-exact-PAF", "catalogue-rows-canonical", "pairwise-inequivalence"],
        "catalogue": {"path": ref["path"], "sha256": ref["sha256"], "rows": count},
    }
    published = entry.get("published_classes")
    if published:
        if n % 2:
            raise ValueError("published_classes describes the published relation at even n")
        result["catalogue"]["published_classes"] = published["count"]
        result["catalogue"]["split"] = published["split"]
        result["catalogue"]["published_checked"] = False
        if not published["split"] and published["count"] != count:
            raise ValueError("a catalogue with one line per published class has as many lines as classes")
        if reproduce:
            classes, closed = published_classes(str(path), ref["sha256"], n)
            if classes != published["count"] or closed != published["split"]:
                raise ValueError("catalogue disagrees with its stated published classes")
            result["catalogue"]["published_checked"] = True
            result["checks"].append("published-classes-recomputed")
    if published and published["split"]:
        # A catalogue that holds whole published classes holds every known class in them; an entry of
        # the cell that is missing from it would show that the published classification is incomplete.
        for q in quadruples.values():
            if q["n"] == n and stratum in q["tracks"] and not contains(path, q["canonical_rows"]):
                raise ValueError(f"quadruple {q['id']} is missing from the catalogue of its cell")
    return result
