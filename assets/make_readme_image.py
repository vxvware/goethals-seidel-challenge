#!/usr/bin/env python3
"""Draw assets/skew-hadamard-356.png, the picture at the top of README.md.

Top: the Goethals-Seidel array of entry n89-eec2e43c042090883c04, a matrix of
order 356, drawn as site/picture.js draws it. The skew row comes first, so the
diagonal blocks are green; light is +1 and dark is -1. The script refuses to
write the file unless H H^T = 356 I and H + H^T = 2 I.

Bottom: the board grid turned on its side. One column per odd n from 1 to 131,
one row per track in the order of verify.common.TRACKS, coloured as the legend
of the site colours a cell. A violet mark is a cell the site marks as new.

Run from the repository root: uv run python assets/make_readme_image.py
"""

from pathlib import Path
import struct
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))
from verify.board import load, summary  # noqa: E402
from verify.common import TRACKS  # noqa: E402
import build as site  # noqa: E402

ENTRY = "n89-eec2e43c042090883c04"
OUT = ROOT / "assets/skew-hadamard-356.png"

# Colours of site/style.css and site/picture.js.
PAPER, PANEL, LINE = (3, 7, 5), (6, 17, 11), (20, 53, 36)
TONES = {"k": ((61, 235, 120), (5, 34, 17)), "s": ((95, 205, 245), (5, 28, 40))}
PLAIN = ((176, 184, 179), (24, 28, 26))
STATUS = {
    "known": (61, 255, 124),
    "published": (95, 215, 255),
    "empty": (44, 58, 49),
    "unresolved": (255, 176, 0),
}
VIOLET = (199, 146, 255)

SCALE = 2  # pixels per matrix entry
MARGIN, PAD, GAP = 12, 6, 14
PITCH, CELL = 11, 9  # grid strip


def gs_array(rows):
    """The array of site/picture.js: circulant X has entry (i, j) equal to x[(j - i) mod n]."""
    n = len(rows[0])
    index = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    a, b, c, d = (np.array(row, dtype=np.int64)[index] for row in rows)

    def xr(x):
        return x[:, ::-1]

    def rx(x):
        return x[::-1, :]

    return np.block(
        [
            [a, xr(b), xr(c), xr(d)],
            [-xr(b), a, -rx(d), rx(c)],
            [-xr(c), rx(d), a, -rx(b)],
            [-xr(d), -rx(c), rx(b), a],
        ]
    )


def cell_state(c):
    """The legend class of a grid cell, as the board grid labels it."""
    if c["quadruples"]:
        return "known"
    if c["artifact_status"] == "published-not-imported":
        return "published"
    return "empty" if c["status"] == "empty" else "unresolved"


def png(pixels):
    """An 8-bit RGB PNG from an array of shape (height, width, 3)."""
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main():
    documents, results, baseline = load(reproduce=False)
    stats = summary(documents, results, baseline)
    rows, pattern = site.display_rows(results[ENTRY])
    n = len(rows[0])
    order = 4 * n
    h = gs_array(rows)
    identity = np.eye(order, dtype=np.int64)
    if pattern[0] != "k":
        raise SystemExit("the first row is not skew")
    if not np.array_equal(h @ h.T, order * identity):
        raise SystemExit("H H^T is not 4n I")
    if not np.array_equal(h + h.T, 2 * identity):
        raise SystemExit("H + H^T is not 2 I")

    side = SCALE * order + 2 * PAD
    orders = list(range(1, 132, 2))
    width = side + 2 * MARGIN
    height = MARGIN + side + GAP + PITCH * len(TRACKS) - (PITCH - CELL) + MARGIN
    if PITCH * len(orders) - (PITCH - CELL) != side:
        raise SystemExit("the grid strip and the matrix panel differ in width")
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = PAPER

    # Matrix panel with a one-pixel border.
    image[MARGIN : MARGIN + side, MARGIN : MARGIN + side] = LINE
    image[MARGIN + 1 : MARGIN + side - 1, MARGIN + 1 : MARGIN + side - 1] = PANEL
    block = np.arange(order) // n
    colours = np.empty((order, order, 3), dtype=np.uint8)
    for y in range(order):
        for x in range(order):
            light, dark = TONES.get(pattern[block[y] ^ block[x]], PLAIN)
            colours[y, x] = light if h[y, x] > 0 else dark
    top = MARGIN + PAD
    image[top : top + SCALE * order, top : top + SCALE * order] = colours.repeat(
        SCALE, axis=0
    ).repeat(SCALE, axis=1)

    # Grid strip.
    by_cell = {(c["n"], c["stratum"]): c for c in stats["cells"]}
    marked = site.new_cells(stats, results)
    strip = MARGIN + side + GAP
    for i, m in enumerate(orders):
        for j, track in enumerate(TRACKS):
            y, x = strip + PITCH * j, MARGIN + PITCH * i
            image[y : y + CELL, x : x + CELL] = STATUS[cell_state(by_cell[m, track])]
            if (track, m) in marked:
                image[y + 2 : y + CELL - 2, x + 2 : x + CELL - 2] = PAPER
                image[y + 3 : y + CELL - 3, x + 3 : x + CELL - 3] = VIOLET

    OUT.write_bytes(png(image))
    print(f"{OUT.relative_to(ROOT)}: {width}x{height}, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
