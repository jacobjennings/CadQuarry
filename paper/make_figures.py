#!/usr/bin/env python3
"""
Assemble paper figures from the committed demo-1k renders.

Outputs into paper/figures/:
  - teaser.png      : variety grid sampled across all families (iso_fr view)
  - families.png    : one labelled representative per family
  - eightview.png   : the 8 standard viewpoints of a single part
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RENDERS = ROOT / "sample" / "demo-1k" / "renders"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BG = (255, 255, 255, 255)
FAMILIES = [
    "plate", "bracket", "revolved", "block", "compound",
    "flanged", "ribbed", "enclosure", "profiled", "gear", "threaded",
]
# Keyed by the figure's number in the compiled paper (Fig 2 = pipeline diagram
# and Fig 5 = distribution chart are not built here, hence the gaps).
FIGURES = {1: "teaser", 3: "families", 4: "eightview"}


def part_dirs(family: str) -> list[Path]:
    return sorted(d for d in RENDERS.iterdir() if d.name.startswith(family + "_") and d.is_dir())


def load(view_dir: Path, view: str) -> Image.Image | None:
    p = view_dir / f"{view}.png"
    if not p.exists():
        return None
    return Image.open(p).convert("RGBA")


def flatten(img: Image.Image, bg=BG) -> Image.Image:
    canvas = Image.new("RGBA", img.size, bg)
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


def font(size: int):
    for name in (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def grid(images: list[Image.Image], cols: int, cell: int, pad: int, bg=(255, 255, 255)) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + pad)
        thumb = im.resize((cell, cell), Image.LANCZOS)
        canvas.paste(thumb, (x, y))
    return canvas


def make_teaser():
    # Proportional-ish sample across families; iso_fr reads well for most.
    # 24 parts in an 8-wide grid = 3 rows, keeping the teaser compact enough
    # that the abstract still lands on page 1.
    picks: list[Path] = []
    per = {f: max(2, 4) for f in FAMILIES}
    for fam in FAMILIES:
        ds = part_dirs(fam)
        random.shuffle(ds)
        picks.extend(ds[: per[fam]])
    random.shuffle(picks)
    picks = picks[:24]
    imgs = []
    for d in picks:
        im = load(d, "iso_fr") or load(d, "iso")
        if im:
            imgs.append(flatten(im))
    out = grid(imgs, cols=8, cell=190, pad=6)
    out.save(OUT / "teaser.png")
    print("teaser:", out.size, len(imgs), "parts")


# Hand-picked representatives that show each family clearly (None = pick at random).
REPRESENTATIVES = {
    "plate": None, "bracket": None, "revolved": None, "block": None,
    "compound": None, "flanged": None, "ribbed": None, "enclosure": None,
    "profiled": None, "gear": None, "threaded": None,
}

# families.png montage geometry (shared by the full build and in-place re-rolls).
FAM_CELL, FAM_PAD, FAM_LABEL_H, FAM_COLS = 230, 10, 30, 6


def _family_part(fam: str) -> Path:
    """The hand-picked representative for a family, or a fresh random one."""
    chosen = REPRESENTATIVES.get(fam)
    if chosen:
        for d in part_dirs(fam):
            if d.name == chosen:
                return d
    return random.choice(part_dirs(fam))


def _draw_family_cell(canvas: Image.Image, draw: ImageDraw.ImageDraw, fnt, i: int, fam: str, d: Path):
    im = flatten(load(d, "iso_fr") or load(d, "iso")).resize((FAM_CELL, FAM_CELL), Image.LANCZOS)
    r, c = divmod(i, FAM_COLS)
    x = FAM_PAD + c * (FAM_CELL + FAM_PAD)
    y = FAM_PAD + r * (FAM_CELL + FAM_LABEL_H + FAM_PAD)
    canvas.paste(im, (x, y))
    tb = draw.textbbox((0, 0), fam, font=fnt)
    draw.text((x + (FAM_CELL - (tb[2] - tb[0])) // 2, y + FAM_CELL + 4), fam, fill=(20, 20, 20), font=fnt)


def make_families():
    """Rebuild the whole grid with a fresh random representative per family."""
    rows = (len(FAMILIES) + FAM_COLS - 1) // FAM_COLS
    W = FAM_COLS * FAM_CELL + (FAM_COLS + 1) * FAM_PAD
    H = rows * (FAM_CELL + FAM_LABEL_H) + (rows + 1) * FAM_PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = font(20)
    for i, fam in enumerate(FAMILIES):
        _draw_family_cell(canvas, draw, fnt, i, fam, _family_part(fam))
    canvas.save(OUT / "families.png")
    print("families:", canvas.size)


def reroll_families(families: set[str]):
    """Re-pick only the named family cells, pasting onto the existing grid so
    every other cell (and the rest of the figure) is left exactly as-is."""
    path = OUT / "families.png"
    if not path.exists():
        make_families()  # nothing to paste onto yet
        return
    canvas = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    fnt = font(20)
    for fam in families:
        _draw_family_cell(canvas, draw, fnt, FAMILIES.index(fam), fam, _family_part(fam))
    canvas.save(path)
    print("families: re-rolled", ", ".join(sorted(families)))


def make_eightview(part: str | None = None):
    order = ["front", "top", "right", "iso", "iso_fr", "iso_fl", "iso_br", "iso_bl"]
    # A compound part shows depth/AO from many angles well.
    if part is None:
        cands = part_dirs("compound") + part_dirs("gear")
        random.shuffle(cands)
        part = next(d.name for d in cands if all((d / f"{v}.png").exists() for v in order))
    d = RENDERS / part
    cell, pad, label_h = 230, 8, 26
    cols = 4
    rows = 2
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = font(18)
    for i, v in enumerate(order):
        im = flatten(load(d, v)).resize((cell, cell), Image.LANCZOS)
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        canvas.paste(im, (x, y))
        tb = draw.textbbox((0, 0), v, font=fnt)
        tw = tb[2] - tb[0]
        draw.text((x + (cell - tw) // 2, y + cell + 3), v, fill=(20, 20, 20), font=fnt)
    canvas.save(OUT / "eightview.png")
    print("eightview:", canvas.size, "part:", part)


def main():
    ap = argparse.ArgumentParser(
        description="Shuffle the paper figure montages. With no arguments, "
        "re-rolls all three; narrow it with --figure and/or --family."
    )
    ap.add_argument(
        "--family",
        action="append",
        metavar="NAME",
        help="Re-roll only these family cells in families.png (repeatable, or "
        "comma-separated). Every other cell, and the rest of the figure, is left "
        f"exactly as it is on disk. Choices: {', '.join(FAMILIES)}.",
    )
    ap.add_argument(
        "--figure",
        action="append",
        metavar="N",
        help="Limit work to these figures (repeatable, or comma-separated): "
        + ", ".join(f"{n}={name}" for n, name in FIGURES.items())
        + ". Default: all (or just families when --family is used).",
    )
    args = ap.parse_args()

    families: set[str] = set()
    for item in args.family or []:
        for name in item.split(","):
            name = name.strip()
            if not name:
                continue
            if name not in FAMILIES:
                ap.error(f"unknown family {name!r}; choose from {', '.join(FAMILIES)}")
            families.add(name)

    figs: set[int] = set()
    for item in args.figure or []:
        for tok in item.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if not tok.isdigit() or int(tok) not in FIGURES:
                choices = ", ".join(f"{n}={name}" for n, name in FIGURES.items())
                ap.error(f"unknown figure {tok!r}; choose from {choices}")
            figs.add(int(tok))
    if not figs:
        # Default: just families when re-rolling specific cells, else all three.
        figs = {3} if families else set(FIGURES)
    if families:
        figs.add(3)  # --family always implies the families figure

    if 1 in figs:
        make_teaser()
    if 3 in figs:
        # Named cells re-roll in place; otherwise rebuild the whole grid.
        reroll_families(families) if families else make_families()
    if 4 in figs:
        make_eightview()


if __name__ == "__main__":
    main()
