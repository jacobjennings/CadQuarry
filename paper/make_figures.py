#!/usr/bin/env python3
"""
Assemble paper figures from the committed demo-1k renders.

Outputs into paper/figures/:
  - teaser.png      : variety grid sampled across all families (iso_fr view)
  - families.png    : one labelled representative per family
  - eightview.png   : the 8 standard viewpoints of a single part
"""
from __future__ import annotations

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


def make_teaser(seed: int = 7):
    rng = random.Random(seed)
    # Proportional-ish sample across families; iso_fr reads well for most.
    picks: list[Path] = []
    per = {f: max(2, 4) for f in FAMILIES}
    for fam in FAMILIES:
        ds = part_dirs(fam)
        rng.shuffle(ds)
        picks.extend(ds[: per[fam]])
    rng.shuffle(picks)
    picks = picks[:48]
    imgs = []
    for d in picks:
        im = load(d, "iso_fr") or load(d, "iso")
        if im:
            imgs.append(flatten(im))
    out = grid(imgs, cols=8, cell=190, pad=6)
    out.save(OUT / "teaser.png")
    print("teaser:", out.size, len(imgs), "parts")


# Hand-picked representatives that show each family clearly.
REPRESENTATIVES = {
    "plate": None, "bracket": None, "revolved": None, "block": None,
    "compound": None, "flanged": None, "ribbed": None, "enclosure": None,
    "profiled": None, "gear": None, "threaded": None,
}


def make_families(seed: int = 3):
    rng = random.Random(seed)
    cell, pad, label_h = 230, 10, 30
    cols = 6
    rows = (len(FAMILIES) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rows * (cell + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = font(20)
    for i, fam in enumerate(FAMILIES):
        ds = part_dirs(fam)
        chosen = REPRESENTATIVES.get(fam)
        d = next((x for x in ds if x.name == chosen), None) if chosen else None
        if d is None:
            rng.shuffle(ds)
            d = ds[0]
        im = load(d, "iso_fr") or load(d, "iso")
        im = flatten(im).resize((cell, cell), Image.LANCZOS)
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        canvas.paste(im, (x, y))
        tb = draw.textbbox((0, 0), fam, font=fnt)
        tw = tb[2] - tb[0]
        draw.text((x + (cell - tw) // 2, y + cell + 4), fam, fill=(20, 20, 20), font=fnt)
    canvas.save(OUT / "families.png")
    print("families:", canvas.size)


def make_eightview(part: str | None = None):
    order = ["front", "top", "right", "iso", "iso_fr", "iso_fl", "iso_br", "iso_bl"]
    # A compound part shows depth/AO from many angles well.
    if part is None:
        cands = part_dirs("compound") + part_dirs("gear")
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


if __name__ == "__main__":
    make_teaser()
    make_families()
    make_eightview()
