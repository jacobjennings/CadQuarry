#!/usr/bin/env python3
"""
Shuffle the paper figures until you like the shown set.

Each run re-rolls the figures you ask for (random every time) and, unless
--no-pdf, recompiles main.pdf so you can eyeball the result. Just keep running
it until you're happy, then commit the figures.

    python3 shuffle.py                            # re-roll all three figures
    python3 shuffle.py --no-pdf                   # figures only, skip the LaTeX build
    python3 shuffle.py --figure 1                 # only the teaser (paper Fig 1)
    python3 shuffle.py --figure 3,4               # families (Fig 3) + eightview (Fig 4)
    python3 shuffle.py --family revolved,profiled # only these cells; the rest stay put
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATEXMK = ["latexmk", "-lualatex", "-shell-escape", "-interaction=nonstopmode", "main.tex"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pdf", action="store_true", help="Regenerate figures only; don't recompile the PDF.")
    ap.add_argument(
        "--family",
        action="append",
        metavar="NAME",
        help="Re-roll only these family cells in families.png (repeatable, or "
        "comma-separated); every other cell stays exactly as it is.",
    )
    ap.add_argument(
        "--figure",
        action="append",
        metavar="N",
        help="Limit work to these figures by their number in the paper (repeatable, "
        "or comma-separated): 1=teaser, 3=families, 4=eightview. Default: all.",
    )
    args = ap.parse_args()

    cmd = [sys.executable, "make_figures.py"]
    for fam in args.family or []:
        cmd += ["--family", fam]
    for fig in args.figure or []:
        cmd += ["--figure", fig]

    print("==> shuffling figures")
    subprocess.run(cmd, cwd=HERE, check=True)

    if not args.no_pdf:
        print("==> rebuilding main.pdf")
        subprocess.run(LATEXMK, cwd=HERE, check=True)

    print("\nDone. Not happy? Run it again. Happy? Commit the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
