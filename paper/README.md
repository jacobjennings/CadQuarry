# CadQuarry paper

LaTeX source for the arXiv paper *"CadQuarry: A Permissively-Licensed,
Procedurally-Generated Dataset of Parametric CAD Programs."*

## Layout

| File | Purpose |
|---|---|
| `main.tex` | The paper (LuaLaTeX). |
| `references.bib` | Bibliography (biblatex/biber). |
| `make_figures.py` | Builds the figure montages from the committed `sample/demo-1k/` renders. |
| `figures/` | Generated figures (`teaser`, `families`, `eightview`). |
| `Makefile` | `make` builds figures + PDF. |

## Build

Requires a full TeX Live (LuaLaTeX, biber), Python + Pillow (figures), and
Pygments (the `minted` code listings). Then:

```bash
make          # regenerate figures, then compile main.pdf
# or directly:
python3 make_figures.py
latexmk -lualatex -shell-escape main.tex
```

`-shell-escape` is required by `minted`. The fonts used (Libertinus Serif/Sans,
Source Code Pro) ship with TeX Live.

## Figures

All renders are CadQuarry's own headless GPU output, drawn from the 1,000-part
sample committed at `../sample/demo-1k/`. Regenerate them (e.g. after rebuilding
the sample) with `make figures`.

To re-roll which parts are shown until you like the set, use `shuffle.py`: each
run randomly re-rolls the figures you ask for and rebuilds the PDF. Keep running
it until you're happy, then commit the figures.

```bash
python3 shuffle.py                            # re-roll all three figures + rebuild PDF
python3 shuffle.py --no-pdf                   # figures only, skip the LaTeX build
python3 shuffle.py --figure 1                 # only the teaser (paper Fig 1)
python3 shuffle.py --figure 3,4               # families (Fig 3) + eightview (Fig 4)
python3 shuffle.py --family revolved,profiled # only these cells; the rest stay put
```

`--family` (repeatable or comma-separated) re-rolls only the named family cells
in `families.png`, pasting fresh picks onto the existing image so every other
cell — and `teaser`/`eightview` — is left exactly as it is on disk. Handy when
the grid is mostly good and one or two examples need swapping; just re-run until
those cells look right.

`--figure` (repeatable or comma-separated) limits the work to specific figures,
numbered as they appear in the compiled paper: `1`=teaser, `3`=families,
`4`=eightview (Fig 2 is the pipeline diagram and Fig 5 the distribution chart,
neither built here). Runs are random and not reproducible by design — there are
no seeds.

## Generative-AI disclosure

Per the paper's disclosure section: the dataset is produced by deterministic
procedural code (no model samples geometry); the generator and this manuscript
were developed with AI coding assistance under the author's direction and review.
