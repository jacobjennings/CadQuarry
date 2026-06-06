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

## Generative-AI disclosure

Per the paper's disclosure section: the dataset is produced by deterministic
procedural code (no model samples geometry); the generator and this manuscript
were developed with AI coding assistance under the author's direction and review.
