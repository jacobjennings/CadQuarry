# Build the CadQuarry paper.
#   make         -> figures + PDF
#   make figures -> regenerate montages from sample/demo-1k renders
#   make clean   -> remove LaTeX aux files (keep main.pdf)
#   make distclean -> also remove main.pdf and generated figures

LATEXMK := latexmk -lualatex -shell-escape -interaction=nonstopmode

main.pdf: main.tex references.bib figures/teaser.png figures/families.png figures/eightview.png
	$(LATEXMK) main.tex

.PHONY: figures
figures:
	python3 make_figures.py

figures/teaser.png figures/families.png figures/eightview.png: make_figures.py
	python3 make_figures.py

.PHONY: clean distclean
clean:
	$(LATEXMK) -c main.tex
	rm -rf _minted* *.bbl *.run.xml

distclean: clean
	rm -f main.pdf
	rm -f figures/teaser.png figures/families.png figures/eightview.png
