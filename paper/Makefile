.PHONY: all clean pdf figures

all: pdf

pdf: main.pdf

main.pdf: main.tex sections/*.tex appendix/*.tex refs.bib figures/*.pdf
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

figures:
	cd .. && python -m experiments.plotting.plot_all

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
	rm -f sections/*.aux appendix/*.aux
