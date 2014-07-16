all: clean
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex


clean:
	rm -f *.log *.aux *.brf *.blg *.bbl *.brf *.out *.dvi *.nav *.snm *.toc
