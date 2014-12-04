all: clean
	pdflatex nips2015.tex
	bibtex nips2015
	pdflatex nips2015.tex
	pdflatex nips2015.tex


clean:
	rm -f *.log *.aux *.brf *.blg *.bbl *.brf *.out *.dvi *.nav *.snm *.toc
