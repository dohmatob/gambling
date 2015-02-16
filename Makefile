all: clean
	pdflatex nips2015.tex
	bibtex nips2015
	pdflatex nips2015.tex
	pdflatex nips2015.tex


experimental: clean
	pdflatex MOR-template.tex
	bibtex MOR-template
	pdflatex MOR-template.tex
	pdflatex MOR-template.tex

clean:
	rm -f *.log *.aux *.brf *.blg *.bbl *.brf *.out *.dvi *.nav *.snm *.toc
