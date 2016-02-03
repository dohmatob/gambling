all: clean
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex


experimental: clean
	pdflatex MOR-template.tex
	bibtex MOR-template
	pdflatex MOR-template.tex
	pdflatex MOR-template.tex

clean:
	rm -f *.log *.aux *.brf *.blg *.bbl *.brf *.out *.dvi *.nav *.snm *.toc
