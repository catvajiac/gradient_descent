FILE=paper

all:
	@pdflatex -shell-escape $(FILE).tex
	@bibtex $(FILE).aux
	@pdflatex -shell-escape $(FILE).tex
	@pdflatex -shell-escape $(FILE).tex

clean:
	@echo "cleaning up..."
	@rm $(FILE).aux $(FILE).bbl $(FILE).blg $(FILE).log $(FILE).out $(FILE).pdf $(FILE).toc
