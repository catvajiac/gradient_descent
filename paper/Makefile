FILE=paper

all:
	@pdflatex $(FILE).tex
	@bibtex $(FILE).aux
	@pdflatex $(FILE).tex
	@pdflatex $(FILE).tex

clean:
	@echo "cleaning up..."
	@rm $(FILE).aux $(FILE).bbl $(FILE).blg $(FILE).log $(FILE).out $(FILE).pdf $(FILE).toc
