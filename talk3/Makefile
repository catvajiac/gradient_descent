NAME=presentation

final: $(NAME).tex
	@latex -shell-escape $(NAME).tex
	@dvips $(NAME).dvi
	@ps2pdf $(NAME).ps
	@echo "Done!"


clean:
	@echo "Cleaning up..."
	@rm $(NAME).aux $(NAME).dvi $(NAME).log $(NAME).nav $(NAME).out
	@rm $(NAME).ps $(NAME).snm $(NAME).toc $(NAME).pdf
