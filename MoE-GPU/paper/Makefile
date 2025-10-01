# Makefile for compiling the research paper

# Main LaTeX file
MAIN = research_paper

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean view arxiv help

# Default target
all: $(MAIN).pdf

# Compile the paper (run twice for references)
$(MAIN).pdf: $(MAIN).tex
	@echo "Compiling LaTeX document..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Running second pass for references..."
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "Done! Output: $(MAIN).pdf"

# Compile with BibTeX (if needed in future)
bibtex: $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).bbl $(MAIN).blg $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f $(MAIN).synctex.gz
	@echo "Clean complete!"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(MAIN).pdf
	@echo "Distclean complete!"

# View the PDF (platform-specific)
view: $(MAIN).pdf
	@echo "Opening PDF..."
ifeq ($(OS),Windows_NT)
	start $(MAIN).pdf
else
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v open > /dev/null; then \
		open $(MAIN).pdf; \
	else \
		echo "Could not find PDF viewer"; \
	fi
endif

# Prepare arXiv submission package
arxiv: $(MAIN).pdf
	@echo "Preparing arXiv submission package..."
	@mkdir -p arxiv_submission
	@cp $(MAIN).tex arxiv_submission/
	@if [ -f $(MAIN).bbl ]; then cp $(MAIN).bbl arxiv_submission/; fi
	@if [ -d figures ]; then cp -r figures arxiv_submission/; fi
	@cd arxiv_submission && tar -czf ../arxiv_submission.tar.gz *
	@echo "arXiv package created: arxiv_submission.tar.gz"
	@echo "Upload this file to arXiv.org"

# Show help
help:
	@echo "Makefile for LaTeX paper compilation"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Compile the paper (default)"
	@echo "  make all      - Same as 'make'"
	@echo "  make bibtex   - Compile with BibTeX"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean- Remove all generated files including PDF"
	@echo "  make view     - Open the PDF"
	@echo "  make arxiv    - Create arXiv submission package"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Output: $(MAIN).pdf"
