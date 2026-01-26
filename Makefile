# Makefile for LLM Evaluation Survey Paper

.PHONY: pdf clean

# Build PDF using latexmk
pdf:
	latexmk -pdf main.tex

# Clean all build artifacts
clean:
	latexmk -C
	rm -f *.bbl *.run.xml *.bcf
