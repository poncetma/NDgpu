PYTHON ?= python
PYTHONPATH := ../..

.PHONY: all figures clean

all: adaptive-bdf-lra-benchmark.pdf

figures:
	MPLCONFIGDIR=/tmp/ndgpu-matplotlib PYTHONPATH=$(PYTHONPATH) $(PYTHON) make_figures.py

build/report.pdf: report.tex figures
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build report.tex

adaptive-bdf-lra-benchmark.pdf: build/report.pdf
	cp build/report.pdf $@

clean:
	rm -rf build adaptive-bdf-lra-benchmark.pdf figures
