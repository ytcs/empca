# Reproduce every figure, table and number of the paper, then build the PDF.
#   make data     -- simulate and run all analyses (about 30 min on 4 cores)
#   make figures  -- re-plot from the saved results only
#   make pdf      -- compile main.pdf
PY ?= python
S = scripts

all: data figures pdf

data:
	cd $(S) && $(PY) run_benchmark.py run ablation
	cd $(S) && $(PY) bench_woodbury.py run
	cd $(S) && $(PY) timing_phase.py
	cd $(S) && $(PY) fig_lowenergy.py

figures:
	cd $(S) && $(PY) run_benchmark.py figures table
	cd $(S) && $(PY) bench_woodbury.py figure
	cd $(S) && $(PY) fig_pulses.py
	cd $(S) && $(PY) fig_manifold.py
	cd $(S) && $(PY) fig_covariance.py
	cd $(S) && $(PY) fig_lowenergy.py
	cd $(S) && $(PY) make_numbers.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C main.tex

.PHONY: all data figures pdf clean
