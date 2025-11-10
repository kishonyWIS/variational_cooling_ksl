# Variational Cooling of Chiral Spin Liquids

This directory contains the LaTeX summary and code for our work on variational cooling of the Kitaev spin liquid (KSL) model using shallow-depth quantum circuits.

## Directory Structure

```
Variational-Cooling-KSL/
├── code/              # Python source files
│   ├── variational_circuit_KSL_numba.py      # Main working file (Numba-optimized)
│   ├── variational_circuit_KSL_faster.py     # Alternative implementation
│   ├── variational_circuit_KSL.py            # Original implementation
│   └── plot_progressive_circuit_expansion_results.py          # Visualization script
├── data/              # Data files
│   ├── progressive_circuit_expansion_results*.csv              # Progressive circuit expansion results
│   ├── variational_circuit_results.csv      # Circuit results
│   ├── optimized_strength_durations*.npz     # Optimized parameters
│   ├── optimized_parameters/                 # Parameter directories
│   └── optimized_parameters_monotonic_large_steps_at_large_p/
├── figures/           # Generated figures (PDFs, etc.)
└── main.tex           # Main LaTeX document

## Contents

- `code/`: Python implementation files for variational circuit optimization
- `data/`: All data files (CSV results, NPZ parameters, parameter directories)
- `figures/`: Directory for figure files (placeholders prepared)
- `main.tex`: Main LaTeX document summarizing the project
- `References.bib`: Bibliography file with references

## Running the Code

The code files in `code/` are set up to:
- Import dependencies from the project root (`translational_invariant_KSL.py`, `time_dependence_functions.py`)
- Save/load data files from the `data/` directory
- Save figures to the `figures/` directory

Run scripts from the project root directory or ensure the project root is in your Python path.

## Figures to Generate

The document includes placeholders for the following figures:

1. `energy_vs_p_by_res.pdf` - Energy density vs. circuit depth $p$ for different training resolutions (already exists in parent directory)
2. `figures/energy_vs_cycles.pdf` - Energy density vs. number of cooling cycles on both training and test grids
3. `figures/energy_heatmap_train.pdf` - Energy difference heatmap on the training grid
4. `figures/energy_heatmap_test.pdf` - Energy difference heatmap on the test grid
5. `figures/parameters_vs_layer.pdf` - Optimized variational parameters vs. layer number
6. `figures/chern_vs_p.pdf` - Chern number vs. circuit depth $p$ on test grid

## Compilation

To compile the LaTeX document:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk`:

```bash
latexmk -pdf main.tex
```

## Project Description

This project combines:
- **Fermionic cooling approach** from CoolingFermions2D (preparing chiral states via fermionic cooling)
- **Variational optimization** from Variational-Cooling-Latex (shallow-depth circuits)

The goal is to compress the large-depth Trotterized cooling protocol (~100 steps) into a shallow variational circuit with only a few layers while maintaining high fidelity ground state preparation of the chiral KSL phase.

