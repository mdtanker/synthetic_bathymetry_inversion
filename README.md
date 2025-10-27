# Synthetic sub-ice shelf bathymetry inversions


This repository contains the code and Jupyter notebooks for running all the inversions and creating the figures for the manuscript: ["Gravity Inversion for Sub-Ice Shelf Bathymetry: Strengths, Limitations, and Insights from Synthetic Modeling"](https://doi.org/10.5194/egusphere-2025-2380).

While most of the code is from external packages, such as [Invert4Geom](https://github.com/mdtanker/invert4geom) for the inversion, there is some code supplied in `src/`.

The Jupyter Notebooks for all the synthetic inversions, as well as creating the figures are found in `docs/`.

Data outputs from these notebooks are saved in `results/`, and figure outputs are saved in `paper/figures/`.

These notebooks can be explored at the following website generated from this repository: https://mdtanker.github.io/synthetic_bathymetry_inversion/


<!-- SPHINX-START-badges -->

## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/synthetic_bathymetry_inversion

## Dependencies

These instructions assume you have Python (>=3.11) installed. If you don't we recommend installing it with [miniforge](https://github.com/conda-forge/miniforge) for a simple and minimal setup.

You can setup your environment in two ways: 1) with `pixi`, or 2) with `conda` (or `mamba`).

### `pixi`
Install `pixi` with the instructions at https://pixi.sh/latest/installation/

Then install the environment defined in `pyproject.toml` with:
```
pixi install
```

## `conda` or `mamba`

Install the required dependencies with either `conda` or `mamba`:

    cd synthetic_bathymetry_inversion

    mamba env create --file environment.yml --name synthetic_bathymetry_inversion

Activate the newly created environment:

    mamba activate synthetic_bathymetry_inversion

## Developer instructions

Export `pixi` environment to a conda `environment.yml` file:
```
pixi workspace export conda-environment environment.yml
```

Create the docs locally:
```
pixi run api
pixi run docs
```

Style-checks and formatting the code / notebooks
```
pixi run style
```

<!-- SPHINX-END-badges -->
