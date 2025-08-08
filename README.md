# Synthetic sub-ice shelf bathymetry inversions

This repository contains the code and Jupyter notebooks for running all the inversions and create the figures for the manuscript: "Gravity Inversion for Sub-Ice Shelf Bathymetry: Strengths, Limitations, and Insights from Synthetic Modeling".

While most of the code is from external packages, such as [Invert4Geom](https://github.com/mdtanker/invert4geom) for the inversion, there is some code supplied in `src/`.

The Jupyter Notebooks for all the synthetic inversions, as well as creating the figures are found in `notebooks/`.

Data outputs from these notebooks are saved in `results/`, and figure outputs are saved in `paper/figures/`.

## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/synthetic_bathymetry_inversion

## Dependencies

These instructions assume you have Python (>=3.11) installed. If you don't we recommend installing it with [miniforge](https://github.com/conda-forge/miniforge) for a simple and minimal setup.

Install the required dependencies with either `conda` or `mamba`:

    cd synthetic_bathymetry_inversion

    mamba env create --file environment.yml --name synthetic_bathymetry_inversion

If you want the specific, pinned, versions for each dependency, replace 'environment.yml' with 'pinned_environment.yml'.

Activate the newly created environment:

    mamba activate synthetic_bathymetry_inversion

Install the local project

    pip install --no-deps -e .


# How to contribute / develop?

See the file `CONTRIBUTING.md` for some detailed instructions on how to work on developing this repository.
