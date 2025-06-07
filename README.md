# Synthetic sub-ice shelf bathymetry inversions

This repository contains the code and Jupyter notebooks for running all the inversions and create the figures for the manuscript: "Gravity Inversion for Sub-Ice Shelf Bathymetry: Strengths, Limitations, and Insights from Synthetic Modeling".

While most of the code is from external packages, such as [Invert4Geom](https://github.com/mdtanker/invert4geom) for the inversion, there is some code supplied in `src/`.

The Jupyter Notebooks for all the synthetic inversions, as well as creating the figures are found in `notebooks/`.

Data outputs from these notebooks are saved in `results/`, and figure outputs are saved in `paper/figures/`.

## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/synthetic_bathymetry_inversion

## Dependencies

Install the required dependencies with either `conda` or `mamba`:

    cd synthetic_bathymetry_inversion

    mamba env create --file environment.yml --name synthetic_bathymetry_inversion

Activate the newly created environment:

    mamba activate synthetic_bathymetry_inversion

Install the local project

    pip install -e .


## Developer instructions

This assumes you have `Make` installed. If you don't, you can just manually input the respective commands from the `Makefile`.

Style-check your code:

    make style

Test your code

    make test

To update dependencies, first add or change dependencies listed in `environment.yml`, then with your conda environment activated run:

    make update

When writing code; use logging to inform users of info, errors, and warnings. In each module `.py` file, import the project-wide logger instance with `from synthetic_bathymetry_inversion import logger` and then for example: `logger.info("log this message")`
