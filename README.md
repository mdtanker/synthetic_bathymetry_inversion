# Synthetic sub-ice shelf bathymetry inversions

## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/synthetic_bathymetry_inversion

## Dependencies

These instructions assume you have `Make` installed. If you don't you can just open up the `Makefile` file and copy and paste the commands into your terminal. This also assumes you have Python installed.

Install the required dependencies with either `conda` or `mamba`:

    cd synthetic_bathymetry_inversion

    make create

Activate the newly created environment:

    conda activate synthetic_bathymetry_inversion

Install the local project

    make install



## Developer instructions

Style-check your code:

    make style

Test your code

    make test

To update dependencies, first add or change dependencies listed in `environment.yml`, then with your conda environment activated run:

    make update

When writing code; use logging to inform users of info, errors, and warnings. In each module `.py` file, import the project-wide logger instance with `from synthetic_bathymetry_inversion import logger` and then for example: `logger.info("log this message")`
