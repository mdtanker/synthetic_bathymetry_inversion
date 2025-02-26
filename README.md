# Research repo template
Template for creating repositories for research projects using Python

Steps:
1) choose a project name (no spaces, capitals or underscores) and replace `project_name` with it in the following locations:
    - `Makefile` top line
    - `environment.yml` top line
    - `src/project_name` folder
    - `project_name.py` in `src/project_name/`
    - line 6 of `pyproject.toml`
    - `project.urls` section of `pyproject.toml`
    - import line of `tests/test_module1.py`
2) add your pip/conda dependencies to `environmental.yml`
3) update `pyproject.yml`
    - update your name and email
    - add a description
    - add keywords
    - replace `mdtanker` with your github username in the `project.urls` section
4) add your code to and rename `src/project_name/module1.py`
5) add tests for `module1.py` to `tests/test_module1.py` and rename to your module
6) remove all the above instructions and add a description of your project here, and tweak the below user instructions as needed


## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/projectname

## Dependencies

These instructions assume you have `Make` installed. If you don't you can just open up the `Makefile` file and copy and paste the commands into your terminal. This also assumes you have Python installed.

Install the required dependencies with either `conda` or `mamba`:

    cd research_repo_template

    make create

Activate the newly created environment:

    conda activate projectname

Install the local project

    make install



## Developer instructions

Style-check your code:

    make style

Test your code

    make test

To update dependencies, first add or change dependencies listed in `environment.yml`, then with your conda environment activated run:

    make update

When writing code; use logging to inform users of info, errors, and warnings. In each module `.py` file, import the project-wide logger instance with `from projectname import logger` and then for example: `logger.info("log this message")`
