PROJECT=synthetic_bathymetry_inversion

####
####
# install commands
####
####

install:
	pip install -e .

remove:
	mamba env remove --name $(PROJECT)

create:
	mamba env create --file environment.yml --name $(PROJECT)

update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

export:
	mamba env export --file pinned_environment.yml --name $(PROJECT) --no-builds

####
####
# style commands
####
####

check:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: check pylint

####
####
# test commands
####
####

test:
	pytest
