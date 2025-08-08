PROJECT=synthetic_bathymetry_inversion

####
####
# install commands
####
####

create:
	mamba env create --file environment.yml --name $(PROJECT)

install:
	pip install --no-deps -e .

update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

remove:
	mamba env remove --name $(PROJECT)

export:
	mamba env export --file pinned_environment.yml --name $(PROJECT) --no-builds
