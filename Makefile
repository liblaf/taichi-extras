SCRIPTS := $(CURDIR)/scripts

all:

include make/*.mk

clean: tetgen-clean
	@ $(RM) --recursive --verbose $(CURDIR)/.pytest_cache
	@ $(RM) --verbose $(CURDIR)/.coverage
	@ $(RM) --verbose $(CURDIR)/coverage.xml

pretty: black prettier

setup: $(SCRIPTS)/setup-cuda.sh tetgen-install
	poetry install
	bash $<

test:
	pytest --cov --cov-report=xml

#####################
# Auxiliary Targets #
#####################

black:
	isort --profile=black $(CURDIR)
	black $(CURDIR)

prettier: $(CURDIR)/.gitignore
	prettier --write --ignore-path=$< $(CURDIR)
