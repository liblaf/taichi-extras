SCRIPTS := $(CURDIR)/scripts

all:

include make/*.mk

clean: tetgen-clean
	@ $(RM) --recursive --verbose $(CURDIR)/.pytest_cache
	@ $(RM) --verbose $(CURDIR)/.coverage
	@ $(RM) --verbose $(CURDIR)/coverage.xml
	@ find $(CURDIR) -name "__pycache__" -exec $(RM) --recursive --verbose {} +
	@ find $(CURDIR) -name "*.pyc"       -exec $(RM) --verbose {} +

pretty: black prettier

setup: $(SCRIPTS)/setup-cuda.sh tetgen-install
	bash $<
	pip install open3d
	poetry install

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
