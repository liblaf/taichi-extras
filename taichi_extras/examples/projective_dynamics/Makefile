DATA   := $(CURDIR)/data
FRAMES := $(CURDIR)/frames
GEN    := $(CURDIR)/gen

TARGET  := cube
TI_ARCH := cuda

OPTIONS += --frame-interval=1
OPTIONS += --max-frames=$(shell echo $$((1 * 30)))
# OPTIONS += --show-window

ENV    := env TI_ARCH=$(TI_ARCH)
PYTHON := $(ENV) python

all: $(DATA)/$(TARGET).2.stl

clean:
	@ $(RM) --recursive --verbose $(CURDIR)/.pytest_cache
	@ $(RM) --recursive --verbose $(DATA)
	@ $(RM) --recursive --verbose $(FRAMES)
	@ $(RM) --verbose *.gif
	@ $(RM) --verbose *.ini
	@ $(RM) --verbose *.mp4
	@ $(RM) --verbose $(CURDIR)/.coverage
	@ $(RM) --verbose $(CURDIR)/coverage.xml
	@ find $(CURDIR) -name "__pycache__" -exec $(RM) --recursive --verbose {} +
	@ find $(CURDIR) -name "*.pyc" -exec $(RM) --verbose {} +

pretty:
	isort --profile=black $(CURDIR)
	black $(CURDIR)

setup:
	$(MAKE) --directory=$(realpath $(CURDIR)/../../..) setup

test:
	$(ENV) pytest --cov --cov-report=term --cov-report=term-missing

view: $(DATA)/$(TARGET).2.stl
	meshlab $<

#####################
# Auxiliary Targets #
#####################

$(DATA):
	@ mkdir --parents --verbose $@

$(DATA)/%.1.node: $(DATA)/%.0.smesh
	tetgen -zpqV $<

$(DATA)/%.0.smesh: $(GEN)/%.py | $(DATA)
	$(PYTHON) $< $@

CONVERT := $(PYTHON) $(dir $(CURDIR))/tetgen/convert.py
$(DATA)/%.1.stl: $(DATA)/%.1.node
	$(CONVERT) $< $@

$(DATA)/%.2.stl: $(DATA)/%.1.node $(CURDIR)/main.py
	$(PYTHON) $(CURDIR)/main.py $(OPTIONS) --output=$@ $<
