METHOD  ?= CG # Sparse
TARGET  ?= bunny
TI_ARCH ?= cuda

DATA   := $(CURDIR)/data/$(TARGET)
GEN    := $(CURDIR)/gen
OUTPUT := $(CURDIR)/output/$(TARGET)

FIXED != if [[ -f $(CURDIR)/fixed/$(TARGET).py ]]; then echo true; else echo false; fi

# OPTIONS += --camera 0.0 0.0 3.0
OPTIONS += --camera 0.0 0.0 0.26 # bunny
OPTIONS += --max-frames=$(shell echo $$((3 * 30)))
OPTIONS += --method=$(METHOD)
OPTIONS += --record
# OPTIONS += --show-window
# OPTIONS += --fixed-stiffness=1e8
ifeq ($(FIXED), true)
  OPTIONS += --fixed=$(DATA)/fixed.node
endif

ENV    := env TI_ARCH=$(TI_ARCH)
PYTHON := $(ENV) python

TARGET_FILES += $(DATA)/undeformed.1.node
TARGET_FILES += $(DATA)/undeformed.smesh
TARGET_FILES += $(OUTPUT)/tet.ply
ifeq ($(FIXED), true)
  TARGET_FILES += $(DATA)/fixed.node
endif

all: $(TARGET_FILES)

clean:
	@ $(RM) --recursive --verbose $(CURDIR)/.pytest_cache
	@ $(RM) --recursive --verbose $(CURDIR)/data
	@ $(RM) --recursive --verbose $(CURDIR)/output
	@ $(RM) --verbose *.ini
	@ $(RM) --verbose $(CURDIR)/.coverage
	@ $(RM) --verbose $(CURDIR)/coverage.xml
	@ find $(CURDIR) -name "__pycache__" -exec $(RM) --recursive --verbose {} +
	@ find $(CURDIR) -name "*.pyc"       -exec $(RM) --verbose {} +

diff: $(OUTPUT)/diff.ply

pretty:
	isort --profile=black $(CURDIR)
	black $(CURDIR)

setup:
	$(MAKE) --directory=$(shell git rev-parse --show-toplevel) setup

test:
	$(ENV) pytest --cov --cov-report=term --cov-report=term-missing

view-tet: $(OUTPUT)/tet.ply
	meshlab $<

view-tri: $(OUTPUT)/tri.ply
	meshlab $<

view-undeformed: $(DATA)/undeformed.ply
	meshlab $<

#####################
# Auxiliary Targets #
#####################

$(DATA) $(OUTPUT):
	@ mkdir --parents --verbose $@

$(DATA)/fixed.node: $(DATA)/undeformed.1.node $(CURDIR)/fixed/$(TARGET).py
	$(PYTHON) $(CURDIR)/fixed/$(TARGET).py $< $@

$(DATA)/undeformed.1.node: $(DATA)/undeformed.smesh
	tetgen -p -Y -q -O -z -V $<

$(DATA)/undeformed.smesh: $(GEN)/$(TARGET).py | $(DATA)
	$(PYTHON) $< $@

CONVERT := $(PYTHON) $(dir $(CURDIR))/tetgen/convert.py
$(DATA)/undeformed.ply: $(DATA)/undeformed.1.node
	$(CONVERT) $< $@

DIFF := $(PYTHON) $(CURDIR)/utils/diff.py
$(OUTPUT)/diff.ply: $(OUTPUT)/old.ply $(OUTPUT)/new.ply
	$(DIFF) --output=$@ $^

ifeq ($(FIXED), true)
  EXTRA_DATA_FILE_LIST += $(DATA)/fixed.node
endif

$(OUTPUT)/new.ply: $(DATA)/undeformed.1.node $(EXTRA_DATA_FILE_LIST) $(CURDIR)/main.py | $(OUTPUT)
	$(PYTHON) $(CURDIR)/main.py $(OPTIONS) --output=$(@D) --no-record --fixed-stiffness=1e6 $<
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $(@D)/tri.ply $@

$(OUTPUT)/old.ply: $(DATA)/undeformed.1.node $(EXTRA_DATA_FILE_LIST) $(CURDIR)/main.py | $(OUTPUT)
	$(PYTHON) $(CURDIR)/main.py $(OPTIONS) --output=$(@D) --no-record --fixed-stiffness=1e5 $<
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $(@D)/tri.ply $@

$(OUTPUT)/tet.ply $(OUTPUT)/tri.ply: $(DATA)/undeformed.1.node $(EXTRA_DATA_FILE_LIST) $(CURDIR)/main.py | $(OUTPUT)
	$(PYTHON) $(CURDIR)/main.py $(OPTIONS) --output=$(@D) $<
