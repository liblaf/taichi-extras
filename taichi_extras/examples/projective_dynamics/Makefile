TARGET  := shell
TI_ARCH := cuda

DATA   := $(CURDIR)/data/$(TARGET)
GEN    := $(CURDIR)/gen
OUTPUT := $(CURDIR)/output/$(TARGET)

FIXED != if [[ -f $(CURDIR)/fixed/$(TARGET).py ]]; then echo true; else echo false; fi

OPTIONS += --no-record
OPTIONS += --max-frames=$(shell echo $$((3 * 30)))
# OPTIONS += --show-window
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

pretty:
	isort --profile=black $(CURDIR)
	black $(CURDIR)

setup:
	$(MAKE) --directory=$(realpath $(CURDIR)/../../..) setup

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
	tetgen -zpqV $<

$(DATA)/undeformed.smesh: $(GEN)/$(TARGET).py | $(DATA)
	$(PYTHON) $< $@

CONVERT := $(PYTHON) $(dir $(CURDIR))/tetgen/convert.py
$(DATA)/undeformed.ply: $(DATA)/undeformed.1.node
	$(CONVERT) $< $@

ifeq ($(FIXED), true)
  EXTRA_DATA_FILE_LIST += $(DATA)/fixed.node
endif
$(OUTPUT)/tet.ply $(OUTPUT)/tri.ply: $(DATA)/undeformed.1.node $(EXTRA_DATA_FILE_LIST) $(CURDIR)/main.py | $(OUTPUT)
	$(PYTHON) $(CURDIR)/main.py $(OPTIONS) --output=$(@D) $<
