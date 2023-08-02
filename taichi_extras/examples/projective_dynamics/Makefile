DATA   := $(CURDIR)/data
FRAMES := $(CURDIR)/frames
GEN    := $(CURDIR)/gen
OUTPUT := $(CURDIR)/output

TARGET  := cube
TI_ARCH := cuda

TARGET_FILES += $(DATA)/$(TARGET)/undeformed.smesh
TARGET_FILES += $(DATA)/$(TARGET)/undeformed.1.node
TARGET_FILES += $(OUTPUT)/$(TARGET)/result.stl

OPTIONS += --frame-interval=1
OPTIONS += --max-frames=$(shell echo $$((3 * 30)))
# OPTIONS += --show-window

ENV    := env TI_ARCH=$(TI_ARCH)
PYTHON := $(ENV) python

all: $(TARGET_FILES)

clean:
	@ $(RM) --recursive --verbose $(CURDIR)/.pytest_cache
	@ $(RM) --recursive --verbose $(DATA)
	@ $(RM) --recursive --verbose $(FRAMES)
	@ $(RM) --recursive --verbose $(OUTPUT)
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

view: $(DATA)/$(TARGET)/undeformed.stl
	meshlab $<

#####################
# Auxiliary Targets #
#####################

$(DATA)/$(TARGET) $(OUTPUT)/$(TARGET):
	@ mkdir --parents --verbose $@

$(DATA)/$(TARGET)/undeformed.1.node: $(DATA)/$(TARGET)/undeformed.smesh
	tetgen -zpqV $<

$(DATA)/$(TARGET)/undeformed.smesh: $(GEN)/$(TARGET).py | $(DATA)/$(TARGET)
	$(PYTHON) $< $@

CONVERT := $(PYTHON) $(dir $(CURDIR))/tetgen/convert.py
$(DATA)/$(TARGET)/undeformed.stl: $(DATA)/$(TARGET)/undeformed.1.node
	$(CONVERT) $< $@

$(OUTPUT)/$(TARGET)/result.stl: $(DATA)/$(TARGET)/undeformed.1.node $(CURDIR)/main.py | $(OUTPUT)/$(TARGET)
	$(PYTHON) $(CURDIR)/main.py $(OPTIONS) --output=$(@D) $<
