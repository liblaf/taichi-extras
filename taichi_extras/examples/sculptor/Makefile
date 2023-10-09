TARGET := chin

DATA   := $(CURDIR)/data/$(TARGET)
AFTER  := $(DATA)/after
BEFORE := $(DATA)/before
RAW    := $(DATA)/raw
OUTPUT := $(DATA)/output
VIDEO  := $(DATA)/video

all: output

CONFIG            := $(DATA)/config.yaml
RAW_COMPONENTS    :=
BEFORE_COMPONENTS :=
BEFORE_ALL        := $(BEFORE)/all.smesh
BEFORE_TET        != echo $(BEFORE)/all.1.{edge,ele,face,node}
AFTER_COMPONENTS  :=
OUTPUT_TET        != echo $(OUTPUT)/all.1.{edge,ele,face,node}
OUTPUT_COMPONENTS :=

include $(CURDIR)/make/$(TARGET).mk

clean:
	@ $(RM) --recursive --verbose $(AFTER)
	@ $(RM) --recursive --verbose $(BEFORE)
	@ $(RM) --recursive --verbose $(RAW)
	@ $(RM) --recursive --verbose $(OUTPUT)
	@ $(RM) --recursive --verbose $(VIDEO)
	@ $(RM) --verbose imgui.ini

pretty: black prettier

OUTPUTS += $(RAW_COMPONENTS)
OUTPUTS += $(BEFORE_COMPONENTS) $(BEFORE_ALL) $(BEFORE_TET)
OUTPUTS += $(AFTER_COMPONENTS)
OUTPUTS += $(OUTPUT_TET) $(OUTPUT_COMPONENTS)
output: $(OUTPUTS)

#####################
# Auxiliary Targets #
#####################

EMPTY:

black:
	isort --profile=black $(CURDIR)
	black $(CURDIR)

prettier:
	prettier --write $(CURDIR)

CMD          := $(CURDIR)/cmd
CMD_FIX      := python $(CMD)/fix.py --binary --join-comp --remove-smallest-components --verbose
CMD_MERGE    := python $(CMD)/merge.py
CMD_TETGEN   := tetgen -p -Y -q -A -O -z -V
CMD_SIMULATE := python $(CMD)/simulate.py --max-frames=90 --no-show-window --video-dir=$(VIDEO)

$(BEFORE)/%.ply: $(RAW)/%.ply $(CMD)/fix.py
	@ mkdir --parents --verbose $(@D)
	$(CMD_FIX) $< $@

$(BEFORE_ALL): $(CONFIG) $(BEFORE_COMPONENTS) $(AFTER_COMPONENTS) $(CMD)/merge.py
	@ mkdir --parents --verbose $(@D)
	$(CMD_MERGE) --prefix=$(DATA) $< $@

$(BEFORE)/%.1.edge $(BEFORE)/%.1.ele $(BEFORE)/%.1.face $(BEFORE)/%.1.node: $(BEFORE)/%.smesh
	$(CMD_TETGEN) $<

$(OUTPUT)/%.1.edge: $(BEFORE)/%.1.edge
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $< $@

$(OUTPUT)/%.1.ele: $(BEFORE)/%.1.ele
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $< $@

$(OUTPUT)/%.1.face: $(BEFORE)/%.1.face
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $< $@

$(OUTPUT)/%.1.node: $(BEFORE)/%.1.node $(CONFIG) $(CMD)/simulate.py
	@ mkdir --parents --verbose $(VIDEO)
	$(CMD_SIMULATE) --config=$(CONFIG) --output=$@ $<
