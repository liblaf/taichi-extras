TARGET := sculptor

DATA := $(CURDIR)/data/$(TARGET)

all: result

include $(CURDIR)/make/$(TARGET).mk

INNER_END != echo $$(($(NUM_INNER) - 1))

RAW_OUTER    := $(DATA)/raw/outer.ply
RAW_INNER    != echo $(DATA)/raw/inner/{00..$(INNER_END)}.ply
BEFORE_OUTER := $(DATA)/before/outer.ply
BEFORE_INNER != echo $(DATA)/before/inner/{00..$(INNER_END)}.ply
TOPOLOGY     := $(DATA)/topology.txt
BEFORE_ALL   := $(DATA)/before/tet/all.smesh
BEFORE_TET   != echo $(DATA)/before/tet/all.1.{edge,ele,face,node}
# AFTER_INNER  :=
RESULT_TET   != echo $(DATA)/result/tet/all.1.{edge,ele,face,node}
RESULT_OUTER := $(DATA)/result/outer.ply
RESULT_INNER != echo $(DATA)/result/inner/{00..$(INNER_END)}.ply

clean:
	@ $(RM) --recursive --verbose $(CURDIR)/data
	@ $(RM) --verbose imgui.ini

pretty: black prettier

FILES += $(RAW_OUTER) $(RAW_INNER)
FILES += $(BEFORE_OUTER) $(BEFORE_INNER) $(TOPOLOGY) $(BEFORE_ALL) $(BEFORE_TET)
FILES += $(AFTER_INNER)
FILES += $(RESULT_TET) $(RESULT_OUTER) $(RESULT_INNER)
result: $(FILES)

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
CMD_FIX      := python $(CMD)/fix.py --binary --no-join-comp --remove-smallest-components --verbose
CMD_TOPOLOGY := python $(CMD)/topology.py
CMD_MERGE    := python $(CMD)/merge.py
CMD_TETGEN   := tetgen -p -Y -q -O -z -V
CMD_SIMULATE := python $(CMD)/simulate.py --max-frames=90 --no-show-window # --video-dir=$(DATA)/result
CMD_SPLIT    := python $(CMD)/split.py

$(DATA)/before/%.ply: $(DATA)/raw/%.ply $(CMD)/fix.py
	@ mkdir --parents --verbose $(@D)
	$(CMD_FIX) $< $@

$(TOPOLOGY): $(BEFORE_OUTER) $(BEFORE_INNER) $(CMD)/topology.py
	@ mkdir --parents --verbose $(@D)
	$(CMD_TOPOLOGY) --output=$@ $(BEFORE_OUTER) $(BEFORE_INNER)

$(BEFORE_ALL): $(BEFORE_OUTER) $(BEFORE_INNER) $(CMD)/merge.py
	@ mkdir --parents --verbose $(@D)
	$(CMD_MERGE) --output=$@ $(BEFORE_OUTER) $(BEFORE_INNER)

$(subst .1.,%,$(BEFORE_TET)): $(BEFORE_ALL)
	$(CMD_TETGEN) $<

$(DATA)/result/tet/all.1.node: $(TOPOLOGY) $(BEFORE_TET) $(AFTER_INNER) $(CMD)/simulate.py
	@ mkdir --parents --verbose $(@D)
	$(CMD_SIMULATE) --topology=$(TOPOLOGY) $(addprefix --inner=, $(AFTER_INNER)) --output=$@ $(DATA)/before/tet/all.1.node

$(DATA)/result/tet/all.1.%: $(DATA)/before/tet/all.1.%
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $< $@

$(subst .,%,$(RESULT_OUTER) $(RESULT_INNER)): $(DATA)/result/tet/all.1.node $(BEFORE_OUTER) $(BEFORE_INNER) $(TOPOLOGY) $(CMD)/split.py
	@ mkdir --parents --verbose $(DATA)/result/inner
	$(CMD_SPLIT) --node=$< --topology=$(TOPOLOGY) --output=$(DATA)/result $(BEFORE_OUTER) $(BEFORE_INNER)
