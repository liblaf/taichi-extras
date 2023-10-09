GEN := $(CURDIR)/gen/$(TARGET)

RAW_COMPONENTS += $(DATA)/raw/face.ply
RAW_COMPONENTS += $(DATA)/raw/mandible.ply
RAW_COMPONENTS += $(DATA)/raw/maxilla.ply
# RAW_COMPONENTS += $(DATA)/raw/skull.ply

BEFORE_COMPONENTS += $(DATA)/before/face.ply
BEFORE_COMPONENTS += $(DATA)/before/mandible.ply
BEFORE_COMPONENTS += $(DATA)/before/maxilla.ply
# BEFORE_COMPONENTS += $(DATA)/before/skull.ply

AFTER_COMPONENTS += $(DATA)/after/mandible.ply

RESULT_COMPONENTS += $(DATA)/result/face.ply
RESULT_COMPONENTS += $(DATA)/result/mandible.ply
RESULT_COMPONENTS += $(DATA)/result/maxilla.ply
# RESULT_COMPONENTS += $(DATA)/result/skull.ply

#####################
# Auxiliary Targets #
#####################

$(subst .,%,$(RAW_COMPONENTS)): $(GEN)/raw.py
	@ mkdir --parents --verbose $(@D)
	python $< $(@D)

$(DATA)/after/%.ply: $(DATA)/before/%.ply $(GEN)/after/%.py
	@ mkdir --parents --verbose $(@D)
	python $(GEN)/after/$*.py $< $@
