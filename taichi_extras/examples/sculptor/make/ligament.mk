GEN := $(CURDIR)/gen/$(TARGET)

RAW_COMPONENTS += $(DATA)/raw/face.ply
RAW_COMPONENTS += $(DATA)/raw/ligament.ply
RAW_COMPONENTS += $(DATA)/raw/skull_left.ply
RAW_COMPONENTS += $(DATA)/raw/skull_right.ply

BEFORE_COMPONENTS += $(DATA)/before/face.ply
BEFORE_COMPONENTS += $(DATA)/before/ligament.ply
BEFORE_COMPONENTS += $(DATA)/before/skull_left.ply
BEFORE_COMPONENTS += $(DATA)/before/skull_right.ply

AFTER_COMPONENTS += $(DATA)/after/skull_left.ply
AFTER_COMPONENTS += $(DATA)/after/skull_right.ply

RESULT_COMPONENTS += $(DATA)/result/face.ply
RESULT_COMPONENTS += $(DATA)/result/ligament.ply
RESULT_COMPONENTS += $(DATA)/result/skull_left.ply
RESULT_COMPONENTS += $(DATA)/result/skull_right.ply

#####################
# Auxiliary Targets #
#####################

$(DATA)/raw/%.ply: $(GEN)/raw/%.py
	@ mkdir --parents --verbose $(@D)
	python $< $@

$(DATA)/after/%.ply: $(DATA)/before/%.ply
	@ install -D --mode="u=rw,go=r" --no-target-directory --verbose $< $@
