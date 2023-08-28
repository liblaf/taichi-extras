NUM_INNER := 1
INNER_END != echo $$(($(NUM_INNER) - 1))

GEN := $(CURDIR)/gen/$(TARGET)

RAW_OUTER   := $(DATA)/raw/outer.ply
RAW_INNER   != echo $(DATA)/raw/inner/{00..$(INNER_END)}.ply
AFTER_INNER := $(DATA)/after/inner/00.ply

$(subst .,%,$(RAW_OUTER) $(RAW_INNER)): $(GEN)/raw.py
	@ mkdir --parents --verbose $(DATA)/raw/inner
	python $< $(DATA)/raw

$(DATA)/after/%.ply: $(DATA)/before/%.ply $(GEN)/after/%.py
	@ mkdir --parents --verbose $(@D)
	python $(GEN)/after/$*.py $< $@
