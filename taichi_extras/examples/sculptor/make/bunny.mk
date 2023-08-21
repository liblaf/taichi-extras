NUM_INNER := 2
INNER_END != echo $$(($(NUM_INNER) - 1))

GEN := $(CURDIR)/gen/$(TARGET)

AFTER_INNER != echo $(DATA)/after/inner/{00..$(INNER_END)}.ply

$(DATA)/raw/%.ply: $(GEN)/raw/%.py
	@ mkdir --parents --verbose $(@D)
	python $< $@

$(DATA)/after/inner/%.ply: $(DATA)/before/inner/%.ply $(GEN)/after/inner/%.py
	@ mkdir --parents --verbose $(@D)
	python $(GEN)/after/inner/$*.py $< $@
