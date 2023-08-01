DATA := $(CURDIR)/data

all: $(DATA)/mesh.stl

clean:
	@ $(RM) --recursive --verbose $(DATA)

pretty:
	isort --profile=black $(CURDIR)
	black $(CURDIR)

view: $(DATA)/mesh.stl
	meshlab $<

#####################
# Auxiliary Targets #
#####################

$(DATA):
	@ mkdir --parents --verbose $@

$(DATA)/%.1.node: $(DATA)/%.smesh $(TETGEN)
	tetgen -zpqV $<

$(DATA)/%.stl: $(DATA)/%.1.node $(CURDIR)/convert.py
	python $(CURDIR)/convert.py $< $@

$(DATA)/mesh.smesh: $(CURDIR)/generate.py | $(DATA)
	python $< $@
