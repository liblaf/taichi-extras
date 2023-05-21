DATA    := $(CURDIR)/data
FRAMES  := $(CURDIR)/frames
OPTIONS := # --max-frames 72 --off-screen

all: run

clean:
	$(RM) --recursive $(DATA)
	$(RM) --recursive $(FRAMES)
	$(RM) *.gif
	$(RM) *.ini
	$(RM) *.mp4

pretty:
	isort --profile black $(CURDIR)
	black $(CURDIR)

run: $(DATA)/mesh.1.node $(DATA)/mesh.1.fixed.txt $(CURDIR)/main.py
	python $(CURDIR)/main.py $(OPTIONS) $<

$(DATA):
	mkdir --parents $@

$(DATA)/mesh.1.fixed.txt: $(DATA)/mesh.1.node $(CURDIR)/fixed_indices.py
	python $(CURDIR)/fixed_indices.py --output $@ --radius 0.5 $<

$(DATA)/mesh.1.node: $(DATA)/mesh.smesh $(TETGEN)
	tetgen -zpqV $<

$(DATA)/mesh.smesh: $(CURDIR)/generate.py | $(DATA)
	python $< $@
