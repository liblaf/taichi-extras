DATA    := $(CURDIR)/data
FRAMES  := $(CURDIR)/frames
ROOT    != git rev-parse --show-toplevel

OPTIONS := --max-frames 72 --off-screen

all: run-cube

clean:
	git clean -d --force -X

pretty: black prettier

test: test-cube # test-spherical-shell

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier: $(ROOT)/.gitignore
	prettier --write --ignore-path $< $(CURDIR)

run-%: $(DATA)/%.1.node $(DATA)/%.1.fixed.txt $(CURDIR)/main.py
	python $(CURDIR)/main.py $(OPTIONS) $<

test-%: $(DATA)/%.1.node $(DATA)/%.1.fixed.txt
	pytest --capture=no --mesh-filepath $< $(CURDIR)/utils/tests

$(DATA):
	mkdir --parents $@

$(DATA)/%.1.fixed.txt: $(DATA)/%.1.node $(CURDIR)/fixed_indices.py
	python $(CURDIR)/fixed_indices.py --output $@ --radius 0.5 $<

.SECONDARY:
$(DATA)/%.1.node: $(DATA)/%.smesh
	tetgen -zpqV $<

$(DATA)/%.smesh: $(CURDIR)/generator/%.py | $(DATA)
	python $< $@
