DATA := $(CURDIR)/data

all: view

clean:
	$(RM) --recursive $(DATA)
	$(RM) *.ini

pretty:
	isort --profile black $(CURDIR)
	black $(CURDIR)

view: $(DATA)/mesh.1.node $(CURDIR)/view.py
	python $(CURDIR)/view.py $<

$(DATA):
	mkdir --parents $@

$(DATA)/mesh.1.node: $(DATA)/mesh.smesh $(TETGEN)
	tetgen -zpqV $<

$(DATA)/mesh.smesh: $(CURDIR)/generate.py | $(DATA)
	python $< $@
