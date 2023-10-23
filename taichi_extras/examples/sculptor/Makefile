TARGET := XXX

# TARGET_FILES   += data/$(TARGET)/pre/01-face.ply
# TARGET_FILES   += data/$(TARGET)/pre/01-skull.ply
# TARGET_FILES   += data/$(TARGET)/pre/02-face.ply
# TARGET_FILES   += data/$(TARGET)/pre/02-skull.ply
# TARGET_FILES   += data/$(TARGET)/pre/03-face.ply
# TARGET_FILES   += data/$(TARGET)/pre/03-skull.ply
TEMPLATE_FILES += $(shell echo data/template/{00..03}-{face,skull}.ply)

all: $(TARGET_FILES) $(TEMPLATE_FILES)

check: data/$(TARGET)/pre/03-face.ply data/$(TARGET)/pre/03-skull.ply
	- $(TETGEN_CHECK) data/$(TARGET)/pre/03-face.ply
	- $(TETGEN_CHECK) data/$(TARGET)/pre/03-skull.ply

clean: clean-target clean-template

clean-target:
	@ $(RM) --verbose $(TARGET_FILES)
	@ $(RM) --verbose data/$(TARGET)/pre/*.edge
	@ $(RM) --verbose data/$(TARGET)/pre/*.ele
	@ $(RM) --verbose data/$(TARGET)/pre/*.face
	@ $(RM) --verbose data/$(TARGET)/pre/*.node

clean-template:
	@ $(RM) --verbose $(TEMPLATE_FILES)

#####################
# Auxiliary Targets #
#####################

CT_TO_MESH   := $(PYTHON) cmd/CT_to_mesh.py
MESH_FIX     := MeshFix
PYTHON       := python
REGISTER     := $(PYTHON) cmd/register.py
SIMPLIFY     := cp # te mesh simplify --face-count = 100000
SMOOTH       := $(PYTHON) cmd/smooth.py
TETGEN       := tetgen -p -Y -q -A -O -z -V
TETGEN_CHECK := $(TETGEN) -d -N -E -F

data/$(TARGET)/pre/01-face.ply: data/$(TARGET)/pre/00-CT.nrrd cmd/CT_to_mesh.py
	$(CT_TO_MESH) --threshold=0.0 $< $@

data/$(TARGET)/pre/01-skull.ply: data/$(TARGET)/pre/00-CT.nrrd cmd/CT_to_mesh.py
	$(CT_TO_MESH) --threshold=300.0 $< $@

data/$(TARGET)/pre/02-%.ply: data/$(TARGET)/pre/01-%.ply
	$(SIMPLIFY) $< $@

data/$(TARGET)/pre/03-%.ply: data/$(TARGET)/pre/02-%.ply data/$(TARGET)/pre/02-%.landmark.txt data/template/02-%.ply data/template/02-%.landmark.txt cmd/register.py
	$(REGISTER) --output=$@ data/template/02-$*.ply $<

data/template/00-%.ply:
	@ mkdir --parents --verbose $(@D)
	wget --output-document=$@ https://github.com/liblaf/sculptor/raw/main/model/template/$*.ply

data/template/01-%.ply: data/template/00-%.ply cmd/template/%.py
	$(PYTHON) cmd/template/$*.py $< $@

data/template/02-%.ply: data/template/01-%.ply
	$(MESH_FIX) $< $@

data/template/03-%.ply: data/template/02-%.ply cmd/smooth.py
	$(SMOOTH) $< $@
