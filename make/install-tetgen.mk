BIN        := $(HOME)/.local/bin
TETGEN_SRC := $(CURDIR)/tetgen1.6.0

tetgen-clean:
	@ $(RM) --recursive --verbose $(TETGEN_SRC)
	@ $(RM) --verbose $(CURDIR)/tetgen1.6.0.tar.gz

tetgen-install: $(BIN)/tetgen

#####################
# Auxiliary Targets #
#####################

$(BIN)/tetgen: $(TETGEN_SRC)/tetgen
	@ install -D --mode="u=rwx,go=rx" --no-target-directory --verbose $< $@

$(CURDIR)/tetgen1.6.0.tar.gz:
	wget --output-document=$@ https://wias-berlin.de/software/tetgen/1.5/src/tetgen1.6.0.tar.gz

$(TETGEN_SRC)/tetgen: $(CURDIR)/tetgen1.6.0.tar.gz
	tar --extract --file=$< --gzip
	$(MAKE) --directory=$(TETGEN_SRC)
