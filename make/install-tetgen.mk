BIN        := $(HOME)/.local/bin
TETGEN_DIR := $(CURDIR)/tetgen1.6.0

tetgen-clean:
	$(RM) --recursive $(TETGEN_DIR)
	$(RM) $(CURDIR)/tetgen1.6.0.tar.gz

tetgen-install: $(BIN)/tetgen

#####################
# Auxiliary Targets #
#####################

$(BIN)/tetgen: $(TETGEN_DIR)/tetgen
	@ install -D --mode="u=rwx,go=rx" --no-target-directory --verbose $< $@

$(CURDIR)/tetgen1.6.0.tar.gz:
	wget --output-document=$@ https://wias-berlin.de/software/tetgen/1.5/src/tetgen1.6.0.tar.gz

$(TETGEN_DIR)/tetgen: $(CURDIR)/tetgen1.6.0.tar.gz
	tar --extract --file=$< --gzip
	$(MAKE) --directory=$(CURDIR)/tetgen1.6.0
