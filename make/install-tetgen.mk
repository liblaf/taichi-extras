BIN        := $(HOME)/.local/bin
TETGEN_DIR := $(CURDIR)/tetgen1.6.0

INSTALL_OPTIONS := -D --mode="u=rwx,go=rx" --no-target-directory --verbose

all: install

clean:
	$(RM) --recursive $(TETGEN_DIR)
	$(RM) $(CURDIR)/tetgen1.6.0.tar.gz

install: $(BIN)/tetgen

$(BIN)/tetgen: $(TETGEN_DIR)/tetgen
	@ install $(INSTALL_OPTIONS) $< $@

$(CURDIR)/tetgen1.6.0.tar.gz:
	wget --output-document=$@ https://wias-berlin.de/software/tetgen/1.5/src/tetgen1.6.0.tar.gz

$(TETGEN_DIR)/tetgen: $(CURDIR)/tetgen1.6.0.tar.gz
	tar --extract --file=$< --gzip
	$(MAKE) --directory=$(CURDIR)/tetgen1.6.0
