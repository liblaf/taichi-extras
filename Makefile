all:

pretty:
	isort --profile black $(CURDIR)
	black $(CURDIR)

deps: $(CURDIR)/pyproject.toml $(CURDIR)/poetry.lock $(CURDIR)/requirements.txt

ALWAYS:

$(CURDIR)/poetry.lock: $(CURDIR)/pyproject.toml ALWAYS
	poetry lock

$(CURDIR)/requirements.txt: $(CURDIR)/poetry.lock
	poetry export --output=$@ --without-hashes --without-urls
