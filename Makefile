all:

clean:
	git clean -d --force -X

lock: $(CURDIR)/pyproject.toml $(CURDIR)/poetry.lock $(CURDIR)/requirements.txt

pretty: black prettier

test:
	pytest --cov --cov-report=xml

ALWAYS:

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier: $(CURDIR)/.gitignore ALWAYS
	prettier --write --ignore-path $< $(CURDIR)

$(CURDIR)/poetry.lock: $(CURDIR)/pyproject.toml ALWAYS
	poetry lock

$(CURDIR)/requirements.txt: $(CURDIR)/poetry.lock
	poetry export --output=$@ --without-hashes --without-urls
