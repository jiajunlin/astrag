# Build automation.

.PHONY: test deploy

test:
	pytest -q

deploy: test
	./scripts/deploy.sh

lint:
	ruff check .
