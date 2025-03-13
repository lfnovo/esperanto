.PHONY: ruff lint test

lint:
	uv run python -m mypy .

ruff:
	ruff check . --fix

test:
	uv run pytest -v


build-docs:
	repomix . --include "**/*.py" --compress --style xml -o ai_docs/all_docs.txt
