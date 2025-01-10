.PHONY: ruff lint test

lint:
	uv run python -m mypy .

ruff:
	ruff check . --fix

test:
	uv run pytest -v
