.PHONY: ruff lint test

lint:
	poetry run python -m mypy .

ruff:
	ruff check . --fix

test:
	poetry run pytest -v