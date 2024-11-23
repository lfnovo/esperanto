.PHONY: ruff lint

lint:
	poetry run python -m mypy .

ruff:
	ruff check . --fix
