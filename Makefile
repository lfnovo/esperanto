.PHONY: ruff lint test

lint:
	uv run python -m mypy .

ruff:
	ruff check . --fix

test:
	uv run pytest -v

uv_sync:
	uv pip compile --extra all --extra dev pyproject.toml -o uv.lock
	uv pip sync uv.lock
	uv pip install -e .