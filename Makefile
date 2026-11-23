.PHONY: install run test lint format

install:
	uv sync --locked --no-editable

run:
	uv run --no-sync uvicorn speech_analyzer.main:app --reload --port 8000

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check src tests
	uv run --no-sync ruff format --check src tests

format:
	uv run --no-sync ruff format src tests
	uv run --no-sync ruff check --fix src tests
