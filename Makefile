.PHONY: install run test lint format

install:
	uv sync

run:
	uv run uvicorn speech_analyzer.main:app --reload --port 8000

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests