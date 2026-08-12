.PHONY: setup test lint typecheck profile clean

setup:
	uv sync --all-groups
	uv run pre-commit install

test:
	uv run pytest --cov=src/rri --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

profile:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_profiling.ipynb

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
