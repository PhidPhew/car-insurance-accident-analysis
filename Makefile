.PHONY: setup test lint typecheck profile warehouse export load-mysql er-diagram clean

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

warehouse:
	uv run python -m rri.warehouse.build

export:
	uv run python -m rri.warehouse.export

load-mysql:
	uv run python scripts/load_to_mysql.py

er-diagram:
	uv run python scripts/generate_er_diagram.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
