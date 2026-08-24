.PHONY: setup test lint typecheck profile warehouse export load-mysql er-diagram clean

setup:
	uv sync --all-groups
	uv run pre-commit install

test:
	# Split in two: PyTorch (rri.models.mlp_entity_embeddings/gru_forecast) and
	# LightGBM (rri.models.lightgbm_baseline) both bundle their own libomp
	# runtime, and loading both in one process reliably segfaults inside
	# LightGBM's Dataset construction partway through the full suite (see
	# tests/conftest.py) -- two separate pytest processes never let both
	# libraries share a process. --cov-append merges the coverage data.
	rm -f .coverage
	uv run pytest --cov=src/rri --cov-report=term-missing \
		--ignore=tests/unit/test_mlp_entity_embeddings.py --ignore=tests/unit/test_gru_forecast.py
	uv run pytest --cov=src/rri --cov-report=term-missing --cov-append \
		tests/unit/test_mlp_entity_embeddings.py tests/unit/test_gru_forecast.py

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
