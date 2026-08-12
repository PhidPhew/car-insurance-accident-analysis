import sys
from pathlib import Path

import pytest

from config.settings import Environment, Settings, detect_environment, settings


def test_random_seed_is_42() -> None:
    assert settings.random_seed == 42


def test_detect_environment_defaults_to_local() -> None:
    assert "google.colab" not in sys.modules
    assert detect_environment() is Environment.LOCAL


def test_detect_environment_detects_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert detect_environment() is Environment.COLAB


def test_settings_derived_paths_are_under_base_dir() -> None:
    assert settings.raw_dir == settings.base_dir / "data" / "00_raw"
    assert settings.staging_dir == settings.data_dir / "10_staging"
    assert settings.mappings_dir == settings.config_dir / "mappings"
    assert settings.warehouse_db_path == settings.warehouse_dir / "warehouse.duckdb"


def test_settings_env_var_overrides_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RRI_RANDOM_SEED", "7")
    overridden = Settings()
    assert overridden.random_seed == 7


def test_ensure_writable_dirs_never_creates_raw_dir(tmp_path: Path) -> None:
    custom = Settings(base_dir=tmp_path)
    custom.ensure_writable_dirs()

    assert not custom.raw_dir.exists()
    assert custom.staging_dir.exists()
    assert custom.reports_dir.exists()
