"""Single source of truth for every filesystem path in the project.

No module outside this file should hard-code a path. Import ``settings``
from here and derive everything else from it.

Two environments are supported:

* **local** — developing in VS Code, project root found by walking up from
  this file until ``pyproject.toml`` or ``.git`` is found.
* **colab** — Google Colab with Google Drive mounted, detected by the
  presence of ``google.colab`` in ``sys.modules``. The actual
  ``drive.mount(...)`` call is deliberately *not* made here — this module
  only decides *where* the project would live once mounted
  (``COLAB_PROJECT_DIR``, overridable via env var). Mounting is a Colab
  notebook's responsibility, not a library import's.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rri.utils.paths import ensure_dir, find_project_root


class Environment(StrEnum):
    """Where the pipeline is currently running."""

    LOCAL = "local"
    COLAB = "colab"


def detect_environment() -> Environment:
    """Detect the runtime environment.

    Colab always injects the ``google.colab`` module, so its presence in
    ``sys.modules`` is a reliable, side-effect-free check (it does not
    import or mount anything).
    """
    if "google.colab" in sys.modules:
        return Environment.COLAB
    return Environment.LOCAL


def _default_base_dir() -> Path:
    if detect_environment() is Environment.COLAB:
        # Design placeholder: the notebook is responsible for calling
        # `drive.mount('/content/drive')` before this path is used.
        # Override with the COLAB_PROJECT_DIR env var if the Drive folder
        # name differs from this default.
        return Path("/content/drive/MyDrive/road-risk-intelligence")
    return find_project_root()


class Settings(BaseSettings):
    """Project-wide configuration, overridable via environment variables or ``.env``.

    Every field prefixed ``RRI_`` in the environment overrides the matching
    field here, e.g. ``RRI_RANDOM_SEED=7``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RRI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Field(default_factory=detect_environment)
    base_dir: Path = Field(default_factory=_default_base_dir)

    random_seed: int = 42

    # -- derived paths (computed in model_post_init, not stored as literal
    #    defaults, since they all depend on base_dir) --
    config_dir: Path = Path()
    mappings_dir: Path = Path()
    data_dir: Path = Path()
    raw_dir: Path = Path()
    staging_dir: Path = Path()
    intermediate_dir: Path = Path()
    warehouse_dir: Path = Path()
    exports_dir: Path = Path()
    models_dir: Path = Path()
    notebooks_dir: Path = Path()
    reports_dir: Path = Path()
    logs_dir: Path = Path()
    tests_dir: Path = Path()
    fixtures_dir: Path = Path()

    def model_post_init(self, __context: object) -> None:
        base = self.base_dir
        self.config_dir = base / "config"
        self.mappings_dir = self.config_dir / "mappings"
        self.data_dir = base / "data"
        self.raw_dir = self.data_dir / "00_raw"
        self.staging_dir = self.data_dir / "10_staging"
        self.intermediate_dir = self.data_dir / "20_intermediate"
        self.warehouse_dir = self.data_dir / "30_warehouse"
        self.exports_dir = self.data_dir / "40_exports"
        self.models_dir = self.data_dir / "50_models"
        self.notebooks_dir = base / "notebooks"
        self.reports_dir = base / "reports"
        self.logs_dir = base / "logs"
        self.tests_dir = base / "tests"
        self.fixtures_dir = self.tests_dir / "fixtures"

    @property
    def warehouse_db_path(self) -> Path:
        return self.warehouse_dir / "warehouse.duckdb"

    def ensure_writable_dirs(self) -> None:
        """Create every directory this project writes to, except ``raw_dir``.

        ``raw_dir`` (data/00_raw) is read-only per CLAUDE.md N1 and must be
        populated manually — this deliberately never creates it, so a
        missing raw directory fails loudly instead of silently appearing
        empty.
        """
        for path in (
            self.staging_dir,
            self.intermediate_dir,
            self.warehouse_dir,
            self.exports_dir,
            self.models_dir,
            self.reports_dir,
            self.logs_dir,
            self.fixtures_dir,
            self.mappings_dir,
        ):
            ensure_dir(path)


settings = Settings()
