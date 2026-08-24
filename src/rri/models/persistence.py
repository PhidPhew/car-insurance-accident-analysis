"""Persisted model artifacts + a unified `predict()` interface (CLAUDE.md rigor checklist,
Phase 5/6 Step 7) -- what the Phase 7 Streamlit dashboard imports and calls, without needing
to know each model tier's internal representation.

**GLM persistence works around a real upstream limitation**: `patsy.DesignInfo` (the object
`rri.models.glm_baseline.FittedGlm` carries to encode new rows the same way as training)
explicitly refuses to pickle (`patsy.util.no_pickling` -- confirmed directly, not assumed:
`joblib.dump` on a `FittedGlm` raises `NotImplementedError`). Rather than hand-roll a
replica of patsy's treatment-coding column-naming convention (fragile, easy to silently
diverge from what the model was actually fit on), this persists three plain, always-picklable
pieces instead of the `FittedGlm`/`DesignInfo` themselves:

- `coefficients` (`result.params`, a `pandas.Series`),
- `reference_frame` (the exact flood-filled training frame the model was fit on), and
- `flood_fill_value`.

At load time, `dmatrices(FORMULA, reference_frame)` deterministically rebuilds a
`DesignInfo` structurally identical to the original (same source data in, same factor
levels/column order out), which `build_design_matrices` then uses to encode new rows exactly
as `FittedGlm`'s own `predict_glm` would -- verified by comparing predictions from the live
`FittedGlm` against this reconstructed path on held-out rows before relying on it (both
produce bit-identical output).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from patsy import build_design_matrices, dmatrices

from config.settings import settings
from rri.models.glm_baseline import (
    FEATURE_FORMULA_RHS,
    FREQUENCY_FORMULA,
    SEVERITY_FORMULA,
    FittedGlm,
    fill_flood_na,
)
from rri.models.gru_forecast import FittedGru
from rri.models.lightgbm_baseline import (
    FittedLgbmClassifier,
    FittedLgbmFrequency,
    FittedLgbmSeverity,
)
from rri.models.mlp_entity_embeddings import FittedMlp

_FILENAMES = {
    "glm_frequency": "glm_frequency.joblib",
    "glm_severity": "glm_severity.joblib",
    "lightgbm_frequency": "lightgbm_frequency.joblib",
    "lightgbm_severity": "lightgbm_severity.joblib",
    "lightgbm_classifier": "lightgbm_classifier.joblib",
    "mlp_classifier": "mlp_classifier.pt",
    "gru_forecast": "gru_forecast.pt",
}


def _path(name: str) -> Path:
    return settings.models_dir / _FILENAMES[name]


@dataclass(frozen=True)
class PersistedGlm:
    coefficients: pd.Series
    reference_frame: pd.DataFrame
    flood_fill_value: float
    offset_col: str | None


def save_glm(
    fitted: FittedGlm, train_reference_frame: pl.DataFrame, flood_fill_value: float, name: str
) -> Path:
    """`train_reference_frame` must be the exact (unfilled) frame `fitted` was trained on --
    `fill_flood_na` is reapplied here so the persisted reference matches what
    `fit_*_glm` actually saw."""
    persisted = PersistedGlm(
        coefficients=fitted.result.params,
        reference_frame=fill_flood_na(train_reference_frame, flood_fill_value).to_pandas(),
        flood_fill_value=flood_fill_value,
        offset_col=fitted.offset_col,
    )
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(persisted, path)
    return path


def load_glm(name: str) -> PersistedGlm:
    result: PersistedGlm = joblib.load(_path(name))
    return result


def predict_persisted_glm(persisted: PersistedGlm, df: pl.DataFrame) -> np.ndarray:
    formula = FREQUENCY_FORMULA if persisted.offset_col else SEVERITY_FORMULA
    _, reference_design = dmatrices(
        formula, data=persisted.reference_frame, return_type="dataframe"
    )

    pdf = fill_flood_na(df, persisted.flood_fill_value).to_pandas()
    (x_new,) = build_design_matrices([reference_design.design_info], pdf, return_type="dataframe")

    linear_predictor = x_new.to_numpy() @ persisted.coefficients.to_numpy()
    if persisted.offset_col:
        linear_predictor = linear_predictor + pdf[persisted.offset_col].to_numpy()
    return np.asarray(np.exp(linear_predictor))


def save_lightgbm(
    fitted: FittedLgbmFrequency | FittedLgbmSeverity | FittedLgbmClassifier, name: str
) -> Path:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)
    return path


def load_lightgbm_frequency() -> FittedLgbmFrequency:
    result: FittedLgbmFrequency = joblib.load(_path("lightgbm_frequency"))
    return result


def load_lightgbm_severity() -> FittedLgbmSeverity:
    result: FittedLgbmSeverity = joblib.load(_path("lightgbm_severity"))
    return result


def load_lightgbm_classifier() -> FittedLgbmClassifier:
    result: FittedLgbmClassifier = joblib.load(_path("lightgbm_classifier"))
    return result


def save_mlp(fitted: FittedMlp) -> Path:
    path = _path("mlp_classifier")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, path)
    return path


def load_mlp() -> FittedMlp:
    result: FittedMlp = torch.load(_path("mlp_classifier"), weights_only=False)
    return result


def save_gru(fitted: FittedGru) -> Path:
    path = _path("gru_forecast")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted, path)
    return path


def load_gru() -> FittedGru:
    result: FittedGru = torch.load(_path("gru_forecast"), weights_only=False)
    return result


__all__ = [
    "FEATURE_FORMULA_RHS",
    "PersistedGlm",
    "settings",
    "load_glm",
    "load_gru",
    "load_lightgbm_classifier",
    "load_lightgbm_frequency",
    "load_lightgbm_severity",
    "load_mlp",
    "predict_persisted_glm",
    "save_glm",
    "save_gru",
    "save_lightgbm",
    "save_mlp",
]
