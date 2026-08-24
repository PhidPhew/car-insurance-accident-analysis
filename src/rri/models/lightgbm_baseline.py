"""LightGBM baselines (CLAUDE.md §10 L1) -- head-to-head against the GLM (L0).

Three models, same `objective=poisson`/`objective=gamma` split as the GLM tier
(CLAUDE.md's core frequency != severity distribution principle applies here too):

- **Frequency**: `objective="poisson"` on the frequency/severity frame, offset by
  `log_exposure` -- LightGBM's sklearn wrapper has no offset argument that also
  applies at predict time, so this uses the native `lgb.train` API directly:
  `init_score=log_exposure` at train time, then `raw_score=True` + manually adding
  the offset back at predict time (`exp(raw_score + log_exposure)`). This is the
  standard, well-documented pattern for exposure offsets in LightGBM -- getting it
  wrong (e.g. relying on the sklearn wrapper's `init_score` alone) silently drops
  the offset at prediction time.
- **Severity**: `objective="gamma"` on the same frame's `severity_index_coarse`,
  same positive-severity row restriction as the GLM (`glm_baseline.severity_training_rows`).
- **Classifier**: `objective="multiclass"` on the record-level severity-classification
  frame, using the *raw* high-cardinality categoricals (CLAUDE.md §10's
  justification for entity embeddings in Phase 6 applies here as LightGBM's own
  native categorical-split handling).

Categorical columns are encoded as pandas `category` dtype with levels fixed from
the training split (`fit_category_levels`) and reapplied unchanged to valid/test --
LightGBM's native categorical handling needs consistent category codes across
splits, same reasoning as the GLM's `build_design_matrices` reuse.

**`avg_days_to_next_holiday` deliberately excluded from `FREQ_SEV_NUMERIC_COLS`**
(Phase 5 Step 4 business-sense review, checked 2026-08-14): it ranked #2 by SHAP
importance in the frequency model, but its partial-dependence curve was
non-monotonic/spiky rather than a coherent trend, and it correlates -0.71 with
`n_holiday_days` in this frame -- symptomatic of fighting that cleaner signal
collinearly rather than adding real information. `n_holiday_days`/`has_songkran`/
`has_newyear` already cover the holiday-risk story (BQ3) without this ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from config.simulation_params import RANDOM_SEED

FREQ_SEV_CATEGORICAL_COLS = ["region", "season", "tourism_tier", "vehicle_group"]
FREQ_SEV_BOOLEAN_COLS = ["has_songkran", "has_newyear"]
FREQ_SEV_NUMERIC_COLS = [
    "rain_lead0",
    "rain_lead1",
    "rain_lead2",
    "rain_lead3",
    "rain_lead4",
    "rain_lead5",
    "flood_pct_high_risk_tambons",
    "ev_share",
    "r35mm_avg_days_2014_2023",
    "n_holiday_days",
    "n_long_weekend_days",
]
FREQ_SEV_FEATURE_COLS = FREQ_SEV_CATEGORICAL_COLS + FREQ_SEV_BOOLEAN_COLS + FREQ_SEV_NUMERIC_COLS

CLASSIFIER_CATEGORICAL_COLS = [
    "region",
    "season",
    "vehicle_group",
    "cause_raw",
    "crash_type_raw",
    "road_geometry_raw",
    "road_code",
    "weather_band",
    "agency",
    "covid_disruption_level",
]
CLASSIFIER_BOOLEAN_COLS = [
    "is_public_holiday",
    "is_weekend",
    "is_long_weekend",
    "is_songkran_7days",
    "is_newyear_7days",
    "is_hazardous",
]
CLASSIFIER_NUMERIC_COLS = [
    "days_to_next_holiday",
    "flood_pct_high_risk_tambons",
    "ev_share",
    "tourist_attraction_count",
    "r35mm_avg_days_2014_2023",
    "km_marker",
]
CLASSIFIER_FEATURE_COLS = (
    CLASSIFIER_CATEGORICAL_COLS + CLASSIFIER_BOOLEAN_COLS + CLASSIFIER_NUMERIC_COLS
)

_LGBM_PARAMS_BASE: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "seed": RANDOM_SEED,
    "verbose": -1,
    "deterministic": True,
}
_MAX_BOOST_ROUNDS = 2000
_EARLY_STOPPING_ROUNDS = 30


def _early_stopping_callbacks() -> list[Any]:
    return [lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]


@dataclass(frozen=True)
class CategoryLevels:
    levels: dict[str, list[str]]


def fit_category_levels(train: pl.DataFrame, categorical_cols: list[str]) -> CategoryLevels:
    """Fixed category levels from `train`, reapplied unchanged to valid/test."""
    return CategoryLevels(
        {
            col: sorted(train[col].cast(pl.Utf8).unique().drop_nulls().to_list())
            for col in categorical_cols
        }
    )


def _to_pandas_features(
    df: pl.DataFrame,
    levels: CategoryLevels,
    feature_cols: list[str],
    categorical_cols: list[str],
    boolean_cols: list[str],
) -> pd.DataFrame:
    pdf = df.select(feature_cols).to_pandas()
    for col in categorical_cols:
        pdf[col] = pd.Categorical(pdf[col].astype(object), categories=levels.levels[col])
    for col in boolean_cols:
        pdf[col] = pdf[col].astype(float)
    return pdf


def freq_sev_feature_matrix(df: pl.DataFrame, levels: CategoryLevels) -> pd.DataFrame:
    """Public feature-matrix builder for the frequency/severity models (e.g. for SHAP)."""
    return _to_pandas_features(
        df, levels, FREQ_SEV_FEATURE_COLS, FREQ_SEV_CATEGORICAL_COLS, FREQ_SEV_BOOLEAN_COLS
    )


def classifier_feature_matrix(df: pl.DataFrame, levels: CategoryLevels) -> pd.DataFrame:
    """Public feature-matrix builder for the severity classifier (e.g. for SHAP)."""
    return _to_pandas_features(
        df, levels, CLASSIFIER_FEATURE_COLS, CLASSIFIER_CATEGORICAL_COLS, CLASSIFIER_BOOLEAN_COLS
    )


@dataclass(frozen=True)
class FittedLgbmFrequency:
    booster: lgb.Booster
    levels: CategoryLevels


def fit_lightgbm_frequency(
    train: pl.DataFrame, valid: pl.DataFrame, levels: CategoryLevels | None = None
) -> FittedLgbmFrequency:
    """Fit the Poisson-objective frequency model, `log_exposure` as the init-score offset."""
    levels = levels or fit_category_levels(train, FREQ_SEV_CATEGORICAL_COLS)
    x_train = freq_sev_feature_matrix(train, levels)
    x_valid = freq_sev_feature_matrix(valid, levels)
    dtrain = lgb.Dataset(
        x_train,
        label=train["accident_count"].to_numpy(),
        init_score=train["log_exposure"].to_numpy(),
        categorical_feature=FREQ_SEV_CATEGORICAL_COLS,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        x_valid,
        label=valid["accident_count"].to_numpy(),
        init_score=valid["log_exposure"].to_numpy(),
        reference=dtrain,
        categorical_feature=FREQ_SEV_CATEGORICAL_COLS,
        free_raw_data=False,
    )
    booster = lgb.train(
        {**_LGBM_PARAMS_BASE, "objective": "poisson"},
        dtrain,
        num_boost_round=_MAX_BOOST_ROUNDS,
        valid_sets=[dvalid],
        callbacks=_early_stopping_callbacks(),
    )
    return FittedLgbmFrequency(booster=booster, levels=levels)


def predict_lightgbm_frequency(fitted: FittedLgbmFrequency, df: pl.DataFrame) -> np.ndarray:
    """Predict frequency, manually re-adding the `log_exposure` offset (see module docstring)."""
    x = freq_sev_feature_matrix(df, fitted.levels)
    raw_score = fitted.booster.predict(
        x, raw_score=True, num_iteration=fitted.booster.best_iteration
    )
    offset = df["log_exposure"].to_numpy()
    return np.asarray(np.exp(np.asarray(raw_score) + offset))


@dataclass(frozen=True)
class FittedLgbmSeverity:
    model: lgb.LGBMRegressor
    levels: CategoryLevels


def fit_lightgbm_severity(
    train: pl.DataFrame, valid: pl.DataFrame, levels: CategoryLevels | None = None
) -> FittedLgbmSeverity:
    """Fit the Gamma-objective severity model (no offset -- not exposure-normalized)."""
    levels = levels or fit_category_levels(train, FREQ_SEV_CATEGORICAL_COLS)
    x_train = freq_sev_feature_matrix(train, levels)
    x_valid = freq_sev_feature_matrix(valid, levels)
    model = lgb.LGBMRegressor(
        objective="gamma", n_estimators=_MAX_BOOST_ROUNDS, **_LGBM_PARAMS_BASE
    )
    model.fit(
        x_train,
        train["severity_index_coarse"].to_numpy(),
        eval_set=[(x_valid, valid["severity_index_coarse"].to_numpy())],
        eval_metric="gamma",
        categorical_feature=FREQ_SEV_CATEGORICAL_COLS,
        callbacks=_early_stopping_callbacks(),
    )
    return FittedLgbmSeverity(model=model, levels=levels)


def predict_lightgbm_severity(fitted: FittedLgbmSeverity, df: pl.DataFrame) -> np.ndarray:
    x = freq_sev_feature_matrix(df, fitted.levels)
    return np.asarray(fitted.model.predict(x))


@dataclass(frozen=True)
class FittedLgbmClassifier:
    model: lgb.LGBMClassifier
    levels: CategoryLevels


def fit_lightgbm_severity_classifier(
    train: pl.DataFrame, valid: pl.DataFrame, levels: CategoryLevels | None = None
) -> FittedLgbmClassifier:
    """Fit the multiclass severity classifier on the record-level frame."""
    levels = levels or fit_category_levels(train, CLASSIFIER_CATEGORICAL_COLS)
    x_train = classifier_feature_matrix(train, levels)
    x_valid = classifier_feature_matrix(valid, levels)
    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=_MAX_BOOST_ROUNDS, **_LGBM_PARAMS_BASE
    )
    model.fit(
        x_train,
        train["severity_class"].to_numpy(),
        eval_set=[(x_valid, valid["severity_class"].to_numpy())],
        eval_metric="multi_logloss",
        categorical_feature=CLASSIFIER_CATEGORICAL_COLS,
        callbacks=_early_stopping_callbacks(),
    )
    return FittedLgbmClassifier(model=model, levels=levels)


def predict_lightgbm_severity_classifier(
    fitted: FittedLgbmClassifier, df: pl.DataFrame
) -> np.ndarray:
    x = classifier_feature_matrix(df, fitted.levels)
    return np.asarray(fitted.model.predict(x))
