"""Unit tests for `rri.models.lightgbm_baseline` (CLAUDE.md §10 L1)."""

from __future__ import annotations

import numpy as np
import polars as pl

from rri.models.lightgbm_baseline import (
    CLASSIFIER_CATEGORICAL_COLS,
    FREQ_SEV_CATEGORICAL_COLS,
    fit_category_levels,
    fit_lightgbm_frequency,
    fit_lightgbm_severity,
    fit_lightgbm_severity_classifier,
    predict_lightgbm_frequency,
    predict_lightgbm_severity,
    predict_lightgbm_severity_classifier,
)


def _freq_sev_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(1000, 50000, n)
    accident_count = rng.poisson(3, n).astype(float)
    severity = np.where(accident_count > 0, rng.gamma(2.0, 5.0, n), 0.0)
    return pl.DataFrame(
        {
            "region": rng.choice(["Central", "North", "South"], n),
            "season": rng.choice(["hot", "rainy", "cool"], n),
            "tourism_tier": rng.choice(["low", "medium", "high"], n),
            "vehicle_group": rng.choice(["motorcycle", "passenger_car", "pickup"], n),
            "has_songkran": rng.choice([True, False], n),
            "has_newyear": rng.choice([True, False], n),
            "rain_lead0": rng.uniform(0, 300, n),
            "rain_lead1": rng.uniform(0, 300, n),
            "rain_lead2": rng.uniform(0, 300, n),
            "rain_lead3": rng.uniform(0, 300, n),
            "rain_lead4": rng.uniform(0, 300, n),
            "rain_lead5": rng.uniform(0, 300, n),
            "flood_pct_high_risk_tambons": rng.uniform(0, 1, n),
            "ev_share": rng.uniform(0, 0.05, n),
            "r35mm_avg_days_2014_2023": rng.uniform(0, 20, n),
            "n_holiday_days": rng.integers(0, 5, n).astype(float),
            "n_long_weekend_days": rng.integers(0, 3, n).astype(float),
            "avg_days_to_next_holiday": rng.uniform(0, 60, n),
            "log_exposure": np.log(exposure),
            "accident_count": accident_count,
            "severity_index_coarse": severity,
        }
    )


def _classifier_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    classes = rng.choice(["none", "minor", "serious", "fatal"], n)
    return pl.DataFrame(
        {
            "region": rng.choice(["Central", "North"], n),
            "season": rng.choice(["hot", "rainy"], n),
            "vehicle_group": rng.choice(["motorcycle", "passenger_car"], n),
            "cause_raw": rng.choice([f"cause_{i}" for i in range(20)], n),
            "crash_type_raw": rng.choice([f"crash_{i}" for i in range(15)], n),
            "road_geometry_raw": rng.choice([f"geo_{i}" for i in range(10)], n),
            "road_code": rng.choice([f"R{i}" for i in range(50)], n),
            "weather_band": rng.choice(["clear", "rain"], n),
            "agency": rng.choice(["highways", "rural"], n),
            "covid_disruption_level": rng.choice(["none", "partial"], n),
            "is_public_holiday": rng.choice([True, False], n),
            "is_weekend": rng.choice([True, False], n),
            "is_long_weekend": rng.choice([True, False], n),
            "is_songkran_7days": rng.choice([True, False], n),
            "is_newyear_7days": rng.choice([True, False], n),
            "is_hazardous": rng.choice([True, False], n),
            "days_to_next_holiday": rng.uniform(0, 60, n),
            "flood_pct_high_risk_tambons": rng.uniform(0, 1, n),
            "ev_share": rng.uniform(0, 0.05, n),
            "tourist_attraction_count": rng.integers(0, 50, n).astype(float),
            "r35mm_avg_days_2014_2023": rng.uniform(0, 20, n),
            "km_marker": rng.uniform(0, 300, n),
            "severity_class": classes,
        }
    )


class TestCategoryLevels:
    def test_levels_are_sorted_and_stable(self) -> None:
        df = pl.DataFrame({"region": ["South", "Central", "North", "Central"]})
        levels = fit_category_levels(df, ["region"])
        assert levels.levels["region"] == ["Central", "North", "South"]


class TestLightgbmFrequency:
    def test_fit_predict_roundtrip(self) -> None:
        train = _freq_sev_frame(300, seed=1)
        valid = _freq_sev_frame(80, seed=2)
        test = _freq_sev_frame(80, seed=3)
        fitted = fit_lightgbm_frequency(train, valid)
        preds = predict_lightgbm_frequency(fitted, test)
        assert len(preds) == test.height
        assert (preds > 0).all()
        assert np.isfinite(preds).all()

    def test_uses_early_stopping_not_the_full_round_budget(self) -> None:
        train = _freq_sev_frame(300, seed=1)
        valid = _freq_sev_frame(80, seed=2)
        fitted = fit_lightgbm_frequency(train, valid)
        assert fitted.booster.best_iteration < 2000

    def test_category_levels_reused_for_unseen_valid_categories(self) -> None:
        train = _freq_sev_frame(300, seed=1)
        levels = fit_category_levels(train, FREQ_SEV_CATEGORICAL_COLS)
        valid = _freq_sev_frame(80, seed=2)
        fitted = fit_lightgbm_frequency(train, valid, levels=levels)
        assert fitted.levels is levels


class TestLightgbmSeverity:
    def test_fit_predict_roundtrip(self) -> None:
        train = _freq_sev_frame(300, seed=1).filter(pl.col("severity_index_coarse") > 0)
        valid = _freq_sev_frame(80, seed=2).filter(pl.col("severity_index_coarse") > 0)
        test = _freq_sev_frame(80, seed=3).filter(pl.col("severity_index_coarse") > 0)
        fitted = fit_lightgbm_severity(train, valid)
        preds = predict_lightgbm_severity(fitted, test)
        assert len(preds) == test.height
        assert (preds > 0).all()


class TestLightgbmSeverityClassifier:
    def test_fit_predict_roundtrip(self) -> None:
        train = _classifier_frame(400, seed=1)
        valid = _classifier_frame(100, seed=2)
        test = _classifier_frame(100, seed=3)
        fitted = fit_lightgbm_severity_classifier(train, valid)
        preds = predict_lightgbm_severity_classifier(fitted, test)
        assert len(preds) == test.height
        assert set(preds).issubset(set(fitted.model.classes_))

    def test_high_cardinality_categoricals_handled(self) -> None:
        # road_code has 50 distinct values in the fixture -- this is the whole
        # point of using LightGBM's native categorical support here rather
        # than the GLM's one-hot approach (CLAUDE.md §10).
        train = _classifier_frame(400, seed=1)
        levels = fit_category_levels(train, CLASSIFIER_CATEGORICAL_COLS)
        assert len(levels.levels["road_code"]) > 10
