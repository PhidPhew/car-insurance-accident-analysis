"""Unit tests for `rri.models.glm_baseline` (CLAUDE.md §10 L0)."""

from __future__ import annotations

import numpy as np
import polars as pl

from rri.models.glm_baseline import (
    compute_flood_fill_value,
    fit_gamma_severity_glm,
    fit_poisson_frequency_glm,
    frequency_training_rows,
    predict_glm,
    severity_training_rows,
)


def _synthetic_frame(n: int = 400, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    regions = rng.choice(["Central", "North", "South"], n)
    seasons = rng.choice(["hot", "rainy", "cool"], n)
    tiers = rng.choice(["low", "medium", "high"], n)
    vehicle_groups = rng.choice(["motorcycle", "passenger_car", "pickup"], n)
    rain_lead0 = rng.uniform(0, 300, n)
    flood_pct = rng.uniform(0, 1, n)
    ev_share = rng.uniform(0, 0.05, n)
    r35mm = rng.uniform(0, 20, n)
    n_holiday_days = rng.integers(0, 5, n).astype(float)
    n_long_weekend = rng.integers(0, 3, n).astype(float)
    has_songkran = rng.choice([True, False], n)
    has_newyear = rng.choice([True, False], n)
    exposure = rng.uniform(1000, 50000, n)
    log_exposure = np.log(exposure)
    accident_count = rng.poisson(3, n)
    severity = np.where(accident_count > 0, rng.gamma(2.0, 5.0, n), 0.0)

    return pl.DataFrame(
        {
            "region": regions,
            "season": seasons,
            "tourism_tier": tiers,
            "vehicle_group": vehicle_groups,
            "rain_lead0": rain_lead0,
            "flood_pct_high_risk_tambons": flood_pct,
            "ev_share": ev_share,
            "r35mm_avg_days_2014_2023": r35mm,
            "n_holiday_days": n_holiday_days,
            "n_long_weekend_days": n_long_weekend,
            "has_songkran": has_songkran,
            "has_newyear": has_newyear,
            "log_exposure": log_exposure,
            "accident_count": accident_count.astype(float),
            "severity_index_coarse": severity,
        }
    )


class TestTrainingRowFilters:
    def test_frequency_rows_require_non_null_exposure(self) -> None:
        df = _synthetic_frame(50).with_columns(
            pl.when(pl.arange(0, 50) < 10)
            .then(None)
            .otherwise(pl.col("log_exposure"))
            .alias("log_exposure")
        )
        result = frequency_training_rows(df)
        assert result.height == 40
        assert result["log_exposure"].null_count() == 0

    def test_severity_rows_require_positive_index_and_accidents(self) -> None:
        df = _synthetic_frame(200)
        result = severity_training_rows(df)
        assert (result["severity_index_coarse"] > 0).all()
        assert (result["accident_count"] > 0).all()
        assert result.height < df.height  # some rows have accident_count == 0


class TestPoissonFrequencyGlm:
    def test_fit_predict_roundtrip(self) -> None:
        train = _synthetic_frame(400, seed=1)
        test = _synthetic_frame(100, seed=2)
        flood_fill = compute_flood_fill_value(train)
        fitted = fit_poisson_frequency_glm(train, flood_fill)
        preds = predict_glm(fitted, test, flood_fill)
        assert len(preds) == test.height
        assert (preds > 0).all()  # Poisson log-link predictions are always positive

    def test_predictions_are_finite(self) -> None:
        train = _synthetic_frame(400, seed=1)
        flood_fill = compute_flood_fill_value(train)
        fitted = fit_poisson_frequency_glm(train, flood_fill)
        preds = predict_glm(fitted, train, flood_fill)
        assert np.isfinite(preds).all()


class TestGammaSeverityGlm:
    def test_fit_predict_roundtrip(self) -> None:
        train = severity_training_rows(_synthetic_frame(400, seed=1))
        test = severity_training_rows(_synthetic_frame(100, seed=2))
        flood_fill = compute_flood_fill_value(train)
        fitted = fit_gamma_severity_glm(train, flood_fill)
        preds = predict_glm(fitted, test, flood_fill)
        assert len(preds) == test.height
        assert (preds > 0).all()  # Gamma log-link predictions are always positive
