"""Unit tests for `rri.models.persistence` (CLAUDE.md rigor checklist, Phase 5/6 Step 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rri.models import gru_forecast as gf
from rri.models import mlp_entity_embeddings as me
from rri.models import persistence as per
from rri.models.glm_baseline import (
    compute_flood_fill_value,
    fit_poisson_frequency_glm,
    predict_glm,
)
from rri.models.gru_forecast import (
    build_gru_windows,
    build_province_month_series,
    fit_gru_forecast,
    predict_gru_forecast,
    seasonal_naive_forecast,
    split_windows,
)
from rri.models.lightgbm_baseline import fit_lightgbm_frequency, predict_lightgbm_frequency
from rri.models.mlp_entity_embeddings import fit_mlp_classifier, predict_mlp_classifier


def _freq_sev_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(1000, 50000, n)
    accident_count = rng.poisson(3, n).astype(float)
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
            "log_exposure": np.log(exposure),
            "accident_count": accident_count,
        }
    )


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(per.settings, "models_dir", tmp_path)


class TestGlmPersistence:
    def test_roundtrip_predictions_match(self) -> None:
        train = _freq_sev_frame(300, seed=1)
        test = _freq_sev_frame(80, seed=2)
        flood_fill = compute_flood_fill_value(train)
        fitted = fit_poisson_frequency_glm(train, flood_fill)
        before = predict_glm(fitted, test, flood_fill)

        per.save_glm(fitted, train, flood_fill, "glm_frequency")
        loaded = per.load_glm("glm_frequency")
        after = per.predict_persisted_glm(loaded, test)

        assert np.allclose(before, after)


class TestLightgbmPersistence:
    def test_roundtrip_predictions_match(self) -> None:
        train = _freq_sev_frame(300, seed=1)
        valid = _freq_sev_frame(80, seed=2)
        test = _freq_sev_frame(80, seed=3)
        fitted = fit_lightgbm_frequency(train, valid)
        before = predict_lightgbm_frequency(fitted, test)

        per.save_lightgbm(fitted, "lightgbm_frequency")
        loaded = per.load_lightgbm_frequency()
        after = predict_lightgbm_frequency(loaded, test)

        assert np.allclose(before, after)


def _classifier_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df = pl.DataFrame(
        {
            "province_th": rng.choice([f"province_{i}" for i in range(10)], n),
            "cause_raw": rng.choice([f"cause_{i}" for i in range(15)], n),
            "crash_type_raw": rng.choice([f"crash_{i}" for i in range(12)], n),
            "road_geometry_raw": rng.choice([f"geo_{i}" for i in range(8)], n),
            "road_code": rng.choice([f"R{i}" for i in range(30)], n),
            "season": rng.choice(["hot", "rainy"], n),
            "vehicle_group": rng.choice(["motorcycle", "passenger_car"], n),
            "weather_band": rng.choice(["clear", "rain"], n),
            "agency": rng.choice(["highways", "rural"], n),
            "covid_disruption_level": rng.choice(["none", "partial"], n),
            "severity_class": rng.choice(["none", "minor", "serious", "fatal"], n),
        }
    )
    for col in me.NUMERIC_COLS:
        df = df.with_columns(pl.Series(col, rng.uniform(0, 10, n)))
    for col in me.BOOLEAN_COLS:
        df = df.with_columns(pl.Series(col, rng.choice([True, False], n)))
    return df


class TestMlpPersistence:
    def test_roundtrip_predictions_match(self) -> None:
        train = _classifier_frame(300, seed=1)
        valid = _classifier_frame(80, seed=2)
        test = _classifier_frame(80, seed=3)
        fitted = fit_mlp_classifier(train, valid)
        before = predict_mlp_classifier(fitted, test)

        per.save_mlp(fitted)
        loaded = per.load_mlp()
        after = predict_mlp_classifier(loaded, test)

        assert (before == after).all()


class TestGruPersistence:
    def test_roundtrip_predictions_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gf, "MAX_EPOCHS", 20)
        monkeypatch.setattr(gf, "EARLY_STOPPING_PATIENCE", 20)

        rng = np.random.default_rng(0)
        months = [(y, m) for y in (2022, 2023, 2024, 2025) for m in range(1, 13)]
        rows = [
            {
                "province_th": "province_0",
                "year": year,
                "month": month,
                "accident_count": int(rng.poisson(10)),
                **{f"rain_lead{h}": float(rng.uniform(0, 200)) for h in range(6)},
            }
            for year, month in months
        ]
        frame = pl.DataFrame(rows)
        series = build_province_month_series(frame)
        windows = build_gru_windows(series)
        splits = split_windows(windows)
        fitted = fit_gru_forecast(splits["train"], splits["valid"])
        before = predict_gru_forecast(fitted, splits["test"])

        per.save_gru(fitted)
        loaded = per.load_gru()
        after = predict_gru_forecast(loaded, splits["test"])

        assert np.allclose(before, after)

    def test_seasonal_naive_still_importable_for_dashboard_comparison(self) -> None:
        # Not persisted (no fitting involved) -- the dashboard recomputes it directly
        # from the warehouse each time, so just confirm the function is reachable
        # from the same place as everything else the dashboard needs.
        assert callable(seasonal_naive_forecast)
