"""Unit tests for `rri.models.gru_forecast` (CLAUDE.md §10 L3, Phase 6 Step 6)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rri.models import gru_forecast as gf
from rri.models.gru_forecast import (
    FORECAST_HORIZON,
    LOOKBACK_MONTHS,
    build_gru_windows,
    build_province_month_series,
    fit_gru_forecast,
    predict_gru_forecast,
    seasonal_naive_forecast,
    split_windows,
)


def _synthetic_freq_sev_frame(n_provinces: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    months = [(y, m) for y in (2022, 2023, 2024, 2025) for m in range(1, 13)]
    rows = []
    for p in range(n_provinces):
        base = 10 + p * 5
        for year, month in months:
            for _vehicle_group in ("motorcycle", "passenger_car"):
                rows.append(
                    {
                        "province_th": f"province_{p}",
                        "year": year,
                        "month": month,
                        "accident_count": int(rng.poisson(base)),
                        **{f"rain_lead{h}": float(rng.uniform(0, 200)) for h in range(6)},
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture
def small_windows() -> gf.GruWindows:
    frame = _synthetic_freq_sev_frame()
    series = build_province_month_series(frame)
    return build_gru_windows(series)


class TestBuildProvinceMonthSeries:
    def test_collapses_vehicle_group_by_summing(self) -> None:
        frame = _synthetic_freq_sev_frame(n_provinces=1)
        series = build_province_month_series(frame)
        # 48 months (2022-2025), one row per (province, year, month)
        assert series.height == 48
        assert "accident_count" in series.columns


class TestBuildGruWindows:
    def test_shapes(self, small_windows: gf.GruWindows) -> None:
        assert small_windows.history.shape[1:] == (LOOKBACK_MONTHS, 3)
        assert small_windows.target_covariates.shape[1:] == (FORECAST_HORIZON, 3)
        assert small_windows.y.shape[1] == FORECAST_HORIZON

    def test_windows_only_from_full_year_target_spans(self, small_windows: gf.GruWindows) -> None:
        # 3 provinces x 3 usable years (2023/2024/2025) x 7 valid start months = 63
        assert len(small_windows.province) == 3 * 3 * 7

    def test_target_start_years_are_2023_to_2025(self, small_windows: gf.GruWindows) -> None:
        assert set(small_windows.target_start_year.tolist()) == {2023, 2024, 2025}


class TestSplitWindows:
    def test_partitions_by_target_year(self, small_windows: gf.GruWindows) -> None:
        splits = split_windows(small_windows)
        assert set(splits["train"].target_start_year.tolist()) <= {2023}
        assert set(splits["valid"].target_start_year.tolist()) == {2024}
        assert set(splits["test"].target_start_year.tolist()) == {2025}

    def test_no_overlap_across_splits(self, small_windows: gf.GruWindows) -> None:
        splits = split_windows(small_windows)
        total = sum(len(s.province) for s in splits.values())
        assert total == len(small_windows.province)


class TestSeasonalNaiveForecast:
    def test_matches_actual_count_from_12_months_earlier(
        self, small_windows: gf.GruWindows
    ) -> None:
        frame = _synthetic_freq_sev_frame()
        series = build_province_month_series(frame)
        splits = split_windows(small_windows)
        test = splits["test"]
        naive = seasonal_naive_forecast(series, test)

        # Spot-check the first test window's first forecast step manually.
        province = test.province[0]
        year = int(test.target_start_year[0])
        expected = series.filter(
            (pl.col("province_th") == province)
            & (pl.col("year") == year - 1)
            & (pl.col("month") == 1)
        )["accident_count"][0]
        assert naive[0, 0] == expected


class TestFitGruForecast:
    def test_fit_predict_roundtrip(
        self, small_windows: gf.GruWindows, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gf, "MAX_EPOCHS", 20)
        monkeypatch.setattr(gf, "EARLY_STOPPING_PATIENCE", 20)
        splits = split_windows(small_windows)
        fitted = fit_gru_forecast(splits["train"], splits["valid"])
        preds = predict_gru_forecast(fitted, splits["test"])
        assert preds.shape == splits["test"].y.shape
        assert (preds >= 0).all()  # Softplus output head, never negative

    def test_training_loss_decreases(
        self, small_windows: gf.GruWindows, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gf, "MAX_EPOCHS", 50)
        monkeypatch.setattr(gf, "EARLY_STOPPING_PATIENCE", 50)
        splits = split_windows(small_windows)
        fitted = fit_gru_forecast(splits["train"], splits["valid"])
        assert fitted.train_losses[-1] < fitted.train_losses[0]
