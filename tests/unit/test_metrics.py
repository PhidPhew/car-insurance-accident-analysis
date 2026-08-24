"""Unit tests for `rri.models.metrics` (CLAUDE.md §10 -- Poisson/Gamma deviance, not RMSE)."""

from __future__ import annotations

import numpy as np
import pytest

from rri.models.metrics import (
    gamma_deviance,
    gini_coefficient,
    lift_table,
    mean_absolute_error,
    poisson_deviance,
)


class TestPoissonDeviance:
    def test_zero_for_perfect_prediction(self) -> None:
        y = np.array([0, 1, 2, 3, 5, 0, 0, 1], dtype=float)
        assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_worse_prediction_has_higher_deviance(self) -> None:
        y = np.array([0, 1, 2, 3, 5, 0, 0, 1], dtype=float)
        good = y + 0.1
        bad = np.full_like(y, y.mean())
        assert poisson_deviance(y, bad) > poisson_deviance(y, good)

    def test_handles_exact_zero_counts(self) -> None:
        # The y=0 convention (y*log(y/mu) treated as 0) must not raise/NaN --
        # most province x month x vehicle_group cells are exactly 0.
        y = np.zeros(10)
        pred = np.full(10, 0.5)
        result = poisson_deviance(y, pred)
        assert np.isfinite(result)
        assert result > 0


class TestGammaDeviance:
    def test_zero_for_perfect_prediction(self) -> None:
        y = np.array([10.0, 50.0, 200.0, 5.0])
        assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_worse_prediction_has_higher_deviance(self) -> None:
        y = np.array([10.0, 50.0, 200.0, 5.0])
        good = y * 1.05
        bad = np.full_like(y, y.mean())
        assert gamma_deviance(y, bad) > gamma_deviance(y, good)


class TestMeanAbsoluteError:
    def test_matches_manual_computation(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx((0.5 + 0.5 + 1.0) / 3)


class TestGiniCoefficient:
    def test_perfect_ranking_is_one(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.poisson(3, 500).astype(float)
        assert gini_coefficient(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_random_ranking_is_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.poisson(3, 2000).astype(float)
        random_pred = rng.random(2000)
        assert abs(gini_coefficient(y, random_pred)) < 0.1

    def test_all_zero_actuals_returns_zero_not_nan(self) -> None:
        y = np.zeros(10)
        pred = np.arange(10, dtype=float)
        assert gini_coefficient(y, pred) == 0.0


class TestLiftTable:
    def test_bins_and_row_count(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.poisson(3, 1000).astype(float)
        table = lift_table(y, y, n_bins=10)
        assert table.height == 10
        assert table["decile"].to_list() == list(range(1, 11))

    def test_monotonic_for_perfect_ranking(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.poisson(3, 1000).astype(float)
        table = lift_table(y, y, n_bins=10)
        means = table["mean_actual"].to_list()
        assert all(means[i] <= means[i + 1] for i in range(len(means) - 1))
