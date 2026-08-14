"""Model evaluation metrics (CLAUDE.md §10).

Frequency (a Poisson-like count) and severity (a Gamma-like right-skewed cost) are
different distributions, so they get different deviance metrics -- RMSE is never
used for either (CLAUDE.md §10's core principle: never score a count target the
same way as a continuous one). `gini_coefficient`/`lift_table` are
distribution-agnostic risk-ranking diagnostics, used for both.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_EPS = 1e-10


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance -- the frequency model's headline metric, not RMSE.

    `D = 2 * mean(y * log(y / mu) - (y - mu))`, with the standard `y=0` convention
    that `y * log(y / mu) = 0` (the limit as y -> 0), so exact zero counts (most
    province x month x vehicle_group cells) contribute cleanly.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), _EPS, None)
    log_ratio = np.where(y_true > 0, y_true * np.log(np.clip(y_true, _EPS, None) / y_pred), 0.0)
    return float(2.0 * np.mean(log_ratio - (y_true - y_pred)))


def gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Gamma deviance -- the severity model's headline metric.

    `D = 2 * mean(-log(y / mu) + (y - mu) / mu)`. Both `y` and `mu` must be
    strictly positive (Gamma support) -- callers are expected to have already
    restricted to the positive-severity subset (see `rri.models.glm_baseline`).
    """
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), _EPS, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), _EPS, None)
    return float(2.0 * np.mean(-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain MAE, reported alongside Gamma deviance for severity (CLAUDE.md Step 2)."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred))))


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Gini coefficient for risk ranking (CLAUDE.md §10).

    Sorts observations by *predicted* risk ascending, traces the Lorenz curve of
    cumulative actual outcome against cumulative population share, and reports
    the area between that curve and the diagonal, normalized by the same curve's
    area under a perfect (sort-by-actual) ranking -- 1.0 is a perfect ranking,
    0.0 is no better than random, and this normalization is what makes Gini
    comparable across models/datasets of different scale (unlike the raw area).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return 0.0

    def _lorenz_area(order: np.ndarray) -> float:
        sorted_actual = y_true[order]
        cum_actual = np.concatenate([[0.0], np.cumsum(sorted_actual)]) / sorted_actual.sum()
        cum_population = np.arange(n + 1) / n
        return float(np.trapezoid(cum_actual, cum_population))

    gini_model = 1.0 - 2.0 * _lorenz_area(np.argsort(y_pred, kind="stable"))
    gini_perfect = 1.0 - 2.0 * _lorenz_area(np.argsort(y_true, kind="stable"))
    if gini_perfect == 0:
        return 0.0
    return float(gini_model / gini_perfect)


def lift_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Decile lift table: predicted-risk bins vs. their actual mean outcome.

    A well-ranked model shows a monotonically increasing `mean_actual` from
    decile 1 (lowest predicted risk) to decile `n_bins` (highest) -- this is
    what the business-sense review (CLAUDE.md Step 4) reads to sanity-check
    ranking quality beyond a single scalar Gini number.
    """
    df = pl.DataFrame({"y_true": y_true, "y_pred": y_pred})
    ranked = df.sort("y_pred").with_row_index("_rank")
    n = ranked.height
    bin_expr = (pl.col("_rank") * n_bins // n).clip(0, n_bins - 1).alias("decile")
    ranked = ranked.with_columns(bin_expr)
    return (
        ranked.group_by("decile")
        .agg(
            pl.len().alias("n"),
            pl.col("y_true").mean().alias("mean_actual"),
            pl.col("y_pred").mean().alias("mean_predicted"),
        )
        .sort("decile")
        .with_columns((pl.col("decile") + 1).alias("decile"))
    )
