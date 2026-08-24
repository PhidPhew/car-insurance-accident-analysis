"""SHAP explainability for the LightGBM models (CLAUDE.md §10 L1, Phase 5 Step 3).

Insurance pricing is regulated -- an unexplainable model is unusable (CLAUDE.md
§3). This module computes SHAP values with `shap.TreeExplainer` (exact, fast for
gradient-boosted trees, no sampling approximation needed) and produces two
artifacts per model: a beeswarm plot (`reports/shap_beeswarm_<name>.png`) and a
ranked feature-importance table with an approximate effect direction, which the
Step 4 business-sense review reads to sanity-check whether the top features make
domain sense.

**Direction is a correlation heuristic, not a causal claim**: for each of the top
features, this correlates the feature's raw value against its own SHAP value
across all rows. Positive correlation reads as "higher feature value pushes the
prediction up." For numeric features this is a clean signal; for categorical
features (`shap`'s stored `data` is the LightGBM-internal category code, not the
label) it's a weaker, code-order-dependent signal -- still directionally useful
since `fit_category_levels` fixes a stable (alphabetical) code order, but the
business-sense review should read categorical directions from the beeswarm plot's
per-level coloring, not the correlation number alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this module only ever writes PNGs, never displays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap


def compute_shap_values(model: object, x: pd.DataFrame) -> shap.Explanation:
    """Exact TreeExplainer SHAP values for a fitted LightGBM booster/regressor/classifier."""
    explainer = shap.TreeExplainer(model)
    return explainer(x)


def save_beeswarm(shap_values: shap.Explanation, output_path: Path, max_display: int = 8) -> Path:
    """Render and save a beeswarm plot; returns the path for convenient chaining."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _numeric_encode(column: np.ndarray) -> np.ndarray:
    """Coerce a SHAP `.data` column to float for the correlation heuristic.

    Numeric columns pass through as-is; string/categorical columns (SHAP stores
    the original label, not the LightGBM-internal code) are factorized into an
    arbitrary-but-stable integer order via `pandas.factorize` -- see the module
    docstring's caveat that categorical direction is a weaker, order-dependent
    signal for exactly this reason.
    """
    try:
        return column.astype(float)
    except (ValueError, TypeError):
        codes, _ = pd.factorize(column)
        return codes.astype(float)


def top_feature_directions(shap_values: shap.Explanation, n: int = 8) -> pl.DataFrame:
    """Top-`n` features by mean |SHAP|, with an approximate effect direction.

    See the module docstring's caveat on categorical-feature direction.
    """
    values = np.asarray(shap_values.values)
    raw_data = np.asarray(shap_values.data, dtype=object)
    feature_names = list(shap_values.feature_names)

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)[:n]

    rows = []
    for i in order:
        feature_values = _numeric_encode(raw_data[:, i])
        shap_col = values[:, i]
        valid = ~np.isnan(feature_values)
        if np.unique(feature_values[valid]).size > 1:
            corr = float(np.corrcoef(feature_values[valid], shap_col[valid])[0, 1])
        else:
            corr = 0.0
        direction = "increases" if corr > 0.05 else "decreases" if corr < -0.05 else "mixed/unclear"
        rows.append(
            {
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "value_shap_correlation": corr,
                "direction": direction,
            }
        )
    return pl.DataFrame(rows)


def multiclass_top_feature_directions(
    shap_values: shap.Explanation, class_names: list[str], target_class: str, n: int = 8
) -> pl.DataFrame:
    """Same as `top_feature_directions`, for one class slice of a multiclass explanation.

    `shap_values.values` for a multiclass TreeExplainer has shape
    `(n_samples, n_features, n_classes)`; this slices out `target_class` before
    delegating. `target_class` defaults to the class the business-sense review
    cares about most (see the Phase 5 Step 3 caller) -- typically "fatal", the
    most business-critical severity tier.
    """
    class_index = class_names.index(target_class)
    sliced = shap.Explanation(
        values=np.asarray(shap_values.values)[:, :, class_index],
        base_values=np.asarray(shap_values.base_values)[:, class_index],
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )
    return top_feature_directions(sliced, n=n)
