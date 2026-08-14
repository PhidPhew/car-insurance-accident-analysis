"""GLM baselines (CLAUDE.md §10 L0) -- the actuarial-standard comparison line.

Poisson GLM (log link, `log_exposure` offset) for frequency, Gamma GLM (log link)
for severity, both fit on `rri.features.frequency_severity`'s province x month x
vehicle_group frame with a fixed formula (no feature search -- this tier's whole
point is being the simple, well-understood baseline LightGBM has to beat).

**Frequency rows**: restricted to the 3 vehicle groups with a real exposure
denominator (`log_exposure` non-null) -- CLAUDE.md §7.2, the other 2 groups have
no registration source to build a Poisson offset from.

**Severity rows**: restricted to `severity_index_coarse > 0`. Gamma's support is
strictly positive, and ~16% of accident-having cells have a coarse index of
exactly 0 (every accident in that cell was property-damage-only, no casualties).
This mirrors standard actuarial practice: a severity model is conditional on
there being something to size a cost for; it does not also predict *whether*
a cell has any costed severity at all (that's the frequency model's job).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices
from patsy.design_info import DesignInfo

#: Shared right-hand-side formula for both GLMs -- deliberately compact (no
#: interaction terms, no feature search) since this tier exists to be the
#: simple, interpretable baseline, not to compete on features.
FEATURE_FORMULA_RHS = (
    "C(region) + C(season) + C(tourism_tier) + C(vehicle_group) + rain_lead0"
    " + flood_pct_high_risk_tambons + ev_share + r35mm_avg_days_2014_2023"
    " + n_holiday_days + n_long_weekend_days + has_songkran + has_newyear"
)

FREQUENCY_FORMULA = f"accident_count ~ {FEATURE_FORMULA_RHS}"
SEVERITY_FORMULA = f"severity_index_coarse ~ {FEATURE_FORMULA_RHS}"


@dataclass(frozen=True)
class FittedGlm:
    result: sm.genmod.generalized_linear_model.GLMResultsWrapper
    design_info: DesignInfo
    offset_col: str | None


def compute_flood_fill_value(train: pl.DataFrame) -> float:
    """Train-set mean of `flood_pct_high_risk_tambons`, for `flood_fill_value` below.

    Computed once from train and reused for valid/test -- fitting the fill value on
    the training split only, then applying it unchanged downstream, avoids leaking
    valid/test statistics into what is nominally a training-time imputation.
    """
    mean = train["flood_pct_high_risk_tambons"].mean()
    if mean is None:
        raise ValueError("train has no non-null flood_pct_high_risk_tambons to average")
    return float(mean)  # type: ignore[arg-type]


def _fill_flood_na(df: pl.DataFrame, fill_value: float) -> pl.DataFrame:
    """Mean-impute `flood_pct_high_risk_tambons` (null for Phuket/Ranong, CLAUDE.md P6).

    `flood_data_available` stays in the frame as the honest missingness signal;
    this fill only exists so the GLM's design matrix has no NaN to choke on.
    """
    return df.with_columns(pl.col("flood_pct_high_risk_tambons").fill_null(fill_value))


def frequency_training_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with a real exposure offset (CLAUDE.md §7.2 -- see module docstring)."""
    return df.filter(pl.col("log_exposure").is_not_null())


def severity_training_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with a strictly-positive severity index (Gamma support, see module docstring)."""
    return df.filter(pl.col("accident_count") > 0, pl.col("severity_index_coarse") > 0)


def fit_poisson_frequency_glm(train: pl.DataFrame, flood_fill_value: float) -> FittedGlm:
    """Fit the Poisson frequency GLM, offset by `log_exposure`."""
    pdf: pd.DataFrame = _fill_flood_na(train, flood_fill_value).to_pandas()
    y, x = dmatrices(FREQUENCY_FORMULA, data=pdf, return_type="dataframe")
    model = sm.GLM(y, x, family=sm.families.Poisson(), offset=pdf["log_exposure"].to_numpy())
    result = model.fit()
    return FittedGlm(result=result, design_info=x.design_info, offset_col="log_exposure")


def fit_gamma_severity_glm(train: pl.DataFrame, flood_fill_value: float) -> FittedGlm:
    """Fit the Gamma severity GLM (no offset -- severity isn't exposure-normalized)."""
    pdf: pd.DataFrame = _fill_flood_na(train, flood_fill_value).to_pandas()
    y, x = dmatrices(SEVERITY_FORMULA, data=pdf, return_type="dataframe")
    model = sm.GLM(y, x, family=sm.families.Gamma(link=sm.families.links.Log()))
    result = model.fit()
    return FittedGlm(result=result, design_info=x.design_info, offset_col=None)


def predict_glm(fitted: FittedGlm, df: pl.DataFrame, flood_fill_value: float) -> np.ndarray:
    """Predict on new rows using the training design matrix's fixed encoding.

    `build_design_matrices` reuses the training formula's category levels, so
    valid/test rows are encoded identically to train even if a category is
    absent from this particular slice.
    """
    pdf: pd.DataFrame = _fill_flood_na(df, flood_fill_value).to_pandas()
    (x,) = build_design_matrices([fitted.design_info], pdf, return_type="dataframe")
    offset = pdf[fitted.offset_col].to_numpy() if fitted.offset_col else None
    prediction = fitted.result.predict(x, offset=offset)
    return np.asarray(prediction)
