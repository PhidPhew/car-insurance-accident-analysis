"""Unit tests for `rri.models.shap_explain` (CLAUDE.md §10, Phase 5 Step 3)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from rri.models.shap_explain import (
    compute_shap_values,
    multiclass_top_feature_directions,
    save_beeswarm,
    top_feature_directions,
)


def _regressor_fixture() -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 300
    x = pd.DataFrame(
        {
            "strong_numeric": rng.uniform(0, 10, n),
            "noise_numeric": rng.uniform(0, 10, n),
            "region": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        }
    )
    # Target driven almost entirely by strong_numeric -- noise_numeric and
    # region should rank far below it in mean |SHAP|.
    y = x["strong_numeric"] * 3.0 + rng.normal(0, 0.1, n)
    model = lgb.LGBMRegressor(n_estimators=50, num_leaves=7, verbose=-1, random_state=0)
    model.fit(x, y, categorical_feature=["region"])
    return model, x


class TestComputeShapValues:
    def test_shape_matches_input(self) -> None:
        model, x = _regressor_fixture()
        sv = compute_shap_values(model, x)
        assert sv.values.shape == (len(x), x.shape[1])


class TestTopFeatureDirections:
    def test_strongest_driver_ranks_first(self) -> None:
        model, x = _regressor_fixture()
        sv = compute_shap_values(model, x)
        table = top_feature_directions(sv, n=3)
        assert table["feature"][0] == "strong_numeric"

    def test_positive_driver_direction_is_increases(self) -> None:
        model, x = _regressor_fixture()
        sv = compute_shap_values(model, x)
        table = top_feature_directions(sv, n=3)
        row = table.filter(table["feature"] == "strong_numeric")
        assert row["direction"][0] == "increases"

    def test_categorical_column_does_not_crash_numeric_encoding(self) -> None:
        model, x = _regressor_fixture()
        sv = compute_shap_values(model, x)
        table = top_feature_directions(sv, n=3)
        assert "region" in table["feature"].to_list() or table.height == 3


class TestSaveBeeswarm:
    def test_writes_a_png_file(self, tmp_path: Path) -> None:
        model, x = _regressor_fixture()
        sv = compute_shap_values(model, x)
        out = save_beeswarm(sv, tmp_path / "beeswarm.png")
        assert out.exists()
        assert out.stat().st_size > 0


def _classifier_fixture() -> tuple[lgb.LGBMClassifier, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(1)
    n = 400
    x = pd.DataFrame(
        {
            "strong_numeric": rng.uniform(0, 10, n),
            "other": rng.uniform(0, 10, n),
        }
    )
    # "fatal" class label strongly driven by strong_numeric being high.
    y = np.where(
        x["strong_numeric"] > 7,
        "fatal",
        rng.choice(["none", "minor", "serious"], n),
    )
    model = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, verbose=-1, random_state=0)
    model.fit(x, y)
    return model, x, sorted(set(y))


class TestMulticlassTopFeatureDirections:
    def test_slices_the_target_class_correctly(self) -> None:
        model, x, class_names = _classifier_fixture()
        sv = compute_shap_values(model, x)
        assert sv.values.ndim == 3
        table = multiclass_top_feature_directions(sv, class_names, "fatal", n=2)
        assert table["feature"][0] == "strong_numeric"
        assert table.filter(table["feature"] == "strong_numeric")["direction"][0] == "increases"

    def test_raises_on_unknown_class(self) -> None:
        model, x, class_names = _classifier_fixture()
        sv = compute_shap_values(model, x)
        with pytest.raises(ValueError, match="not in list"):
            multiclass_top_feature_directions(sv, class_names, "does_not_exist", n=2)
