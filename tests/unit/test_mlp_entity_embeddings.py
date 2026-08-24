"""Unit tests for `rri.models.mlp_entity_embeddings` (CLAUDE.md §10 L2, Phase 6 Step 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from rri.models.mlp_entity_embeddings import (
    BOOLEAN_COLS,
    EMBEDDING_COLS,
    NUMERIC_COLS,
    ONEHOT_COLS,
    embedding_dim_for_cardinality,
    encode_column,
    encode_onehot,
    fit_category_encoder,
    fit_mlp_classifier,
    fit_numeric_scaler,
    fit_onehot_encoder,
    predict_mlp_classifier,
    project_province_embeddings_2d,
    province_embedding_table,
    save_province_embedding_plot,
    transform_numeric,
)


def _synthetic_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    provinces = [f"province_{i}" for i in range(10)]
    classes = rng.choice(["none", "minor", "serious", "fatal"], n)
    df = pl.DataFrame(
        {
            "province_th": rng.choice(provinces, n),
            "cause_raw": rng.choice([f"cause_{i}" for i in range(15)], n),
            "crash_type_raw": rng.choice([f"crash_{i}" for i in range(12)], n),
            "road_geometry_raw": rng.choice([f"geo_{i}" for i in range(8)], n),
            "road_code": rng.choice([f"R{i}" for i in range(30)], n),
            "season": rng.choice(["hot", "rainy"], n),
            "vehicle_group": rng.choice(["motorcycle", "passenger_car"], n),
            "weather_band": rng.choice(["clear", "rain"], n),
            "agency": rng.choice(["highways", "rural"], n),
            "covid_disruption_level": rng.choice(["none", "partial"], n),
            "severity_class": classes,
        }
    )
    for col in NUMERIC_COLS:
        df = df.with_columns(pl.Series(col, rng.uniform(0, 10, n)))
    for col in BOOLEAN_COLS:
        df = df.with_columns(pl.Series(col, rng.choice([True, False], n)))
    return df


class TestEmbeddingDimForCardinality:
    def test_clamped_between_4_and_8(self) -> None:
        assert embedding_dim_for_cardinality(2) == 4
        assert embedding_dim_for_cardinality(100_000) == 8

    def test_increases_with_cardinality(self) -> None:
        assert embedding_dim_for_cardinality(5) <= embedding_dim_for_cardinality(3000)


class TestCategoryEncoder:
    def test_unseen_value_maps_to_zero(self) -> None:
        train_col = pl.Series(["a", "b", "a", "c"])
        encoder = fit_category_encoder(train_col)
        result = encode_column(pl.Series(["a", "unseen"]), encoder)
        assert result[1] == 0
        assert result[0] != 0

    def test_vocab_size_includes_unknown_slot(self) -> None:
        train_col = pl.Series(["a", "b", "c"])
        encoder = fit_category_encoder(train_col)
        assert encoder.vocab_size == 4  # 3 categories + reserved index 0


class TestOnehotEncoder:
    def test_unseen_value_is_all_zero_row(self) -> None:
        encoder = fit_onehot_encoder(pl.Series(["x", "y"]))
        result = encode_onehot(pl.Series(["x", "z"]), encoder)
        assert result[0].sum() == 1
        assert result[1].sum() == 0


class TestNumericScaler:
    def test_transform_is_standardized(self) -> None:
        train = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = fit_numeric_scaler(train, ["a"])
        transformed = transform_numeric(train, ["a"], scaler)
        assert abs(transformed.mean()) < 1e-5

    def test_null_filled_with_train_mean(self) -> None:
        train = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = fit_numeric_scaler(train, ["a"])
        test = pl.DataFrame({"a": [None]}, schema={"a": pl.Float64})
        transformed = transform_numeric(test, ["a"], scaler)
        assert transformed[0, 0] == 0.0  # train mean, standardized -> 0


class TestFitMlpClassifier:
    def test_fit_predict_roundtrip(self) -> None:
        train = _synthetic_frame(300, seed=1)
        valid = _synthetic_frame(80, seed=2)
        test = _synthetic_frame(80, seed=3)
        fitted = fit_mlp_classifier(train, valid)
        preds = predict_mlp_classifier(fitted, test)
        assert len(preds) == test.height
        assert set(preds).issubset(set(fitted.class_names))

    def test_early_stopping_restores_best_valid_loss_epoch(self) -> None:
        train = _synthetic_frame(300, seed=1)
        valid = _synthetic_frame(80, seed=2)
        fitted = fit_mlp_classifier(train, valid)
        assert fitted.best_epoch < len(fitted.valid_losses)
        # best_epoch tracks the best loss beyond a small improvement tolerance (see
        # fit_mlp_classifier), so it may trail the bare argmin by a hair -- not exceed it.
        assert fitted.valid_losses[fitted.best_epoch] <= min(fitted.valid_losses) + 1e-3

    def test_deterministic_with_fixed_seed(self) -> None:
        train = _synthetic_frame(200, seed=1)
        valid = _synthetic_frame(50, seed=2)
        test = _synthetic_frame(50, seed=3)
        fitted_a = fit_mlp_classifier(train, valid)
        fitted_b = fit_mlp_classifier(train, valid)
        preds_a = predict_mlp_classifier(fitted_a, test)
        preds_b = predict_mlp_classifier(fitted_b, test)
        assert (preds_a == preds_b).all()

    def test_embedding_columns_and_onehot_columns_disjoint(self) -> None:
        assert set(EMBEDDING_COLS).isdisjoint(set(ONEHOT_COLS))


class TestProvinceEmbeddingTable:
    def test_returns_only_real_provinces(self) -> None:
        train = _synthetic_frame(300, seed=1)
        valid = _synthetic_frame(80, seed=2)
        fitted = fit_mlp_classifier(train, valid)
        table = province_embedding_table(fitted)
        assert "Unknown" not in table["province_th"].to_list()
        assert table.height == 10  # the fixture's 10 synthetic provinces

    def test_projection_and_plot(self, tmp_path: Path) -> None:
        train = _synthetic_frame(300, seed=1)
        valid = _synthetic_frame(80, seed=2)
        fitted = fit_mlp_classifier(train, valid)
        table = province_embedding_table(fitted)
        projected = project_province_embeddings_2d(table)
        assert projected.height == table.height
        assert {"tsne_x", "tsne_y"} <= set(projected.columns)

        dim_geography = pl.DataFrame(
            {
                "province_th": table["province_th"].to_list(),
                "province_en": [f"Province {i}" for i in range(table.height)],
                "region": ["Central"] * table.height,
            }
        )
        out = save_province_embedding_plot(projected, dim_geography, tmp_path / "tsne.png")
        assert out.exists()
        assert out.stat().st_size > 0
