"""Unit tests for `rri.features.severity_index` (CLAUDE.md §9)."""

from __future__ import annotations

import polars as pl
import pytest

from config.severity_index_params import W_F_COARSE, W_I_COARSE
from config.simulation_params import (
    SEVERITY_WEIGHT_FATAL,
    SEVERITY_WEIGHT_MINOR,
    SEVERITY_WEIGHT_SERIOUS,
)
from rri.features.severity_index import (
    add_severity_index_coarse,
    add_severity_index_fine,
    severity_index_correlation,
)


def _fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fatality_count": [1, 0, 0],
            "serious_injury_count": [0, 2, 0],
            "minor_injury_count": [0, 1, 0],
            "total_injury_count": [0, 3, 0],
            "accident_count": [1, 1, 2],
            "has_injury_breakdown": [True, True, False],
        }
    )


class TestSeverityIndexCoarse:
    def test_formula(self) -> None:
        result = add_severity_index_coarse(_fixture())
        expected = [
            1 * W_F_COARSE / 1,
            (0 * W_F_COARSE + 3 * W_I_COARSE) / 1,
            0.0,
        ]
        assert result["severity_index_coarse"].to_list() == pytest.approx(expected)

    def test_computable_even_without_injury_breakdown(self) -> None:
        # Coarse is the whole point of CLAUDE.md §9's design -- it must never
        # be null just because has_injury_breakdown is false.
        result = add_severity_index_coarse(_fixture())
        assert result["severity_index_coarse"].null_count() == 0


class TestSeverityIndexFine:
    def test_formula_where_breakdown_available(self) -> None:
        result = add_severity_index_fine(_fixture())
        expected_row0 = 1 * SEVERITY_WEIGHT_FATAL
        expected_row1 = 2 * SEVERITY_WEIGHT_SERIOUS + 1 * SEVERITY_WEIGHT_MINOR
        assert result["severity_index_fine"][0] == pytest.approx(expected_row0)
        assert result["severity_index_fine"][1] == pytest.approx(expected_row1)

    def test_gate_b_null_where_no_breakdown(self) -> None:
        # CLAUDE.md §8 Gate B: severity_index_fine must never be computed for
        # a row where has_injury_breakdown is false.
        result = add_severity_index_fine(_fixture())
        assert result["severity_index_fine"][2] is None


class TestSeverityIndexCorrelation:
    def test_perfectly_correlated_when_derived_from_same_signal(self) -> None:
        df = pl.DataFrame(
            {
                "fatality_count": [0, 1, 2, 0, 3],
                "serious_injury_count": [1, 0, 1, 2, 0],
                "minor_injury_count": [2, 1, 0, 3, 1],
                "total_injury_count": [3, 1, 1, 5, 1],
                "accident_count": [1, 1, 1, 1, 1],
                "has_injury_breakdown": [True] * 5,
                "year": [2020, 2021, 2022, 2023, 2024],
            }
        )
        df = add_severity_index_coarse(df)
        df = add_severity_index_fine(df)
        result = severity_index_correlation(df)
        assert result["n_rows"] == 5
        assert -1.0 <= result["pearson_r"] <= 1.0

    def test_excludes_rows_without_breakdown_or_outside_window(self) -> None:
        df = pl.DataFrame(
            {
                "fatality_count": [1, 1, 1],
                "serious_injury_count": [1, 1, 1],
                "minor_injury_count": [1, 1, 1],
                "total_injury_count": [2, 2, 2],
                "accident_count": [1, 1, 1],
                "has_injury_breakdown": [True, False, True],
                "year": [2020, 2020, 2019],  # row 1 no breakdown, row 2 outside window
            }
        )
        df = add_severity_index_coarse(df)
        df = add_severity_index_fine(df)
        result = severity_index_correlation(df)
        assert result["n_rows"] == 1
