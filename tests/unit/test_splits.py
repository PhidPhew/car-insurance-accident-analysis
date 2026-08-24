"""Unit tests for `rri.features.splits` (CLAUDE.md N6 -- time-based split only)."""

from __future__ import annotations

import polars as pl
import pytest

from rri.features.splits import (
    TEST_YEAR,
    TRAIN_END_YEAR,
    VALID_YEAR,
    TimeSplit,
    assert_time_based_split,
    time_based_split,
)


def _fixture() -> pl.DataFrame:
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    return pl.DataFrame({"year": years, "value": list(range(7))})


class TestTimeBasedSplit:
    def test_partitions_by_year_boundary(self) -> None:
        split = time_based_split(_fixture())
        assert split.train["year"].to_list() == [2020, 2021, 2022, 2023]
        assert split.valid["year"].to_list() == [2024]
        assert split.test["year"].to_list() == [2025]

    def test_year_after_test_is_excluded_not_leaked(self) -> None:
        split = time_based_split(_fixture())
        all_years = (
            split.train["year"].to_list()
            + split.valid["year"].to_list()
            + split.test["year"].to_list()
        )
        assert 2026 not in all_years

    def test_boundaries_match_module_constants(self) -> None:
        assert TRAIN_END_YEAR == 2023
        assert VALID_YEAR == 2024
        assert TEST_YEAR == 2025


class TestAssertTimeBasedSplit:
    def test_valid_split_does_not_raise(self) -> None:
        split = time_based_split(_fixture())
        assert_time_based_split(split)  # no raise

    def test_raises_on_train_leaking_a_future_year(self) -> None:
        bad = TimeSplit(
            train=pl.DataFrame({"year": [2023, 2024]}),
            valid=pl.DataFrame({"year": [2024]}),
            test=pl.DataFrame({"year": [2025]}),
        )
        with pytest.raises(AssertionError, match="leaks a year"):
            assert_time_based_split(bad)

    def test_raises_on_valid_containing_a_non_2024_year(self) -> None:
        bad = TimeSplit(
            train=pl.DataFrame({"year": [2022, 2023]}),
            valid=pl.DataFrame({"year": [2023, 2024]}),
            test=pl.DataFrame({"year": [2025]}),
        )
        with pytest.raises(AssertionError, match="valid split must be exactly"):
            assert_time_based_split(bad)

    def test_raises_on_random_kfold_style_split(self) -> None:
        # Simulates the regression this guard rail exists to catch: a random
        # shuffle that scatters years across all three splits.
        bad = TimeSplit(
            train=pl.DataFrame({"year": [2020, 2024, 2025]}),
            valid=pl.DataFrame({"year": [2021, 2024]}),
            test=pl.DataFrame({"year": [2022, 2025]}),
        )
        with pytest.raises(AssertionError):
            assert_time_based_split(bad)
