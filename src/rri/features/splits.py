"""Time-based train/valid/test split (CLAUDE.md N6) -- never random K-fold.

Every model-facing frame in this project is split the same way: train on
everything up to and including `TRAIN_END_YEAR`, validate on `VALID_YEAR`,
test on `TEST_YEAR`. `assert_time_based_split` re-derives the same boundary
check a second time, independent of however the split was actually produced,
so a future edit that accidentally reintroduces a random split (e.g. a stray
`sample(frac=...)`) fails loudly instead of silently leaking future rows into
training.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

TRAIN_END_YEAR = 2023
VALID_YEAR = 2024
TEST_YEAR = 2025


@dataclass(frozen=True)
class TimeSplit:
    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame


def time_based_split(df: pl.DataFrame, year_col: str = "year") -> TimeSplit:
    """Split `df` by `year_col` into train (<=2023) / valid (2024) / test (2025).

    Rows outside `[.., TEST_YEAR]` (e.g. an incomplete future year) are silently
    excluded from all three splits -- callers that want them as a holdout should
    filter separately before calling this.
    """
    train = df.filter(pl.col(year_col) <= TRAIN_END_YEAR)
    valid = df.filter(pl.col(year_col) == VALID_YEAR)
    test = df.filter(pl.col(year_col) == TEST_YEAR)
    split = TimeSplit(train=train, valid=valid, test=test)
    assert_time_based_split(split, year_col=year_col)
    return split


def assert_time_based_split(split: TimeSplit, year_col: str = "year") -> None:
    """Hard-assert N6: no year overlap and the boundary years are exactly right.

    Raises `AssertionError` (not a silent log) so CI and every caller fail loudly
    on a regression toward random/K-fold splitting.
    """
    if split.train.height:
        max_train_year = int(split.train[year_col].max())  # type: ignore[arg-type]
        assert max_train_year <= TRAIN_END_YEAR, (
            f"train split leaks a year > {TRAIN_END_YEAR}: max={max_train_year}"
        )
    if split.valid.height:
        valid_years = set(split.valid[year_col].unique().to_list())
        assert valid_years <= {VALID_YEAR}, (
            f"valid split must be exactly {VALID_YEAR}: {valid_years}"
        )
    if split.test.height:
        test_years = set(split.test[year_col].unique().to_list())
        assert test_years <= {TEST_YEAR}, f"test split must be exactly {TEST_YEAR}: {test_years}"
