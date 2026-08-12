import random

from scripts.make_fixtures import (
    find_out_of_bounds_coord_indices,
    first_index_where,
    is_ambiguous_dmy,
    is_excel_fraction_time,
    to_float,
    top_up_random,
)


class TestToFloat:
    def test_parses_valid_number(self) -> None:
        assert to_float("13.5") == 13.5

    def test_blank_string_is_none(self) -> None:
        assert to_float("") is None
        assert to_float("   ") is None

    def test_none_input_is_none(self) -> None:
        assert to_float(None) is None

    def test_non_numeric_is_none(self) -> None:
        assert to_float("abc") is None


class TestIsAmbiguousDmy:
    def test_both_components_le_12_is_ambiguous(self) -> None:
        assert is_ambiguous_dmy("5/5/2019") is True

    def test_day_over_12_is_unambiguous(self) -> None:
        assert is_ambiguous_dmy("25/3/2019") is False

    def test_month_over_12_is_unambiguous(self) -> None:
        assert is_ambiguous_dmy("3/25/2019") is False

    def test_excel_serial_is_not_ambiguous(self) -> None:
        assert is_ambiguous_dmy("45746") is False

    def test_empty_string_is_not_ambiguous(self) -> None:
        assert is_ambiguous_dmy("") is False


class TestIsExcelFractionTime:
    def test_fraction_below_one_is_excel_time(self) -> None:
        assert is_excel_fraction_time("0.375") is True
        assert is_excel_fraction_time("0.625") is True

    def test_clock_string_is_not_excel_time(self) -> None:
        assert is_excel_fraction_time("21:10") is False

    def test_midnight_placeholder_is_not_excel_fraction(self) -> None:
        assert is_excel_fraction_time("0:00") is False

    def test_value_at_or_above_one_is_not_excel_time(self) -> None:
        assert is_excel_fraction_time("1.0") is False

    def test_empty_string_is_not_excel_time(self) -> None:
        assert is_excel_fraction_time("") is False


class TestFindOutOfBoundsCoordIndices:
    def test_flags_lat_and_lon_outside_thailand_bounds(self) -> None:
        rows = [
            {"LATITUDE": "13.75", "LONGITUDE": "100.5"},  # valid (Bangkok-ish)
            {"LATITUDE": "0.0", "LONGITUDE": "100.5"},  # lat out of bounds
            {"LATITUDE": "13.75", "LONGITUDE": "200.0"},  # lon out of bounds
            {"LATITUDE": "", "LONGITUDE": ""},  # blank, not flagged
        ]

        assert find_out_of_bounds_coord_indices(rows) == {1, 2}


class TestFirstIndexWhere:
    def test_returns_first_matching_index(self) -> None:
        rows = [{"x": "a"}, {"x": "b"}, {"x": "b"}]
        assert first_index_where(rows, lambda r: r["x"] == "b") == 1

    def test_returns_none_when_no_match(self) -> None:
        rows = [{"x": "a"}]
        assert first_index_where(rows, lambda r: r["x"] == "z") is None


class TestTopUpRandom:
    def test_fills_up_to_target_deterministically(self) -> None:
        rows = [{"i": i} for i in range(20)]
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result1 = top_up_random(rows, {0, 1}, 10, rng1)
        result2 = top_up_random(rows, {0, 1}, 10, rng2)

        assert len(result1) == 10
        assert {0, 1} <= result1
        assert result1 == result2  # same seed -> identical selection

    def test_does_not_exceed_available_rows(self) -> None:
        rows = [{"i": i} for i in range(3)]
        rng = random.Random(42)

        result = top_up_random(rows, set(), 100, rng)

        assert result == {0, 1, 2}

    def test_never_shrinks_existing_selection(self) -> None:
        rows = [{"i": i} for i in range(5)]
        rng = random.Random(42)

        result = top_up_random(rows, {0, 1, 2, 3, 4}, 2, rng)

        assert result == {0, 1, 2, 3, 4}
