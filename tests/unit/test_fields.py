from datetime import time

import polars as pl

from rri.cleaning.fields import (
    add_injury_breakdown_flag,
    clean_coordinates,
    convert_times,
    parse_time_value,
)


class TestCleanCoordinates:
    def test_in_bounds_coordinates_are_kept_and_valid(self) -> None:
        df = pl.DataFrame({"LATITUDE": ["13.75"], "LONGITUDE": ["100.5"]})
        result, coord_log = clean_coordinates(df)
        assert result["LATITUDE"].to_list() == [13.75]
        assert result["LONGITUDE"].to_list() == [100.5]
        assert result["is_geo_valid"].to_list() == [True]
        assert coord_log.rows_lat_out_of_bounds == 0
        assert coord_log.rows_lon_out_of_bounds == 0
        assert coord_log.rows_is_geo_valid_false == 0

    def test_out_of_bounds_latitude_is_nulled(self) -> None:
        df = pl.DataFrame({"LATITUDE": ["99.0"], "LONGITUDE": ["100.5"]})
        result, coord_log = clean_coordinates(df)
        assert result["LATITUDE"].to_list() == [None]
        assert result["LONGITUDE"].to_list() == [100.5]
        assert result["is_geo_valid"].to_list() == [False]
        assert coord_log.rows_lat_out_of_bounds == 1
        assert coord_log.rows_lon_out_of_bounds == 0

    def test_out_of_bounds_longitude_is_nulled(self) -> None:
        df = pl.DataFrame({"LATITUDE": ["13.75"], "LONGITUDE": ["0.5"]})
        result, coord_log = clean_coordinates(df)
        assert result["LONGITUDE"].to_list() == [None]
        assert result["is_geo_valid"].to_list() == [False]
        assert coord_log.rows_lon_out_of_bounds == 1

    def test_already_null_coordinate_stays_null_and_invalid(self) -> None:
        df = pl.DataFrame({"LATITUDE": [None], "LONGITUDE": ["100.5"]})
        result, coord_log = clean_coordinates(df)
        assert result["LATITUDE"].to_list() == [None]
        assert result["is_geo_valid"].to_list() == [False]
        assert coord_log.rows_lat_out_of_bounds == 0  # null, not out-of-bounds
        assert coord_log.rows_is_geo_valid_false == 1

    def test_boundary_values_are_valid(self) -> None:
        df = pl.DataFrame({"LATITUDE": ["5.0", "21.0"], "LONGITUDE": ["97.0", "106.0"]})
        result, coord_log = clean_coordinates(df)
        assert result["is_geo_valid"].to_list() == [True, True]
        assert coord_log.rows_is_geo_valid_false == 0

    def test_custom_bounds_and_column_names(self) -> None:
        df = pl.DataFrame({"lat": ["50.0"], "lon": ["10.0"]})
        result, coord_log = clean_coordinates(
            df,
            lat_column="lat",
            lon_column="lon",
            lat_bounds=(0.0, 100.0),
            lon_bounds=(0.0, 20.0),
        )
        assert result["is_geo_valid"].to_list() == [True]
        assert coord_log.rows_total == 1


class TestParseTimeValue:
    def test_none_returns_none(self) -> None:
        assert parse_time_value(None) is None

    def test_clock_string(self) -> None:
        assert parse_time_value("21:10") == time(21, 10)

    def test_clock_string_single_digit_hour(self) -> None:
        assert parse_time_value("9:05") == time(9, 5)

    def test_excel_fraction_09_00(self) -> None:
        assert parse_time_value("0.375") == time(9, 0)

    def test_excel_fraction_15_00(self) -> None:
        assert parse_time_value("0.625") == time(15, 0)

    def test_literal_midnight(self) -> None:
        assert parse_time_value("0:00") == time(0, 0)

    def test_unparseable_returns_none(self) -> None:
        assert parse_time_value("garbage") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_time_value("") is None

    def test_out_of_range_hour_returns_none(self) -> None:
        assert parse_time_value("25:10") is None

    def test_fraction_of_one_returns_none(self) -> None:
        assert parse_time_value("1.0") is None

    def test_fraction_rounding_to_hour_24_returns_none(self) -> None:
        assert parse_time_value("0.9999999") is None


class TestConvertTimes:
    def test_parses_clock_and_excel_fraction_columns(self) -> None:
        df = pl.DataFrame({"เวลา": ["21:10", "0.375"]})
        result, time_log = convert_times(df)
        assert result["accident_time"].to_list() == [time(21, 10), time(9, 0)]
        assert time_log.rows_parsed == 2
        assert time_log.rows_unparseable == 0

    def test_unparseable_value_is_null_and_counted(self) -> None:
        df = pl.DataFrame({"เวลา": ["garbage"]})
        result, time_log = convert_times(df)
        assert result["accident_time"].to_list() == [None]
        assert time_log.rows_unparseable == 1

    def test_literal_midnight_flag_set_only_for_exact_string(self) -> None:
        df = pl.DataFrame({"เวลา": ["0:00", "00:00", "9:00"]})
        result, time_log = convert_times(df)
        assert result["time_is_literal_midnight"].to_list() == [True, False, False]
        assert time_log.rows_literal_midnight == 1

    def test_null_raw_value_is_not_flagged_as_midnight(self) -> None:
        df = pl.DataFrame({"เวลา": [None]})
        result, time_log = convert_times(df)
        assert result["time_is_literal_midnight"].to_list() == [False]
        assert result["accident_time"].to_list() == [None]
        assert time_log.rows_literal_midnight == 0


class TestAddInjuryBreakdownFlag:
    def test_2019_rows_flagged_false(self) -> None:
        df = pl.DataFrame({"source_file": ["accident2019.csv", "accident2020.csv"]})
        result, breakdown_log = add_injury_breakdown_flag(df)
        assert result["has_injury_breakdown"].to_list() == [False, True]
        assert breakdown_log.rows_without_breakdown == 1
        assert breakdown_log.rows_total == 2

    def test_no_2019_rows_all_true(self) -> None:
        df = pl.DataFrame({"source_file": ["accident2020.csv", "accident2021.csv"]})
        result, breakdown_log = add_injury_breakdown_flag(df)
        assert result["has_injury_breakdown"].to_list() == [True, True]
        assert breakdown_log.rows_without_breakdown == 0

    def test_custom_column_and_source_file_name(self) -> None:
        df = pl.DataFrame({"src": ["special.csv", "other.csv"]})
        result, breakdown_log = add_injury_breakdown_flag(
            df,
            source_file_column="src",
            no_breakdown_source_file="special.csv",
            new_column="has_breakdown",
        )
        assert result["has_breakdown"].to_list() == [False, True]
        assert breakdown_log.rows_without_breakdown == 1
