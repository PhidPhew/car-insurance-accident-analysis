from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from rri.cleaning.extract import (
    add_extract_metadata,
    coalesce_twin_columns,
    compute_row_hash,
    convert_buddhist_year,
    csv_encoding_for,
    pivot_rainfall_leads,
    read_raw_csv,
    rename_headers,
)


class TestCsvEncodingFor:
    def test_bom_files_use_utf8_sig(self) -> None:
        assert csv_encoding_for("thailand_accidents_2019_2026_combined.csv") == "utf-8-sig"
        assert csv_encoding_for("car_registered_new_2565_2569_combined.csv") == "utf-8-sig"

    def test_no_bom_file_uses_plain_utf8(self) -> None:
        assert csv_encoding_for("stt_car_fuel_at_25690228.csv") == "utf-8"

    def test_unknown_filename_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown raw CSV filename"):
            csv_encoding_for("some_new_file.csv")


class TestReadRawCsv:
    def test_reads_bom_file_correctly(self, tmp_path: Path) -> None:
        path = tmp_path / "thailand_accidents_2019_2026_combined.csv"
        path.write_bytes("﻿a,b\n1,2\n".encode("utf-8-sig"))
        result = read_raw_csv(path)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == ["1"]

    def test_reads_all_columns_as_string(self, tmp_path: Path) -> None:
        path = tmp_path / "stt_car_fuel_at_25690228.csv"
        path.write_text("n\n007\n")
        result = read_raw_csv(path)
        assert result["n"].to_list() == ["007"]  # would be "7" if parsed as int

    def test_placeholder_slash_value_not_coerced_to_null(self, tmp_path: Path) -> None:
        path = tmp_path / "stt_car_fuel_at_25690228.csv"
        path.write_text("date\n//\n")
        result = read_raw_csv(path)
        assert result["date"].to_list() == ["//"]

    def test_known_null_tokens_become_null(self, tmp_path: Path) -> None:
        path = tmp_path / "stt_car_fuel_at_25690228.csv"
        path.write_text("a,b,c,d\n,NA,N/A,-\n")
        result = read_raw_csv(path)
        assert result.row(0) == (None, None, None, None)


class TestComputeRowHash:
    def test_deterministic_across_calls(self) -> None:
        df = pl.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
        first = compute_row_hash(df).to_list()
        second = compute_row_hash(df).to_list()
        assert first == second

    def test_different_rows_get_different_hashes(self) -> None:
        df = pl.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
        hashes = compute_row_hash(df).to_list()
        assert hashes[0] != hashes[1]

    def test_hash_is_hex_string(self) -> None:
        df = pl.DataFrame({"a": ["1"]})
        h = compute_row_hash(df).to_list()[0]
        assert len(h) == 16
        int(h, 16)  # raises ValueError if not valid hex


class TestAddExtractMetadata:
    def test_adds_all_three_columns(self) -> None:
        df = pl.DataFrame({"a": ["1", "2"]})
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        result = add_extract_metadata(df, "some_file.csv", ts)
        assert set(result.columns) >= {"_source_file", "_ingested_at", "_row_hash"}
        assert result["_source_file"].to_list() == ["some_file.csv", "some_file.csv"]
        assert result.height == 2


class TestRenameHeaders:
    def test_renames_present_column(self) -> None:
        df = pl.DataFrame({"Foreast_Month": ["1"], "other": ["x"]})
        result = rename_headers(df, {"Foreast_Month": "Forecast_Month"})
        assert "Forecast_Month" in result.columns
        assert "Foreast_Month" not in result.columns
        assert "other" in result.columns

    def test_absent_mapping_key_is_a_noop(self) -> None:
        df = pl.DataFrame({"a": ["1"]})
        result = rename_headers(df, {"not_present": "renamed"})
        assert result.columns == ["a"]


class TestCoalesceTwinColumns:
    def test_coalesces_zero_overlap_pair_into_old_column_name(self) -> None:
        df = pl.DataFrame(
            {
                "old_a": ["1", None, None],
                "new_a": [None, "2", None],
            }
        )
        result, logs = coalesce_twin_columns(df, pairs=(("old_a", "new_a"),))
        assert "new_a" not in result.columns
        assert result["old_a"].to_list() == ["1", "2", None]
        assert logs[0].rows_from_old == 1
        assert logs[0].rows_from_new == 1
        assert logs[0].rows_still_null == 1
        assert logs[0].total_rows == 3

    def test_raises_on_overlap_rather_than_silently_dropping_new_value(self) -> None:
        df = pl.DataFrame({"old_a": ["1", "2"], "new_a": ["1", None]})
        with pytest.raises(ValueError, match="overlapping rows"):
            coalesce_twin_columns(df, pairs=(("old_a", "new_a"),))

    def test_multiple_pairs_processed_independently(self) -> None:
        df = pl.DataFrame(
            {
                "old_a": ["1", None],
                "new_a": [None, "2"],
                "old_b": [None, "x"],
                "new_b": ["y", None],
            }
        )
        result, logs = coalesce_twin_columns(df, pairs=(("old_a", "new_a"), ("old_b", "new_b")))
        assert result["old_a"].to_list() == ["1", "2"]
        assert result["old_b"].to_list() == ["y", "x"]
        assert len(logs) == 2


class TestConvertBuddhistYear:
    def test_subtracts_543(self) -> None:
        df = pl.DataFrame({"ปี พ.ศ.": ["2565", "2569"]})
        result = convert_buddhist_year(df, "ปี พ.ศ.")
        assert result["year_ce"].to_list() == [2022, 2026]

    def test_original_column_untouched(self) -> None:
        df = pl.DataFrame({"ปี พ.ศ.": ["2565"]})
        result = convert_buddhist_year(df, "ปี พ.ศ.")
        assert result["ปี พ.ศ."].to_list() == ["2565"]


class TestPivotRainfallLeads:
    def test_pivots_six_leads_into_columns_not_dropping_rows(self) -> None:
        df = pl.DataFrame(
            {
                "Initial_Year": ["2021"] * 6,
                "Initial_Month": ["1"] * 6,
                "Forecast_Year": ["2021"] * 6,
                "Forecast_Month": ["1", "2", "3", "4", "5", "6"],
                "Province_ID": ["10"] * 6,
                "Province_NameTH": ["กรุงเทพมหานคร"] * 6,
                "Rainfall": ["5.0", "15.0", "8.0", "180.0", "252.0", "175.0"],
            }
        )
        result = pivot_rainfall_leads(df)
        assert result.height == 1  # one (initial period, province) group
        for i in range(6):
            assert f"rain_lead{i}" in result.columns
        row = result.to_dicts()[0]
        assert row["rain_lead0"] == 5.0
        assert row["rain_lead5"] == 175.0

    def test_handles_year_wraparound_in_lead_calculation(self) -> None:
        # Initial Nov (month 11) forecasting into Jan/Feb of the following year
        # must compute lead via the year difference, not just month difference.
        df = pl.DataFrame(
            {
                "Initial_Year": ["2021", "2021"],
                "Initial_Month": ["11", "11"],
                "Forecast_Year": ["2021", "2022"],
                "Forecast_Month": ["11", "1"],
                "Province_ID": ["10", "10"],
                "Province_NameTH": ["กรุงเทพมหานคร", "กรุงเทพมหานคร"],
                "Rainfall": ["1.0", "2.0"],
            }
        )
        result = pivot_rainfall_leads(df)
        row = result.to_dicts()[0]
        assert row["rain_lead0"] == 1.0
        assert row["rain_lead2"] == 2.0

    def test_multiple_groups_produce_multiple_rows(self) -> None:
        rows = []
        for province in ("10", "11"):
            for month in range(1, 7):
                rows.append(
                    {
                        "Initial_Year": "2021",
                        "Initial_Month": "1",
                        "Forecast_Year": "2021",
                        "Forecast_Month": str(month),
                        "Province_ID": province,
                        "Province_NameTH": f"P{province}",
                        "Rainfall": "1.0",
                    }
                )
        df = pl.DataFrame(rows)
        result = pivot_rainfall_leads(df)
        assert result.height == 2
