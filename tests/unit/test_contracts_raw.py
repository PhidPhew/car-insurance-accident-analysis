from typing import Any

import pandera.polars as pa
import polars as pl
import pytest

from rri.contracts.raw import (
    ACCIDENTS_RAW_SCHEMA,
    GATE_1_ACCIDENT_COUNT,
    GATE_1_FATALITY_COUNT,
    GATE_2_ACCIDENT_COUNT,
    GATE_2_FATALITY_COUNT,
    TWIN_COLUMN_PAIRS,
    _in_thailand_lat,
    _in_thailand_lon,
    check_gate1_reconciliation,
    check_gate2_reconciliation,
    check_injury_totals,
    check_province_normalization,
    check_source_file_date_formats,
    check_twin_column_pairs,
    run_accidents_raw_contract,
)


def _minimal_accidents_df(**overrides: Any) -> pl.DataFrame:
    """One valid row for every column ACCIDENTS_RAW_SCHEMA / run_accidents_raw_contract needs."""
    base = {
        "ACC_CODE": ["A1"],
        "source_file": ["accident2023.csv"],
        "วันที่เกิดเหตุ": ["4/12/2023"],
        "วันที่รายงาน": ["4/15/2023"],
        "จังหวัด": ["กรุงเทพมหานคร"],
        "LATITUDE": ["13.75"],
        "LONGITUDE": ["100.5"],
        "รถยนต์นั่งส่วนบุคคล": [None],
        "รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ": ["1"],
        "รถบรรทุกไม่เกิน10ล้อ": [None],
        "รถบรรทุกมากกว่า6ล้อไม่เกิน10ล้อ": [None],
        "รถบรรทุกมากกว่า10ล้อ": [None],
        "รถบรรทุกมากกว่า10ล้อ(รถพ่วง)": [None],
        "รถอื่นๆ": [None],
        "อื่นๆ": [None],
        "ผู้บาดเจ็บสาหัส": ["1"],
        "ผู้บาดเจ็บเล็กน้อย": ["2"],
        "รวมจำนวนผู้บาดเจ็บ": ["3"],
        "ผู้เสียชีวิต": ["0"],
    }
    base.update(overrides)
    return pl.DataFrame(base)


class TestGateChecks:
    def test_gate1_passes_when_counts_match(self) -> None:
        # Build a frame with exactly GATE_1_ACCIDENT_COUNT rows summing to
        # GATE_1_FATALITY_COUNT fatalities without materializing 161k rows.
        df = pl.concat(
            [
                pl.DataFrame({"ผู้เสียชีวิต": ["1"] * GATE_1_FATALITY_COUNT}),
                pl.DataFrame({"ผู้เสียชีวิต": ["0"] * (GATE_1_ACCIDENT_COUNT - GATE_1_FATALITY_COUNT)}),
            ]
        )
        results = check_gate1_reconciliation(df)
        assert all(r.passed for r in results)

    def test_gate1_fails_when_counts_mismatch(self) -> None:
        df = pl.DataFrame({"ผู้เสียชีวิต": ["1", "2"]})
        results = check_gate1_reconciliation(df)
        assert not any(r.passed for r in results)
        accident_count = next(r for r in results if r.label == "gate1_accident_count")
        assert accident_count.expected == GATE_1_ACCIDENT_COUNT
        assert accident_count.actual == 2

    def test_gate2_checks_a_separate_deduped_frame(self) -> None:
        df = pl.concat(
            [
                pl.DataFrame({"ผู้เสียชีวิต": ["1"] * GATE_2_FATALITY_COUNT}),
                pl.DataFrame({"ผู้เสียชีวิต": ["0"] * (GATE_2_ACCIDENT_COUNT - GATE_2_FATALITY_COUNT)}),
            ]
        )
        results = check_gate2_reconciliation(df)
        assert all(r.passed for r in results)


class TestTwinColumnPairs:
    def test_all_pairs_pass_when_never_both_populated(self) -> None:
        df = _minimal_accidents_df()
        results = check_twin_column_pairs(df)
        assert len(results) == len(TWIN_COLUMN_PAIRS)
        assert all(r.passed for r in results)

    def test_pair_fails_when_both_populated_on_same_row(self) -> None:
        df = _minimal_accidents_df(
            **{"รถยนต์นั่งส่วนบุคคล": ["1"], "รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ": ["1"]}
        )
        results = check_twin_column_pairs(df)
        broken = next(r for r in results if r.old_column == "รถยนต์นั่งส่วนบุคคล")
        assert not broken.passed
        assert broken.overlap_row_count == 1


class TestSourceFileDateFormats:
    def test_mdy_file_with_no_counter_evidence_passes(self) -> None:
        df = pl.DataFrame(
            {"source_file": ["accident2023.csv"] * 2, "วันที่เกิดเหตุ": ["4/25/2023", "1/1/2023"]}
        )
        results = check_source_file_date_formats(df)
        result_2023 = next(r for r in results if r.source_file == "accident2023.csv")
        assert result_2023.passed

    def test_mdy_file_with_dmy_evidence_fails(self) -> None:
        # 25/4/2023 -- first component > 12, only possible under DMY, which
        # contradicts accident2023.csv's established MDY format.
        df = pl.DataFrame({"source_file": ["accident2023.csv"], "วันที่เกิดเหตุ": ["25/4/2023"]})
        results = check_source_file_date_formats(df)
        result_2023 = next(r for r in results if r.source_file == "accident2023.csv")
        assert not result_2023.passed
        assert result_2023.counter_evidence_count == 1

    def test_excel_serial_file_with_a_slash_date_fails(self) -> None:
        df = pl.DataFrame({"source_file": ["accident2024.csv"], "วันที่เกิดเหตุ": ["4/12/2023"]})
        results = check_source_file_date_formats(df)
        result_2024 = next(r for r in results if r.source_file == "accident2024.csv")
        assert not result_2024.passed

    def test_covers_all_eight_known_files(self) -> None:
        df = pl.DataFrame({"source_file": ["accident2023.csv"], "วันที่เกิดเหตุ": ["4/12/2023"]})
        results = check_source_file_date_formats(df)
        assert len(results) == 8


class TestProvinceNormalization:
    def test_passes_on_known_provinces_only(self) -> None:
        df = pl.DataFrame({"จังหวัด": ["กรุงเทพมหานคร", "เชียงใหม่", None]})
        result = check_province_normalization(df)
        assert result.canonical_province_count == 77
        assert result.orphan_values == ()
        assert result.passed

    def test_fails_and_lists_orphan_values(self) -> None:
        df = pl.DataFrame({"จังหวัด": ["กรุงเทพมหานคร", "Atlantis"]})
        result = check_province_normalization(df)
        assert not result.passed
        assert "Atlantis" in result.orphan_values


class TestInjuryTotals:
    def test_passes_when_serious_plus_minor_equals_total(self) -> None:
        df = pl.DataFrame(
            {
                "ผู้บาดเจ็บสาหัส": ["1", "0"],
                "ผู้บาดเจ็บเล็กน้อย": ["2", "0"],
                "รวมจำนวนผู้บาดเจ็บ": ["3", "0"],
            }
        )
        result = check_injury_totals(df)
        assert result.mismatched_row_count == 0

    def test_flags_mismatched_rows(self) -> None:
        df = pl.DataFrame(
            {
                "ผู้บาดเจ็บสาหัส": ["1", "5"],
                "ผู้บาดเจ็บเล็กน้อย": ["2", "0"],
                "รวมจำนวนผู้บาดเจ็บ": ["3", "999"],
            }
        )
        result = check_injury_totals(df)
        assert result.mismatched_row_count == 1
        assert result.mismatch_rate == pytest.approx(0.5)

    def test_zero_rows_has_zero_rate_not_division_error(self) -> None:
        df = pl.DataFrame(
            {"ผู้บาดเจ็บสาหัส": [], "ผู้บาดเจ็บเล็กน้อย": [], "รวมจำนวนผู้บาดเจ็บ": []},
            schema={
                "ผู้บาดเจ็บสาหัส": pl.Utf8,
                "ผู้บาดเจ็บเล็กน้อย": pl.Utf8,
                "รวมจำนวนผู้บาดเจ็บ": pl.Utf8,
            },
        )
        result = check_injury_totals(df)
        assert result.mismatch_rate == 0.0


class TestCoordinateBoundsHelpers:
    def test_none_is_treated_as_in_bounds(self) -> None:
        assert _in_thailand_lat(None) is True
        assert _in_thailand_lon(None) is True

    def test_non_numeric_string_is_out_of_bounds(self) -> None:
        assert _in_thailand_lat("garbage") is False
        assert _in_thailand_lon("garbage") is False


class TestAccidentsRawSchema:
    def test_valid_row_passes(self) -> None:
        df = _minimal_accidents_df()
        ACCIDENTS_RAW_SCHEMA.validate(df)

    def test_does_not_assert_a_date_format_shape(self) -> None:
        # Deliberately not a slash date, not an Excel serial -- schema only
        # requires present + non-null, so this must still pass.
        df = _minimal_accidents_df(**{"วันที่เกิดเหตุ": ["not-a-normal-date-string"]})
        ACCIDENTS_RAW_SCHEMA.validate(df)

    def test_null_report_date_fails(self) -> None:
        df = _minimal_accidents_df(**{"วันที่รายงาน": [None]})
        with pytest.raises(pa.errors.SchemaErrors):
            ACCIDENTS_RAW_SCHEMA.validate(df, lazy=True)

    def test_out_of_bounds_latitude_fails(self) -> None:
        df = _minimal_accidents_df(**{"LATITUDE": ["99.0"]})
        with pytest.raises(pa.errors.SchemaErrors):
            ACCIDENTS_RAW_SCHEMA.validate(df, lazy=True)

    def test_non_numeric_latitude_and_longitude_fail(self) -> None:
        df = _minimal_accidents_df(**{"LATITUDE": ["not-a-number"], "LONGITUDE": ["also-not"]})
        with pytest.raises(pa.errors.SchemaErrors):
            ACCIDENTS_RAW_SCHEMA.validate(df, lazy=True)

    def test_null_latitude_and_longitude_pass(self) -> None:
        # A single-row all-None list would infer dtype Null, not String --
        # use an explicitly-typed Series so this exercises the "value is
        # None" branch of the bounds check, not a dtype mismatch.
        df = _minimal_accidents_df(
            **{
                "LATITUDE": pl.Series([None], dtype=pl.Utf8),
                "LONGITUDE": pl.Series([None], dtype=pl.Utf8),
            }
        )
        ACCIDENTS_RAW_SCHEMA.validate(df)

    def test_unknown_source_file_fails(self) -> None:
        df = _minimal_accidents_df(**{"source_file": ["accident2099.csv"]})
        with pytest.raises(pa.errors.SchemaErrors):
            ACCIDENTS_RAW_SCHEMA.validate(df, lazy=True)

    def test_extra_unlisted_columns_are_allowed(self) -> None:
        df = _minimal_accidents_df(**{"some_other_raw_column": ["whatever"]})
        ACCIDENTS_RAW_SCHEMA.validate(df)


class TestRunAccidentsRawContract:
    def test_all_clean_row_produces_no_column_schema_failures(self) -> None:
        report = run_accidents_raw_contract(_minimal_accidents_df())
        assert report.column_schema_failure_cases.height == 0
        assert all(r.passed for r in report.twin_columns)
        assert report.province.passed
        assert report.injury_totals.mismatched_row_count == 0

    def test_known_bad_row_surfaces_every_expected_failure(self) -> None:
        df = _minimal_accidents_df(
            **{
                "วันที่รายงาน": [None],
                "LATITUDE": ["99.0"],
                "จังหวัด": ["Atlantis"],
                "ผู้บาดเจ็บสาหัส": ["999"],
            }
        )
        report = run_accidents_raw_contract(df)
        failed_columns = set(report.column_schema_failure_cases["column"].to_list())
        assert {"วันที่รายงาน", "LATITUDE"} <= failed_columns
        assert not report.province.passed
        assert report.injury_totals.mismatched_row_count == 1
        # gate1 is expected to fail too -- this is a 1-row synthetic frame,
        # nowhere near the real 161,307-row count.
        assert not any(g.passed for g in report.gate1)
