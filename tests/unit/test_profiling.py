from pathlib import Path

import polars as pl

from rri.utils.profiling import (
    check_value_when_both_populated,
    check_within_group_agreement,
    classify_date_string,
    classify_time_string,
    compute_dedup_impact,
    detect_delimiter,
    detect_line_ending,
    find_null_correlated_column_groups,
    population_rate_by_group,
    profile_column,
    profile_csv_file,
    profile_date_column_formats,
    profile_duplicates,
    profile_file_metadata,
    profile_json_structure,
    profile_time_column_formats,
)


class TestProfileColumn:
    def test_reports_dtype_and_null_stats(self) -> None:
        df = pl.DataFrame({"x": ["1", "2", None, "4"]})
        result = profile_column(df, "x")
        assert result.null_count == 1
        assert result.null_pct == 25.0
        assert result.distinct_count == 3

    def test_sample_values_excludes_nulls(self) -> None:
        df = pl.DataFrame({"x": [None, "a", "b"]})
        result = profile_column(df, "x")
        assert None not in result.sample_values
        assert set(result.sample_values) == {"a", "b"}

    def test_min_max_only_for_numeric_columns(self) -> None:
        numeric_df = pl.DataFrame({"n": [3, 1, 2]})
        result = profile_column(numeric_df, "n")
        assert result.min_value == "1"
        assert result.max_value == "3"

        string_df = pl.DataFrame({"s": ["c", "a", "b"]})
        string_result = profile_column(string_df, "s")
        assert string_result.min_value is None
        assert string_result.max_value is None

    def test_all_null_column_has_zero_percent_when_empty(self) -> None:
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Utf8)})
        result = profile_column(df, "x")
        assert result.null_pct == 0.0


class TestProfileDuplicates:
    def test_full_row_duplicate_count(self) -> None:
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = profile_duplicates(df)
        assert result.full_row_duplicate_count == 1

    def test_candidate_key_duplicate_detection(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["A1", "A1", "A2", "A3"],
                "year": [2020, 2021, 2020, 2020],
            }
        )
        # Same code reused across different years is NOT a key collision
        # (real key is (code, year)) -- but code alone collides here since
        # "A1" appears with two different years, i.e. this checks the
        # composite key (code, year) has no true duplicate.
        result = profile_duplicates(df, candidate_key_columns=["code", "year"])
        assert result.candidate_key_duplicate_group_count == 0

    def test_candidate_key_duplicate_when_composite_key_repeats(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A1", "A2"], "year": [2020, 2020, 2020]})
        result = profile_duplicates(df, candidate_key_columns=["code", "year"])
        assert result.candidate_key_duplicate_group_count == 1
        assert result.candidate_key_duplicate_row_count == 2

    def test_candidate_key_null_count(self) -> None:
        df = pl.DataFrame({"code": ["A1", None, "A3"]})
        result = profile_duplicates(df, candidate_key_columns=["code"])
        assert result.candidate_key_null_count == 1

    def test_no_candidate_key_returns_zeroed_key_stats(self) -> None:
        df = pl.DataFrame({"a": [1, 1]})
        result = profile_duplicates(df, candidate_key_columns=None)
        assert result.candidate_key_columns is None
        assert result.candidate_key_duplicate_group_count == 0


class TestFindNullCorrelatedColumnGroups:
    def test_identifies_columns_never_populated_on_the_same_row(self) -> None:
        # The real P3 pattern: old_name used pre-2023, new_name from 2023 on
        # -- never both non-null on the same row, each ~50% null here.
        df = pl.DataFrame(
            {
                "old_name": ["a", None, None, "d"],
                "new_name": [None, "b", "c", None],
                "unrelated": ["p", "q", "r", "s"],
            }
        )
        pairs = find_null_correlated_column_groups(df)
        assert len(pairs) == 1
        assert set(pairs[0]) == {"old_name", "new_name"}

    def test_no_pair_when_columns_overlap_in_non_null_rows(self) -> None:
        # Both non-null on row 0 -- not a twin-column pattern.
        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, None]})
        pairs = find_null_correlated_column_groups(df, min_combined_coverage_pct=0)
        assert pairs == []

    def test_fully_populated_columns_are_not_paired(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        pairs = find_null_correlated_column_groups(df)
        assert pairs == []

    def test_unbalanced_null_split_still_detected(self) -> None:
        # The real P3 pattern: one twin populated in most rows, the other
        # in only a few -- individual null percentages need not match.
        df = pl.DataFrame(
            {
                "old_name": ["a", "b", "c", "d", "e", None, None, None, None, None],
                "new_name": [None, None, None, None, None, "f", "g", None, None, None],
            }
        )
        pairs = find_null_correlated_column_groups(df)
        assert len(pairs) == 1
        assert set(pairs[0]) == {"old_name", "new_name"}

    def test_low_combined_coverage_pair_excluded_by_default_threshold(self) -> None:
        # Never overlap, but together cover only 4/20 = 20% of rows -- below
        # the default 50% min_combined_coverage_pct, so likely a
        # coincidence between two unrelated sparse columns, not a real
        # twin-column pattern.
        n = 20
        a: list[str | None] = [None] * n
        b: list[str | None] = [None] * n
        a[0], a[1] = "x", "y"
        b[2], b[3] = "p", "q"
        df = pl.DataFrame({"a": a, "b": b})
        pairs = find_null_correlated_column_groups(df)
        assert pairs == []

    def test_reports_pairs_independently_not_merged_cliques(self) -> None:
        # Three mutually-exclusive columns (e.g. one-hot categorical
        # indicators) should surface as 3 separate pairs, not one group of
        # 3 -- merging would bury genuine old/new-name twins in noise from
        # unrelated mutually-exclusive columns.
        df = pl.DataFrame(
            {
                "a": ["x", None, None, None, None, None],
                "b": [None, "y", None, None, None, None],
                "c": [None, None, "z", None, None, None],
            }
        )
        pairs = find_null_correlated_column_groups(df, min_combined_coverage_pct=0)
        pair_sets = {frozenset(p) for p in pairs}
        assert pair_sets == {
            frozenset({"a", "b"}),
            frozenset({"a", "c"}),
            frozenset({"b", "c"}),
        }


class TestClassifyDateString:
    def test_empty_string(self) -> None:
        assert classify_date_string("") == "empty"
        assert classify_date_string("   ") == "empty"

    def test_ambiguous_slash_date(self) -> None:
        assert classify_date_string("5/5/2019") == "slash_ambiguous"

    def test_unambiguous_slash_date(self) -> None:
        assert classify_date_string("25/3/2019") == "slash_unambiguous"
        assert classify_date_string("3/25/2019") == "slash_unambiguous"

    def test_excel_serial(self) -> None:
        assert classify_date_string("45746") == "excel_serial"

    def test_other(self) -> None:
        assert classify_date_string("not-a-date") == "other"


class TestProfileDateColumnFormats:
    def test_counts_each_category(self) -> None:
        values = ["5/5/2019", "25/3/2019", "45746", "", "garbage"]
        counts = profile_date_column_formats(values)
        assert counts == {
            "slash_ambiguous": 1,
            "slash_unambiguous": 1,
            "excel_serial": 1,
            "empty": 1,
            "other": 1,
        }


class TestClassifyTimeString:
    def test_empty(self) -> None:
        assert classify_time_string("") == "empty"

    def test_clock_string(self) -> None:
        assert classify_time_string("21:10") == "clock_string"
        assert classify_time_string("0:00") == "clock_string"

    def test_excel_fraction(self) -> None:
        assert classify_time_string("0.375") == "excel_fraction"

    def test_value_at_or_above_one_is_other(self) -> None:
        assert classify_time_string("1.5") == "other"

    def test_other(self) -> None:
        assert classify_time_string("garbage") == "other"


class TestProfileTimeColumnFormats:
    def test_counts_each_category(self) -> None:
        values = ["21:10", "0.375", "0.625", ""]
        counts = profile_time_column_formats(values)
        assert counts == {"clock_string": 1, "excel_fraction": 2, "empty": 1}


class TestProfileJsonStructure:
    def test_list_of_records(self) -> None:
        data = [{"a": 1, "b": [1, 2], "c": {"nested": True}}, {"a": 2, "b": [], "c": {}}]
        result = profile_json_structure(data)
        assert result.root_type == "list"
        assert result.record_count == 2
        assert set(result.keys) == {"a", "b", "c"}
        assert result.array_valued_keys == ("b",)
        assert result.nested_object_keys == ("c",)
        assert result.max_depth >= 1

    def test_dict_root(self) -> None:
        data = {"years": [{"year": 2025, "holidays": [1, 2, 3]}]}
        result = profile_json_structure(data)
        assert result.root_type == "dict"
        assert result.record_count is None
        assert "years" in result.keys
        assert result.array_valued_keys == ("years",)

    def test_empty_list(self) -> None:
        result = profile_json_structure([])
        assert result.record_count == 0
        assert result.keys == ()

    def test_scalar_root(self) -> None:
        result = profile_json_structure(42)
        assert result.root_type == "int"
        assert result.record_count is None


class TestFileMetadataAndCsvProfile:
    def test_detects_bom_and_encoding(self, tmp_path: Path) -> None:
        path = tmp_path / "with_bom.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8-sig")

        metadata = profile_file_metadata(path)
        assert metadata.has_bom is True
        assert metadata.encoding == "utf-8-sig"

    def test_no_bom(self, tmp_path: Path) -> None:
        path = tmp_path / "no_bom.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")

        metadata = profile_file_metadata(path)
        assert metadata.has_bom is False
        assert metadata.encoding == "utf-8"

    def test_detect_delimiter_comma(self, tmp_path: Path) -> None:
        path = tmp_path / "comma.csv"
        path.write_text("a,b,c\n1,2,3\n1,2,3\n", encoding="utf-8")
        assert detect_delimiter(path) == ","

    def test_detect_delimiter_semicolon(self, tmp_path: Path) -> None:
        path = tmp_path / "semi.csv"
        path.write_text("a;b;c\n1;2;3\n1;2;3\n", encoding="utf-8")
        assert detect_delimiter(path) == ";"

    def test_detect_line_ending_lf(self, tmp_path: Path) -> None:
        path = tmp_path / "lf.csv"
        path.write_bytes(b"a,b\n1,2\n")
        assert detect_line_ending(path) == "LF"

    def test_detect_line_ending_crlf(self, tmp_path: Path) -> None:
        path = tmp_path / "crlf.csv"
        path.write_bytes(b"a,b\r\n1,2\r\n")
        assert detect_line_ending(path) == "CRLF"

    def test_profile_csv_file_end_to_end(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.csv"
        path.write_text(
            "code,year,value\nA1,2020,10\nA1,2020,10\nA2,2021,20\n",
            encoding="utf-8-sig",
        )

        result = profile_csv_file(path, candidate_key_columns=["code", "year"])

        assert result.row_count == 3
        assert result.column_count == 3
        assert result.metadata.has_bom is True
        assert result.duplicates.full_row_duplicate_count == 1
        column_names = {c.name for c in result.columns}
        assert column_names == {"code", "year", "value"}


class TestCheckWithinGroupAgreement:
    def test_all_groups_agree(self) -> None:
        df = pl.DataFrame({"key": ["A", "A", "B", "B"], "agency": ["x", "x", "y", "y"]})
        result = check_within_group_agreement(df, ["key"], "agency")
        assert result.groups_checked == 2
        assert result.agree_count == 2
        assert result.conflict_count == 0
        assert result.agree_pct == 100.0

    def test_conflicting_group_detected(self) -> None:
        df = pl.DataFrame({"key": ["A", "A", "B", "B"], "agency": ["x", "z", "y", "y"]})
        result = check_within_group_agreement(df, ["key"], "agency")
        assert result.agree_count == 1
        assert result.conflict_count == 1

    def test_null_vs_populated_counts_as_conflict(self) -> None:
        df = pl.DataFrame({"key": ["A", "A"], "note": [None, "text"]})
        result = check_within_group_agreement(df, ["key"], "note")
        assert result.conflict_count == 1
        assert result.agree_count == 0

    def test_single_row_groups_are_not_counted(self) -> None:
        df = pl.DataFrame({"key": ["A", "B", "C"], "agency": ["x", "y", "z"]})
        result = check_within_group_agreement(df, ["key"], "agency")
        assert result.groups_checked == 0

    def test_null_group_keys_excluded_not_lumped_into_one_group(self) -> None:
        # Regression case: many unrelated rows sharing a null key must not
        # be treated as one giant duplicate group by polars' group_by.
        df = pl.DataFrame(
            {
                "key": ["A", "A", None, None, None],
                "agency": ["x", "x", "p", "q", "r"],
            }
        )
        result = check_within_group_agreement(df, ["key"], "agency")
        assert result.groups_checked == 1  # only the real "A" group
        assert result.agree_count == 1
        assert result.conflict_count == 0

    def test_agree_pct_zero_when_no_groups_checked(self) -> None:
        df = pl.DataFrame({"key": ["A"], "agency": ["x"]})
        result = check_within_group_agreement(df, ["key"], "agency")
        assert result.agree_pct == 0.0


class TestComputeDedupImpact:
    def test_collapses_duplicate_groups_to_one_row(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["A1", "A1", "A2"],
                "fatalities": [2, 2, 0],
            }
        )
        impact = compute_dedup_impact(df, "code", ["fatalities"])
        assert impact.rows_before == 3
        assert impact.rows_after == 2
        assert impact.sums_before["fatalities"] == 4.0
        assert impact.sums_after["fatalities"] == 2.0

    def test_null_key_rows_pass_through_unchanged(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["A1", "A1", None, None],
                "fatalities": [1, 1, 5, 5],
            }
        )
        impact = compute_dedup_impact(df, "code", ["fatalities"])
        # both null-key rows survive (nothing to group them by), only the
        # true duplicate pair (A1, A1) collapses to 1
        assert impact.rows_after == 3
        assert impact.sums_after["fatalities"] == 11.0  # 1 + 5 + 5

    def test_no_duplicates_leaves_totals_unchanged(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A2", "A3"], "fatalities": [1, 2, 3]})
        impact = compute_dedup_impact(df, "code", ["fatalities"])
        assert impact.rows_before == impact.rows_after == 3
        assert impact.sums_before == impact.sums_after == {"fatalities": 6.0}


class TestCheckValueWhenBothPopulated:
    def test_matches_hypothesis_when_expected_value_dominates(self) -> None:
        df = pl.DataFrame(
            {
                "primary": ["other", "other", "other"],
                "secondary": ["detail1", "detail2", "detail3"],
            }
        )
        result = check_value_when_both_populated(df, "primary", "secondary", "other")
        assert result.rows_both_populated == 3
        assert result.matching_value_count == 3
        assert result.matching_pct == 100.0

    def test_refutes_hypothesis_when_expected_value_absent(self) -> None:
        df = pl.DataFrame(
            {
                "primary": ["straight_road", "curved_road"],
                "secondary": ["flat", "hilly"],
            }
        )
        result = check_value_when_both_populated(df, "primary", "secondary", "other")
        assert result.matching_value_count == 0
        assert result.matching_pct == 0.0

    def test_ignores_rows_where_either_is_null(self) -> None:
        df = pl.DataFrame(
            {
                "primary": ["other", None, "other"],
                "secondary": ["x", "y", None],
            }
        )
        result = check_value_when_both_populated(df, "primary", "secondary", "other")
        assert result.rows_both_populated == 1

    def test_zero_rows_both_populated_has_zero_pct_not_division_error(self) -> None:
        df = pl.DataFrame({"primary": [None], "secondary": [None]})
        result = check_value_when_both_populated(df, "primary", "secondary", "other")
        assert result.matching_pct == 0.0


class TestPopulationRateByGroup:
    def test_reports_total_and_populated_counts_per_group(self) -> None:
        df = pl.DataFrame(
            {
                "year": ["2025", "2025", "2026", "2026", "2026"],
                "value": ["a", None, "b", None, None],
            }
        )
        result = population_rate_by_group(df, "year", "value")
        by_year = {row["year"]: row for row in result.to_dicts()}
        assert by_year["2025"]["total_rows"] == 2
        assert by_year["2025"]["populated_rows"] == 1
        assert by_year["2026"]["total_rows"] == 3
        assert by_year["2026"]["populated_rows"] == 1

    def test_group_with_zero_population_reports_zero(self) -> None:
        df = pl.DataFrame({"year": ["2019", "2019"], "value": [None, None]})
        result = population_rate_by_group(df, "year", "value")
        assert result["populated_rows"].to_list() == [0]
