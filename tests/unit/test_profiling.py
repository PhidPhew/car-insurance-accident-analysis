from datetime import date
from pathlib import Path

import polars as pl
import pytest

from rri.utils.profiling import (
    check_duplicate_date_string_consistency,
    check_value_when_both_populated,
    check_within_group_agreement,
    classify_date_string,
    classify_slash_date_side_evidence,
    classify_time_string,
    compute_dedup_impact,
    compute_report_lag_days,
    count_resolved_by_normalization,
    count_rows_over_lag_threshold,
    detect_delimiter,
    detect_line_ending,
    discriminate_transposed_date_pairs,
    find_conflicting_groups,
    find_null_correlated_column_groups,
    find_transposed_date_string_pairs,
    normalize_for_synonym_check,
    parse_accident_date,
    parse_excel_serial_date,
    parse_slash_date,
    population_rate_by_group,
    profile_column,
    profile_csv_file,
    profile_date_column_formats,
    profile_duplicates,
    profile_file_metadata,
    profile_json_structure,
    profile_slash_date_side_evidence,
    profile_time_column_formats,
    report_lag_quantiles_by_group,
    sample_deterministic,
    summarize_report_lag,
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


class TestClassifySlashDateSideEvidence:
    def test_mdy_evidence_when_second_component_over_12(self) -> None:
        assert classify_slash_date_side_evidence("4/25/2023") == "mdy_evidence"

    def test_dmy_evidence_when_first_component_over_12(self) -> None:
        assert classify_slash_date_side_evidence("25/4/2023") == "dmy_evidence"

    def test_both_over_12(self) -> None:
        assert classify_slash_date_side_evidence("13/14/2023") == "both_over_12"

    def test_ambiguous_when_neither_over_12(self) -> None:
        assert classify_slash_date_side_evidence("4/12/2023") == "ambiguous"

    def test_non_slash_string_returns_none(self) -> None:
        assert classify_slash_date_side_evidence("45746") is None
        assert classify_slash_date_side_evidence("") is None


class TestProfileSlashDateSideEvidence:
    def test_counts_each_category_and_skips_non_slash_values(self) -> None:
        values = ["4/25/2023", "25/4/2023", "13/14/2023", "4/12/2023", "45746", ""]
        counts = profile_slash_date_side_evidence(values)
        assert counts == {
            "mdy_evidence": 1,
            "dmy_evidence": 1,
            "both_over_12": 1,
            "ambiguous": 1,
        }

    def test_zero_dmy_evidence_across_whole_file_is_the_decisive_signal(self) -> None:
        # Mirrors the real accident2023.csv finding: many mdy_evidence rows,
        # zero dmy_evidence rows anywhere in the file -- not just a majority.
        values = ["4/25/2023"] * 100 + ["4/12/2023"] * 50
        counts = profile_slash_date_side_evidence(values)
        assert counts.get("dmy_evidence", 0) == 0
        assert counts["mdy_evidence"] == 100


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


class TestFindConflictingGroups:
    def test_returns_key_and_sorted_distinct_values_for_conflicting_groups(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["A1", "A1", "A2", "A2"],
                "location": ["straight road", "curved road", "flat", "flat"],
            }
        )
        result = find_conflicting_groups(df, ["code"], "location")
        assert len(result) == 1
        key, values = result[0]
        assert key == ("A1",)
        assert values == ["curved road", "straight road"]

    def test_agreeing_groups_excluded(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A1"], "location": ["flat", "flat"]})
        assert find_conflicting_groups(df, ["code"], "location") == []

    def test_null_group_keys_excluded(self) -> None:
        df = pl.DataFrame({"code": [None, None], "location": ["x", "y"]})
        assert find_conflicting_groups(df, ["code"], "location") == []

    def test_null_values_within_group_ignored_not_treated_as_conflict(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A1"], "location": ["flat", None]})
        assert find_conflicting_groups(df, ["code"], "location") == []

    def test_single_row_groups_excluded(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A2"], "location": ["x", "y"]})
        assert find_conflicting_groups(df, ["code"], "location") == []


class TestSampleDeterministic:
    def test_same_seed_gives_same_sample(self) -> None:
        items = list(range(100))
        assert sample_deterministic(items, 10, seed=42) == sample_deterministic(items, 10, seed=42)

    def test_different_seed_can_give_different_sample(self) -> None:
        items = list(range(100))
        a = sample_deterministic(items, 10, seed=1)
        b = sample_deterministic(items, 10, seed=2)
        assert a != b

    def test_n_larger_than_items_returns_all(self) -> None:
        items = [1, 2, 3]
        assert sorted(sample_deterministic(items, 10, seed=42)) == [1, 2, 3]


class TestNormalizeForSynonymCheck:
    def test_collapses_no_slope_and_flat_synonym(self) -> None:
        assert normalize_for_synonym_check("ทางตรง+ไม่มีความลาดชัน") == normalize_for_synonym_check(
            "ทางตรง+ที่ราบ"
        )

    def test_strips_elaboration_suffix_after_slash(self) -> None:
        assert normalize_for_synonym_check(
            "ทางตรง+ไม่มีความลาดชัน/ที่ราบ"
        ) == normalize_for_synonym_check("ทางตรง+ไม่มีความลาดชัน")

    def test_removes_internal_whitespace(self) -> None:
        assert normalize_for_synonym_check("ทางสามแยก (Y)") == normalize_for_synonym_check(
            "ทางสามแยก(Y)"
        )

    def test_genuinely_different_values_stay_different(self) -> None:
        assert normalize_for_synonym_check("ทางตรง") != normalize_for_synonym_check("ทางโค้ง")


class TestCountResolvedByNormalization:
    def test_synonym_pair_counts_as_resolved(self) -> None:
        conflicts: list[tuple[tuple[str, ...], list[str]]] = [
            (("A1",), ["ทางตรง+ไม่มีความลาดชัน", "ทางตรง+ที่ราบ"])
        ]
        resolved, still = count_resolved_by_normalization(conflicts)
        assert resolved == 1
        assert still == 0

    def test_genuinely_different_pair_stays_conflicting(self) -> None:
        conflicts: list[tuple[tuple[str, ...], list[str]]] = [(("A1",), ["ทางตรง", "ทางโค้ง"])]
        resolved, still = count_resolved_by_normalization(conflicts)
        assert resolved == 0
        assert still == 1

    def test_mixed_set(self) -> None:
        conflicts: list[tuple[tuple[str, ...], list[str]]] = [
            (("A1",), ["ทางตรง+ไม่มีความลาดชัน", "ทางตรง+ที่ราบ"]),
            (("A2",), ["ทางตรง", "ทางโค้ง"]),
        ]
        resolved, still = count_resolved_by_normalization(conflicts)
        assert resolved == 1
        assert still == 1


class TestParseSlashDate:
    def test_month_first(self) -> None:
        assert parse_slash_date("4/12/2023", month_first=True) == date(2023, 4, 12)

    def test_day_first(self) -> None:
        assert parse_slash_date("4/12/2023", month_first=False) == date(2023, 12, 4)

    def test_two_digit_year_expands_to_2000s(self) -> None:
        assert parse_slash_date("4/12/23", month_first=True) == date(2023, 4, 12)

    def test_invalid_calendar_date_returns_none(self) -> None:
        # month_first=False reads this as month=40, day=13 -- not a real month
        assert parse_slash_date("13/40/2023", month_first=False) is None

    def test_non_slash_string_returns_none(self) -> None:
        assert parse_slash_date("45746", month_first=True) is None


class TestCheckDuplicateDateStringConsistency:
    def test_identical_strings_parse_identically(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["A1", "A1"],
                "raw": ["4/12/2023", "4/12/2023"],
                "parsed": [date(2023, 4, 12), date(2023, 4, 12)],
            }
        )
        result = check_duplicate_date_string_consistency(df, "code", "raw", "parsed")
        assert result.groups_checked == 1
        assert result.mismatched_groups == 0

    def test_identical_strings_parsing_to_different_dates_is_flagged(self) -> None:
        # Simulates a parser/format-detection bug: same raw string, two parsed dates.
        df = pl.DataFrame(
            {
                "code": ["A1", "A1"],
                "raw": ["4/12/2023", "4/12/2023"],
                "parsed": [date(2023, 4, 12), date(2023, 12, 4)],
            }
        )
        result = check_duplicate_date_string_consistency(df, "code", "raw", "parsed")
        assert result.mismatched_groups == 1
        assert result.mismatched_group_keys == ["A1"]

    def test_transposed_pair_group_excluded_not_flagged(self) -> None:
        # This is the case the user's correction was about: a duplicate group
        # whose two rows have DIFFERENT (transposed) raw strings must not be
        # counted here at all -- it belongs in the transposed-pair report.
        df = pl.DataFrame(
            {
                "code": ["A1", "A1"],
                "raw": ["4/12/2023", "12/4/2023"],
                "parsed": [date(2023, 4, 12), date(2023, 12, 4)],
            }
        )
        result = check_duplicate_date_string_consistency(df, "code", "raw", "parsed")
        assert result.groups_checked == 0
        assert result.mismatched_groups == 0

    def test_null_group_key_excluded(self) -> None:
        df = pl.DataFrame(
            {
                "code": [None, None],
                "raw": ["4/12/2023", "4/12/2023"],
                "parsed": [date(2023, 4, 12), date(2023, 4, 12)],
            }
        )
        result = check_duplicate_date_string_consistency(df, "code", "raw", "parsed")
        assert result.groups_checked == 0

    def test_single_row_group_excluded(self) -> None:
        df = pl.DataFrame({"code": ["A1"], "raw": ["4/12/2023"], "parsed": [date(2023, 4, 12)]})
        result = check_duplicate_date_string_consistency(df, "code", "raw", "parsed")
        assert result.groups_checked == 0


class TestFindTransposedDateStringPairs:
    def test_transposed_pair_detected(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A1"], "date": ["4/12/2023", "12/4/2023"]})
        assert find_transposed_date_string_pairs(df, "code", "date") == ["A1"]

    def test_non_transposed_pair_excluded(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A1"], "date": ["4/12/2023", "4/13/2023"]})
        assert find_transposed_date_string_pairs(df, "code", "date") == []

    def test_three_distinct_raw_dates_excluded(self) -> None:
        df = pl.DataFrame(
            {"code": ["A1", "A1", "A1"], "date": ["4/12/2023", "12/4/2023", "5/1/2023"]}
        )
        assert find_transposed_date_string_pairs(df, "code", "date") == []

    def test_identical_strings_excluded(self) -> None:
        df = pl.DataFrame({"code": ["A1", "A1"], "date": ["4/12/2023", "4/12/2023"]})
        assert find_transposed_date_string_pairs(df, "code", "date") == []

    def test_single_row_group_excluded(self) -> None:
        df = pl.DataFrame({"code": ["A1"], "date": ["4/12/2023"]})
        assert find_transposed_date_string_pairs(df, "code", "date") == []

    def test_null_group_key_excluded(self) -> None:
        df = pl.DataFrame({"code": [None, None], "date": ["4/12/2023", "12/4/2023"]})
        assert find_transposed_date_string_pairs(df, "code", "date") == []

    def test_three_row_group_with_two_distinct_transposed_dates_detected(self) -> None:
        # Mirrors the real 23 three-row groups: one candidate date repeated.
        df = pl.DataFrame(
            {
                "code": ["A1", "A1", "A1"],
                "date": ["4/12/2023", "4/12/2023", "12/4/2023"],
            }
        )
        assert find_transposed_date_string_pairs(df, "code", "date") == ["A1"]

    def test_non_slash_raw_value_excluded_not_crashed(self) -> None:
        # A group with 2 distinct raw values where one isn't a plain slash
        # date (e.g. a stray Excel serial, see the accident2020 P1 note)
        # must be skipped, not raise.
        df = pl.DataFrame({"code": ["A1", "A1"], "date": ["45746", "4/12/2023"]})
        assert find_transposed_date_string_pairs(df, "code", "date") == []


class TestDiscriminateTransposedDatePairs:
    def test_resolves_unique_using_report_date_gap(self) -> None:
        # row1 (April 12) reports 3 days later -> valid.
        # row2 (December 4, the transposed reading) would report BEFORE the
        # accident happened -> invalid. Only one candidate survives.
        df = pl.DataFrame(
            {
                "code": ["A", "A"],
                "accident": ["4/12/2023", "12/4/2023"],
                "report": ["4/15/2023", "4/15/2023"],
            }
        )
        result = discriminate_transposed_date_pairs(
            df, "code", ["A"], "accident", "report", month_first=True
        )
        assert result.resolved_unique == 1
        assert result.both_valid == 0
        assert result.neither_valid == 0
        resolution = result.resolutions[0]
        assert resolution.outcome == "resolved_unique"
        assert resolution.valid_dates == [date(2023, 4, 12)]

    def test_both_candidates_valid_is_undecidable(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["B", "B"],
                "accident": ["4/12/2023", "12/4/2023"],
                "report": ["4/20/2023", "12/10/2023"],
            }
        )
        result = discriminate_transposed_date_pairs(
            df, "code", ["B"], "accident", "report", month_first=True
        )
        assert result.both_valid == 1
        assert result.resolved_unique == 0
        assert result.neither_valid == 0

    def test_neither_candidate_valid_is_flagged(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["C", "C"],
                "accident": ["4/12/2023", "12/4/2023"],
                "report": ["6/1/2024", "6/1/2024"],
            }
        )
        result = discriminate_transposed_date_pairs(
            df, "code", ["C"], "accident", "report", month_first=True
        )
        assert result.neither_valid == 1
        assert result.resolved_unique == 0
        assert result.both_valid == 0

    def test_three_row_group_requires_all_rows_of_a_candidate_to_pass(self) -> None:
        # Candidate "April 12" has two rows: one valid, one with a huge gap.
        # Because not ALL rows for that candidate pass, the whole candidate
        # is invalid -- the surviving December 4 (one row, valid) wins.
        df = pl.DataFrame(
            {
                "code": ["D", "D", "D"],
                "accident": ["4/12/2023", "4/12/2023", "12/4/2023"],
                "report": ["4/20/2023", "8/1/2024", "12/10/2023"],
            }
        )
        result = discriminate_transposed_date_pairs(
            df, "code", ["D"], "accident", "report", month_first=True
        )
        assert result.resolved_unique == 1
        assert result.resolutions[0].valid_dates == [date(2023, 12, 4)]

    def test_unparseable_report_date_makes_that_candidate_invalid(self) -> None:
        # A malformed report date (e.g. a stray Excel serial, see the
        # accident2020 P1 note) must invalidate its candidate, not raise.
        df = pl.DataFrame(
            {
                "code": ["E", "E"],
                "accident": ["4/12/2023", "12/4/2023"],
                "report": ["not-a-date", "12/10/2023"],
            }
        )
        result = discriminate_transposed_date_pairs(
            df, "code", ["E"], "accident", "report", month_first=True
        )
        assert result.resolved_unique == 1
        assert result.resolutions[0].valid_dates == [date(2023, 12, 4)]

    def test_unparseable_raw_accident_date_skipped_as_candidate(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["F", "F"],
                "accident": ["not-a-date", "12/4/2023"],
                "report": ["1/1/2023", "12/10/2023"],
            }
        )
        result = discriminate_transposed_date_pairs(
            df, "code", ["F"], "accident", "report", month_first=True
        )
        assert result.resolutions[0].candidate_dates == [date(2023, 12, 4)]
        assert result.resolved_unique == 1


class TestParseExcelSerialDate:
    def test_known_serial(self) -> None:
        # The real accident2020.csv anomaly noted in CLAUDE.md P1.
        assert parse_excel_serial_date("44178") == date(2020, 12, 13)

    def test_invalid_string_returns_none(self) -> None:
        assert parse_excel_serial_date("not-a-number") is None

    def test_non_positive_returns_none(self) -> None:
        assert parse_excel_serial_date("0") is None
        assert parse_excel_serial_date("-5") is None


class TestParseAccidentDate:
    def test_dmy(self) -> None:
        assert parse_accident_date("25/3/2023", "dmy") == date(2023, 3, 25)

    def test_mdy(self) -> None:
        assert parse_accident_date("3/25/2023", "mdy") == date(2023, 3, 25)

    def test_excel_serial(self) -> None:
        assert parse_accident_date("44178", "excel_serial") == date(2020, 12, 13)

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown date_format"):
            parse_accident_date("1/1/2023", "ymd")


class TestComputeReportLagDays:
    def test_parses_per_file_format_and_computes_lag(self) -> None:
        df = pl.DataFrame(
            {
                "source_file": ["f_mdy.csv", "f_mdy.csv", "f_dmy.csv"],
                "accident": ["4/12/2023", "1/1/2023", "25/3/2023"],
                "report": ["4/15/2023", "1/10/2023", "28/3/2023"],
            }
        )
        formats = {"f_mdy.csv": "mdy", "f_dmy.csv": "dmy"}
        result = compute_report_lag_days(df, "source_file", "accident", "report", formats)
        assert result.height == 3
        lag_by_accident = dict(
            zip(result["accident"].to_list(), result["lag_days"].to_list(), strict=True)
        )
        assert lag_by_accident["4/12/2023"] == 3
        assert lag_by_accident["1/1/2023"] == 9
        assert lag_by_accident["25/3/2023"] == 3

    def test_unparseable_rows_dropped(self) -> None:
        df = pl.DataFrame(
            {
                "source_file": ["f.csv", "f.csv"],
                "accident": ["4/12/2023", "not-a-date"],
                "report": ["4/15/2023", "4/20/2023"],
            }
        )
        result = compute_report_lag_days(df, "source_file", "accident", "report", {"f.csv": "mdy"})
        assert result.height == 1

    def test_source_file_not_in_format_map_excluded(self) -> None:
        df = pl.DataFrame(
            {"source_file": ["unknown.csv"], "accident": ["4/12/2023"], "report": ["4/15/2023"]}
        )
        result = compute_report_lag_days(
            df, "source_file", "accident", "report", {"other.csv": "mdy"}
        )
        assert result.height == 0

    def test_no_matching_files_returns_empty_frame_with_lag_column(self) -> None:
        df = pl.DataFrame(
            {"source_file": ["x.csv"], "accident": ["1/1/2023"], "report": ["1/2/2023"]}
        )
        result = compute_report_lag_days(df, "source_file", "accident", "report", {})
        assert result.height == 0
        assert "lag_days" in result.columns


class TestSummarizeReportLag:
    def test_basic_stats(self) -> None:
        df = pl.DataFrame({"lag_days": [1, 2, 3, 4, 5]})
        profile = summarize_report_lag(df)
        assert profile.rows_parsed == 5
        assert profile.rows_negative_lag == 0
        assert profile.median_days == 3.0
        assert profile.max_days == 5.0
        assert profile.count_over_90_days == 0

    def test_negative_lag_excluded_from_percentiles_but_counted(self) -> None:
        df = pl.DataFrame({"lag_days": [-5, -3, 10, 20, 30]})
        profile = summarize_report_lag(df)
        assert profile.rows_parsed == 5
        assert profile.rows_negative_lag == 2
        assert profile.max_days == 30.0
        assert profile.median_days == 20.0

    def test_count_over_90_days(self) -> None:
        df = pl.DataFrame({"lag_days": [10, 95, 200]})
        profile = summarize_report_lag(df)
        assert profile.count_over_90_days == 2

    def test_all_negative_returns_zeroed_stats_not_a_crash(self) -> None:
        df = pl.DataFrame({"lag_days": [-1, -2]})
        profile = summarize_report_lag(df)
        assert profile.rows_parsed == 2
        assert profile.rows_negative_lag == 2
        assert profile.median_days == 0.0
        assert profile.max_days == 0.0


class TestCountRowsOverLagThreshold:
    def test_counts_rows_above_threshold_excluding_negative(self) -> None:
        df = pl.DataFrame({"lag_days": [-10, 5, 50, 91, 200]})
        assert count_rows_over_lag_threshold(df, 90) == 2
        assert count_rows_over_lag_threshold(df, 40) == 3

    def test_threshold_boundary_is_exclusive(self) -> None:
        df = pl.DataFrame({"lag_days": [90]})
        assert count_rows_over_lag_threshold(df, 90) == 0


class TestReportLagQuantilesByGroup:
    def test_group_breakdown(self) -> None:
        df = pl.DataFrame({"agency": ["A", "A", "B", "B"], "lag_days": [10, 20, 100, 200]})
        result = report_lag_quantiles_by_group(df, "agency")
        rows = {r["agency"]: r for r in result.to_dicts()}
        assert rows["A"]["n"] == 2
        assert rows["A"]["median_days"] == 15.0
        assert rows["B"]["median_days"] == 150.0

    def test_negative_lag_excluded_from_group_quantiles(self) -> None:
        df = pl.DataFrame({"agency": ["A", "A"], "lag_days": [-5, 10]})
        result = report_lag_quantiles_by_group(df, "agency")
        assert result.height == 1
        assert result["n"][0] == 1
