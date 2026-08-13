from datetime import date

import polars as pl

from rri.cleaning.dates import (
    apply_date_certainty,
    check_year_match,
    day_of_month_histogram,
    parse_dates,
    resolve_transposed_pairs_by_likelihood,
)


class TestParseDates:
    def test_parses_per_file_format(self) -> None:
        df = pl.DataFrame(
            {
                "source_file": ["accident2023.csv", "accident2019.csv"],
                "วันที่เกิดเหตุ": ["4/12/2023", "25/3/2019"],
                "วันที่รายงาน": ["4/15/2023", "28/3/2019"],
            }
        )
        result = parse_dates(df, {"accident2023.csv": "mdy", "accident2019.csv": "dmy"})
        assert result["accident_date"].to_list() == [date(2023, 4, 12), date(2019, 3, 25)]
        assert result["report_date"].to_list() == [date(2023, 4, 15), date(2019, 3, 28)]

    def test_unparseable_value_becomes_null_not_an_error(self) -> None:
        df = pl.DataFrame(
            {
                "source_file": ["accident2022.csv"],
                "วันที่เกิดเหตุ": ["02/16/2022"],
                "วันที่รายงาน": ["//"],
            }
        )
        result = parse_dates(df, {"accident2022.csv": "mdy"})
        assert result["accident_date"].to_list() == [date(2022, 2, 16)]
        assert result["report_date"].to_list() == [None]

    def test_null_raw_value_becomes_null(self) -> None:
        df = pl.DataFrame(
            {
                "source_file": ["accident2026.csv"],
                "วันที่เกิดเหตุ": ["1/1/2026"],
                "วันที่รายงาน": [None],
            }
        )
        result = parse_dates(df, {"accident2026.csv": "dmy"})
        assert result["report_date"].to_list() == [None]

    def test_excel_serial_format(self) -> None:
        df = pl.DataFrame(
            {
                "source_file": ["accident2024.csv"],
                "วันที่เกิดเหตุ": ["45746"],
                "วันที่รายงาน": ["45750"],
            }
        )
        result = parse_dates(df, {"accident2024.csv": "excel_serial"})
        assert result["accident_date"].to_list()[0] is not None


class TestCheckYearMatch:
    def test_zero_mismatches_on_matching_years(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "2"],
                "accident_date": [date(2023, 4, 12), date(2019, 3, 25)],
                "ปีที่เกิดเหตุ": ["2023", "2019"],
            }
        )
        result = check_year_match(df)
        assert result.rows_checked == 2
        assert result.rows_mismatched == 0
        assert result.rows_unparseable == 0

    def test_detects_mismatch(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1"],
                "accident_date": [date(2023, 4, 12)],
                "ปีที่เกิดเหตุ": ["2022"],
            }
        )
        result = check_year_match(df)
        assert result.rows_mismatched == 1
        assert result.mismatch_examples == (("1", 2023, "2022"),)

    def test_null_accident_date_counted_as_unparseable_not_mismatch(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1"],
                "accident_date": pl.Series([None], dtype=pl.Date),
                "ปีที่เกิดเหตุ": ["2023"],
            }
        )
        result = check_year_match(df)
        assert result.rows_unparseable == 1
        assert result.rows_mismatched == 0

    def test_max_examples_caps_returned_list(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "2", "3"],
                "accident_date": [date(2023, 1, 1)] * 3,
                "ปีที่เกิดเหตุ": ["2022", "2021", "2020"],
            }
        )
        result = check_year_match(df, max_examples=2)
        assert result.rows_mismatched == 3
        assert len(result.mismatch_examples) == 2


class TestDayOfMonthHistogram:
    def test_counts_by_day(self) -> None:
        df = pl.DataFrame(
            {"accident_date": [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 15)]}
        )
        histogram = day_of_month_histogram(df)
        assert histogram[1] == 2
        assert histogram[15] == 1
        assert histogram[2] == 0

    def test_covers_all_31_days_even_if_unused(self) -> None:
        df = pl.DataFrame({"accident_date": [date(2023, 1, 1)]})
        histogram = day_of_month_histogram(df)
        assert set(histogram.keys()) == set(range(1, 32))

    def test_ignores_null_dates(self) -> None:
        df = pl.DataFrame({"accident_date": pl.Series([date(2023, 1, 1), None], dtype=pl.Date)})
        histogram = day_of_month_histogram(df)
        assert sum(histogram.values()) == 1


class TestResolveTransposedPairsByLikelihood:
    def test_resolves_when_one_candidate_much_closer_to_median(self) -> None:
        # code "A": accident "4/12/2023" (Apr 12) reports 10 days later,
        # agency median lag is 10 -> deviation 0. The transposed candidate
        # "12/4/2023" (Dec 4) reporting the same date is a huge negative
        # gap -> not viable at all. Only one viable candidate -> resolved.
        df = pl.DataFrame(
            {
                "code": ["A", "A"],
                "accident_raw": ["4/12/2023", "12/4/2023"],
                "report_raw": ["4/22/2023", "4/22/2023"],
                "agency": ["highways", "highways"],
            }
        )
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["A"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 10.0},
            agency_max_lag_days={"highways": 400.0},
        )
        assert len(resolutions) == 1
        result = resolutions[0]
        assert result.outcome == "resolved"
        assert result.winner_date == date(2023, 4, 12)
        assert result.loser_dates == (date(2023, 12, 4),)
        assert result.margin_days is None  # only one viable candidate, no comparison made

    def test_picks_closer_candidate_when_both_viable(self) -> None:
        # Both candidates produce a non-negative, sane lag, but one is much
        # closer to the agency's median lag than the other.
        df = pl.DataFrame(
            {
                "code": ["B", "B"],
                "accident_raw": ["1/2/2023", "2/1/2023"],
                "report_raw": ["1/12/2023", "1/12/2023"],
                "agency": ["highways", "highways"],
            }
        )
        # candidate Jan 2 -> lag 10 (median is 10, deviation 0)
        # candidate Feb 1 -> lag -20 (negative -> not viable at all)
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["B"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 10.0},
            agency_max_lag_days={"highways": 400.0},
        )
        assert resolutions[0].outcome == "resolved"
        assert resolutions[0].winner_date == date(2023, 1, 2)

    def test_records_margin_when_two_candidates_are_comparable(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["C", "C"],
                "accident_raw": ["1/10/2023", "10/1/2023"],
                "report_raw": ["1/20/2023", "10/11/2023"],
                "agency": ["highways", "highways"],
            }
        )
        # candidate Jan 10 -> lag 10 (median 10, deviation 0)
        # candidate Oct 1 -> lag 10 too (deviation 0) -- both equally close
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["C"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 10.0},
            agency_max_lag_days={"highways": 400.0},
        )
        result = resolutions[0]
        assert result.outcome == "resolved"
        assert result.margin_days == 0.0

    def test_ambiguous_when_no_candidate_has_any_usable_report_date(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["D", "D"],
                "accident_raw": ["1/10/2023", "10/1/2023"],
                "report_raw": ["//", "//"],
                "agency": ["highways", "highways"],
            }
        )
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["D"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 10.0},
            agency_max_lag_days={"highways": 400.0},
        )
        assert resolutions[0].outcome == "ambiguous"
        assert resolutions[0].winner_date is None

    def test_three_row_group_uses_mean_deviation_across_all_rows_of_a_candidate(self) -> None:
        # Candidate "Apr 12" has 2 rows: one with lag exactly at median (dev 0),
        # one far off (dev 90) -> mean deviation 45. Candidate "Dec 4" has 1
        # row with deviation 5 -> should win despite fewer rows.
        df = pl.DataFrame(
            {
                "code": ["E", "E", "E"],
                "accident_raw": ["4/12/2023", "4/12/2023", "12/4/2023"],
                "report_raw": ["4/22/2023", "7/21/2023", "12/9/2023"],
                "agency": ["highways", "highways", "highways"],
            }
        )
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["E"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 10.0},
            agency_max_lag_days={"highways": 400.0},
        )
        assert resolutions[0].winner_date == date(2023, 12, 4)

    def test_different_agencies_use_their_own_median(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["F", "F"],
                "accident_raw": ["1/10/2023", "10/1/2023"],
                "report_raw": ["1/15/2023", "10/2/2023"],
                "agency": ["rural", "rural"],
            }
        )
        # candidate Jan 10 -> lag 5 (rural median 5, deviation 0)
        # candidate Oct 1 -> lag 1 (deviation 4)
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["F"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"rural": 5.0},
            agency_max_lag_days={"rural": 400.0},
        )
        assert resolutions[0].winner_date == date(2023, 1, 10)


class TestApplyDateCertainty:
    def test_defaults_to_certain_for_untouched_rows(self) -> None:
        df = pl.DataFrame({"code": ["X"], "accident_date": [date(2023, 1, 1)]})
        result = apply_date_certainty(df, [], "code")
        assert result["date_certainty"].to_list() == ["certain"]

    def test_winner_row_is_certain_loser_row_is_ambiguous(self) -> None:
        from rri.cleaning.dates import LikelihoodResolution

        df = pl.DataFrame(
            {
                "code": ["A", "A"],
                "accident_date": [date(2023, 4, 12), date(2023, 12, 4)],
            }
        )
        resolutions = [
            LikelihoodResolution(
                group_key="A",
                candidate_dates=(date(2023, 4, 12), date(2023, 12, 4)),
                winner_date=date(2023, 4, 12),
                loser_dates=(date(2023, 12, 4),),
                margin_days=None,
                outcome="resolved",
            )
        ]
        result = apply_date_certainty(df, resolutions, "code")
        assert result["date_certainty"].to_list() == ["certain", "ambiguous"]

    def test_ambiguous_outcome_marks_both_rows_ambiguous(self) -> None:
        from rri.cleaning.dates import LikelihoodResolution

        df = pl.DataFrame(
            {
                "code": ["D", "D"],
                "accident_date": [date(2023, 1, 10), date(2023, 10, 1)],
            }
        )
        resolutions = [
            LikelihoodResolution(
                group_key="D",
                candidate_dates=(date(2023, 1, 10), date(2023, 10, 1)),
                winner_date=None,
                loser_dates=(date(2023, 1, 10), date(2023, 10, 1)),
                margin_days=None,
                outcome="ambiguous",
            )
        ]
        result = apply_date_certainty(df, resolutions, "code")
        assert result["date_certainty"].to_list() == ["ambiguous", "ambiguous"]
