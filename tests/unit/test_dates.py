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

    def test_two_hard_valid_candidates_stay_ambiguous_even_with_a_likelihood_winner(self) -> None:
        # Both candidates survive the hard chronology filter against the
        # single shared report date -- a likelihood winner is still picked
        # (for reference/tie-break), but the group MUST stay "ambiguous":
        # two chronologically possible answers is a guess, not a fact.
        df = pl.DataFrame(
            {
                "code": ["C", "C"],
                "accident_raw": ["5/1/2023", "1/5/2023"],
                "report_raw": ["6/1/2023", "6/1/2023"],
                "agency": ["highways", "highways"],
            }
        )
        # candidate May 1 -> lag 31 (median 31, deviation 0)
        # candidate Jan 5 -> lag 147 (deviation 116) -- both non-negative, both <= 400
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["C"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 31.0},
            agency_max_lag_days={"highways": 400.0},
        )
        result = resolutions[0]
        assert result.outcome == "ambiguous"
        assert result.winner_date == date(2023, 5, 1)
        assert result.margin_days == 116.0

    def test_hard_filter_uses_every_report_date_in_the_group_not_just_the_own_row(self) -> None:
        # This is the bug the checkpoint review caught: candidate "Mar 1"
        # looks self-consistently valid against ITS OWN row's report date
        # (also transposed, "4/1" -> Apr 1, giving a plausible 31-day lag)
        # -- but is chronologically impossible against the OTHER row's
        # report date (Jan 4, which is before Mar 1). The hard filter must
        # check every report date in the group, not just the paired one.
        df = pl.DataFrame(
            {
                "code": ["G", "G"],
                "accident_raw": ["1/3/2023", "3/1/2023"],
                "report_raw": ["1/4/2023", "4/1/2023"],
                "agency": ["rural", "rural"],
            }
        )
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["G"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"rural": 5.0},
            agency_max_lag_days={"rural": 400.0},
        )
        result = resolutions[0]
        assert result.outcome == "resolved"
        assert result.winner_date == date(2023, 1, 3)
        assert result.loser_dates == (date(2023, 3, 1),)
        assert result.had_order_violation is True

    def test_had_order_violation_is_false_when_both_candidates_hard_valid(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["C", "C"],
                "accident_raw": ["5/1/2023", "1/5/2023"],
                "report_raw": ["6/1/2023", "6/1/2023"],
                "agency": ["highways", "highways"],
            }
        )
        resolutions = resolve_transposed_pairs_by_likelihood(
            df,
            "code",
            ["C"],
            "accident_raw",
            "report_raw",
            "agency",
            month_first=True,
            agency_median_lag_days={"highways": 31.0},
            agency_max_lag_days={"highways": 400.0},
        )
        assert resolutions[0].had_order_violation is False

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
        # Candidates "Mar 4" and "Apr 3" (only 30 days apart) both survive
        # the hard filter against all 3 group report dates. "Mar 4" has 2
        # rows (lags 89, 119 -- mean deviation 15 from median 90); "Apr 3"
        # has 1 row (lag 73 -- deviation 17). Mar 4 wins on a narrower
        # margin despite having a farther-off individual row, because the
        # score is a mean across every row sharing that candidate.
        df = pl.DataFrame(
            {
                "code": ["E", "E", "E"],
                "accident_raw": ["3/4/2023", "3/4/2023", "4/3/2023"],
                "report_raw": ["6/1/2023", "7/1/2023", "6/15/2023"],
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
            agency_median_lag_days={"highways": 90.0},
            agency_max_lag_days={"highways": 400.0},
        )
        result = resolutions[0]
        assert result.outcome == "ambiguous"
        assert result.winner_date == date(2023, 3, 4)
        assert result.margin_days == 2.0

    def test_different_agencies_use_their_own_median(self) -> None:
        df = pl.DataFrame(
            {
                "code": ["F", "F"],
                "accident_raw": ["1/10/2023", "10/1/2023"],
                "report_raw": ["1/15/2023", "10/2/2023"],
                "agency": ["rural", "rural"],
            }
        )
        # candidate Jan 10 -> valid against both group report dates (Jan 15, Oct 2)
        # candidate Oct 1 -> chronologically impossible against Jan 15 (report
        # predates it) -> eliminated by the hard filter, regardless of median
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
        assert resolutions[0].outcome == "resolved"
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
                had_order_violation=True,
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
                had_order_violation=False,
            )
        ]
        result = apply_date_certainty(df, resolutions, "code")
        assert result["date_certainty"].to_list() == ["ambiguous", "ambiguous"]

    def test_ambiguous_outcome_with_a_recorded_winner_still_marks_both_ambiguous(self) -> None:
        # The checkpoint-review fix: even when stage 2 picked a likelihood
        # winner for reference, outcome="ambiguous" must still mark BOTH
        # rows ambiguous -- a recorded winner is not the same as certainty.
        from rri.cleaning.dates import LikelihoodResolution

        df = pl.DataFrame(
            {
                "code": ["C", "C"],
                "accident_date": [date(2023, 5, 1), date(2023, 1, 5)],
            }
        )
        resolutions = [
            LikelihoodResolution(
                group_key="C",
                candidate_dates=(date(2023, 5, 1), date(2023, 1, 5)),
                winner_date=date(2023, 5, 1),
                loser_dates=(date(2023, 1, 5),),
                margin_days=116.0,
                outcome="ambiguous",
                had_order_violation=False,
            )
        ]
        result = apply_date_certainty(df, resolutions, "code")
        assert result["date_certainty"].to_list() == ["ambiguous", "ambiguous"]
