import polars as pl
import pytest

from rri.cleaning.dedup import assert_casualty_agreement, deduplicate_accidents


class TestAssertCasualtyAgreement:
    def test_passes_when_all_groups_agree(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1", "2"],
                "ผู้เสียชีวิต": [0, 0, 1],
                "ผู้บาดเจ็บสาหัส": [1, 1, 0],
                "ผู้บาดเจ็บเล็กน้อย": [0, 0, 0],
                "รวมจำนวนผู้บาดเจ็บ": [1, 1, 0],
            }
        )
        assert_casualty_agreement(df)  # does not raise

    def test_raises_on_disagreement(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1"],
                "ผู้เสียชีวิต": [0, 5],
                "ผู้บาดเจ็บสาหัส": [1, 1],
                "ผู้บาดเจ็บเล็กน้อย": [0, 0],
                "รวมจำนวนผู้บาดเจ็บ": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="disagree on 'ผู้เสียชีวิต'"):
            assert_casualty_agreement(df)

    def test_null_group_keys_excluded(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": [None, None],
                "ผู้เสียชีวิต": [0, 999],
                "ผู้บาดเจ็บสาหัส": [0, 0],
                "ผู้บาดเจ็บเล็กน้อย": [0, 0],
                "รวมจำนวนผู้บาดเจ็บ": [0, 0],
            }
        )
        assert_casualty_agreement(df)  # does not raise -- null keys never grouped


def _base_columns() -> dict[str, list[str | int | None]]:
    return {
        "ผู้เสียชีวิต": [0, 0],
        "ผู้บาดเจ็บสาหัส": [0, 0],
        "ผู้บาดเจ็บเล็กน้อย": [0, 0],
        "รวมจำนวนผู้บาดเจ็บ": [0, 0],
    }


class TestDeduplicateAccidents:
    def test_collapses_duplicate_group_to_one_row(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1", "2"],
                "date_certainty": ["certain", "certain", "certain"],
                "_row_hash": ["aaa", "bbb", "ccc"],
                "field_a": ["x", "y", "z"],
                "ผู้เสียชีวิต": [0, 0, 1],
                "ผู้บาดเจ็บสาหัส": [0, 0, 0],
                "ผู้บาดเจ็บเล็กน้อย": [0, 0, 0],
                "รวมจำนวนผู้บาดเจ็บ": [0, 0, 0],
            }
        )
        result, log = deduplicate_accidents(df, columns_for_null_count=["field_a"])
        assert result.height == 2
        assert log.rows_before == 3
        assert log.rows_after == 2
        assert log.groups_collapsed == 1

    def test_prefers_certain_over_ambiguous(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1"],
                "date_certainty": ["ambiguous", "certain"],
                "_row_hash": ["aaa", "bbb"],
                "field_a": ["x", "y"],
                **_base_columns(),
            }
        )
        result, _ = deduplicate_accidents(df, columns_for_null_count=["field_a"])
        assert result["date_certainty"].to_list() == ["certain"]
        assert result["field_a"].to_list() == ["y"]

    def test_tie_break_prefers_fewer_nulls(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1"],
                "date_certainty": ["certain", "certain"],
                "_row_hash": ["aaa", "bbb"],
                "field_a": [None, "complete"],
                "field_b": [None, "also_complete"],
                **_base_columns(),
            }
        )
        result, _ = deduplicate_accidents(df, columns_for_null_count=["field_a", "field_b"])
        assert result["field_a"].to_list() == ["complete"]

    def test_tie_break_falls_back_to_row_hash_for_determinism(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1"],
                "date_certainty": ["certain", "certain"],
                "_row_hash": ["zzz", "aaa"],
                "field_a": ["second_by_hash", "first_by_hash"],
                **_base_columns(),
            }
        )
        result, _ = deduplicate_accidents(df, columns_for_null_count=["field_a"])
        assert result["field_a"].to_list() == ["first_by_hash"]

    def test_null_acc_code_rows_pass_through_unchanged(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": [None, None, "1"],
                "date_certainty": ["certain", "certain", "certain"],
                "_row_hash": ["aaa", "bbb", "ccc"],
                "field_a": ["x", "y", "z"],
                "ผู้เสียชีวิต": [0, 0, 0],
                "ผู้บาดเจ็บสาหัส": [0, 0, 0],
                "ผู้บาดเจ็บเล็กน้อย": [0, 0, 0],
                "รวมจำนวนผู้บาดเจ็บ": [0, 0, 0],
            }
        )
        result, log = deduplicate_accidents(df, columns_for_null_count=["field_a"])
        assert result.height == 3  # both null-key rows survive untouched
        assert log.groups_collapsed == 0

    def test_no_duplicates_leaves_everything_unchanged(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "2"],
                "date_certainty": ["certain", "certain"],
                "_row_hash": ["aaa", "bbb"],
                "field_a": ["x", "y"],
                **_base_columns(),
            }
        )
        result, log = deduplicate_accidents(df, columns_for_null_count=["field_a"])
        assert result.height == 2
        assert log.groups_collapsed == 0

    def test_raises_before_deduping_on_casualty_disagreement(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1"],
                "date_certainty": ["certain", "certain"],
                "_row_hash": ["aaa", "bbb"],
                "field_a": ["x", "y"],
                "ผู้เสียชีวิต": [0, 999],
                "ผู้บาดเจ็บสาหัส": [0, 0],
                "ผู้บาดเจ็บเล็กน้อย": [0, 0],
                "รวมจำนวนผู้บาดเจ็บ": [0, 0],
            }
        )
        with pytest.raises(ValueError, match="disagree on"):
            deduplicate_accidents(df, columns_for_null_count=["field_a"])

    def test_examples_capture_kept_and_dropped_certainty(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1"],
                "date_certainty": ["ambiguous", "certain"],
                "_row_hash": ["aaa", "bbb"],
                "field_a": ["x", "y"],
                **_base_columns(),
            }
        )
        _, log = deduplicate_accidents(df, columns_for_null_count=["field_a"])
        assert len(log.examples) == 1
        example = log.examples[0]
        assert example.group_key == "1"
        assert example.rows_in_group == 2
        assert example.kept_certainty == "certain"
        assert example.dropped_certainties == ("ambiguous",)

    def test_example_count_caps_number_of_groups_logged(self) -> None:
        df = pl.DataFrame(
            {
                "ACC_CODE": ["1", "1", "2", "2", "3", "3"],
                "date_certainty": ["certain"] * 6,
                "_row_hash": ["a1", "a2", "b1", "b2", "c1", "c2"],
                "field_a": ["x"] * 6,
                "ผู้เสียชีวิต": [0] * 6,
                "ผู้บาดเจ็บสาหัส": [0] * 6,
                "ผู้บาดเจ็บเล็กน้อย": [0] * 6,
                "รวมจำนวนผู้บาดเจ็บ": [0] * 6,
            }
        )
        _, log = deduplicate_accidents(df, columns_for_null_count=["field_a"], example_count=2)
        assert log.groups_collapsed == 3
        assert len(log.examples) == 2
