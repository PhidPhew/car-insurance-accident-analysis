from pathlib import Path

import polars as pl

from rri.cleaning.log import CleaningLogEntry, cleaning_log_to_frame, write_cleaning_log


class TestCleaningLogToFrame:
    def test_empty_list_returns_empty_frame_with_expected_schema(self) -> None:
        result = cleaning_log_to_frame([])
        assert result.height == 0
        expected = {"step", "rule", "rows_before", "rows_affected", "description"}
        assert set(result.columns) == expected

    def test_entries_become_rows_in_order(self) -> None:
        entries = [
            CleaningLogEntry("step1", "rule_a", 100, 5, "did a thing"),
            CleaningLogEntry("step1", "rule_b", 100, 0, "did nothing"),
        ]
        result = cleaning_log_to_frame(entries)
        assert result.height == 2
        assert result["rule"].to_list() == ["rule_a", "rule_b"]
        assert result["rows_affected"].to_list() == [5, 0]


class TestWriteCleaningLog:
    def test_writes_readable_csv(self, tmp_path: Path) -> None:
        entries = [CleaningLogEntry("step1", "rule_a", 10, 2, "desc")]
        out_path = tmp_path / "cleaning_log.csv"
        write_cleaning_log(entries, str(out_path))
        reloaded = pl.read_csv(out_path)
        assert reloaded.height == 1
        assert reloaded["rule"].to_list() == ["rule_a"]
