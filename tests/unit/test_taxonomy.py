from pathlib import Path

import polars as pl

from rri.cleaning.taxonomy import (
    TaxonomyMapping,
    apply_taxonomy,
    load_taxonomy_mapping,
)


class TestTaxonomyMapping:
    def test_group_for_known_value(self) -> None:
        mapping = TaxonomyMapping({"a": "group_a"})
        assert mapping.group_for("a") == "group_a"

    def test_group_for_none_is_none(self) -> None:
        mapping = TaxonomyMapping({"a": "group_a"})
        assert mapping.group_for(None) is None

    def test_group_for_unknown_value_is_none(self) -> None:
        mapping = TaxonomyMapping({"a": "group_a"})
        assert mapping.group_for("unknown") is None


class TestLoadTaxonomyMapping:
    def test_loads_csv_into_lookup(self, tmp_path: Path) -> None:
        path = tmp_path / "test_groups.csv"
        content = "raw_value,canonical_group,row_count\nขับเร็ว,speeding,100\n"
        path.write_text(content, encoding="utf-8")
        mapping = load_taxonomy_mapping(path)
        assert mapping.group_for("ขับเร็ว") == "speeding"

    def test_works_without_row_count_column(self, tmp_path: Path) -> None:
        path = tmp_path / "test_groups.csv"
        path.write_text("raw_value,canonical_group\na,group_a\n", encoding="utf-8")
        mapping = load_taxonomy_mapping(path)
        assert mapping.group_for("a") == "group_a"


class TestApplyTaxonomy:
    def test_maps_known_values(self) -> None:
        df = pl.DataFrame({"cause": ["a", "b", "a"]})
        mapping = TaxonomyMapping({"a": "group_a", "b": "group_b"})
        result, log = apply_taxonomy(df, "cause", mapping)
        assert result["cause_group"].to_list() == ["group_a", "group_b", "group_a"]
        assert log.rows_mapped == 3
        assert log.rows_null == 0
        assert log.rows_unmapped == 0

    def test_null_values_stay_null_not_counted_as_unmapped(self) -> None:
        df = pl.DataFrame({"cause": ["a", None]})
        mapping = TaxonomyMapping({"a": "group_a"})
        result, log = apply_taxonomy(df, "cause", mapping)
        assert result["cause_group"].to_list() == ["group_a", None]
        assert log.rows_null == 1
        assert log.rows_unmapped == 0

    def test_unmapped_non_null_value_becomes_null_and_is_logged(self) -> None:
        df = pl.DataFrame({"cause": ["a", "not_in_table"]})
        mapping = TaxonomyMapping({"a": "group_a"})
        result, log = apply_taxonomy(df, "cause", mapping)
        assert result["cause_group"].to_list() == ["group_a", None]
        assert log.rows_unmapped == 1
        assert log.unmapped_values == ("not_in_table",)

    def test_custom_output_column_name(self) -> None:
        df = pl.DataFrame({"cause": ["a"]})
        mapping = TaxonomyMapping({"a": "group_a"})
        result, _ = apply_taxonomy(df, "cause", mapping, new_column="canonical_cause")
        assert "canonical_cause" in result.columns
        assert "cause_group" not in result.columns

    def test_original_column_untouched(self) -> None:
        df = pl.DataFrame({"cause": ["a"]})
        mapping = TaxonomyMapping({"a": "group_a"})
        result, _ = apply_taxonomy(df, "cause", mapping)
        assert result["cause"].to_list() == ["a"]

    def test_duplicate_unmapped_values_reported_once(self) -> None:
        df = pl.DataFrame({"cause": ["x", "x", "x"]})
        mapping = TaxonomyMapping({})
        _, log = apply_taxonomy(df, "cause", mapping)
        assert log.rows_unmapped == 3
        assert log.unmapped_values == ("x",)
