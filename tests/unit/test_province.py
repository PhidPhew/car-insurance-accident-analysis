import pytest

from rri.cleaning.province import (
    load_canonical_province_names,
    match_province_values,
    normalize_province,
)


class TestNormalizeProvinceCanonical:
    def test_already_canonical_name_passes_through(self) -> None:
        assert normalize_province("กรุงเทพมหานคร") == "กรุงเทพมหานคร"

    def test_ordinary_province_name_matches_directly(self) -> None:
        assert normalize_province("ลพบุรี") == "ลพบุรี"


class TestNormalizeProvincePrefixStripping:
    def test_strips_jor_dot_prefix(self) -> None:
        assert normalize_province("จ.นนทบุรี") == "นนทบุรี"

    def test_strips_full_changwat_prefix(self) -> None:
        assert normalize_province("จังหวัดนนทบุรี") == "นนทบุรี"

    def test_bangkok_never_strips_its_own_leading_characters(self) -> None:
        # กรุงเทพ... does not start with the "จ." prefix, so this is really
        # a no-op guard -- but explicit, since Bangkok is the one province
        # covered by CLAUDE.md's flood-file quirk of having NO "จ." prefix
        # while every other province in that file does.
        assert normalize_province("กรุงเทพมหานคร") == "กรุงเทพมหานคร"


class TestNormalizeProvinceWhitespaceAndFormatting:
    def test_normalizes_multiple_internal_spaces(self) -> None:
        assert normalize_province("จ.  นนทบุรี") == "นนทบุรี"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_province("  ลพบุรี  ") == "ลพบุรี"


class TestNormalizeProvinceAliases:
    def test_english_name_maps_to_thai(self) -> None:
        assert normalize_province("Bangkok") == "กรุงเทพมหานคร"
        assert normalize_province("Krabi") == "กระบี่"
        assert normalize_province("Kanchanaburi") == "กาญจนบุรี"

    def test_ayutthaya_short_form_maps_to_full_official_name(self) -> None:
        # Found via an exact diff between the accidents file's 77 province
        # names and the rainfall file's 77 canonical names -- the accidents
        # source_file uses the short form.
        assert normalize_province("อยุธยา") == "พระนครศรีอยุธยา"

    def test_bangkok_informal_variants(self) -> None:
        assert normalize_province("กรุงเทพฯ") == "กรุงเทพมหานคร"
        assert normalize_province("กทม.") == "กรุงเทพมหานคร"
        assert normalize_province("กรุงเทพ") == "กรุงเทพมหานคร"

    def test_flood_file_jor_prefixed_english_lookalike_still_resolves(self) -> None:
        # flood's PROV_T format: "จ.กระบี่" (จ. prefix + plain Thai name).
        assert normalize_province("จ.กระบี่") == "กระบี่"


class TestNormalizeProvinceEdgeCases:
    def test_none_returns_none(self) -> None:
        assert normalize_province(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_province("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_province("   ") is None

    def test_unknown_name_returns_none(self) -> None:
        assert normalize_province("Atlantis") is None

    def test_unknown_thai_looking_name_returns_none(self) -> None:
        assert normalize_province("จังหวัดสมมติ") is None


class TestMatchProvinceValues:
    def test_all_matched_gives_full_rate(self) -> None:
        result = match_province_values("test_source", ["กรุงเทพมหานคร", "ลพบุรี", "Bangkok"])
        assert result.total_distinct_values == 3
        assert result.matched_count == 3
        assert result.unmatched_values == ()
        assert result.match_rate == 1.0

    def test_partial_match_lists_unmatched_values(self) -> None:
        result = match_province_values("test_source", ["กรุงเทพมหานคร", "Atlantis", "Narnia"])
        assert result.total_distinct_values == 3
        assert result.matched_count == 1
        assert set(result.unmatched_values) == {"Atlantis", "Narnia"}
        assert result.match_rate == pytest.approx(1 / 3)

    def test_ignores_blank_and_whitespace_only_values(self) -> None:
        result = match_province_values("test_source", ["กรุงเทพมหานคร", "", "   "])
        assert result.total_distinct_values == 1
        assert result.matched_count == 1

    def test_deduplicates_repeated_values(self) -> None:
        result = match_province_values("test_source", ["ลพบุรี", "ลพบุรี", "ลพบุรี"])
        assert result.total_distinct_values == 1

    def test_empty_input_has_zero_rate_not_division_error(self) -> None:
        result = match_province_values("test_source", [])
        assert result.total_distinct_values == 0
        assert result.match_rate == 0.0


class TestLoadCanonicalProvinceNames:
    def test_returns_exactly_77_provinces(self) -> None:
        assert len(load_canonical_province_names()) == 77

    def test_includes_bangkok(self) -> None:
        assert "กรุงเทพมหานคร" in load_canonical_province_names()
