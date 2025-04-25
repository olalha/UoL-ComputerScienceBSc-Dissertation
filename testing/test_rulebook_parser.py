import pytest
from pathlib import Path

from input_manager.rulebook_parser import parse_rulebook_excel

# Path to test rulebooks directory
RULEBOOKS_DIR = Path(__file__).parent / "testing/testing_excel_rulebooks"

def test_valid_rulebook():
    """Test that a valid rulebook is correctly parsed."""
    rulebook_path = RULEBOOKS_DIR / "Rulebook_full_valid.xlsx"
    result = parse_rulebook_excel(rulebook_path)
    
    # Check that parsing was successful
    assert result is not None
    
    # Verify key properties of the parsed rulebook
    assert "content_title" in result
    assert "collection_mode" in result
    assert "total" in result
    assert "content_rules" in result
    assert "collection_ranges" in result
    
    # Check collection mode is valid
    assert result["collection_mode"] in ("word", "chunk")
    
    # Check content rules structure
    assert len(result["content_rules"]) > 0
    for topic, rule in result["content_rules"].items():
        assert "total_proportion" in rule
        assert "sentiment_proportion" in rule
        assert "chunk_min_wc" in rule
        assert "chunk_max_wc" in rule
        assert "chunk_pref" in rule
        assert "chunk_wc_distribution" in rule
        
        # Validate value constraints
        assert 0 <= rule["total_proportion"] <= 1
        assert len(rule["sentiment_proportion"]) == 3
        assert abs(sum(rule["sentiment_proportion"]) - 1) < 1e-9
        assert rule["chunk_min_wc"] > 0
        assert rule["chunk_max_wc"] > rule["chunk_min_wc"]
        assert 0 <= rule["chunk_pref"] <= 1
        assert rule["chunk_wc_distribution"] > 0
    
    # Check collection ranges
    assert len(result["collection_ranges"]) > 0
    sum_fractions = 0
    prev_end = None
    for range_dict in result["collection_ranges"]:
        start, end = range_dict["range"]
        assert end >= start
        if prev_end is not None:
            assert start == prev_end + 1
        prev_end = end
        sum_fractions += range_dict["target_fraction"]
    
    # Check target fractions sum to 1
    assert abs(sum_fractions - 1) < 1e-9


def test_invalid_sentiment_ratio():
    """Test that a rulebook with invalid sentiment ratios is rejected."""
    rulebook_path = RULEBOOKS_DIR / "Rulbeook_sentiment_ration_invalid.xlsx"
    result = parse_rulebook_excel(rulebook_path)
    
    # The parsing should fail
    assert result is None


def test_invalid_mode():
    """Test that a rulebook with invalid collection mode is rejected."""
    rulebook_path = RULEBOOKS_DIR / "Rulebook_mode_invalid.xlsx"
    result = parse_rulebook_excel(rulebook_path)
    
    # The parsing should fail
    assert result is None


def test_invalid_ranges():
    """Test that a rulebook with invalid collection ranges is rejected."""
    rulebook_path = RULEBOOKS_DIR / "Rulebook_ranges_invalid.xlsx"
    result = parse_rulebook_excel(rulebook_path)
    
    # The parsing should fail
    assert result is None


def test_invalid_topic_proportion():
    """Test that a rulebook with invalid topic proportions is rejected."""
    rulebook_path = RULEBOOKS_DIR / "Rulebook_topic_proportion_invalid.xlsx"
    result = parse_rulebook_excel(rulebook_path)
    
    # The parsing should fail
    assert result is None