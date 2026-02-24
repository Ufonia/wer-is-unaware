"""Tests for metrics.cleaning — transcript normalization."""

from metrics.cleaning import get_clean_transcript


def test_number_to_words():
    result = get_clean_transcript("3 cats")
    assert "three" in result
    assert "cats" in result
    assert "3" not in result


def test_ordinal_number_to_words():
    result = get_clean_transcript("1st place")
    assert "first" in result
    assert "1" not in result


def test_decimal_number_to_words():
    result = get_clean_transcript("2.5 mg")
    assert "2" not in result
    assert "two" in result


def test_filler_removal():
    result = get_clean_transcript("uh um hello", remove_non_lexical_tokens=True)
    assert "uh" not in result
    assert "um" not in result
    assert "hello" in result


def test_filler_kept_by_default():
    result = get_clean_transcript("uh hello")
    assert "uh" in result
    assert "hello" in result


def test_punctuation_removal():
    result = get_clean_transcript("Hello, world!")
    assert "," not in result
    assert "!" not in result
    assert "hello" in result
    assert "world" in result


def test_lowercasing():
    result = get_clean_transcript("Hello World")
    assert result == "hello world"


def test_hyphen_replaced_with_space():
    result = get_clean_transcript("twenty-three")
    assert "-" not in result
    assert "twenty" in result
    assert "three" in result


def test_empty_string():
    result = get_clean_transcript("")
    assert result == ""


def test_non_string_input():
    result = get_clean_transcript(None)
    assert result == ""


def test_non_string_number_input():
    result = get_clean_transcript(42)
    assert result == ""


def test_whitespace_normalization():
    result = get_clean_transcript("hello    world")
    assert result == "hello world"
