"""Transcript cleaning and normalization for metric computation."""

import json
import logging
import re
from pathlib import Path
from typing import List

import jiwer
from num2words import num2words

log = logging.getLogger(__name__)

_NON_LEXICAL_TOKENS_PATH = Path(__file__).parent / "data" / "non_lexical_tokens.json"
with open(_NON_LEXICAL_TOKENS_PATH, "r") as _f:
    NON_LEXICAL_TOKENS: List[str] = json.load(_f)


def _convert_numbers_to_words(text: str) -> str:
    """Convert ordinal and cardinal numbers to words (en_GB)."""
    lang = "en_GB"
    # Pass 1: ordinals (1st, 2nd, 3rd, 4th, ...)
    text = re.sub(
        r"(\d+)(st|nd|rd|th)",
        lambda m: num2words(int(m.group(1)), to="ordinal", lang=lang),
        text,
    )
    # Pass 2: remaining cardinals (integers or decimals)
    text = re.sub(
        r"(\d+(\.\d+)?)",
        lambda m: num2words(m.group(1), lang=lang),
        text,
    )
    return text


def get_transformation(remove_non_lexical_tokens: bool):
    """Build a jiwer Compose transformation pipeline."""
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemoveSpecificWords(
                NON_LEXICAL_TOKENS if remove_non_lexical_tokens else []
            ),
            jiwer.SubstituteRegexes({r"-": " "}),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
        ]
    )
    return transformation


def get_clean_transcript(
    transcript: str, remove_non_lexical_tokens: bool = True
) -> str:
    """Clean and normalize a transcript string.

    Steps:
    1. Convert numbers to words (e.g. "3" → "three")
    2. Lowercase
    3. Optionally remove non-lexical tokens (uh, um, etc.)
    4. Replace hyphens with spaces
    5. Remove punctuation
    6. Normalize whitespace

    Args:
        transcript: Raw transcript text.
        remove_non_lexical_tokens: If True, remove filler words.

    Returns:
        Cleaned transcript string.
    """
    if not isinstance(transcript, str):
        log.warning(f"Non-string input received: {transcript}")
        return ""

    transcript_with_words = _convert_numbers_to_words(transcript)
    transformation = get_transformation(remove_non_lexical_tokens)
    transformed_text = transformation(transcript_with_words)[0]
    return " ".join(transformed_text)
