import pytest
import re

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def clamp_star_rating(rating) -> int:
    try:
        val = float(rating)
        if val < 1:
            return 1
        elif val > 5:
            return 5
        return int(val)
    except (ValueError, TypeError):
        return None

def assign_binary_sentiment(star_rating: int):
    if star_rating is None or star_rating == 3:
        return None
    return 1 if star_rating > 3 else 0

def is_short_review(text: str) -> bool:
    if not text:
        return True
    words = len(text.split())
    chars = len(text)
    return words < 5 or chars < 30

def test_text_normalization():
    raw = "   This   IS A GREAT   product!!  \n\t  "
    expected = "this is a great product!!"
    assert normalize_text(raw) == expected

def test_star_rating_clamping():
    assert clamp_star_rating(5) == 5
    assert clamp_star_rating(0) == 1
    assert clamp_star_rating(-2.5) == 1
    assert clamp_star_rating(7.8) == 5
    assert clamp_star_rating("4") == 4
    assert clamp_star_rating("invalid") is None

def test_binary_sentiment_labeling():
    assert assign_binary_sentiment(5) == 1
    assert assign_binary_sentiment(4) == 1
    assert assign_binary_sentiment(3) is None
    assert assign_binary_sentiment(2) == 0
    assert assign_binary_sentiment(1) == 0

def test_short_review_detection():
    assert is_short_review("Good") is True
    assert is_short_review("Nice product") is True
    assert is_short_review("") is True
    long_review = "This item arrived on time and functions very well indeed."
    assert is_short_review(long_review) is False
