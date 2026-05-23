"""Text analysis helpers for song titles.

This file keeps the title and sentiment analysis separate from the main
modelling workflow.
"""

from collections import Counter
import re

import pandas as pd

CUSTOM_STOPWORDS = {
    "feat",
    "remix",
    "version",
    "with",
    "the",
    "and",
    "for",
    "from",
}


def clean_title(title: str) -> list[str]:
    """Split a song title into cleaned lowercase words."""
    if pd.isna(title):
        return []

    words = re.findall(r"[A-Za-z]+", str(title).lower())
    words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 1]

    return words


def most_common_title_words(data: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the most common words used in song titles."""
    counter: Counter[str] = Counter()

    for title in data["name"].dropna().unique():
        counter.update(clean_title(title))

    common_words = counter.most_common(top_n)
    return pd.DataFrame(common_words, columns=["word", "count"])
