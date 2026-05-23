"""Optional text analysis helpers for song titles.

This part was included in the original coursework, but the main public workflow
focuses more on data preparation and modelling.
"""

import re
from collections import Counter

import pandas as pd

BASIC_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "to",
    "in",
    "with",
    "feat",
    "featuring",
    "remix",
    "version",
}


def clean_title(title: str) -> list[str]:
    """Tokenise a song title and remove simple stopwords."""
    if pd.isna(title):
        return []

    title = str(title).lower()
    title = re.sub(r"[^a-zA-Z0-9\s]", " ", title)
    words = [word for word in title.split() if word not in BASIC_STOPWORDS]
    return words


def most_common_title_words(data: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the most common words used in song titles."""
    counter: Counter[str] = Counter()

    for title in data["name"].dropna().unique():
        counter.update(clean_title(title))

    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])
