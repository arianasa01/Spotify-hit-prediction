"""
Text and sentiment analysis helpers for Spotify song titles.

This part is kept separate because the main modelling workflow does not depend on it.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import nltk
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud


OUTPUT_DIR = Path("outputs")


def download_nltk_resources() -> None:
    """Download the NLTK resources needed for sentiment analysis."""
    nltk.download("stopwords")
    nltk.download("vader_lexicon")


def build_stopwords() -> set:
    """Build a combined stopword list for song title analysis."""
    custom_words = {
        "song",
        "music",
        "track",
        "top",
        "hit",
        "feat",
        "version",
        "remix",
        "ft",
        "featuring",
        "with",
        "la",
        "de",
        "que",
        "na",
        "el",
        "en",
        "mi",
    }

    return STOPWORDS.union(custom_words)


def generate_wordcloud(titles: Iterable[str], output_name: str, output_dir: Path = OUTPUT_DIR) -> None:
    """Generate a word cloud from song titles."""
    output_dir.mkdir(parents=True, exist_ok=True)

    text = " ".join(str(title) for title in titles if pd.notna(title))
    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=build_stopwords(),
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / output_name, dpi=150)
    plt.close()


def create_country_wordcloud(data: pd.DataFrame, country_code: str, output_dir: Path = OUTPUT_DIR) -> None:
    """Create a song title word cloud for one country."""
    country_data = data[data["country"] == country_code]
    titles = country_data["name"].dropna().unique()
    generate_wordcloud(titles, f"wordcloud_{country_code}.png", output_dir)


def score_title_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment scores for song titles."""
    download_nltk_resources()

    analyser = SentimentIntensityAnalyzer()

    scored_data = data[["spotify_id", "name", "country", "is_explicit"]].copy()
    scored_data["name"] = scored_data["name"].fillna("").astype(str)
    scored_data = scored_data.drop_duplicates(subset=["spotify_id", "name"])

    scored_data["title_sentiment"] = scored_data["name"].apply(lambda value: analyser.polarity_scores(value)["compound"])

    return scored_data
