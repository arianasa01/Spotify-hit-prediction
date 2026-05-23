"""
Exploratory analysis helpers for the Spotify Hit Prediction project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


OUTPUT_DIR = Path("outputs")


def plot_popularity_by_month(data: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """Create a boxplot of average popularity by month."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="year_month", y="avg_popularity")
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Average popularity")
    plt.tight_layout()
    plt.savefig(output_dir / "popularity_by_month.png", dpi=150)
    plt.close()


def plot_audio_correlation(data: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """Create a correlation heatmap for audio features and popularity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "avg_popularity",
    ]

    corr = data[columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", center=0)
    plt.tight_layout()
    plt.savefig(output_dir / "audio_feature_correlation.png", dpi=150)
    plt.close()


def create_country_feature_map(raw_data: pd.DataFrame, feature: str, output_dir: Path = OUTPUT_DIR) -> None:
    """Create a choropleth map for a selected audio feature."""
    output_dir.mkdir(parents=True, exist_ok=True)

    country_data = (
        raw_data.dropna(subset=["country"])
        .groupby("country")[feature]
        .mean()
        .reset_index()
    )

    fig = px.choropleth(
        country_data,
        locations="country",
        locationmode="ISO-3",
        color=feature,
        title=f"Average {feature} by country",
    )

    fig.write_html(output_dir / f"{feature}_country_map.html")
