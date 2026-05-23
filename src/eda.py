"""Exploratory data analysis helpers for the Spotify project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path("outputs")


def plot_popularity_by_month(data: pd.DataFrame) -> None:
    """Create a boxplot of average popularity by month."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="year_month", y="avg_popularity")
    plt.xlabel("Month")
    plt.ylabel("Average popularity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "popularity_by_month.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(data: pd.DataFrame, features: list[str]) -> None:
    """Create a correlation heatmap for selected numeric features."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    correlation_matrix = data[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
