"""Data preparation for the Spotify Hit Prediction project.

This script converts the raw daily Spotify chart data into a monthly
modelling dataset. The raw CSV is not included in the repository because the
file is large.
"""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/universal_top_spotify_songs.csv")
PROCESSED_DATA_PATH = Path("data/processed/spotify_monthly_2024.csv")

REQUIRED_COLUMNS = [
    "spotify_id",
    "name",
    "artists",
    "daily_rank",
    "country",
    "snapshot_date",
    "popularity",
    "is_explicit",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

TECHNICAL_FEATURES = [
    "key",
    "mode",
    "duration_ms",
    "is_explicit",
]


def check_raw_file_exists(raw_path: Path = RAW_DATA_PATH) -> None:
    """Check that the raw dataset file exists before running the project."""
    if not raw_path.exists():
        message = f"""
Raw data file was not found.

Please download the dataset from Kaggle:
https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated

Then place the CSV file here:
{raw_path}
"""
        raise FileNotFoundError(message)


def load_raw_data(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Spotify dataset."""
    check_raw_file_exists(raw_path)
    data = pd.read_csv(raw_path)
    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Check that the dataset contains the columns needed for this project."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(missing_columns)
        )


def create_monthly_dataset(
    data: pd.DataFrame,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> pd.DataFrame:
    """Create a monthly song-level dataset for modelling.

    The raw dataset contains one row per song per daily chart entry. This
    function aggregates the data so each row represents one song in one month.
    """
    validate_columns(data)

    data = data.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    data = data.dropna(subset=["snapshot_date", "spotify_id"])

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    data = data[(data["snapshot_date"] >= start) & (data["snapshot_date"] <= end)].copy()

    data["year_month"] = data["snapshot_date"].dt.to_period("M").astype(str)

    monthly_metrics = (
        data.groupby(["spotify_id", "year_month"])
        .agg(
            avg_popularity=("popularity", "mean"),
            max_popularity=("popularity", "max"),
            min_popularity=("popularity", "min"),
            avg_rank=("daily_rank", "mean"),
            best_rank=("daily_rank", "min"),
            worst_rank=("daily_rank", "max"),
            chart_days=("snapshot_date", "nunique"),
        )
        .reset_index()
    )

    static_columns = [
        "name",
        "artists",
        "country",
        *AUDIO_FEATURES,
        *TECHNICAL_FEATURES,
    ]

    static_features = (
        data.sort_values("snapshot_date")
        .groupby("spotify_id")[static_columns]
        .first()
        .reset_index()
    )

    monthly_data = monthly_metrics.merge(static_features, on="spotify_id", how="left")
    monthly_data["duration_min"] = monthly_data["duration_ms"] / (1000 * 60)
    monthly_data = monthly_data.drop(columns=["duration_ms"])
    monthly_data["is_explicit"] = monthly_data["is_explicit"].astype(int)

    return monthly_data


def save_monthly_dataset(
    monthly_data: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save the monthly dataset to the processed data folder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_data.to_csv(output_path, index=False)


def build_processed_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load the raw dataset, create the monthly dataset and save it."""
    raw_data = load_raw_data(raw_path)
    monthly_data = create_monthly_dataset(raw_data)
    save_monthly_dataset(monthly_data, output_path)

    print("Processed dataset created")
    print(f"Rows: {monthly_data.shape[0]}")
    print(f"Columns: {monthly_data.shape[1]}")
    print(f"Saved to: {output_path}")

    return monthly_data


if __name__ == "__main__":
    build_processed_dataset()
