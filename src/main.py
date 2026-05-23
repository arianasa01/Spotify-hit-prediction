"""Main project runner.

Run this file from the root of the repository:

    python src/main.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_preparation import (  # noqa: E402
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    build_processed_dataset,
)
from src.modelling import run_all_models  # noqa: E402


def main() -> None:
    """Run the project workflow."""
    print("Spotify Hit Prediction project")
    print("--------------------------------")

    if not RAW_DATA_PATH.exists():
        print(
            f"""
Raw dataset not found.

Please download the dataset from Kaggle:
https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated

Then place it here:
{RAW_DATA_PATH}
"""
        )
        return

    monthly_data = build_processed_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    run_all_models(monthly_data)

    print("\nDone. Check the outputs folder for results.")


if __name__ == "__main__":
    main()
