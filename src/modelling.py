"""
Modelling code for the Spotify Hit Prediction project.

The workflow includes:
1. Regression models for average monthly popularity
2. A classification model for long-term chart presence
"""

from pathlib import Path

import pandas as pd

from scipy.stats import randint, uniform

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


PROCESSED_DATA_PATH = Path("data/processed/spotify_monthly_2024.csv")
OUTPUT_DIR = Path("outputs")

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

NUMERIC_FEATURES = [
    *AUDIO_FEATURES,
    "duration_min",
    "is_explicit",
]

CATEGORICAL_FEATURES = [
    "key",
    "mode",
]


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed monthly dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run data_preparation.py first."
        )

    return pd.read_csv(path)


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps for numeric and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def grouped_train_test_split(data: pd.DataFrame, test_size: float = 0.2):
    """
    Split the data while keeping the same song out of both train and test sets.

    This is safer than a normal random row split because the same song can appear
    in multiple monthly rows.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(data, groups=data["spotify_id"]))

    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()

    return train_data, test_data


def compare_regression_models(data: pd.DataFrame) -> pd.DataFrame:
    """Train and compare regression models for average monthly popularity."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    target = "avg_popularity"

    model_data = data.dropna(subset=features + [target, "spotify_id"]).copy()

    train_data, test_data = grouped_train_test_split(model_data)

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    preprocessor = create_preprocessor()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "K Nearest Neighbours": KNeighborsRegressor(),
        "Support Vector Regressor": SVR(),
        "Stochastic Gradient Descent": SGDRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Dummy Mean": DummyRegressor(strategy="mean"),
        "Dummy Median": DummyRegressor(strategy="median"),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "model": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

    results_df = pd.DataFrame(results).sort_values("rmse")
    return results_df


def create_song_level_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly rows into one row per song for the long-term hit classifier.

    This avoids training and testing on different rows of the same song.
    """
    song_months = data.groupby("spotify_id")["year_month"].nunique().rename("months_in_chart")

    first_values = (
        data.sort_values("year_month")
        .groupby("spotify_id")
        .first()
        .reset_index()
    )

    song_data = first_values.merge(song_months, on="spotify_id", how="left")
    song_data["long_term_hit"] = (song_data["months_in_chart"] > 2).astype(int)

    return song_data


def engineer_classification_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features for the classification model."""
    model_data = data.copy()

    model_data["energy_loudness"] = model_data["energy"] * model_data["loudness"]
    model_data["dance_energy"] = model_data["danceability"] * model_data["energy"]
    model_data["acoust_instru"] = model_data["acousticness"] * model_data["instrumentalness"]

    return model_data


def train_long_term_classifier(data: pd.DataFrame):
    """Train a Gradient Boosting classifier for long-term chart presence."""
    song_data = create_song_level_dataset(data)
    song_data = engineer_classification_features(song_data)

    feature_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_min",
        "is_explicit",
        "energy_loudness",
        "dance_energy",
        "acoust_instru",
    ]

    target = "long_term_hit"

    model_data = song_data.dropna(subset=feature_columns + [target]).copy()

    X = model_data[feature_columns]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )

    param_distributions = {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": uniform(0.03, 0.17),
        "model__subsample": uniform(0.6, 0.4),
        "model__min_samples_split": randint(2, 10),
        "model__min_samples_leaf": randint(1, 5),
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=25,
        cv=3,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_short_term", "actual_long_term"],
        columns=["predicted_short_term", "predicted_long_term"],
    )

    summary = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "best_parameters": search.best_params_,
    }

    return best_model, report_df, cm_df, summary


def run_all_models(data: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """Run the regression and classification workflows and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    regression_results = compare_regression_models(data)
    regression_results.to_csv(output_dir / "regression_results.csv", index=False)

    best_model, class_report, confusion, summary = train_long_term_classifier(data)
    class_report.to_csv(output_dir / "classification_report.csv")
    confusion.to_csv(output_dir / "confusion_matrix.csv")

    summary_df = pd.DataFrame(
        [
            {
                "accuracy": summary["accuracy"],
                "f1_macro": summary["f1_macro"],
                "best_parameters": str(summary["best_parameters"]),
            }
        ]
    )
    summary_df.to_csv(output_dir / "classification_summary.csv", index=False)

    print("\nRegression results")
    print(regression_results)

    print("\nClassification summary")
    print(summary_df)


if __name__ == "__main__":
    monthly_data = load_processed_data()
    run_all_models(monthly_data)
