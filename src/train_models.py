"""Reproducible baseline and regression model training."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import mlflow
import mlflow.sklearn


from src.data_validation import validate_parking_data
from src.feature_engineering import (
    chronological_split,
    create_forecast_targets,
    create_historical_features,
)
from src.feature_schema import FEATURE_COLUMNS, TARGET_COLUMN


def build_preprocessing() -> ColumnTransformer:
    categorical_features = ["SystemCodeNumber"]
    numeric_features = [
        column for column in FEATURE_COLUMNS
        if column not in categorical_features
    ]

    return ColumnTransformer([
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric_features,
        ),
    ])


def clip_to_capacity(predictions: np.ndarray, capacity: pd.Series) -> np.ndarray:
    return np.clip(np.asarray(predictions), 0, capacity.to_numpy())


def evaluate(y_true: pd.Series, predictions: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def train(data_path: Path, artifact_dir: Path) -> dict:
    """Run the complete training workflow from a CSV file."""
    raw_df = pd.read_csv(data_path)
    validated_df = validate_parking_data(raw_df)
    historical_df = create_historical_features(validated_df)
    forecast_df = create_forecast_targets(historical_df)
    train_df, validation_df, test_df = chronological_split(forecast_df)

    print("Original observations:", len(raw_df))
    print("Validated observations:", len(validated_df))
    print("Valid forecasting examples:", len(forecast_df))
    print("Held-out test examples:", len(test_df))

    return train_from_splits(
        train_df, validation_df, artifact_dir
    )


def train_from_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifact_dir: Path,
) -> dict:
    """Train and evaluate models using prepared data splits."""
   
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("parking-occupancy-prediction")

    
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    train_model_df = train_df.dropna(subset=required_columns).copy()
    validation_model_df = validation_df.dropna(subset=required_columns).copy()

    if train_model_df.empty:
        raise ValueError("No valid training examples remain.")

    if validation_model_df.empty:
        raise ValueError("No valid validation examples remain.")

    X_train = train_model_df[FEATURE_COLUMNS]
    y_train = train_model_df[TARGET_COLUMN]
    X_validation = validation_model_df[FEATURE_COLUMNS]
    y_validation = validation_model_df[TARGET_COLUMN]

    linear_model = Pipeline([
        ("preprocessing", build_preprocessing()),
        ("model", LinearRegression()),
    ])

    random_forest_model = Pipeline([
        ("preprocessing", build_preprocessing()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        ),
    ])

    linear_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    baseline_predictions = clip_to_capacity(
        X_validation["Occupancy"].to_numpy(),
        X_validation["Capacity"],
    )
    linear_predictions = clip_to_capacity(
        linear_model.predict(X_validation),
        X_validation["Capacity"],
    )
    random_forest_predictions = clip_to_capacity(
        random_forest_model.predict(X_validation),
        X_validation["Capacity"],
    )

    metrics = {
        "Persistence Baseline": evaluate(y_validation, baseline_predictions),
        "Linear Regression": evaluate(y_validation, linear_predictions),
        "Random Forest": evaluate(y_validation, random_forest_predictions),
    }

    # Track the persistence baseline in MLflow
    with mlflow.start_run(run_name="Persistence Baseline"):
        mlflow.log_param("model_type", "persistence")
        mlflow.log_param("forecast_horizon_minutes", 30)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        mlflow.log_metric("validation_mae", metrics["Persistence Baseline"]["MAE"])
        mlflow.log_metric("validation_rmse", metrics["Persistence Baseline"]["RMSE"])

    # Track Linear Regression in MLflow
    with mlflow.start_run(run_name="Linear Regression"):
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("forecast_horizon_minutes", 30)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        mlflow.log_metric(
            "validation_mae",
            metrics["Linear Regression"]["MAE"]
        )
        mlflow.log_metric(
            "validation_rmse",
            metrics["Linear Regression"]["RMSE"]
        )

        mlflow.sklearn.log_model(
            sk_model=linear_model,
            name="linear_regression",
            skops_trusted_types=["numpy.dtype"]
        )

    # Track Random Forest in MLflow
    with mlflow.start_run(run_name="Random Forest"):
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("forecast_horizon_minutes", 30)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        mlflow.log_params({
            "n_estimators": 200,
            "random_state": 42,
            "min_samples_leaf": 2,
            "n_jobs": -1,
        })

        mlflow.log_metric(
            "validation_mae",
            metrics["Random Forest"]["MAE"]
        )
        mlflow.log_metric(
            "validation_rmse",
            metrics["Random Forest"]["RMSE"]
        )

        mlflow.sklearn.log_model(
            sk_model=random_forest_model,
            name="random_forest",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(linear_model, artifact_dir / "linear_regression.joblib")
    joblib.dump(random_forest_model, artifact_dir / "random_forest.joblib")

    with (artifact_dir / "validation_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    print("Training examples:", len(train_model_df))
    print("Validation examples:", len(validation_model_df))
    print("\nValidation metrics:")
    print(pd.DataFrame(metrics).T.sort_values("RMSE").to_string())

    return metrics


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train baseline and regression parking occupancy models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "dataset.csv",
        help="Path to the parking dataset.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=project_root / "artifacts",
        help="Directory for trained model and metrics artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments.data, arguments.artifacts)
