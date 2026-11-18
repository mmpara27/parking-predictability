"""Reusable task functions for the parking prediction pipeline."""

import json
import logging
import math
from pathlib import Path

import pandas as pd

from src.data_validation import validate_parking_data
from src.feature_engineering import (
    chronological_split,
    create_forecast_targets,
    create_historical_features,
)
from src.train_models import train_from_splits


logger = logging.getLogger(__name__)


def ingest_data(data_path: str, run_dir: str) -> str:
    """Load the source dataset and save a run-specific copy."""
    source = Path(data_path)
    output_dir = Path(run_dir)

    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    df = pd.read_csv(source)

    if df.empty:
        raise ValueError("The source dataset is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "raw_data.pkl"
    df.to_pickle(output_path)

    logger.info("Ingested %s observations.", len(df))

    return str(output_path)


def validate_data(raw_data_path: str) -> str:
    """Validate the ingested parking dataset."""
    input_path = Path(raw_data_path)
    raw_df = pd.read_pickle(input_path)

    validated_df = validate_parking_data(raw_df)

    if validated_df.empty:
        raise ValueError("No observations remain after validation.")

    output_path = input_path.parent / "validated_data.pkl"
    validated_df.to_pickle(output_path)

    logger.info(
        "Validated %s of %s observations.",
        len(validated_df),
        len(raw_df),
    )

    return str(output_path)


def preprocess_data(validated_data_path: str) -> dict:
    """Create forecasting features, targets, and chronological splits."""
    input_path = Path(validated_data_path)
    validated_df = pd.read_pickle(input_path)

    historical_df = create_historical_features(validated_df)
    forecast_df = create_forecast_targets(historical_df)

    if forecast_df.empty:
        raise ValueError("No valid forecasting examples were created.")

    train_df, validation_df, test_df = chronological_split(
        forecast_df
    )

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    split_paths = {}

    for name, df in splits.items():
        if df.empty:
            raise ValueError(f"The {name} split is empty.")

        output_path = input_path.parent / f"{name}.pkl"
        df.to_pickle(output_path)

        split_paths[name] = str(output_path)

        logger.info("%s split: %s observations.", name, len(df))

    summary = {
        "forecasting_examples": len(forecast_df),
        "train_examples": len(train_df),
        "validation_examples": len(validation_df),
        "test_examples": len(test_df),
    }

    summary_path = input_path.parent / "preprocessing_summary.json"

    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)

    return split_paths


def train_models(split_paths: dict) -> str:
    """Train models from the prepared training and validation splits."""
    train_path = Path(split_paths["train"])
    validation_path = Path(split_paths["validation"])

    train_df = pd.read_pickle(train_path)
    validation_df = pd.read_pickle(validation_path)

    artifact_dir = train_path.parent / "models"

    metrics = train_from_splits(
        train_df,
        validation_df,
        artifact_dir,
    )

    logger.info("Training completed. Metrics: %s", metrics)

    return str(artifact_dir)


def verify_artifacts(artifact_dir: str) -> str:
    """Verify that expected models and valid metrics were generated."""
    output_dir = Path(artifact_dir)

    expected_files = [
        "linear_regression.joblib",
        "random_forest.joblib",
        "validation_metrics.json",
    ]

    for filename in expected_files:
        file_path = output_dir / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Missing training artifact: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise ValueError(f"Empty training artifact: {file_path}")

    metrics_path = output_dir / "validation_metrics.json"

    with metrics_path.open() as file:
        metrics = json.load(file)

    expected_models = {
        "Persistence Baseline",
        "Linear Regression",
        "Random Forest",
    }

    if set(metrics) != expected_models:
        raise ValueError("Unexpected models in validation metrics.")

    for model_name, model_metrics in metrics.items():
        for metric_name in ["MAE", "RMSE"]:
            value = model_metrics.get(metric_name)

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Missing or invalid {metric_name} for {model_name}"
                )

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid {metric_name} for {model_name}: {value}"
                )

    logger.info("All expected training artifacts verified.")

    return str(output_dir)
