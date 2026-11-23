"""Send clean held-out observations through the parking prediction API."""

import argparse
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_validation import validate_parking_data
from src.feature_engineering import (
    chronological_split,
    create_forecast_targets,
    create_historical_features,
)
from src.feature_schema import FEATURE_COLUMNS, TARGET_COLUMN


DEFAULT_DATA_PATH = Path("data/dataset.csv")
DEFAULT_OUTPUT_PATH = Path("artifacts/heldout_api_metrics.json")
DEFAULT_API_URL = "http://127.0.0.1:8000"


def prepare_heldout_examples(data_path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Prepare clean examples exclusively from the held-out test split."""
    raw_df = pd.read_csv(data_path)
    validated_df = validate_parking_data(raw_df)
    historical_df = create_historical_features(validated_df)
    forecast_df = create_forecast_targets(historical_df)
    _, _, test_df = chronological_split(forecast_df)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return test_df.dropna(subset=required_columns).copy()


def build_api_request(row: pd.Series) -> dict:
    """Convert one held-out row to the FastAPI request payload."""
    return {
        "SystemCodeNumber": str(row["SystemCodeNumber"]),
        "Capacity": int(row["Capacity"]),
        "Occupancy": float(row["Occupancy"]),
        "hour": int(row["hour"]),
        "day_of_week": int(row["day_of_week"]),
        "is_weekend": int(row["is_weekend"]),
        "month": int(row["month"]),
        "occupancy_lag_1": float(row["occupancy_lag_1"]),
        "occupancy_lag_2": float(row["occupancy_lag_2"]),
        "occupancy_change_1": float(row["occupancy_change_1"]),
        "occupancy_rolling_mean_3": float(row["occupancy_rolling_mean_3"]),
    }


def calculate_metrics(targets: list[float], predictions: list[float]) -> dict:
    """Calculate held-out API prediction metrics."""
    if not targets:
        return {"MAE": None, "RMSE": None}

    y_true = np.asarray(targets, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def validate_heldout_api(
    data_path=DEFAULT_DATA_PATH,
    api_url=DEFAULT_API_URL,
    output_path=DEFAULT_OUTPUT_PATH,
    limit=None,
    transport=None,
) -> dict:
    """Send held-out observations to the API and save baseline evidence."""
    examples = prepare_heldout_examples(data_path)
    heldout_examples_available = len(examples)
    if limit is not None:
        examples = examples.head(limit)

    targets = []
    predictions = []
    successful_requests = 0
    failed_requests = 0

    with httpx.Client(base_url=api_url, timeout=30.0, transport=transport) as client:
        for _, row in examples.iterrows():
            try:
                response = client.post("/predict", json=build_api_request(row))
                if response.status_code != 200:
                    failed_requests += 1
                    continue

                predicted = float(response.json()["predicted_occupancy"])
                targets.append(float(row[TARGET_COLUMN]))
                predictions.append(predicted)
                successful_requests += 1
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                failed_requests += 1

    metrics = calculate_metrics(targets, predictions)
    evidence = {
        "dataset": str(data_path),
        "api_url": api_url,
        "heldout_examples_available": heldout_examples_available,
        "examples_requested": len(examples),
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "metrics": metrics,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    evidence = validate_heldout_api(
        data_path=args.data,
        api_url=args.api_url,
        output_path=args.output,
        limit=args.limit,
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
