import math
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from src.data_validation import validate_parking_data
from src.feature_engineering import (
    create_historical_features,
    create_forecast_targets,
    chronological_split,
)
from src.feature_schema import FEATURE_COLUMNS


def test_registered_model_prediction():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    )

    model = mlflow.sklearn.load_model(
        "models:/parking-occupancy-random-forest/1"
    )

    raw_df = pd.read_csv("data/dataset.csv")
    validated_df = validate_parking_data(raw_df)
    historical_df = create_historical_features(validated_df)
    forecast_df = create_forecast_targets(historical_df)

    _, _, test_df = chronological_split(forecast_df)

    valid_test_df = test_df.dropna(subset=FEATURE_COLUMNS)
    assert not valid_test_df.empty, "No valid held-out samples"

    sample = valid_test_df[FEATURE_COLUMNS].head(1).copy()

    raw_prediction = float(model.predict(sample)[0])
    capacity = float(sample["Capacity"].iloc[0])

    assert capacity > 0
    assert math.isfinite(raw_prediction), "Prediction must be finite"

    predicted_occupancy = max(0, min(raw_prediction, capacity))
    predicted_available = capacity - predicted_occupancy

    assert 0 <= predicted_occupancy <= capacity
    assert 0 <= predicted_available <= capacity

    print(
        f"\nPASS: Registered model prediction successful"
        f"\nInput features: {len(sample.columns)}"
        f"\nPredicted occupancy: {predicted_occupancy:.2f}"
        f"\nPredicted available spaces: {predicted_available:.2f}"
    )
