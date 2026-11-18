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

# Connect to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered Random Forest model
model = mlflow.sklearn.load_model(
    "models:/parking-occupancy-random-forest/1"
)

# Prepare a real example from the held-out test split
raw_df = pd.read_csv("data/dataset.csv")
validated_df = validate_parking_data(raw_df)
historical_df = create_historical_features(validated_df)
forecast_df = create_forecast_targets(historical_df)

_, _, test_df = chronological_split(forecast_df)

valid_test_df = test_df.dropna(subset=FEATURE_COLUMNS)
sample = valid_test_df[FEATURE_COLUMNS].head(1).copy()

# Generate a prediction
raw_prediction = float(model.predict(sample)[0])

capacity = float(sample["Capacity"].iloc[0])
predicted_occupancy = max(0, min(raw_prediction, capacity))
predicted_available = capacity - predicted_occupancy

print("PASS: Registered model prediction successful")
print("Input features:", len(sample.columns))
print("Predicted occupancy:", round(predicted_occupancy, 2))
print("Predicted available spaces:", round(predicted_available, 2))
