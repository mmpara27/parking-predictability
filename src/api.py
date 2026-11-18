"""FastAPI application for parking occupancy prediction."""

import os

import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.feature_schema import FEATURE_COLUMNS

app = FastAPI(
    title="Parking Availability Prediction API",
    version="1.0.0",
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/parking-occupancy-random-forest/1",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the registered model once when the application starts.
model = mlflow.sklearn.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
    }

class ParkingPredictionRequest(BaseModel):
    SystemCodeNumber: str
    Capacity: int = Field(gt=0)
    Occupancy: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    month: int = Field(ge=1, le=12)
    occupancy_lag_1: float = Field(ge=0)
    occupancy_lag_2: float = Field(ge=0)
    occupancy_change_1: float
    occupancy_rolling_mean_3: float = Field(ge=0)


@app.post("/predict")
def predict(request: ParkingPredictionRequest):
    # Convert the validated request to the training feature schema
    input_data = pd.DataFrame(
        [request.model_dump()]
    )[FEATURE_COLUMNS]

    # Predict occupancy approximately 30 minutes ahead
    raw_prediction = float(model.predict(input_data)[0])

    capacity = float(request.Capacity)
    predicted_occupancy = max(
        0.0,
        min(raw_prediction, capacity)
    )

    predicted_available_spaces = capacity - predicted_occupancy

    return {
        "facility": request.SystemCodeNumber,
        "forecast_horizon_minutes": 30,
        "predicted_occupancy": round(predicted_occupancy, 2),
        "predicted_available_spaces": round(
            predicted_available_spaces, 2
        ),
        "capacity": request.Capacity,
        "model_uri": MODEL_URI,
    }
