"""FastAPI application for parking occupancy prediction."""

import os

import pandas as pd
import mlflow
import mlflow.sklearn
from functools import lru_cache
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field, model_validator

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
    "http://127.0.0.1:5001",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@lru_cache(maxsize=1)
def get_model():
    """Load and cache the registered MLflow model."""
    return mlflow.sklearn.load_model(MODEL_URI)

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

    @model_validator(mode="after")
    def validate_capacity(self):
        """Ensure observed occupancy values do not exceed capacity."""
        occupancy_fields = [
            "Occupancy",
            "occupancy_lag_1",
            "occupancy_lag_2",
            "occupancy_rolling_mean_3",
        ]

        for field in occupancy_fields:
            if getattr(self, field) > self.Capacity:
                raise ValueError(
                    f"{field} cannot exceed facility capacity"
                )

        return self

class ParkingPredictionResponse(BaseModel):
    facility: str
    forecast_horizon_minutes: int
    predicted_occupancy: float = Field(ge=0)
    predicted_available_spaces: float = Field(ge=0)
    capacity: int = Field(gt=0)
    model_uri: str
    model_version: str

@app.post(
    "/predict",
    response_model=ParkingPredictionResponse,
)
def predict(
    request: ParkingPredictionRequest,
    model=Depends(get_model),
):
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
        "model_version": MODEL_URI.rsplit("/", 1)[-1],
    }
