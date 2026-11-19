"""Unit tests for the FastAPI prediction endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_model
from src.feature_schema import FEATURE_COLUMNS


class FakeModel:
    """Mock model that returns a predetermined occupancy prediction."""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, data):
        assert list(data.columns) == FEATURE_COLUMNS
        return [self.prediction]


@pytest.fixture
def client():
    app.dependency_overrides[get_model] = lambda: FakeModel(75.5)

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def valid_request():
    return {
        "SystemCodeNumber": "BHMBCCMKT01",
        "Capacity": 100,
        "Occupancy": 70,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "month": 11,
        "occupancy_lag_1": 68,
        "occupancy_lag_2": 65,
        "occupancy_change_1": 2,
        "occupancy_rolling_mean_3": 67.67,
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_success(client, valid_request):
    response = client.post("/predict", json=valid_request)

    assert response.status_code == 200

    result = response.json()

    assert result["facility"] == "BHMBCCMKT01"
    assert result["forecast_horizon_minutes"] == 30
    assert result["predicted_occupancy"] == 75.5
    assert result["predicted_available_spaces"] == 24.5
    assert result["capacity"] == 100
    assert result["model_version"] == "1"


def test_invalid_capacity(client, valid_request):
    valid_request["Capacity"] = 0

    response = client.post("/predict", json=valid_request)

    assert response.status_code == 422


def test_occupancy_exceeds_capacity(client, valid_request):
    valid_request["Occupancy"] = 120

    response = client.post("/predict", json=valid_request)

    assert response.status_code == 422


def test_missing_required_field(client, valid_request):
    del valid_request["hour"]

    response = client.post("/predict", json=valid_request)

    assert response.status_code == 422


def test_prediction_clipped_to_capacity(client, valid_request):
    app.dependency_overrides[get_model] = lambda: FakeModel(150)

    response = client.post("/predict", json=valid_request)

    assert response.status_code == 200
    assert response.json()["predicted_occupancy"] == 100
    assert response.json()["predicted_available_spaces"] == 0


def test_negative_prediction_clipped_to_zero(client, valid_request):
    app.dependency_overrides[get_model] = lambda: FakeModel(-20)

    response = client.post("/predict", json=valid_request)

    assert response.status_code == 200
    assert response.json()["predicted_occupancy"] == 0
    assert response.json()["predicted_available_spaces"] == 100