"""Tests for drift simulation and schema-anomaly monitoring."""

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from scripts.simulate_drift import (
    create_valid_drifted_data,
    malformed_requests,
    validate_schema_anomalies,
)
from scripts.validate_heldout_api import build_api_request
from src.api import ParkingPredictionRequest, app, get_model
from src.feature_schema import FEATURE_COLUMNS


def test_create_valid_drifted_data_preserves_schema_and_capacity():
    reference = pd.DataFrame([{
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
    }])

    drifted = create_valid_drifted_data(reference)

    assert list(drifted.columns) == FEATURE_COLUMNS
    assert drifted["month"].iloc[0] in range(1, 13)
    assert drifted["hour"].iloc[0] in range(24)
    assert drifted["day_of_week"].iloc[0] in range(7)
    assert drifted["Occupancy"].iloc[0] <= 100
    assert drifted["occupancy_lag_1"].iloc[0] <= 100
    assert drifted["occupancy_change_1"].iloc[0] == (
        drifted["Occupancy"].iloc[0] - drifted["occupancy_lag_1"].iloc[0]
    )
    assert drifted["occupancy_rolling_mean_3"].iloc[0] == drifted[
        ["Occupancy", "occupancy_lag_1", "occupancy_lag_2"]
    ].iloc[0].mean()
    ParkingPredictionRequest.model_validate(build_api_request(drifted.iloc[0]))


def test_malformed_requests_are_separate_scenarios():
    valid = {
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

    scenarios = malformed_requests(valid)

    assert len(scenarios) == 6
    assert "missing_hour" in {item["name"] for item in scenarios}
    assert "occupancy_exceeds_capacity" in {item["name"] for item in scenarios}


def test_schema_anomalies_count_expected_422_responses(monkeypatch):
    valid = {
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

    monkeypatch.setitem(app.dependency_overrides, get_model, lambda: None)
    with TestClient(app) as client:
        transport = httpx.MockTransport(
            lambda request: client.post(
                request.url.path,
                content=request.content,
                headers={"Content-Type": request.headers["Content-Type"]},
            )
        )

        evidence = validate_schema_anomalies(
            "http://testserver",
            valid,
            transport=transport,
        )

    assert evidence["scenarios_requested"] == 6
    assert evidence["scenarios_passed"] == 6
    assert evidence["status_counts"] == {"422": 6}
    details = {
        entry["scenario"]: entry["response"]["detail"][0]
        for entry in evidence["scenarios"]
    }
    assert details["missing_hour"]["loc"] == ["body", "hour"]
    assert details["wrong_capacity_type"]["loc"] == ["body", "Capacity"]
    assert details["negative_occupancy"]["loc"] == ["body", "Occupancy"]
    assert details["zero_capacity"]["loc"] == ["body", "Capacity"]
    assert "Occupancy cannot exceed" in details["occupancy_exceeds_capacity"]["msg"]
    assert "occupancy_lag_1 cannot exceed" in details["lag_exceeds_capacity"]["msg"]


def test_build_api_request_uses_feature_schema():
    row = pd.Series({
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
    })

    payload = build_api_request(row)

    assert set(payload) == set(FEATURE_COLUMNS)
