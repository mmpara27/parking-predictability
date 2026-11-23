import json

import httpx
import numpy as np
import pandas as pd

from scripts.validate_heldout_api import (
    build_api_request,
    calculate_metrics,
    validate_heldout_api,
)
from src.feature_schema import FEATURE_COLUMNS, TARGET_COLUMN


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
        TARGET_COLUMN: 75,
    })

    request = build_api_request(row)

    assert set(request) == set(FEATURE_COLUMNS)
    assert request["Capacity"] == 100
    assert request["Occupancy"] == 70


def test_calculate_metrics():
    metrics = calculate_metrics([10, 20, 30], [12, 18, 35])

    assert np.isclose(metrics["MAE"], 3.0)
    assert np.isclose(metrics["RMSE"], np.sqrt((4 + 4 + 25) / 3))


def test_validate_heldout_api_saves_metrics_and_request_counts(tmp_path, monkeypatch):
    heldout = pd.DataFrame([
        {
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
            TARGET_COLUMN: 75,
        },
        {
            "SystemCodeNumber": "BHMBCCMKT02",
            "Capacity": 100,
            "Occupancy": 50,
            "hour": 15,
            "day_of_week": 2,
            "is_weekend": 0,
            "month": 11,
            "occupancy_lag_1": 48,
            "occupancy_lag_2": 46,
            "occupancy_change_1": 2,
            "occupancy_rolling_mean_3": 48,
            TARGET_COLUMN: 55,
        },
    ])

    monkeypatch.setattr(
        "scripts.validate_heldout_api.prepare_heldout_examples",
        lambda data_path: heldout,
    )

    def handler(request):
        payload = json.loads(request.content)
        predicted = 77 if payload["SystemCodeNumber"] == "BHMBCCMKT01" else 40
        return httpx.Response(200, json={"predicted_occupancy": predicted})

    transport = httpx.MockTransport(handler)
    output_path = tmp_path / "heldout_api_metrics.json"

    evidence = validate_heldout_api(
        data_path="data/dataset.csv",
        api_url="http://test",
        output_path=output_path,
        transport=transport,
    )

    assert evidence["heldout_examples_available"] == 2
    assert evidence["examples_requested"] == 2
    assert evidence["successful_requests"] == 2
    assert evidence["failed_requests"] == 0
    assert np.isclose(evidence["metrics"]["MAE"], 8.5)
    assert np.isclose(evidence["metrics"]["RMSE"], np.sqrt(114.5))
    assert json.loads(output_path.read_text())["metrics"] == evidence["metrics"]


def test_validate_heldout_api_counts_failed_requests(tmp_path, monkeypatch):
    heldout = pd.DataFrame([{
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
        TARGET_COLUMN: 75,
    }])

    monkeypatch.setattr(
        "scripts.validate_heldout_api.prepare_heldout_examples",
        lambda data_path: heldout,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(422, json={"detail": "invalid"})
    )

    evidence = validate_heldout_api(
        data_path="data/dataset.csv",
        api_url="http://test",
        output_path=tmp_path / "heldout_api_metrics.json",
        transport=transport,
    )

    assert evidence["successful_requests"] == 0
    assert evidence["failed_requests"] == 1
