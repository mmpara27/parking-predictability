"""Tests for reproducible parking occupancy model training."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_models import (
    build_preprocessing,
    clip_to_capacity,
    evaluate,
)


class TrainModelsTests(unittest.TestCase):
    def test_predictions_are_clipped_to_capacity(self):
        predictions = np.array([-10, 25, 150])
        capacity = pd.Series([100, 100, 100])

        clipped = clip_to_capacity(predictions, capacity)

        np.testing.assert_array_equal(clipped, np.array([0, 25, 100]))

    def test_mae_and_rmse_are_calculated(self):
        y_true = pd.Series([10, 20, 30])
        predictions = np.array([12, 20, 27])

        metrics = evaluate(y_true, predictions)

        self.assertAlmostEqual(metrics["MAE"], 5 / 3)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(13 / 3))

    def test_preprocessing_pipeline_can_transform_model_features(self):
        data = pd.DataFrame({
            "SystemCodeNumber": ["A", "B", "A", "C"],
            "Capacity": [100, 200, 150, 300],
            "Occupancy": [20, 30, 40, 50],
            "hour": [8, 9, 10, 11],
            "day_of_week": [0, 1, 2, 3],
            "is_weekend": [0, 0, 0, 0],
            "month": [10, 10, 10, 11],
            "occupancy_lag_1": [18, 28, 38, 48],
            "occupancy_lag_2": [16, 26, 36, 46],
            "occupancy_change_1": [2, 2, 2, 2],
            "occupancy_rolling_mean_3": [18, 28, 38, 48],
        })

        transformed = build_preprocessing().fit_transform(data)

        self.assertEqual(transformed.shape[0], 4)
        self.assertGreater(transformed.shape[1], len(data.columns) - 1)

    def test_saved_validation_metrics_have_expected_models(self):
        metrics_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "validation_metrics.json"
        )

        if not metrics_path.exists():
            self.skipTest("Run python -m src.train_models first")

        with metrics_path.open() as file:
            metrics = json.load(file)

        self.assertEqual(
            set(metrics),
            {
                "Persistence Baseline",
                "Linear Regression",
                "Random Forest",
            },
        )


if __name__ == "__main__":
    unittest.main()
