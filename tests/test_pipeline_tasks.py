"""Tests for reusable parking training pipeline tasks."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline_tasks import (
    ingest_data,
    validate_data,
    preprocess_data,
    train_models,
    verify_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"


class PipelineTasksTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ingest_valid_dataset(self):
        result = ingest_data(
            str(DATA_PATH),
            str(self.run_dir),
        )

        self.assertTrue(Path(result).is_file())
        self.assertFalse(pd.read_pickle(result).empty)

    def test_ingest_missing_dataset(self):
        with self.assertRaises(FileNotFoundError):
            ingest_data(
                str(self.run_dir / "missing.csv"),
                str(self.run_dir),
            )

    def test_ingest_empty_dataset(self):
        empty_path = self.run_dir / "empty.csv"
        pd.DataFrame(columns=["Capacity"]).to_csv(
            empty_path, index=False
        )

        with self.assertRaises(ValueError):
            ingest_data(str(empty_path), str(self.run_dir))

    def test_validate_valid_dataset(self):
        raw_path = ingest_data(
            str(DATA_PATH),
            str(self.run_dir),
        )

        result = validate_data(raw_path)
        validated = pd.read_pickle(result)

        self.assertFalse(validated.empty)
        self.assertTrue(
            (validated["Occupancy"] <= validated["Capacity"]).all()
        )
        self.assertTrue((validated["Occupancy"] >= 0).all())

    def test_validate_missing_columns(self):
        invalid_path = self.run_dir / "invalid.pkl"

        pd.DataFrame({
            "Occupancy": [10, 20]
        }).to_pickle(invalid_path)

        with self.assertRaises(ValueError):
            validate_data(str(invalid_path))

    def test_preprocess_creates_splits(self):
        raw_path = ingest_data(
            str(DATA_PATH),
            str(self.run_dir),
        )

        validated_path = validate_data(raw_path)
        splits = preprocess_data(validated_path)

        self.assertEqual(
            set(splits),
            {"train", "validation", "test"},
        )

        for path in splits.values():
            self.assertTrue(Path(path).is_file())
            self.assertFalse(pd.read_pickle(path).empty)

        train = pd.read_pickle(splits["train"])
        validation = pd.read_pickle(splits["validation"])
        test = pd.read_pickle(splits["test"])

        self.assertLess(
            train["TargetTimestamp"].max(),
            validation["LastUpdated"].min(),
        )

        self.assertLess(
            validation["TargetTimestamp"].max(),
            test["LastUpdated"].min(),
        )

    def test_verify_missing_artifacts(self):
        with self.assertRaises(FileNotFoundError):
            verify_artifacts(str(self.run_dir))

    def test_verify_empty_artifact(self):
        (self.run_dir / "linear_regression.joblib").touch()

        with self.assertRaises(ValueError):
            verify_artifacts(str(self.run_dir))

    def test_verify_invalid_metrics(self):
        self.create_mock_artifacts()

        metrics = {
            "Unexpected Model": {
                "MAE": 1.0,
                "RMSE": 2.0,
            }
        }

        metrics_path = self.run_dir / "validation_metrics.json"
        metrics_path.write_text(json.dumps(metrics))

        with self.assertRaises(ValueError):
            verify_artifacts(str(self.run_dir))

    def create_mock_artifacts(self):
        for name in [
            "linear_regression.joblib",
            "random_forest.joblib",
        ]:
            (self.run_dir / name).write_bytes(b"test artifact")

    def test_verify_valid_artifact_structure(self):
        self.create_mock_artifacts()

        metrics = {
            model: {"MAE": 1.0, "RMSE": 2.0}
            for model in [
                "Persistence Baseline",
                "Linear Regression",
                "Random Forest",
            ]
        }

        metrics_path = self.run_dir / "validation_metrics.json"
        metrics_path.write_text(json.dumps(metrics))

        result = verify_artifacts(str(self.run_dir))

        self.assertEqual(result, str(self.run_dir))

    def test_end_to_end_pipeline(self):
        raw_path = ingest_data(
            str(DATA_PATH),
            str(self.run_dir),
        )

        validated_path = validate_data(raw_path)
        splits = preprocess_data(validated_path)
        artifact_dir = train_models(splits)

        result = verify_artifacts(artifact_dir)

        self.assertTrue(Path(result).is_dir())

        metrics_path = Path(result) / "validation_metrics.json"
        metrics = json.loads(metrics_path.read_text())

        self.assertAlmostEqual(
            metrics["Random Forest"]["MAE"],
            18.775141,
            places=5,
        )

        self.assertAlmostEqual(
            metrics["Random Forest"]["RMSE"],
            33.672739,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
