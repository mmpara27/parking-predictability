"""Airflow DAG for training parking occupancy prediction models."""

import hashlib
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from src.pipeline_tasks import (
    ingest_data,
    validate_data,
    preprocess_data,
    train_models,
    verify_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "pipeline_runs"


@dag(
    dag_id="parking_training",
    description="Train and validate parking occupancy prediction models.",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "parking-predictability",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["parking", "machine-learning", "training"],
)
def parking_training_pipeline():
    """Orchestrate the end-to-end model training workflow."""

    @task(task_id="ingest_data")
    def ingestion_task() -> str:
        context = get_current_context()
        run_id = context["run_id"]

        conf = context["dag_run"].conf or {}

        data_path = Path(
            conf.get("data_path", str(DATA_PATH))
        )

        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

        run_hash = hashlib.sha256(
            run_id.encode("utf-8")
        ).hexdigest()[:16]

        run_dir = RUNS_DIR / run_hash

        return ingest_data(
            str(data_path),
            str(run_dir),
        )

    @task(task_id="validate_data")
    def validation_task(raw_data_path: str) -> str:
        return validate_data(raw_data_path)

    @task(task_id="preprocess_data", multiple_outputs=False)
    def preprocessing_task(validated_data_path: str) -> dict:
        return preprocess_data(validated_data_path)

    @task(task_id="train_models")
    def training_task(split_paths: dict) -> str:
        return train_models(split_paths)

    @task(task_id="verify_artifacts")
    def verification_task(artifact_dir: str) -> str:
        return verify_artifacts(artifact_dir)

    raw_data = ingestion_task()
    validated_data = validation_task(raw_data)
    split_paths = preprocessing_task(validated_data)
    artifact_dir = training_task(split_paths)
    verification_task(artifact_dir)


parking_training_dag = parking_training_pipeline()
