"""Tests for the Airflow parking training DAG."""

import unittest
from pathlib import Path

from airflow.models import DagBag


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingDAGTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dag_bag = DagBag(
            dag_folder=str(PROJECT_ROOT / "dags"),
            include_examples=False,
        )

    def test_no_import_errors(self):
        self.assertEqual(self.dag_bag.import_errors, {})

    def test_dag_exists(self):
        dag = self.dag_bag.get_dag("parking_training")
        self.assertIsNotNone(dag)

    def test_expected_tasks(self):
        dag = self.dag_bag.get_dag("parking_training")

        expected_tasks = {
            "ingest_data",
            "validate_data",
            "preprocess_data",
            "train_models",
            "verify_artifacts",
        }

        self.assertEqual(set(dag.task_ids), expected_tasks)

    def test_task_dependencies(self):
        dag = self.dag_bag.get_dag("parking_training")

        expected_dependencies = {
            "ingest_data": {"validate_data"},
            "validate_data": {"preprocess_data"},
            "preprocess_data": {"train_models"},
            "train_models": {"verify_artifacts"},
            "verify_artifacts": set(),
        }

        for task_id, downstream in expected_dependencies.items():
            self.assertEqual(
                dag.get_task(task_id).downstream_task_ids,
                downstream,
            )

    def test_dag_configuration(self):
        dag = self.dag_bag.get_dag("parking_training")

        self.assertFalse(dag.catchup)
        self.assertEqual(dag.max_active_runs, 1)
        self.assertIsNone(dag.schedule_interval)

    def test_task_retries(self):
        dag = self.dag_bag.get_dag("parking_training")

        for task in dag.tasks:
            self.assertEqual(task.retries, 1)


if __name__ == "__main__":
    unittest.main()
