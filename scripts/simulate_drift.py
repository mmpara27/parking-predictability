"""Simulate feature drift and malformed FastAPI requests."""

import argparse
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from scripts.validate_heldout_api import (
    DEFAULT_API_URL,
    DEFAULT_DATA_PATH,
    build_api_request,
    prepare_heldout_examples,
)
from src.feature_schema import FEATURE_COLUMNS

DEFAULT_DRIFT_OUTPUT = Path("artifacts/evidently_drift_report.html")
DEFAULT_DRIFT_METRICS = Path("artifacts/drift_metrics.json")
DEFAULT_SCHEMA_LOG = Path("artifacts/schema_validation_log.json")


def create_valid_drifted_data(reference: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """Create physically valid distribution shifts in numerical features."""
    drifted = reference[FEATURE_COLUMNS].copy()
    rng = np.random.default_rng(random_seed)

    occupancy_columns = ["Occupancy", "occupancy_lag_1", "occupancy_lag_2"]
    capacity = drifted["Capacity"].astype(float)
    for column in occupancy_columns:
        shifted = drifted[column] * 1.35 + rng.normal(25, 8, len(drifted))
        drifted[column] = shifted.clip(lower=0, upper=capacity)
    drifted["occupancy_change_1"] = (
        drifted["Occupancy"] - drifted["occupancy_lag_1"]
    )
    drifted["occupancy_rolling_mean_3"] = drifted[occupancy_columns].mean(axis=1)

    # Shift valid calendar inputs while preserving their valid ranges.
    drifted["month"] = ((drifted["month"].astype(int) + 3 - 1) % 12 + 1)
    drifted["day_of_week"] = ((drifted["day_of_week"].astype(int) + 2) % 7)
    drifted["is_weekend"] = drifted["day_of_week"].isin([5, 6]).astype(int)
    drifted["hour"] = ((drifted["hour"].astype(int) + 6) % 24)
    return drifted


def generate_evidently_drift_report(
    reference: pd.DataFrame,
    drifted: pd.DataFrame,
    html_output: Path,
) -> dict:
    """Generate an Evidently Data Drift report and machine-readable summary."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    evaluation = report.run(current_data=drifted, reference_data=reference)

    html_output = Path(html_output)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(str(html_output))

    return {
        "evidently_version": _evidently_version(),
        "reference_examples": len(reference),
        "drifted_examples": len(drifted),
        "report_data": json.loads(evaluation.json()),
    }


def _evidently_version() -> str:
    import evidently

    return getattr(evidently, "__version__", "unknown")


def malformed_requests(valid_request: dict) -> list[dict]:
    """Build representative malformed API requests."""
    missing = dict(valid_request)
    del missing["hour"]

    wrong_type = dict(valid_request)
    wrong_type["Capacity"] = "not-a-number"

    negative = dict(valid_request)
    negative["Occupancy"] = -1

    zero_capacity = dict(valid_request)
    zero_capacity["Capacity"] = 0

    exceeds_capacity = dict(valid_request)
    exceeds_capacity["Occupancy"] = exceeds_capacity["Capacity"] + 1

    lag_exceeds_capacity = dict(valid_request)
    lag_exceeds_capacity["occupancy_lag_1"] = lag_exceeds_capacity["Capacity"] + 1

    return [
        {"name": "missing_hour", "payload": missing},
        {"name": "wrong_capacity_type", "payload": wrong_type},
        {"name": "negative_occupancy", "payload": negative},
        {"name": "zero_capacity", "payload": zero_capacity},
        {"name": "occupancy_exceeds_capacity", "payload": exceeds_capacity},
        {"name": "lag_exceeds_capacity", "payload": lag_exceeds_capacity},
    ]


def validate_schema_anomalies(api_url: str, valid_request: dict, transport=None) -> dict:
    """Send malformed requests separately and log their validation responses."""
    entries = []
    status_counts = {}

    with httpx.Client(base_url=api_url, timeout=30.0, transport=transport) as client:
        for scenario in malformed_requests(valid_request):
            try:
                response = client.post("/predict", json=scenario["payload"])
                status = response.status_code
                try:
                    response_body = response.json()
                except json.JSONDecodeError:
                    response_body = response.text
                has_detail = isinstance(response_body, dict) and bool(response_body.get("detail"))
                status_counts[str(status)] = status_counts.get(str(status), 0) + 1
                entries.append({
                    "scenario": scenario["name"],
                    "status_code": status,
                    "expected_status": 422,
                    "passed": status == 422 and has_detail,
                    "response": response_body,
                })
            except httpx.RequestError as exc:
                status_counts["request_error"] = status_counts.get("request_error", 0) + 1
                entries.append({
                    "scenario": scenario["name"],
                    "status_code": None,
                    "expected_status": 422,
                    "passed": False,
                    "error": str(exc),
                })

    return {
        "api_url": api_url,
        "scenarios_requested": len(entries),
        "status_counts": status_counts,
        "scenarios_passed": sum(entry["passed"] for entry in entries),
        "scenarios": entries,
    }


def run_monitoring(
    data_path=DEFAULT_DATA_PATH,
    api_url=DEFAULT_API_URL,
    drift_output=DEFAULT_DRIFT_OUTPUT,
    drift_metrics_output=DEFAULT_DRIFT_METRICS,
    schema_output=DEFAULT_SCHEMA_LOG,
    transport=None,
) -> dict:
    """Run drift simulation and separate malformed-request validation."""
    reference = prepare_heldout_examples(data_path)
    if reference.empty:
        raise ValueError("No complete held-out examples are available for drift monitoring.")
    drifted = create_valid_drifted_data(reference)

    drift_evidence = generate_evidently_drift_report(
        reference[FEATURE_COLUMNS],
        drifted,
        drift_output,
    )

    drift_metrics_output = Path(drift_metrics_output)
    drift_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    drift_metrics_output.write_text(
        json.dumps(drift_evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    valid_request = build_api_request(drifted.iloc[0])
    schema_evidence = validate_schema_anomalies(api_url, valid_request, transport=transport)

    schema_output = Path(schema_output)
    schema_output.parent.mkdir(parents=True, exist_ok=True)
    schema_output.write_text(
        json.dumps(schema_evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "drift_report": str(drift_output),
        "drift_metrics": drift_evidence,
        "schema_validation": schema_evidence,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--drift-output", type=Path, default=DEFAULT_DRIFT_OUTPUT)
    parser.add_argument("--drift-metrics", type=Path, default=DEFAULT_DRIFT_METRICS)
    parser.add_argument("--schema-output", type=Path, default=DEFAULT_SCHEMA_LOG)
    return parser.parse_args()


def main():
    args = parse_args()
    evidence = run_monitoring(
        data_path=args.data,
        api_url=args.api_url,
        drift_output=args.drift_output,
        drift_metrics_output=args.drift_metrics,
        schema_output=args.schema_output,
    )
    print(json.dumps({
        "drift_report": evidence["drift_report"],
        "schema_validation": evidence["schema_validation"],
    }, indent=2))
    schema = evidence["schema_validation"]
    return int(schema["scenarios_passed"] != schema["scenarios_requested"])


if __name__ == "__main__":
    raise SystemExit(main())
