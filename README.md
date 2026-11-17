# Parking Availability Prediction Using MLOps

An end-to-end Machine Learning Operations (MLOps) project to predict parking occupancy and estimate available parking spaces using historical data from Birmingham, United Kingdom.

**Course:** ADSP 32021 — Machine Learning Operations  
**Institution:** University of Chicago  
**Team:** Emmalucia E., Alison C., Monica P.  
**Final deadline:** December 2, 2026

## Project Overview

This project aims to build an automated machine learning pipeline that forecasts parking occupancy approximately 30 minutes ahead.

The system will estimate available parking spaces using:

```text
Available spaces = Capacity − Predicted occupancy
```

Predicted occupancy will be constrained to the valid range between zero and facility capacity.

The planned MLOps workflow includes data validation, feature engineering, model training, experiment tracking, deployment, and monitoring.

For the original project scope and proposed architecture, see [Project Proposal](project_proposal.md).

## Dataset

**Source:** [UCI Machine Learning Repository — Parking Birmingham](https://archive.ics.uci.edu/dataset/482/parking+birmingham)

The dataset contains historical occupancy observations from 30 parking facilities.

| Property | Value |
|---|---|
| Original observations | 35,717 |
| Parking facilities | 30 |
| Validated observations | 35,116 |
| Forecasting examples with complete historical features | 25,793 |
| Forecast horizon | Approximately 30 minutes |
| Model input features | 11 |

Original columns:

- `SystemCodeNumber` — Parking facility identifier
- `Capacity` — Total parking capacity
- `Occupancy` — Occupied parking spaces
- `LastUpdated` — Observation timestamp

## Repository Structure

```text
parking-predictability/
├── data/
│   └── dataset.csv
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_feature_engineering.ipynb
├── src/
│   ├── data_validation.py
│   ├── feature_engineering.py
│   └── feature_schema.py
├── project_proposal.md
└── README.md
```

Additional files and directories will be added as model training, automation, deployment, and monitoring are implemented.

## Data Validation

The data validation stage checks:

- Missing values and invalid timestamps
- Duplicate observations
- Nonpositive parking capacities
- Negative occupancy values
- Occupancy exceeding facility capacity
- Consistency of capacity values within each facility

**Validation results:**

| Metric | Count |
|---|---:|
| Original records | 35,717 |
| Duplicate records removed | 216 |
| Physically invalid occupancy records removed | 385 |
| Final validated records | 35,116 |

Reusable validation logic is implemented in `src/data_validation.py`.

## Feature Engineering

The project uses a facility-specific forecasting approach.

For each observation, the prediction target is the next recorded occupancy when the next observation occurs between 25 and 35 minutes later.

Historical features are constructed using information available at prediction time.

**Model features:**

| Category | Features |
|---|---|
| Facility | `SystemCodeNumber`, `Capacity` |
| Current occupancy | `Occupancy` |
| Calendar | `hour`, `day_of_week`, `is_weekend`, `month` |
| Historical occupancy | `occupancy_lag_1`, `occupancy_lag_2`, `occupancy_change_1`, `occupancy_rolling_mean_3` |

The shared feature definitions are maintained in `src/feature_schema.py`.

Reusable feature engineering functions are implemented in `src/feature_engineering.py`.

## Chronological Data Splitting

To reduce temporal leakage, observations are divided chronologically rather than randomly.

| Dataset | Observations |
|---|---:|
| Training | 18,423 |
| Validation | 3,509 |
| Test | 3,833 |

Forecast targets crossing split boundaries are excluded.

The test dataset is reserved for final evaluation and is not used during model development.

**Verification completed:**

- All 11 input features are available.
- Training and validation features contain no missing values.
- Target columns are excluded from model inputs.
- Feature leakage checks pass.

## Planned MLOps Pipeline

The following components are planned for subsequent development.

| Component | Technology | Purpose |
|---|---|---|
| Pipeline orchestration | Apache Airflow | Automate data preparation and model training |
| Model training | scikit-learn | Compare baseline and regression models |
| Experiment tracking | MLflow | Record model parameters, metrics, and artifacts |
| Model serving | FastAPI | Expose a prediction endpoint |
| Containerization | Docker | Package the inference application |
| Model monitoring | Evidently AI | Detect data drift and monitor input distributions |

## Project Progress

As of November 17, 2026:

| Component | Status |
|---|---|
| Project proposal and repository setup | Completed |
| Data audit and validation | Completed |
| Feature engineering and chronological splits | Implemented and tested |
| Reusable feature schema | Implemented and tested |
| Baseline and regression models | Pending |
| Airflow pipeline | Pending |
| MLflow tracking and registration | Pending |
| FastAPI and Docker deployment | Pending |
| Monitoring and drift simulation | Pending |

## Running the Project

The project currently uses Python and pandas for data validation and feature engineering.

To explore the completed work, open the following notebooks in VS Code or Jupyter:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_feature_engineering.ipynb`

Reusable Python functions are available in the `src/` directory.

Full environment setup and end-to-end execution instructions will be added as the automated pipeline is implemented.

## Project Documentation

- [Project Proposal](project_proposal.md) — Original objectives, architecture, timeline, and deliverables
- `notebooks/01_data_audit.ipynb` — Dataset exploration and validation
- `notebooks/02_feature_engineering.ipynb` — Forecast target construction, historical features, and chronological splitting

## Project Team

- Emmalucia E.
- Alison C.
- Monica P.

University of Chicago — ADSP 32021 Machine Learning Operations, Fall 2026.

