# Model Loading Contract

## 1. Model Information

| Property | Value |
|---|---|
| Project | Parking Availability Prediction Using MLOps |
| MLflow experiment | `parking-occupancy-prediction` |
| Registered model | `parking-occupancy-random-forest` |
| Registered version | 1 |
| Model type | Random Forest Regressor |
| Framework | scikit-learn |
| Model format | MLflow scikit-learn pipeline (`skops`) |
| Forecasting horizon | Approximately 30 minutes |

The registered model includes the fitted preprocessing pipeline and Random Forest estimator.

## 2. Model Loading

The model is stored in the MLflow Model Registry and can be loaded using its registered model URI.

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_URI = "models:/parking-occupancy-random-forest/1"

model = mlflow.sklearn.load_model(MODEL_URI)
```

The local MLflow tracking server must be running and accessible. For deployment, the tracking URI should be configurable rather than hardcoded.

## 3. Required Input Features

The model expects 11 input features defined in `src/feature_schema.py`.

| Feature | Description |
|---|---|
| `SystemCodeNumber` | Parking facility identifier |
| `Capacity` | Total parking capacity |
| `Occupancy` | Current occupied spaces |
| `hour` | Hour of the current observation |
| `day_of_week` | Day of the week |
| `is_weekend` | Weekend indicator |
| `month` | Month of the current observation |
| `occupancy_lag_1` | Previous valid occupancy observation |
| `occupancy_lag_2` | Second previous valid occupancy observation |
| `occupancy_change_1` | Change between current and previous occupancy |
| `occupancy_rolling_mean_3` | Mean occupancy across the current and two previous valid observations |

Historical features must be calculated separately for each facility using observations available at prediction time.

The model expects a pandas DataFrame containing these feature columns.

## 4. Prediction Output

The model predicts the number of occupied parking spaces approximately 30 minutes ahead.

The raw model prediction is a continuous numerical value.

Predictions should be constrained to the physically valid range:

```python
predicted_occupancy = max(
    0,
    min(predicted_occupancy, capacity)
)
```

Estimated available parking spaces are calculated as:

```python
predicted_available_spaces = capacity - predicted_occupancy
```

The deployed application is responsible for applying these constraints because the saved scikit-learn pipeline does not include prediction clipping.

## 5. Model Evaluation

Validation results for the registered Random Forest model:

| Metric | Value |
|---|---:|
| MAE | 18.7751 |
| RMSE | 33.6727 |

These metrics were calculated using the validation dataset. The held-out test dataset is reserved for final evaluation.

## 6. Deployment Requirements

The inference application must:

1. Connect to the configured MLflow tracking server.
2. Load the registered model version.
3. Validate the required input feature schema.
4. Generate predictions using the fitted scikit-learn pipeline.
5. Constrain predicted occupancy between zero and facility capacity.
6. Calculate estimated available parking spaces.

The model should be loaded during application initialization rather than downloaded for every prediction request.


## 7. Verification Status

- [x] Model successfully registered in MLflow.
- [x] Registered Version 1 successfully loaded in Python.
- [x] Prediction tested using a valid 11-feature input record.
- [x] FastAPI deployment successfully loads the registered model.
- [x] End-to-end inference verified through the deployment.

### FastAPI Inference Test Results

The registered Random Forest model was successfully
loaded and tested through the FastAPI `/predict` endpoint.

- HTTP response: 200 OK
- Input features: 11
- Test input: Synthetic parking observation
- Facility capacity: 577 spaces
- Predicted occupancy: 224.56 spaces
- Predicted available spaces: 352.44 spaces
- Forecast horizon: Approximately 30 minutes
- Registered model: parking-occupancy-random-forest, Version 1
- Result: PASSED

### Local Inference Test Results

The registered Random Forest model was tested using
`tests/test_registered_model.py`.

- Input features: 11
- Predicted occupancy: 222.3 spaces
- Predicted available spaces: 354.7 spaces
- Facility capacity: 577 spaces
- Result: PASSED
