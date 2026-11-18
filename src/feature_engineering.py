
"""
Feature engineering for Parking Birmingham occupancy forecasting.

Creates leakage-safe calendar and historical occupancy features
using information available at prediction time.
"""

import pandas as pd


def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar and historical occupancy features.

    Parameters
    ----------
    df : pd.DataFrame
        Validated parking observations.

    Returns
    -------
    pd.DataFrame
        Observations with calendar and historical features.
        Invalid lag sequences contain NaN values.
    """

    data = df.copy()

    # Sort chronologically within each parking facility
    data = data.sort_values(
        ["SystemCodeNumber", "LastUpdated"]
    ).reset_index(drop=True)

    # Calendar features
    data["hour"] = data["LastUpdated"].dt.hour
    data["day_of_week"] = data["LastUpdated"].dt.day_of_week
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)
    data["month"] = data["LastUpdated"].dt.month

    # Historical occupancy features
    groups = data.groupby("SystemCodeNumber")

    data["occupancy_lag_1"] = groups["Occupancy"].shift(1)
    data["occupancy_lag_2"] = groups["Occupancy"].shift(2)

    data["occupancy_change_1"] = (
        data["Occupancy"] - data["occupancy_lag_1"]
    )

    # Rolling mean using current and two previous observations
    data["occupancy_rolling_mean_3"] = (
        data.groupby("SystemCodeNumber")["Occupancy"]
        .transform(lambda x: x.rolling(window=3, min_periods=3).mean())
    )

    # Historical time intervals
    data["lag_interval_1"] = (
        groups["LastUpdated"].diff().dt.total_seconds() / 60
    )

    data["lag_interval_2"] = (
        groups["LastUpdated"].diff(2).dt.total_seconds() / 60
        - data["lag_interval_1"]
    )

    # Only accept consecutive historical intervals of 25–35 minutes
    valid_lags = (
        data["lag_interval_1"].between(25, 35)
        & data["lag_interval_2"].between(25, 35)
    )

    lag_columns = [
        "occupancy_lag_1",
        "occupancy_lag_2",
        "occupancy_change_1",
        "occupancy_rolling_mean_3",
    ]

    data.loc[~valid_lags, lag_columns] = float("nan")

    return data




def create_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct approximately 30-minute future occupancy targets.

    Uses the next observation within each parking facility,
    provided it occurs 25–35 minutes after the current timestamp.

    Future-derived columns must never be used as model inputs.
    """

    data = df.copy()

    # Sort observations chronologically within each facility
    data = data.sort_values(
        ["SystemCodeNumber", "LastUpdated"]
    ).reset_index(drop=True)

    groups = data.groupby("SystemCodeNumber")

    # Identify the next observation and its occupancy
    data["TargetTimestamp"] = groups["LastUpdated"].shift(-1)
    data["TargetOccupancy"] = groups["Occupancy"].shift(-1)

    # Calculate the actual forecasting horizon
    data["ForecastHorizonMinutes"] = (
        data["TargetTimestamp"] - data["LastUpdated"]
    ).dt.total_seconds() / 60

    # Retain only examples with valid future targets
    data = data.loc[
        data["ForecastHorizonMinutes"].between(25, 35)
    ].copy()

    return data





def chronological_split(df: pd.DataFrame):
    """
    Split forecasting examples chronologically into
    training, validation, and test datasets.

    Uses global prediction timestamps and excludes examples
    whose target timestamps cross split boundaries.

    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """

    data = df.sort_values(
        ["LastUpdated", "SystemCodeNumber"]
    ).reset_index(drop=True)

    # Establish global chronological boundaries
    unique_times = data["LastUpdated"].sort_values().unique()

    train_boundary = pd.Timestamp(
        unique_times[int(len(unique_times) * 0.70)]
    )

    test_boundary = pd.Timestamp(
        unique_times[int(len(unique_times) * 0.85)]
    )

    # Training period
    train_df = data.loc[
        (data["LastUpdated"] < train_boundary)
        & (data["TargetTimestamp"] < train_boundary)
    ].copy()

    # Validation period
    val_df = data.loc[
        (data["LastUpdated"] >= train_boundary)
        & (data["LastUpdated"] < test_boundary)
        & (data["TargetTimestamp"] < test_boundary)
    ].copy()

    # Held-out test period
    test_df = data.loc[
        data["LastUpdated"] >= test_boundary
    ].copy()

    # Verify temporal separation
    assert (
        train_df["TargetTimestamp"].max()
        < val_df["LastUpdated"].min()
    ), "Training and validation periods overlap."

    assert (
        val_df["TargetTimestamp"].max()
        < test_df["LastUpdated"].min()
    ), "Validation and test periods overlap."

    return train_df, val_df, test_df
