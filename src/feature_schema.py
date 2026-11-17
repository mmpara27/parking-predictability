
"""Shared feature schema for parking occupancy prediction."""

FEATURE_COLUMNS = [
    "SystemCodeNumber",
    "Capacity",
    "Occupancy",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "occupancy_lag_1",
    "occupancy_lag_2",
    "occupancy_change_1",
    "occupancy_rolling_mean_3",
]

TARGET_COLUMN = "TargetOccupancy"

CATEGORICAL_FEATURES = [
    "SystemCodeNumber",
]

NUMERIC_FEATURES = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURES
]
