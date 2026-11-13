"""
Data validation and cleaning for the Parking Birmingham dataset.

Removes duplicate records, parses timestamps, and excludes
observations that violate parking capacity constraints.
"""

import pandas as pd


def validate_parking_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean parking occupancy observations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Parking Birmingham dataset.

    Returns
    -------
    pd.DataFrame
        Validated observations sorted by facility and timestamp.
    """

    required_columns = [
        "SystemCodeNumber",
        "Capacity",
        "Occupancy",
        "LastUpdated",
    ]

    # Ensure all required columns exist
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Preserve the original input DataFrame
    cleaned = df.copy()

    # Remove exact duplicate observations
    cleaned = cleaned.drop_duplicates()

    # Convert numeric columns
    for column in ["Capacity", "Occupancy"]:
        cleaned[column] = pd.to_numeric(
            cleaned[column], errors="coerce"
        )

    # Convert timestamps
    cleaned["LastUpdated"] = pd.to_datetime(
        cleaned["LastUpdated"], errors="coerce"
    )

    # Remove missing or invalid values
    cleaned = cleaned.dropna(subset=required_columns)

    # Enforce physical parking constraints
    cleaned = cleaned.loc[
        (cleaned["Capacity"] > 0)
        & (cleaned["Occupancy"] >= 0)
        & (cleaned["Occupancy"] <= cleaned["Capacity"])
    ]

    # Ensure one observation per facility and timestamp
    if cleaned.duplicated(
        subset=["SystemCodeNumber", "LastUpdated"]
    ).any():
        raise ValueError(
            "Conflicting observations exist for the same "
            "facility and timestamp."
        )

    # Sort chronologically within each facility
    cleaned = cleaned.sort_values(
        ["SystemCodeNumber", "LastUpdated"]
    ).reset_index(drop=True)

    return cleaned
