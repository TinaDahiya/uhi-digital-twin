
"""Evaluate Random Forest performance by year and dominant land cover."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --------------------------------------------------
# PATHS
# --------------------------------------------------

RESIDUAL_PATH = Path("models/rf_analysis/rf_residuals.csv")
ML_DATA_PATH = Path("data/processed/delhi/delhi_ml_ready.parquet")
OUTPUT_DIR = Path("models/rf_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# METRICS FUNCTION
# --------------------------------------------------

def calculate_metrics(group):
    actual = group["SUHII"]
    predicted = group["predicted_SUHII"]

    return pd.Series({
        "observations": len(group),
        "R2": r2_score(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted),
    })

# --------------------------------------------------
# LOAD TEST PREDICTIONS
# --------------------------------------------------

df = pd.read_csv(RESIDUAL_PATH)

print("Test prediction columns:", df.columns.tolist())

# --------------------------------------------------
# YEAR-WISE PERFORMANCE
# --------------------------------------------------

year_results = (
    df.groupby("year")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

print("\nYEAR-WISE PERFORMANCE")
print(year_results.to_string(index=False))

year_results.to_csv(
    OUTPUT_DIR / "year_performance.csv", index=False
)

# --------------------------------------------------
# LOAD LAND COVER FEATURES
# --------------------------------------------------

ml_df = pd.read_parquet(ML_DATA_PATH)

print("\nML data columns:", ml_df.columns.tolist())

# Join by cell_id and year
landcover_columns = [
    "DW_trees",
    "DW_grass",
    "DW_water",
    "DW_flooded_vegetation",
    "DW_bare",
    "DW_snow_and_ice",
]

available_columns = [
    c for c in landcover_columns if c in ml_df.columns
]

if not available_columns:
    raise ValueError("No Dynamic World land-cover columns found.")

merge_columns = ["cell_id", "year"] + available_columns

df = df.merge(
    ml_df[merge_columns],
    on=["cell_id", "year"],
    how="left",
    validate="many_to_one",
)

# --------------------------------------------------
# DOMINANT LAND COVER
# --------------------------------------------------

df["dominant_landcover"] = (
    df[available_columns]
    .idxmax(axis=1)
    .str.replace("DW_", "", regex=False)
)

# --------------------------------------------------
# LAND-COVER-WISE PERFORMANCE
# --------------------------------------------------

landcover_results = (
    df.groupby("dominant_landcover")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

print("\nLAND-COVER-WISE PERFORMANCE")
print(landcover_results.to_string(index=False))

landcover_results.to_csv(
    OUTPUT_DIR / "landcover_performance.csv", index=False
)

# --------------------------------------------------
# SAVE JSON SUMMARY
# --------------------------------------------------

summary = {
    "year_performance": year_results.to_dict(orient="records"),
    "landcover_performance": landcover_results.to_dict(
        orient="records"
    ),
}

with open(OUTPUT_DIR / "group_performance.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nAnalysis completed.")
print(f"Results saved in: {OUTPUT_DIR}")