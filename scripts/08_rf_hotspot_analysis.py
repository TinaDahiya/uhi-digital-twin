
"""Evaluate Random Forest performance in the hottest SUHII locations."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

INPUT_PATH = Path("models/rf_analysis/rf_residuals.csv")
OUTPUT_DIR = Path("models/rf_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Available columns:", df.columns.tolist())

# Automatically identify actual and predicted columns
actual_candidates = ["SUHII"]
pred_candidates = ["predicted_SUHII"]

actual_col = next(
    (c for c in actual_candidates if c in df.columns), None
)
pred_col = next(
    (c for c in pred_candidates if c in df.columns), None
)

if actual_col is None or pred_col is None:
    raise ValueError(
        "Could not identify actual/predicted columns. "
        "Check the printed column names and update actual_col/pred_col."
    )

df = df.dropna(subset=[actual_col, pred_col]).copy()

actual = df[actual_col].to_numpy()
predicted = df[pred_col].to_numpy()

# --------------------------------------------------
# IDENTIFY HOTTEST 10% OF ACTUAL TEST OBSERVATIONS
# --------------------------------------------------

threshold = np.percentile(actual, 90)

hot_mask = actual >= threshold

hot_actual = actual[hot_mask]
hot_predicted = predicted[hot_mask]

# --------------------------------------------------
# METRICS FOR HOTTEST 10%
# --------------------------------------------------

hot_metrics = {
    "hot_observations": int(hot_mask.sum()),
    "suhii_threshold_90th_percentile": float(threshold),
    "hot_r2": float(r2_score(hot_actual, hot_predicted)),
    "hot_rmse": float(np.sqrt(
        mean_squared_error(hot_actual, hot_predicted)
    )),
    "hot_mae": float(
        mean_absolute_error(hot_actual, hot_predicted)
    ),
}

# --------------------------------------------------
# HOTSPOT CLASSIFICATION PERFORMANCE
# --------------------------------------------------

# Actual hotspots: actual SUHII >= actual 90th percentile
actual_hotspots = hot_mask

# Predicted hotspots: predicted SUHII >= same threshold
predicted_hotspots = predicted >= threshold

hot_metrics.update({
    "hotspot_precision": float(
        precision_score(
            actual_hotspots,
            predicted_hotspots,
            zero_division=0,
        )
    ),
    "hotspot_recall": float(
        recall_score(
            actual_hotspots,
            predicted_hotspots,
            zero_division=0,
        )
    ),
    "hotspot_f1": float(
        f1_score(
            actual_hotspots,
            predicted_hotspots,
            zero_division=0,
        )
    ),
})

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_path = OUTPUT_DIR / "hotspot_analysis.json"

with open(output_path, "w") as f:
    json.dump(hot_metrics, f, indent=4)

print("\nHOTSPOT ANALYSIS RESULTS")
print("-" * 40)

for key, value in hot_metrics.items():
    print(f"{key}: {value}")

print(f"\nResults saved to: {output_path}")
