
"""Random Forest residual analysis."""

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "delhi"
    / "delhi_ml_ready.parquet"
)

STATE_PATH = (
    ROOT / "data" / "processed" / "delhi"
    / "preprocess_state.json"
)

# Use the tuned model for final diagnostics.
# Change this if your saved tuned model has a different name.
MODEL_PATH = ROOT / "models" / "random_forest_delhi_initial.joblib"

OUTPUT_DIR = ROOT / "models" / "rf_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD MODEL AND DATA
# ==================================================

print("\nLoading model and dataset...")

with open(STATE_PATH, "r", encoding="utf-8") as f:
    state = json.load(f)

features = state["final_features"]
target = state["target"]

model = joblib.load(MODEL_PATH)

df = pd.read_parquet(DATA_PATH)

test_df = df[df["split"] == "test"].copy()

X_test = test_df[features]
y_test = test_df[target]

print(f"Test observations: {len(test_df)}")


# ==================================================
# STEP 1: PREDICTIONS AND RESIDUALS
# ==================================================

print("\nCalculating predictions and residuals...")

y_pred = model.predict(X_test)

test_df["predicted_SUHII"] = y_pred
test_df["residual"] = y_test - y_pred
test_df["absolute_error"] = test_df["residual"].abs()

print("\nResidual statistics:")

print(test_df["residual"].describe())


# ==================================================
# STEP 2: SAVE RESIDUAL DATA
# ==================================================

residual_columns = [
    "cell_id",
    "year",
    "lon",
    "lat",
    "block_id",
    "SUHII",
    "predicted_SUHII",
    "residual",
    "absolute_error",
]

residual_df = test_df[residual_columns].copy()

residual_df.to_csv(
    OUTPUT_DIR / "rf_residuals.csv",
    index=False,
)

print("\nResidual data saved.")


# ==================================================
# STEP 3: RESIDUAL DISTRIBUTION
# ==================================================

plt.figure(figsize=(9, 6))

plt.hist(
    test_df["residual"],
    bins=40,
    edgecolor="black",
)

plt.axvline(
    0,
    linestyle="--",
    color="red",
)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Random Forest Residual Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=300,
)

plt.close()


# ==================================================
# STEP 4: ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
)

plt.xlabel("Actual SUHII")
plt.ylabel("Predicted SUHII")
plt.title("Actual vs Predicted SUHII")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
)

plt.close()


# ==================================================
# STEP 5: RESIDUALS VS PREDICTED
# ==================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_pred,
    test_df["residual"],
    alpha=0.5,
)

plt.axhline(
    0,
    linestyle="--",
    color="red",
)

plt.xlabel("Predicted SUHII")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted SUHII")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residuals_vs_predicted.png",
    dpi=300,
)

plt.close()


# ==================================================
# STEP 6: GEOGRAPHICAL RESIDUAL MAP
# ==================================================

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    test_df["lon"],
    test_df["lat"],
    c=test_df["residual"],
    cmap="coolwarm",
    vmin=-2,
    vmax=2,
    s=12,
)

plt.colorbar(
    scatter,
    label="Residual (Actual - Predicted)",
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Random Forest Residuals")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "spatial_residual_map.png",
    dpi=300,
)

plt.close()


# ==================================================
# STEP 7: ERROR SUMMARY
# ==================================================

summary = {
    "test_rows": int(len(test_df)),
    "mean_residual": float(test_df["residual"].mean()),
    "mean_absolute_error": float(
        mean_absolute_error(y_test, y_pred)
    ),
    "rmse": float(
        mean_squared_error(y_test, y_pred) ** 0.5
    ),
    "maximum_absolute_error": float(
        test_df["absolute_error"].max()
    ),
}

with open(
    OUTPUT_DIR / "residual_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=4)

print("\nResidual analysis complete.")
print(summary)

print(f"\nResults saved in: {OUTPUT_DIR}")
