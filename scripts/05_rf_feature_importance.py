
"""Random Forest feature importance analysis."""

import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


# ==================================================
# Paths
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

MODEL_PATH = (
    ROOT / "models"
    / "random_forest_delhi_initial.joblib"
)

OUTPUT_DIR = ROOT / "models" / "rf_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Load model and dataset
# ==================================================

print("\nLoading Random Forest and dataset...")

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
print(f"Features: {len(features)}")


# ==================================================
# STEP 1: Built-in Feature Importance
# ==================================================

print("\nCalculating built-in feature importance...")

builtin_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
})

builtin_importance = builtin_importance.sort_values(
    "importance",
    ascending=False,
)

builtin_importance.to_csv(
    OUTPUT_DIR / "builtin_feature_importance.csv",
    index=False,
)

print("\nBuilt-in Feature Importance:")
print(builtin_importance.to_string(index=False))


# ==================================================
# STEP 2: Permutation Importance
# ==================================================

print("\nCalculating permutation importance...")

permutation_result = permutation_importance(
    estimator=model,
    X=X_test,
    y=y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_importance_df = pd.DataFrame({
    "feature": features,
    "importance_mean": permutation_result.importances_mean,
    "importance_std": permutation_result.importances_std,
})

permutation_importance_df = (
    permutation_importance_df.sort_values(
        "importance_mean",
        ascending=False,
    )
)

permutation_importance_df.to_csv(
    OUTPUT_DIR / "permutation_importance.csv",
    index=False,
)

print("\nPermutation Importance:")
print(permutation_importance_df.to_string(index=False))


# ==================================================
# STEP 3: Plot Built-in Feature Importance
# ==================================================

print("\nCreating feature importance plots...")

plt.figure(figsize=(10, 7))

plt.barh(
    builtin_importance["feature"],
    builtin_importance["importance"],
)

plt.gca().invert_yaxis()

plt.xlabel("Impurity-based Importance")
plt.ylabel("Feature")
plt.title("Random Forest Built-in Feature Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "builtin_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# STEP 4: Plot Permutation Importance
# ==================================================

plt.figure(figsize=(10, 7))

plt.barh(
    permutation_importance_df["feature"],
    permutation_importance_df["importance_mean"],
    xerr=permutation_importance_df["importance_std"],
)

plt.gca().invert_yaxis()

plt.xlabel("Increase in RMSE after Permutation")
plt.ylabel("Feature")
plt.title("Random Forest Permutation Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "permutation_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# STEP 5: Combined Importance Comparison
# ==================================================

combined = builtin_importance.merge(
    permutation_importance_df,
    on="feature",
)

combined.to_csv(
    OUTPUT_DIR / "feature_importance_comparison.csv",
    index=False,
)

print("\nFeature importance analysis complete.")

print(f"Results saved in: {OUTPUT_DIR}")
