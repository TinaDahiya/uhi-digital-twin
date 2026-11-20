
"""Consolidated comparison of all four trained model families."""

import json
from pathlib import Path

import joblib
import pandas as pd

from uhi.models.evaluation import calculate_metrics


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "delhi" / "delhi_ml_ready.parquet"
STATE_PATH = ROOT / "data" / "processed" / "delhi" / "preprocess_state.json"

RESULTS_DIR = ROOT / "models" / "comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Update paths here if your model files are stored differently.
MODEL_PATHS = {
    "Random Forest": {
        "initial": ROOT / "models" / "random_forest_delhi_initial.joblib",
        "tuned": ROOT / "models" / "random_forest_delhi_tuned.joblib",
    },
    "XGBoost": {
        "initial": ROOT / "models" / "xgboost_delhi_initial.joblib",
        "tuned": ROOT / "models" / "xgboost_delhi_tuned.joblib",
    },
    "LightGBM": {
        "initial": ROOT / "models" / "lightgbm" / "lightgbm_delhi_initial.joblib",
        "tuned": ROOT / "models" / "lightgbm" / "lightgbm_delhi_tuned.joblib",
    },
    "Gradient Boosting": {
        "initial": ROOT / "models" / "gradient_boosting" / "gradient_boosting_delhi_initial.joblib",
        "tuned": ROOT / "models" / "gradient_boosting" / "gradient_boosting_delhi_tuned.joblib",
    },
}


# ==================================================
# LOAD DATA
# ==================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

with open(STATE_PATH, "r", encoding="utf-8") as f:
    state = json.load(f)

features = state["final_features"]
target = state["target"]

df = pd.read_parquet(DATA_PATH)

test_df = df[df["split"] == "test"].copy()

X_test = test_df[features]
y_test = test_df[target]

print(f"Test observations: {len(test_df)}")
print(f"Features: {len(features)}")


# ==================================================
# MODEL EVALUATION
# ==================================================

results = []

print("\n" + "=" * 60)
print("EVALUATING MODELS")
print("=" * 60)

for model_name, versions in MODEL_PATHS.items():

    for version, model_path in versions.items():

        if not model_path.exists():
            print(f"\nWARNING: Missing model: {model_path}")
            continue

        print(f"\nEvaluating {model_name} ({version})")

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        result = {
            "model": model_name,
            "version": version,
            **metrics,
        }

        results.append(result)

        print(f"R2:   {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")


# ==================================================
# RESULTS TABLE
# ==================================================

results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError(
        "No models were evaluated. Check MODEL_PATHS."
    )

results_df = results_df.sort_values(
    by="rmse",
    ascending=True,
)

results_df["rank_rmse"] = range(1, len(results_df) + 1)

print("\n" + "=" * 60)
print("CONSOLIDATED MODEL COMPARISON")
print("=" * 60)

print(
    results_df[
        ["model", "version", "r2", "rmse", "mae", "rank_rmse"]
    ].to_string(index=False)
)


# ==================================================
# SAVE RESULTS
# ==================================================

CSV_PATH = RESULTS_DIR / "model_comparison.csv"
JSON_PATH = RESULTS_DIR / "model_comparison.json"

results_df.to_csv(CSV_PATH, index=False)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results_df.to_dict(orient="records"), f, indent=4)

print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)

print(f"CSV:  {CSV_PATH}")
print(f"JSON: {JSON_PATH}")

best_model = results_df.iloc[0]

print("\nBest model by test RMSE:")
print(f"Model: {best_model['model']} ({best_model['version']})")
print(f"RMSE: {best_model['rmse']:.4f}")
print(f"R2:   {best_model['r2']:.4f}")
print(f"MAE:  {best_model['mae']:.4f}")
