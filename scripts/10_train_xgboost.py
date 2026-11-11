
"""Complete XGBoost training and evaluation workflow."""

import json
from pathlib import Path

import joblib
import pandas as pd

from uhi.models.xgboost_model import build_xgboost
from uhi.models.evaluation import (
    evaluate_mean_baseline,
    spatial_cross_validation,
    calculate_metrics,
)
from uhi.models.tuning import tune_xgboost
from uhi.training.train import train_model


# ==================================================
# CONFIGURATION
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

MODEL_DIR = ROOT / "models" / "xgboost"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_MODEL_PATH = MODEL_DIR / "xgboost_delhi_initial.joblib"
TUNED_MODEL_PATH = MODEL_DIR / "xgboost_delhi_tuned.joblib"
RESULTS_PATH = MODEL_DIR / "xgboost_complete_results.json"
TUNING_CSV_PATH = MODEL_DIR / "xgboost_tuning_search.csv"

N_SPLITS = 5
N_ITER = 25
RANDOM_STATE = 42


# ==================================================
# LOAD DATA
# ==================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

with open(STATE_PATH, "r", encoding="utf-8") as f:
    state = json.load(f)

features = state["final_features"]
target = state["target"]

df = pd.read_parquet(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(features)}")
print(f"Target: {target}")


# ==================================================
# TRAIN-TEST SEPARATION
# ==================================================

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

groups = train_df["block_id"]

print(f"Training observations: {len(train_df)}")
print(f"Testing observations: {len(test_df)}")
print(f"Training geographical blocks: {groups.nunique()}")


# ==================================================
# STEP 1: MEAN BASELINE
# ==================================================

print("\n" + "=" * 60)
print("STEP 1: MEAN BASELINE")
print("=" * 60)

baseline_model, baseline_metrics = evaluate_mean_baseline(
    X_train, y_train, X_test, y_test
)

print("\nMean Baseline Test Performance:")
for name, value in baseline_metrics.items():
    print(f"{name}: {value}")


# ==================================================
# STEP 2: INITIAL XGBOOST SPATIAL CV
# ==================================================

print("\n" + "=" * 60)
print("STEP 2: INITIAL XGBOOST SPATIAL CV")
print("=" * 60)

initial_model = build_xgboost(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
)

initial_cv_results, initial_cv_summary = spatial_cross_validation(
    model=initial_model,
    X=X_train,
    y=y_train,
    groups=groups,
    n_splits=N_SPLITS,
)

print("\nInitial Spatial CV Summary:")
print(initial_cv_summary)


# ==================================================
# STEP 3: INITIAL XGBOOST TRAINING
# ==================================================

print("\n" + "=" * 60)
print("STEP 3: INITIAL XGBOOST")
print("=" * 60)

initial_model = train_model(
    model_name="xgboost",
    X_train=X_train,
    y_train=y_train,
    model_params={
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": RANDOM_STATE,
    },
)

initial_predictions = initial_model.predict(X_test)

initial_test_metrics = calculate_metrics(
    y_test, initial_predictions
)

print("\nInitial XGBoost Test Performance:")
for name, value in initial_test_metrics.items():
    print(f"{name}: {value}")

joblib.dump(initial_model, INITIAL_MODEL_PATH)


# ==================================================
# STEP 4: HYPERPARAMETER TUNING
# ==================================================

print("\n" + "=" * 60)
print("STEP 4: XGBOOST HYPERPARAMETER TUNING")
print("=" * 60)

search, tuned_model, tuned_cv_metrics, search_results = tune_xgboost(
    X_train=X_train,
    y_train=y_train,
    groups=groups,
    n_iter=N_ITER,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
)

print("\nBest Hyperparameters:")
for name, value in search.best_params_.items():
    print(f"{name}: {value}")

print("\nBest Spatial CV Metrics:")
for name, value in tuned_cv_metrics.items():
    print(f"{name}: {value}")

search_results.to_csv(TUNING_CSV_PATH, index=False)


# ==================================================
# STEP 5: TUNED XGBOOST TEST EVALUATION
# ==================================================

print("\n" + "=" * 60)
print("STEP 5: TUNED XGBOOST TEST EVALUATION")
print("=" * 60)

tuned_predictions = tuned_model.predict(X_test)

tuned_test_metrics = calculate_metrics(
    y_test, tuned_predictions
)

print("\nTuned XGBoost Test Performance:")
for name, value in tuned_test_metrics.items():
    print(f"{name}: {value}")

joblib.dump(tuned_model, TUNED_MODEL_PATH)


# ==================================================
# STEP 6: FINAL COMPARISON
# ==================================================

print("\n" + "=" * 60)
print("STEP 6: MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Mean Baseline": baseline_metrics,
    "Initial XGBoost": initial_test_metrics,
    "Tuned XGBoost": tuned_test_metrics,
}).T

print(comparison.to_string())


# ==================================================
# STEP 7: SAVE RESULTS
# ==================================================

results = {
    "dataset": {
        "total_rows": len(df),
        "training_rows": len(train_df),
        "testing_rows": len(test_df),
        "training_blocks": int(groups.nunique()),
        "features": features,
        "target": target,
    },
    "baseline": baseline_metrics,
    "initial_xgboost": {
        "parameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": RANDOM_STATE,
        },
        "spatial_cv_folds": initial_cv_results.to_dict(
            orient="records"
        ),
        "spatial_cv_summary": initial_cv_summary.to_dict(),
        "test_metrics": initial_test_metrics,
    },
    "tuned_xgboost": {
        "best_parameters": search.best_params_,
        "spatial_cv_metrics": tuned_cv_metrics,
        "test_metrics": tuned_test_metrics,
    },
    "comparison": comparison.to_dict(orient="index"),
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 60)
print("XGBOOST WORKFLOW COMPLETE")
print("=" * 60)

print(f"Initial model: {INITIAL_MODEL_PATH}")
print(f"Tuned model: {TUNED_MODEL_PATH}")
print(f"Results: {RESULTS_PATH}")
print(f"Tuning search: {TUNING_CSV_PATH}")