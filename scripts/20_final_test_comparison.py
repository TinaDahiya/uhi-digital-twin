"""Frozen reference vs 3-km candidate: final spatial holdout comparison."""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.neighbors import BallTree
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uhi.preprocess.load import load_raw
from uhi.preprocess.clean import clean_rows
from uhi.preprocess.split import assign_blocks
from uhi.preprocess.transforms import add_log_transforms, year_center, Winsorizer
from uhi.preprocess import filters as F

state_path = ROOT / "data/processed/delhi/preprocess_state.json"
config_path = ROOT / "configs/pipeline.yaml"
state = json.loads(state_path.read_text(encoding="utf-8"))
pp = yaml.safe_load(config_path.read_text(encoding="utf-8"))["preprocess"]

raw = load_raw(ROOT / "data/raw", "delhi", years=state["years"])
d, _ = clean_rows(raw, pp["min_valid_frac"], pp["placeholder_values"])
d["block_id"] = assign_blocks(
    d, state["split"]["n_blocks"], state["split"]["bounds"]
)
d["split"] = d.block_id.map(state["split"]["block_map"])
if not d.split.isin(["train", "test"]).all():
    raise ValueError("Invalid frozen split.")
if d.duplicated(["cell_id", "year"]).any():
    raise ValueError("Duplicate cell-year rows.")

fit = d.split.eq("train").to_numpy()
test = d.split.eq("test").to_numpy()
if not fit.any() or not test.any():
    raise ValueError("Empty train/test split.")
if set(d.loc[fit, "cell_id"]) & set(d.loc[test, "cell_id"]):
    raise ValueError("Cell overlap between train and test.")
if set(d.loc[fit, "block_id"]) & set(d.loc[test, "block_id"]):
    raise ValueError("Block overlap between train and test.")

features, _ = F.candidate_features(
    pp["dw_drop_classes"], pp["dw_reference_class"]
)
d = add_log_transforms(d)
if not np.isfinite(d[features + ["SUHII", "lon", "lat"]].to_numpy(float)).all():
    raise ValueError("Missing/non-finite values require explicit handling.")

z, year_means = year_center(d, pp["year_center_features"], fit)
base_winsor = Winsorizer(*pp["winsor_quantiles"]).fit(z.loc[fit], features)
z = base_winsor.transform(z)
base_features, _ = F.drop_zero_variance(z.loc[fit], features)

# Only predictor columns enter the neighbourhood calculation.
context_cols = [
    "NDVI", "NDBI", "Albedo",
    "DW_water", "DW_trees", "DW_grass", "DW_flooded_vegetation",
    "DW_crops", "DW_shrub", "DW_built", "DW_bare", "DW_snow_and_ice"
]
imagery = d[["cell_id", "year", "lat", "lon"] + context_cols].copy()
if not np.isfinite(imagery[context_cols].to_numpy(float)).all():
    raise ValueError("Invalid context predictors.")

context = pd.DataFrame(index=d.index)
context_names = ["ctx_" + c for c in context_cols]
context[context_names] = np.nan
context["ctx_count"] = 0
context["ctx_fallback"] = 0

for year in sorted(imagery.year.unique()):
    area = imagery.loc[imagery.year.eq(year)]
    coords = np.radians(area[["lat", "lon"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    neighbours = tree.query_radius(coords, r=(3.0 + 1e-8) / 6371.0088)
    values = area[context_cols].to_numpy(float)
    ids = area.cell_id.to_numpy()

    for row_index, own_id, indices in zip(area.index, ids, neighbours):
        indices = indices[ids[indices] != own_id]
        if len(indices):
            means = values[indices].mean(axis=0)
        else:
            pool = values[ids != own_id]
            if not len(pool):
                raise ValueError("No fallback imagery donors.")
            means = pool.mean(axis=0)
        context.loc[row_index, context_names] = means
        context.loc[row_index, "ctx_count"] = len(indices)
        context.loc[row_index, "ctx_fallback"] = int(len(indices) == 0)

if context.isna().any().any():
    raise ValueError("Missing context values.")

ctx_winsor = Winsorizer(*pp["winsor_quantiles"]).fit(
    context.loc[fit], context_names
)
context_clipped = ctx_winsor.transform(context)
ctx_features, _ = F.drop_zero_variance(
    context_clipped.loc[fit], list(context_clipped.columns)
)

params = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, reg_lambda=1.0,
    objective="reg:squarederror", random_state=42, n_jobs=2
)

# Thresholds are frozen from training targets before test scoring.
low, high = d.loc[fit, "SUHII"].quantile([0.1, 0.9]).to_numpy()
out = ROOT / "models/final_evaluation" / datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)
out.mkdir(parents=True, exist_ok=False)

def metrics(actual, pred):
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    if len(actual) == 0:
        return dict(n=0, rmse=None, mae=None, bias=None, r2=None)
    error = actual - pred
    return dict(
        n=len(actual),
        rmse=float(np.sqrt(np.mean(error ** 2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(np.mean(error)),
        r2=float(r2_score(actual, pred))
        if len(actual) > 1 and np.var(actual) > 0 else None
    )

inputs = {
    "reference": z[base_features],
    "neighbour_3km": pd.concat(
        [z[base_features], context_clipped[ctx_features]], axis=1
    )
}
summary_rows = []
year_rows = []
tail_rows = []
prediction_frames = []

print(f"Train: {fit.sum()} | Test: {test.sum()}", flush=True)
for label, X in inputs.items():
    print("Fitting frozen configuration:", label, flush=True)
    model = XGBRegressor(**params)
    model.fit(X.loc[fit], d.loc[fit, "SUHII"])
    pred = model.predict(X.loc[test])
    actual = d.loc[test, "SUHII"].to_numpy()

    summary_rows.append({"variant": label, **metrics(actual, pred)})
    frame = d.loc[test, [
        "cell_id", "year", "block_id", "lon", "lat", "SUHII"
    ]].copy()
    frame["prediction"] = pred
    frame["variant"] = label
    frame["residual"] = actual - pred
    prediction_frames.append(frame)

    for year, group in frame.groupby("year"):
        year_rows.append({
            "variant": label, "year": int(year),
            **metrics(group.SUHII, group.prediction)
        })
    for name, mask in [
        ("hot", actual >= high), ("cold", actual <= low)
    ]:
        tail_rows.append({
            "variant": label, "subset": name,
            **metrics(actual[mask], pred[mask])
        })

    # Save estimator plus preprocessing information needed for reuse.
    joblib.dump({
        "model": model,
        "features": list(X.columns),
        "base_features": base_features,
        "context_features": ctx_features if label == "neighbour_3km" else [],
        "year_means": year_means,
        "base_winsor_bounds": base_winsor.bounds_,
        "context_winsor_bounds": ctx_winsor.bounds_
            if label == "neighbour_3km" else {},
        "context_radius_km": 3.0 if label == "neighbour_3km" else None,
        "context_columns": context_cols if label == "neighbour_3km" else [],
        "imagery_policy": "same-year available study-area predictors; exclude own cell",
        "fallback": "same-year predictor mean excluding own cell",
        "note": "Rebuild predictors with this workflow before calling model.predict."
    }, out / f"{label}_bundle.joblib")

summary = pd.DataFrame(summary_rows)
years = pd.DataFrame(year_rows)
tails = pd.DataFrame(tail_rows)
summary.to_csv(out / "test_summary.csv", index=False)
years.to_csv(out / "test_year_metrics.csv", index=False)
tails.to_csv(out / "test_tail_metrics.csv", index=False)
pd.concat(prediction_frames, ignore_index=True).to_csv(
    out / "test_predictions.csv", index=False
)

metadata = {
    "parameters": params,
    "train_rows": int(fit.sum()), "test_rows": int(test.sum()),
    "hot_threshold_from_train": float(high),
    "cold_threshold_from_train": float(low),
    "test_context_fallback_share": float(
        context.loc[test, "ctx_fallback"].mean()
    ),
    "imagery_policy": (
        "Available study-area predictors include test-area imagery. "
        "Test targets are used only for scoring. "
        "CV development used outer-training-area imagery."
    ),
    "test_history": "This spatial holdout was evaluated in earlier project work.",
    "python": sys.version,
    "versions": {p: version(p) for p in [
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost"
    ]},
    "sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [state_path, config_path, Path(__file__)] + [
            ROOT / f"data/raw/delhi_{y}.csv" for y in state["years"]
        ]
    }
}
(out / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\nFINAL TEST COMPARISON")
print(summary.round(5).to_string(index=False))
print("\nHOT/COLD TEST METRICS - TRAINING-DERIVED THRESHOLDS")
print(tails.round(5).to_string(index=False))
print("\nYEAR-WISE TEST METRICS")
print(years.round(5).to_string(index=False))
print("\nResults:", out)
print("Do not tune configurations against these test results.")
