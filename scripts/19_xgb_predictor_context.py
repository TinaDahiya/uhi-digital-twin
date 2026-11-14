"""Training-only XGBoost experiments on frozen spatial folds."""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
fold_path = ROOT / "models/experiments/spatial_cv_folds_v1.csv"

state = json.loads(state_path.read_text(encoding="utf-8"))
pp = yaml.safe_load(config_path.read_text(encoding="utf-8"))["preprocess"]

raw_paths = [ROOT / f"data/raw/delhi_{y}.csv" for y in state["years"]]
raw = load_raw(ROOT / "data/raw", "delhi", years=state["years"])
data, _ = clean_rows(raw, pp["min_valid_frac"], pp["placeholder_values"])
data["block_id"] = assign_blocks(
    data, state["split"]["n_blocks"], state["split"]["bounds"]
)
data["split"] = data.block_id.map(state["split"]["block_map"])
if data["split"].isna().any():
    raise ValueError("Unassigned blocks found.")

train = data.loc[data.split == "train"].reset_index(drop=True)
train = add_log_transforms(train)
features, _ = F.candidate_features(
    pp["dw_drop_classes"], pp["dw_reference_class"]
)
if train[features + ["SUHII"]].isna().any().any():
    raise ValueError("Missing data requires fold-safe imputation.")
if not np.isfinite(train[features + ["SUHII"]].to_numpy(float)).all():
    raise ValueError("Non-finite values found.")

keys = ["cell_id", "year", "block_id"]
if train.duplicated(["cell_id", "year"]).any():
    raise ValueError("Duplicate cell-year rows.")

saved = pd.read_csv(fold_path)
if saved.duplicated(keys).any():
    raise ValueError("Duplicate fold assignments.")

# Explicitly align the manifest with current training-row order.
identity = train[keys].copy()
identity["_row"] = np.arange(len(identity))
aligned = identity.merge(
    saved[keys + ["fold"]], on=keys, how="outer",
    validate="one_to_one", indicator=True
)
if not aligned["_merge"].eq("both").all():
    raise ValueError("Saved folds do not match current training data.")
aligned = aligned.sort_values("_row").reset_index(drop=True)
fold_ids = aligned["fold"].astype(int)
if set(fold_ids) != {1, 2, 3, 4, 5}:
    raise ValueError("Invalid folds.")
if aligned.groupby("block_id").fold.nunique().max() != 1:
    raise ValueError("A geographic block spans multiple folds.")
if aligned.groupby("cell_id").fold.nunique().max() != 1:
    raise ValueError("A cell spans multiple folds.")

baseline = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, reg_lambda=1.0,
    objective="reg:squarederror", random_state=42, n_jobs=2
)
capacity = {
    **baseline,
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 1,
    "reg_lambda": 0.5
}
variants = [
    ("reference", baseline, False),
    ("neighbour_1km", baseline, False),
    ("neighbour_3km", baseline, False)
]
radii = {"reference": None, "neighbour_1km": 1.0, "neighbour_3km": 3.0}

out = ROOT / "models/experiments" / (
    "xgb_predictor_context_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
out.mkdir(parents=True, exist_ok=False)

metadata = {
    "python": sys.version,
    "versions": {p: version(p) for p in [
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost"
    ]},
    "variants": {
        label: {
            "parameters": params,
            "tail_weight": 1.0,
            "neighbour_radius_km": radii[label],
            "donors": "same-year outer-training-area predictor imagery, excluding own cell",
            "fallback": "same-year outer-training-area predictor mean, excluding own cell"
        }
        for label, params, weighted in variants
    },
    "thresholds": "10th/90th percentiles of each fold's fitting targets",
    "test_evaluated": False,
    "sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [state_path, config_path, fold_path, Path(__file__)] + raw_paths
    }
}
(out / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
aligned[keys + ["fold"]].to_csv(out / "fold_assignments.csv", index=False)

def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not len(actual):
        return {"rmse": None, "mae": None, "bias": None}
    residual = actual - predicted
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual))
    }

from sklearn.neighbors import BallTree

context_cols = [
    "NDVI", "NDBI", "Albedo",
    "DW_water", "DW_trees", "DW_grass", "DW_flooded_vegetation",
    "DW_crops", "DW_shrub", "DW_built", "DW_bare", "DW_snow_and_ice"
]
required = context_cols + ["lon", "lat"]
if not np.isfinite(train[required].to_numpy(float)).all():
    raise ValueError("Missing/non-finite neighbourhood inputs.")

def build_context(fit_mask, radius_km):
    """Use available predictor imagery across the outer training area; no targets."""
    result = pd.DataFrame(index=train.index)
    names = ["ctx_" + c for c in context_cols]
    result[names] = np.nan
    result["ctx_count"] = 0
    result["ctx_fallback"] = 0

    for year in sorted(train.year.unique()):
        donor = train.loc[
            train.year.eq(year),
            ["cell_id", "lat", "lon"] + context_cols
        ]
        query = train.loc[train.year.eq(year)]

        if len(donor) < 2:
            raise ValueError("Too few same-year fitting donors.")

        donor_coords = np.radians(donor[["lat", "lon"]].to_numpy())
        query_coords = np.radians(query[["lat", "lon"]].to_numpy())
        tree = BallTree(donor_coords, metric="haversine")
        indices = tree.query_radius(
            query_coords, r=(radius_km + 1e-8) / 6371.0088
        )

        values = donor[context_cols].to_numpy(float)
        donor_ids = donor.cell_id.to_numpy()
        query_ids = query.cell_id.to_numpy()

        for row_index, own_id, neighbours in zip(
            query.index, query_ids, indices
        ):
            neighbours = neighbours[donor_ids[neighbours] != own_id]
            count = len(neighbours)

            if count:
                means = values[neighbours].mean(axis=0)
            else:
                # Exclude own cell from fallback too.
                pool = values[donor_ids != own_id]
                if not len(pool):
                    raise ValueError("No fallback donors available.")
                means = pool.mean(axis=0)

            result.loc[row_index, names] = means
            result.loc[row_index, "ctx_count"] = count
            result.loc[row_index, "ctx_fallback"] = int(count == 0)

    if result.isna().any().any():
        raise ValueError("Neighbourhood construction left missing values.")
    return result

# These are deterministic predictor summaries, not fitted target features.
# Only outer-training-area imagery is used; held-out test rows are excluded.
context_cache = {
    label: build_context(np.ones(len(train), dtype=bool), radius)
    for label, radius in radii.items() if radius is not None
}
for label, context in context_cache.items():
    print(
        f"{label}: available-imagery fallback="
        f"{context['ctx_fallback'].mean():.1%}",
        flush=True
    )

records = []
predictions = []

print(f"Training rows: {len(train)} | Blocks: {train.block_id.nunique()}")
print("Positive bias = underprediction; negative bias = overprediction.")
print("Output:", out, flush=True)

for fold in range(1, 6):
    fit_mask = fold_ids.ne(fold).to_numpy()
    val_mask = ~fit_mask

    transformed, _ = year_center(
        train, pp["year_center_features"], fit_mask
    )
    winsor = Winsorizer(*pp["winsor_quantiles"]).fit(
        transformed.loc[fit_mask], features
    )
    transformed = winsor.transform(transformed)
    cols, _ = F.drop_zero_variance(
        transformed.loc[fit_mask], features
    )

    X_fit = transformed.loc[fit_mask, cols]
    X_val = transformed.loc[val_mask, cols]
    y_fit = train.loc[fit_mask, "SUHII"]
    y_val = train.loc[val_mask, "SUHII"]

    low, high = y_fit.quantile([0.1, 0.9]).to_numpy()
    fit_tail = (y_fit <= low) | (y_fit >= high)
    hot = y_val.to_numpy() >= high
    cold = y_val.to_numpy() <= low

    base_fit = X_fit.copy()
    base_val = X_val.copy()

    for label, params, weighted in variants:
        X_fit = base_fit.copy()
        X_val = base_val.copy()

        radius = radii[label]
        fallback_share = 0.0
        if radius is not None:
            context = context_cache[label]

            # Fit context clipping only on fold-fitting rows.
            context_winsor = Winsorizer(*pp["winsor_quantiles"]).fit(
                context.loc[fit_mask],
                [c for c in context.columns if c.startswith("ctx_DW_")
                 or c in ("ctx_NDVI", "ctx_NDBI", "ctx_Albedo")]
            )
            clipped_context = context_winsor.transform(context)
            context_features, _ = F.drop_zero_variance(
                clipped_context.loc[fit_mask],
                list(clipped_context.columns)
            )

            X_fit = pd.concat([
                X_fit, clipped_context.loc[fit_mask, context_features]
            ], axis=1)
            X_val = pd.concat([
                X_val, clipped_context.loc[val_mask, context_features]
            ], axis=1)

            fallback_share = float(
                context.loc[val_mask, "ctx_fallback"].mean()
            )
            print(
                f"Fold {fold} | {label} | "
                f"validation neighbour fallback={fallback_share:.1%}",
                flush=True
            )

        model = XGBRegressor(**params)
        weights = np.where(fit_tail, 2.0, 1.0) if weighted else None
        if weights is not None:
            weights = weights / weights.mean()
        model.fit(X_fit, y_fit, sample_weight=weights)

        pred = model.predict(X_val)
        fit_pred = model.predict(X_fit)

        overall = metrics(y_val, pred)
        hot_scores = metrics(y_val.to_numpy()[hot], pred[hot])
        cold_scores = metrics(y_val.to_numpy()[cold], pred[cold])

        row = {
            "variant": label, "fold": fold,
            "n_features": X_fit.shape[1],
            "validation_context_fallback_share": fallback_share,
            **overall,
            "r2": float(r2_score(y_val, pred)),
            "fit_rmse": metrics(y_fit, fit_pred)["rmse"],
            "hot_n": int(hot.sum()), "cold_n": int(cold.sum()),
            "hot_rmse": hot_scores["rmse"],
            "hot_bias": hot_scores["bias"],
            "cold_rmse": cold_scores["rmse"],
            "cold_bias": cold_scores["bias"],
            "low_threshold": float(low), "high_threshold": float(high)
        }
        records.append(row)

        frame = train.loc[val_mask, keys + ["SUHII"]].copy()
        frame["fold"] = fold
        frame["variant"] = label
        frame["prediction"] = pred
        frame["hot"] = hot
        frame["cold"] = cold
        predictions.append(frame)

        pd.DataFrame(records).to_csv(
            out / "fold_results.csv", index=False
        )
        pd.concat(predictions, ignore_index=True).to_csv(
            out / "oof_predictions.csv", index=False
        )
        print(
            f"Fold {fold} | {label} | RMSE={overall['rmse']:.4f} | "
            f"R2={row['r2']:.4f} | hot RMSE={hot_scores['rmse']:.4f}",
            flush=True
        )

scores = pd.DataFrame(records)
summary = scores.groupby("variant")[[
    "rmse", "mae", "r2", "fit_rmse",
    "hot_rmse", "hot_bias", "cold_rmse", "cold_bias"
]].mean()
summary = summary.rename(columns={
    "rmse": "cv_rmse", "mae": "cv_mae", "r2": "cv_r2"
})

oof = pd.concat(predictions, ignore_index=True)
year_rows = []
for (label, year), group in oof.groupby(["variant", "year"]):
    year_rows.append({
        "variant": label, "year": int(year),
        **metrics(group.SUHII, group.prediction)
    })
pd.DataFrame(year_rows).to_csv(out / "year_metrics.csv", index=False)
summary.to_csv(out / "summary.csv")

reference = scores[scores.variant == "reference"].set_index("fold")
paired = []
for label, group in scores.groupby("variant"):
    group = group.set_index("fold").sort_index()
    delta = group.rmse - reference.rmse
    paired.append({
        "variant": label,
        "folds_with_lower_rmse": int((delta < 0).sum()),
        "mean_rmse_delta": float(delta.mean()),
        "mean_hot_rmse_delta": float(
            (group.hot_rmse - reference.hot_rmse).mean()
        )
    })
paired = pd.DataFrame(paired)
paired.to_csv(out / "paired_comparison.csv", index=False)

print("\nFINAL COMPARISON - MEAN OF FIVE FOLDS")
print(summary.round(5).to_string())
print("\nPAIRED COMPARISON - NEGATIVE DELTA IS BETTER")
print(paired.round(5).to_string(index=False))
print("\nYEAR-WISE VALIDATION ERRORS")
print(pd.DataFrame(year_rows).round(5).to_string(index=False))
print("\nResults:", out)
print("No held-out test evaluation; no existing models overwritten.")
