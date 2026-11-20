"""Training-only spatial CV comparison with persisted fold assignments."""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
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
state = json.loads(state_path.read_text(encoding="utf-8"))
pp = yaml.safe_load(config_path.read_text(encoding="utf-8"))["preprocess"]

raw = load_raw(ROOT / "data/raw", "delhi", years=state["years"])
data, _ = clean_rows(raw, pp["min_valid_frac"], pp["placeholder_values"])
data["block_id"] = assign_blocks(
    data, state["split"]["n_blocks"], state["split"]["bounds"]
)
data["split"] = data.block_id.map(state["split"]["block_map"])
if data["split"].isna().any():
    raise ValueError("Unassigned blocks found; frozen split cannot be preserved.")

# Test rows are excluded before transforms, fitting or evaluation.
train = data.loc[data.split == "train"].reset_index(drop=True)
train = add_log_transforms(train)
candidates, _ = F.candidate_features(
    pp["dw_drop_classes"], pp["dw_reference_class"]
)
if train[candidates + ["SUHII"]].isna().any().any():
    raise ValueError("Missing values found; fold-safe imputation is required.")
if train.duplicated(["cell_id", "year"]).any():
    raise ValueError("Duplicate cell-year rows found.")
if train.block_id.nunique() < 5:
    raise ValueError("At least five training blocks are required.")

out = ROOT / "models/experiments" / (
    "xgb_fold_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
out.mkdir(parents=True, exist_ok=False)

# Persist exact validation assignments; independent of future library defaults.
manifest_path = ROOT / "models/experiments/spatial_cv_folds_v1.csv"
keys = ["cell_id", "year", "block_id"]
if manifest_path.exists():
    saved = pd.read_csv(manifest_path)
    manifest = train[keys].merge(
        saved, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not manifest["_merge"].eq("both").all():
        raise ValueError("Saved folds do not match current training rows.")
    manifest = manifest.drop(columns="_merge")
else:
    manifest = train[keys].copy()
    manifest["fold"] = 0
    for fold, (_, val_idx) in enumerate(
        GroupKFold(5).split(train, groups=train.block_id), 1
    ):
        manifest.loc[val_idx, "fold"] = fold
    manifest.to_csv(manifest_path, index=False)

if set(manifest["fold"]) != {1, 2, 3, 4, 5}:
    raise ValueError("Invalid saved fold numbers.")
if manifest.groupby("block_id")["fold"].nunique().max() != 1:
    raise ValueError("A block spans multiple validation folds.")
if manifest.groupby("cell_id")["fold"].nunique().max() != 1:
    raise ValueError("A cell spans multiple validation folds.")

manifest.to_csv(out / "fold_assignments.csv", index=False)
metadata = {
    "python": sys.version,
    "versions": {p: version(p) for p in [
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost"
    ]},
    "sha256": {},
    "test_evaluated": False,
}
for path in [state_path, config_path, manifest_path, Path(__file__)] + [
    ROOT / f"data/raw/delhi_{year}.csv" for year in state["years"]
]:
    metadata["sha256"][str(path.relative_to(ROOT))] = (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
(out / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

results = []
predictions = []
print(f"Training rows: {len(train)}; blocks: {train.block_id.nunique()}")
print(f"Fold manifest: {manifest_path}", flush=True)

for fold in range(1, 6):
    fit_mask = manifest["fold"].ne(fold).to_numpy()
    val_mask = ~fit_mask

    z, _ = year_center(train, pp["year_center_features"], fit_mask)
    winsor = Winsorizer(*pp["winsor_quantiles"]).fit(
        z.loc[fit_mask], candidates
    )
    z = winsor.transform(z)
    all_features, _ = F.drop_zero_variance(z.loc[fit_mask], candidates)
    filtered, _ = F.spearman_filter(
        z.loc[fit_mask], all_features,
        pp["spearman_threshold"], pp["spearman_keep_rule"],
        target=train.loc[fit_mask, "SUHII"],
        tiers=pp["spearman_priority_tiers"],
    )
    filtered, _, _ = F.vif_filter(
        z.loc[fit_mask], filtered,
        pp["vif_threshold"], pp["spearman_keep_rule"],
        pp["spearman_priority_tiers"],
    )

    xgb_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=1.0,
        objective="reg:squarederror", random_state=42, n_jobs=2,
    )
    variants = [
        ("rf_filtered", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=2
        ), filtered),
        ("xgb_filtered", XGBRegressor(**xgb_params), filtered),
        ("xgb_all_candidates", XGBRegressor(**xgb_params), all_features),
    ]

    for label, model, features in variants:
        model.fit(z.loc[fit_mask, features], train.loc[fit_mask, "SUHII"])
        pred = model.predict(z.loc[val_mask, features])
        actual = train.loc[val_mask, "SUHII"]

        row = dict(
            variant=label, fold=fold, n_features=len(features),
            rmse=float(np.sqrt(mean_squared_error(actual, pred))),
            mae=float(mean_absolute_error(actual, pred)),
            r2=float(r2_score(actual, pred)),
            features=features,
        )
        results.append(row)

        frame = train.loc[val_mask, keys + ["SUHII"]].copy()
        frame["variant"] = label
        frame["fold"] = fold
        frame["prediction"] = pred
        predictions.append(frame)

        (out / "fold_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(
            f"Fold {fold} | {label} | features={len(features)} | "
            f"RMSE={row['rmse']:.4f} | R2={row['r2']:.4f}",
            flush=True,
        )

scores = pd.DataFrame(results)
scores.drop(columns="features").to_csv(out / "fold_results.csv", index=False)
pd.concat(predictions, ignore_index=True).to_csv(
    out / "oof_predictions.csv", index=False
)
summary = scores.groupby("variant")[["rmse", "mae", "r2"]].agg(["mean", "std"])
summary.to_csv(out / "summary.csv")
print("\nFINAL SPATIAL-CV COMPARISON")
print(summary.to_string())
print("\nResults:", out)
print("Held-out test set was not evaluated.")
