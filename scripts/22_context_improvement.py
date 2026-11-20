"""Training-only 3-km context improvement search. CV selection scores are development results, not independent validation."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree

CONTEXT = [
    "NDVI", "NDBI", "Albedo", "DW_water", "DW_trees", "DW_grass",
    "DW_flooded_vegetation", "DW_crops", "DW_shrub", "DW_built",
    "DW_bare", "DW_snow_and_ice",
]
KEYS = ["cell_id", "year", "block_id"]


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not len(y):
        return dict(n=0, rmse=None, mae=None, bias=None, r2=None)
    error = y - pred
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
                r2=float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None)


def build_context(imagery, radius_km=3.0):
    """Input is deliberately restricted to identifiers, coordinates and predictors."""
    allowed = ["cell_id", "year", "lat", "lon"] + CONTEXT
    if list(imagery.columns) != allowed:
        raise ValueError("Unexpected neighbourhood inputs; targets are forbidden.")
    out = pd.DataFrame(index=imagery.index)
    names = ["ctx_" + c for c in CONTEXT]
    out[names] = np.nan
    out["ctx_count"] = 0
    out["ctx_fallback"] = 0
    for _, area in imagery.groupby("year", sort=True):
        if len(area) < 2:
            raise ValueError("Insufficient same-year imagery donors.")
        coordinates = np.radians(area[["lat", "lon"]].to_numpy(float))
        neighbours = BallTree(coordinates, metric="haversine").query_radius(
            coordinates, r=(radius_km + 1e-8) / 6371.0088)
        values = area[CONTEXT].to_numpy(float)
        ids = area.cell_id.to_numpy()
        for position, (index, indices) in enumerate(zip(area.index, neighbours)):
            indices = indices[ids[indices] != ids[position]]
            pool = values[indices] if len(indices) else values[ids != ids[position]]
            if not len(pool):
                raise ValueError("No eligible fallback donors.")
            out.loc[index, names] = pool.mean(axis=0)
            out.loc[index, "ctx_count"] = len(indices)
            out.loc[index, "ctx_fallback"] = int(len(indices) == 0)
    if not np.isfinite(out.to_numpy(float)).all():
        raise ValueError("Invalid neighbourhood features.")
    return out


TAIL_WEIGHTS = {
    "baseline": 1.0, "tail_1p5": 1.5, "tail_2": 2.0,
    "capacity": 1.0, "capacity_tail_1p5": 1.5, "regularized": 1.0,
}

def make_models(threads, trees):
    from xgboost import XGBRegressor
    base = dict(n_estimators=trees, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=1.0, objective="reg:squarederror", random_state=42,
        n_jobs=threads)
    capacity = dict(base, n_estimators=5 if trees == 5 else 600,
                    max_depth=8, learning_rate=0.03,
                    min_child_weight=1, reg_lambda=0.5)
    regularized = dict(base, n_estimators=5 if trees == 5 else 600,
                       max_depth=4, learning_rate=0.03,
                       min_child_weight=5, reg_lambda=5.0)
    return {name: XGBRegressor(**(
        capacity if name.startswith("capacity") else
        regularized if name == "regularized" else base))
        for name in TAIL_WEIGHTS}


def recommendation(scores, smoke):
    """Conservative development gate; never an independent accuracy estimate."""
    if smoke:
        return dict(status="SMOKE_ONLY", selected=None)
    means = scores.groupby("model").mean(numeric_only=True)
    base = means.loc["baseline"]
    baseline_folds = scores[scores.model.eq("baseline")].set_index("fold")
    candidates = []
    for name, row in means.iterrows():
        if name == "baseline":
            continue
        paired = scores[scores.model.eq(name)].set_index("fold")
        improved = int((paired.rmse < baseline_folds.rmse).sum())
        needed = ["rmse", "hot_rmse", "cold_rmse", "bias"]
        finite = np.isfinite(row[needed].to_numpy(float)).all()
        passed = bool(finite and row.rmse < base.rmse
            and row.hot_rmse < base.hot_rmse
            and row.cold_rmse < base.cold_rmse
            and abs(row.bias) <= abs(base.bias) and improved >= 3)
        candidates.append(dict(model=name, passes=passed,
            folds_with_lower_rmse=improved, cv_rmse=float(row.rmse)))
    eligible = [x for x in candidates if x["passes"]]
    selected = min(eligible, key=lambda x: x["cv_rmse"])["model"] if eligible else "baseline"
    return dict(status="CV_CANDIDATE" if eligible else "KEEP_BASELINE",
        selected=selected, comparisons=candidates,
        caution="Selection on these folds is optimistic; fresh validation is required.",
        gate="Lower mean fold overall/hot/cold RMSE, no worse absolute mean bias, >=3 folds improved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="One fold, five trees: execution check only, not research results.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from uhi.preprocess.load import load_raw
    from uhi.preprocess.clean import clean_rows
    from uhi.preprocess.split import assign_blocks
    from uhi.preprocess.transforms import add_log_transforms, year_center, Winsorizer
    from uhi.preprocess import filters as F

    state_path = root / "data/processed/delhi/preprocess_state.json"
    config_path = root / "configs/pipeline.yaml"
    fold_path = root / "models/experiments/spatial_cv_folds_v1.csv"
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    pp = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))["preprocess"]
    raw = load_raw(root / "data/raw", "delhi", years=state["years"])
    data, _ = clean_rows(raw, pp["min_valid_frac"], pp["placeholder_values"])
    data["block_id"] = assign_blocks(data, state["split"]["n_blocks"], state["split"]["bounds"])
    data["split"] = data.block_id.map(state["split"]["block_map"])
    if not data.split.isin(["train", "test"]).all():
        raise ValueError("Unassigned blocks found.")
    # Explicitly discard holdout rows before context construction or any fit.
    train = add_log_transforms(data.loc[data.split.eq("train")].reset_index(drop=True))
    del data, raw
    features, _ = F.candidate_features(pp["dw_drop_classes"], pp["dw_reference_class"])
    if set(features) & {"SUHII", "LST", "LST_ref", "lon", "lat", "year", "block_id", "DW_label"}:
        raise ValueError("Forbidden direct model inputs.")
    required = list(dict.fromkeys(features + CONTEXT + ["SUHII", "lat", "lon"]))
    if not np.isfinite(train[required].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite data require a separate fold-safe handling design.")
    if train.duplicated(["cell_id", "year"]).any():
        raise ValueError("Duplicate cell-year observations.")
    saved = pd.read_csv(fold_path)
    if saved.duplicated(KEYS).any():
        raise ValueError("Duplicate saved fold assignments.")
    identity = train[KEYS].assign(_row=np.arange(len(train)))
    aligned = identity.merge(saved[KEYS + ["fold"]], on=KEYS, how="outer",
                             validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError("Frozen folds do not match this dataset.")
    aligned = aligned.sort_values("_row").reset_index(drop=True)
    if set(aligned.fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected exactly five saved folds.")
    for group in ["cell_id", "block_id"]:
        if aligned.groupby(group).fold.nunique().max() != 1:
            raise ValueError(f"{group} spans multiple folds.")
    fold_ids = aligned.fold.astype(int).to_numpy()
    context = build_context(train[["cell_id", "year", "lat", "lon"] + CONTEXT].copy())
    folds = [1] if args.smoke else [1, 2, 3, 4, 5]
    trees = 5 if args.smoke else 300
    # Resolve all optional dependencies before creating an output directory.
    templates = make_models(args.threads, trees)
    out = root / "models/experiments" / (
        ("SMOKE_" if args.smoke else "") + "context_improvement_" +
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    inputs = [state_path, config_path, fold_path, Path(__file__).resolve()] + [
        root / f"data/raw/delhi_{year}.csv" for year in state["years"]]
    inputs += sorted((root / "src/uhi/preprocess").glob("*.py"))
    metadata = dict(
        python=sys.version, platform=platform.platform(), smoke=args.smoke,
        test_evaluated=False, tuning_performed=False,
        context_policy="same-year outer-training-area available predictor imagery, own cell excluded",
        scope="Training-only model selection; scores are not independent validation or causal evidence.",
        parameters={name: model.get_params() for name, model in templates.items()},
        tail_weights=TAIL_WEIGHTS,
        selection_policy="Lower overall/hot/cold mean fold RMSE, no worse absolute mean bias, >=3 improved folds",
        validation_note="Model selection CV only; not independent validation.",
        versions={p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]},
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    aligned[KEYS + ["fold"]].to_csv(out / "fold_assignments.csv", index=False)
    rows, prediction_frames, preprocess_records = [], [], []
    print(f"Train rows={len(train)} | blocks={train.block_id.nunique()} | SMOKE={args.smoke}", flush=True)
    for fold in folds:
        fit, val = fold_ids != fold, fold_ids == fold
        transformed, means = year_center(train, pp["year_center_features"], fit)
        bw = Winsorizer(*pp["winsor_quantiles"]).fit(transformed.loc[fit], features)
        transformed = bw.transform(transformed)
        cols, _ = F.drop_zero_variance(transformed.loc[fit], features)
        cw = Winsorizer(*pp["winsor_quantiles"]).fit(context.loc[fit], ["ctx_" + c for c in CONTEXT])
        clipped_context = cw.transform(context)
        ctx_cols, _ = F.drop_zero_variance(clipped_context.loc[fit], list(context.columns))
        inputs_by_variant = {
                "context_3km": pd.concat([transformed[cols], clipped_context[ctx_cols]], axis=1),
        }
        preprocess_records.append(dict(fold=fold, year_means=means,
            base_features=cols, context_features=ctx_cols,
            base_bounds=bw.bounds_, context_bounds=cw.bounds_))
        y_fit = train.loc[fit, "SUHII"]
        actual = train.loc[val, "SUHII"].to_numpy()
        low, high = y_fit.quantile([0.1, 0.9]).to_numpy()
        hot, cold = actual >= high, actual <= low
        for variant, X in inputs_by_variant.items():
            for family, model in make_models(args.threads, trees).items():
                tail = (y_fit.to_numpy() <= low) | (y_fit.to_numpy() >= high)
                weights = np.where(tail, TAIL_WEIGHTS[family], 1.0)
                weights = weights / weights.mean()
                model.fit(X.loc[fit], y_fit, sample_weight=weights)
                pred = model.predict(X.loc[val])
                row = dict(model=family, variant=variant, fold=fold, n_features=X.shape[1],
                    **metrics(actual, pred), fit_rmse=metrics(y_fit, model.predict(X.loc[fit]))["rmse"],
                    hot_n=int(hot.sum()), cold_n=int(cold.sum()),
                    hot_rmse=metrics(actual[hot], pred[hot])["rmse"],
                    cold_rmse=metrics(actual[cold], pred[cold])["rmse"],
                    hot_bias=metrics(actual[hot], pred[hot])["bias"],
                    cold_bias=metrics(actual[cold], pred[cold])["bias"],
                    hot_threshold=float(high), cold_threshold=float(low))
                rows.append(row)
                frame = train.loc[val, KEYS + ["SUHII"]].copy()
                frame = frame.assign(model=family, variant=variant, fold=fold,
                                     prediction=pred, hot=hot, cold=cold)
                prediction_frames.append(frame)
                pd.DataFrame(rows).to_csv(out / "fold_results.csv", index=False)
                pd.concat(prediction_frames, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
                print(f"Fold {fold} | {family}/{variant} | RMSE={row['rmse']:.5f} | R2={row['r2']:.5f}", flush=True)
    scores = pd.DataFrame(rows)
    summary = scores.groupby(["model", "variant"])[[
        "rmse", "mae", "r2", "bias", "fit_rmse", "hot_rmse", "cold_rmse", "hot_bias", "cold_bias"]].mean()
    summary.to_csv(out / "summary.csv")
    scores.groupby(["model", "variant"])[["rmse", "mae", "r2"]].std().to_csv(out / "fold_std.csv")
    decision = recommendation(scores, args.smoke)
    (out / "recommendation.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    oof = pd.concat(prediction_frames, ignore_index=True)
    year_metrics = [dict(model=m, variant=v, year=int(y), **metrics(g.SUHII, g.prediction))
                    for (m, v, y), g in oof.groupby(["model", "variant", "year"])]
    pd.DataFrame(year_metrics).to_csv(out / "year_metrics.csv", index=False)
    (out / "fold_preprocessing.json").write_text(json.dumps(preprocess_records, indent=2), encoding="utf-8")
    print("\nMEAN SPATIAL-CV SCORES\n" + summary.round(5).to_string())
    print("\nDEVELOPMENT RECOMMENDATION\n" + json.dumps(decision, indent=2))
    print("\nResults:", out)
    print("Smoke check only; do not report these scores." if args.smoke else
          "30 fits complete. Training-only development search; no existing model overwritten.")


if __name__ == "__main__":
    main()
