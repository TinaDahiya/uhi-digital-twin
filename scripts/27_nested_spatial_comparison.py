"""Limited equal-trial-budget nested spatial development comparison. No fresh-data claims."""
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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



FAMILIES = ["rf", "gb", "xgb", "lgbm"]
VARIANTS = ["cell_only", "cell_context", "cell_context_contrast"]


def grids(threads, smoke=False):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    specs = {
        "rf": (RandomForestRegressor, dict(random_state=42, n_jobs=threads), [
            dict(n_estimators=300, min_samples_leaf=2, max_features=1.0),
            dict(n_estimators=300, min_samples_leaf=4, max_features=.8, max_depth=18),
            dict(n_estimators=450, min_samples_leaf=1, max_features=.8)]),
        "gb": (GradientBoostingRegressor, dict(random_state=42, subsample=.8), [
            dict(n_estimators=300, learning_rate=.05, max_depth=3, min_samples_leaf=2),
            dict(n_estimators=450, learning_rate=.03, max_depth=2, min_samples_leaf=4),
            dict(n_estimators=300, learning_rate=.03, max_depth=4, min_samples_leaf=4)]),
        "xgb": (XGBRegressor, dict(random_state=42, n_jobs=threads,
            objective="reg:squarederror", subsample=.8, colsample_bytree=.8), [
            dict(n_estimators=300, learning_rate=.05, max_depth=6, min_child_weight=3, reg_lambda=1.),
            dict(n_estimators=500, learning_rate=.03, max_depth=4, min_child_weight=5, reg_lambda=5.),
            dict(n_estimators=500, learning_rate=.03, max_depth=8, min_child_weight=3, reg_lambda=2.)]),
        "lgbm": (LGBMRegressor, dict(random_state=42, n_jobs=threads,
            verbosity=-1, subsample=.8, subsample_freq=0, colsample_bytree=.8), [
            dict(n_estimators=300, learning_rate=.05, num_leaves=31, reg_lambda=1.),
            dict(n_estimators=500, learning_rate=.03, num_leaves=15, reg_lambda=3., min_child_samples=30),
            dict(n_estimators=300, learning_rate=.03, num_leaves=63, reg_lambda=3., min_child_samples=30)])
    }
    result = {}
    for family, (cls, base, candidates) in specs.items():
        params = [{**base, **c} for c in candidates]
        if smoke:
            params = [{**p, "n_estimators": 5} for p in params[:2]]
        result[family] = (cls, params)
    return result


def load_data(root):
    sys.path.insert(0, str(root / "src"))
    from uhi.preprocess.load import load_raw
    from uhi.preprocess.clean import clean_rows
    from uhi.preprocess.split import assign_blocks
    from uhi.preprocess.transforms import add_log_transforms
    from uhi.preprocess import filters as F
    state_path = root / "data/processed/delhi/preprocess_state.json"
    config_path = root / "configs/pipeline.yaml"
    fold_path = root / "models/experiments/spatial_cv_folds_v1.csv"
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    pp = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))["preprocess"]
    data, _ = clean_rows(load_raw(root / "data/raw", "delhi", state["years"]),
                         pp["min_valid_frac"], pp["placeholder_values"])
    data["block_id"] = assign_blocks(data, state["split"]["n_blocks"], state["split"]["bounds"])
    data["split"] = data.block_id.map(state["split"]["block_map"])
    if not data.split.isin(["train", "test"]).all():
        raise ValueError("Unassigned frozen blocks.")
    if set(data.loc[data.split.eq("train"), "cell_id"]) & set(data.loc[data.split.eq("test"), "cell_id"]):
        raise ValueError("Cell overlap with original holdout.")
    train = add_log_transforms(data[data.split.eq("train")].reset_index(drop=True))
    del data
    features, _ = F.candidate_features(pp["dw_drop_classes"], pp["dw_reference_class"])
    if set(features) & {"SUHII", "LST", "LST_ref", "lon", "lat", "year", "block_id", "DW_label"}:
        raise ValueError("Forbidden direct predictors.")
    required = list(dict.fromkeys(features + CONTEXT + ["SUHII", "lat", "lon"]))
    if not np.isfinite(train[required].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite inputs need a separate fold-safe design.")
    if train.duplicated(["cell_id", "year"]).any():
        raise ValueError("Duplicate cell-year.")
    saved = pd.read_csv(fold_path)
    if saved.duplicated(KEYS).any():
        raise ValueError("Duplicate frozen fold keys.")
    aligned = train[KEYS].assign(_row=np.arange(len(train))).merge(
        saved[KEYS + ["fold"]], on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError("Saved folds do not match training observations.")
    aligned = aligned.sort_values("_row").reset_index(drop=True)
    if set(aligned.fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five frozen folds.")
    for key in ["cell_id", "block_id"]:
        if aligned.groupby(key).fold.nunique().max() != 1:
            raise ValueError(key + " spans folds.")
    files = [state_path, config_path, fold_path] + [
        root / f"data/raw/delhi_{y}.csv" for y in state["years"]]
    files += sorted((root / "src/uhi/preprocess").glob("*.py"))
    return train, pp, features, aligned, files


def context_sets(train):
    imagery = train[["cell_id", "year", "lat", "lon"] + CONTEXT].copy()
    ctx = build_context(imagery, 3.0)
    delta = pd.DataFrame({"delta3_" + c: train[c] - ctx["ctx_" + c]
                          for c in CONTEXT}, index=train.index)
    return {"cell_only": pd.DataFrame(index=train.index),
            "cell_context": ctx,
            "cell_context_contrast": pd.concat([ctx, delta], axis=1)}


def prepare(train, pp, features, contexts, fit):
    """Fit every preprocessing parameter on the supplied fitting mask only."""
    from uhi.preprocess.transforms import year_center, Winsorizer
    from uhi.preprocess.filters import drop_zero_variance
    z, means = year_center(train, pp["year_center_features"], fit)
    bw = Winsorizer(*pp["winsor_quantiles"]).fit(z.loc[fit], features)
    z = bw.transform(z)
    base_cols, _ = drop_zero_variance(z.loc[fit], features)
    result, details = {}, {}
    for name, context in contexts.items():
        clip = [c for c in context if not c.endswith(("ctx_count", "ctx_fallback"))]
        cw = Winsorizer(*pp["winsor_quantiles"]).fit(context.loc[fit], clip)
        transformed = cw.transform(context)
        cols, _ = drop_zero_variance(transformed.loc[fit], list(context.columns))
        result[name] = pd.concat([z[base_cols], transformed[cols]], axis=1)
        details[name] = dict(context_features=cols, context_bounds=cw.bounds_)
    return result, dict(year_means=means, base_features=base_cols,
                        base_bounds=bw.bounds_, contexts=details)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def atomic_csv(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temp, index=False)
    temp.replace(path)


def main():
    from sklearn.model_selection import GroupKFold
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    root = args.root.resolve()
    print("Loading frozen training data; original holdout will not be scored.", flush=True)
    train, pp, features, aligned, files = load_data(root)
    specs = grids(args.threads, args.smoke)
    outer_folds = [1] if args.smoke else [1, 2, 3, 4, 5]
    inner_k = 2 if args.smoke else 3
    plan = dict(smoke=args.smoke, inner_folds=inner_k, outer_folds=outer_folds,
                variants=VARIANTS, parameters={f: ps for f, (_, ps) in specs.items()})
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    hashes["nested_script"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    versions = {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"]}
    contract = dict(plan=plan, hashes=hashes, versions=versions,
                    python=sys.version, platform=platform.platform())
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    expected_tasks = len(outer_folds) * len(FAMILIES) * len(VARIANTS)
    expected_fits = expected_tasks * (len(specs["rf"][1]) * inner_k + 1)
    metadata = dict(contract=contract, signature=signature,
        expected_tasks=expected_tasks, expected_fits=expected_fits,
        outer_holdout_scored=False, radius_km=3.,
        policy="Same-year outer-training-area available predictor imagery includes CV validation imagery; own cell and targets excluded.",
        limitation="Limited equal-trial budget; compute/capacity not equal. Prior feature development used these observations/folds. Nested tuning does not undo that adaptation. No fresh-data or novelty claim.")
    if args.resume is not None:
        out = args.resume.resolve()
        existing = json.loads((out / "metadata.json").read_text())
        if existing["signature"] != signature:
            raise ValueError("Resume refused: inputs, script, runtime, parameters or smoke mode changed.")
    else:
        out = root / "models/experiments" / (
            ("SMOKE_" if args.smoke else "") + "nested_spatial_" +
            datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "metadata.json", metadata)
        atomic_csv(out / "outer_fold_assignments.csv", aligned[KEYS + ["fold"]])
    (out / "tasks").mkdir(exist_ok=True)
    print(f"Rows={len(train)} | blocks={train.block_id.nunique()} | planned fits={expected_fits} | smoke={args.smoke}", flush=True)
    print("RESULTS:", out, flush=True)
    contexts = context_sets(train)
    fold_ids = aligned.fold.astype(int).to_numpy()
    y = train.SUHII.to_numpy(float)
    all_rows, all_predictions, all_tuning = [], [], []
    fit_calls = 0
    for outer in outer_folds:
        fit_outer = fold_ids != outer
        val_outer = ~fit_outer
        outer_indices = np.flatnonzero(fit_outer)
        splits = []
        assignments = []
        for inner, (a, b) in enumerate(GroupKFold(n_splits=inner_k).split(
                outer_indices, groups=train.loc[fit_outer, "block_id"]), start=1):
            fit = np.zeros(len(train), dtype=bool)
            val = fit.copy()
            fit[outer_indices[a]], val[outer_indices[b]] = True, True
            assert not (fit & val_outer).any() and not (val & val_outer).any()
            if set(train.loc[fit, "block_id"]) & set(train.loc[val, "block_id"]):
                raise ValueError("Inner spatial overlap.")
            X, prep = prepare(train, pp, features, contexts, fit)
            splits.append((inner, fit, val, X))
            frame = train.loc[fit_outer, KEYS].copy()
            frame["inner_fold"] = inner
            frame["role"] = np.where(fit[outer_indices], "fit", "validation")
            assignments.append(frame)
            prep_path = out / f"preprocess_outer{outer}_inner{inner}.json"
            if not prep_path.exists():
                atomic_json(prep_path, prep)
        atomic_csv(out / f"inner_assignments_outer{outer}.csv", pd.concat(assignments, ignore_index=True))
        X_outer, prep_outer = prepare(train, pp, features, contexts, fit_outer)
        atomic_json(out / f"preprocess_outer{outer}_refit.json", prep_outer)
        # Original outer-validation targets are first used below for final scoring only.
        low, high = np.quantile(y[fit_outer], [.1, .9])
        for family in FAMILIES:
            cls, candidates = specs[family]
            for variant in VARIANTS:
                task_id = f"outer{outer}_{family}_{variant}"
                task_path = out / "tasks" / (task_id + ".json")
                pred_path = out / "tasks" / (task_id + ".csv")
                if task_path.exists() and pred_path.exists():
                    task = json.loads(task_path.read_text())
                    pred_frame = pd.read_csv(pred_path)
                    if task.get("prediction_sha256") != hashlib.sha256(pred_path.read_bytes()).hexdigest():
                        raise ValueError("Checkpoint prediction file changed or corrupted.")
                    if task["signature"] != signature:
                        raise ValueError("Task signature mismatch.")
                    print("RESUMED", task_id, flush=True)
                else:
                    started = time.monotonic()
                    trials = []
                    for trial, params in enumerate(candidates):
                        inner_scores = []
                        for inner, fit, val, matrices in splits:
                            X = matrices[variant]
                            model = cls(**params)
                            model.fit(X.loc[fit], y[fit], sample_weight=np.ones(fit.sum()))
                            fit_calls += 1
                            score = metrics(y[val], model.predict(X.loc[val]))["rmse"]
                            inner_scores.append(score)
                            trials.append(dict(outer_fold=outer, model=family, variant=variant,
                                trial=trial, inner_fold=inner, rmse=score))
                        print(f"{task_id} | trial {trial+1}/{len(candidates)} | inner RMSE={np.mean(inner_scores):.5f}", flush=True)
                    tuning = pd.DataFrame(trials)
                    means = tuning.groupby("trial").rmse.mean()
                    selected = int(means.idxmin())  # stable first-trial tie break
                    model = cls(**candidates[selected])
                    X = X_outer[variant]
                    model.fit(X.loc[fit_outer], y[fit_outer], sample_weight=np.ones(fit_outer.sum()))
                    fit_calls += 1
                    predicted = model.predict(X.loc[val_outer])
                    actual = y[val_outer]
                    hot, cold = actual >= high, actual <= low
                    row = dict(outer_fold=outer, model=family, variant=variant,
                        selected_trial=selected, selected_inner_rmse=float(means.loc[selected]),
                        **metrics(actual, predicted),
                        fit_rmse=metrics(y[fit_outer], model.predict(X.loc[fit_outer]))["rmse"],
                        hot_n=int(hot.sum()), cold_n=int(cold.sum()),
                        hot_rmse=metrics(actual[hot], predicted[hot])["rmse"],
                        cold_rmse=metrics(actual[cold], predicted[cold])["rmse"],
                        hot_bias=metrics(actual[hot], predicted[hot])["bias"],
                        cold_bias=metrics(actual[cold], predicted[cold])["bias"],
                        hot_threshold=float(high), cold_threshold=float(low),
                        seconds=float(time.monotonic()-started))
                    pred_frame = train.loc[val_outer, KEYS + ["SUHII"]].copy()
                    pred_frame = pred_frame.assign(outer_fold=outer, model=family,
                        variant=variant, prediction=np.asarray(predicted, float), hot=hot, cold=cold)
                    task = dict(signature=signature, row=row,
                        selected_parameters=candidates[selected], inner_trials=trials)
                    atomic_csv(pred_path, pred_frame)
                    task["prediction_sha256"] = hashlib.sha256(pred_path.read_bytes()).hexdigest()
                    atomic_json(task_path, task)  # task completion marker written last
                    print(f"OUTER RESULT {task_id} | RMSE={row['rmse']:.5f} | R2={row['r2']:.5f}", flush=True)
                all_rows.append(task["row"])
                all_tuning.extend(task["inner_trials"])
                all_predictions.append(pred_frame)
                atomic_csv(out / "outer_results.csv", pd.DataFrame(all_rows))
                atomic_csv(out / "inner_trials.csv", pd.DataFrame(all_tuning))
                atomic_csv(out / "oof_predictions.csv", pd.concat(all_predictions, ignore_index=True))
    scores = pd.DataFrame(all_rows)
    if len(scores) != expected_tasks:
        raise ValueError("Incomplete nested result.")
    oof = pd.concat(all_predictions, ignore_index=True)
    if not args.smoke:
        if oof.groupby(["model", "variant"]).size().ne(len(train)).any():
            raise ValueError("Incomplete OOF coverage.")
        if oof.duplicated(KEYS + ["model", "variant"]).any():
            raise ValueError("Duplicate OOF predictions.")
    columns = ["rmse", "mae", "r2", "bias", "fit_rmse", "hot_rmse", "cold_rmse", "hot_bias", "cold_bias"]
    summary = scores.groupby(["model", "variant"])[columns].mean().reset_index()
    atomic_csv(out / "summary.csv", summary)
    atomic_csv(out / "outer_std.csv", scores.groupby(["model", "variant"])[["rmse", "mae", "r2"]].std().reset_index())
    pooled = [dict(model=m, variant=v, **metrics(g.SUHII, g.prediction))
              for (m, v), g in oof.groupby(["model", "variant"])]
    atomic_csv(out / "pooled_oof_metrics.csv", pd.DataFrame(pooled))
    yearly = [dict(model=m, variant=v, year=int(yr), **metrics(g.SUHII, g.prediction))
              for (m, v, yr), g in oof.groupby(["model", "variant", "year"])]
    atomic_csv(out / "year_metrics.csv", pd.DataFrame(yearly))
    comparisons = []
    for family, g in scores.groupby("model"):
        proposed = g[g.variant.eq("cell_context_contrast")].set_index("outer_fold")
        for variant in ["cell_only", "cell_context"]:
            base = g[g.variant.eq(variant)].set_index("outer_fold")
            comparisons.append(dict(model=family, baseline=variant,
                proposed="cell_context_contrast",
                outer_folds_improved=int((proposed.rmse < base.rmse).sum()),
                mean_rmse_delta=float((proposed.rmse-base.rmse).mean()),
                mean_hot_rmse_delta=float((proposed.hot_rmse-base.hot_rmse).mean()),
                mean_cold_rmse_delta=float((proposed.cold_rmse-base.cold_rmse).mean())))
    atomic_csv(out / "paired_comparison.csv", pd.DataFrame(comparisons))
    atomic_json(out / "COMPLETED.json", dict(status="SMOKE_ONLY" if args.smoke else "NESTED_DEVELOPMENT_COMPLETE",
        completed_tasks=len(scores), planned_fits=expected_fits, fit_calls_this_invocation=fit_calls,
        selected_model=None, original_holdout_evaluated=False))
    print("\nMEAN OUTER SPATIAL-CV RESULTS\n" + summary.round(5).to_string(index=False), flush=True)
    print("\nPAIRED NESTED COMPARISON\n" + pd.DataFrame(comparisons).round(5).to_string(index=False), flush=True)
    print("RESULTS:", out, flush=True)
    print("SMOKE ONLY: do not report scores." if args.smoke else
          "Nested development evaluation complete. No production model saved or overwritten.", flush=True)


if __name__ == "__main__":
    main()
