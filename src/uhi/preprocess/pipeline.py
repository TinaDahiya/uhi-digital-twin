"""Preprocessing pipeline (spec section 1, steps F -> I; multi-year).

Ordering note: the spec diagram lists the split (I) after the filters (H), but spec section 8 rule 3 forbids test cells
from fitting any filter/transform. We split first, then fit winsorization, Spearman and VIF on TRAIN rows only.

Only TWO files are persisted (see save_outputs): the ML-ready table and `preprocess_state.json` (frozen split, final
features, winsor bounds, decision log, gate results). All other diagnostics are recomputed on demand by the notebook.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import filters as F
from .clean import clean_rows
from .impute import feature_units, impute_nearest
from .load import derive_dw_label, load_raw
from .schema import CONT_RAW, DW_ALL, DW_LABEL, LEAKY, META, TARGET
from .split import assign_blocks, assign_split, compute_bounds
from .transforms import Winsorizer, add_log_transforms, year_center
from .validate import run_gates

STATE_VERSION = 4


def _dec(log, feature, stage, action, reason, **metrics):
    log.append({"feature": feature, "stage": stage, "action": action, "reason": reason, "metrics": metrics})


def _core(raw, pp, aoi_geojson=None, frozen=None):
    d, clean_log = clean_rows(raw, pp["min_valid_frac"], pp["placeholder_values"], aoi_geojson)
    sp = pp["split"]
    bounds = (frozen or {}).get("bounds") or compute_bounds(raw)
    d["block_id"] = assign_blocks(d, sp["n_blocks"], bounds)
    d["split"], block_map = assign_split(d, sp["train_frac"], sp["seed"], (frozen or {}).get("block_map"))

    # Missing FEATURE values: nearest-cell imputation inside the same year and the same split (or removal if disabled).
    imp = pp.get("impute", {})
    impute_log = None
    if imp.get("enabled", True):
        n0 = len(d)
        d, impute_log = impute_nearest(d, feature_units(), imp.get("max_distance_km", 3.0), imp.get("max_null_share", 0.3))
        r1, r2 = impute_log["rows_removed_too_many_missing"], impute_log["rows_removed_no_donor_in_range"]
        clean_log["steps"].append({"step": f"too many missing features (> {imp.get('max_null_share', 0.3):.0%} of groups)", "removed": r1, "remaining": n0 - r1})
        clean_log["steps"].append({"step": f"no donor cell within {imp.get('max_distance_km', 3.0)} km for a missing feature", "removed": r2, "remaining": len(d)})
    else:
        n0 = len(d)
        d = d.dropna(subset=[c for c in CONT_RAW + DW_ALL if c not in ("SUHII", "LST", "LST_ref")])
        d["n_imputed"] = 0
        clean_log["steps"].append({"step": "missing feature values (imputation disabled)", "removed": n0 - len(d), "remaining": len(d)})
    clean_log["rows_out"] = int(len(d))
    clean_log["rows_out_per_year"] = {int(k): int(v) for k, v in d.groupby("year").size().items()}
    d = derive_dw_label(d)
    d[DW_LABEL] = d[DW_LABEL].astype(int)
    d = add_log_transforms(d)
    yc_cols = list(pp.get("year_center_features") or [])
    year_means = {}
    if yc_cols:
        d, year_means = year_center(d, yc_cols, d.split == "train")
    train = d[d.split == "train"]
    rule, tiers = pp.get("spearman_keep_rule", "priority"), pp.get("spearman_priority_tiers")

    dec = []
    for c in LEAKY:
        _dec(dec, c, "leakage_guard", "excluded", "SUHII = LST - local rural LST; any LST-derived column would leak the target (spec 8.1).")
    _dec(dec, "NTL", "transform", "replaced", "Raw radiance is heavily right-skewed; replaced by NTL_log = log1p(NTL).")
    _dec(dec, "Pop", "transform", "replaced", "Raw density is heavily right-skewed; replaced by Pop_log = log1p(Pop).")
    for c in ("lon", "lat"):
        _dec(dec, c, "metadata", "excluded", "Coordinates enter only the separately labelled spatial-aware model, never the driver-only baseline (spec 8.4).")
    for c in ("city", "year", "cell_id", "block_id", "valid_frac", "split", DW_LABEL):
        _dec(dec, c, "metadata", "excluded", "Identifier / bookkeeping / map-only column, not a physical driver.")

    for c in yc_cols:
        _dec(dec, c, "year_centering", "transformed",
             "Takes one value per city per year (coarse 1-7 km / 5 km grid), so with few years it would let the model identify the "
             "year. Centred on that year's training-cell mean: only within-year spatial variation is kept.")
    feats, removed_up = F.candidate_features(pp["dw_drop_classes"], pp["dw_reference_class"])
    shares = {}
    for col, kind in removed_up:
        sh = float(train[col].mean())
        if kind == "class":
            shares[col] = sh
            ok = sh <= pp["dw_drop_max_share"]
            _dec(dec, col, "dw_class", "dropped",
                 f"Class averages {sh:.4f} cover in the training cells. " +
                 ("Negligible, so dropping it loses no information." if ok else
                  f"NOT negligible (> {pp['dw_drop_max_share']}); dropping it discards real signal - reconsider."), mean_share=sh)
        else:
            _dec(dec, col, "dw_reference", "dropped",
                 f"Reference category for the compositional Dynamic World probabilities (they sum to ~1, so one must go to avoid "
                 f"perfect collinearity). Mean share {sh:.3f}; its effect is implicit in the others. Simulations must keep the "
                 f"class shares summing to 1.", mean_share=sh)

    wz = Winsorizer(*pp["winsor_quantiles"]).fit(train, feats)
    dw = wz.transform(d)
    train_w = dw[dw.split == "train"]
    clipped = {c: float(((train[c] < lo) | (train[c] > hi)).mean()) for c, (lo, hi) in wz.bounds_.items()}

    feats, zv = F.drop_zero_variance(train_w, feats)
    for r in zv:
        _dec(dec, r["feature"], "zero_variance", "dropped", r["reason"])
    candidates = list(feats)
    corr_all = train_w[candidates].corr(method="spearman")
    vif_start = pd.Series(F.compute_vif(train_w[candidates].to_numpy(float)), index=candidates)

    feats, sp_log = F.spearman_filter(train_w, feats, pp["spearman_threshold"], rule, train_w[TARGET], tiers)
    for r in sp_log:
        _dec(dec, r["feature"], "spearman", "dropped", r["reason"], rho=r["rho"], kept_instead=r["kept_instead"])
    feats, vif_log, vif_final = F.vif_filter(train_w, feats, pp["vif_threshold"], rule, tiers)
    for r in vif_log:
        _dec(dec, r["feature"], "vif", "dropped", r["reason"], vif=r["vif"])
    vmap = dict(zip(vif_final.feature, vif_final.VIF))
    for c in feats:
        t = float(train_w[c].corr(train_w[TARGET], method="spearman"))
        _dec(dec, c, "final", "kept", f"Survived every filter. VIF={vmap[c]:.2f}. Diagnostic only (not used to select): "
             f"Spearman rho with SUHII on train = {t:+.3f}.", vif=float(vmap[c]), rho_with_target=t)

    ml = dw[META + [TARGET] + feats + [DW_LABEL]].reset_index(drop=True)
    return dict(clean_df=d, ml=ml, feats=feats, clean_log=clean_log, decisions=dec, winsor_bounds=wz.bounds_,
                winsor_clipped_frac=clipped, corr_all=corr_all, vif_start=vif_start, vif_final=vif_final,
                block_map=block_map, bounds=bounds, impute_log=impute_log, dw_dropped_shares=shares, year_means=year_means, yc_cols=yc_cols,
                split_info={"train_row_frac": float((ml.split == "train").mean())})


def _hash(df):
    return int(pd.util.hash_pandas_object(df, index=False).sum()) ^ hash(tuple(df.columns))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def load_state(out_dir, city):
    p = Path(out_dir) / city / "preprocess_state.json"
    return json.loads(p.read_text()) if p.exists() else None


def preprocess(pipe, city, raw_dir, years=None, aoi_geojson=None, state=None, expected_years=None):
    """Run everything IN MEMORY. Nothing is written. `state` (previous preprocess_state.json) freezes the split."""
    pp = pipe["preprocess"]
    frozen = None
    if state:
        s = state["split"]
        if s["n_blocks"] != pp["split"]["n_blocks"]:
            raise ValueError("n_blocks changed since the frozen split was written; delete preprocess_state.json to re-split.")
        frozen = {"block_map": s["block_map"], "bounds": s["bounds"]}
    raw = load_raw(raw_dir, city, years)
    elog = Path(raw_dir) / f"{city}_export_log.json"
    provenance = json.loads(elog.read_text()) if elog.exists() else None
    res = _core(raw, pp, aoi_geojson, frozen)
    repro_ok = _hash(res["ml"]) == _hash(_core(raw, pp, aoi_geojson, frozen)["ml"])
    res["raw"] = raw
    res["gates"] = run_gates(dict(
        raw_rows=len(raw), clean_df=res["clean_df"], clean_log=res["clean_log"], ml=res["ml"], feats=res["feats"],
        gates_cfg=pp["gates"], min_rows_per_block=pp["min_rows_per_block"], expected_pop_max=pp.get("expected_pop_max"),
        repro_ok=repro_ok, expected_years=expected_years, dw_dropped_shares=res["dw_dropped_shares"],
        dw_max_share=pp["dw_drop_max_share"], split_info=res["split_info"], impute_log=res["impute_log"],
        impute_warn=pp.get("impute", {}).get("warn_if_imputed_share", 0.05), provenance=provenance))
    res["provenance"] = provenance
    res["years"] = sorted(int(y) for y in raw.year.unique())
    return res


def _feature_spec(res):
    """How each model input is produced from raw values: what a simulator must apply to a scenario value."""
    spec = []
    for f in res["feats"]:
        lo, hi = res["winsor_bounds"][f]
        if f in ("NTL_log", "Pop_log"):
            t, src = "log1p(max(x, 0))", f.replace("_log", "")
        elif f in res["yc_cols"]:
            t, src = "x - year_mean[year][feature]", f
        else:
            t, src = "identity", f
        spec.append({"feature": f, "source_column": src, "transform": t, "winsor_lower": lo, "winsor_upper": hi,
                     "applied_order": "transform -> clip to [winsor_lower, winsor_upper]"})
    return spec


def build_state(res, pipe, city, provenance=None):
    pp = pipe["preprocess"]
    return _jsonable({
        "state_version": STATE_VERSION, "city": city, "years": res["years"], "target": TARGET,
        "final_features": res["feats"], "feature_spec": _feature_spec(res),
        "year_centering": {"features": res["yc_cols"], "year_means_train_cells": res["year_means"]},
        "non_feature_columns": META + [DW_LABEL], "excluded_leaky": LEAKY,
        "split": {"n_blocks": pp["split"]["n_blocks"], "train_frac_requested": pp["split"]["train_frac"],
                  "seed": pp["split"]["seed"], "bounds": res["bounds"], "block_map": res["block_map"],
                  "train_row_frac": res["split_info"]["train_row_frac"]},
        "winsorization": {"quantiles": pp["winsor_quantiles"], "fit_on": "train split only",
                          "bounds": res["winsor_bounds"], "train_clipped_fraction": res["winsor_clipped_frac"]},
        "selection_rule": {"spearman_keep_rule": pp["spearman_keep_rule"], "tiers": pp.get("spearman_priority_tiers"),
                           "spearman_threshold": pp["spearman_threshold"], "vif_threshold": pp["vif_threshold"],
                           "dw_reference_class": pp["dw_reference_class"]},
        "imputation": {"settings": pp.get("impute"), "log": res["impute_log"],
                       "target_imputed": False, "note": "SUHII and LST are never imputed; features only."},
        "row_cleaning": res["clean_log"], "decisions": res["decisions"], "validation": res["gates"],
        "provenance": provenance})


def save_table(df, path):
    path = Path(path)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception:  # pyarrow missing
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")


def run_preprocessing(pipe, city, raw_dir, out_dir, years=None, aoi_geojson=None, expected_years=None):
    state = load_state(out_dir, city)
    res = preprocess(pipe, city, raw_dir, years, aoi_geojson, state, expected_years)
    prov = res["provenance"]
    out = Path(out_dir) / city
    out.mkdir(parents=True, exist_ok=True)
    table = save_table(res["ml"], out / f"{city}_ml_ready")
    (out / "preprocess_state.json").write_text(json.dumps(build_state(res, pipe, city, prov), indent=2))
    dec = res["decisions"]
    return {"city": city, "years": res["years"], "rows": len(res["ml"]),
            "rows_per_year": res["ml"].groupby("year").size().to_dict(), "features_kept": res["feats"],
            "features_dropped": [x["feature"] for x in dec if x["action"] == "dropped"],
            "gates": {g["gate"]: g["status"] for g in res["gates"]},
            "files_written": [str(table), str(out / "preprocess_state.json")],
            "split_reused_from_state": bool(state)}
