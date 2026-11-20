"""Freeze a historical-time validation protocol and acquire Delhi 2023 separately.
No model evaluation or accuracy claim. Requires existing script 27 for the frozen
search space. External-source operations run only for explicit modes.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def build_protocol(root):
    sys.path.insert(0, str(root / "src"))
    from uhi.config import load_configs, get_city, plan_periods
    pipe, cities = load_configs(root / "configs/pipeline.yaml", root / "configs/cities.yaml")
    pipe = copy.deepcopy(pipe)
    city, year = "delhi", 2023
    state_path = root / "data/processed/delhi/preprocess_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    if year in state["years"]:
        raise ValueError("2023 is already in the frozen training years: not an additional year.")
    if (root / "data/raw/delhi_2023.csv").exists():
        raise ValueError("Existing raw Delhi 2023 found. Its prior use must be checked before calling it unseen.")
    if pipe.get("window") != "full_year" or pipe.get("tropomi", {}).get("mode") != "period":
        raise ValueError("This protocol expects the existing full-year / period-source setup.")
    cc = get_city(cities, city)
    if cc["aoi"]["source"] == "custom":
        raise ValueError("A validated city boundary is required.")
    nested_path = root / "scripts/27_nested_spatial_comparison.py"
    spec = importlib.util.spec_from_file_location("frozen_nested_protocol", nested_path)
    nested = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nested)
    params = {k: candidates for k, (_, candidates) in nested.grids(2, smoke=False).items()}
    tracked = [state_path, root / "configs/pipeline.yaml", root / "configs/cities.yaml", root / "src/uhi/config.py",
               root / "models/experiments/spatial_cv_folds_v1.csv", nested_path]
    tracked += [root / f"data/raw/delhi_{y}.csv" for y in state["years"]]
    tracked += sorted((root / "src/uhi/preprocess").glob("*.py"))
    tracked += sorted((root / "src/uhi/gee").glob("*.py"))
    files = {str(p.relative_to(root)): digest(p) for p in tracked}
    pipe["years"] = [year]
    # Keep the same full-year acquisition policy; 2023 is complete.
    period = plan_periods(pipe, cc, [year])[year]
    if period["partial"] or period["months"] != 12:
        raise ValueError("Expected all 12 months of 2023.")
    protocol = dict(schema=1, city=city, validation_year=year,
        training_years=state["years"], protocol_status="FROZEN_CANDIDATE_TEMPORAL_VALIDATION",
        prior_use_status="Not in supplied training snapshot; use in earlier project work still needs verification.",
        task="Historical-time transfer to 2023 at potentially overlapping Delhi locations; not future forecasting or new-city validation.",
        primary_comparison="XGBoost: cell_context_contrast versus cell_context",
        secondary_comparisons="All four learners and all three feature sets; report all, not only winners.",
        primary_metric="RMSE; also report MAE, R2, overall/hot/cold bias and RMSE.",
        families=nested.FAMILIES, variants=nested.VARIANTS,
        radius_km=3., target="SUHII using unchanged project acquisition definition",
        full_fit_selection="Same three configurations per family/variant; select by 3-fold spatial CV on original fitting rows only, then refit those rows.",
        candidate_parameters=params,
        external_labels_policy="Never use 2023 labels for tuning, filtering decisions, clipping, feature selection, thresholds or model selection. Score once after all estimators and transforms are frozen.",
        year_centering_policy="Training years use their fitting-row means. For unseen 2023, use the 2023 predictor-only study-area mean for configured year-centered columns; predeclared transductive adaptation, no labels.",
        context_policy="Same-year available predictor imagery for each domain; own cell excluded; no thermal/target neighbour features. 2023 context built from 2023 imagery only.",
        winsor_policy="Use fitted original-training bounds, never fit external-label-driven bounds.",
        tail_policy="Thresholds fixed from original fitting targets before external scoring.",
        missing_policy="Do not impute or drop hot/cold targets. Report invalid/missing rows and stop for a revised, separately documented protocol if required predictors are missing.",
        original_holdout_policy="Do not use original Delhi spatial-holdout labels for fitting or selection.",
        limitations=[
            "2023 predates training years: retrospective temporal transfer, not prospective forecasting.",
            "Locations may overlap training years: not a spatially independent external city.",
            "2026 training composite is partial while 2023 is full-year; period mismatch must be reported.",
            "Target construction uses vegetation/built-up rules also present as predictors; predictive skill is not a causal mitigation effect.",
            "This protocol does not establish prior-art novelty."],
        source_hashes=files,
        acquisition_config=pipe, city_config=cc,
        period={k:([str(x) for x in v] if isinstance(v,tuple) else str(v) if hasattr(v,"isoformat") else v)
                for k,v in period.items()},
        acquisition_script_sha256=digest(Path(__file__)))
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    pipe["export"]["asset_folder"] = f"uhi_validation_2023_{signature[:10]}"
    protocol["acquisition_config"] = pipe
    # Fingerprint the final object; asset suffix derives from the pre-folder fingerprint.
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, signature, pipe, cc, period


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=["prepare","preflight","export","fetch"], default="prepare")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol, signature, pipe, cc, period = build_protocol(root)
    directory = root / "validation/delhi_2023"
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "protocol.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError("Frozen protocol inputs/source changed. Existing protocol was not overwritten.")
    else:
        atomic_json(frozen, dict(signature=signature,
            frozen_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol))
    raw_dir = directory / "raw"
    print("PROTOCOL:", frozen, flush=True)
    print("YEAR: 2023 | complete calendar year | original training years:", protocol["training_years"], flush=True)
    print("MODE:", args.mode, flush=True)
    print("This is candidate historical-time validation, not future forecasting.", flush=True)
    if args.mode == "prepare":
        print("Offline preparation complete. No Earth Engine request, export, model fit or score performed.")
        return
    from uhi.gee.init import init_ee
    ee = init_ee(pipe["export"]["gee_project"])
    if args.mode in ["preflight", "export"]:
        from uhi.gee.availability import latest_dates, covered_through
        from uhi.gee.export import get_aoi, asset_id
        latest = latest_dates(ee)
        coverage, binding, per_source = covered_through(latest)
        print("Common source coverage ends at:", coverage, "| binding source:", binding, flush=True)
        for name, end in sorted(per_source.items()):
            print(" ", name, end, flush=True)
        if coverage < period["end_excl"]:
            raise ValueError("Not all sources cover the complete validation year.")
        if float(get_aoi(ee, cc).area(100).getInfo()) <= 0:
            raise ValueError("Delhi boundary is empty.")
        print("Validation asset:", asset_id(pipe, "delhi", 2023), flush=True)
        if args.mode == "preflight":
            print("Preflight complete. No export started.")
            return
        guard = directory / "export_request.json"
        if guard.exists():
            raise ValueError("An export was already requested. Use --mode fetch; do not launch a duplicate.")
        atomic_json(guard, dict(status="REQUESTING", signature=signature,
            asset_id=asset_id(pipe, "delhi", 2023)))
        from uhi.gee.export import run_export
        try:
            entry = run_export(ee, pipe, "delhi", cc, 2023, period, log_dir=str(raw_dir))
        except Exception as error:
            atomic_json(guard, dict(status="REVIEW_REQUIRED", signature=signature,
                error=str(error),
                note="An export may have started before an error. Inspect Earth Engine tasks before retrying."))
            raise
        atomic_json(guard, dict(status="REQUESTED", signature=signature, entry=entry))
        print("Export requested. When the remote task finishes, run --mode fetch.", flush=True)
        return
    log_path = raw_dir / "delhi_export_log.json"
    if not log_path.exists():
        raise ValueError("No separate validation export log. Run --mode export first.")
    entry = json.loads(log_path.read_text())["2023"]
    from uhi.gee.export import asset_id
    if entry["asset_id"] != asset_id(pipe, "delhi", 2023):
        raise ValueError("Validation asset does not match frozen protocol.")
    dest = raw_dir / "delhi_2023.csv"
    if dest.exists():
        print("Validation CSV already exists; not overwritten:", dest)
        return
    from uhi.gee.fetch import fetch_asset
    # An unfinished asset may raise a service error; do not pretend data exists.
    frame = fetch_asset(ee, entry["asset_id"], entry["columns"])
    temp = dest.with_suffix(".csv.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(dest)
    atomic_json(directory / "DATA_RECEIPT.json", dict(signature=signature,
        file=dest.name, sha256=digest(dest), rows=len(frame),
        fetched_utc=datetime.now(timezone.utc).isoformat(),
        note="No model scoring performed; prior use remains to be verified."))
    print("Downloaded validation data:", dest, "| rows:", len(frame))
    print("No validation scores calculated. Keep labels out of tuning.")


if __name__ == "__main__":
    main()
