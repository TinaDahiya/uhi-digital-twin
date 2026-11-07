#!/usr/bin/env python
"""Raw per-year CSVs -> ONE ML-ready table + preprocess_state.json. Review the decisions in notebooks/01_preprocessing_review.ipynb."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uhi.config import get_city, load_configs, resolve_years  # noqa: E402
from uhi.preprocess.pipeline import run_preprocessing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--years", nargs="+", type=int, help="default: configs/pipeline.yaml (latest n_years)")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--aoi-geojson")
    a = ap.parse_args()
    pipe, cities = load_configs()
    city = a.city or pipe["city"]
    if a.years:
        pipe["years"] = a.years
    years = resolve_years(pipe, get_city(cities, city))
    print(json.dumps(run_preprocessing(pipe, city, a.raw_dir, a.out_dir, years, a.aoi_geojson, years), indent=2))


if __name__ == "__main__":
    main()
