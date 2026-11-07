#!/usr/bin/env python
"""Download finished Earth Engine assets into data/raw/<city>_<year>.csv (one small file per year)."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uhi.config import load_configs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args()
    pipe, _ = load_configs()
    city = a.city or pipe["city"]
    log_path = Path(a.raw_dir) / f"{city}_export_log.json"
    if not log_path.exists():
        sys.exit(f"{log_path} not found - run scripts/01_export_gee.py first")
    log = json.loads(log_path.read_text())
    from uhi.gee.fetch import fetch_asset
    from uhi.gee.init import init_ee
    ee = init_ee(pipe["export"]["gee_project"])
    for year, e in sorted(log.items()):
        out = Path(a.raw_dir) / f"{city}_{year}.csv"
        if out.exists():
            print(f"[skip] {out} exists")
            continue
        try:
            df = fetch_asset(ee, e["asset_id"], e["columns"])
        except Exception as ex:  # asset not finished yet
            print(f"[wait] {year}: {str(ex)[:120]}")
            continue
        df.to_csv(out, index=False)
        print(f"[ok] {out} ({len(df)} cells)")
        if pipe["export"]["delete_asset_after_fetch"]:
            ee.data.deleteAsset(e["asset_id"])


if __name__ == "__main__":
    main()
