#!/usr/bin/env python
"""Check what every source covers, plan each year's window, then start one Earth Engine batch export per year.

Default window (configs/pipeline.yaml): full calendar years; the still-running year ends at the latest COMPLETE month
that ALL sources cover (the binding source is printed).

  python scripts/01_export_gee.py --preflight-only     # show coverage + the exact windows, start nothing
  python scripts/01_export_gee.py                      # start the exports
  python scripts/01_export_gee.py --years 2024 2025 2026 --dry-run   # plan without touching Earth Engine
"""
import argparse, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uhi.config import get_city, load_configs, plan_periods, resolve_years, utm_epsg  # noqa: E402


def show(plan):
    for y, p in plan.items():
        tag = "PARTIAL (ends at the latest complete month all sources cover)" if p["partial"] else "full"
        print(f"   {y}: {p['start']} .. {p['end_excl']} (excl)  = {p['months']} months  [{tag}]  tropomi {p['tropomi'][0]}..{p['tropomi'][1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--years", nargs="+", type=int)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pipe, cities = load_configs()
    city = a.city or pipe["city"]
    cc = get_city(cities, city)
    if a.years:
        pipe["years"] = a.years
    years = resolve_years(pipe, cc)
    print(f"[plan] {city} years={years} window={pipe.get('window')} align_months={pipe.get('align_months', False)} "
          f"utm=EPSG:{utm_epsg(cc)} cell={pipe['grid']['cell_size_m']} m")
    if a.dry_run:
        show(plan_periods(pipe, cc, years))
        return

    from uhi.gee.availability import covered_through, latest_dates
    from uhi.gee.init import init_ee
    ee = init_ee(pipe["export"]["gee_project"])
    latest = latest_dates(ee)
    cov, binding, per = covered_through(latest)
    print("[coverage] imagery exists up to (excl):")
    for k, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"   {k:<15} {v}{'   <- binding source' if k == binding else ''}")
    plan = plan_periods(pipe, cc, years, covered_through=cov)
    show(plan)
    short = [y for y, p in plan.items() if not p["partial"] and cov < p["end_excl"]]
    if short:
        sys.exit(f"[abort] a source does not cover the full year(s) {short}; check the coverage table above.")
    if a.preflight_only:
        return
    from uhi.gee.export import run_export
    for y in years:
        e = run_export(ee, pipe, city, cc, y, plan[y])
        print(f"[started] {y} ({e['months_in_window']} months) -> {e['asset_id']} (Landsat scenes: {e['landsat_scenes_used']})")
    print("Next: wait for the tasks to finish, then python scripts/02_fetch_gee.py")


if __name__ == "__main__":
    main()
