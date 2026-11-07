"""Step F: row cleaning. No imputation. Hot/cold rows are NEVER removed (hotspots are the object of study)."""
import numpy as np
import pandas as pd

from .schema import CONT_RAW, DW_ALL


def clean_rows(df: pd.DataFrame, min_valid_frac=0.5, placeholders=(-9999,), aoi_geojson=None):
    log = {"rows_in": int(len(df)), "steps": [], "rows_in_per_year": {int(k): int(v) for k, v in df.groupby("year").size().items()}}
    d = df.copy()
    cols = CONT_RAW + DW_ALL

    def step(name, n_before):
        log["steps"].append({"step": name, "removed": int(n_before - len(d)), "remaining": int(len(d))})

    # 1. placeholder values -> NaN (then removed with the null rows)
    n_ph = int(d[cols].isin(list(placeholders)).sum().sum())
    for p in placeholders:
        d[cols] = d[cols].replace(p, np.nan)
    log["placeholder_cells_found"] = n_ph

    # 2. mostly-masked cells (water / cloud gaps / outside AOI)
    n = len(d)
    d = d[d["valid_frac"] >= min_valid_frac]
    step(f"valid_frac < {min_valid_frac}", n)

    # 3. rows whose TARGET (or LST / coordinates) is missing can never be repaired: removed. Missing FEATURES stay NaN
    #    here; pipeline.py imputes them (nearest cell) or, if imputation is off, removes those rows.
    n = len(d)
    d = d.dropna(subset=["SUHII", "LST", "LST_ref", "lon", "lat", "cell_id"])
    step("missing target / LST / coordinates", n)

    # 4. optional: outside AOI polygon (centroid test)
    if aoi_geojson:
        try:
            import json
            from shapely.geometry import Point, shape
            gj = json.load(open(aoi_geojson))
            geoms = [shape(f["geometry"]) for f in gj["features"]] if gj.get("type") == "FeatureCollection" else [shape(gj)]
            from shapely.ops import unary_union
            poly = unary_union(geoms)
            n = len(d)
            d = d[[poly.contains(Point(x, y)) or poly.touches(Point(x, y)) for x, y in zip(d["lon"], d["lat"])]]
            step("centroid outside AOI polygon", n)
        except ImportError:
            log["aoi_check"] = "skipped (shapely not installed)"
    else:
        log["aoi_check"] = "skipped (no AOI geojson given; grid was already filtered to AOI in GEE)"

    log["rows_out"] = int(len(d))
    log["rows_out_per_year"] = {int(k): int(v) for k, v in d.groupby("year").size().items()}
    return d.reset_index(drop=True), log
