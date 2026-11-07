"""Stage 1+2 driver: build the composite, aggregate to 1 km cells, export CSVs to Drive."""
import json
from datetime import datetime
from pathlib import Path

from ..config import months_between, utm_epsg
from . import landsat, sources, suhii as suhii_mod
from .grid import build_grid

CONT_BANDS = ["SUHII", "LST", "LST_ref", "NDVI", "NDBI", "NDWI", "BSI", "Albedo", "NTL", "Pop",
              "SO2", "CO", "NO2", "O3", "AAI", "Rain_period"] + sources.DW_RENAME


def get_aoi(ee, city_cfg):
    a = city_cfg["aoi"]
    if a["source"] == "custom":
        raise NotImplementedError(f"Custom AOI required: {a.get('note')}")
    fc = ee.FeatureCollection(a["source"]).filter(ee.Filter.inList(a["filter_field"], a["filter_values"]))
    return fc.geometry()


def build_images(ee, aoi, city_cfg, pipe, year, period):
    s0, s1 = period["start"], period["end_excl"]
    t0, t1 = period["tropomi"]
    crs = f"EPSG:{utm_epsg(city_cfg)}"
    ls, n_scenes = landsat.landsat_composite(ee, aoi, s0, s1, pipe["landsat"]["max_cloud"])
    sh = suhii_mod.compute_suhii(ee, ls, pipe["suhii"], crs)
    pop, pop_year = sources.gpw_population(ee, year)
    probs, label = sources.dynamic_world(ee, aoi, s0, s1)
    rain = sources.chirps_rain(ee, s0, s1, pipe["rain"]["include_30d_pre"])
    bands = [sh, ls.select(["NDVI", "NDBI", "NDWI", "BSI", "Albedo"]).addBands(ls.select("LST")),
             sources.viirs_ntl(ee, aoi, s0, s1), pop, sources.tropomi(ee, t0, t1), rain, probs]
    img = bands[0]
    for b in bands[1:]:
        img = img.addBands(b)
    img = img.select(CONT_BANDS + (["Rain_30d_pre"] if pipe["rain"]["include_30d_pre"] else []))
    # valid_frac = share of the FULL cell that has a valid SUHII pixel INSIDE the AOI. It must be built from
    # unmasked bands and added AFTER clip(), otherwise clip() masks the outside-AOI pixels out of the
    # denominator and every border cell looks fully valid.
    # unmask(0, False): without sameFootprint=False, unmask() on a clipped image keeps the clip footprint, the outside
    # pixels stay masked and drop out of the denominator again (this was the remaining cause of valid_frac ~ 1).
    inside = ee.Image.constant(1).clip(aoi).unmask(0, False)
    valid = img.select("SUHII").mask().unmask(0, False).And(inside).rename("valid_frac")
    img = img.clip(aoi).addBands(valid).toFloat()
    return img, label, n_scenes, pop_year, (s0, s1, t0, t1)


def asset_id(pipe, city_key, year):
    ex = pipe["export"]
    if not ex.get("gee_project"):
        raise ValueError("Set export.gee_project in configs/pipeline.yaml (your Earth Engine cloud project id).")
    return f"projects/{ex['gee_project']}/assets/{ex['asset_folder']}/{city_key}_{year}"


def ensure_asset_folder(ee, pipe):
    ex = pipe["export"]
    folder = f"projects/{ex['gee_project']}/assets/{ex['asset_folder']}"
    try:
        ee.data.getAsset(folder)
    except Exception:
        ee.data.createAsset({"type": "FOLDER"}, folder)


def run_export(ee, pipe, city_key, city_cfg, year, period, log_dir="data/raw"):
    """Start ONE batch task for `year`: 1 km cell means of every band -> an Earth Engine table asset."""
    epsg = utm_epsg(city_cfg)
    aoi = get_aoi(ee, city_cfg)
    grid, n_cells = build_grid(ee, aoi, epsg, pipe["grid"]["cell_size_m"])
    img, _label, n_scenes, pop_year, (s0, s1, t0, t1) = build_images(ee, aoi, city_cfg, pipe, year, period)
    ex = pipe["export"]
    fc = img.reduceRegions(collection=grid, reducer=ee.Reducer.mean(), scale=ex["scale_m"],
                           crs=f"EPSG:{epsg}", tileScale=ex["tile_scale"])
    cols = ["cell_id", "lon", "lat", "valid_frac"] + CONT_BANDS + (["Rain_30d_pre"] if pipe["rain"]["include_30d_pre"] else [])
    # A table asset needs a geometry on every feature. Keep only the attribute columns and use the cell
    # centroid as a light point geometry (the fetch step reads attributes only).
    fc = fc.map(lambda f: ee.Feature(ee.Geometry.Point([f.get("lon"), f.get("lat")]), f.toDictionary(cols)))
    aid = asset_id(pipe, city_key, year)
    ensure_asset_folder(ee, pipe)
    task = ee.batch.Export.table.toAsset(collection=fc, description=f"uhi_{city_key}_{year}", assetId=aid)
    task.start()

    entry = {"year": year, "asset_id": aid, "columns": cols, "window": [str(s0), str(s1)], "months_in_window": months_between(s0, s1),
             "partial_year": bool(period["partial"]), "window_mode": pipe.get("window", "full_year"),
             "tropomi_window": [str(t0), str(t1)], "epsg": epsg, "cell_size_m": pipe["grid"]["cell_size_m"],
             "grid_cells_before_aoi_filter": n_cells, "landsat_scenes_used": n_scenes.getInfo(),
             "gpw_year_used": pop_year, "export_started_utc": datetime.utcnow().isoformat(timespec="seconds")}
    path = Path(log_dir) / f"{city_key}_export_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(path.read_text()) if path.exists() else {}
    log[str(year)] = entry
    path.write_text(json.dumps(log, indent=2))
    return entry
