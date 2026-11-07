"""Tiny in-code fixture (NOT a data generator): 3 years x 24x24 cells in the per-year CSV schema, just enough to
exercise the pipeline's logic in tests. Contains no scientific claim."""
import numpy as np
import pandas as pd

DW = ["DW_water", "DW_trees", "DW_grass", "DW_flooded_vegetation", "DW_crops", "DW_shrub", "DW_built", "DW_bare", "DW_snow_and_ice"]


def year_frame(year, seed, n=24):
    r = np.random.default_rng(seed + year)
    lon, lat = (a.ravel() for a in np.meshgrid(np.linspace(76.85, 77.35, n), np.linspace(28.40, 28.88, n)))
    u = np.clip(np.exp(-(np.hypot(lon - 77.2, lat - 28.63) / 0.2) ** 2) + r.normal(0, .08, lon.size), 0, 1)
    ndvi = .55 - .45 * u + r.normal(0, .04, lon.size)
    ndbi = -.25 + .4 * u + r.normal(0, .03, lon.size)
    logit = np.column_stack([-3 + 0 * u, .5 - 2 * u, .2 - u, -9 + 0 * u, 1 - u, .3 - u, -2 + 4 * u, -.5 + 0 * u, -9 + 0 * u]) + r.normal(0, .3, (lon.size, 9))
    p = np.exp(logit); p /= p.sum(1, keepdims=True)
    lst = 41 + 6 * u + r.normal(0, .7, lon.size); ref = 39.5 + r.normal(0, .3, lon.size)
    d = pd.DataFrame({"cell_id": [f"c{i}" for i in range(lon.size)], "lon": lon, "lat": lat, "valid_frac": 1.0,
                      "SUHII": lst - ref, "LST": lst, "LST_ref": ref, "NDVI": ndvi, "NDBI": ndbi,
                      "NDWI": -.2 - .9 * ndvi + r.normal(0, .01, lon.size), "BSI": ndbi + r.normal(0, .01, lon.size),
                      "Albedo": .18 - .05 * u + r.normal(0, .01, lon.size), "NTL": np.exp(.3 + 3 * u + r.normal(0, .4, lon.size)),
                      "Pop": np.exp(3 + 6 * u + r.normal(0, .6, lon.size)), "SO2": r.normal(2e-4, 5e-5, lon.size),
                      "CO": .035 + .01 * u + r.normal(0, 5e-4, lon.size), "NO2": 8e-5 + 1e-4 * r.random(lon.size),
                      "O3": .125 + r.normal(0, 3e-4, lon.size), "AAI": .6 + r.random(lon.size), "Rain_period": 20 + 15 * (lat - 28.4) + r.normal(0, 1, lon.size)})
    for k, c in enumerate(DW):
        d[c] = p[:, k]
    return d


def write_years(raw_dir, city="delhi", years=(2024, 2025, 2026), seed=0):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for y in years:
        year_frame(y, seed).to_csv(raw_dir / f"{city}_{y}.csv", index=False)
