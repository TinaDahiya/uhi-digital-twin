"""Nearest-cell imputation of missing FEATURE values.

Rules that keep it honest:
* The target (SUHII) and LST are never imputed (rows missing them are removed in clean.py).
* A donor must be in the SAME year and the SAME split, so no train/test information crosses over.
* A value is copied from the single NEAREST cell that has it, and only if that cell is within `max_distance_km`;
  otherwise the row is removed (never extrapolated).
* Columns that must stay consistent (the nine Dynamic World shares, which sum to 1) are copied together from ONE donor.
* A row missing more than `max_null_share` of its feature groups is removed instead of repaired.
* A feature that is entirely missing for a whole year cannot be imputed: that is an export problem and raises an error.
"""
import numpy as np
import pandas as pd

from .schema import DW_ALL

SINGLE = ["NDVI", "NDBI", "NDWI", "BSI", "Albedo", "NTL", "Pop", "SO2", "CO", "NO2", "O3", "AAI", "Rain_period"]


def feature_units():
    """unit name -> columns copied together from one donor."""
    units = {c: [c] for c in SINGLE}
    units["DW"] = list(DW_ALL)
    return units


def _xy_km(d):
    lat = d["lat"].to_numpy(float)
    return np.column_stack([d["lon"].to_numpy(float) * 111.32 * np.cos(np.deg2rad(lat)), lat * 110.57])


def impute_nearest(df: pd.DataFrame, units=None, max_distance_km=3.0, max_null_share=0.3):
    """Returns (frame with `n_imputed`, log). Expects `year`, `split`, `cell_id`, `lon`, `lat` columns."""
    units = units or feature_units()
    d = df.sort_values(["year", "cell_id"], kind="stable").reset_index(drop=True)   # deterministic donor order
    xy = _xy_km(d)
    nulls = pd.DataFrame({u: d[cols].isna().any(axis=1) for u, cols in units.items()})

    for u in units:                                   # fail fast: nothing to copy from
        for y, g in d.groupby("year"):
            if nulls.loc[g.index, u].all():
                raise ValueError(f"Feature '{u}' has no valid value in any cell of {y}; this cannot be imputed. "
                                 f"Check the Earth Engine export for that year/source.")

    too_many = (nulls.sum(axis=1) / len(units)) > max_null_share
    drop_no_donor = np.zeros(len(d), bool)
    n_imp = np.zeros(len(d), int)
    per_unit = {u: {"imputed": 0, "dist_km": []} for u in units}

    for (_, _), grp in d.groupby(["year", "split"]):
        idx = grp.index.to_numpy()
        for u, cols in units.items():
            miss = nulls.loc[idx, u].to_numpy()
            targets = idx[miss & ~too_many.loc[idx].to_numpy()]
            if targets.size == 0:
                continue
            donors = idx[~miss]
            for t in targets:
                if donors.size == 0:
                    drop_no_donor[t] = True
                    continue
                dist = np.hypot(*(xy[donors] - xy[t]).T)
                j = int(np.argmin(dist))                 # ties -> first donor = smallest cell_id
                if dist[j] > max_distance_km:
                    drop_no_donor[t] = True
                    continue
                d.loc[t, cols] = d.loc[donors[j], cols].to_numpy()
                n_imp[t] += 1
                per_unit[u]["imputed"] += 1
                per_unit[u]["dist_km"].append(float(dist[j]))

    d["n_imputed"] = n_imp
    keep = ~(too_many.to_numpy() | drop_no_donor)
    log = {
        "rule": "nearest cell with a value, same year, same split",
        "max_distance_km": max_distance_km, "max_null_share": max_null_share,
        "rows_before": int(len(d)), "rows_with_any_missing": int(nulls.any(axis=1).sum()),
        "rows_removed_too_many_missing": int(too_many.sum()),
        "rows_removed_no_donor_in_range": int((drop_no_donor & ~too_many.to_numpy()).sum()),
        "rows_with_imputed_values": int(((n_imp > 0) & keep).sum()),
        "units": {u: {"imputed": v["imputed"], "share_of_rows": v["imputed"] / max(len(d), 1),
                      "median_donor_km": float(np.median(v["dist_km"])) if v["dist_km"] else None,
                      "max_donor_km": float(np.max(v["dist_km"])) if v["dist_km"] else None}
                  for u, v in per_unit.items() if v["imputed"] or nulls[u].any()},
    }
    return d[keep].reset_index(drop=True), log
