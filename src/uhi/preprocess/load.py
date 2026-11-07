"""Load one CSV per year (data/raw/<city>_<year>.csv) and stack them."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DW_ALL, DW_LABEL, REQUIRED_RAW


def derive_dw_label(df: pd.DataFrame) -> pd.DataFrame:
    """Dominant class per cell = argmax of the period-mean class probabilities (maps/area stats only, never a feature)."""
    d = df.copy()
    P = d[DW_ALL].to_numpy(float)
    d[DW_LABEL] = np.where(np.isnan(P).any(axis=1), np.nan, np.argmax(np.nan_to_num(P, nan=-1.0), axis=1))
    return d


def available_years(raw_dir, city: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(city)}_(\d{{4}})\.csv$")
    return sorted(int(m.group(1)) for p in Path(raw_dir).glob(f"{city}_*.csv") if (m := pat.match(p.name)))


def load_raw(raw_dir, city: str, years=None) -> pd.DataFrame:
    have = available_years(raw_dir, city)
    years = sorted(years) if years else have
    missing = [y for y in years if y not in have]
    if missing or not years:
        raise FileNotFoundError(f"Missing raw files for {city} years {missing or 'any'} in {raw_dir}; found {have}")
    frames = []
    for y in years:
        d = pd.read_csv(Path(raw_dir) / f"{city}_{y}.csv").drop(columns=["system:index", ".geo"], errors="ignore")
        absent = [c for c in REQUIRED_RAW if c not in d.columns]
        if absent:
            raise ValueError(f"{city}_{y}.csv is missing columns {absent}")
        d.insert(0, "year", y)
        d.insert(0, "city", city)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return derive_dw_label(df)
