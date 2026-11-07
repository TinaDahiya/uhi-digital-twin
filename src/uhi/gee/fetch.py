"""Pull a finished Earth Engine table asset straight into the project folder (no Drive / Cloud Storage)."""
import pandas as pd


def fetch_asset(ee, asset_id: str, columns: list[str], chunk: int = 1000) -> pd.DataFrame:
    fc = ee.FeatureCollection(asset_id).sort("cell_id")
    n = fc.size().getInfo()
    rows = []
    for off in range(0, n, chunk):
        feats = fc.toList(chunk, off).getInfo()
        rows += [f["properties"] for f in feats]
    df = pd.DataFrame(rows)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Asset {asset_id} lacks expected columns {missing}")
    if len(df) != n:
        raise RuntimeError(f"Fetched {len(df)} rows, asset has {n}")
    return df[columns]
