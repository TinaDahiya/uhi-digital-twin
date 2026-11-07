"""Pre-flight: is every source actually available through the end of the requested window?"""
from datetime import date, datetime, timedelta, timezone

COLLECTIONS = {
    "Landsat 8 L2": "LANDSAT/LC08/C02/T1_L2",
    "Landsat 9 L2": "LANDSAT/LC09/C02/T1_L2",
    "VIIRS monthly": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
    "TROPOMI SO2": "COPERNICUS/S5P/OFFL/L3_SO2",
    "TROPOMI CO": "COPERNICUS/S5P/OFFL/L3_CO",
    "TROPOMI NO2": "COPERNICUS/S5P/OFFL/L3_NO2",
    "TROPOMI O3": "COPERNICUS/S5P/OFFL/L3_O3",
    "TROPOMI AAI": "COPERNICUS/S5P/OFFL/L3_AER_AI",
    "CHIRPS daily": "UCSB-CHG/CHIRPS/DAILY",
    "Dynamic World": "GOOGLE/DYNAMICWORLD/V1",
}


def latest_dates(ee) -> dict:
    out = {}
    for name, cid in COLLECTIONS.items():
        ms = ee.ImageCollection(cid).aggregate_max("system:time_start").getInfo()
        out[name] = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() if ms else None
    return out


def covered_through(latest: dict):
    """(date, binding source, per-source dict): imagery of EVERY source exists up to (excluding) this date.
    VIIRS is monthly with a month-start timestamp, so its coverage ends at the start of the following month."""
    cov = {}
    for name, d in latest.items():
        if d is None:
            raise RuntimeError(f"{name}: no images found")
        if name.startswith("VIIRS"):
            cov[name] = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        else:
            cov[name] = d + timedelta(days=1)
    binding = min(cov, key=cov.get)
    return cov[binding], binding, cov
