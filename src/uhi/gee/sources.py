"""Spec 3.4-3.8: non-Landsat sources. Raw (un-logged) values are exported; transforms happen in Python (stage G)."""

DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
DW_RENAME = ["DW_water", "DW_trees", "DW_grass", "DW_flooded_vegetation", "DW_crops", "DW_shrub",
             "DW_built", "DW_bare", "DW_snow_and_ice"]

TROPOMI = {  # output name -> (collection, band)
    "SO2": ("COPERNICUS/S5P/OFFL/L3_SO2", "SO2_column_number_density"),
    "CO": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"),
    "NO2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
    "O3": ("COPERNICUS/S5P/OFFL/L3_O3", "O3_column_number_density"),
    "AAI": ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index"),
}


def viirs_ntl(ee, aoi, start, end_excl):
    img = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
           .filterDate(start.isoformat(), end_excl.isoformat()).select("avg_rad").median())
    return img.max(0).rename("NTL")           # negative radiance -> 0; log1p applied in Python


def gpw_population(ee, year):
    """GPW only has 2000/05/10/15/20 -> nearest census year (2020 for any year >= 2019)."""
    ref = min([2000, 2005, 2010, 2015, 2020], key=lambda y: abs(y - year))
    col = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
    img = col.filterDate(f"{ref}-01-01", f"{ref}-12-31").first().select("population_density")
    return img.rename("Pop"), ref


def tropomi(ee, start, end_excl):
    bands = []
    for name, (cid, band) in TROPOMI.items():
        bands.append(ee.ImageCollection(cid).filterDate(start.isoformat(), end_excl.isoformat())
                     .select(band).mean().rename(name))
    return ee.Image.cat(bands)


def chirps_rain(ee, start, end_excl, include_30d_pre=False):
    """Total rainfall (mm) over the compositing period (a full year for full-year windows)."""
    from datetime import timedelta
    col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    out = col.filterDate(start.isoformat(), end_excl.isoformat()).sum().rename("Rain_period")
    if include_30d_pre:
        pre = col.filterDate((start - timedelta(days=30)).isoformat(), start.isoformat()).sum().rename("Rain_30d_pre")
        out = out.addBands(pre)
    return out


def dynamic_world(ee, aoi, start, end_excl):
    col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi)
           .filterDate(start.isoformat(), end_excl.isoformat()))
    probs = col.select(DW_CLASSES).mean().rename(DW_RENAME)
    label = col.select("label").mode().rename("DW_label")
    return probs, label
