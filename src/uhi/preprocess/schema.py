TARGET = "SUHII"
LEAKY = ["LST", "LST_ref"]                       # spec section 8, rule 1
META = ["city", "year", "cell_id", "lon", "lat", "block_id", "valid_frac", "n_imputed", "split"]
DW_LABEL = "DW_label"
CONT_RAW = ["SUHII", "LST", "LST_ref", "NDVI", "NDBI", "NDWI", "BSI", "Albedo", "NTL", "Pop",
            "SO2", "CO", "NO2", "O3", "AAI", "Rain_period"]
DW_ALL = ["DW_water", "DW_trees", "DW_grass", "DW_flooded_vegetation", "DW_crops", "DW_shrub",
          "DW_built", "DW_bare", "DW_snow_and_ice"]
REQUIRED_RAW = ["cell_id", "lon", "lat", "valid_frac"] + CONT_RAW + DW_ALL   # DW_label is derived in load.py
BASE_FEATURES = ["NDVI", "NDBI", "NDWI", "BSI", "Albedo", "NTL_log", "Pop_log",
                 "SO2", "CO", "NO2", "O3", "AAI", "Rain_period"]
