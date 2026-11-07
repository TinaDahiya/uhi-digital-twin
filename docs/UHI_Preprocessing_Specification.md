# UHI Digital Twin: Preprocessing Specification

Everything that happens between the raw satellite/gridded sources and the ML step. It covers what we take from each source, what it is converted into, and the checks the output must pass.

**Status of this document**
- Landsat, VIIRS, GPW, TROPOMI and the SUHII target were already pulled for Mumbai (the v2/v3 datasets). Their ranges below come from those runs.
- CHIRPS, albedo and Dynamic World have **not been pulled yet**. Their rows are the plan, and their expected ranges are marked *not yet observed*.
- Two formula problems in the earlier datasets are corrected here (Section 11). The earlier Mumbai results should be re-run after fixing them.

---

## 1. Pipeline order

```
RAW SOURCES (GEE)
   │
   ▼
[A] Per-source cleaning        mask, scale factors, unit conversion
   │
   ▼
[B] Index / feature derivation NDVI, NDBI, NDWI, BSI, Albedo, DW probabilities
   │
   ▼
[C] Temporal compositing       one value per pixel per year (season window)
   │
   ▼
[D] SUHII target               local rural reference, per pixel
   │
   ▼
[E] Spatial aggregation        pixels → analysis cells (one row per cell per year)
   │
   ▼
[F] Row cleaning               nulls, QA, outliers
   │
   ▼
[G] Transforms                 log1p on skewed features
   │
   ▼
[H] Leakage guard + filters    drop LST-derived, Spearman, VIF
   │
   ▼
[I] Spatial block split        70:30 by geographic block
   │
   ▼
ML-READY TABLE
```

---

## 2. Conventions applied to every source

| Item | Rule |
|---|---|
| **Study years** | 2019 onward, rolling window (see Architecture Section 4) |
| **Season window** | Per city from `CITY_CONFIGS` (Mumbai Mar–May, Delhi Apr–Jun). Hyderabad and Bangalore need IMD confirmation before use |
| **Landsat / VIIRS / CHIRPS / Dynamic World window** | The full season window |
| **TROPOMI window** | A narrow 14-day window inside the season (compute-driven choice, documented as a limitation) |
| **Analysis CRS** | UTM zone derived from the city's longitude, not hardcoded. Mumbai, Delhi and Bangalore: EPSG:32643. Hyderabad (78.4°E): EPSG:32644 |
| **Float type** | Cast every band to Float32 before export |
| **Missing values** | Kept as null through steps A–E, resolved only in step F |
| **Reproducibility** | Fixed random seeds (year-based for sampling, 42 for splits) |

---

## 3. Per-source specifications

### 3.1 Landsat 8/9 Collection 2 Level-2 (LST and reflectance)

**Inputs:** `ST_B10`, `QA_PIXEL`, `SR_B2`, `SR_B3`, `SR_B4`, `SR_B5`, `SR_B6`, `SR_B7`

**Steps**
1. Merge Landsat 8 (`LC08`) and Landsat 9 (`LC09`).
2. Keep scenes with `CLOUD_COVER < 40`.
3. Mask using `QA_PIXEL` bits 0 (fill), 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (shadow).
4. Thermal: `LST_K = ST_B10 × 0.00341802 + 149.0`, then `LST_C = LST_K − 273.15`.
5. Reflectance: `SR = SR_Bx × 0.0000275 − 0.2`.
6. Derive the indices in 3.2 per scene, then take the **median** across scenes in the season window.

**Output:** `LST` (°C). It is used only to build SUHII and is never a model input.

**Observed range (Mumbai, 2019–2024 samples):** 28–57 °C, median about 44 °C. This is land-surface skin temperature, not air temperature.

### 3.2 Spectral indices from Landsat reflectance

| Feature | Formula (Landsat 8/9 bands) | Meaning |
|---|---|---|
| `NDVI` | (NIR − Red) / (NIR + Red) = (B5 − B4) / (B5 + B4) | Vegetation |
| `NDBI` | (SWIR1 − NIR) / (SWIR1 + NIR) = (B6 − B5) / (B6 + B5) | Built-up |
| `NDWI` | (Green − NIR) / (Green + NIR) = (B3 − B5) / (B3 + B5) | Surface water / moisture |
| `BSI` | ((SWIR1 + Red) − (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)) = ((B6 + B4) − (B5 + B2)) / ((B6 + B4) + (B5 + B2)) | Bare soil |

**Not used:** `NDMI` is dropped at definition. With our formulas `NDMI = (NIR − SWIR1)/(NIR + SWIR1)` is exactly `−NDBI`, so including it adds nothing and only produces a ρ = −1.00 that is pure algebra.

**Check after building:** `BSI` can correlate with `NDBI` (both use SWIR1). Let the VIF filter decide, and do not drop either by hand.

### 3.3 Albedo

**Inputs:** the same reflectance bands as 3.2.

**Steps:** broadband albedo as a weighted sum of the bands plus an intercept (Liang-type formula), clipped to [0, 1], then median-composited like everything else from Landsat.

**Open item:** the published Liang coefficients were derived for ETM+. Before publication, confirm the coefficient set used for OLI bands against a citation. The alternative is the MODIS MCD43A3 product (about 500 m), which is coarser but needs no coefficient choice.

**Output:** `Albedo` (unitless, 0–1). *Not yet observed.*

### 3.4 VIIRS-DNB monthly night-time lights

**Input:** `avg_rad`, collection `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`.

**Steps**
1. Median over the season window.
2. Set negative radiance to 0.
3. `NTL_log = log1p(avg_rad)`.

**Why log:** the raw range is about 0.3–1,200 with a median near 2, so a few bright pixels dominate otherwise.

**Output:** `NTL_log`.

### 3.5 GPWv4.11 population density

**Input:** `population_density`.

**Steps**
1. Use the nearest available census year. GPW only has 2000, 2005, 2010, 2015 and 2020, so **every study year from 2019 onward maps to the 2020 layer**.
2. `Pop_log = log1p(population_density)`.

**Consequence to state in the paper:** population is effectively static in our window. It can explain spatial variation, but it cannot explain year-to-year change.

**Output:** `Pop_log`.

### 3.6 Sentinel-5P TROPOMI (OFFL)

**Inputs:** column densities for SO₂, CO, NO₂, O₃ and the UV aerosol index (`COPERNICUS/S5P/OFFL/L3_*`).

**Steps**
1. Mean over the 14-day window.
2. Rely on the product's own QA filtering (no extra masking).
3. Resample to the analysis grid by mean.

**Output:** `SO2`, `CO`, `NO2`, `O3`, `AAI`.

**Known properties (Mumbai):** SO₂ can be slightly negative (retrieval noise, not an error), and the native resolution is much coarser than the analysis cell, so these vary smoothly across space.

### 3.7 CHIRPS daily rainfall

**Input:** daily precipitation (mm).

**Steps**
1. Sum over the season window → `Rain_season`.
2. Optional: add a 30-day antecedent sum before the window start → `Rain_30d_pre`.

**Caution:** CHIRPS is about 5 km, coarser than a typical city's heat pattern. It may vary more between years than across the city. Keep it only if it survives the VIF filter and carries non-trivial SHAP in a test run. The same coarse-resolution reasoning is why ERA5 humidity was dropped.

**Output:** `Rain_season`. *Not yet observed.*

### 3.8 Dynamic World land cover

**Input:** the nine per-class probability bands plus the `label` band.

**Steps**
1. Mean of each class probability over the season window.
2. Keep the modal `label` per pixel for maps and per-class area statistics (not as an ML feature).
3. Drop `snow_and_ice` and `flooded_vegetation` (about zero here). The remaining probabilities sum to roughly 1, which creates perfect collinearity.
4. Drop one more class (the largest, usually `crops`) as the reference category, or let VIF remove it.

**Output:** `DW_built`, `DW_trees`, `DW_grass`, `DW_crops`, `DW_shrub`, `DW_bare`, `DW_water`.

**Caution:** `DW_built` and `NDBI` both describe built-up land and will likely collide in the correlation filter. That is expected, and whichever survives is a result worth reporting.

### 3.9 Supporting layers (not model features)

| Layer | Use |
|---|---|
| JRC Global Surface Water `occurrence` | `.unmask(0).gt(50)` gives a water mask. Always `unmask(0)` before inverting |
| LGD / GAUL boundaries | AOI construction |

---

## 4. Target construction: SUHII

For each pixel at the Landsat composite resolution:

1. **Rural mask** = `NDBI ≤ 0` AND `NDVI > 0.2` AND not water (JRC).
2. **Local reference** = focal mean of `LST` over rural pixels within a **3 km** radius (`reduceNeighborhood` with `skipMasked=False`).
3. **SUHII = LST − local reference.**

**Rules**
- Compute SUHII at the pixel level first, aggregate to the cell afterwards (Section 5).
- Never compute SUHII from already-aggregated LST, because that changes the definition.
- The two settings that previously broke it, `.unmask(0)` on the water layer and `skipMasked=False`, must stay.

**Observed range (Mumbai):** about −15 to +14 °C, median about 2.7 °C.

---

## 5. Spatial aggregation to analysis cells

The earlier datasets took one 30 m pixel at each randomly drawn point. That makes Landsat features noisy point values while pollutant and population features are smooth, which hurts the model.

**Planned replacement**
1. Build a regular grid (1 km cells) over the AOI.
2. For each cell and year, compute the **mean** of every continuous band (including SUHII) over the pixels inside the cell.
3. For categorical data (Dynamic World `label`), take the **mode**.
4. Store the cell centroid as `lon`, `lat`, and the share of valid pixels as `valid_frac`.

**Rule:** drop cells with `valid_frac < 0.5`, so cells mostly covered by water or cloud gaps do not enter the dataset.

This is the largest planned accuracy gain, and it should be tested explicitly against the point-sampled dataset.

---

## 6. Row cleaning

1. Remove rows with nulls in any candidate feature or the target. No imputation. The earlier Mumbai data had zero nulls after sampling.
2. Remove placeholder values (such as −9999) if any source emits them.
3. Remove cells outside the AOI polygon.
4. Do not remove rows for being hot or cold. Hotspots are the thing we are studying.

---

## 7. Transforms

| Variable | Transform | Reason |
|---|---|---|
| NTL | `log1p` | Heavy right tail |
| Population | `log1p` | Heavy right tail |
| Rainfall | None unless the distribution is strongly skewed | Check after the first pull |
| Indices, albedo, DW probabilities | None | Already bounded |

**Outliers**
- Winsorize **features** at the 1st and 99th percentiles, with the bounds computed on the **training** split only and applied to the test split.
- Leave the **SUHII target** untouched for evaluation. If extreme target values hurt training, clip the target in training only and report every metric against the raw test target.

**Scaling:** tree models do not need it. The Spearman filter is rank-based and VIF is invariant to linear scaling, so z-scoring also changes neither. It is applied only if we want an exact replication of the source paper's workflow.

---

## 8. Leakage rules

1. `LST` and anything computed from it (the rural reference, any LST-derived layer) never enters the feature matrix.
2. The spatial-lag feature (mean SUHII of nearby cells) uses the target. It is a separate, labelled model variant, always computed using **training-block neighbours only**, and never mixed into the driver-only results.
3. Cells in the test blocks are never used to fit any transform, filter threshold, or imputation value.
4. `lon` and `lat` enter the model only in the spatial-aware variant, never in the driver-only baseline.

---

## 9. Feature filtering (after the transforms)

1. **Spearman filter:** compute on the training split. For each pair with |ρ| > 0.8, drop one. Record which pair triggered each drop.
2. **VIF filter:** drop the highest-VIF variable repeatedly until all VIF ≤ 8.
3. Save both diagnostic tables. They are evidence in the report, not just intermediate steps.

---

## 10. Final table schema

One row per cell per year.

| Column | Type | Notes |
|---|---|---|
| `city` | string | e.g. `mumbai` |
| `year` | int | |
| `cell_id` | string | grid index |
| `lon`, `lat` | float | cell centroid (EPSG:4326) |
| `block_id` | string | spatial block for CV |
| `valid_frac` | float | share of valid pixels in the cell |
| `SUHII` | float | **target** |
| `NDVI`, `NDBI`, `NDWI`, `BSI` | float | spectral drivers |
| `Albedo` | float | |
| `NTL_log`, `Pop_log` | float | |
| `SO2`, `CO`, `NO2`, `O3`, `AAI` | float | |
| `Rain_season` | float | |
| `DW_built`, `DW_trees`, `DW_grass`, `DW_crops`, `DW_shrub`, `DW_bare`, `DW_water` | float | |
| `DW_label` | int | mode class, maps only |

The final model matrix is this table minus the metadata columns and minus whatever the filters removed.

---

## 11. Corrections to the earlier Mumbai datasets

| Issue | What was wrong | Fix | Impact |
|---|---|---|---|
| **`NDBaI` was mislabelled** | The code computed (SWIR1 − SWIR2)/(SWIR1 + SWIR2), which is a moisture-type index (NBR2), not the bareness index. This likely explains its +0.85 correlation with NDVI | Replace with `BSI` (Section 3.2) | The "NDBaI reverses direction vs the paper" finding and the "NDBaI ranks #2" claim in the report need re-checking after a re-run |
| **`NDMI` was redundant by construction** | With our formulas it equals −NDBI exactly | Drop it at definition | The "ρ = −1.00 matches the paper" point is algebra, not a finding |
| **Point sampling noise** | Single-pixel samples for 30 m features vs smooth coarse features | Cell aggregation (Section 5) | Expected accuracy gain, to be tested |
| **Z-score described as required** | It is not needed for Spearman or VIF | Optional only | None on results |

---

## 12. Validation gates

A dataset version is only used for ML if it passes all of these.

| Check | Pass condition |
|---|---|
| Row count | Matches cells × years after `valid_frac` filtering |
| Nulls | 0 in the final features and target |
| LST sanity | Within about 25–60 °C for Mumbai/Delhi pre-monsoon |
| SUHII distribution | Median near 0–4 °C, no value beyond about ±25 °C |
| Population max | Matches the GPW source value (40,274 for Mumbai) |
| NTL | Non-negative, `NTL_log` max about 7 |
| Dynamic World | Class probabilities sum to about 1 before dropping classes |
| Spatial coverage | Every spatial block that appears in train or test has enough rows |
| Reproducibility | Re-running the same year with the same seed gives identical rows |
| Independent literature check | LST distribution consistent with published pre-monsoon values for the same city |

---

## 13. Open decisions

1. **Albedo source:** Landsat-derived (30 m, needs a coefficient citation) or MODIS MCD43A3 (500 m, no coefficients).
2. **Analysis cell size:** 1 km (matches the coarsest feature) or 500 m / 100 m (more rows, but coarse features are then repeated across cells).
3. **CHIRPS:** keep only if it survives filtering, or drop up front.
4. **Dynamic World reference class:** drop `crops`, or let VIF choose.
5. **Season windows for Hyderabad and Bangalore:** need IMD normals before the pipeline is used there.
6. **Whether to re-run the existing Mumbai models** with `BSI` replacing `NDBaI` before any comparison table goes into the paper.
