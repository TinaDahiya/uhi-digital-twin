# UHI Digital Twin: Preprocessing Report (Delhi, 2024-2026) and Modelling Plan

**Project:** UHI Digital Twin (Delhi first; Mumbai, Hyderabad, Bangalore later)
**Prepared:** 8 November 2026 | **Pipeline version:** 0.3 | **State file version:** 4
**Code:** `src/uhi/` | **Config:** `configs/pipeline.yaml`, `configs/cities.yaml` | **Review notebook:** `notebooks/01_preprocessing_review.ipynb`
**Audience:** anyone who will model, explain, extend or audit this work, including people who were not in the original discussions.

**How to read this document.** Part A records exactly what was done to the data and why. Part B is the plan for the model, including what we can honestly claim as our contribution. Part C lists decisions the team still has to make. Where something is an assumption or untested, it is marked as such.

---

# PART A: PREPROCESSING

## A1. Verdict on the current outputs

The preprocessing is **internally consistent and ready for model training**. Every automated check passed except two expected warnings and one item that was skipped. No bug was found in this run. Four things need the team's attention before or during modelling (Section A10 and Part C); none of them is an error.

| Check | Result |
|---|---|
| Rows | 4,521 = 1,507 cells x 3 years (390 border-cell rows removed) |
| Missing values | None found, so imputation was never triggered on this data |
| Train/test | 70.1% of rows in train; no cell appears on both sides in any year |
| Redundancy | All 15 kept features have VIF <= 4.24 and no pair above |rho| 0.8 |
| Reproducibility | Two independent runs give an identical table |
| Warnings | `spatial_coverage` (one thin edge block) and `period_coverage` (2026 covers fewer months), both expected |
| Skipped / manual | Population-maximum gate (value not set); literature check of Delhi temperatures (a person must do it) |

---

## A2. Context: what was measured, when, from where

### A2.1 Time window, year by year (exact)

| Year | Compositing window (end exclusive) | Months | TROPOMI window | Landsat scenes used | Population layer |
|---|---|---|---|---|---|
| 2024 | 1 Jan 2024 to 31 Dec 2024 | 12 (full year) | whole window | 77 | GPW 2020 |
| 2025 | 1 Jan 2025 to 31 Dec 2025 | 12 (full year) | whole window | 77 | GPW 2020 |
| 2026 | 1 Jan 2026 to 30 Sep 2026 (ends 1 Oct) | 9 (partial year) | whole window | 61 | GPW 2020 |

- **2026 is partial by design.** It ends at the latest complete month that every source covers. The export log (`data/raw/delhi_export_log.json`) records the windows.
- **Each year is one composite.** Every satellite layer is the median (or mean, or sum, depending on the layer) over that window, not a time series. One row per cell per year.
- Exports were started on 4 November 2026. Earth Engine assets were fetched into `data/raw/delhi_<year>.csv`.
- The earlier design used only the April-June pre-monsoon season. That version remains available with `window: season` and is documented in `docs/DECISIONS.md`.

### A2.2 Data sources

| Source | Provides | Native resolution | Used as |
|---|---|---|---|
| Landsat 8 + 9 Collection 2 Level-2 (scene cloud cover < 40%) | Land-surface temperature (LST), NDVI, NDBI, NDWI, BSI, albedo | 30 m | LST builds the target; indices and albedo are drivers |
| VIIRS-DNB monthly (VCMSLCFG) | Night-time light radiance | ~500 m | Driver (`NTL_log`) |
| CIESIN GPWv4.11 | Population density | ~1 km | Driver (`Pop_log`) |
| Sentinel-5P TROPOMI (OFFL) | SO2, CO, NO2, O3, aerosol index (AAI) | ~1-7 km | Context drivers |
| CHIRPS daily | Total rainfall over the window | ~5 km | Context driver (`Rain_period`) |
| Dynamic World | Land-cover class probabilities (9 classes) | 10 m | Drivers (`DW_*`) |
| JRC Global Surface Water | Water mask (occurrence > 50%) | 30 m | Excludes water from the rural reference only |
| FAO GAUL 2015 level 1 "Delhi" | City boundary | vector | Area of interest only (to be cross-checked against the LGD boundary) |

Landsat 8/9 pass over Delhi in the mid-morning (about 10:00-10:30 local time). **SUHII here is a daytime, surface-skin-temperature contrast.** It is not air temperature, not night-time heat island intensity, and not human heat stress.

---

## A3. Pipeline overview

```
Earth Engine (per year)                          Python (this repo)
-------------------------------                  ------------------------------------------------------------
Landsat composite -> LST, indices, albedo        load the 3 yearly CSVs and stack
SUHII (pixel level, 3 km rural reference)   ->   row cleaning (valid_frac, missing target)
other sources composited over the window         geographic blocks -> train/test split (frozen)
mean over each 1 km cell -> table asset          nearest-cell imputation of missing FEATURES (same year + split)
fetch to data/raw/delhi_<year>.csv               log transforms (NTL, population)
                                                 year-centring of coarse context variables (train means)
                                                 winsorize features (train percentiles)
                                                 correlation filter -> VIF filter (train only)
                                                 -> delhi_ml_ready.parquet + preprocess_state.json
```

Standing rules for every step: **the target is never transformed, clipped or imputed; hot and cold cells are never removed; nothing the test cells contain influences any fitted quantity.**

---

## A4. The target variable: how SUHII is built

For every 30 m pixel:
1. **Rural reference mask:** a pixel is "rural" if NDBI <= 0, NDVI > 0.2, and it is not water (JRC occurrence <= 50%). All three use the same window's composite.
2. **Local reference temperature:** the mean LST of rural pixels within 3 km of the pixel.
3. **SUHII = pixel LST minus that local reference.**

SUHII is then averaged inside each 1 km cell. The order (pixel first, cell second) must not be reversed.

**Meaning.** SUHII +3 means the cell's surface is about 3 °C hotter than nearby green land. Negative means cooler than it (water, dense trees, shade, irrigation). There is **no urban mask**: every pixel gets a value, and urban pixels are simply those that tend to come out positive.

**Built-in properties to remember:**
- "Rural" means green land, not countryside. Dry fallow and bare ground fail the NDVI test and are not part of the reference, so they can score hot against it.
- NDVI and NDBI help define the reference and are also model features. Rural-like cells sit near zero by construction, and the model can partly learn the mask itself.
- A full-year SUHII (this run) is the median over all clear-sky scenes of the window. It mixes seasons and is a different quantity from the earlier April-June SUHII: medians are about 0.4-0.6 °C here against about 0.7-1.3 °C before.

---

## A5. Step by step: what was done, why, and what the numbers say

### A5.1 Aggregation to 1 km cells
Pixels were averaged into a regular 1 km grid (UTM zone 43N, EPSG:32643) so that 30 m and 1-7 km layers describe the same unit. Single-pixel sampling was rejected because it made Landsat features noisy while pollutant features stayed smooth. Each cell records `valid_frac`: the share of the cell that has a valid SUHII value **inside Delhi**.

### A5.2 Row cleaning

| Step | Rows removed | Remaining |
|---|---|---|
| Start (3 years x 1,637 grid cells) | | 4,911 |
| `valid_frac` < 0.5 (cells mostly outside Delhi or masked) | 390 (130 per year) | 4,521 |
| Missing target / LST / coordinates | 0 | 4,521 |
| Too many missing features (> 30% of groups) | 0 | 4,521 |
| No donor cell within 3 km for a missing feature | 0 | 4,521 |

- The same 130 border cells go every year, so each year has exactly 1,507 cells.
- **Why:** a cell more than half outside Delhi describes mostly non-Delhi land and rests on few pixels.
- **History:** the first exports had a bug that hid the outside-Delhi area, so this filter removed nothing. It was found from the cell counts, fixed (two separate causes), and the data re-exported.

### A5.3 Imputation of missing features (new in version 0.3)
- **What it does:** a missing feature value is copied from the nearest cell that has it, in the **same year and the same train/test split**, only if that cell is within 3 km. The nine land-cover shares are copied together from one donor so they still sum to 1.
- **What it never does:** it never imputes the target (SUHII) or LST. A row missing them is removed, because an invented answer would corrupt both training and testing. It never borrows across the train/test boundary and never extrapolates beyond 3 km; such rows are removed instead. A row missing more than 30% of its feature groups is removed, not repaired. A feature missing for a whole year raises an error.
- **Why nearest cell:** neighbouring cells are similar in space, so a copied value is a reasonable stand-in and introduces no model-fitted assumption.
- **Result in this run:** no missing feature values were found, so nothing was imputed. The logic is covered by unit tests but has not yet been exercised on real data.
- Each row carries `n_imputed` (zero everywhere in this run) so the model team can down-weight or audit imputed rows later.

### A5.4 Transforms

| Variable | Transform | Why |
|---|---|---|
| Night-time lights | `NTL_log = log1p(max(NTL, 0))` | Raw radiance is extremely right-skewed |
| Population density | `Pop_log = log1p(Pop)` | Same reason |
| Indices, albedo, land-cover shares | none | Already bounded |
| SUHII (target) | **none** | Metrics are always reported in real °C |

### A5.5 Train/test split by geographic block
- Delhi's bounding box is cut into a 10 x 10 block grid. **Whole blocks** go to train or test, so the model is always tested on places it has never seen.
- Blocks depend only on location, so a cell is on the same side in all three years. A gate fails if any cell appears in both.
- Blocks are visited in a seeded hash order (seed 42) and sent to train while the running train share is below 70%.
- The assignment is **frozen** in `preprocess_state.json`: adding a year later cannot move an old test cell into training.

| Split | Rows | Cells | Blocks | SUHII mean | SUHII std |
|---|---|---|---|---|---|
| Train | 3,171 | 1,057 | 55 | 0.69 | 1.24 |
| Test | 1,350 | 450 | 20 | 0.48 | 1.21 |

The test blocks are about 0.2 °C cooler on average. This is ordinary spatial variation, so test scores describe those particular parts of Delhi. Test blocks touch training blocks, so scores will be mildly optimistic.

### A5.6 Year-centring of coarse context variables
- **Problem:** SO2, CO, NO2, O3, AAI and `Rain_period` come from 1-7 km and 5 km grids and take essentially one value per city per year (raw mean `Rain_period`: about 975 mm in 2024, 1,043 mm in 2025, 852 mm in 2026). With three years a model could use them to recognise the year and appear to learn a rainfall effect from three data points. Ordinary spatial cross-validation would not reveal this.
- **Fix:** each of the six has its year's **training-cell mean** subtracted, keeping only the within-year spatial pattern. Test cells never contribute. The means are stored in `preprocess_state.json`.
- **Not centred:** land-surface drivers (NDBI, albedo, land cover), because their between-year differences can be physical. See A10, finding 1.

### A5.7 Land-cover (Dynamic World) handling
- The nine class probabilities sum to about 1 in every cell-year (checked: 100% within +/-0.05).
- **`crops` is the reference class and is dropped** (mean share 14.8%), because keeping all nine would make them perfectly collinear. Its effect is implicit in the others. Simulations must keep the class shares summing to 1.
- `snow_and_ice` and `flooded_vegetation` are **kept as candidates**: in this full-year data they average about 6% and 4% cover (raw yearly means 0.087 / 0.058 / 0.049 and 0.041 / 0.043 / 0.041), not negligible. `snow_and_ice` cannot be real snow in Delhi; it is almost certainly bright surfaces, dust or haze, and is treated as a non-actionable context class.
- `DW_label` (the class with the highest period-mean probability per cell) is stored **for maps only**, never as a feature.

### A5.8 Winsorization of features
- Each feature is clipped to the **1st and 99th percentile of the training cells** (about 2% of training values clipped per feature). The same limits apply to test cells. The target is never clipped.
- The values in the parquet are therefore clipped values. Any scenario value fed to the model must go through the same transform and the same limits (stored in `preprocess_state.json -> feature_spec`).

Limits for the 15 final features (snapshot; the JSON is authoritative):

| Feature | Lower | Upper |
|---|---|---|
| NDBI | -0.19991 | 0.05293 |
| Albedo | 0.12705 | 0.19080 |
| NTL_log | 1.35552 | 4.56275 |
| Pop_log | 6.78540 | 10.78744 |
| SO2 (centred) | -0.00008 | 0.00009 |
| CO (centred) | -0.00148 | 0.00196 |
| O3 (centred) | -0.00028 | 0.00034 |
| AAI (centred) | -0.16206 | 0.12850 |
| Rain_period (centred, mm) | -147.17561 | 141.78861 |
| DW_water | 0.03759 | 0.18194 |
| DW_trees | 0.03344 | 0.43869 |
| DW_grass | 0.02617 | 0.08279 |
| DW_flooded_vegetation | 0.03101 | 0.07753 |
| DW_bare | 0.04351 | 0.26822 |
| DW_snow_and_ice | 0.04034 | 0.10277 |

### A5.9 Redundancy filtering (both fitted on training cells only)

**Step A: Spearman filter (|rho| > 0.8).** When two features exceed that, one is removed. *Which one* is decided by a priority list fixed **before** looking at any result, so SUHII can never steer the choice:

| Tier | Meaning | Members |
|---|---|---|
| 1 | Land-surface properties a planner can change | NDVI, NDBI, Albedo, DW_built, DW_trees, DW_grass, DW_water |
| 2 | Indirect land / activity proxies | NDWI, BSI, DW_shrub, DW_bare, NTL_log, Pop_log |
| 3 | Coarse atmosphere / climate context | SO2, CO, NO2, O3, AAI, Rain_period |
| (unlisted) | Lowest priority | DW_flooded_vegetation, DW_snow_and_ice |

Within the same tier the feature more correlated with the other remaining features is removed. Pairs are resolved strongest first.

| Removed | Kept instead | |rho| | Reason |
|---|---|---|---|
| BSI | NDBI | 0.987 | NDBI tier 1, BSI tier 2 |
| NDWI | NDVI | 0.978 | NDVI tier 1, NDWI tier 2 |
| DW_shrub | DW_built | 0.851 | DW_built tier 1, DW_shrub tier 2 |
| DW_built | DW_grass | 0.844 | Same tier; mean |rho| 0.355 vs 0.341 (**near-tie**) |
| NDVI | DW_trees | 0.852 | Same tier; mean |rho| 0.301 vs 0.296 (**near-tie**) |
| NO2 | NTL_log | 0.807 | NTL_log tier 2, NO2 tier 3 (just above the 0.8 threshold) |

**Step B: VIF filter (threshold 8).** Before filtering, VIFs were extreme (NDVI 236, BSI 173, NDWI 148, NDBI 94). After the correlation step nothing exceeded 8, so the VIF step removed nothing. The highest VIF among kept features is 4.24.

---

## A6. The final model inputs (15 features)

Values are cell means for one year (see A5.6 and A5.8 for treatments). "Rho" is the Spearman correlation with SUHII on training cells, for information only; the default selection rule never uses it.

| Feature | What it measures | Source | Treatment | VIF | Rho with SUHII |
|---|---|---|---|---|---|
| NDBI | Built-up / dry bare surface index | Landsat | clipped | 3.72 | +0.701 |
| Albedo | Broadband surface reflectivity | Landsat | clipped | 2.59 | +0.357 |
| DW_trees | Tree-cover probability | Dynamic World | clipped | 2.51 | -0.558 |
| DW_grass | Grass probability | Dynamic World | clipped | 3.29 | -0.235 |
| DW_water | Water probability | Dynamic World | clipped | 2.28 | +0.071 |
| DW_flooded_vegetation | Flooded-vegetation probability | Dynamic World | clipped | 3.78 | -0.412 |
| DW_bare | Bare-ground probability | Dynamic World | clipped | 3.22 | +0.488 |
| DW_snow_and_ice | "Snow/ice" probability (bright-surface artefact in Delhi) | Dynamic World | clipped | 1.23 | -0.019 |
| NTL_log | Night-time light (human activity) | VIIRS | log1p, clipped | 4.24 | +0.112 |
| Pop_log | Population density (2020, same all years) | GPW | log1p, clipped | 1.81 | +0.062 |
| CO | CO column | TROPOMI | year-centred, clipped | 3.75 | -0.048 |
| SO2 | SO2 column | TROPOMI | year-centred, clipped | 1.24 | -0.039 |
| O3 | O3 column | TROPOMI | year-centred, clipped | 1.95 | +0.014 |
| AAI | UV aerosol index | TROPOMI | year-centred, clipped | 2.92 | +0.123 |
| Rain_period | Total rainfall over the window | CHIRPS | year-centred, clipped | 2.22 | -0.076 |

**Reading the rho column.** These are associations in Delhi's geography on these composites, not cause and effect. The strongest are NDBI (+0.70), tree cover (-0.56), bare ground (+0.49) and flooded vegetation (-0.41). The signs and sizes differ from the earlier April-June run (for example nightlights were -0.36 then and are +0.11 now), as expected for a different quantity.

### Columns in the table that are not features
`city`, `year`, `cell_id`, `lon`, `lat`, `block_id`, `valid_frac`, `n_imputed`, `split` (bookkeeping), `SUHII` (target), `DW_label` (maps only).

### Features removed, and why

| Column | Why |
|---|---|
| `LST`, `LST_ref` | They build the target; including them would leak the answer |
| `lon`, `lat`, `year` | Reserved for a separately labelled spatial-aware variant |
| Raw `NTL`, `Pop` | Replaced by log versions |
| `DW_crops` | Reference class |
| `BSI`, `NDWI`, `NDVI`, `DW_shrub`, `DW_built`, `NO2` | Removed by the correlation filter; full reasons in `preprocess_state.json -> decisions` |

---

## A7. Validation results

| Gate | Status | Meaning |
|---|---|---|
| row_count | PASS | 4,911 in, 390 removed, 4,521 out |
| nulls | PASS | No missing values in target or features |
| lst_sanity | PASS | Cell LST 27.8-46.3 °C (allowed 10-60); yearly medians 36.0 / 32.5 / 39.8 °C |
| suhii_median | PASS | Yearly medians 0.58 / 0.39 / 0.62 °C (band 0-4 comes from Mumbai) |
| suhii_extremes | PASS | Largest |SUHII| is 6.21 °C (limit 25) |
| population_max | SKIP | Set `expected_pop_max` from the GPW source to enable |
| ntl | PASS | NTL_log 1.17-5.19, non-negative |
| dynamic_world_sum | PASS | 100% of cell-years sum to 1 +/- 0.05 |
| dw_dropped_class_share | PASS | Nothing dropped up front except the reference class |
| years_present | PASS | 2024, 2025, 2026 present |
| cells_per_year | PASS | 1,507 each |
| split_cell_integrity | PASS | 1,057 train cells, 450 test cells, no overlap |
| spatial_coverage | WARN | One block has fewer than 5 rows (thin edge block); harmless |
| imputation | PASS | Nothing needed imputing |
| period_coverage | WARN | 12 / 12 / 9 months: years cover different calendar months (see A10) |
| reproducibility | PASS | Two independent runs give an identical table |
| literature_check | MANUAL | A person must compare Delhi LST with published values |

---

## A8. Outliers in SUHII: what we do and why

**The SUHII target is not clipped, trimmed or removed.** Box-plot points beyond the whiskers are 2.9%, 2.4% and 3.6% of cells in 2024, 2025 and 2026 (range about -3.9 to +6.2 °C).

| Year | Median | p05 | p95 | Max |
|---|---|---|---|---|
| 2024 | 0.58 | -1.30 | 2.97 | 6.07 |
| 2025 | 0.39 | -1.12 | 2.20 | 5.46 |
| 2026 | 0.62 | -1.17 | 3.11 | 6.21 |

Reasons: they are the signal the twin exists to find; they are physically plausible (the largest value, 6.21 °C, is far inside the 25 °C sanity limit); cloud, shadow and border artefacts are already filtered upstream; and feature outliers are handled separately by winsorization.

**Evidence they are real (notebook section 8b):**
- **Persistence.** 16 cells are among the hottest 2% in all three years, and 14 among the coldest 2%. Chance alone would give about 0.01. Artefacts do not repeat; real places do.
- **Land cover.** The hottest 2% of cell-years are dominated by crops (49%, likely dry fallow), built-up (31%) and bare ground (20%). The coldest are trees (43%) and water (26%).
- **Location.** One cell (`703_3160`, near 28.55 N, 77.08 E) is the hottest or among the hottest in every year, with neighbouring cells. That area lies near Indira Gandhi International Airport (about 28.56 N, 77.10 E), so runways and aprons are a plausible cause. This is an inference from coordinates; confirm on a map.
- Several of the coldest cells (for example near 28.59 N, 77.27 E) appear to follow the Yamuna, which fits.

If extremes hurt training, the permitted remedy is to clip or down-weight the target **in training only** (or use a robust loss such as Huber) and still report every metric against the raw test target.

---

## A9. Decisions and the reasoning behind them

| # | Decision | Alternatives considered | Why this one |
|---|---|---|---|
| 1 | Three years: 2024 and 2025 full, 2026 to latest complete month | April-June only; more years | More of the year and more data, with all sources consistent. 2026 is partial, so see A10 |
| 2 | 1 km analysis cells | 30 m points; 100-500 m cells | Matches the coarsest layer; avoids noisy single pixels. Finer cells are a candidate for later |
| 3 | SUHII = daytime skin temperature minus a local 3 km rural reference | City-wide rural mean | A local reference removes regional gradients; it is the project's chosen definition |
| 4 | Split before any fitted step | Filters on all data | Test cells must never influence a fitted quantity |
| 5 | Geographic block split with a frozen map | Random cell split; per-year split | A random split leaks through neighbours; a per-year split would put the same place in both sets |
| 6 | Priority tiers for correlated pairs | Unsupervised mean |rho|; target correlation | Outcome-blind and keeps variables a planner can change. The unsupervised rule flipped on tiny differences |
| 7 | Year-centre six coarse context variables | Leave raw; drop them | They would act as year labels |
| 8 | Keep `snow_and_ice`, `flooded_vegetation` as candidates | Drop as "negligible" | They average about 6% and 4% cover |
| 9 | `crops` as reference class | Let VIF choose | An explicit baseline for the compositional shares |
| 10 | Winsorize features at train 1/99 percentile | No clipping; z-score | Limits extreme feature values; no scaling needed for tree models, Spearman or VIF |
| 11 | Impute missing features from the nearest cell (same year and split, within 3 km); never impute the target | No imputation; model-based imputation | Requested; restricted so it cannot leak across the split or invent labels |
| 12 | Never remove hot/cold cells; never transform SUHII | Trim outliers | Hotspots are the object of study |
| 13 | Export via Earth Engine assets, one file per year | Google Drive; many files | Keeps storage in the project; fewer files; reproducible fetch |
| 14 | Persist only the ML table and the state file | Save every diagnostic | Diagnostics are recomputed in the notebook |

---

## A10. Findings from this run that matter for modelling

1. **Year-level shifts in some land features could act as year labels.** Raw yearly means: albedo 0.152 / 0.147 / 0.164; `DW_snow_and_ice` 0.087 / 0.058 / 0.049; NDBI -0.057 / -0.077 / -0.057. Albedo in 2026 sits roughly one within-year spread above the other years, and the yearly albedo ordering matches the yearly SUHII ordering (2026 > 2024 > 2025). With only three years, a model could use these levels to identify the year. Part of this may be physical (a wetter 2025) and part composite effects from the partial 2026 window. Land features are deliberately not year-centred, so **Stage 5 must run the within-year ablation** (Part B, Section B5).
2. **The 2026 composite is not like-for-like with 2024/2025.** It covers 9 months, so its medians are seasonally biased: median LST is 39.8 °C against 36.0 and 32.5 °C. SUHII, being a difference, is far less affected (0.62 vs 0.58 and 0.39). `align_months: true` would trim every year to Jan-Sep (needs re-export of 2024 and 2025).
3. **Two key levers were removed by near-ties.** NDVI lost to `DW_trees` on 0.301 vs 0.296 mean |rho|, and `DW_built` lost to `DW_grass` on 0.355 vs 0.341. These are exactly the near-tie sensitivities the priority tiers were meant to avoid, and they now decide the fate of the two most intuitive intervention variables. `DW_built` is not a model input, yet in Dynamic World a road is mostly classed as built-up. A road scenario must therefore be expressed through NDBI, `DW_bare`, albedo and the vegetation shares.
4. **`DW_snow_and_ice` adds risk and little signal.** It is a likely artefact, correlates -0.019 with SUHII, and its yearly mean falls by 44% between 2024 and 2026.
5. **Threshold sensitivity.** NO2 was removed at |rho| 0.807 against a 0.8 threshold.
6. **Plausibility checks still to do by hand.** Yearly `Rain_period` totals (975 / 1,043 / 852 mm) look high for Delhi against my recollection of its long-term annual rainfall (roughly 600-800 mm); verify against IMD. They enter the model only as year-centred values. Also confirm that two years having exactly 77 Landsat scenes is a coincidence (check `delhi_export_log.json`).
7. **Imputation has not been exercised on real data**, because no feature values were missing.

---

## A11. Files, schema and how to use them

| File | Contents |
|---|---|
| `data/raw/delhi_<year>.csv` | One file per year: cell means of every band (input to preprocessing) |
| `data/raw/delhi_export_log.json` | Windows, months, scene counts, asset ids per year |
| `data/processed/delhi/delhi_ml_ready.parquet` | The model table (4,521 rows) |
| `data/processed/delhi/preprocess_state.json` | Final features, `feature_spec` (how to rebuild each input), winsor limits, year means, frozen split map, imputation log, decision log, gate results |

Parquet columns: `city, year, cell_id, lon, lat, block_id, valid_frac, n_imputed, split, SUHII, <15 features>, DW_label`.

```python
import pandas as pd, json
df = pd.read_parquet("data/processed/delhi/delhi_ml_ready.parquet")
state = json.load(open("data/processed/delhi/preprocess_state.json"))
X_cols = state["final_features"]
train, test = df[df.split == "train"], df[df.split == "test"]
# cross-validate with GroupKFold(groups=train["block_id"]); never plain K-fold
```

To reproduce: `scripts/01_export_gee.py`, `02_fetch_gee.py`, `03_preprocess.py`, then review the notebook. All tunable choices live in `configs/pipeline.yaml`. Delete `preprocess_state.json` only when you intend to re-split.

---

# PART B: THE MODEL (NEXT STEP) AND OUR CONTRIBUTION

## B1. What already exists (so we claim only what is ours)

A quick literature search (not a systematic review; **verify every item before citing**) shows that the basic idea of "machine learning on satellite data plus what-if simulation of surface temperature" is already published. We should not present it as new.

| Existing work | What it does | Implication for us |
|---|---|---|
| Oxoli et al., 2026 (Politecnico di Milano) | Random-forest model of land surface temperature, driven by surface-material fractions and urban canopy parameters, used for what-if simulation, aimed at urban digital twins | A tree model plus what-if on LST is established |
| Fleckenstein et al., 2025 (IBM Research) | Fine-tunes a geospatial foundation model; simulates greening by replacing built-up pixels; checks physical realism against observed park cooling | Physical-realism testing of simulations has precedent, with a different method |
| Jyothi and Mesapam, 2025 (NIT Warangal) | Random-forest UHI digital twin with a 3D platform for an Indian city | An Indian digital-twin prototype exists |
| Li (2023) GeoShapley, and a green-space cooling study in Chinese cities | Spatially varying explanations of drivers; applied to land surface temperature | GeoShapley on LST is not new |
| Kattenborn et al., 2022, and related work | Random cross-validation inflates performance under spatial autocorrelation | Spatial block validation is established practice in remote sensing, so it is good practice, not novelty |

So "better than existing solutions" cannot rest on the model family, on SHAP or GeoShapley, or on having a what-if simulator. It has to come from the properties below, and each must be shown by an experiment.

## B2. What we can credibly contribute (hypotheses to prove, not claims)

| # | Contribution | The gap it addresses | How we will prove it |
|---|---|---|---|
| C1 | **A leak-safe, auditable, outcome-blind data pipeline** for any Indian city from free global data: frozen block split across years, train-only fits, year-label guard, a-priori actionable feature tiers, an explained decision for every feature | Many ML-UHI studies report one random split and a model; few publish an audit trail of why each input was kept (to be checked in a proper review) | Release the pipeline; show that a random split overstates accuracy compared with our block split on the same data |
| C2 | **Intervention validity as a measured property**, not only R-squared: sign tests, small-change sensitivity, extrapolation guard, and **back-tests on real land-cover changes** | Prediction accuracy does not tell us whether a model responds correctly to a hypothetical change | Pre-defined sanity tests plus before/after satellite comparisons for real developments (needs earlier years) |
| C3 | **Detectability honesty:** each scenario result comes with an interval and a verdict on whether the effect is large enough to separate from model error | Small interventions (a highway is a few percent of a 1 km cell) can look precise but be noise | Calibrated prediction intervals; a "minimum detectable effect" map |
| C4 | **Actionable, physically consistent scenarios:** footprint-to-cell mapping, land-cover shares kept summing to 1, correlated features moved together, monotonic response constraints | Unconstrained tree models give step-like or backwards responses (for example more albedo predicting more heat) | Compare constrained vs unconstrained models on the sanity tests |
| C5 | **Equity-weighted prioritisation:** cooling gain multiplied by exposed population, to rank where interventions help most people | Hotspot maps do not say whom to help first | Ward-level ranking compared with a hotspot-only ranking. Needs a yearly population layer, since GPW 2020 is static |
| C6 | **Cross-city transfer:** train on Delhi, test on Mumbai/Hyderabad/Bangalore with no retraining | Most studies are single-city | Leave-one-city-out score against a local model. High reward, high risk |

C1, C2 and C3 are the most defensible and are within reach on Delhi alone. C6 is the strongest generalisation claim but needs more cities.

## B3. Contribution to society

- **Targeting.** Delhi's heat planning needs to know where to plant trees or retrofit roofs first. The twin should rank locations by cooling gain and by the number of people exposed, instead of only mapping hot spots.
- **Honest planning tool.** Telling planners when an intervention is too small to detect protects public money from over-promised benefits.
- **Low cost and replicable.** It uses only free global data, so cities without local sensor networks or modelling teams can reproduce it. The audit trail lets officials and reviewers see why each variable was used.
- **Limits to state openly.** The target is daytime surface temperature. It is not air temperature, night-time heat (which matters more for health) or heat stress. A claim of health benefit would need a further link to such measures.

## B4. Model implementation plan (Stage 5)

**Inputs:** `delhi_ml_ready.parquet`, `preprocess_state.json`. **Target:** `SUHII`. **Features:** `final_features`. Metadata columns are never features.

**Validation rules (fixed before any run):**
- Tune and compare with `GroupKFold` on `block_id` inside the train split. Do not touch `split == "test"` until the final report.
- Report R-squared, RMSE and MAE, overall and per year, per dominant land cover, and for the hottest 10% of cells (hotspots are what we care about).
- Report scores against the raw, unclipped target.

**Models:**
1. Baselines: mean predictor, ridge regression, random forest, gradient boosting, XGBoost, LightGBM.
2. Simulation-grade candidates: monotone-constrained gradient boosting; an explainable additive model (EBM or GAM) whose responses are smooth; an ensemble with quantile or conformal intervals.
3. Labelled variants: **A** driver-only (primary, used for scenarios); **B** adds neighbourhood context (mean of surrounding cells, built from training cells only); **C** spatial-aware (coordinates, spatial lag from training neighbours only), for map quality only, never mixed with A.

**Diagnostics:** residual maps and Moran's I (are errors spatially clustered?); learning curve against number of blocks; permutation importance by block; SHAP stability across folds; partial-dependence and ICE plots checked for sign and plausibility (albedo especially).

**Explainability:** SHAP globally and per cell; GeoShapley to separate feature effects from location effects, noting it is established (B1).

## B5. Required experiments before trusting the model

| Experiment | Question | Pass condition (set before running) |
|---|---|---|
| Within-year ablation (year-centred target and year-centred land features) | Is the model learning years instead of places? (A10, finding 1) | Scores and top drivers broadly unchanged |
| Random split vs block split | How much does leakage inflate scores? | Documents the gap (this is evidence for C1) |
| Drop `DW_snow_and_ice` | Does it carry any signal beyond noise? | No loss in block-CV score |
| Tier variants (A10, finding 3) | Does protecting NDVI and `DW_built` change accuracy or drivers? | Reported either way |
| Threshold sensitivity (|rho| 0.75 / 0.8 / 0.85, VIF 5 / 8 / 10) | Do conclusions depend on arbitrary cut-offs? | Top drivers stable |
| Sign and sensitivity tests | Does more built-up never predict cooler, more trees never hotter? Does a 1-5% land-cover change move predictions or fall in a flat step? | Responses have the right sign and are not purely step-like |

## B6. Simulator plan (Stage 7)

1. **Scenario input:** a polygon or line (for example a road with a width), plus a land-cover change type.
2. **Footprint to cells:** intersect with the 1 km grid and compute the fraction of each cell changed.
3. **Feature change:** area-weighted mixing between the current cell value and the surface-type value of the new surface (values from literature or from 30 m pixels over existing roads); update land-cover shares so they still sum to 1.
4. **Apply the pipeline's own transforms:** `feature_spec` (log1p, year centring) and the winsorization limits.
5. **Predict before and after** with the ensemble; report the change with an interval and the detectability verdict (C3).
6. **Guards:** refuse scenarios whose features fall outside the training support; propagate changes to correlated features together.
7. **Known gaps to display:** no spillover to neighbouring cells, no traffic heat or tailpipe emissions, a static 2020 population layer.

## B7. Validation of the simulator and go/no-go

1. **Predictive:** held-out block accuracy against baselines (B4).
2. **Sanity:** the sign and sensitivity tests (B5).
3. **Back-test (the key evidence):** find real developments or greening between earlier years and 2026 (needs data back to 2019; identify candidates from OpenStreetMap or land-cover change), predict their effect, and compare it with the observed satellite change. Report where the model fails.
4. **Transfer:** leave-one-city-out once a second city exists.

**Go/no-go.** Agree the thresholds before seeing results. If a road-sized change is smaller than model error, say so, and use the simulator for larger-area scenarios or move to finer cells (100-250 m) or neighbourhood features.

## B8. Suggested order of work

1. Settle the open decisions in Part C.
2. Train baselines and the simulation-grade candidates with block-grouped cross-validation.
3. Run the experiments in B5.
4. Build the simulator, then back-test it. For this, extend the data to earlier years.
5. Add the equity layer once a yearly population layer is chosen.
6. Repeat on a second city to test transfer.

---

# PART C: OPEN DECISIONS FOR THE TEAM

| # | Decision | Options | My recommendation | Note |
|---|---|---|---|---|
| 1 | Year-comparability of land features | Keep as is; centre land features by year; `align_months: true` (re-export 2024 and 2025 as Jan-Sep) | Keep the current data and run the within-year ablation first; re-export aligned only if the ablation shows a problem | Re-export is the expensive option |
| 2 | Priority of NDVI and `DW_built` | Keep tiers; put NDVI and `DW_built` in their own first tier | Decide by the simulation need (a road is "built") and document it | Any change after seeing results is results-informed and must be recorded as such |
| 3 | `DW_snow_and_ice` | Keep; drop | Test in the ablation, lean towards dropping | Likely artefact, no relation to SUHII |
| 4 | Correlation threshold | 0.75 / 0.8 / 0.85 | Keep 0.8 and report sensitivity | NO2 sits at 0.807 |
| 5 | Population layer | Keep GPW 2020; use a yearly product | Keep for now; revisit for the equity layer (C5) | Population is the same in all years |
| 6 | Population gate and literature check | Set `expected_pop_max`; compare Delhi LST with published values | Do both before publication | Open since the first run |
| 7 | More years | Stay at three; extend back to 2019 | Extend, because back-tests (C2) need earlier data | Sentinel-5P limits 2019 as the floor |

---

## Glossary

- **SUHII:** surface urban heat island intensity; surface temperature relative to a nearby green reference.
- **LST:** land-surface temperature (skin temperature seen from space).
- **NDVI / NDBI / NDWI / BSI:** vegetation, built-up, water and bare-soil spectral indices.
- **VIF:** variance inflation factor; how redundant a feature is given the others (above 8 is treated as too redundant).
- **Spearman rho:** rank correlation between two variables.
- **Winsorize:** clip extreme values to a chosen percentile instead of deleting them.
- **Reference class:** the land-cover class left out so the class shares are not perfectly collinear.
- **Block:** a geographic tile; whole blocks are assigned to train or test.
- **Spatial cross-validation:** testing on whole areas held out from training, so neighbouring similar cells cannot leak the answer.
- **Back-test:** checking a simulator by predicting a change that really happened and comparing it with what was observed.
- **Detectability:** whether an effect is large enough to be told apart from model error.

## Sources consulted for Part B (quick search; verify before citing)
- Oxoli, Vavassori, Brovelli (2026), prototype Earth-observation framework for simulating urban intervention effects on LST, Politecnico di Milano.
- Fleckenstein, Kreismann, Rigotti et al. (2025), fine-tuned geospatial foundation model for UHI simulation, arXiv 2510.18773 (IBM Research).
- Jyothi and Mesapam (2025), digital twin for climate-resilient planning and UHI mitigation, ISPRS Archives XLVIII-M-10-2025.
- Li (2023), GeoShapley, arXiv 2312.03675; Zhang et al., green-space cooling with GeoShapley across five Chinese cities.
- Kattenborn et al. (2022), spatially autocorrelated samples inflate performance assessment, ISPRS Open Journal of Photogrammetry and Remote Sensing.
