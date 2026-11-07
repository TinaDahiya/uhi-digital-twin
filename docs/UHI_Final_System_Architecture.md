# UHI Digital Twin — Final System Architecture (Mumbai Reference Implementation)

This document specifies the complete, final pipeline architecture: every data source, why it's included, what goes in and comes out of each stage, and how the whole system fits together. This is the reference architecture for Mumbai and the generalizable template for Delhi (primary target), Hyderabad, Bangalore, and any future Indian city added to the project.

**Note on scope:** this architecture has moved beyond reproducing the source paper's methodology. Data source selection, study period, and system design decisions below are made for this project's own goals (an operational, multi-city, decision-support digital twin), not to match the paper's original choices.

---

## 1. High-Level Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 0: CITY CONFIGURATION                                        │
│  Input:  city name                                                  │
│  Output: AOI geometry, summer months, region center                 │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 1: DATA ACQUISITION (Google Earth Engine)                    │
│  Input:  AOI geometry, year, date range                             │
│  Pulls:  Landsat 8/9 (LST + spectral indices + albedo), VIIRS       │
│          (nightlights), CIESIN (population), Sentinel-5P (5         │
│          pollutants), CHIRPS (rainfall), Dynamic World (LULC class) │
│  Output: composite image (13+ bands) → sampled point table (CSV)    │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 2: SUHII COMPUTATION                                         │
│  Input:  LST, NDVI, NDBI bands                                      │
│  Output: SUHII band (locally-referenced heat island intensity)      │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 3: FEATURE ENGINEERING                                       │
│  Input:  tabular dataset (all years combined)                       │
│  Steps:  Z-score norm → Spearman filter (|ρ|>0.8) → VIF filter (≤8) │
│  Output: final feature list + cleaned dataset                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 4: SPATIAL BLOCK SPLIT                                       │
│  Input:  lon/lat per row                                            │
│  Output: train/test masks (70:30, geographically independent)       │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 5: MODEL TRAINING                                            │
│  Input:  X (features), y (SUHII), spatial groups                    │
│  Models: RandomForest, GradientBoosting, XGBoost, LightGBM          │
│  + Novel: spatial-lag augmented variant                             │
│  Output: trained models, R²/RMSE/MAE metrics per model              │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 6: EXPLAINABILITY (XAI)                                      │
│  Input:  best trained model, test set                               │
│  Methods: SHAP (global + spatial), PDP, dependence plots,           │
│           + Novel: GeoShapley (feature/geo/interaction decomposition)│
│  Output: driver rankings, spatial SHAP maps, geo-interaction values │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────┐
│  STAGE 7: DIGITAL TWIN OUTPUT LAYER  (not yet built)                │
│  Input:  spatial predictions, SHAP maps, LULC classification        │
│  Output: interactive map/dashboard for periodic monitoring,         │
│          hotspot identification, driver attribution, and a          │
│          what-if scenario simulator for mitigation planning         │
│          (NOT real-time - see Section 6 for actual update cadence)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete Data Source Table (FINALIZED)

Every input to the system, what it provides, and why it's included. This list reflects the project's own final decisions, made independently of the source paper.

| # | Data Source | Provider/Platform | Variable(s) Extracted | Native Resolution | Temporal Availability | Purpose in Pipeline |
|---|---|---|---|---|---|---|
| 1 | **Landsat 8/9 Collection 2 Level-2** | USGS/NASA, via GEE | LST (thermal), NDVI, NDBI, NDWI, NDBaI, NDMI (spectral indices) | 30m | 2013–present | Core surface temperature + land-cover-character indices; LST feeds SUHII computation directly |
| 2 | **VIIRS-DNB Monthly (VCMSLCFG)** | NOAA, via GEE | Night-time light radiance (NTL) | ~500m | 2012–present | Proxy for anthropogenic/urbanization intensity; does not saturate over bright cores (unlike older DMSP-OLS sensors) |
| 3 | **CIESIN GPWv4.11** | Columbia Univ. CIESIN, via GEE | Population density | ~1km | 5-year steps (2000/05/10/15/20) | Socio-economic driver of heat retention/anthropogenic heat; also enables population-weighted exposure/equity analysis |
| 4 | **Sentinel-5P TROPOMI (OFFL)** | ESA Copernicus, via GEE | SO₂, CO, NO₂, O₃, Aerosol Index | ~1–7km (varies by pollutant) | June 2018–present | Atmospheric pollutant drivers of SUHII |
| 5 | **CHIRPS Daily** | UCSB Climate Hazards Center, via GEE | Rainfall | ~5km | 1981–present | Climatological driver; moisture availability affects surface cooling capacity |
| 6 | **Albedo** | Derived from Landsat SR bands (broadband albedo formula) or MODIS MCD43A3 | Surface reflectivity | 30m (Landsat-derived) or ~500m (MODIS) | Matches host sensor | Captures solar radiation absorption independent of what NDBI/NDBaI measure — low-albedo surfaces (dark asphalt, certain roofing) absorb more shortwave radiation, directly elevating LST through a distinct physical mechanism |
| 7 | **Dynamic World** | Google/WRI, via GEE | Pre-classified 9-class land cover (probabilistic, near-real-time) | 10m | 2015–present | Categorical land-use/land-cover classification — closes the LULC gap without requiring manual supervised classification (an upgrade over the source paper's manual MODIS-based approach) |
| — | *(Supporting layer, not a driver feature)* JRC Global Surface Water | EC Joint Research Centre, via GEE | Water occurrence mask | 30m | Historical composite | Used internally to exclude water pixels from the SUHII "rural reference" mask — not a modeling input itself |
| — | *(Supporting layer)* LGD Sub-district / FAO GAUL boundaries | India LGD / FAO, via GEE | Administrative boundaries | Vector | Current / 2015 snapshot | AOI construction only, not a modeling feature |

**Explicitly excluded going forward:** ERA5-Land air temperature/humidity. Decision: at coarse resolution (~9–11km for ERA5-Land, or ~31km for plain ERA5), this variable would be nearly constant across a single city's extent, actively undermining the project's core spatial-heterogeneity analysis (spatial SHAP, GeoShapley). Not worth the integration cost given the resolution mismatch with the project's actual scientific focus.

---

## 3. Multi-City Applicability — Confirmed

**Yes — every finalized data source in Section 2 has national (India-wide) and, in fact, global coverage, with no city-specific availability restrction.** This was a deliberate selection criterion, not a coincidence:

| Source | Coverage |
|---|---|
| Landsat 8/9 | Global |
| VIIRS-DNB | Global |
| CIESIN GPWv4.11 | Global |
| Sentinel-5P TROPOMI | Global |
| CHIRPS | Quasi-global (50°S–50°N) — fully covers all of India |
| Albedo (Landsat-derived or MODIS) | Global |
| Dynamic World | Global |

This means the exact same data-acquisition functions apply unchanged to Delhi, Hyderabad, Bangalore, or any other Indian city added later — the only per-city work is the AOI boundary construction (Section 5) and confirming the correct local peak-heat season. No data source needs to be swapped or substituted city-to-city.

---

## 4. Study Period — Project's Own Recommendation

**This is no longer being chosen to match the source paper — it's chosen for what best serves an operational, ongoing monitoring system.**

**Binding constraint, unchanged:** Sentinel-5P TROPOMI only exists from **June 2018** onward. This sets the earliest possible start point for a dataset using all finalized sources consistently — **2019** is the first full calendar year with every source available.

**Recommended range: 2019 – most recent complete year, as a rolling window — not a fixed historical span.**

Reasoning, thought through for this project specifically rather than inherited from the paper:

- **Don't hardcode a fixed end year.** The source paper picked fixed historical snapshots (2002/2014/2024) because it was a one-time retrospective study. This project is meant to be an ongoing monitoring system — it should be designed to **append the newest available year automatically** as time passes, rather than being frozen at "2024" the way a static paper's dataset would be. Practically: re-run Stage 1 once per year (once that year's summer season has fully passed and clear-sky Landsat passes are available), and add it to the training data.
- **Keep 2019 as the fixed starting point**, not because of the paper, but because it's the earliest point where every finalized source is simultaneously available — going earlier would mean dropping pollutants or Dynamic World for those years, creating an inconsistent dataset across your timeline.
- **A 6–7 year window (2019–2025, growing to 2019–2026, etc.) is enough to support genuine trend analysis** (year-over-year SUHII change, the historical explorer mode) while staying within a period where urban development patterns are still reasonably comparable — going back to, say, 2005 would mean spanning a very different city (pre-metro-expansion Mumbai, Delhi, etc.), making trend comparisons less meaningful for current planning decisions anyway.

**Bottom line recommendation to write into your methodology:** *"Study period: 2019 onward, treated as a rolling window that expands with each newly completed year, rather than a fixed historical range — reflecting the project's design as an operational monitoring system rather than a one-time retrospective study."*

---

## 5. Stage-by-Stage Input/Output Specification

### Stage 0 — City Configuration
- **Input:** a city key string (`"mumbai"`, `"delhi"`, etc.)
- **Process:** looks up `CITY_CONFIGS` — state/district list (or `None` for whole-UT cities), climatologically-appropriate summer months (verified per-city against IMD normals), map center coordinates
- **Output:** parameters consumed by every downstream stage; per-city AOI precision decisions (e.g., Mumbai's 15-taluka boundary) are handled as documented, separately-validated boundary files where a generic administrative union would be inaccurate

### Stage 1 — Data Acquisition
- **Input:** AOI geometry (`ee.Geometry`), target year, summer-month range
- **Process:**
  1. Landsat 8+9 collection filtered by cloud cover (<40%), masked (5-bit QA_PIXEL cloud/shadow/cirrus mask), median-composited over the full summer window → LST + 5 spectral indices + albedo
  2. VIIRS monthly composite, median over same window → NTL
  3. GPWv4 nearest-available-year population raster
  4. TROPOMI: 5 separate pollutant pulls, narrow time-window aggregation (a compute-cost-driven design choice, not a scientific one — see Section 6 for the technical reasoning)
  5. CHIRPS daily rainfall, summed over the summer window
  6. Dynamic World median/mode composite → categorical land-cover class per pixel
  7. All bands merged into one composite image, sampled at N points per year, exported as CSV via background export tasks
- **Output:** one CSV per year, each row = one sampled point with all raw band values, LULC class, and lon/lat

### Stage 2 — SUHII Computation
- **Input:** LST, NDVI, NDBI bands from the Stage 1 composite
- **Process:** for each pixel, build a "rural mask" (NDBI≤0 AND NDVI>0.2 AND not-water via JRC layer), compute a focal mean of LST over rural pixels within a 3km radius, subtract from the pixel's own LST
- **Output:** SUHII band — the ML target variable throughout the rest of the pipeline

### Stage 3 — Feature Engineering
- **Input:** combined multi-year tabular dataset
- **Process:** Spearman correlation filter (|ρ|>0.8) → sequential VIF elimination (≤8) → explicit exclusion of raw LST (target leakage)
- **Output:** final feature list + diagnostic correlation/VIF tables

### Stage 4 — Spatial Block Split
- **Input:** lon/lat columns
- **Process:** 10×10 geographic grid, block-level 70:30 train/test allocation
- **Output:** train/test mask + spatial group labels

### Stage 5 — Model Training
- **Input:** final features, SUHII target, spatial groups
- **Process:** 4 ensemble models (RF, GB, XGBoost, LightGBM), 5-fold spatial GroupKFold CV
- **Output:** trained models + metrics table
- **Novel addition:** spatial-lag augmented variant (neighboring SUHII values + lon/lat as inputs)

### Stage 6 — Explainability
- **Input:** best-performing trained model, test set
- **Process:** SHAP (global + spatial), PDP, dependence plots, GeoShapley decomposition
- **Output:** driver ranking, spatial SHAP maps, geo-interaction values

### Stage 7 — Digital Twin Output Layer (not yet built)
- **Input:** trained model, spatial SHAP/GeoShapley outputs, Dynamic World LULC layer, historical multi-year dataset
- **Intended output:**
  - Current-state SUHII map (most recent completed year)
  - Historical explorer (year slider across the rolling 2019+ window)
  - Click-a-point local SHAP driver breakdown
  - Year-over-year change maps and per-location trend lines
  - What-if scenario simulator (adjust a location's feature values, see predicted SUHII change) — the project's primary decision-support novelty
  - Population-weighted exposure/equity view

---

## 6. Data Latency — Honest System-Level Constraint

This system is a **periodic monitoring and scenario-planning tool, not a real-time dashboard.** Every data source has an inherent update lag:

| Source | Revisit / Update Frequency | Effective Lag |
|---|---|---|
| Landsat 8/9 | 16-day per satellite, 8-day combined | Up to 8 days between usable clear-sky passes (longer in monsoon) — this is the binding constraint on the whole system |
| VIIRS-DNB Monthly | Daily passes, monthly composite used | Up to ~1 month |
| Sentinel-5P TROPOMI (OFFL) | Daily coverage, ~5hr–few days processing latency | Effectively near-daily, though our aggregation window is intentionally wider |
| CHIRPS | Daily | ~2 days typical latency |
| CIESIN Population | 5-year steps | Up to 5 years stale by design — least current input in the system |
| Dynamic World | Near-real-time (Sentinel-2 based) | Matches Sentinel-2 revisit, ~5 days |

**Bottom line:** the freshest possible SUHII estimate lags real conditions by roughly 1–2 weeks minimum (Landsat-bound), and population context can be years out of date. This is standard for satellite-based UHI monitoring generally (the source paper itself only ever reports discrete historical years) and should be stated plainly in the digital twin's UI, not hidden — it positions the tool honestly as a planning aid, not a live weather service.

---

## 7. Known Gaps / Future Work

| Item | Status |
|---|---|
| Full-resolution raster export for map visualization | Currently produced via point sampling; full-raster export for the Stage 7 map layer needs a more careful, incremental retry (single-variable, single-year exports, not a bulk multi-year batch) |
| GeoShapley re-run on the largest available dataset | Only run on an earlier, smaller dataset version so far |
| Nowcasting / sub-daily prediction | Explicitly out of scope — no data source in the finalized list updates fast enough to support this meaningfully |
| Per-city summer-month validation | Confirmed for Mumbai/Delhi; needs explicit IMD-normals confirmation for Hyderabad/Bangalore before their pipelines are finalized |
