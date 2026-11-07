# Decisions, assumptions, caveats (review before trusting any model result)

## 1. What is persisted (and why only this)
| File | Why it is kept |
|---|---|
| `data/raw/<city>_<year>.csv` (1 per year) | Expensive Earth Engine output; avoids re-running 30 m SUHII computation |
| `data/raw/<city>_export_log.json` (1 per city) | Provenance: season/TROPOMI windows, Landsat scene counts, GPW year, asset ids |
| `data/processed/<city>/<city>_ml_ready.parquet` | The ONLY input the ML stage reads |
| `data/processed/<city>/preprocess_state.json` | Frozen train/test block map, final features, winsor bounds, full decision log, gate results |

Everything else (correlation matrices, VIF tables, drop logs, plots) is recomputed in memory by
`notebooks/01_preprocessing_review.ipynb`. Nothing diagnostic is written to disk.

## 2. Data transfer / storage
Earth Engine now exports one table per year to an **EE asset** (your EE quota, not Google Drive) and
`scripts/02_fetch_gee.py` pulls it straight into `data/raw/`. Size is small: ~1,600 cells x ~30 columns is well under 1 MB
per year; even 100 m cells (~100x more rows) would be tens of MB per year (rough estimate). If a city ever outgrows that,
switch the export to Cloud Storage + Parquet. `export.delete_asset_after_fetch: true` frees EE storage after download.

## 3. How retain/remove decisions stay unbiased
| Risk | Safeguard |
|---|---|
| Test cells influencing preprocessing | Split first; winsorization, Spearman, VIF fit on train rows only |
| Same place in train (one year) and test (another) | Blocks depend on location only; gate `split_cell_integrity` fails if any cell is in both |
| Split shifting when a year is added | Block map + bounds frozen in `preprocess_state.json`; new blocks assigned by seed-hash |
| Outcome steering feature choice | Default `spearman_keep_rule: priority` never looks at SUHII. Tiers are set a priori (planner-controllable > proxy > atmospheric) |
| Silent assumptions | Any Dynamic World class dropped up front must average <= 1% cover or a WARN is raised (Delhi: snow_and_ice 4.3%, flooded_vegetation 3.7% -> now kept as candidates); every column has exactly one logged outcome (tested) |
| Target leakage | LST/LST_ref excluded; lon/lat/year/block never features; hot/cold cells never removed; the target is never imputed |
| Imputation leaking across the split | A missing feature is copied only from a donor in the SAME year and SAME split, within 3 km (tested) |
| Year identification | Coarse context features (SO2, CO, NO2, O3, AAI, Rain_period: one value per city-year) are centred on the year's TRAIN-cell mean so they cannot act as year labels; land-surface drivers are not centred |
| Outliers | Features winsorized at train 1st/99th percentile; SUHII target untouched |

**What remains a judgement call** (all in `configs/pipeline.yaml`): the tier lists, |rho| > 0.8, VIF <= 8, 1/99 winsorization,
`min_valid_frac` 0.5, 1 km cells, `crops` as DW reference. They are not fitted to outcomes, but they are choices.
Dropping a correlated variable does not remove its information: it is carried by the survivor, so what-if simulations must move
correlated features together, and Dynamic World shares must still sum to 1.

## 4. Deviations from the spec
| Spec | Here | Reason |
|---|---|---|
| Filters (H) before split (I) | Split first | Spec 8.3 forbids test cells from fitting any filter |
| Log in source step | Raw NTL/Pop exported, `log1p` in Python | Spec's own order puts transforms after aggregation |
| `DW_label` = mode of pixel labels | Argmax of the cell's season-mean class probabilities, derived in Python | Removes a second export table; used only for maps/area stats, never a feature |
| Which member of a pair to drop undefined | Priority tiers (see 3) | Previous unsupervised rule flipped on 0.007 differences and removed NDVI and DW_built |
| VIF: drop highest | Among variables over the threshold, drop the lowest-priority tier first | Same reason |
| Peak-heat season only | **Full-year windows** (see section 8) | Your instruction: more of the year, 2026 up to the latest complete month |
| No imputation | **Nearest-cell imputation of FEATURES only** (see section 8) | Your instruction; restricted so it cannot leak or invent labels |
| Imputation in step F, before the split | Split first, then impute | Same-split donors only; spec 8.3 |
| TROPOMI: 14-day window | Whole window by default (`tropomi.mode: period`); `narrow` still available | A 14-day mean next to a full-year composite would be inconsistent |
| `Rain_season` | Renamed `Rain_period` (total rainfall over the row's window) | It is no longer a season |
| `valid_frac` | Full cell incl. outside-AOI area. **two bugs fixed**: clip() hid outside pixels, and unmask(0) kept the clipped footprint (needs `unmask(0, False)`), so valid_frac was ~1 everywhere. Re-export needed | Border cells must be dropped when < 50% inside Delhi |

## 5. Not verified
1. **Earth Engine code has never run** (sandbox cannot reach EE). Untested: asset export, `toList` chunked fetch, band names,
   runtime limits. First thing to do: `scripts/01_export_gee.py --preflight-only`, then one year, then all three.
2. **Source coverage for 2026** (VIIRS monthly, Dynamic World are the likely laggards). The pre-flight aborts and tells you to use earlier years.
3. The 3 km pixel-level SUHII neighbourhood is heavy. If it times out: raise `export.tile_scale`, then `suhii.reference_scale_m: 120` (coarser reference; slightly changes the definition).
4. Preprocessing was tested with a tiny in-code fixture (`tests/fixtures.py`, ~30 lines, kept only so CI guards the logic); real Delhi 3-year results are yours to review in the notebook.
5. Albedo uses a Liang-type formula mapped to OLI bands: confirm the coefficients against a citation before publication.

## 6. Scientific caveats for interpreting the model
- **A full-year SUHII is a different quantity from the pre-monsoon one.** It is the median of all clear-sky scenes over the year
  (cool winter, hazy post-monsoon, sparse cloudy monsoon scenes included), so it describes year-round surface contrast, not the
  summer peak. The twin's heat-mitigation story is about summer; if that is the focus, keep a season-specific model too.
- **The partial 2026 composite is not like-for-like with 2024/2025.** It lacks the last months of the year, so its medians
  (LST, NDVI, NDBI, albedo, rain total) are seasonally biased relative to a full year. The `period_coverage` gate warns about
  this; `align_months: true` trims every year to the same months if you want clean year comparisons.
- Delhi's monsoon (Jul-Sep) leaves few cloud-free Landsat scenes: those months contribute little to a yearly median.
- **Partly circular target**: the rural reference uses NDVI and NDBI, which are also features. Rural-like cells have SUHII near 0 by
  construction, and the model can learn the mask boundary. Treat NDVI/NDBI importance with this in mind.
- Population is the 2020 GPW layer for all years (cannot explain change over time). TROPOMI (1-7 km) and CHIRPS (5 km) vary smoothly
  across a city and can act as location proxies; check SHAP for this.
- Years are pooled for filtering, so the same cell contributes three correlated rows. Use `GroupKFold(block_id)` and do not read
  CV scores as independent-sample accuracy.
- Season-median composites smooth out single hot days. SUHII is a skin-temperature contrast, not air temperature.
- Thresholds in the gates (SUHII median 0-4, etc.) come from Mumbai; the "literature check" gate is manual.

## 7. What the ML-ready parquet can and cannot do (simulation / what-if)
The parquet is the **training table**, not the simulator. A highway scenario needs four more things:
1. **A trained, validated model** (Stage 5). Tree ensembles are step functions: a small input change often moves the prediction
   by 0 or by a jump. Use monotonic constraints (NDBI up, vegetation down) and compare with a smooth baseline.
2. **An intervention -> feature-change mapping.** A highway is ~40-60 m wide, i.e. a few percent of a 1 km cell. Its footprint
   must be intersected with the grid and converted into area-weighted changes of NDBI, Albedo, vegetation/land-cover shares
   (asphalt endmember values from literature or from 30 m pixels over existing roads). The twin must also apply, in order,
   the transform in `preprocess_state.json -> feature_spec` (log1p / year centring), then clip to the winsor bounds.
3. **Consistency rules:** Dynamic World shares must keep summing to 1 (the dropped reference class `crops` and any other
   non-listed class absorb the remainder); correlated features move together (a dropped feature's information lives in the survivor).
4. **Honest uncertainty:** the model is associative, learned from spatial differences between cells, not a causal estimate of
   building a road. It has no neighbour spillover, no traffic heat or tailpipe emissions (TROPOMI is 1-7 km, NTL is ~500 m), and the
   SUHII reference itself is local. Expected effects for a few percent of a cell are small compared with SUHII spread
   (std about 1.6 C), so report a range, not a number. Validate on a real road built between 2019 and today (needs earlier years).

## 8. Changes in version 0.3 (full-year window, nearest-cell imputation)
**Window.** `window: full_year`: 2024 and 2025 are 1 Jan-31 Dec. 2026 runs from 1 Jan to the end of the latest COMPLETE month
that EVERY source covers; `scripts/01_export_gee.py` prints each source's coverage and the binding one (VIIRS monthly is the
likely limiter). Exact windows are written to `data/raw/<city>_export_log.json` and the `period_coverage` gate reports months
per year. `window: season` restores the April-June design.

**Imputation.** What is and is not done:
| Done | Not done |
|---|---|
| A missing feature value is copied from the nearest cell that has it (same year, same split, within 3 km) | The target SUHII (and LST) is never imputed: a row without it is removed, because invented labels would corrupt training and testing |
| The nine Dynamic World shares are copied together from one donor so they still sum to 1 | No cross-split borrowing, no extrapolation beyond 3 km: the row is removed instead |
| Rows missing >30% of feature groups are removed | A feature missing for a whole year raises an error (an export problem, not an imputation job) |
| `n_imputed` per row is stored; the gate warns if >5% of any feature was imputed | |

Before this change the Apr-Jun data had 0 missing values, so imputation changes nothing there; it matters for full-year
composites, where cloudy months can leave gaps. Cell-level (not 30 m pixel-level) imputation was chosen because the model
table is built from 1 km cells and cell values are what the gates and split reason about.

**Compute warning.** A full-year composite uses roughly 3-4x more Landsat scenes and (by default) a full-period TROPOMI mean,
so each export will take longer and cost more EECU-seconds than the 3-month runs. Fallbacks, in order: raise
`export.tile_scale`; `tropomi.mode: narrow`; `suhii.reference_scale_m: 120`. Never run all three years before one year succeeds.

**Gates.** `lst_range` widened to [10, 60] (full-year medians include cool months). The SUHII median band 0-4 is a WARN only and
comes from Mumbai's pre-monsoon runs.

**Old data.** Raw CSVs from the Apr-Jun run use the column `Rain_season` and are rejected by the new loader; re-export.
