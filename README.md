# UHI Digital Twin (Delhi first)

Satellite SUHII pipeline for Indian cities: Earth Engine -> 1 km cells -> preprocessing -> (later) spatial ML, XAI, twin.
Specs: `docs/UHI_Final_System_Architecture.md`, `docs/UHI_Preprocessing_Specification.md`.
**Read `docs/DECISIONS.md` first** (what is kept, why decisions are unbiased, what is untested).

| Stage | State |
|---|---|
| Config, 3-year window (`configs/`, `uhi.config`) | done, tested |
| Earth Engine export + fetch (`uhi.gee`) | written, **not yet run against Earth Engine** |
| Preprocessing (`uhi.preprocess`) | done, 36 tests; review on real data via the notebook |
| Models / XAI / Twin | stubs |

## Workflow
```bash
pip install -e ".[gee,geo,parquet,notebook,dev]"
pytest -q
earthengine authenticate          # set export.gee_project in configs/pipeline.yaml first

python scripts/01_export_gee.py --preflight-only   # coverage per source + the exact window of each year
python scripts/01_export_gee.py                    # one EE batch export per year (assets, not Drive)
python scripts/02_fetch_gee.py                     # when tasks finish -> data/raw/delhi_<year>.csv
python scripts/03_preprocess.py                    # -> data/processed/delhi/ (2 files)
jupyter lab notebooks/01_preprocessing_review.ipynb   # review every decision before modelling
```
Study window (`configs/pipeline.yaml`): `window: full_year` = 1 Jan-31 Dec per year, the running year ending at the latest
complete month all sources cover; `window: season` = April-June. `align_months: true` makes every year cover the same months.
Missing feature values are imputed from the nearest cell (same year and split); the target is never imputed.
Change years with `years:` or `--years 2023 2024 2025`.

## Persisted outputs
`data/raw/<city>_<year>.csv`, `data/raw/<city>_export_log.json`, `data/processed/<city>/<city>_ml_ready.parquet`,
`data/processed/<city>/preprocess_state.json`. Diagnostics live in the notebook, not on disk.

## Layout
```
configs/    cities.yaml, pipeline.yaml
src/uhi/    config.py | gee/ (landsat, suhii, sources, grid, export, fetch, availability) | preprocess/ | models/ xai/ twin/ (stubs)
scripts/    01_export_gee.py  02_fetch_gee.py  03_preprocess.py
notebooks/  01_preprocessing_review.ipynb
tests/      config + preprocessing tests (tiny in-code fixture)
docs/       specs, DECISIONS.md, PREPROCESSING_REPORT.md
```
Adding a city: add AOI, IMD-confirmed season and TROPOMI window to `configs/cities.yaml` (UTM zone is derived from longitude).
