# Delhi Urban Heat Digital Twin

**From satellite observations to spatial heat estimates and interactive scenario exploration.**

This project maps Delhi's *surface urban heat island intensity* (SUHII): how much warmer a location's land surface is than a local rural reference. It combines Earth observation data, spatial machine learning and a scenario explorer to investigate where heat concentrates and how predictions respond to changes in measured environmental indicators.

> **Scope:** Satellite-derived land-surface heat, not air temperature. Scenario outputs show model sensitivity to edited predictors; they are not measured cooling effects, causal claims or forecasts.

## At a glance

| Stage | What is available |
| --- | --- |
| Data | Google Earth Engine export pipeline and Delhi observations for 2024–2026 |
| Preparation | 1 km cells, documented preprocessing, spatial train/test blocks and provenance |
| Modelling | Random Forest, Gradient Boosting, XGBoost and LightGBM comparisons; neighbourhood and contrast feature experiments |
| Evaluation | Saved spatial cross-validation runs and a 1,350-row spatial test comparison |
| Exploration | Streamlit app for historical cell-level predictor scenarios and downloadable results |

**Saved test comparison:** XGBoost with 3 km neighbourhood context reached **RMSE 0.618 °C and R² 0.739**, versus **RMSE 0.689 °C and R² 0.675** for its reference configuration ([test metrics](models/final_evaluation/20261114_231907_965985/test_summary.csv)). The test area had already been evaluated in earlier project work, so this is **not an untouched final holdout**. A separate [development benchmark](models/experiments/novelty_benchmark_20261117_111010_101180/summary.csv) compares four model families and feature sets; its cross-validation scores should not be treated as test scores.

## How it works

1. **Observe:** Earth Engine assembles Landsat 8/9 land-surface temperature and spectral indicators with land cover, night lights, population, atmospheric and rainfall data.
2. **Prepare:** Aggregate to 1 km cells, compute SUHII against a local rural reference, clean inputs and split by spatial block before fitting preprocessing decisions.
3. **Compare:** Evaluate models and neighbourhood features with saved fold assignments, predictions and metrics.
4. **Explore:** Load a saved XGBoost model in the Streamlit app, change historical predictors for one cell and inspect the predicted difference.

The app currently uses the saved **3 km neighbourhood XGBoost** bundle. It does **not** run the newer context-and-contrast benchmark model. Its map shows cell locations; predicted changes appear in the table and downloadable CSV.

## Explore the simulation

From the repository root, use Python 3.10+:

```bash
python -m venv .venv
# Activate .venv with your shell's usual command.
python -m pip install -r requirements.txt
python -m pip install streamlit
python -m streamlit run simulation/app.py
```

The app uses the committed Delhi raw data, preprocessing state and saved model bundle. Before showing scenarios, it verifies that its baseline predictions reproduce the saved test predictions. Change **NDVI, NDBI, albedo** or transfer Dynamic World *built* probability to *trees*; inspect baseline and scenario SUHII, their difference and exported CSV/JSON. Dynamic World values are class probabilities, not verified land-area fractions. Related environmental variables are not automatically adjusted.

## Reproduce the data pipeline

New Earth Engine exports require authentication and `export.gee_project` in [pipeline.yaml](configs/pipeline.yaml). Existing Delhi data and saved outputs can be explored without exporting again.

```bash
python -m pip install -e ".[gee,geo,parquet,notebook,dev]"
python -m pytest -q
earthengine authenticate
python scripts/01_export_gee.py --preflight-only
python scripts/01_export_gee.py
python scripts/02_fetch_gee.py
python scripts/03_preprocess.py
```

For modelling, start with the [preprocessing notebook](notebooks/01_preprocessing_review.ipynb), then inspect the [saved test comparison](scripts/20_final_test_comparison.py) and [feature benchmark](scripts/24_novelty_benchmark.py). Results are in [models/experiments](models/experiments/). Review each script's paths and settings before rerunning it.

## Repository guide

| Path | Contents |
| --- | --- |
| [configs](configs/) | City and pipeline settings |
| [src/uhi](src/uhi/) | Earth Engine acquisition and preprocessing modules |
| [data](data/) | Delhi raw observations and processed data |
| [scripts](scripts/) | Export, preprocessing, training and evaluation |
| [models](models/) | Saved models, diagnostics and results |
| [notebooks](notebooks/) | Analysis and comparison walkthroughs |
| [simulation](simulation/) | Scenario engine, Streamlit interface and checks |
| [docs](docs/) | Architecture, specifications and decisions |

## Interpreting the results

- Spatial separation reduces the risk of evaluating on nearby training locations. Read [decisions and limitations](docs/DECISIONS.md) for leakage controls and assumptions.
- Metrics describe prediction against satellite-derived SUHII. Lower predicted SUHII under an edited input does not establish that a real intervention will cause that cooling.
- The 2026 study window is incomplete relative to prior full years; consult the pipeline configuration and export log before comparing years.

**Project focus:** An explainable, reproducible foundation for urban heat monitoring and careful what-if analysis, starting with Delhi.
