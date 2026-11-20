"""Run from project root: python -m streamlit run simulation/app.py"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uhi.preprocess.load import load_raw
from uhi.preprocess.clean import clean_rows
from uhi.preprocess.filters import candidate_features
from uhi.preprocess.transforms import add_log_transforms
from engine import predict, design, replay_check

st.set_page_config(page_title='Delhi UHI Scenario Explorer', layout='wide')
st.title('Delhi UHI Scenario Explorer')
st.caption('Saved XGBoost · 3-km neighbourhood · historical predictor sensitivity')
st.info('Predictions represent SUHII relative to a local rural reference, in °C. These are model responses to edited predictors, not validated intervention effects or future forecasts.')

bundles = sorted((ROOT / 'models/final_evaluation').glob('*/neighbour_3km_bundle.joblib'))
if not bundles:
    st.error('No neighbour_3km_bundle.joblib found under models/final_evaluation.'); st.stop()
selected = st.sidebar.selectbox('Saved model bundle', bundles, format_func=lambda p: p.parent.name)

@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)

@st.cache_data
def load_inputs(signature):
    state = json.loads((ROOT / 'data/processed/delhi/preprocess_state.json').read_text(encoding='utf-8-sig'))
    pp = yaml.safe_load((ROOT / 'configs/pipeline.yaml').read_text(encoding='utf-8-sig'))['preprocess']
    raw = load_raw(ROOT / 'data/raw', 'delhi', years=state['years'])
    raw, _ = clean_rows(raw, pp['min_valid_frac'], pp['placeholder_values'])
    features, _ = candidate_features(pp['dw_drop_classes'], pp['dw_reference_class'])
    checked = add_log_transforms(raw)
    needed = features + ['SUHII', 'lat', 'lon']
    if not np.isfinite(checked[needed].to_numpy(float)).all():
        raise ValueError('Inputs contain missing values. This bundle training script did not impute them; stop and inspect data.')
    if raw.duplicated(['cell_id', 'year']).any():
        raise ValueError('Duplicate cell/year rows.')
    return raw

try:
    files = [ROOT / 'data/processed/delhi/preprocess_state.json', ROOT / 'configs/pipeline.yaml']
    files += sorted((ROOT / 'data/raw').glob('delhi_*.csv'))
    signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    bundle = load_bundle(str(selected), selected.stat().st_mtime_ns)
    raw = load_inputs(signature)
    @st.cache_data
    def verify(path, modified, signature):
        return replay_check(load_inputs(signature), load_bundle(path, modified), Path(path).parent)
    n, error = verify(str(selected), selected.stat().st_mtime_ns, signature)
    st.sidebar.success(f'Baseline replay passed: {n} held-out rows; max error {error:.2g} °C')
except Exception as exc:
    st.error(str(exc)); st.stop()

year = st.sidebar.selectbox('Historical year', sorted(raw.year.unique()), index=len(raw.year.unique()) - 1)
area = raw.loc[raw.year.eq(year)].copy()
cell = st.sidebar.selectbox('Cell', area.cell_id.tolist())
idx = area.index[area.cell_id.eq(cell)][0]
row = area.loc[idx]
st.sidebar.caption(f'Latitude {row.lat:.5f} · Longitude {row.lon:.5f}')
st.subheader('Scenario inputs')
st.caption('Edit measured predictor values directly. NDVI is an index, not green-cover percentage. Related spectral changes must be supplied by the user; they are not inferred from land-cover edits.')
scenario = area.copy()
columns = st.columns(3)
for container, col, bounds in zip(columns, ['NDVI', 'NDBI', 'Albedo'], [(-1., 1.), (-1., 1.), (0., 1.)]):
    with container:
        value = st.number_input(col, value=float(row[col]), step=0.01, key=f'{year}_{cell}_{col}')
        if not bounds[0] <= value <= bounds[1]:
            st.error(f'{col} must lie in {bounds}.'); st.stop()
        scenario.loc[idx, col] = value
        if value < area[col].min() or value > area[col].max():
            st.warning('Outside this year’s observed range; extrapolation may be unreliable.')

st.caption('Dynamic World values are mean class probabilities, not verified land-area fractions. The optional transfer below preserves their original sum.')
transfer = st.slider('Transfer DW_built probability to DW_trees (percentage points)', 0.0, float(max(0, row.DW_built * 100)), 0.0, step=0.1, key=f'transfer_{year}_{cell}') / 100
scenario.loc[idx, 'DW_built'] -= transfer
scenario.loc[idx, 'DW_trees'] += transfer
if scenario.loc[idx, 'DW_trees'] > 1 + 1e-8:
    st.error('DW_trees exceeds 1. Reduce the transfer.'); st.stop()

try:
    before, after = predict(area, bundle), predict(scenario, bundle)
    pos = area.index.get_loc(idx)
    delta = after - before
    a, b, c = st.columns(3)
    a.metric('Baseline predicted SUHII', f'{before[pos]:.3f} °C')
    b.metric('Scenario predicted SUHII', f'{after[pos]:.3f} °C')
    c.metric('Scenario − baseline', f'{delta[pos]:+.3f} °C')
    st.caption('Negative difference = lower predicted SUHII. Context is rebuilt for every cell in the selected year, using the saved radius, self-exclusion and fallback policy.')
    X0, X1 = design(area, bundle), design(scenario, bundle)
    if np.array_equal(X0.to_numpy(), X1.to_numpy()):
        st.warning('These edits produce no change in model inputs, possibly because of clipping or unused features.')
    result = area[['cell_id', 'year', 'lat', 'lon']].copy()
    result['baseline_suhii_c'] = before
    result['scenario_suhii_c'] = after
    result['delta_suhii_c'] = delta
    result['selected_cell'] = result.cell_id.eq(cell)
    st.subheader('Study cells')
    st.map(result[['lat', 'lon']])
    st.caption('Map shows cell locations. The table below contains predicted changes; map points are not a temperature heatmap.')
    st.dataframe(result.sort_values('delta_suhii_c'), hide_index=True, use_container_width=True)
    st.download_button('Download scenario results', result.to_csv(index=False), 'scenario_results.csv', 'text/csv')
    changes = {col: {'baseline': float(row[col]), 'scenario': float(scenario.loc[idx, col])} for col in ['NDVI','NDBI','Albedo','DW_built','DW_trees']}
    details = dict(bundle=str(selected.relative_to(ROOT)), year=int(year), cell_id=str(cell), inputs=changes, interpretation='SUHII predictor sensitivity; not causal intervention evidence', fixed_local_rural_reference=True)
    st.download_button('Download scenario settings', json.dumps(details, indent=2), 'scenario_settings.json', 'application/json')
    st.caption('Local rural reference is held conceptually fixed. A physical intervention that changes rural reference conditions requires separate reference reconstruction. This is not the latest nested contrast model.')
except Exception as exc:
    st.error(str(exc))
