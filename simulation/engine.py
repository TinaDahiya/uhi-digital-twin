"""Inference for scripts/20_final_test_comparison.py bundles; never fits a model."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def context_features(raw, bundle):
    cols = bundle['context_columns']
    result = pd.DataFrame(index=raw.index)
    for _, area in raw.groupby('year', sort=True):
        if len(area) < 2:
            raise ValueError('At least two same-year cells are required.')
        coords = np.radians(area[['lat', 'lon']].to_numpy(float))
        radius = (float(bundle['context_radius_km']) + 1e-8) / 6371.0088
        neighbours = BallTree(coords, metric='haversine').query_radius(coords, r=radius)
        values, ids = area[cols].to_numpy(float), area.cell_id.to_numpy()
        for pos, (idx, near) in enumerate(zip(area.index, neighbours)):
            near = near[ids[near] != ids[pos]]
            pool = values[near] if len(near) else values[ids != ids[pos]]
            result.loc[idx, ['ctx_' + c for c in cols]] = pool.mean(axis=0)
            result.loc[idx, 'ctx_count'] = len(near)
            result.loc[idx, 'ctx_fallback'] = int(len(near) == 0)
    return result


def design(raw, bundle):
    d = raw.copy()
    d['NTL_log'] = np.log1p(d.NTL.clip(lower=0))
    d['Pop_log'] = np.log1p(d.Pop.clip(lower=0))
    for year, area in d.groupby('year'):
        means = bundle['year_means'].get(int(year), bundle['year_means'].get(str(int(year))))
        if means is None:
            raise ValueError(f'No saved centering means for year {year}.')
        for col, mean in means.items():
            d.loc[area.index, col] -= mean
    for col, (lo, hi) in bundle['base_winsor_bounds'].items():
        d[col] = d[col].clip(lo, hi)
    X = d[bundle['base_features']].copy()
    if bundle['context_features']:
        ctx = context_features(raw, bundle)
        for col, (lo, hi) in bundle['context_winsor_bounds'].items():
            ctx[col] = ctx[col].clip(lo, hi)
        X = pd.concat([X, ctx[bundle['context_features']]], axis=1)
    X = X[bundle['features']]
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError('Non-finite model inputs. No automatic imputation is applied.')
    return X


def predict(raw, bundle):
    return np.asarray(bundle['model'].predict(design(raw, bundle)), float)


def replay_check(raw, bundle, folder):
    path = Path(folder) / 'test_predictions.csv'
    if not path.exists():
        raise ValueError('Missing test_predictions.csv beside bundle; cannot verify inference replay.')
    saved = pd.read_csv(path)
    saved = saved.loc[saved.variant.eq('neighbour_3km')].copy()
    if saved.empty or saved.duplicated(['cell_id', 'year']).any():
        raise ValueError('Saved neighbour_3km test predictions missing or duplicated.')
    current = raw[['cell_id', 'year']].copy()
    current['replayed'] = predict(raw, bundle)
    matched = saved.merge(current, on=['cell_id', 'year'], how='left', validate='one_to_one')
    if matched.replayed.isna().any():
        raise ValueError('Some saved test cells are missing from reconstructed imagery.')
    error = float(np.max(np.abs(matched.prediction - matched.replayed)))
    if error > 1e-5:
        raise ValueError(f'Inference replay mismatch: max error {error:.8g}. Check raw data and bundle versions.')
    return len(matched), error
