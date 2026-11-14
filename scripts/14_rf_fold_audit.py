"""Training-only 5-fold RF audit. No held-out target evaluation or model overwrites.
Place in scripts/14_rf_fold_audit.py; run from the project environment.
Requires the uploaded no-missing-data Delhi version; assertions stop on missing values.
The 21-feature variant bypasses correlation/VIF selection only; other transforms stay fold-fitted.
"""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error,r2_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uhi.preprocess.load import load_raw
from uhi.preprocess.clean import clean_rows
from uhi.preprocess.split import assign_blocks
from uhi.preprocess.transforms import add_log_transforms,year_center,Winsorizer
from uhi.preprocess import filters as F
s=json.loads((ROOT / 'data/processed/delhi/preprocess_state.json').read_text()); pp=yaml.safe_load((ROOT / 'configs/pipeline.yaml').read_text())['preprocess']
r=load_raw(ROOT / 'data/raw','delhi');d,log=clean_rows(r,pp['min_valid_frac'],pp['placeholder_values']);d['block_id']=assign_blocks(d,s['split']['n_blocks'],s['split']['bounds']);d['split']=d.block_id.map(s['split']['block_map'])
assert d['split'].notna().all()
print('RAW',len(r),'CLEAN',len(d),'NULLS',int(d.isna().sum().sum()),flush=True)
print('LST medians',d.groupby('year').LST.median().to_dict(),flush=True)
print('DW label counts',d.groupby('year').DW_label.value_counts().to_dict(),flush=True)
t=add_log_transforms(d[d.split=='train'].reset_index(drop=True));assert not t.isna().any().any()
print('TRAIN',len(t),'BLOCKS',t.block_id.nunique(),flush=True)
results=[]
# Predeclared: original filtering vs all candidate drivers; same folds and RF params.
for fold,(a,b) in enumerate(GroupKFold(5).split(t,groups=t.block_id),1):
 mask=np.zeros(len(t),bool);mask[a]=True
 z,_=year_center(t,pp['year_center_features'],mask)
 candidates,_=F.candidate_features(pp['dw_drop_classes'],pp['dw_reference_class'])
 wz=Winsorizer(*pp['winsor_quantiles']).fit(z.iloc[a],candidates);z=wz.transform(z)
 features,_=F.drop_zero_variance(z.iloc[a],candidates)
 filtered,_=F.spearman_filter(z.iloc[a],features,pp['spearman_threshold'],pp['spearman_keep_rule'],tiers=pp['spearman_priority_tiers'])
 filtered,_,_=F.vif_filter(z.iloc[a],filtered,pp['vif_threshold'],pp['spearman_keep_rule'],pp['spearman_priority_tiers'])
 for label,cols in [('fold_filtered',filtered),('all_candidates',features)]:
  m=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=2)
  m.fit(z.iloc[a][cols],t.iloc[a].SUHII);pred=m.predict(z.iloc[b][cols]);y=t.iloc[b].SUHII
  row=dict(variant=label,fold=fold,n_features=len(cols),features=cols,rmse=float(np.sqrt(mean_squared_error(y,pred))),r2=float(r2_score(y,pred)))
  results.append(row);print(json.dumps(row),flush=True)
from datetime import datetime
out = ROOT / 'models' / 'experiments' / ('rf_fold_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
out.mkdir(parents=True, exist_ok=False)
(out / 'fold_results.json').write_text(json.dumps(results,indent=2), encoding='utf-8')
pd.DataFrame(results).drop(columns='features').to_csv(out / 'fold_results.csv', index=False)
print('Results saved:', out)
print(pd.DataFrame(results).groupby('variant')[['rmse','r2']].agg(['mean','std']).to_string(),flush=True)
