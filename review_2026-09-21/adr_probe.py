import sys
sys.path.insert(0,'/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/src')
import numpy as np,pandas as pd
from data_loading import load_raw,add_derived
from reconstruction import restore_outliers
from splitting import split
from simulation import SimConfig,simulate_enhanced,simulate_conventional
from adr_algorithm import AdrParameters
raw=add_derived(load_raw())
df=restore_outliers(raw.loc[raw.rh>2].reset_index(drop=True))
tr,te=split(df)
pred=np.load('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/outputs/final/test_predictions.npz')
print('n',len(te),'max achievable SF10',np.mean(te.snr>-15)*100,'max achievable SF12',np.mean(te.snr>-20)*100)
print('by device',te.groupby('device_id',observed=True).snr.agg(['min','mean','max']))
rows=[]
for cap in [10,12]:
 for rule in ['residual','threshold']:
  for name in ['ANN','SVR','RF','ADR']:
   lm=11 if name=='ADR' else 4
   cfg=SimConfig(lm_values=(lm,),variant='text',rule=rule,clamp_tp=True,adr_window_excl_current=True,params=AdrParameters(max_sf=cap))
   res=simulate_conventional(te,cfg) if name=='ADR' else simulate_enhanced(te,pred[name],cfg)
   rows.append(res.assign(cap=cap,rule=rule,scheme=name))
res=pd.concat(rows)
print(res[['cap','rule','scheme','LM','pdr','mean_energy_j','mean_sf','mean_tp_dbm','frac_tp_clamped_high']].to_string(index=False))
res.to_csv('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/review_2026-09-21/adr_rule_probe.csv',index=False)
# conventional temporal baseline: same residual rule and repaired physical threshold
for order in ['test_order','chronological']:
 for rule in ['residual','threshold']:
  cfg=SimConfig(lm_values=tuple(range(16)),variant='text',rule=rule,clamp_tp=True,adr_window_excl_current=True,adr_window_order=order,params=AdrParameters(max_sf=10))
  r=simulate_conventional(te,cfg)
  h=r[r.pdr>=99]
  print('ADR 99%',order,rule, 'LM '+str(h.LM.min()) if len(h) else 'NOT ACHIEVABLE; max '+str(r.pdr.max()),'LM4 pdr',r.loc[r.LM==4,'pdr'].iloc[0])
