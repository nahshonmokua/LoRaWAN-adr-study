import sys
from pathlib import Path
sys.path.insert(0,str(Path('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/src')))
import numpy as np,pandas as pd
from data_loading import load_raw,add_derived
from reconstruction import restore_outliers
from splitting import split
from cpls_models import MLRCpls,make_X,make_y
raw=add_derived(load_raw());clean=raw[~(raw.rh<=2)].reset_index(drop=True)
def score(tr,te,snr_key):
 xtr=make_X(tr);xte=make_X(te);xtr[:,-1]=tr[snr_key];xte[:,-1]=te[snr_key]
 est=MLRCpls().fit(xtr,make_y(tr));r=make_y(te)-est.predict(xte)
 return dict(rmse=float(np.sqrt(np.mean(r*r))),bias=float(np.mean(r)),LM99=float(np.quantile(r,.99)),train_rows=len(tr),test_rows=len(te))
for label,df in [('released_clean',clean),('synthetic',restore_outliers(clean))]:
 df=df.sort_values(['device_id','timestamp','index'],kind='stable').reset_index(drop=True)
 df['snr_lag1']=df.groupby('device_id',observed=True).snr.shift(1)
 df['snr_past20_mean']=df.groupby('device_id',observed=True).snr.transform(lambda x:x.shift(1).rolling(20,min_periods=1).mean())
 df=df[df.snr_lag1.notna()].reset_index(drop=True)
 random=split(df)
 within=df.groupby('device_id',observed=True).cumcount();lens=df.groupby('device_id',observed=True)['device_id'].transform('size');mask=within<np.floor(.8*lens)
 for spl,(tr,te) in [('random',random),('temporal_per_node',(df[mask],df[~mask]))]:
  for k in ['snr','snr_lag1','snr_past20_mean']:
   print(label,spl,k,score(tr,te,k),flush=True)
