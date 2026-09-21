import sys,json,os,time
from pathlib import Path
sys.path.insert(0, str(Path('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/src')))
import numpy as np,pandas as pd
from scipy import stats
from data_loading import load_raw,add_derived
from reconstruction import restore_outliers
from splitting import split
from cpls_models import MLRCpls,make_X,make_y,make_rf
from sklearn.metrics import mean_squared_error

def fit_score(train,test):
 m=MLRCpls().fit(make_X(train),make_y(train));r=make_y(test)-m.predict(make_X(test))
 return dict(train_rows=len(train),test_rows=len(test),rmse=float(np.sqrt(np.mean(r*r))),mae=float(np.mean(abs(r))),bias=float(r.mean()))
raw=add_derived(load_raw());clean=raw[~(raw.rh<=2)].reset_index(drop=True);rec=restore_outliers(clean)
print('profiles',clean.groupby('device_id',observed=True).agg(n=('index','size'),distances=('distance','nunique'),distance=('distance','first'),start=('timestamp','min'),end=('timestamp','max'),BP_mean=('bp','mean'),BP_sd=('bp','std')).to_json(),flush=True)
print('duplicates',int(clean.duplicated(['device_id','timestamp']).sum()),flush=True)
print('raw-target-consistency',np.max(abs(clean.experimental_pl-clean.pl_from_link_budget)),flush=True)
for label,df in [('released_clean',clean),('synthetic',rec)]:
 tr,te=split(df);print(label,'random',fit_score(tr,te),flush=True)
 m=MLRCpls().fit(make_X(tr),make_y(tr));r=make_y(tr)-m.predict(make_X(tr))
 print(label,'residual',dict(excess_kurtosis=float(stats.kurtosis(r)),q01=float(np.quantile(r,.01)),q99=float(np.quantile(r,.99))),flush=True)
 # Later 20% packets within each node, avoiding any mixed time boundary
 df=df.sort_values(['device_id','timestamp'],kind='stable')
 within=df.groupby('device_id',observed=True).cumcount();lens=df.groupby('device_id',observed=True)['device_id'].transform('size')
 mask=within<np.floor(.8*lens);print(label,'temporal_per_node',fit_score(df[mask],df[~mask]),flush=True)
 for node in df.device_id.unique():
  mask=df.device_id!=node;print(label,'holdout_'+node,fit_score(df[mask],df[~mask]),flush=True)
tr,te=split(rec);idx=np.random.default_rng(42).choice(len(tr),100000,replace=False)
Xtr,Xte,ytr,yte=make_X(tr),make_X(te),make_y(tr),make_y(te)
for label,cols in [('RF100k_all',list(range(7))),('RF100k_distance_frequency_SNR',[0,1,6]),('RF100k_distance_frequency_weather',[0,1,2,3,4,5])]:
 start=time.time();model=make_rf(n_jobs=4).fit(Xtr[idx][:,cols],ytr[idx]);pred=model.predict(Xte[:,cols]);rmse=float(np.sqrt(np.mean((pred-yte)**2)))
 print(label,dict(rmse=rmse,seconds=time.time()-start),flush=True)
print('finished',flush=True)
