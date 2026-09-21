import sys
sys.path.insert(0,'/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/src')
import numpy as np,pandas as pd
from data_loading import load_raw,add_derived
from reconstruction import restore_outliers
from splitting import split
raw=add_derived(load_raw())
clean=raw[raw.rh>2].reset_index(drop=True)
restored=restore_outliers(clean)
tr,te=split(restored)
_,te_clean=split(clean)
rows=[]
for name,d in [('raw_all',raw),('clean_all',clean),('clean_test',te_clean),('restored_all',restored),('restored_test',te)]:
 print(name,'SF10 max PDR',np.mean(d.snr>-15)*100,'SF12 max PDR',np.mean(d.snr>-20)*100)
 for order in ['test_order','chronological']:
  x=d.sort_values(['device_id','timestamp'],kind='mergesort') if order=='chronological' else d
  mx=x.groupby('device_id',observed=True).snr.transform(lambda s:s.rolling(20,min_periods=1).max().shift(1).fillna(s))
  err=mx-x.snr
  rows.append(dict(data=name,order=order,lm99=np.quantile(err,.99),lm95=np.quantile(err,.95),pdr4=np.mean(err<4)*100,pdr0=np.mean(err<0)*100))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/review_2026-09-21/adr_baseline_probe.csv',index=False)
# chronological preceding 20 samples in the full clean campaign, score original test row ids
for name,d,test in [('clean_all_history',clean,te_clean),('restored_all_history',restored,te)]:
 x=d.sort_values(['device_id','timestamp'],kind='mergesort').copy()
 x['err']=x.groupby('device_id',observed=True).snr.transform(lambda s:s.rolling(20,min_periods=1).max().shift(1).fillna(s))-x.snr
 err=x.set_index('index').loc[test['index'],'err']
 print(name,dict(lm99=np.quantile(err,.99),lm95=np.quantile(err,.95),pdr4=np.mean(err<4)*100,pdr0=np.mean(err<0)*100))
