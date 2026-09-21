import sys
sys.path.insert(0,'/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/src')
import numpy as np,pandas as pd
from data_loading import load_raw,add_derived
from reconstruction import restore_outliers
from splitting import split
from adr_algorithm import AdrParameters,enhanced_adr_vectorized
from simulation import _adr_from_snrmax
from energy import PowerModel,time_on_air
raw=add_derived(load_raw()); _,te=split(restore_outliers(raw[raw.rh>2].reset_index(drop=True)))
pred=np.load('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/outputs/final/test_predictions.npz')
snrmax=te.groupby('device_id',observed=True).snr.transform(lambda x:x.rolling(20,min_periods=1).max().shift(1).fillna(x)).to_numpy()
lb={c:te[c].to_numpy(float) for c in ('ltx','gtx','lrx','grx')}
rows=[]
for cap in [10,12]:
 p=AdrParameters(max_sf=cap)
 for name in ['ADR','ANN','SVR','RF']:
  if name=='ADR': tp,sf=_adr_from_snrmax(snrmax,te.sf.to_numpy(),11,p)
  else: tp,sf,*_=enhanced_adr_vectorized(pred[name],te.ptx.to_numpy(),te.sf.to_numpy(),4,te.noise_power_rssi.to_numpy(),p,'text',True,lb)
  for quant in [None,2]:
   tp2=tp if quant is None else np.minimum(20,2*np.ceil(tp/2))
   for payload in [1,14]:
    toa=time_on_air(sf,payload)
    rows.append(dict(cap=cap,scheme=name,tp_quant=quant or 0,payload_bytes=payload,mean_energy_mJ=np.mean(PowerModel().energy_j(tp2,toa))*1000,mean_toa_ms=np.mean(toa)*1000))
r=pd.DataFrame(rows)
r['energy_savings_pct']=r.groupby(['cap','tp_quant','payload_bytes']).mean_energy_mJ.transform(lambda s:100*(s.iloc[0]-s)/s.iloc[0])
print(r.to_string(index=False));r.to_csv('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/review_2026-09-21/adr_energy_probe.csv',index=False)
print('ToA SF7 SF10 SF12, PHY bytes1',time_on_air([7,10,12],1)*1000,'PHY bytes14',time_on_air([7,10,12],14)*1000)
print('frame_lengths',raw.frame_length.value_counts().to_dict())
