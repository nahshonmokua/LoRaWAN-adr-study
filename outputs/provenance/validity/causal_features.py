"""Causal evaluation: the EN can only know the PREVIOUS packet's SNR / noise floor before transmitting.
Chronological per-device lag features; conventional ADR with chronological history.  Released and
reconstructed data.  LM needed for 95/99 % under the residual rule.  Run from the project root after run_final.py (~8 min, four ANN fits); writes causal_features.csv here."""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from cpls_models import MLRCpls, make_ann, make_X, make_y
from data_loading import load_cached
from reconstruction import restore_outliers
from splitting import split
import paper_spec as spec
raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True)
def lagged(df):
    d = df.sort_values(["device_id", "timestamp"], kind="mergesort").copy()
    g = d.groupby("device_id", observed=True)
    d["snr_prev"] = g.snr.shift(1); d["snr_prev20"] = g.snr.transform(lambda x: x.rolling(20, min_periods=1).mean().shift(1))
    d["snr_max_prev20"] = g.snr.transform(lambda x: x.rolling(20, min_periods=1).max().shift(1))
    return d.dropna(subset=["snr_prev", "snr_prev20", "snr_max_prev20"]).reset_index(drop=True)
def lm_for(resid, q): return float(np.percentile(resid, q))
rows = []
for dname, base in (("released", clean), ("reconstructed", restore_outliers(clean, seed=42))):
    d = lagged(base); tr, te = split(d)
    adr = te.snr_max_prev20.values - te.snr.values
    rows.append(dict(data=dname, scheme="conventional ADR, chronological prev-20", LM95=lm_for(adr, 95), LM99=lm_for(adr, 99)))
    for feat in ("snr", "snr_prev", "snr_prev20"):
        Xtr, Xte = make_X(tr).copy(), make_X(te).copy(); Xtr[:, 6] = tr[feat].values; Xte[:, 6] = te[feat].values
        ytr, yte = make_y(tr), make_y(te)
        mlr = MLRCpls().fit(Xtr, ytr); r = yte - mlr.predict(Xte)
        rows.append(dict(data=dname, scheme=f"MLR with {feat}", rmse=float(np.sqrt(np.mean(r**2))), LM95=lm_for(r, 95), LM99=lm_for(r, 99)))
        if feat in ("snr", "snr_prev"):
            t0 = time.time(); ann = make_ann(**spec.ANN_BEST).fit(Xtr, ytr); r = yte - ann.predict(Xte)
            rows.append(dict(data=dname, scheme=f"ANN with {feat}", rmse=float(np.sqrt(np.mean(r**2))), LM95=lm_for(r, 95), LM99=lm_for(r, 99)))
            print(f"  {dname} ANN {feat}: rmse {np.sqrt(np.mean(r**2)):.3f} LM99 {lm_for(r, 99):.2f} ({time.time()-t0:.0f}s)", flush=True)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(pathlib.Path(__file__).with_name("causal_features.csv"), index=False); print("CAUSAL DONE")
