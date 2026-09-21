"""The released CSV against its own data descriptor (Data 8(1):4, 2023) and against the IoT-J paper.

Descriptor: 930,753 rows; outliers = Mahalanobis distance over 11 variables, removed if Md > 29.59
(chi2, 10 dof, p = 0.001); its MLR on the file: test RMSE 1.840.  IoT-J: training set 792,600 rows
=> 990,750 rows AFTER its outlier step; MLR test RMSE 1.951.
Run from the project root:  python outputs/provenance/released_file_check.py   (writes released_file_check.csv)
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from data_loading import load_cached
from splitting import split
from cpls_models import MLRCpls, make_X, make_y
import paper_spec as spec

raw = load_cached()
cols = ["distance", "frequency", "sf", "frame_length", "temperature", "rh", "bp", "pm2_5", "toa", "energy", "experimental_pl"]
X = raw[cols].to_numpy(float); D = X - X.mean(0); md2 = np.einsum("ij,jk,ik->i", D, np.linalg.inv(np.cov(X, rowvar=False)), D)
keep = md2 <= 29.59

def mlr_rmse(df):
    tr, te = split(df.reset_index(drop=True)); est = MLRCpls().fit(make_X(tr), make_y(tr))
    return float(np.sqrt(np.mean((make_y(te) - est.predict(make_X(te))) ** 2)))

out = dict(released_rows=len(raw), iotj_rows_after_outlier_step=int(round(792_600 / 0.8)),
           mahalanobis_removed=int((~keep).sum()), mahalanobis_removed_pct=float((~keep).mean() * 100),
           mlr_rmse_released=mlr_rmse(raw), mlr_rmse_after_mahalanobis=mlr_rmse(raw[keep]),
           mlr_rmse_after_dht22_drop=mlr_rmse(raw[~(raw.rh <= 2)]),
           descriptor_mlr_rmse=spec.DESCRIPTOR["mlr"]["test_rmse"], iotj_mlr_rmse=1.951,
           sf_11_12_rows=int((raw.sf >= 11).sum()))
pd.DataFrame([out]).to_csv(pathlib.Path(__file__).with_suffix(".csv"), index=False)
print(pd.Series(out).to_string())
