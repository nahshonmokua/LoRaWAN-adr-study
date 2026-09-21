"""What carries the MLR's gain over the distance-only model?  Feature ablation of eq. (6) (frequency
term fixed at 20 log10 f, OLS), reconstructed data, test set.  R2 = squared Pearson correlation (the
paper's definition).  `node` = one-hot end-node indicator, to test whether barometric pressure acts as
a node identifier (EN3 sits 17-23 hPa below the other three; within-node sd 1.6 hPa).
Run from the project root:  python outputs/provenance/mlr_feature_ablation.py  (writes the .csv next to it)
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from data_loading import load_cached
from reconstruction import restore_outliers
from splitting import split

raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True); tr, te = split(restore_outliers(clean, seed=42))
NODES = sorted(tr.device_id.unique())

def design(df, feats):
    cols = {"dist": 10 * np.log10(df.distance.values), "T": df.temperature.values, "RH": df.rh.values,
            "BP": df.bp.values, "PM": df.pm2_5.values, "SNR": df.snr.values}
    X = [np.ones(len(df))] + [cols[f] for f in feats if f != "node"]
    if "node" in feats: X += [(df.device_id.values == n).astype(float) for n in NODES[1:]]
    return np.column_stack(X)

def fit_eval(feats):
    ytr = tr.experimental_pl.values - 20 * np.log10(tr.frequency.values); yte = te.experimental_pl.values - 20 * np.log10(te.frequency.values)
    b, *_ = np.linalg.lstsq(design(tr, feats), ytr, rcond=None); p = design(te, feats) @ b; r = yte - p
    return float(np.sqrt(np.mean(r ** 2))), float(np.corrcoef(p, yte)[0, 1] ** 2)

SETS = [("distance only (SPLMSF form)", ["dist"]), ("distance + T, RH, PM", ["dist", "T", "RH", "PM"]),
        ("distance + BP", ["dist", "BP"]), ("distance + SNR", ["dist", "SNR"]), ("distance + node", ["dist", "node"]),
        ("distance + node + SNR", ["dist", "node", "SNR"]), ("distance + BP + SNR", ["dist", "BP", "SNR"]),
        ("all but BP", ["dist", "T", "RH", "PM", "SNR"]), ("all but SNR", ["dist", "T", "RH", "PM", "BP"]),
        ("all (eq. 6)", ["dist", "T", "RH", "PM", "BP", "SNR"]), ("all + node", ["dist", "T", "RH", "PM", "BP", "SNR", "node"]),
        ("distance + node + SNR + T, RH, PM", ["dist", "node", "SNR", "T", "RH", "PM"])]
rows = [dict(features=n, test_rmse=fit_eval(f)[0], test_r2=fit_eval(f)[1]) for n, f in SETS]
D = pd.DataFrame(rows); D.to_csv(pathlib.Path(__file__).with_suffix(".csv"), index=False); print(D.round(4).to_string(index=False))
bp = te.groupby("device_id").bp.agg(["mean", "std"]).round(1); print("\nBP by node (hPa):\n" + bp.to_string())
