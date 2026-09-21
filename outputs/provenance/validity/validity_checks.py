"""Validity checks prompted by the independent review of 21 Sep 2026 (review_2026-09-21/REVIEW.md).

  A. delivery feasibility: the residual rule (final) is blind to the selected SF/TP; the receiver-threshold
     rule is applied to the same selections under SF <= 12 and SF <= 10
  B. released vs reconstructed data: MLR RMSE, residual excess kurtosis, fitted Student-t nu
  C. conventional-ADR window: shuffled test order (final) / chronological test history / the 20 packets
     preceding each test packet in the full campaign - released and reconstructed data
  D. leave-one-node-out MLR on the released data (four fixed links)
  E. Fig. 11 discrepancy per scheme, RMSE and maximum
  F. achieved PDR at the paper's integer operating points
Run from the project root after run_final.py (~2 min):  python outputs/provenance/validity/validity_checks.py
Writes validity_summary.csv (named scalars, read by make_final_report.py) and validity_checks.txt here.
The causal-feature experiment is causal_features.py (~8 min).
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from scipy import stats
from adr_algorithm import snr_limit_of
from cpls_models import MLRCpls, make_X, make_y
from data_loading import load_cached
from reconstruction import restore_outliers
from splitting import split
import run_final as rf

HERE = pathlib.Path(__file__).resolve().parent; F = ROOT / "outputs/final"; out = {}; lines = []
def say(s): print(s, flush=True); lines.append(s)
raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True)
recon = restore_outliers(clean, frac_rssi_only=rf.CONFIG["restore_frac_rssi_only"], frac_coupled=rf.CONFIG["restore_frac_coupled"],
                         outlier_db=tuple(rf.CONFIG["outlier_db"]), seed=rf.SEED)
tr_c, te_c = split(clean); tr, te = split(recon)

# ---- A
z = np.load(F / "test_predictions.npz"); npow = te.noise_power_rssi.to_numpy(float); rssi = te.rssi.to_numpy(float)
lb = (-te.ltx + te.gtx + te.grx - te.lrx).to_numpy(float); pl_true = te.experimental_pl.to_numpy(float); sf_log = te.sf.to_numpy(float)
def en_sel(pred, lm, sf_max):
    SFS = np.arange(7, sf_max + 1); me20 = (20 + lb - pred)[:, None] - (npow[:, None] + snr_limit_of(SFS)[None, :] + lm); feas = me20 >= 0
    idx = np.where(feas.any(1), feas.argmax(1), len(SFS) - 1); sf = SFS[idx]; me = me20[np.arange(len(sf)), idx]
    return np.clip(20 - np.maximum(me, 0), 2, 20), sf
def thr(tp, sf): return float((rssi + (tp - 20) > npow + snr_limit_of(sf)).mean() * 100)
say("A. PDR at the paper's 99 % operating LM (ML models: LM 4): residual rule vs receiver-threshold rule, same selections")
for sf_max in (12, 10):
    for k in ("ANN", "SVR", "RF"):
        tp, sf = en_sel(z[k], 4.0, sf_max); res = float(((pl_true - z[k]) < 4.0).mean() * 100); t = thr(tp, sf)
        out[f"pdr99_residual_{k}"] = res; out[f"pdr99_threshold_sf{sf_max}_{k}"] = t
        say(f"   SF<={sf_max} {k}: residual {res:.2f} %  threshold {t:.2f} %  mean SF {sf.mean():.2f}")
out["threshold_ceiling_sf10"] = thr(np.full(len(te), 20.0), np.full(len(te), 10)); out["threshold_ceiling_sf12"] = thr(np.full(len(te), 20.0), np.full(len(te), 12))
out["threshold_at_logged_sf"] = thr(np.full(len(te), 20.0), sf_log)
out["share_received_below_limit_released"] = float((te_c.snr.values < snr_limit_of(te_c.sf.values)).mean() * 100)
say(f"   everyone at 20 dBm: SF 10 -> {out['threshold_ceiling_sf10']:.2f} %, SF 12 -> {out['threshold_ceiling_sf12']:.2f} %, logged SF -> {out['threshold_at_logged_sf']:.2f} %")
say(f"   received test packets whose logged SNR is below Table 2's limit for their own SF: {out['share_received_below_limit_released']:.2f} % (released)")
# ---- B
for name, (a, b) in (("released", (tr_c, te_c)), ("reconstructed", (tr, te))):
    est = MLRCpls().fit(make_X(a), make_y(a)); r = make_y(b) - est.predict(make_X(b))
    sub = r[np.random.default_rng(0).choice(len(r), 100_000, replace=False)]; nu = float(stats.t.fit(sub)[0])
    out[f"mlr_rmse_{name}"] = float(np.sqrt(np.mean(r ** 2))); out[f"mlr_kurtosis_{name}"] = float(stats.kurtosis(r)); out[f"mlr_t_nu_{name}"] = nu
    say(f"B. MLR {name:13s}: test RMSE {out[f'mlr_rmse_{name}']:.4f}  excess kurtosis {out[f'mlr_kurtosis_{name}']:+.3f}  fitted t nu {nu:.4g}")
# ---- C
def prev20(d, window_order):
    if window_order == "chronological": d = d.sort_values(["device_id", "timestamp"], kind="mergesort")
    m = d.groupby("device_id", observed=True).snr.transform(lambda x: x.rolling(20, min_periods=1).max().shift(1)).fillna(d.snr)
    return pd.Series((m - d.snr).values, index=d["index"].values)
say("C. conventional ADR: LM needed for 99 / 95 % (percentiles of SNRmax - SNR)")
for dname, (full, tst) in (("released", (clean, te_c)), ("reconstructed", (recon, te))):
    for wname, mg in (("shuffled test order (final)", prev20(tst.reset_index(drop=True), "test").values),
                      ("chronological test history", prev20(tst, "chronological").values),
                      ("full campaign, preceding 20", prev20(full, "chronological").loc[tst["index"].values].values)):
        key = f"adr_lm99_{dname}_{wname.split(' ')[0]}"; out[key] = float(np.percentile(mg, 99)); out[key.replace('99', '95')] = float(np.percentile(mg, 95))
        say(f"   {dname:13s} {wname:30s}: 99 % at {out[key]:.2f} dB, 95 % at {out[key.replace('99','95')]:.2f} dB")
# ---- D
say("D. leave-one-node-out MLR (released data):")
for node in sorted(clean.device_id.unique()):
    a, b = clean[clean.device_id != node], clean[clean.device_id == node]
    est = MLRCpls().fit(make_X(a), make_y(a)); r = make_y(b) - est.predict(make_X(b)); out[f"lono_rmse_{node}"] = float(np.sqrt(np.mean(r ** 2))); out[f"lono_bias_{node}"] = float(r.mean())
    say(f"   hold out {node}: RMSE {out[f'lono_rmse_{node}']:.2f} dB, bias {out[f'lono_bias_{node}']:+.2f} dB")
# ---- E
cm = pd.read_csv(F / "figs_vs_paper_digitized.csv"); f11 = cm[cm.figure == "fig11"]
say("E. Fig. 11 |ours - paper| per scheme: RMSE / max (LM):")
for s in f11.scheme.unique():
    g = f11[f11.scheme == s]; i = g["diff"].abs().idxmax(); out[f"fig11_rmse_{s}"] = float(np.sqrt(np.mean(g["diff"] ** 2))); out[f"fig11_max_{s}"] = float(abs(g.loc[i, "diff"])); out[f"fig11_max_lm_{s}"] = int(g.loc[i, "LM"])
    say(f"   {s:8s} rmse {out[f'fig11_rmse_{s}']:.2f}  max {out[f'fig11_max_{s}']:.2f} at LM {out[f'fig11_max_lm_{s}']}")
# ---- F
E = pd.read_csv(F / "fig12_13_energy_toa.csv"); say("F. achieved PDR at the paper-integer operating points:")
for p in (80, 85, 90, 95, 99):
    for k in ("ANN", "ADR"):
        r = E[(E.scheme == k) & (E.pdr == p)].iloc[0]; out[f"achieved_pdr_{k}_{p}"] = float(r.pdr_at_LM); out[f"op_lm_{k}_{p}"] = float(r.LM_dB)
    say(f"   label {p} %: ANN at LM {out[f'op_lm_ANN_{p}']:.0f} achieves {out[f'achieved_pdr_ANN_{p}']:.2f} %; ADR at LM {out[f'op_lm_ADR_{p}']:.0f} achieves {out[f'achieved_pdr_ADR_{p}']:.2f} %")
pd.Series(out, name="value").rename_axis("quantity").reset_index().to_csv(HERE / "validity_summary.csv", index=False)
(HERE / "validity_checks.txt").write_text("\n".join(lines) + "\n"); say("VALIDITY CHECKS DONE")
