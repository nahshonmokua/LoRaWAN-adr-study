"""Is "ANN best" (Figs. 12/13) a property of the model class?  Retrain the ANN with other seeds and
score every model at the SAME operating LMs (the paper's ANN LMs 1/2/2/3/4 vs ADR 6/7/8/9/11), final
configuration.  Run from the project root after run_final.py; writes ann_seed_sensitivity.csv here.
"""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from adr_algorithm import AdrParameters, snr_limit_of
from cpls_models import make_ann, make_X, make_y
from data_loading import load_cached
from energy import PowerModel, improvement_pct, time_on_air
from reconstruction import restore_outliers
from simulation import SimConfig, simulate_conventional
from splitting import split
import paper_spec as spec
F = str(ROOT / "outputs/final") + "/"
raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True); tr, te = split(restore_outliers(clean, seed=42))
Xtr, ytr, Xte, yte = make_X(tr), make_y(tr), make_X(te), make_y(te)
lb = (-te.ltx + te.gtx + te.grx - te.lrx).to_numpy(float); npow = te.noise_power_rssi.to_numpy(float); pm = PowerModel(); SFS = np.arange(7, 13)
LMI = tuple(range(0, 16)); LEVELS = (80, 85, 90, 95, 99)
adr = simulate_conventional(te, SimConfig(lm_values=LMI, rule="residual", adr_tp_step=None, adr_sf_mode="both", adr_window_excl_current=True, params=AdrParameters(max_sf=12)))
def en_point(pred, lm):
    me20 = (20 + lb - pred)[:, None] - (npow[:, None] + snr_limit_of(SFS)[None, :] + lm); feas = me20 >= 0
    sf = SFS[np.where(feas.any(1), feas.argmax(1), 5)]; me_s = me20[np.arange(len(sf)), np.where(feas.any(1), feas.argmax(1), 5)]
    tp = np.clip(20 - np.maximum(me_s, 0), 2, 20); toa = time_on_air(sf, 1); e = pm.energy_j(tp, toa)
    return e.mean(), toa.mean(), ((yte - pred) < lm).mean() * 100, (sf >= 11).mean() * 100
def score(pred, name):
    rows = []
    for i, lvl in enumerate(LEVELS):
        lm, lm_adr = spec.PDR_TARGETS_INTEGER["ANN"][lvl], spec.PDR_TARGETS_INTEGER["ADR"][lvl]
        ra = adr[np.isclose(adr.LM, lm_adr)].iloc[0]; e, t, pdr, s11 = en_point(pred, lm)
        rows.append(dict(model=name, pdr_level=lvl, LM=lm, energy=improvement_pct(ra.mean_energy_j, e), toa=improvement_pct(ra.mean_toa_s, t), pdr_at_LM=pdr, pct_sf11=s11))
    return pd.DataFrame(rows)
z = np.load(F + "test_predictions.npz"); out = [score(z["ANN"], "ANN seed 42 (final)"), score(z["SVR"], "SVR (final)"), score(z["RF"], "RF (final)")]
for seed in (1, 7, 2024):
    t0 = time.time(); est = make_ann(**spec.ANN_BEST)
    est.set_params(**{k: seed for k in est.get_params() if k.endswith("random_state")}); est.fit(Xtr, ytr); pred = est.predict(Xte)
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2))); bias = float(np.mean(pred - yte)); en3 = te.device_id.to_numpy() == "EN3"; bias3 = float(np.mean((pred - yte)[en3]))
    out.append(score(pred, f"ANN seed {seed} (rmse {rmse:.3f}, bias {bias:+.3f}, EN3 bias {bias3:+.3f})")); print(f"seed {seed}: rmse {rmse:.4f} bias {bias:+.3f} EN3 bias {bias3:+.3f} ({time.time()-t0:.0f}s)", flush=True)
D = pd.concat(out, ignore_index=True)
print("\nenergy improvement at 80/85/90/95/99 (paper ANN: 4/15/17/20/43.5; SVR -1/10/14/16/40.6; RF -4/7/11/11/38.7)")
for m, g in D.groupby("model", sort=False): print(f"  {m:62s} E {np.round(g.energy).astype(int).tolist()}   ToA {np.round(g.toa).astype(int).tolist()}   %SF>=11 at LM4 {g.pct_sf11.iloc[-1]:.1f}")
z2 = np.load(F + "test_predictions.npz"); en3 = te.device_id.to_numpy() == "EN3"
for k in ("ANN", "SVR", "RF"): print(f"  final {k}: bias {np.mean(z2[k]-yte):+.3f} dB, EN3 bias {np.mean((z2[k]-yte)[en3]):+.3f} dB, EN3 rmse {np.sqrt(np.mean((z2[k]-yte)[en3]**2)):.3f}")
D.to_csv(pathlib.Path(__file__).with_name("ann_seed_sensitivity.csv"), index=False); print("SEEDS DONE")
