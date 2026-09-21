"""Is Algorithm 1 stateful?  Test of assumption A13.

Hypothesis: `current_tp` / `current_sf` are the end node's *current* settings - the previous
packet's outputs - rather than 20 dBm and the logged SF for every packet (the final run).
Algebra says the TP set-point  PL_hat + noise + SNR_limit(sf) + LM - link_budget  does not depend
on the starting TP, so state can only act through the branch choice (a node below 20 dBm sees a
negative margin and takes the SF-raise branch).  `inc_order="tp_first"` is the cost-aware
alternative: spend TP headroom before raising SF in the increase branch.

Run from the project root:  python outputs/provenance/adr_mechanism/adr_stateful.py
Writes stateful_fit.csv next to this file.  `carry=()` must reproduce simulate_enhanced exactly.
"""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from adr_algorithm import AdrParameters, snr_limit_of
from conventional_models import SPLMSF, SPLMSFT, friis_pl
from data_loading import load_cached
from energy import PowerModel, improvement_pct, time_on_air
from reconstruction import restore_outliers
from simulation import SimConfig, simulate_conventional, simulate_enhanced
from splitting import split
import paper_spec as spec


def simulate_enhanced_stateful(df, cpls_pred, cfg, carry=("tp", "sf"), inc_order="sf_first"):
    """Corrected Algorithm 1 run per device in `df` order, vectorised over the LM grid."""
    p = cfg.params; LMS = np.asarray(cfg.lm_values, float); nL = len(LMS)
    pred = np.asarray(cpls_pred, float); dev = df["device_id"].to_numpy()
    npow = df[cfg.noise_column].to_numpy(float); sf_log = df["sf"].to_numpy(float)
    lbv = (-df["ltx"] + df["gtx"] + df["grx"] - df["lrx"]).to_numpy(float)
    TP = np.empty((len(df), nL)); SF = np.empty((len(df), nL), int)
    for d in pd.unique(dev):
        idx = np.flatnonzero(dev == d)
        tp = np.full(nL, p.max_tp); sf = np.full(nL, sf_log[idx[0]])
        for i in idx:
            if "tp" not in carry: tp = np.full(nL, p.max_tp)
            if "sf" not in carry: sf = np.full(nL, sf_log[i])
            rssi = tp + lbv[i] - pred[i]                                        # line 4
            me = rssi - (npow[i] + snr_limit_of(sf) + LMS)                      # line 5
            inc = me < 0
            if inc.any():                                                       # lines 6-17
                if inc_order == "tp_first":                                     # cost-aware: TP headroom first
                    tp_new = np.minimum(tp - me, p.max_tp)
                    rssi = np.where(inc, rssi + (tp_new - tp), rssi); tp = np.where(inc, tp_new, tp)
                    me = rssi - (npow[i] + snr_limit_of(sf) + LMS)
                for _ in range(p.max_sf - p.min_sf + 1):
                    step = inc & (me < 0) & (sf < p.max_sf)
                    if not step.any(): break
                    sf = np.where(step, sf + 1, sf); me = np.where(step, rssi - (npow[i] + snr_limit_of(sf) + LMS), me)
                tp = np.where(inc, np.minimum(tp - me, p.max_tp), tp)
            dec = me >= 0
            if dec.any():                                                       # lines 18-31
                broke = np.zeros(nL, bool)
                for _ in range(p.max_sf - p.min_sf + 1):
                    step = dec & ~broke & (me > 0) & (sf > p.min_sf)
                    if not step.any(): break
                    sf_try = np.where(step, sf - 1, sf); me_try = rssi - (npow[i] + snr_limit_of(sf_try) + LMS)
                    hit = step & (me_try <= 0)
                    sf = np.where(step & ~hit, sf_try, sf); me = np.where(step, me_try, me); broke |= hit
                sp = tp - me
                tp = np.where(dec, np.clip(sp, p.min_tp, p.max_tp) if cfg.clamp_tp else np.maximum(sp, p.min_tp), tp)
            TP[i] = tp; SF[i] = sf
    pm = PowerModel(); pl_true = df["experimental_pl"].to_numpy(float); rows = []
    for j, lm in enumerate(LMS):
        tp, sf = TP[:, j], SF[:, j]; ok = (pl_true - pred) < lm
        toa = time_on_air(sf, cfg.payload_bytes); e = pm.energy_j(tp, toa)
        rows.append(dict(LM=float(lm), pdr=float(ok.mean() * 100), mean_tp_dbm=float(tp.mean()), mean_sf=float(sf.mean()),
                         mean_toa_s=float(toa.mean()), mean_energy_j=float(e.mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).resolve().parent; F = ROOT / "outputs/final"; D = HERE.parent / "paper_digitized"
    T11 = pd.read_csv(D / "paper_fig11_digitized.csv", index_col=0).clip(upper=100)
    T12 = pd.read_csv(D / "paper_fig12_digitized.csv", index_col=0); T13 = pd.read_csv(D / "paper_fig13_digitized.csv", index_col=0)
    NAME = {"ANN": "ANN", "SVR": "SVR", "RF": "RF", "MLR": "MLR", "FRIIS": "Friis", "SPLMSF": "SPLMSF", "SPLMSFT": "SPLMSFT"}
    raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True)
    tr, te = split(restore_outliers(clean, seed=42))
    z = np.load(F / "test_predictions.npz"); preds = {k: z[k] for k in z.files}
    rng = np.random.default_rng(42); y_tr = tr.experimental_pl.values
    sp = SPLMSF().fit(tr.distance, y_tr); spt = SPLMSFT(nu=spec.APPENDIX["t_dof_nu"]).fit(tr.distance, y_tr)
    preds["Friis"] = np.asarray(friis_pl(te.distance, te.frequency), float)
    preds["SPLMSF"] = sp.predict(te.distance) + sp.sample_shadowing(len(te), rng)
    preds["SPLMSFT"] = spt.predict(te.distance) + spt.sample_shadowing(len(te), rng)
    LMS = tuple(np.round(np.arange(0, 15.001, 0.25), 2)); LEVELS = (80, 85, 90, 95, 99)
    ce = SimConfig(lm_values=LMS, clamp_tp=True, variant="corrected", rule="residual", params=AdrParameters(max_sf=12))
    ca = SimConfig(lm_values=LMS, rule="residual", adr_tp_step=3.0, adr_sf_mode="both", params=AdrParameters(max_sf=12))
    adr = simulate_conventional(te, ca)
    a = simulate_enhanced(te, preds["ANN"], ce); b = simulate_enhanced_stateful(te, preds["ANN"], ce, carry=())
    assert np.abs(a.pdr - b.pdr).max() == 0 and np.abs(a.mean_toa_s - b.mean_toa_s).max() == 0, "carry=() must equal the stateless simulator"
    print("self-check: carry=() reproduces simulate_enhanced exactly", flush=True)

    def at(cur, k, pdr): h = cur[k][cur[k].pdr >= pdr]; return None if h.empty else h.iloc[0]
    out = []
    VARIANTS = [("stateless (final run)", (), "sf_first"), ("carry TP+SF", ("tp", "sf"), "sf_first"),
                ("carry SF", ("sf",), "sf_first"), ("carry TP", ("tp",), "sf_first"),
                ("carry TP+SF, TP before SF", ("tp", "sf"), "tp_first"), ("carry TP, TP before SF", ("tp",), "tp_first")]
    for label, carry, order in VARIANTS:
        t0 = time.time()
        cur = {k: (simulate_enhanced(te, preds[k], ce) if not carry and order == "sf_first"
                   else simulate_enhanced_stateful(te, preds[k], ce, carry, order)) for k in preds}
        e_err, t_err = [], []
        for pdr in LEVELS:
            ra = adr[adr.pdr >= pdr].iloc[0]
            for pk, ok in NAME.items():
                h = at(cur, ok, pdr)
                if h is None: continue
                e, t = improvement_pct(ra.mean_energy_j, h.mean_energy_j), improvement_pct(ra.mean_toa_s, h.mean_toa_s)
                e_err.append(e - T13.loc[pdr, pk]); t_err.append(t - T12.loc[pdr, pk])
                out.append(dict(variant=label, scheme=ok, pdr=pdr, LM=h.LM, energy_pct=e, toa_pct=t,
                                paper_energy_pct=T13.loc[pdr, pk], paper_toa_pct=T12.loc[pdr, pk], mean_sf=h.mean_sf, mean_tp_dbm=h.mean_tp_dbm))
        p11 = np.concatenate([np.array([cur[ok].loc[np.isclose(cur[ok].LM, l), "pdr"].iloc[0] for l in range(12)]) - T11[pk].values[:12]
                              for pk, ok in NAME.items() if pk != "FRIIS"])
        fe, ft, f11 = np.sqrt(np.mean(np.square(e_err))), np.sqrt(np.mean(np.square(t_err))), np.sqrt(np.nanmean(p11 ** 2))
        ann = [f"{improvement_pct(adr[adr.pdr >= q].iloc[0].mean_toa_s, at(cur, 'ANN', q).mean_toa_s):+.0f}" for q in LEVELS]
        print(f"{label:28s} energy RMSE {fe:5.1f}  ToA RMSE {ft:5.1f}  Fig11 RMSE {f11:4.1f}   ANN ToA % at 80/85/90/95/99: {' '.join(ann)}"
              f"   (paper {' '.join(f'{v:+.0f}' for v in T12['ANN'])})   {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(HERE / "stateful_fit.csv", index=False); print("written stateful_fit.csv")
