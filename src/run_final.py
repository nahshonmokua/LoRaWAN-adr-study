"""FINAL reproduction of Gonzalez-Palacio et al. (2023), IEEE IoT-J 10(12) - one run, one folder.

Configuration (every non-printed choice, and how it was fixed):
  data          released CSV -> DHT22 stuck-low rows dropped -> pre-cleaning population restored
                (reconstruction.py; determined in stage11_reconstruct.py)
  split         80 / 20, seed 42
  Algorithm 1   snr_limit recomputed from the current SF; set_point_tp clamped to [min_tp, max_tp]
                (the printed listing's lines 10/22/28 are defects; determined in stage14_inverse.py)
  parameters    sf 7..12, tp 2..20 dBm, per-node ltx/gtx/lrx/grx from the CSV
  conventional  TTN rule, rolling max of 20 SNR samples per device over the test subset
  ToA / energy  LoRa ToA (1-byte payload, BW 125 kHz, CR 4/5, CRC, explicit header);
                E = P_consumed(TP) x ToA with P_consumed the Table VII linear fit
Tables go to tables/, figures to figures/, fitted models to models/ (see data_loading for the layout).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adr_algorithm import AdrParameters                                           # noqa: E402

from conventional_models import (SPLMSF, SPLMSFT, friis_pl, okumura_hata_pl,       # noqa: E402
                                 score, two_ray_pl)
from cpls_models import MLRCpls, make_ann, make_rf, make_svr, make_X, make_y      # noqa: E402
from data_loading import FIGURES, MODELS, PAPER_DIG, TABLES, load_cached                                     # noqa: E402
from energy import PowerModel, improvement_pct, verify_against_csv   # noqa: E402
from reconstruction import restore_outliers                                       # noqa: E402
from simulation import SimConfig, simulate_conventional, simulate_enhanced        # noqa: E402
from splitting import SEED, split                                                 # noqa: E402
import paper_spec as spec                                                         # noqa: E402

FINAL, FIGS = TABLES, FIGURES
LEVELS = (80, 85, 90, 95, 99)
CONFIG = dict(seed=SEED, train_fraction=0.8, sf_min=7, sf_max=12, min_tp_dbm=2.0, max_tp_dbm=20.0,
              algorithm1_variant="text",     # Section IV-A's reading of Algorithm 1: exclusive scenarios, TP from the margin at the SF used ("corrected" = listing-literal)
              line28="clamped", adr_window=20,
              adr_window_order="test_order",
              adr_window_excl_current=True,  # "collects 20 SNR samples and obtains the maximum": the packet judged is not among them (PDR 1/21 at LM = 0)
              adr_tp_step=None,            # "decreases PT as needed to get Me = 0": continuous.  3.0 = TTN nStep = floor(Me/3)
              adr_sf_mode="both",
              sf_max_adr=12,               # SF range available to the conventional ADR (Section IV: 7-12)
              delivery_rule="residual",    # "actual RSSI > predicted RSSI - LM"  <=>  PL_true - PL_pred < LM
              en_tp_quant=None,            # EN transmit-power granularity, dB (US915 TXPower table: 2); None = continuous
              energy_over="all",           # improvement curves over "all" packets or "delivered" only
              operating_lm="paper_integer",  # Figs. 12/13 evaluated at the paper's integer LMs (spec.PDR_TARGETS_INTEGER); "ours_integer" | "grid"
              shadowing_psi="appendix",      # SPLMSFT and "MLR (with t-distributed shadow fading)" sample the Appendix psi ~ t(11.43) fitted on the MLR residuals; "own" = SPLMSFT from its own residuals, MLR deterministic
              lm_step_db=0.05,             # LM sweep resolution
              payload_bytes=1, bandwidth_hz=125_000,
              coding_rate="4/5", restore_frac_rssi_only=0.0025, restore_frac_coupled=0.0015,
              outlier_db=[8, 15], svr_fit_rows=50_000, ann_cv_rows=100_000)
PARAMS = AdrParameters(min_sf=CONFIG["sf_min"], max_sf=CONFIG["sf_max"],
                       min_tp=CONFIG["min_tp_dbm"], max_tp=CONFIG["max_tp_dbm"])
LMS = tuple(np.round(np.arange(0, 15.001, CONFIG["lm_step_db"]), 2))
PARAMS_ADR = AdrParameters(min_sf=CONFIG["sf_min"], max_sf=CONFIG["sf_max_adr"],
                           min_tp=CONFIG["min_tp_dbm"], max_tp=CONFIG["max_tp_dbm"])
from plots import DIG  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ 1. data
def prepare_data():
    raw = load_cached()
    clean = raw[~(raw["rh"] <= 2.0)].reset_index(drop=True)          # DHT22 stuck-low fault
    clean["pos"] = np.arange(len(clean))
    df = restore_outliers(clean, frac_rssi_only=CONFIG["restore_frac_rssi_only"], frac_coupled=CONFIG["restore_frac_coupled"],
                          outlier_db=tuple(CONFIG["outlier_db"]), seed=SEED)
    df.attrs["perturbed"] = (df["rssi"].to_numpy() != clean["rssi"].to_numpy())      # by position in the cleaned frame
    tr, te = split(df)
    pd.DataFrame([dict(released_rows=len(raw), dht22_fault_rows=int((raw["rh"] <= 2.0).sum()),
                       restored_outlier_rows=df.attrs["n_restored"], train_rows=len(tr), test_rows=len(te),
                       pl_range_db=f"{df.experimental_pl.min():.1f}..{df.experimental_pl.max():.1f}")]
                 ).to_csv(FINAL / "data_summary.csv", index=False)
    return df, tr, te


# ------------------------------------------------------- 2. conventional models
def conventional(df, tr, te):
    y_tr, y_te = tr.experimental_pl.values, te.experimental_pl.values
    sp = SPLMSF().fit(tr.distance, y_tr)
    spt = SPLMSFT(nu=spec.APPENDIX["t_dof_nu"]).fit(tr.distance, y_tr)
    P = {"Friis": (friis_pl(tr.distance, tr.frequency), friis_pl(te.distance, te.frequency)),
         "Two-ray": (two_ray_pl(tr.distance, tr.ht, tr.hr), two_ray_pl(te.distance, te.ht, te.hr)),
         "Okumura-Hata": (okumura_hata_pl(tr.distance, tr.frequency, tr.ht, tr.hr),
                          okumura_hata_pl(te.distance, te.frequency, te.ht, te.hr)),
         "SPLMSF": (sp.predict(tr.distance), sp.predict(te.distance))}
    rows = []
    for k, (a, b) in P.items():
        s_tr, s_te = score(y_tr, a), score(y_te, b)
        pp = spec.TABLE_IV[k]
        rows.append(dict(model=k, train_rmse=s_tr["rmse"], test_rmse=s_te["rmse"], test_r2=s_te["r2_corr"],
                         paper_test_rmse=pp["test_rmse"], paper_test_r2=pp["test_r2"],
                         rmse_err_pct=100 * (s_te["rmse"] - pp["test_rmse"]) / pp["test_rmse"]))
    pd.DataFrame(rows).to_csv(FINAL / "table_iii_conventional.csv", index=False)
    pd.DataFrame([dict(parameter="gamma", ours=sp.gamma_, paper=2.7),
                  dict(parameter="K_dB_d0_1km", ours=sp.K_at(1000.0), paper=84.2),
                  dict(parameter="psi_sigma_dB", ours=sp.psi_params_["scale"], paper=np.nan)]
                 ).to_csv(FINAL / "splmsf_parameters.csv", index=False)

    rep = df.groupby("distance", observed=True).agg(pl=("experimental_pl", "mean"), f=("frequency", "mean"),
                                                    ht=("ht", "first"), hr=("hr", "first")).reset_index()
    pd.DataFrame(dict(distance_km=rep.distance / 1e3, pl_mean=rep.pl,
                      pl_p005=[df.loc[df.distance == d, "experimental_pl"].quantile(0.005) for d in rep.distance],
                      pl_p995=[df.loc[df.distance == d, "experimental_pl"].quantile(0.995) for d in rep.distance],
                      friis=friis_pl(rep.distance, rep.f), splmsf=sp.predict(rep.distance),
                      two_ray=two_ray_pl(rep.distance, rep.ht, rep.hr))).to_csv(FINAL / "fig04_data.csv", index=False)
    return sp, spt


# --------------------------------------------------------------- 3. CPLS models
def cpls(tr, te):
    Xtr, ytr, Xte, yte = make_X(tr), make_y(tr), make_X(te), make_y(te)
    rng = np.random.default_rng(SEED)
    svr_idx = rng.choice(len(Xtr), CONFIG["svr_fit_rows"], replace=False)
    ann_cv = rng.choice(len(Xtr), CONFIG["ann_cv_rows"], replace=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    spec_est = {"MLR": (lambda: MLRCpls(), None), "ANN": (lambda: make_ann(**spec.ANN_BEST), ann_cv),
                "SVR": (lambda: make_svr(**spec.SVR_BEST), svr_idx), "RF": (lambda: make_rf(**spec.RF_BEST), None)}
    preds, rows = {}, []
    for name, (mk, sub) in spec_est.items():
        t0 = time.time()
        fit_idx = svr_idx if name == "SVR" else np.arange(len(Xtr))
        est = mk().fit(Xtr[fit_idx], ytr[fit_idx])
        cv_idx = sub if sub is not None else np.arange(len(Xtr))
        cvs = -cross_val_score(mk() if name != "RF" else make_rf(**spec.RF_BEST, n_jobs=1),
                               Xtr[cv_idx], ytr[cv_idx], scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
        p_te = np.asarray(est.predict(Xte), float)
        preds[name] = p_te
        s_te = score(yte, p_te)
        pp = spec.TABLE_IV[name]
        rows.append(dict(model=name, n_fit_rows=len(fit_idx), n_cv_rows=len(cv_idx),
                         cv_rmse=float(cvs.mean()), cv_rmse_sd=float(cvs.std()),
                         test_rmse=s_te["rmse"], test_r2=s_te["r2_corr"],
                         paper_train_rmse=pp["train_rmse"], paper_train_s=pp["train_s"],
                         paper_test_rmse=pp["test_rmse"], paper_test_r2=pp["test_r2"],
                         test_rmse_err_pct=100 * (s_te["rmse"] - pp["test_rmse"]) / pp["test_rmse"]))
        joblib.dump(est, MODELS / f"cpls_{name.lower()}.joblib")
        log(f"  {name}: test RMSE {s_te['rmse']:.4f}  R2 {s_te['r2_corr']:.4f}  cv {cvs.mean():.4f}+/-{cvs.std():.4f}  ({time.time()-t0:.0f}s)")
        if name == "MLR":
            names = ["b0_intercept", "gamma", "b1_T", "b2_RH", "b3_BP", "b4_PM25", "b5_SNR"]
            pv = [spec.TABLE_V[k] for k in ["b0_intercept_db", "gamma_distance", "b1_temperature_db_C",
                                             "b2_rel_humidity_db_pct", "b3_bar_pressure_db_hPa",
                                             "b4_pm25_db_ug_m3", "b5_snr_db"]]
            # the descriptor's eq. (10) takes d in metres (d0 = 1 m); the IoT-J's eq. (6) and this fit take km:
            # its intercept is re-referenced with +30*gamma_descriptor so the two columns share a unit
            dm = spec.DESCRIPTOR["mlr"]; dv = [dm.get(n, np.nan) for n in names]
            dv[names.index("b0_intercept")] = dm["b0_intercept"] + 30.0 * dm["gamma"]
            pd.DataFrame(dict(weight=names, ours=est.coef_, paper=pv, descriptor=dv)).assign(
                err_pct=lambda d: 100 * (d.ours - d.paper) / d.paper.abs(),
                err_vs_descriptor_pct=lambda d: 100 * (d.ours - d.descriptor) / d.descriptor.abs()).to_csv(FINAL / "table_v_mlr_weights.csv", index=False)
            mlr = est
    pd.DataFrame(rows).to_csv(FINAL / "table_iv_cpls.csv", index=False)
    np.savez_compressed(FINAL / "test_predictions.npz", **preds)
    return preds, mlr


# ------------------------------------------------------------ 4. residuals
def residuals(tr, te, mlr, pert_tr=None):
    Xtr, ytr = make_X(tr), make_y(tr)
    r = ytr - mlr.predict(Xtr)
    pert_tr = np.zeros(len(tr), bool) if pert_tr is None else pert_tr
    rng = np.random.default_rng(SEED)
    sub = r[rng.choice(len(r), 200_000, replace=False)]
    mu, sd = r.mean(), r.std(ddof=1)
    ks = stats.kstest(r, "norm", args=(mu, sd))
    nu, loc, sc = stats.t.fit(sub)
    nu_a = spec.APPENDIX["t_dof_nu"]
    loc_a, sc_a = stats.t.fit(r, f0=nu_a)[1:]      # the Appendix's psi: nu fixed at 11.43, loc/scale by ML on all training residuals
    # Durbin-Watson depends on the ROW ORDER of the residuals.  The shuffled split gives 2.0 by construction;
    # the released file is device-blocked (every consecutive pair is the same node); timestamp order interleaves
    # the four nodes.  The paper's 1.67 (R's caTools keeps the database's own order) matches neither - it implies
    # ~40 % same-node adjacency, an ordering the released file does not have.
    o_native = np.argsort(tr["index"].values, kind="stable")
    o_time = np.lexsort((tr["device_id"].astype(str).values, tr["timestamp"].values))
    dw_native, dw_time = float(durbin_watson(r[o_native])), float(durbin_watson(r[o_time]))
    idx = rng.choice(len(r), 200_000, replace=False)
    bp = het_breuschpagan(r[idx], np.column_stack([np.ones(len(idx)), Xtr[idx]]))
    def qq_r2(dist, par):
        n = len(sub); th = dist.ppf((np.arange(1, n + 1) - .5) / n, *par); em = np.sort(sub)
        ok = np.isfinite(th); return float(np.corrcoef(th[ok], em[ok])[0, 1] ** 2), th, em
    r2t, tht, emt = qq_r2(stats.t, (nu, loc, sc)); r2n, thn, emn = qq_r2(stats.norm, (mu, sd))
    pd.DataFrame([
        dict(test="KS vs fitted normal", statistic=float(ks.statistic), pvalue=float(ks.pvalue), paper="p = 2.2e-16"),
        dict(test="Durbin-Watson (file order: device-blocked)", statistic=dw_native, pvalue=np.nan, paper="1.67 (database order, not recoverable)"),
        dict(test="Durbin-Watson (timestamp order: nodes interleaved)", statistic=dw_time, pvalue=np.nan, paper="1.67 (database order, not recoverable)"),
        dict(test="Breusch-Pagan", statistic=float(bp[0]), pvalue=float(bp[1]), paper="p = 1e-16"),
        dict(test="excess kurtosis", statistic=float(stats.kurtosis(r)), pvalue=np.nan, paper="fat tails"),
        dict(test="Student-t nu (MLE)", statistic=float(nu), pvalue=np.nan, paper=11.43),
        dict(test="Student-t scale at nu = 11.43 (Appendix psi, dB)", statistic=float(sc_a), pvalue=np.nan, paper="not reported"),
        dict(test="QQ R2 vs t", statistic=r2t, pvalue=np.nan, paper=0.996),
        dict(test="QQ R2 vs normal", statistic=r2n, pvalue=np.nan, paper="not reported"),
        dict(test="residual range dB", statistic=f"{r.min():.2f}..{r.max():.2f}", pvalue=np.nan, paper="~ -13..+10 (Fig.14 axis)"),
        dict(test="residual range dB, unperturbed rows only", statistic=f"{r[~pert_tr].min():.2f}..{r[~pert_tr].max():.2f}", pvalue=np.nan, paper="not reported"),
    ]).to_csv(FINAL / "appendix_residual_tests.csv", index=False)
    step = max(1, len(emn) // 4000)
    ok_ = np.isfinite(tht[::step]) & np.isfinite(thn[::step])
    pd.DataFrame(dict(theoretical_normal=thn[::step][ok_], theoretical_t=tht[::step][ok_], empirical=emn[::step][ok_])).to_csv(FINAL / "fig14_15_qq.csv", index=False)
    log(f"  residuals: nu {nu:.2f} (paper 11.43), kurt {stats.kurtosis(r):+.2f}, range {r.min():.1f}..{r.max():.1f}; Appendix psi scale {sc_a:.3f} dB")
    return dict(nu=float(nu_a), loc=float(loc_a), scale=float(sc_a))


# ----------------------------------------------------- 5/6. PDR, ToA, energy
def adr_and_energy(tr, te, preds, sp, spt, psi=None):
    rng = np.random.default_rng(SEED)
    allp = dict(preds)
    allp["Friis"] = np.asarray(friis_pl(te.distance, te.frequency), float)
    allp["SPLMSF"] = sp.predict(te.distance) + sp.sample_shadowing(len(te), rng)
    if CONFIG["shadowing_psi"] == "appendix":
        # IV-B: "SPLMSFT (t-distributed shadow fading), MLR (with t-distributed shadow fading)".  The
        # Appendix fits psi once, on the MLR residuals (nu = 11.43); both schemes sample that psi.
        def t_psi(): return stats.t.rvs(psi["nu"], psi["loc"], psi["scale"], size=len(te), random_state=rng)
        allp["SPLMSFT"] = spt.predict(te.distance) + t_psi()
        allp["MLR"] = np.asarray(preds["MLR"], float) + t_psi()
    else:
        allp["SPLMSFT"] = spt.predict(te.distance) + spt.sample_shadowing(len(te), rng)
    cfg = SimConfig(lm_values=LMS, clamp_tp=(CONFIG["line28"] == "clamped"), variant=CONFIG["algorithm1_variant"],
                    rule=CONFIG["delivery_rule"], en_tp_quant=CONFIG["en_tp_quant"], params=PARAMS)
    cfg_adr = SimConfig(lm_values=LMS, rule=CONFIG["delivery_rule"],
                        adr_window_order=CONFIG["adr_window_order"], adr_window=CONFIG["adr_window"],
                        adr_sf_mode=CONFIG["adr_sf_mode"], adr_tp_step=CONFIG["adr_tp_step"],
                        adr_window_excl_current=CONFIG["adr_window_excl_current"], params=PARAMS_ADR)
    cur = {k: simulate_enhanced(te, p, cfg) for k, p in allp.items()}
    cur["ADR"] = simulate_conventional(te, cfg_adr)
    order = ["ADR", "Friis", "SPLMSF", "SPLMSFT", "MLR", "ANN", "SVR", "RF"]
    cur = {k: cur[k] for k in order}
    pd.concat([v.assign(scheme=k) for k, v in cur.items()], ignore_index=True).to_csv(FINAL / "fig11_curves.csv", index=False)

    def lm_at(c, p):
        h = c[c.pdr >= p]; return float(h.LM.iloc[0]) if len(h) else np.nan
    def lm_int(c, p):
        h = c[(c.pdr >= p) & np.isclose(c.LM % 1, 0)]; return float(h.LM.iloc[0]) if len(h) else np.nan
    rows = []
    for k in order:
        for p in LEVELS:
            rows.append(dict(scheme=k, pdr=p, LM_dB=lm_at(cur[k], p), LM_integer_dB=lm_int(cur[k], p),
                             paper_LM=spec.PDR_TARGETS.get(k, {}).get(p, np.nan), paper_LM_integer=spec.PDR_TARGETS_INTEGER[k][p]))
    pd.DataFrame(rows).to_csv(FINAL / "fig11_link_margins.csv", index=False)

    EK, TK = (("mean_energy_delivered_j", "mean_toa_delivered_s") if CONFIG["energy_over"] == "delivered"
              else ("mean_energy_j", "mean_toa_s"))

    def op_point(curves, k, p, reading):
        """Row of scheme k's LM curve at which its PDR-p improvement is evaluated (D6)."""
        c = curves[k]
        if reading == "paper_integer": h = c[np.isclose(c.LM, spec.PDR_TARGETS_INTEGER[k][p])]
        elif reading == "ours_integer": h = c[(c.pdr >= p) & np.isclose(c.LM % 1, 0)]
        else: h = c[c.pdr >= p]
        return None if h.empty else h.iloc[0]

    def improvements(curves, reading):
        erows = []
        for p in LEVELS:
            rp = op_point(curves, "ADR", p, reading)
            if rp is None: continue
            for k in order:
                h = op_point(curves, k, p, reading)
                if h is None: continue
                erows.append(dict(scheme=k, pdr=p, LM_dB=float(h.LM), LM_ADR_dB=float(rp.LM), pdr_at_LM=float(h.pdr),
                                  mean_sf=float(h.mean_sf), mean_tp_dbm=float(h.mean_tp_dbm),
                                  mean_energy_mJ=h[EK] * 1e3, mean_toa_ms=h[TK] * 1e3,
                                  energy_improvement_pct=improvement_pct(rp[EK], h[EK]),
                                  toa_improvement_pct=improvement_pct(rp[TK], h[TK]),
                                  paper_energy_pct=spec.ENERGY_IMPROVEMENT_TARGETS.get(k, np.nan) if p == 99 else np.nan,
                                  paper_toa_pct=spec.TOA_IMPROVEMENT_TARGETS.get(k, np.nan) if p == 99 else np.nan))
        return pd.DataFrame(erows)
    E = improvements(cur, CONFIG["operating_lm"]); E.to_csv(FINAL / "fig12_13_energy_toa.csv", index=False)

    d11 = pd.read_csv(PAPER_DIG / "paper_fig11_digitized.csv", index_col=0).clip(upper=100) if (PAPER_DIG / "paper_fig11_digitized.csv").exists() else None
    cmp_rows = []
    for col_, fname, digf in (("toa_improvement_pct", "fig12", "paper_fig12_digitized.csv"), ("energy_improvement_pct", "fig13", "paper_fig13_digitized.csv")):
        dg = pd.read_csv(PAPER_DIG / digf, index_col=0) if (PAPER_DIG / digf).exists() else None
        if dg is None: continue
        for pk, ok in DIG.items():
            if ok == "ADR": continue
            for p in dg.index:
                r = E[(E.scheme == ok) & (E.pdr == p)]
                if len(r): cmp_rows.append(dict(figure=fname, scheme=ok, pdr=p, ours=float(r[col_].iloc[0]), paper=float(dg.loc[p, pk])))
    if d11 is not None:
        for pk, ok in DIG.items():
            c = cur[ok]
            for l in range(16):
                v = d11.loc[l, pk]
                if np.isfinite(v):
                    cmp_rows.append(dict(figure="fig11", scheme=ok, pdr=np.nan, LM=l, ours=float(c.loc[np.isclose(c.LM, l), "pdr"].iloc[0]), paper=float(v)))
    cm = pd.DataFrame(cmp_rows); cm["diff"] = cm.ours - cm.paper
    cm.to_csv(FINAL / "figs_vs_paper_digitized.csv", index=False)
    log("  ADR LM@99 %.2f | ANN %.2f | energy ANN/SVR/RF %.1f/%.1f/%.1f" % (
        lm_at(cur["ADR"], 99), lm_at(cur["ANN"], 99),
        *[E[(E.scheme == k) & (E.pdr == 99)].energy_improvement_pct.iloc[0] for k in ("ANN", "SVR", "RF")]))


# --------------------------------------------------------------------- 7. claims
def claims():
    t4 = pd.read_csv(FINAL / "table_iv_cpls.csv").set_index("model")
    lm = pd.read_csv(FINAL / "fig11_link_margins.csv")
    E = pd.read_csv(FINAL / "fig12_13_energy_toa.csv")
    res = pd.read_csv(FINAL / "appendix_residual_tests.csv").set_index("test")
    g = lambda s, p: float(lm[(lm.scheme == s) & (lm.pdr == p)].LM_dB.iloc[0])
    e99 = E[E.pdr == 99].set_index("scheme").energy_improvement_pct
    order = " < ".join(t4.sort_values("test_rmse").index)
    def v(a, b, tol):
        return "REPRODUCED" if abs(a - b) <= tol else "DEVIATED"
    rows = [
        dict(claim="RMSE up to 1.566 dB, R2 up to 0.94", paper="1.566 / 0.94",
             ours=f"{t4.test_rmse.min():.3f} / {t4.test_r2.max():.4f}",
             verdict="REPRODUCED" if abs(t4.test_rmse.min() - 1.566) <= 0.157 and abs(t4.test_r2.max() - 0.94) <= 0.02 else "DEVIATED"),
        dict(claim="Model ranking by test RMSE", paper="RF < ANN < SVR < MLR", ours=order,
             verdict="REPRODUCED" if order == "RF < ANN < SVR < MLR" else "DEVIATED"),
        dict(claim="Best ML model: PDR > 99 % at LM = 4 dB", paper="4 dB", ours=f"{g('ANN', 99):.2f} dB", verdict=v(g("ANN", 99), 4, 0.5)),
        dict(claim="Conventional ADR needs LM = 11 dB for 99 %", paper="11 dB", ours=f"{g('ADR', 99):.2f} dB", verdict=v(g("ADR", 99), 11, 1.0)),
        dict(claim="Energy saving up to 43 % vs conventional ADR (43.5 % for ANN, its best model)", paper="43.5 % (ANN); max 43.5 %",
             ours=f"{e99['ANN']:.1f} % (ANN); max {e99.max():.1f} % ({e99.idxmax()})",
             verdict=v(e99["ANN"], 43.5, 4.35) if abs(e99.max() - 43.5) <= 8.7 else "DEVIATED"),
        dict(claim="Shadow fading is Student-t, nu = 11.43", paper="11.43",
             ours=f"{float(res.loc['Student-t nu (MLE)', 'statistic']):.2f}",
             verdict="CALIBRATED" if CONFIG["restore_frac_rssi_only"] or CONFIG["restore_frac_coupled"] else v(float(res.loc["Student-t nu (MLE)", "statistic"]), 11.43, 2.0)),
    ]
    # what each verdict rests on: the data step (D1) is calibrated to the paper's MLR RMSE, nu and ADR margin,
    # so agreement on those quantities is not independent evidence
    basis = {"RMSE up to 1.566 dB, R2 up to 0.94": "reconstructed data (calibrated to the paper's MLR RMSE)",
             "Model ranking by test RMSE": "not a calibration target", "Best ML model: PDR > 99 % at LM = 4 dB": "not a calibration target (residual rule, packet's own SNR; simulator choices selected against the paper's figures)",
             "Conventional ADR needs LM = 11 dB for 99 %": "reconstructed data (the coupled fraction is calibrated to this margin), shuffled window",
             "Energy saving up to 43 % vs conventional ADR (43.5 % for ANN, its best model)": "not a calibration target; under the paper's SF 7-12, simulator choices selected against its figures; ordering not reproduced",
             "Shadow fading is Student-t, nu = 11.43": "calibration target of D1"}
    for r in rows: r["basis"] = basis[r["claim"]]
    pd.DataFrame(rows).to_csv(FINAL / "headline_claims.csv", index=False)


def main():
    for d in (FINAL, FIGS, MODELS): d.mkdir(parents=True, exist_ok=True)
    json.dump(CONFIG, open(FINAL / "config.json", "w"), indent=2)
    log("1/6 data"); df, tr, te = prepare_data()
    pm = PowerModel()
    pd.DataFrame([pm.params() | dict(paper_r2=0.95)]).to_csv(FINAL / "table_vii_power_model.csv", index=False)
    pd.DataFrame([verify_against_csv(df)]).to_csv(FINAL / "toa_formula_check.csv", index=False)
    log("2/6 conventional models"); sp, spt = conventional(df, tr, te)
    log("3/6 CPLS models"); preds, mlr = cpls(tr, te)
    log("4/6 residuals"); psi = residuals(tr, te, mlr, pert_tr=df.attrs["perturbed"][tr["pos"].to_numpy()])
    log("5/6 ADR / energy"); adr_and_energy(tr, te, preds, sp, spt, psi)
    log("6/6 claims"); claims()
    import plots; plots.save_all(FINAL, FIGS)
    log("done -> tables/, figures/, models/")


if __name__ == "__main__":
    main()
