"""Statement-by-statement audit of the PDR/energy simulation against Section IV's text.

Reproduces the evidence behind determinations D2, D3, D5, D6, D7 (src/assumptions.py) using the
deliverable's own simulators, scored against the digitized figures in ../paper_digitized/:
  1. Fig. 11: the Appendix psi for MLR and SPLMSFT (D7); the previous-20 ADR window (D3)
  2. Figs. 12/13: operating-point reading (D6) x Algorithm-1 TP semantics (D2) x ADR set-point (D5)
  3. Figs. 12/13 under the paper's reading: ADR SF policy, SF caps, delivered-only averaging
  4. Energy/airtime profiles vs LM: the paper's, extracted from its own Figs. 12/13, against ours
Run from the project root after run_final.py:  python outputs/provenance/adr_mechanism/text_audit.py
Writes text_audit_fig11.csv, text_audit_scores.csv, text_audit_adr_policy.csv, adr_profile.csv here.
"""
import sys, time, itertools, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import joblib, numpy as np, pandas as pd
from scipy import stats
from adr_algorithm import AdrParameters
from conventional_models import SPLMSF, SPLMSFT, friis_pl
from cpls_models import make_X, make_y
from data_loading import load_cached
from energy import improvement_pct
from reconstruction import restore_outliers
from simulation import SimConfig, simulate_conventional, simulate_enhanced
from splitting import split
import paper_spec as spec

HERE = pathlib.Path(__file__).resolve().parent; FINAL = ROOT / "outputs/final"; DIG = HERE.parent / "paper_digitized"
T11 = pd.read_csv(DIG / "paper_fig11_digitized.csv", index_col=0).clip(upper=100)
T12 = pd.read_csv(DIG / "paper_fig12_digitized.csv", index_col=0); T13 = pd.read_csv(DIG / "paper_fig13_digitized.csv", index_col=0)
NAME = {"ANN": "ANN", "SVR": "SVR", "RF": "RF", "MLR": "MLR", "FRIIS": "Friis", "SPLMSF": "SPLMSF", "SPLMSFT": "SPLMSFT"}
LEVELS = (80, 85, 90, 95, 99); LMS = tuple(np.round(np.arange(0, 15.001, 0.25), 2))

raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True); tr, te = split(restore_outliers(clean, seed=42))
z = np.load(FINAL / "test_predictions.npz"); preds = {k: np.asarray(z[k], float) for k in z.files}
y_tr, y_te = tr.experimental_pl.values, te.experimental_pl.values
sp = SPLMSF().fit(tr.distance, y_tr); spt = SPLMSFT(nu=spec.APPENDIX["t_dof_nu"]).fit(tr.distance, y_tr)
mlr = joblib.load(FINAL / "models/cpls_mlr.joblib"); r_tr = make_y(tr) - mlr.predict(make_X(tr))
nu_a = spec.APPENDIX["t_dof_nu"]; loc_a, sc_a = stats.t.fit(r_tr, f0=nu_a)[1:]
print(f"Appendix psi: t(nu={nu_a}) loc {loc_a:.3f} scale {sc_a:.3f} dB (MLR training residual sd {r_tr.std():.3f})", flush=True)
rng = np.random.default_rng(42); n = len(te)
def t_psi(): return stats.t.rvs(nu_a, loc_a, sc_a, size=n, random_state=rng)
P = dict(preds); P["Friis"] = np.asarray(friis_pl(te.distance, te.frequency), float)
P["SPLMSF"] = sp.predict(te.distance) + sp.sample_shadowing(n, rng)
P["SPLMSFT_own"] = spt.predict(te.distance) + spt.sample_shadowing(n, rng); P["SPLMSFT"] = spt.predict(te.distance) + t_psi()
P["MLR_det"] = preds["MLR"]; P["MLR"] = preds["MLR"] + t_psi()

# ---- 1. Fig. 11 checks
rows = []
for k, pk in (("MLR_det", "MLR"), ("MLR", "MLR"), ("SPLMSFT_own", "SPLMSFT"), ("SPLMSFT", "SPLMSFT")):
    c = np.array([((y_te - P[k]) < l).mean() * 100 for l in range(16)])
    rows.append(dict(check="psi", scheme=k, fig11_rmse=float(np.sqrt(np.nanmean((c[:12] - T11[pk].values[:12]) ** 2))), **{f"pdr_lm{l}": c[l] for l in range(0, 8)}))
for excl in (False, True):
    c = simulate_conventional(te, SimConfig(lm_values=tuple(range(16)), rule="residual", adr_window_excl_current=excl, params=AdrParameters(max_sf=12)))
    rows.append(dict(check="window", scheme="ADR previous 20" if excl else "ADR incl. current", fig11_rmse=float(np.sqrt(np.mean((c.pdr.values[:12] - T11["ADR"].values[:12]) ** 2))), **{f"pdr_lm{l}": c.pdr.values[l] for l in range(0, 8)}))
F = pd.DataFrame(rows); F.to_csv(HERE / "text_audit_fig11.csv", index=False); print("\n=== 1. Fig. 11 ===\n" + F.round(1).to_string(index=False), flush=True)

# ---- 2. reading x TP semantics x ADR set-point
def op(curve, k, level, reading):
    i = LEVELS.index(level)
    if reading == "paper_integer": h = curve[np.isclose(curve.LM, spec.PDR_TARGETS_INTEGER[k][level])]
    elif reading == "ours_integer": h = curve[(curve.pdr >= level) & np.isclose(curve.LM % 1, 0)]
    else: h = curve[curve.pdr >= level]
    return None if h.empty else h.iloc[0]
def score(cur, adr, reading, ek="mean_energy_j", tk="mean_toa_s"):
    e, t, det = [], [], []
    for pk, ok in NAME.items():
        for lvl in LEVELS:
            ra, h = op(adr, "ADR", lvl, reading), op(cur[ok], ok, lvl, reading)
            if ra is None or h is None: continue
            e.append(improvement_pct(ra[ek], h[ek]) - T13.loc[lvl, pk]); t.append(improvement_pct(ra[tk], h[tk]) - T12.loc[lvl, pk]); det.append(ok in ("ANN", "SVR", "RF", "Friis"))
    e, t, det = np.array(e), np.array(t), np.array(det)
    return dict(energy_rmse=np.sqrt(np.mean(e ** 2)), toa_rmse=np.sqrt(np.mean(t ** 2)), energy_rmse_ml_friis=np.sqrt(np.mean(e[det] ** 2)), toa_rmse_ml_friis=np.sqrt(np.mean(t[det] ** 2)))
t0 = time.time(); EN, ADR = {}, {}
for var in ("corrected", "text"):
    for sfm in (12, 10):
        cfg = SimConfig(lm_values=LMS, clamp_tp=True, variant=var, rule="residual", params=AdrParameters(max_sf=sfm))
        EN[(var, sfm)] = {k: simulate_enhanced(te, P[k], cfg) for k in NAME.values()}
        print(f"  EN {var} sf<={sfm} {time.time()-t0:.0f}s", flush=True)
for step, mode, sfm in itertools.product((None, 3.0), ("both", "down_only", "up_only", "fixed"), (12, 10)):
    ADR[("continuous" if step is None else "nStep3", mode, sfm)] = simulate_conventional(te, SimConfig(lm_values=LMS, rule="residual", adr_tp_step=step, adr_sf_mode=mode, adr_window_excl_current=True, params=AdrParameters(max_sf=sfm)))
print(f"  ADR curves {time.time()-t0:.0f}s", flush=True)
rows = []
for var, step, reading in itertools.product(("corrected", "text"), ("continuous", "nStep3"), ("grid", "ours_integer", "paper_integer")):
    rows.append(dict(tp_semantics=var, adr_set_point=step, reading=reading, **score(EN[(var, 12)], ADR[(step, "both", 12)], reading)))
S = pd.DataFrame(rows).sort_values("energy_rmse"); S.to_csv(HERE / "text_audit_scores.csv", index=False)
print("\n=== 2. reading x TP semantics x ADR set-point (SF 7-12, both, all packets) ===\n" + S.round(1).to_string(index=False), flush=True)

# ---- 3. ADR policy under the paper's reading
rows = []
for (var, sfm_en), (step, mode, sfm_adr), over in itertools.product(EN, ADR, ("all", "delivered")):
    ek, tk = ("mean_energy_j", "mean_toa_s") if over == "all" else ("mean_energy_delivered_j", "mean_toa_delivered_s")
    rows.append(dict(tp_semantics=var, sf_max_en=sfm_en, adr_set_point=step, adr_sf_mode=mode, sf_max_adr=sfm_adr, energy_over=over, **score(EN[(var, sfm_en)], ADR[(step, mode, sfm_adr)], "paper_integer", ek, tk)))
A = pd.DataFrame(rows); A["total"] = np.sqrt((A.energy_rmse ** 2 + A.toa_rmse ** 2) / 2); A = A.sort_values("total"); A.to_csv(HERE / "text_audit_adr_policy.csv", index=False)
print("\n=== 3. ADR policy under the paper's reading (top 12 of %d) ===\n" % len(A) + A.head(12).round(1).to_string(index=False), flush=True)

# ---- 4. profiles vs LM: the paper's (from its own Figs. 12/13) against ours
def paper_ratio(T, k, p1, p2):   # same scheme LM at p1 and p2  =>  X_ADR(L2)/X_ADR(L1) = (1 - I1)/(1 - I2)
    return (1 - T.loc[p1, k] / 100) / (1 - T.loc[p2, k] / 100)
pairs = [("SVR", 80, 85, "6->7"), ("RF", 80, 85, "6->7"), ("ANN", 85, 90, "7->8"), ("SVR", 90, 95, "8->9"), ("RF", 90, 95, "8->9")]
rows = [dict(source=f"paper via {k} at LM {spec.PDR_TARGETS_INTEGER[k][p1]}", step=s, energy_ratio=paper_ratio(T13, k, p1, p2), toa_ratio=paper_ratio(T12, k, p1, p2)) for k, p1, p2, s in pairs]
for key in (("continuous", "both", 12), ("continuous", "down_only", 12), ("continuous", "fixed", 12), ("nStep3", "both", 12), ("continuous", "both", 10)):
    c = ADR[key]; E = {l: c.loc[np.isclose(c.LM, l), "mean_energy_j"].iloc[0] for l in range(6, 12)}; T = {l: c.loc[np.isclose(c.LM, l), "mean_toa_s"].iloc[0] for l in range(6, 12)}
    for a, b in ((6, 7), (7, 8), (8, 9), (9, 11)):
        rows.append(dict(source=f"ours ADR {key[0]} {key[1]} sf<={key[2]}", step=f"{a}->{b}", energy_ratio=E[b] / E[a], toa_ratio=T[b] / T[a]))
Pf = pd.DataFrame(rows); Pf.to_csv(HERE / "adr_profile.csv", index=False)
print("\n=== 4. ADR energy/airtime ratio between operating LMs ===\n" + Pf.round(3).to_string(index=False))
print(f"\nlogged SF >= 11 in the test set: {(te.sf >= 11).mean()*100:.1f} %")
print("\nTEXT AUDIT DONE")
