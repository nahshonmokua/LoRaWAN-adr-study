"""Observation 3 of Fig. 11: "the t-distribution improved the PDR since SPLMSF has a worse behavior than
SPLMSFT".  Is it the t SHAPE or the SCALE of the sampled psi?  Under the residual rule the PDR curve of
a parametric scheme is P(residual - psi_sampled < LM); SPLMSF samples its own residual sd, SPLMSFT the
Appendix psi (t, nu = 11.43, fitted on the MLR residuals - a narrower term from a better model).
Run from the project root after run_final.py:  python outputs/provenance/psi_scale_check.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from scipy import stats
from conventional_models import SPLMSF
from data_loading import load_cached
from reconstruction import restore_outliers
from splitting import split
import paper_spec as spec

raw = load_cached(); clean = raw[~(raw.rh <= 2)].reset_index(drop=True); tr, te = split(restore_outliers(clean, seed=42))
y_tr, y_te = tr.experimental_pl.values, te.experimental_pl.values
sp = SPLMSF().fit(tr.distance, y_tr); base = sp.predict(te.distance); rng = np.random.default_rng(42); n = len(te)
rs = pd.read_csv(ROOT / "outputs/final/appendix_residual_tests.csv")
sc_a = float(rs[rs.test.str.startswith("Student-t scale")].statistic.iloc[0]); nu = spec.APPENDIX["t_dof_nu"]
sd_t, sd_own = sc_a * np.sqrt(nu / (nu - 2)), sp.psi_params_["scale"]
D = pd.read_csv(ROOT / "outputs/provenance/paper_digitized/paper_fig11_digitized.csv", index_col=0).clip(upper=100)
LM = list(range(0, 12))
def curve(pred): return np.array([((y_te - pred) < l).mean() * 100 for l in LM])
cases = {"SPLMSF + normal, own sd (final SPLMSF)": base + rng.normal(0, sd_own, n),
         "SPLMSF + t, own sd": base + stats.t.rvs(nu, 0, sd_own / np.sqrt(nu / (nu - 2)), size=n, random_state=rng),
         "SPLMSF + normal, Appendix sd": base + rng.normal(0, sd_t, n),
         "SPLMSF + t, Appendix scale (final SPLMSFT)": base + stats.t.rvs(nu, 0, sc_a, size=n, random_state=rng)}
rows = []
for k, p in cases.items():
    c = curve(p); ref = "SPLMSFT" if "Appendix" in k else "SPLMSF"
    rows.append(dict(case=k, psi_sd=sd_t if "Appendix" in k else sd_own, rmse_vs_paper_SPLMSF=np.sqrt(np.nanmean((c - D["SPLMSF"].values[:12]) ** 2)),
                     rmse_vs_paper_SPLMSFT=np.sqrt(np.nanmean((c - D["SPLMSFT"].values[:12]) ** 2)), **{f"pdr_lm{l}": c[i] for i, l in enumerate(LM)}))
out = pd.DataFrame(rows); out.to_csv(pathlib.Path(__file__).with_suffix(".csv"), index=False)
print(f"psi sd: own {sd_own:.3f} dB, Appendix {sd_t:.3f} dB"); print(out.round(1).to_string(index=False))
