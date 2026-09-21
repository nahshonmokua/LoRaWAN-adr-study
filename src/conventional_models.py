"""Stage 2 - conventional path loss models: Friis, Two-ray, Okumura-Hata, SPLMSF, SPLMSFT.

All models take a DataFrame with the released column names and return path loss in dB.
Importable and dataset-agnostic: nothing here reads a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

C_LIGHT = 299_792_458.0


# ------------------------------------------------------------------ metrics
def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def r2_determination(y, yhat) -> float:
    """Textbook coefficient of determination, 1 - SSE/SST."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def r2_correlation(y, yhat) -> float:
    """Squared Pearson correlation.  This is what the paper's Tables III/IV report (A5)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if np.std(yhat) == 0:
        return float("nan")
    return float(np.corrcoef(y, yhat)[0, 1] ** 2)


def score(y, yhat) -> dict:
    return dict(rmse=rmse(y, yhat), r2_corr=r2_correlation(y, yhat), r2_det=r2_determination(y, yhat))


# ---------------------------------------------------------------- the models
def friis_pl(d_m, f_hz) -> np.ndarray:
    """Free-space path loss, PL = 20log10(4*pi*d*f/c).  No fitted parameter."""
    d_m, f_hz = np.asarray(d_m, float), np.asarray(f_hz, float)
    return 20.0 * np.log10(4.0 * np.pi * d_m * f_hz / C_LIGHT)


def two_ray_pl(d_m, ht_m, hr_m) -> np.ndarray:
    """Plane-earth two-ray model, PL = 40log10(d) - 20log10(ht) - 20log10(hr)  (A6)."""
    d_m = np.asarray(d_m, float)
    return 40.0 * np.log10(d_m) - 20.0 * np.log10(np.asarray(ht_m, float)) \
        - 20.0 * np.log10(np.asarray(hr_m, float))


def okumura_hata_pl(d_m, f_hz, ht_m, hr_m) -> np.ndarray:
    """Hata urban (small/medium city) formula (A6).  f in MHz, d in km, heights in m."""
    f = np.asarray(f_hz, float) / 1e6
    d = np.asarray(d_m, float) / 1000.0
    hb = np.asarray(ht_m, float)          # base-station-side (here the EN mast)
    hm = np.asarray(hr_m, float)          # mobile-side (here the GW)
    a_hm = (1.1 * np.log10(f) - 0.7) * hm - (1.56 * np.log10(f) - 0.8)
    return (69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(hb) - a_hm
            + (44.9 - 6.55 * np.log10(hb)) * np.log10(d))


class SPLMSF:
    """Simplified path loss model with lognormal shadow fading, eq. (2).

        PL = -K + 10*gamma*log10(d/d0) + psi,   psi ~ N(0, sigma)

    `-K` and `gamma` are fitted by OLS on log10(d/d0); psi is fitted afterwards on
    the residuals.  d0 = 1 m (A7).
    """
    name = "SPLMSF"
    shadow_dist = "lognormal (normal in dB)"

    def __init__(self, d0_m: float = 1.0):
        self.d0_m = d0_m
        self.K_ = self.gamma_ = None
        self.psi_params_: dict = {}

    def fit(self, d_m, pl_db) -> "SPLMSF":
        x = np.log10(np.asarray(d_m, float) / self.d0_m)
        A = np.column_stack([np.ones_like(x), 10.0 * x])
        coef, *_ = np.linalg.lstsq(A, np.asarray(pl_db, float), rcond=None)
        minus_K, self.gamma_ = float(coef[0]), float(coef[1])
        self.K_ = -minus_K
        resid = np.asarray(pl_db, float) - self.predict(d_m)
        self.psi_params_ = dict(loc=float(resid.mean()), scale=float(resid.std(ddof=1)))
        return self

    def predict(self, d_m) -> np.ndarray:
        return -self.K_ + 10.0 * self.gamma_ * np.log10(np.asarray(d_m, float) / self.d0_m)

    def sample_shadowing(self, n: int, rng) -> np.ndarray:
        return rng.normal(self.psi_params_["loc"], self.psi_params_["scale"], n)

    def K_at(self, d0_m: float) -> float:
        """Intercept re-referenced to another d0.  Fig. 4's caption (K=84.2, gamma=2.7)
        is only consistent with d0 = 1 km and PL = +K + 10*gamma*log10(d/d0)."""
        return float(self.predict(np.array([d0_m]))[0])

    def params(self) -> dict:
        return dict(K_db=self.K_, K_at_d0_1km=self.K_at(1000.0), gamma=self.gamma_,
                    d0_m=self.d0_m, **{f"psi_{k}": v for k, v in self.psi_params_.items()})


class SPLMSFT(SPLMSF):
    """SPLMSF whose shadow-fading term is Student-t instead of normal.

    The deterministic part is identical (same OLS fit); only psi changes.  nu, loc
    and scale are obtained by maximum likelihood, exactly as the Appendix describes.
    """
    name = "SPLMSFT"
    shadow_dist = "Student-t"

    def __init__(self, d0_m: float = 1.0, nu: float | None = None):
        super().__init__(d0_m)
        self.nu_fixed = nu        # pass 11.43 to force the paper's Appendix value

    def fit(self, d_m, pl_db) -> "SPLMSFT":
        super().fit(d_m, pl_db)
        resid = np.asarray(pl_db, float) - self.predict(d_m)
        if self.nu_fixed is None:
            nu, loc, scale = stats.t.fit(resid)
        else:
            nu = self.nu_fixed
            loc, scale = stats.t.fit(resid, f0=nu)[1:]
        self.psi_params_ = dict(nu=float(nu), loc=float(loc), scale=float(scale))
        return self

    def sample_shadowing(self, n: int, rng) -> np.ndarray:
        p = self.psi_params_
        return stats.t.rvs(p["nu"], loc=p["loc"], scale=p["scale"], size=n,
                           random_state=rng)


def fit_t_dof(residuals) -> dict:
    """MLE Student-t fit of the residuals (Appendix: nu = 11.43)."""
    nu, loc, scale = stats.t.fit(np.asarray(residuals, float))
    return dict(nu=float(nu), loc=float(loc), scale=float(scale))


def predict_all_conventional(df: pd.DataFrame, splmsf: SPLMSF | None = None) -> pd.DataFrame:
    """Deterministic predictions of every conventional model for a measurement frame."""
    out = pd.DataFrame(index=df.index)
    out["Friis"] = friis_pl(df["distance"], df["frequency"])
    out["Two-ray"] = two_ray_pl(df["distance"], df["ht"], df["hr"])
    out["Okumura-Hata"] = okumura_hata_pl(df["distance"], df["frequency"], df["ht"], df["hr"])
    if splmsf is not None:
        out["SPLMSF"] = splmsf.predict(df["distance"])
    return out
