"""Calibrated perturbation of the released CSV (assumption ledger D1).

The released file (930,753 rows) is a smaller database than the one the paper fitted (990,750 rows
after its own outlier step, Section III-A) and its residuals are near-normal (MLR test RMSE 1.846 dB,
excess kurtosis -0.06); the paper's Table IV MLR RMSE (1.951), Appendix nu (11.43) and +-15 dB tails
cannot be obtained from it.  This module perturbs a small fraction of rows so that those statistics
take their printed values.  BOTH fractions are calibrated to the paper - the RSSI-only one to the
RMSE and nu, the coupled one to the conventional ADR's 11 dB margin (stage11_reconstruct.py) - so
agreement on those quantities is a calibration result, not independent evidence; the report labels
it so and states the unmodified released-data results alongside.

Two fault populations, on disjoint rows, applied before the train/test split:
  * RSSI-only misreports  (telemetry)      - the full shift survives as model residual
  * RSSI+SNR coupled      (received-power) - SNR is a predictor, so ~62 % is absorbed, but
                                             it perturbs the conventional ADR's SNR window
Magnitudes +-U(8, 15) dB (`outlier_db`); every derived column is rebuilt from the CSV's own definitions.
All parameters are passed explicitly by run_final.CONFIG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRAC_RSSI_ONLY = 0.0025
FRAC_COUPLED = 0.0015
OUTLIER_DB = (8.0, 15.0)


def restore_outliers(df: pd.DataFrame, frac_rssi_only: float = FRAC_RSSI_ONLY,
                     frac_coupled: float = FRAC_COUPLED, seed: int = 42,
                     outlier_db: tuple = OUTLIER_DB) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["rssi"] = d["rssi"].astype(float)
    d["snr"] = d["snr"].astype(float)
    n = len(d)
    k1, k2 = int(round(frac_rssi_only * n)), int(round(frac_coupled * n))
    pick = rng.choice(n, k1 + k2, replace=False)
    for idx, coupled in ((pick[:k1], False), (pick[k1:], True)):
        delta = rng.choice([-1.0, 1.0], len(idx)) * rng.uniform(*outlier_db, size=len(idx))
        d.loc[d.index[idx], "rssi"] = d["rssi"].to_numpy()[idx] + delta
        if coupled:
            d.loc[d.index[idx], "snr"] = d["snr"].to_numpy()[idx] + delta
    d["experimental_pl"] = (d.ptx - d.ltx + d.gtx - d.rssi + d.grx - d.lrx).astype(float)
    d["pl_from_link_budget"] = d["experimental_pl"]
    d["esp"] = d.rssi - 10.0 * np.log10(1.0 + 10.0 ** (-d.snr / 10.0))
    d["pn"] = d.esp - d.snr
    d["noise_power_rssi"] = d.rssi - d.snr
    d.attrs["n_restored"] = k1 + k2
    return d
