"""Stage 7 - LoRa time-on-air and the Table VII power model.

The ToA expression is the standard Semtech SX1276 / LoRaWAN one (the paper only
cites it, A18).  It is *verified exactly* against the released `toa` column: with
BW = 125 kHz, CR = 4/5, CRC on, explicit header and an 8-symbol preamble it
reproduces every row of LoRaWAN_PathLossMeasurements.csv to floating-point
precision (see `verify_against_csv`).
"""
from __future__ import annotations

import numpy as np

BANDWIDTH_HZ = 125_000
PREAMBLE_SYMBOLS = 8
SUPPLY_VOLTAGE_V = 3.3
TABLE_VII = [(7, 20), (13, 29), (17, 87), (20, 120)]      # (dBm, mA)


def low_data_rate_optimize(sf, bandwidth_hz=BANDWIDTH_HZ) -> np.ndarray:
    """DE = 1 when the symbol period exceeds 16 ms (SF11/SF12 at 125 kHz)."""
    return (2.0 ** np.asarray(sf, float) / bandwidth_hz > 16e-3).astype(float)


def time_on_air(sf, payload_bytes=1, bandwidth_hz=BANDWIDTH_HZ, coding_rate=1,
                explicit_header=True, crc=True, preamble=PREAMBLE_SYMBOLS) -> np.ndarray:
    """LoRa time on air, in seconds.

        T_sym     = 2^SF / BW
        T_preamble= (n_preamble + 4.25) * T_sym
        n_payload = 8 + max(ceil((8*PL - 4*SF + 28 + 16*CRC - 20*IH)
                                 / (4*(SF - 2*DE))) * (CR + 4), 0)
        ToA       = T_preamble + n_payload * T_sym

    `coding_rate` is the CR index 1..4, i.e. 4/(4+CR); the paper uses 4/5 -> 1.
    """
    sf = np.asarray(sf, float)
    pl = np.asarray(payload_bytes, float)
    de = low_data_rate_optimize(sf, bandwidth_hz)
    ih = 0.0 if explicit_header else 1.0
    crc_bits = 16.0 if crc else 0.0

    t_sym = 2.0 ** sf / bandwidth_hz
    t_preamble = (preamble + 4.25) * t_sym
    num = 8.0 * pl - 4.0 * sf + 28.0 + crc_bits - 20.0 * ih
    den = 4.0 * (sf - 2.0 * de)
    n_payload = 8.0 + np.maximum(np.ceil(num / den) * (coding_rate + 4.0), 0.0)
    return t_preamble + n_payload * t_sym


def verify_against_csv(df, atol=1e-9) -> dict:
    """Check `time_on_air` against the released `toa` column using each row's own SF
    and frame_length.  Returns the max absolute error and the mismatch count."""
    pred = time_on_air(df["sf"].to_numpy(), df["frame_length"].to_numpy())
    err = np.abs(pred - df["toa"].to_numpy(float))
    return dict(max_abs_error_s=float(err.max()), n_mismatch=int((err > atol).sum()), n=len(df))


class PowerModel:
    """Linear fit of consumed power on radiated power, from Table VII.

        P_consumed[mW] = a + b * P_radiated[mW],   P_consumed = V * I

    The paper: "We transformed the given power to mW.  Then we fitted a simple
    linear regression to get a closed-form expression to forecast the power
    consumption of different transmission powers (from 0 to 20 dBm), obtaining an
    R2 = 0.95."
    """

    def __init__(self, table=TABLE_VII, voltage_v=SUPPLY_VOLTAGE_V):
        self.voltage_v = voltage_v
        dbm = np.array([p for p, _ in table], float)
        ma = np.array([i for _, i in table], float)
        self.p_radiated_mw_ = 10.0 ** (dbm / 10.0)
        self.p_consumed_mw_ = voltage_v * ma
        A = np.column_stack([np.ones_like(self.p_radiated_mw_), self.p_radiated_mw_])
        coef, *_ = np.linalg.lstsq(A, self.p_consumed_mw_, rcond=None)
        self.intercept_mw_, self.slope_ = float(coef[0]), float(coef[1])
        yhat = A @ coef
        ss_res = float(np.sum((self.p_consumed_mw_ - yhat) ** 2))
        ss_tot = float(np.sum((self.p_consumed_mw_ - self.p_consumed_mw_.mean()) ** 2))
        self.r2_ = 1.0 - ss_res / ss_tot

    def consumed_power_mw(self, tp_dbm) -> np.ndarray:
        """Forecast consumed power (mW) for a transmission power in dBm."""
        p_rad_mw = 10.0 ** (np.asarray(tp_dbm, float) / 10.0)
        return self.intercept_mw_ + self.slope_ * p_rad_mw

    def energy_j(self, tp_dbm, toa_s) -> np.ndarray:
        """E = P_T x ToA (A19: P_T is the *consumed* power from this fit)."""
        return self.consumed_power_mw(tp_dbm) * 1e-3 * np.asarray(toa_s, float)

    def params(self) -> dict:
        return dict(intercept_mw=self.intercept_mw_, slope_mw_per_mw=self.slope_,
                    r2=self.r2_, voltage_v=self.voltage_v)


def table_vii_interpretations() -> "list[dict]":
    """Which reading of 'fitted a simple linear regression ... obtaining an R2 = 0.95'?

    Only one of the plausible readings gives 0.95, which pins the interpretation (A19).
    """
    dbm = np.array([p for p, _ in TABLE_VII], float)
    ma = np.array([i for _, i in TABLE_VII], float)
    mw = 10.0 ** (dbm / 10.0)

    def r2(x, y):
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        return float(1.0 - res @ res / ((y - y.mean()) @ (y - y.mean())))

    return [
        dict(reading="consumed power [mW] = a + b * radiated power [mW]  (used)", r2=r2(mw, SUPPLY_VOLTAGE_V * ma)),
        dict(reading="current [mA] = a + b * radiated power [mW]", r2=r2(mw, ma)),
        dict(reading="current [mA] = a + b * radiated power [dBm]", r2=r2(dbm, ma)),
        dict(reading="consumed power [mW] = a + b * radiated power [dBm]", r2=r2(dbm, SUPPLY_VOLTAGE_V * ma)),
    ]


def improvement_pct(reference, candidate) -> float:
    """Percent improvement of `candidate` over `reference` (lower is better)."""
    reference, candidate = float(reference), float(candidate)
    return 100.0 * (reference - candidate) / reference
