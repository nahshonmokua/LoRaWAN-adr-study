"""Stage 6 - PDR / ToA / energy simulation over a link-margin sweep (Figs. 11-13).

Decision rule
-------------
The paper states it as: "if the actual RSSI was greater than the predicted RSSI,
the packet was delivered successfully; if the actual RSSI was less than the
predicted RSSI, the packet was dropped."

Taken at face value this is degenerate.  Algorithm 1's estimated RSSI and the
actual RSSI share the same transmission power, so

    actual_rssi > estimated_rssi   <=>   PL_predicted > PL_true

which contains no LM at all: PDR would be a horizontal line at P(over-prediction),
about 50 % for any unbiased model.  Fig. 11 plainly is not a horizontal line.

What *does* reproduce Fig. 11 is comparing the actual received power at the
transmission power Algorithm 1 chose against the demodulation threshold, with the
link margin acting as the headroom it was budgeted as:

    delivered  <=>  rssi_logged + (tp_new - tp_logged)  >  noise_power + SNR_limit(sf_new)

When the TP is not clamped this reduces to the intuitive statement

    delivered  <=>  (PL_true - PL_predicted) < LM

i.e. the packet survives exactly when the model's under-prediction of path loss is
smaller than the margin that was set aside for it.  The same rule applied to the
conventional TTN ADR becomes (SNR_rolling_max - SNR_actual) < LM.

Both readings are implemented (`rule="threshold"` and `rule="literal"`) so the
reproduction can show why the literal one cannot have produced Fig. 11.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adr_algorithm import (AdrParameters, enhanced_adr_vectorized,
                           snr_limit_of)
from energy import PowerModel, time_on_air


@dataclass
class SimConfig:
    lm_values: tuple = tuple(range(0, 16))       # "varying the LM term from 0 to 15 dB [15]"
    payload_bytes: int = 1                       # "we normalized the calculation to 1 byte"
    variant: str = "corrected"                   # Algorithm 1 variant
    rule: str = "threshold"                      # "threshold" | "literal"
    clamp_tp: bool = False
    # Algorithm 1 is RSSI-referenced, so the noise floor must be too (see data_loading).
    noise_column: str = "noise_power_rssi"
    # How the conventional ADR's "20 SNR samples" window is formed (A17).
    # How the conventional ADR's "20 SNR samples" are collected (A17).  The paper says only
    # "the NS collects 20 SNR samples and obtains the maximum".
    #   test_order     - per device, rows in the order they appear in the frame (shuffled test set)
    #   chronological  - per device, sorted by timestamp
    #   per_device_max - the running maximum ever seen for that device
    #   expanding      - per device, expanding maximum in chronological order
    #   per_device_sf  - per (device, SF), rolling window, chronological
    #   blocks         - per device, non-overlapping blocks of `adr_window` rows
    adr_window_order: str = "test_order"
    adr_window: int = 20
    # "the NS collects 20 SNR samples and obtains the maximum": the packet being decided is not
    # yet among them.  True = maximum over the 20 previous samples (gives the paper's 4.8 % at
    # LM = 0 = 1/21); False = window includes the current sample (PDR = 0 at LM = 0).
    adr_window_excl_current: bool = False
    # How the conventional ADR moves SF: "both" (up when Me<0, down when Me>0), "down_only"
    # (TTN: ADR never raises SF; the device's own backoff does), "fixed" (TP only).
    adr_sf_mode: str = "both"
    adr_tp_step: float | None = None     # TTN integer steps (3 dB); None = continuous
    en_tp_quant: float | None = None     # EN transmit power granularity (US915 TXPower table: 2 dB), ceil
    energy_over: str = "all"             # "all" packets or "delivered" only, for the improvement curves
    en_sf_up: bool = True                # False: the EN never raises SF above its logged value (TP maxes out instead)
    params: AdrParameters = AdrParameters()


def _delivered(rssi_logged, tp_new, tp_logged, sf_new, noise_power, rule, cpls_pred=None,
               pl_true=None, params=AdrParameters(), lm=None):
    if rule == "threshold":
        rssi_at_new_tp = np.asarray(rssi_logged, float) + (np.asarray(tp_new, float)
                                                           - np.asarray(tp_logged, float))
        return rssi_at_new_tp > (np.asarray(noise_power, float) + snr_limit_of(sf_new))
    if rule == "literal":
        # actual RSSI > estimated RSSI  <=>  PL_predicted > PL_true
        return np.asarray(cpls_pred, float) > np.asarray(pl_true, float)
    if rule == "residual":
        # "actual RSSI greater than the predicted RSSI" with the predicted RSSI carrying the
        # link margin:  rssi_meas > rssi_pred(20 dBm) - LM  <=>  PL_true - PL_pred < LM.
        # No demodulation limit is consulted: SF/TP set energy and ToA, not delivery.
        return (np.asarray(pl_true, float) - np.asarray(cpls_pred, float)) < lm
    raise ValueError(rule)


def simulate_enhanced(df: pd.DataFrame, cpls_pred: np.ndarray, cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    """Sweep LM for one CPLS model driving Algorithm 1."""
    pm = PowerModel()
    tp_logged = df["ptx"].to_numpy(float)
    rssi = df["rssi"].to_numpy(float)
    npow = df[cfg.noise_column].to_numpy(float)
    sf0 = df["sf"].to_numpy(float)
    pl_true = df["experimental_pl"].to_numpy(float)
    # per-node link-budget terms (see enhanced_adr_vectorized docstring)
    lb = {c: df[c].to_numpy(float) for c in ("ltx", "gtx", "lrx", "grx")}

    rows = []
    for lm in cfg.lm_values:
        tp, sf, me, _ = enhanced_adr_vectorized(cpls_pred, tp_logged, sf0, lm, npow,
                                                cfg.params, cfg.variant, cfg.clamp_tp,
                                                link_budget=lb)
        if not cfg.en_sf_up:
            raised = sf > sf0
            sf = np.where(raised, sf0, sf).astype(int); tp = np.where(raised, cfg.params.max_tp, tp)
        if cfg.en_tp_quant:
            tp = np.minimum(cfg.params.max_tp, cfg.en_tp_quant * np.ceil(tp / cfg.en_tp_quant))
        ok = _delivered(rssi, tp, tp_logged, sf, npow, cfg.rule, cpls_pred, pl_true, cfg.params, lm=lm)
        toa = time_on_air(sf, cfg.payload_bytes)
        e = pm.energy_j(tp, toa)
        rows.append(dict(LM=lm, pdr=float(ok.mean() * 100), mean_tp_dbm=float(tp.mean()),
                         mean_sf=float(sf.mean()), mean_toa_s=float(toa.mean()),
                         mean_energy_j=float(e.mean()),
                         mean_toa_delivered_s=float(toa[ok].mean()) if ok.any() else np.nan,
                         mean_energy_delivered_j=float(e[ok].mean()) if ok.any() else np.nan,
                         frac_tp_clamped_high=float((tp >= cfg.params.max_tp).mean()),
                         frac_tp_clamped_low=float((tp <= cfg.params.min_tp).mean())))
    return pd.DataFrame(rows)


def simulate_conventional(df: pd.DataFrame, cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    """Sweep LM for the conventional TTN ADR of Section IV, quoted verbatim by the paper:

      1) the NS collects 20 SNR samples and obtains the maximum
      2) Me = SNRmax - SNRlimit - LM
      3) it varies the SF and decreases PT as needed to get Me = 0

    `cfg.adr_window_order` selects how the 20-sample window is formed (A17):

      "test_order"    - the rows are taken per device in the order they appear in
                        `df`.  For the shuffled test set this is a random draw from
                        that device's whole campaign, which is what an EN-simulator
                        iterating over the test subset produces.  Reproduces the
                        paper's Fig. 11 ADR curve to within ~1 point up to LM = 5.
      "chronological" - rows sorted by device then timestamp.  Consecutive logged
                        samples are one minute apart and strongly autocorrelated, so
                        the rolling max sits only ~1.2 dB above the current sample
                        and the ADR looks far better than the paper reports.
    """
    pm = PowerModel()
    chrono = cfg.adr_window_order in ("chronological", "per_device_max", "expanding", "per_device_sf")
    d = df.sort_values(["device_id", "timestamp"], kind="mergesort") if chrono else df
    if cfg.adr_window_order == "per_device_sf":
        d = df.sort_values(["device_id", "sf", "timestamp"], kind="mergesort")
    tp_logged = d["ptx"].to_numpy(float)
    rssi = d["rssi"].to_numpy(float)
    npow = d[cfg.noise_column].to_numpy(float)
    sf0 = d["sf"].to_numpy(float)
    snr = d["snr"].to_numpy(float)
    dev = d["device_id"].to_numpy()

    # SNRmax does not depend on LM, so build it once
    ser = pd.Series(snr)
    grp = pd.Series(dev)

    def _rolling_max(x):
        m = x.rolling(cfg.adr_window, min_periods=1).max()
        return m.shift(1).fillna(x) if cfg.adr_window_excl_current else m
    if cfg.adr_window_order == "per_device_max":
        snr_max = ser.groupby(grp).transform("max").to_numpy()
    elif cfg.adr_window_order == "expanding":
        snr_max = ser.groupby(grp).transform(lambda x: x.expanding().max()).to_numpy()
    elif cfg.adr_window_order == "blocks":
        blk = grp.astype(str) + "_" + (ser.groupby(grp).cumcount() // cfg.adr_window).astype(str)
        snr_max = ser.groupby(blk).transform("max").to_numpy()
    elif cfg.adr_window_order == "per_device_sf":
        key = grp.astype(str) + "_" + pd.Series(sf0).astype(int).astype(str)
        snr_max = ser.groupby(key).transform(_rolling_max).to_numpy()
    else:                                     # test_order / chronological
        snr_max = ser.groupby(grp).transform(_rolling_max).to_numpy()

    rows = []
    for lm in cfg.lm_values:
        tp, sf = _adr_from_snrmax(snr_max, sf0, lm, cfg.params, cfg.adr_sf_mode, cfg.adr_tp_step)
        if cfg.rule == "residual":
            ok = (snr_max - snr) < lm          # the NS's SNR estimate minus the margin vs the real SNR
        else:
            ok = _delivered(rssi, tp, tp_logged, sf, npow, "threshold")
        toa = time_on_air(sf, cfg.payload_bytes)
        e = pm.energy_j(tp, toa)
        rows.append(dict(LM=lm, pdr=float(ok.mean() * 100), mean_tp_dbm=float(tp.mean()),
                         mean_sf=float(sf.mean()), mean_toa_s=float(toa.mean()),
                         mean_energy_j=float(e.mean()),
                         mean_toa_delivered_s=float(toa[ok].mean()) if ok.any() else np.nan,
                         mean_energy_delivered_j=float(e[ok].mean()) if ok.any() else np.nan,
                         frac_tp_clamped_high=float((tp >= cfg.params.max_tp).mean()),
                         frac_tp_clamped_low=float((tp <= cfg.params.min_tp).mean())))
    return pd.DataFrame(rows)


def _adr_from_snrmax(snr_max, sf0, lm, params, sf_mode="both", tp_step=None):
    """Steps 2-3 of the TTN rule, given a precomputed SNRmax per row."""
    sf = np.asarray(sf0, float).copy()
    me = snr_max - snr_limit_of(sf) - lm
    if tp_step:
        # vectorised TTN nStep rule (integer steps; the final run uses the continuous rule)
        nstep = np.floor(me / tp_step).astype(int)
        tp = np.full(len(sf), 20.0)
        for _ in range(params.max_sf - params.min_sf):
            m = (nstep > 0) & (sf > params.min_sf) & (sf_mode != "fixed")
            sf[m] -= 1; nstep[m] -= 1
        for _ in range(int((20.0 - params.min_tp) / tp_step) + 1):
            m = (nstep > 0) & (tp - tp_step >= params.min_tp)
            tp[m] -= tp_step; nstep[m] -= 1
        for _ in range(int((params.max_tp - 20.0) / tp_step) + 1):
            m = (nstep < 0) & (tp + tp_step <= params.max_tp)
            tp[m] += tp_step; nstep[m] += 1
        for _ in range(params.max_sf - params.min_sf):
            m = (nstep < 0) & (sf < params.max_sf) & (sf_mode == "both")
            sf[m] += 1; nstep[m] += 1
        return tp, sf.astype(int)
    if sf_mode == "fixed":
        tp = np.clip(20.0 - np.maximum(me, 0.0), params.min_tp, params.max_tp)
        return tp, sf.astype(int)
    for _ in range(params.max_sf - params.min_sf + 1):
        up = (me < 0) & (sf < params.max_sf) & (sf_mode in ("both", "up_only"))
        if not up.any():
            break
        sf[up] += 1
        me[up] = snr_max[up] - snr_limit_of(sf[up]) - lm
    for _ in range(params.max_sf - params.min_sf + 1):
        down = (me > 0) & (sf > params.min_sf) & (sf_mode != "up_only")
        if not down.any():
            break
        cand = np.where(down, sf - 1, sf)
        cand_me = snr_max - snr_limit_of(cand) - lm
        ok = down & (cand_me > 0)
        if not ok.any():
            break
        sf[ok] = cand[ok]; me[ok] = cand_me[ok]
    tp = np.clip(20.0 - np.maximum(me, 0.0), params.min_tp, params.max_tp)
    return tp, sf.astype(int)


def lm_for_pdr(curve: pd.DataFrame, target_pct: float) -> float:
    """Smallest integer LM in the sweep whose PDR reaches `target_pct`."""
    hit = curve.loc[curve["pdr"] >= target_pct, "LM"]
    return float(hit.min()) if len(hit) else np.nan


def lm_table(curves: dict[str, pd.DataFrame], targets=(80, 85, 90, 95, 99)) -> pd.DataFrame:
    return pd.DataFrame([{"scheme": k, **{f"LM@{t}%": lm_for_pdr(v, t) for t in targets}}
                         for k, v in curves.items()])

