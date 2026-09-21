"""Closed-loop conventional (TTN) ADR, as Section IV of the paper describes it.

The paper names three drawbacks of the conventional ADR that its simulation captures:
  1) the LM must be over-estimated in varying channels;
  2) if the sample rate is low the SNR window does not track the channel;
  3) "if the NS does not receive measurement data from an EN (a packet drop), the
     corresponding SNR measurement will not be considered".
(2) and (3) only exist in a STATEFUL simulation: the NS cuts TP, later packets arrive with
lower SNR, dropped packets never enter the window, so the window is survivorship-biased
upward and the link is held at the edge.  An open-loop sweep over logged SNRs (all measured
at 20 dBm) cannot show either effect.  This module is the stateful version.

Per device, iterating the test rows in the order they appear:
    snr_actual = snr_logged + (tp - 20)                 # logged at 20 dBm
    delivered  = snr_actual > snr_limit(sf)
    if delivered: window.append(snr_actual)             # drops are not observed by the NS
    every `update_every` received packets, once the window holds `window` samples:
        Me = max(window) - snr_limit(sf) - LM
        lower SF while Me stays >= 0; then tp -= Me (clipped to [min_tp, max_tp]);
        if Me < 0: raise tp (and, if allowed, SF) to cover it
Energy / ToA are taken at the (tp, sf) the packet was actually sent with.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from adr_algorithm import snr_limit_of
from energy import PowerModel, time_on_air


def run_device(snr_logged, sf_init, lm, *, window=20, update_every=1, min_tp=2.0, max_tp=20.0,
               min_sf=7, max_sf=12, allow_sf_up=True, reset_window_on_update=False, backoff=None,
               tp_step=None):
    """`backoff`: LoRaWAN device-side ADR backoff.  After `backoff` consecutive uplinks with no
    downlink (here: consecutive drops, since a received packet always yields an ADR command),
    the device sets TP to max; if already at max it raises SF one step.  LoRaWAN default is
    ADR_ACK_LIMIT + ADR_ACK_DELAY = 64 + 32 = 96.  None disables it (the pure NS loop)."""
    n = len(snr_logged)
    tp, sf = max_tp, int(sf_init)
    win = deque(maxlen=window)
    tps = np.empty(n); sfs = np.empty(n, int); ok = np.zeros(n, bool)
    since = 0; drops = 0
    for i in range(n):
        tps[i], sfs[i] = tp, sf
        s = snr_logged[i] + (tp - max_tp)
        if not (s > snr_limit_of(sf)):
            drops += 1
            if backoff and drops >= backoff:
                drops = 0
                if tp < max_tp: tp = max_tp
                elif sf < max_sf: sf += 1
                win.clear()                       # the NS's stale window no longer applies
            continue
        drops = 0
        if True:
            ok[i] = True
            win.append(s)
            since += 1
            if len(win) >= window and since >= update_every:
                since = 0
                me = max(win) - snr_limit_of(sf) - lm
                if tp_step:
                    # TTN reference ADR: nStep = floor(Me / 3); each positive step first raises the
                    # data rate (SF - 1), then cuts TP by 3 dB; negative steps raise TP by 3 dB only.
                    nstep = int(np.floor(me / tp_step))
                    while nstep > 0 and sf > min_sf:
                        sf -= 1; nstep -= 1
                    while nstep > 0 and tp - tp_step >= min_tp:
                        tp -= tp_step; nstep -= 1
                    while nstep < 0 and tp + tp_step <= max_tp:
                        tp += tp_step; nstep += 1
                    while allow_sf_up and nstep < 0 and sf < max_sf:
                        sf += 1; nstep += 1
                elif me >= 0:
                    while sf > min_sf and me - 2.5 >= 0:      # lowering SF raises snr_limit by 2.5
                        sf -= 1; me -= 2.5
                    tp = max(min_tp, tp - me)
                else:
                    need = -me
                    room = max_tp - tp
                    tp = min(max_tp, tp + need)
                    need -= room
                    while allow_sf_up and need > 0 and sf < max_sf:
                        sf += 1; need -= 2.5
                if reset_window_on_update:
                    win.clear()
    return tps, sfs, ok


def simulate(te: pd.DataFrame, lm_values, *, sf_init="logged", **kw) -> pd.DataFrame:
    """PDR / energy / ToA of the closed-loop ADR over an LM sweep, per-device stateful."""
    pm = PowerModel()
    dev = te["device_id"].to_numpy()
    snr = te["snr"].to_numpy(float)
    sf_log = te["sf"].to_numpy(int)
    rows = []
    for lm in lm_values:
        tp_all = np.empty(len(te)); sf_all = np.empty(len(te), int); ok_all = np.zeros(len(te), bool)
        for d in pd.unique(dev):
            m = np.flatnonzero(dev == d)
            s0 = sf_log[m[0]] if sf_init == "logged" else int(sf_init)
            t, s, o = run_device(snr[m], s0, lm, **kw)
            tp_all[m], sf_all[m], ok_all[m] = t, s, o
        toa = time_on_air(sf_all, 1)
        e = pm.energy_j(tp_all, toa)
        rows.append(dict(LM=float(lm), pdr=float(ok_all.mean() * 100), mean_tp_dbm=float(tp_all.mean()),
                         mean_sf=float(sf_all.mean()), mean_toa_s=float(toa.mean()),
                         mean_energy_j=float(e.mean())))
    return pd.DataFrame(rows)
