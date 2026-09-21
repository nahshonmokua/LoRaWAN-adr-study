"""Stage 5 - the paper's enhanced ADR algorithm (Algorithm 1), plus conventional TTN ADR.

`enhanced_adr` is a line-for-line transcription of Algorithm 1 as printed on page
10735.  Two documented defects in the printed listing are preserved in the
`as_printed` variant and repaired in the `corrected` variant:

  * line 4 subtracts `ltx` twice and never uses `lrx`, contradicting eq. (1).
  * line 28 clamps `set_point_tp` only from BELOW (`>= min_tp`).  When the
    decrease branch breaks out at lines 23-26, line 27's `margin_excess` is the
    (negative) value computed at the decremented SF, so line 29 *raises* the
    transmission power with no upper bound.  Measured effect: 12-18 % of rows end
    above max_tp, by at most one SNR_limit step (2.5 dB).
  * lines 10 and 22 re-evaluate `margin_excess` with the identical right-hand side
    of line 5, and `snr_limit` is declared a scalar *parameter* (line 2) rather
    than a function of the spreading factor.  As printed, neither while-loop can
    ever change `margin_excess`, so the loops run to their bounds unconditionally:
    the increase branch always lands on sf_max and the decrease branch on sf_min.
    The surrounding prose ("we decrease the SF as needed to guarantee that the Me
    is greater than zero") only makes sense with snr_limit = snr_limit(current_sf).

Both variants are exposed so the reproduction can quantify the difference.  A third, `text`,
is `corrected` plus the two points where the listing contradicts Section IV-A's description:
the two scenarios are exclusive (line 18 is printed as a second `if`, so both blocks run), and
TP is set from the margin at the SF actually used (the decrease branch's `break` leaves
margin_excess at the decremented SF, which makes line 27 raise power above max_tp).  `text` is
the variant of the final run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Section IV: SF 7 -> -7.5 dB ... SF 12 -> -20 dB, steps of -2.5 dB
SNR_LIMIT_BY_SF: dict[int, float] = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


@dataclass(frozen=True)
class AdrParameters:
    """Algorithm 1, line 2: `parameters: min_sf, max_sf, min_tp, max_tp, ltx, gtx,
    lrx, grx, snr_limit`.  min_tp/max_tp are never printed in the paper (A14)."""
    min_sf: int = 7
    max_sf: int = 12
    min_tp: float = 2.0
    max_tp: float = 20.0
    ltx: float = 1.0
    gtx: float = 2.9955
    lrx: float = 4.25
    grx: float = 4.16094
    snr_limit: float = -7.5          # the line-2 scalar, used by the `as_printed` variant


def snr_limit_of(sf) -> np.ndarray | float:
    """SNR_limit(SF) per Section IV.  Linear in SF so it also accepts arrays."""
    return -7.5 - 2.5 * (np.asarray(sf, float) - 7.0)


# --------------------------------------------------------------- scalar form
def enhanced_adr(d, f, T, RH, BP, PM, SNR, pl_model, current_tp, current_sf, LM,
                 noise_power, params: AdrParameters = AdrParameters(),
                 variant: str = "corrected"):
    """Algorithm 1.  Signature matches line 1 of the printed listing.

    Parameters
    ----------
    pl_model : callable(d, f, T, RH, BP, PM, SNR) -> float
        The `cpls_model` of line 3.  Any of the Stage 2/3 models can be passed.
    variant : {"as_printed", "corrected"}
        See the module docstring.

    Returns
    -------
    (current_tp, current_sf, margin_excess, adjusted_sf)
        The first two are Algorithm 1's line-32 outputs; the last two are exposed
        for the Stage 6 simulation.
    """
    p = params
    if variant not in ("as_printed", "corrected", "text"):
        raise ValueError(variant)

    def snr_limit() -> float:
        # `as_printed`: the line-2 scalar parameter, independent of current_sf.
        # `corrected`: recomputed from the SF the algorithm is currently holding.
        return p.snr_limit if variant == "as_printed" else float(snr_limit_of(current_sf))

    # 3: estimated_cpls <- cpls_model(d, f, T, RH, BP, PM, SNR)
    estimated_cpls = float(pl_model(d, f, T, RH, BP, PM, SNR))

    # 4: estimated_rssi <- current_tp - ltx + gtx - estimated_cpls + grx - ltx
    #    (`- ltx` twice is verbatim from the paper; the corrected variant uses lrx)
    tail = p.ltx if variant == "as_printed" else p.lrx
    estimated_rssi = current_tp - p.ltx + p.gtx - estimated_cpls + p.grx - tail

    # 5: margin_excess <- estimated_rssi - (noise_power + snr_limit + LM)
    margin_excess = estimated_rssi - (noise_power + snr_limit() + LM)

    adjusted_sf = False
    took_increase = margin_excess < 0
    if margin_excess < 0:                                            # 6
        adjusted_sf = False                                          # 7
        while margin_excess < 0 and current_sf < p.max_sf:           # 8
            current_sf = current_sf + 1                              # 9
            margin_excess = estimated_rssi - (noise_power + snr_limit() + LM)   # 10
            if margin_excess > 0:                                    # 11
                adjusted_sf = True                                   # 12
        set_point_tp = current_tp - margin_excess                    # 13
        if set_point_tp <= p.max_tp:                                 # 14
            current_tp = set_point_tp                                # 15
        else:                                                        # 16
            current_tp = p.max_tp                                    # 17

    # "text": Section IV-A describes lines 6-17 and 18-31 as two exclusive scenarios; as printed,
    # line 18 is a second `if` and both blocks run whenever the first one restores the margin.
    if margin_excess >= 0 and not (took_increase and variant == "text"):   # 18
        adjusted_sf = False                                          # 19
        while margin_excess > 0 and current_sf > p.min_sf:           # 20
            current_sf = current_sf - 1                              # 21
            margin_excess = estimated_rssi - (noise_power + snr_limit() + LM)   # 22
            if margin_excess <= 0:                                   # 23
                current_sf = current_sf + 1                          # 24
                adjusted_sf = True                                   # 25
                if variant == "text":
                    # "decrease the SF and transmission power": TP is set from the margin at
                    # the SF actually used.  As printed, line 27 sees the margin of the
                    # decremented SF (negative), which RAISES TP above max_tp.
                    margin_excess = estimated_rssi - (noise_power + snr_limit() + LM)
                break                                                # 26
        set_point_tp = current_tp - margin_excess                    # 27
        if set_point_tp >= p.min_tp:                                 # 28
            current_tp = set_point_tp                                # 29
        else:                                                        # 30
            current_tp = p.min_tp                                    # 31

    return current_tp, current_sf, margin_excess, adjusted_sf        # 32


# ----------------------------------------------------------- vectorised form
def enhanced_adr_vectorized(estimated_cpls, current_tp, current_sf, LM, noise_power,
                            params: AdrParameters = AdrParameters(),
                            variant: str = "corrected", clamp_tp: bool = False,
                            link_budget: dict | None = None):
    """Same control flow as `enhanced_adr`, evaluated over whole arrays.

    `clamp_tp=True` additionally clips the output to [min_tp, max_tp], repairing the
    line-28 defect described in the module docstring.  Default False = faithful.

    `link_budget` optionally supplies PER-ROW ltx/gtx/lrx/grx arrays.  Algorithm 1
    declares these as parameters because they are "fixed once the node is deployed",
    but they differ BETWEEN nodes: in the released data ltx is 1 or 11.75 dB and gtx
    is 2.90-8.54 dBi.  Using one node's values for all four ENs puts a ~10 dB error
    into line 4, so the simulation always passes the per-row columns.

    Takes the CPLS prediction directly (line 3 is the expensive part and is done
    once per model by the caller).  Verified against `enhanced_adr` row by row in
    tests/test_adr_equivalence.py.
    """
    p = params
    cpls = np.asarray(estimated_cpls, float)
    tp = np.asarray(current_tp, float).astype(float) * np.ones_like(cpls)
    sf = np.asarray(current_sf, float).astype(float) * np.ones_like(cpls)
    npow = np.asarray(noise_power, float) * np.ones_like(cpls)

    lb = link_budget or {}
    ltx = np.asarray(lb.get("ltx", p.ltx), float) * np.ones_like(cpls)
    gtx = np.asarray(lb.get("gtx", p.gtx), float) * np.ones_like(cpls)
    lrx = np.asarray(lb.get("lrx", p.lrx), float) * np.ones_like(cpls)
    grx = np.asarray(lb.get("grx", p.grx), float) * np.ones_like(cpls)
    tail = ltx if variant == "as_printed" else lrx      # line 4 subtracts ltx twice as printed
    rssi = tp - ltx + gtx - cpls + grx - tail

    def sl(sf_arr):
        return np.full_like(sf_arr, p.snr_limit) if variant == "as_printed" else snr_limit_of(sf_arr)

    me = rssi - (npow + sl(sf) + LM)
    adjusted = np.zeros_like(cpls, dtype=bool)

    # ---- lines 6-17: increase branch
    inc = me < 0
    if inc.any():
        # while margin_excess < 0 and current_sf < max_sf  (at most max_sf-min_sf steps)
        for _ in range(p.max_sf - p.min_sf + 1):
            step = inc & (me < 0) & (sf < p.max_sf)
            if not step.any():
                break
            sf[step] += 1
            me[step] = rssi[step] - (npow[step] + sl(sf[step]) + LM)
            adjusted |= step & (me > 0)
        set_point = tp - me
        tp = np.where(inc, np.where(set_point <= p.max_tp, set_point, p.max_tp), tp)

    # ---- lines 18-31: decrease branch ("text": exclusive with the increase branch)
    dec = (me >= 0) & ~inc if variant == "text" else (me >= 0)
    if dec.any():
        adjusted[dec] = False                        # line 19
        broke = np.zeros_like(dec)
        for _ in range(p.max_sf - p.min_sf + 1):
            step = dec & ~broke & (me > 0) & (sf > p.min_sf)
            if not step.any():
                break
            sf[step] -= 1
            me[step] = rssi[step] - (npow[step] + sl(sf[step]) + LM)
            # lines 23-26: restore the SF but KEEP the margin_excess computed at the
            # decremented SF - line 27 then uses that (negative) value, exactly as printed.
            hit = step & (me <= 0)
            sf[hit] += 1
            if variant == "text":                    # margin at the SF actually used
                me[hit] = rssi[hit] - (npow[hit] + sl(sf[hit]) + LM)
            adjusted |= hit
            broke |= hit
        set_point = tp - me
        tp = np.where(dec, np.where(set_point >= p.min_tp, set_point, p.min_tp), tp)

    if clamp_tp:
        tp = np.clip(tp, p.min_tp, p.max_tp)
    return tp, sf.astype(int), me, adjusted


# ------------------------------------------------- conventional (TTN) ADR
def conventional_adr(snr_series, sf_series, LM, current_tp=20.0,
                     params: AdrParameters = AdrParameters(), window: int = 20):
    """The TTN ADR of Section IV, quoted verbatim by the paper:

      1) the NS collects 20 SNR samples and obtains the maximum
      2) Me = SNRmax - SNRlimit - LM
      3) it varies the SF and decreases PT as needed to get Me = 0

    `snr_series` / `sf_series` must already be in per-device chronological order.
    The rolling maximum is taken over the previous `window` samples (A17).
    """
    import pandas as pd

    snr = pd.Series(np.asarray(snr_series, float))
    snr_max = snr.rolling(window, min_periods=1).max().to_numpy()
    sf = np.asarray(sf_series, float).astype(float).copy()
    tp = np.full(len(snr), float(current_tp))

    me = snr_max - snr_limit_of(sf) - LM
    # step 3: raise SF while the link cannot close, lower it while there is slack
    for _ in range(params.max_sf - params.min_sf + 1):
        up = (me < 0) & (sf < params.max_sf)
        if not up.any():
            break
        sf[up] += 1
        me[up] = snr_max[up] - snr_limit_of(sf[up]) - LM
    for _ in range(params.max_sf - params.min_sf + 1):
        down = (me > 0) & (sf > params.min_sf)
        if not down.any():
            break
        cand_sf = np.where(down, sf - 1, sf)
        cand_me = snr_max - snr_limit_of(cand_sf) - LM
        ok = down & (cand_me > 0)
        if not ok.any():
            break
        sf[ok] = cand_sf[ok]
        me[ok] = cand_me[ok]

    # decrease PT to drive the remaining excess to zero
    tp = np.clip(tp - np.maximum(me, 0.0), params.min_tp, params.max_tp)
    return tp, sf.astype(int), me
