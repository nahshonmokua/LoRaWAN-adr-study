"""Stage 14 - reverse-engineer the paper's simulation configuration.

Rather than document the under-specification and stop, treat it as an inverse problem:
the paper publishes ~16 numbers; the configuration that produced them has ~6 unknown
discrete parameters.  Search the parameter space and find which setting reproduces the
published numbers best.

UNKNOWNS (none of these is printed in the paper)
    sf_max                 Algorithm 1 line 2
    min_tp, max_tp         Algorithm 1 line 2
    line-28 handling       as printed | repaired  (see adr_algorithm docstring)
    Algorithm 1 variant    corrected | as_printed
    ADR SNR window order   readings of "the NS collects 20 SNR samples"
    ADR window length      the paper says 20

TARGETS (all printed in the paper)
    Fig. 11  ADR   LM at 80/85/90/95/99 % PDR = 6, 7, 8, 9, 11 dB
    Fig. 11  Friis LM at 80/85/90/95/99 % PDR = 0, 1, 2, 3, 5 dB
    Fig. 11  ANN/SVR/RF LM at 95 % and 99 %    = 3, 4 dB
    Sec IV-C energy improvement at 99 % PDR    = 43.5, 40.6, 38.7 %
    Sec IV-C ToA improvement at 99 % PDR       = 32.7, 29.9, 27.5 %

Two grids:
  --grid full     the round-1 factorial (1 152 joint configurations)
  --grid refined  prunes what round 1 ruled out unanimously (line-28 as printed, the
                  as_printed variant, sf_max = 10 which cannot reach 99 % PDR) and
                  refines where the optimum sat (min_tp, window length)
Two data sources:
  --data clean             the released CSV as published
  --data reconstructed_C   Stage 11's pre-cleaning reconstruction (mode C)

CAVEAT: 6 parameters against 16 targets is constrained but not heavily so.  A good fit is
evidence about what they did, not proof.  Per-target residuals are reported so an
over-fitted solution is visible rather than hidden behind one aggregate score.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adr_algorithm import AdrParameters                                          # noqa: E402
from conventional_models import friis_pl                                         # noqa: E402
from data_loading import OUTPUTS                                                 # noqa: E402
from energy import improvement_pct                                               # noqa: E402
from simulation import SimConfig, simulate_conventional, simulate_enhanced       # noqa: E402
from splitting import split                                                      # noqa: E402

TABLES = OUTPUTS / "tables"
LMS = tuple(np.round(np.arange(0, 15.001, 0.25), 2))
LEVELS = (80, 85, 90, 95, 99)

TARGET_ADR = {80: 6.0, 85: 7.0, 90: 8.0, 95: 9.0, 99: 11.0}
TARGET_FRIIS = {80: 0.0, 85: 1.0, 90: 2.0, 95: 3.0, 99: 5.0}
TARGET_ML = {95: 3.0, 99: 4.0}
TARGET_E = {"ANN": 43.5, "SVR": 40.6, "RF": 38.7}
TARGET_T = {"ANN": 32.7, "SVR": 29.9, "RF": 27.5}

GRIDS = {
    "full": dict(SF_MAX=(10, 11, 12),
                 TP_BOUNDS=((2.0, 20.0), (2.0, 14.0), (7.0, 20.0), (7.0, 14.0)),
                 LINE28=(False, True), VARIANT=("corrected", "as_printed"),
                 ADR_ORDER=("test_order", "chronological", "per_device_max",
                            "expanding", "per_device_sf", "blocks"),
                 ADR_WINDOW=(10, 20, 50, 100)),
    "refined": dict(SF_MAX=(11, 12),
                    TP_BOUNDS=((2.0, 20.0), (5.0, 20.0), (7.0, 20.0), (10.0, 20.0)),
                    LINE28=(True,), VARIANT=("corrected",),
                    ADR_ORDER=("test_order", "blocks", "expanding", "per_device_max"),
                    ADR_WINDOW=(20, 50, 100, 200)),
}


def load_data(source: str):
    if source == "clean":
        # released CSV with the DHT22 fault rows dropped, models fitted on it (no restoration)
        import joblib
        from cpls_models import make_X
        from data_loading import load_cached
        raw = load_cached()
        df = raw[~(raw["rh"] <= 2.0)].reset_index(drop=True)
        tr, te = split(df)
        mdir = OUTPUTS / "final" / "models"
        preds = {k: np.asarray(joblib.load(mdir / f"cpls_{k.lower()}.joblib").predict(make_X(te)), float)
                 for k in ("MLR", "ANN", "SVR", "RF")}
        preds["Friis"] = np.asarray(friis_pl(te["distance"], te["frequency"]), float)
    else:
        sfx = source.replace("reconstructed", "")          # "_C"
        te = pd.read_pickle(OUTPUTS / f"reconstructed_test{sfx}.pkl")
        z = np.load(OUTPUTS / f"reconstructed_predictions{sfx}.npz")
        preds = {k: z[k] for k in z.files}
        preds["Friis"] = np.asarray(friis_pl(te["distance"], te["frequency"]), float)
    return te, {k: preds[k] for k in ("Friis", "MLR", "ANN", "SVR", "RF")}


def lm_at(curve, pdr):
    h = curve[curve.pdr >= pdr]
    return float(h.LM.iloc[0]) if len(h) else np.nan


def row_at(curve, pdr):
    h = curve[curve.pdr >= pdr]
    return h.iloc[0] if len(h) else None


def main(source: str, grid: str) -> None:
    G = GRIDS[grid]
    te, preds = load_data(source)
    tag = f"{source}_{grid}"
    print(f"data={source}  grid={grid}  test rows={len(te):,}", flush=True)

    enh = {}
    combos = list(product(G["SF_MAX"], G["TP_BOUNDS"], G["LINE28"], G["VARIANT"]))
    for i, (sfm, (lo, hi), l28, var) in enumerate(combos, 1):
        cfg = SimConfig(lm_values=LMS, clamp_tp=l28, variant=var,
                        params=AdrParameters(max_sf=sfm, min_tp=lo, max_tp=hi))
        enh[(sfm, lo, hi, l28, var)] = {k: simulate_enhanced(te, p, cfg) for k, p in preds.items()}
        print(f"  enhanced {i}/{len(combos)}", flush=True)

    adr = {}
    combos = list(product(G["ADR_ORDER"], G["ADR_WINDOW"], G["SF_MAX"], G["TP_BOUNDS"]))
    for i, (order, win, sfm, (lo, hi)) in enumerate(combos, 1):
        cfg = SimConfig(lm_values=LMS, adr_window_order=order, adr_window=win,
                        params=AdrParameters(max_sf=sfm, min_tp=lo, max_tp=hi))
        adr[(order, win, sfm, lo, hi)] = simulate_conventional(te, cfg)
        if i % 24 == 0 or i == len(combos):
            print(f"  conventional {i}/{len(combos)}", flush=True)

    rows, detail = [], []
    for (sfm, lo, hi, l28, var), curves in enh.items():
        e_friis = [abs(lm_at(curves["Friis"], p) - TARGET_FRIIS[p]) for p in LEVELS]
        e_ml = [abs(lm_at(curves[k], p) - TARGET_ML[p]) for k in ("ANN", "SVR", "RF") for p in (95, 99)]
        for (order, win, sfm2, lo2, hi2), a in adr.items():
            if (sfm2, lo2, hi2) != (sfm, lo, hi):
                continue
            e_adr = [abs(lm_at(a, p) - TARGET_ADR[p]) for p in LEVELS]
            ra = row_at(a, 99)
            e_en, e_toa, vals = [], [], {}
            for k in ("ANN", "SVR", "RF"):
                rk = row_at(curves[k], 99)
                if ra is None or rk is None:
                    e_en.append(np.nan); e_toa.append(np.nan)
                    vals[f"E_{k}"] = vals[f"T_{k}"] = np.nan
                    continue
                vals[f"E_{k}"] = improvement_pct(ra.mean_energy_j, rk.mean_energy_j)
                vals[f"T_{k}"] = improvement_pct(ra.mean_toa_s, rk.mean_toa_s)
                e_en.append(abs(vals[f"E_{k}"] - TARGET_E[k]))
                e_toa.append(abs(vals[f"T_{k}"] - TARGET_T[k]))
            mae = lambda v: float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan
            key = dict(sf_max=sfm, min_tp=lo, max_tp=hi,
                       line28="repaired" if l28 else "as printed", variant=var,
                       adr_order=order, adr_window=win)
            rows.append(dict(**key,
                             mae_adr_lm=mae(e_adr), mae_friis_lm=mae(e_friis), mae_ml_lm=mae(e_ml),
                             mae_energy_pp=mae(e_en), mae_toa_pp=mae(e_toa),
                             n_unreachable=int(np.sum(~np.isfinite(e_adr + e_friis + e_ml + e_en + e_toa)))))
            detail.append(dict(**key,
                               **{f"ADR_LM{p}": lm_at(a, p) for p in LEVELS},
                               **{f"Friis_LM{p}": lm_at(curves["Friis"], p) for p in LEVELS},
                               **{f"{k}_LM{p}": lm_at(curves[k], p) for k in ("ANN", "SVR", "RF") for p in (95, 99)},
                               **vals))
    res = pd.DataFrame(rows)
    res["score"] = (res.mae_adr_lm.fillna(9) + res.mae_friis_lm.fillna(9) + res.mae_ml_lm.fillna(9)
                    + res.mae_energy_pp.fillna(90) / 10 + res.mae_toa_pp.fillna(90) / 10)
    res = res.sort_values("score").reset_index(drop=True)
    res.to_csv(TABLES / f"stage14_inverse_{tag}.csv", index=False)
    det = pd.DataFrame(detail)
    det = det.set_index(list(res.columns[:7])).loc[
        [tuple(r) for r in res[res.columns[:7]].itertuples(index=False)]].reset_index()
    det.to_csv(TABLES / f"stage14_inverse_{tag}_detail.csv", index=False)

    pd.set_option("display.width", 260)
    print(f"\n=== {tag}: best 12 of {len(res)} (lower score = closer to the paper) ===")
    print(res.head(12).round(3).to_string(index=False))
    print("\n=== winner's actual values vs the paper ===")
    w = det.iloc[0]
    print("ADR   LM 80/85/90/95/99:", [round(w[f"ADR_LM{p}"], 2) for p in LEVELS], " paper [6,7,8,9,11]")
    print("Friis LM 80/85/90/95/99:", [round(w[f"Friis_LM{p}"], 2) for p in LEVELS], " paper [0,1,2,3,5]")
    for k in ("ANN", "SVR", "RF"):
        print(f"{k:5s} LM 95/99: [{w[f'{k}_LM95']:.2f}, {w[f'{k}_LM99']:.2f}]  paper [3, 4]   "
              f"energy {w[f'E_{k}']:.1f} (paper {TARGET_E[k]})   ToA {w[f'T_{k}']:.1f} (paper {TARGET_T[k]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["clean", "reconstructed_C"], default="clean")
    ap.add_argument("--grid", choices=list(GRIDS), default="full")
    a = ap.parse_args()
    main(a.data, a.grid)
