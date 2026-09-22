"""Every figure of the reproduction, drawn from outputs/final/*.csv only.

Used by run_final.py (saves PNGs) and by notebooks/figures.ipynb (shows them inline), so there is
one drawing of each figure.  Each function returns a matplotlib Figure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "outputs" / "final"
PAPER_DIG = ROOT / "data" / "paper_digitized"
STYLE = {"ADR": ("0.45", "s"), "ANN": ("#E8A33D", "o"), "Friis": ("#5BC0EB", "^"), "MLR": ("#2E9E5B", "o"),
         "RF": ("#E03C31", "x"), "SVR": ("#D6409F", "*"), "SPLMSF": ("#2B6CB0", "D"), "SPLMSFT": ("#D2601A", "v")}
DIG = {"ANN": "ANN", "SVR": "SVR", "RF": "RF", "MLR": "MLR", "FRIIS": "Friis", "SPLMSF": "SPLMSF", "SPLMSFT": "SPLMSFT", "ADR": "ADR"}
ORDER = ("ANN", "SVR", "RF", "MLR", "Friis", "SPLMSFT", "SPLMSF")


def _dig(name):
    p = PAPER_DIG / name
    return pd.read_csv(p, index_col=0) if p.exists() else None


def _hollow(ax, x, y, key, s):
    """The paper's digitized points: hollow markers (unfilled marker styles take a plain colour)."""
    col, m = STYLE[key]
    if m == "x": ax.scatter(x, y, marker=m, s=s, color=col, lw=1.3, zorder=6)
    else: ax.scatter(x, y, marker=m, s=s, facecolor="none" if m != "*" else col, edgecolor=col, lw=1.3, zorder=6)


def fig04(final: Path = FINAL):
    """Fig. 4 - conventional models against the measured path loss, per distance."""
    d = pd.read_csv(final / "fig04_data.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for r in d.itertuples(): ax.vlines(r.distance_km, r.pl_min, r.pl_max, color="0.75", lw=6)
    for col, lab, c in (("pl_mean", "Av. PL (measured)", "#F8766D"), ("friis", "Friis", "#7CAE00"), ("splmsf", "SPLMSF", "#00BFC4"), ("two_ray", "Two-ray", "#C77CFF")):
        ax.scatter(d.distance_km, d[col], s=90, c=c, zorder=5, label=lab)
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Path Loss (dB)"); ax.set_title("Fig. 4 - conventional models")
    ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); return fig


def fig11(final: Path = FINAL, curves: str = "fig11_curves.csv", title: str = "Fig. 11 - PDR vs link margin"):
    """Fig. 11 - PDR vs LM for every scheme; hollow markers are the paper's curve, digitized."""
    c = pd.read_csv(final / curves); d11 = _dig("paper_fig11_digitized.csv")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for k in ("ADR", "Friis", "SPLMSF", "SPLMSFT", "MLR", "ANN", "SVR", "RF"):
        g = c[c.scheme == k]; ax.plot(g.LM, g.pdr, color=STYLE[k][0], lw=1.6, label=k)
    if d11 is not None:
        for pk, ok in DIG.items(): _hollow(ax, d11.index, d11[pk].clip(upper=100).values, ok, 34)
    ax.set_xlabel("LM (dB)"); ax.set_ylabel("PDR (%)"); ax.set_ylim(0, 103); ax.grid(alpha=.3)
    ax.set_title(f"{title}   (lines: reproduced, hollow markers: paper, digitized)")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); return fig


def _improvement(final, table, col, ylab, title, digf=None, levels=(80, 85, 90, 95, 99)):
    E = pd.read_csv(final / table); dg = _dig(digf) if digf else None
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for k in ORDER:
        g = E[E.scheme == k]
        if len(g): ax.plot(g.pdr, g[col], marker=STYLE[k][1], color=STYLE[k][0], lw=1.6, label=k)
    if dg is not None:
        for pk, ok in DIG.items():
            if ok != "ADR": _hollow(ax, dg.index, dg[pk].values, ok, 60)
    ax.axhline(0, color="0.45", ls="--"); ax.set_xlabel("PDR (%)"); ax.set_ylabel(ylab); ax.grid(alpha=.3); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); return fig


def fig12(final: Path = FINAL):
    """Fig. 12 - ToA improvement vs the conventional ADR at the paper's operating LMs."""
    return _improvement(final, "fig12_13_energy_toa.csv", "toa_improvement_pct", "ToA improvement vs ADR (%)",
                        "Fig. 12 - lines: reproduced (paper's operating LMs), hollow markers: paper (digitized)", "paper_fig12_digitized.csv")


def fig13(final: Path = FINAL):
    """Fig. 13 - energy improvement vs the conventional ADR at the paper's operating LMs."""
    return _improvement(final, "fig12_13_energy_toa.csv", "energy_improvement_pct", "Energy improvement vs ADR (%)",
                        "Fig. 13 - lines: reproduced (paper's operating LMs), hollow markers: paper (digitized)", "paper_fig13_digitized.csv")


def fig13_matched(final: Path = FINAL):
    """Energy improvement at EQUAL achieved PDR: (a) paper's SF 7-12, residual rule; (b) SF <= 10, residual rule
    (energy only - the rule is blind to SF/TP); (c) SF <= 10, receiver-threshold rule (95/99 % unreachable)."""
    sens = pd.read_csv(final / "fig12_13_reading_sensitivity.csv"); g = sens[sens.reading == "grid"]
    panels = [(g, "(a) SF 7-12, residual rule, matched PDR")]
    for f_, t_ in (("fig12_13_energy_toa_sf10.csv", "(b) SF <= 10, residual rule (PDR unchanged by construction)"),
                   ("fig12_13_energy_toa_threshold_sf10.csv", "(c) SF <= 10, threshold rule, matched achieved PDR")):
        if (final / f_).exists(): panels.append((pd.read_csv(final / f_), t_))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), sharey=True)
    for ax, (E, t_) in zip(np.atleast_1d(axes), panels):
        for k in ORDER:
            h = E[E.scheme == k]
            if len(h): ax.plot(h.pdr, h.energy_improvement_pct, marker=STYLE[k][1], color=STYLE[k][0], lw=1.5, label=k)
        ax.axhline(0, color="0.45", ls="--"); ax.set_xlabel("PDR (%)"); ax.set_title(t_, fontsize=9.5); ax.grid(alpha=.3); ax.set_xlim(78, 100)
    np.atleast_1d(axes)[0].set_ylabel("Energy improvement vs ADR (%)"); np.atleast_1d(axes)[0].legend(fontsize=7.5)
    fig.tight_layout(); return fig


def fig14_15(final: Path = FINAL):
    """Figs. 14-15 - QQ plots of the MLR residuals against a normal and a Student-t."""
    q = pd.read_csv(final / "fig14_15_qq.csv"); meta = pd.read_csv(final / "appendix_residual_tests.csv").set_index("test")
    nu = float(meta.loc["Student-t nu (MLE)", "statistic"]); r2t = float(meta.loc["QQ R2 vs t", "statistic"]); r2n = float(meta.loc["QQ R2 vs normal", "statistic"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, (th, em, r2, ttl) in zip(ax, [(q.theoretical_normal, q.empirical, r2n, "Fig. 14 - QQ vs Normal"),
                                         (q.theoretical_t, q.empirical, r2t, f"Fig. 15 - QQ vs Student-t, $\\nu$={nu:.2f}")]):
        a.plot(th, em, ".", ms=3); lim = [float(th.min()), float(th.max())]
        a.plot(lim, lim, "r-", lw=1); a.set_title(f"{ttl}  (R$^2$={r2:.4f})"); a.grid(alpha=.3); a.set_xlabel("theoretical quantiles (dB)")
    ax[0].set_ylabel("residual quantiles (dB)"); fig.tight_layout(); return fig


ALL = {"fig04": fig04, "fig11": fig11, "fig12": fig12, "fig13": fig13, "fig13_matched": fig13_matched, "fig14_15": fig14_15}


def save_all(final: Path = FINAL, figs: Path | None = None, dpi: int = 160) -> list[Path]:
    figs = figs or final / "figures"; figs.mkdir(parents=True, exist_ok=True); done = []
    for name, fn in ALL.items():
        try: f = fn(final); f.savefig(figs / f"{name}.png", dpi=dpi); plt.close(f); done.append(figs / f"{name}.png")
        except FileNotFoundError as e: print(f"  {name}: skipped ({e.filename} missing)")
    return done
