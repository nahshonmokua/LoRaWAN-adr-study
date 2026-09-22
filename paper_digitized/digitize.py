"""Digitize a ggplot-style figure: detect the grey panel and its white gridlines, map pixels
to data via the known tick values, classify curve pixels by colour, and read each series at
requested x positions as the median y of its pixels in a narrow column band."""
import numpy as np
from PIL import Image

SERIES = ["ADR", "ANN", "FRIIS", "MLR", "RF", "SPLMSF", "SPLMSFT", "SVR"]   # legend order


def panel_and_grid(a):
    """Panel bbox from the ggplot grey; gridlines = narrow white runs inside it.  Returns the
    MAJOR gridline centres (thicker runs) and all runs, so ticks can be matched to labels."""
    grey = (np.abs(a[:, :, 0] - 235) < 8) & (np.abs(a[:, :, 1] - 235) < 8) & (np.abs(a[:, :, 2] - 235) < 8)
    white = (a.min(axis=2) > 248)
    rows = np.flatnonzero(grey.mean(axis=1) > 0.15); cols = np.flatnonzero(grey.mean(axis=0) > 0.15)
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    iy, ix = slice(y0 + 12, y1 - 12), slice(x0 + 12, x1 - 12)          # interior only

    def runs(profile, offset, lo, hi):
        idx = np.flatnonzero(profile > 0.6)
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        out = []
        for g in groups:
            if not len(g): continue
            c, w = g.mean() + offset, g[-1] - g[0] + 1
            if w <= 10 and lo - 1 <= c <= hi + 1:      # a gridline, inside the panel proper
                out.append(c)
        return np.array(out)

    v = runs(white[iy, x0 - 4:x1 + 5].mean(axis=0), x0 - 4, x0, x1)
    h = runs(white[y0 - 4:y1 + 5, ix].mean(axis=1), y0 - 4, y0, y1)
    return (y0, y1, x0, x1), v, h


def calibrate(lines_px, first_tick, spacing, descending=False):
    """Equally spaced gridlines; the one at the axis minimum is `first_tick`.  For y the
    pixel order is reversed (pixel y grows downward while data grows upward)."""
    px = np.sort(lines_px)
    if descending: px = px[::-1]
    vals = first_tick + spacing * np.arange(len(px))
    return np.polyfit(vals, px, 1), len(px)


def legend_colors(a, box):
    """Sample the legend key colours top-to-bottom.  box = (x0, x1, y0, y1) of the key column."""
    x0, x1, y0, y1 = box
    sub = a[y0:y1, x0:x1].astype(int)
    sat = sub.max(axis=2) - sub.min(axis=2)
    dark_grey = (sat < 12) & (sub.mean(axis=2) < 200) & (sub.mean(axis=2) > 90)
    marker = (sat > 60) | dark_grey
    rows = np.flatnonzero(marker.mean(axis=1) > 0.05)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 15) + 1)
    cols = []
    for g in groups:
        px = sub[g[0]:g[-1] + 1][marker[g[0]:g[-1] + 1]]
        cols.append(px.mean(axis=0))
    return cols


def classify(a, colors, tol=55):
    """Nearest-colour label per pixel, or -1 if far from every series colour."""
    flat = a.reshape(-1, 3).astype(float)
    d = np.stack([np.linalg.norm(flat - c, axis=1) for c in colors], axis=1)
    lab = d.argmin(axis=1); lab[d.min(axis=1) > tol] = -1
    return lab.reshape(a.shape[:2])


def read_series(a, lab, k, xs_px, panel, half=4):
    y0, y1, x0, x1 = panel
    out = []
    for xp in xs_px:
        band = lab[y0:y1, int(xp - half):int(xp + half + 1)]
        ys = np.flatnonzero((band == k).any(axis=1)) + y0
        out.append(np.median(ys) if len(ys) else np.nan)
    return np.array(out)


def digitize(png, x_first, x_step, y_first, y_step, x_query, legend_box=None, colors=None, tol=55,
             y_anchor_series=None):
    a = np.asarray(Image.open(png).convert("RGB")).astype(int)
    panel, v, h = panel_and_grid(a)
    px_per_x, nx = calibrate(v, x_first, x_step)
    if colors is None: colors = legend_colors(a, legend_box)
    lab = classify(a, colors, tol)
    if y_anchor_series is None:
        px_per_y, ny = calibrate(h, y_first, y_step, descending=True)
    else:
        # A series known to be a flat line at y = 0 (the ADR reference in Figs. 12/13): its
        # modal pixel row is data 0.  Spacing = the smallest gap between detected gridlines,
        # which survives missed lines.  Robust to annotations crossing the panel.
        k = SERIES.index(y_anchor_series); y0, y1, x0, x1 = panel
        rows = np.flatnonzero((lab[y0:y1, x0 + 20:x1 - 20] == k).sum(axis=1) > 0.3 * (x1 - x0 - 40)) + y0
        zero_px = np.median(rows)
        gaps = np.diff(np.sort(h)); step_px = gaps[gaps > 8].min()
        px_per_y = np.array([-step_px / y_step, zero_px]); ny = len(h)
    xs_px = np.polyval(px_per_x, np.array(x_query, float))
    res = {}
    for k, name in enumerate(SERIES[:len(colors)]):
        ypx = read_series(a, lab, k, xs_px, panel)
        res[name] = (ypx - px_per_y[1]) / px_per_y[0]
    y_top = (panel[0] - px_per_y[1]) / px_per_y[0]; y_bot = (panel[1] - px_per_y[1]) / px_per_y[0]
    x_l = (panel[2] - px_per_x[1]) / px_per_x[0]; x_r = (panel[3] - px_per_x[1]) / px_per_x[0]
    info = dict(panel=panel, n_x=nx, n_y=ny, colors=[np.array(c).round().tolist() for c in colors],
                x_map=f"{px_per_x[0]:.2f} px/unit, panel x {x_l:.1f}..{x_r:.1f}",
                y_map=f"{-px_per_y[0]:.2f} px/unit, panel y {y_bot:.1f}..{y_top:.1f}")
    return res, info


if __name__ == "__main__":
    import pandas as pd
    res, info = digitize("fig11_hi.png", 0, 2.5, 0, 12.5, x_query=list(range(0, 16)),
                         legend_box=(1370, 1460, 470, 1400))
    print("Fig11:", info["n_x"], "x-lines,", info["n_y"], "y-lines |", info["x_map"], "|", info["y_map"])
    colors = [np.array(c) for c in info["colors"]]
    df = pd.DataFrame(res, index=list(range(0, 16))); df.index.name = "LM"
    print(df.round(1).to_string()); df.to_csv("paper_fig11_digitized.csv")
    page = Image.open("hi-12.png")
    for fig, box in (("fig12", (2200, 300, 3450, 1450)), ("fig13", (2200, 1550, 3450, 2700))):
        page.crop(box).save(f"{fig}_hi.png")
        r, i = digitize(f"{fig}_hi.png", 80, 2.5, -75, 12.5, x_query=[80, 85, 90, 95, 99], colors=colors,
                        y_anchor_series="ADR")
        print(f"\n{fig}:", i["n_x"], "x-lines,", i["n_y"], "y-lines |", i["x_map"], "|", i["y_map"])
        d = pd.DataFrame(r, index=[80, 85, 90, 95, 99]); d.index.name = "PDR"
        print(d.round(1).to_string()); d.to_csv(f"paper_{fig}_digitized.csv")
