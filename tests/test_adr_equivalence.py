"""The vectorised Algorithm 1 must agree with the line-for-line scalar transcription."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from adr_algorithm import AdrParameters, enhanced_adr, enhanced_adr_vectorized, snr_limit_of  # noqa: E402


def _case_grid(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        cpls=rng.uniform(80, 125, n),
        tp=rng.uniform(2, 20, n),
        sf=rng.integers(7, 13, n).astype(float),
        npow=rng.uniform(-105, -70, n),
    )


def test_vectorised_matches_scalar():
    g = _case_grid()
    params = AdrParameters()
    for variant in ("as_printed", "corrected", "text"):
        for LM in (0.0, 3.0, 7.0, 15.0):
            tp_v, sf_v, me_v, adj_v = enhanced_adr_vectorized(
                g["cpls"], g["tp"], g["sf"], LM, g["npow"], params, variant)
            for i in range(len(g["cpls"])):
                tp_s, sf_s, me_s, adj_s = enhanced_adr(
                    0, 0, 0, 0, 0, 0, 0,
                    pl_model=lambda *a, _c=g["cpls"][i]: _c,
                    current_tp=g["tp"][i], current_sf=int(g["sf"][i]), LM=LM,
                    noise_power=g["npow"][i], params=params, variant=variant)
                assert sf_v[i] == sf_s, (variant, LM, i, sf_v[i], sf_s)
                assert np.isclose(tp_v[i], tp_s), (variant, LM, i, tp_v[i], tp_s)
                assert np.isclose(me_v[i], me_s), (variant, LM, i)
                assert bool(adj_v[i]) == bool(adj_s), (variant, LM, i)


def test_as_printed_loops_are_degenerate():
    """Documented defect: as printed, the while-loops cannot change margin_excess,
    so SF is driven to max_sf (increase branch) or min_sf (decrease branch)."""
    g = _case_grid(500, seed=1)
    p = AdrParameters()
    _, sf, me, _ = enhanced_adr_vectorized(g["cpls"], g["tp"], g["sf"], 5.0, g["npow"], p, "as_printed")
    rssi = g["tp"] - p.ltx + p.gtx - g["cpls"] + p.grx - p.ltx
    me0 = rssi - (g["npow"] + p.snr_limit + 5.0)
    assert np.all(sf[me0 < 0] == p.max_sf)
    assert np.all(sf[me0 > 0] == p.min_sf)


def test_sf_always_within_bounds():
    g = _case_grid(500, seed=2)
    p = AdrParameters()
    _, sf, _, _ = enhanced_adr_vectorized(g["cpls"], g["tp"], g["sf"], 4.0, g["npow"], p, "corrected")
    assert np.all(sf >= p.min_sf) and np.all(sf <= p.max_sf)


def test_line28_lets_tp_exceed_max_tp():
    """Documented defect: line 28 clamps only from below, so the decrease branch can
    raise TP above max_tp by up to one SNR_limit step (2.5 dB)."""
    g = _case_grid(5000, seed=3)
    p = AdrParameters()
    tp, _, _, _ = enhanced_adr_vectorized(g["cpls"], g["tp"], g["sf"], 4.0, g["npow"], p, "corrected")
    assert (tp > p.max_tp).any()
    assert tp.max() <= p.max_tp + 2.5 + 1e-9
    tp_c, _, _, _ = enhanced_adr_vectorized(g["cpls"], g["tp"], g["sf"], 4.0, g["npow"], p,
                                            "corrected", clamp_tp=True)
    assert tp_c.max() <= p.max_tp + 1e-9




def test_text_variant_tp_from_final_sf():
    """Variant "text": the two scenarios are exclusive and TP is set from the margin at the SF
    actually used, so from 20 dBm the set point never exceeds 20 dBm.  As printed, the decrease
    branch's `break` leaves margin_excess at the decremented SF and line 27 pushes TP above max."""
    g = _case_grid(2000, seed=3)
    p = AdrParameters()
    tp, sf, me, _ = enhanced_adr_vectorized(g["cpls"], 20.0, g["sf"], 4.0, g["npow"], p, "text")
    rssi = 20.0 - p.ltx + p.gtx - g["cpls"] + p.grx - p.lrx
    me_final = rssi - (g["npow"] + snr_limit_of(sf) + 4.0)
    assert np.allclose(me, me_final)
    assert np.all(tp <= 20.0 + 1e-9)
    assert np.allclose(tp, np.maximum(20.0 - np.maximum(me_final, 0.0), p.min_tp))
    tp_c, _, _, _ = enhanced_adr_vectorized(g["cpls"], 20.0, g["sf"], 4.0, g["npow"], p, "corrected")
    assert np.any(tp_c > 20.0), "the printed listing does push TP above max_tp"


if __name__ == "__main__":
    test_vectorised_matches_scalar(); print("scalar == vectorised: OK")
    test_as_printed_loops_are_degenerate(); print("as-printed degeneracy confirmed: OK")
    test_sf_always_within_bounds(); print("SF bounds: OK")
    test_line28_lets_tp_exceed_max_tp(); print("line-28 unbounded-TP defect confirmed: OK")
    test_text_variant_tp_from_final_sf(); print("text variant: TP from the margin at the final SF, never above max_tp: OK")
