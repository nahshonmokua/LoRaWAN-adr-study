"""Behaviour added or fixed after the independent review of 21 Sep 2026:
configuration forwarding, cache invalidation, and the dependence of each delivery rule on the selected SF/TP."""
import sys, tempfile, time, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data_loading as dl                                   # noqa: E402
import reconstruction                                        # noqa: E402
import run_final                                             # noqa: E402
from adr_algorithm import snr_limit_of                       # noqa: E402
from simulation import _delivered                            # noqa: E402


def _tiny_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(
        index=np.arange(n), timestamp=pd.date_range("2021-11-01", periods=n, freq="min"),
        device_id=pd.Categorical(rng.choice(["EN1", "EN2", "EN3", "EN4"], n)),
        distance=rng.choice([2140, 3450, 6100, 8260], n), ht=8.0, hr=5.0, ptx=20.0, ltx=1.0, gtx=2.9, lrx=4.25, grx=4.161,
        frequency=904_100_000, sf=rng.choice([7, 8, 9, 10], n), frame_length=9, temperature=22.0, rh=80.0, bp=845.0, pm2_5=5.0,
        rssi=rng.uniform(-105, -63, n), snr=rng.uniform(-18, 12, n), toa=0.1, energy=0.1)).assign(
        experimental_pl=lambda d: d.ptx - d.ltx + d.gtx - d.rssi + d.grx - d.lrx)


def test_config_fractions_and_amplitudes_are_forwarded():
    """prepare_data must pass CONFIG's restoration fractions and amplitude bounds to restore_outliers
    (until 21 Sep 2026 it did not; the defaults happened to equal CONFIG)."""
    captured = {}
    def fake_restore(df, **kw):
        captured.update(kw); out = df.copy(); out.attrs["n_restored"] = 0; return out
    saved = (run_final.load_cached, run_final.restore_outliers, run_final.FINAL, run_final.split)
    try:
        run_final.load_cached = lambda: dl.add_derived(_tiny_frame())
        run_final.restore_outliers = fake_restore
        run_final.split = lambda df: (df.iloc[:300], df.iloc[300:])
        with tempfile.TemporaryDirectory() as tmp:
            run_final.FINAL = Path(tmp)
            run_final.CONFIG["restore_frac_rssi_only"], run_final.CONFIG["restore_frac_coupled"], run_final.CONFIG["outlier_db"] = 0.123, 0.045, [1, 2]
            run_final.prepare_data()
        assert captured["frac_rssi_only"] == 0.123 and captured["frac_coupled"] == 0.045 and tuple(captured["outlier_db"]) == (1, 2)
    finally:
        run_final.load_cached, run_final.restore_outliers, run_final.FINAL, run_final.split = saved
        run_final.CONFIG["restore_frac_rssi_only"], run_final.CONFIG["restore_frac_coupled"], run_final.CONFIG["outlier_db"] = 0.0025, 0.0015, [8, 15]
    z = reconstruction.restore_outliers(dl.add_derived(_tiny_frame()), frac_rssi_only=0.0, frac_coupled=0.0)
    assert z.attrs["n_restored"] == 0 and np.array_equal(z.rssi.to_numpy(), _tiny_frame().rssi.to_numpy())
    z = reconstruction.restore_outliers(dl.add_derived(_tiny_frame()), frac_rssi_only=0.05, frac_coupled=0.0, outlier_db=(0.0, 0.0))
    assert np.allclose(z.rssi.to_numpy(), _tiny_frame().rssi.to_numpy())


def test_cache_is_invalidated_when_the_csv_changes():
    """load_cached must rebuild when the CSV's size/mtime changes and must not serve a pickle for a missing CSV."""
    calls = []
    saved = (dl.CSV_PATH, dl.CACHE, dl.load_raw)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "data.csv"; csv.write_text("a\n1\n")
            dl.CSV_PATH, dl.CACHE = csv, Path(tmp) / "raw.pkl"
            dl.load_raw = lambda chunksize=None: (calls.append(1), _tiny_frame())[1]
            dl.load_cached(); assert len(calls) == 1
            dl.load_cached(); assert len(calls) == 1, "second call must hit the cache"
            csv.write_text("a\n1\n2\n"); os.utime(csv, (time.time() + 5, time.time() + 5))
            dl.load_cached(); assert len(calls) == 2, "a changed CSV must rebuild the cache"
            csv.unlink()
            try:
                dl.load_cached(); raise AssertionError("a missing CSV must not be served from the cache")
            except FileNotFoundError:
                pass
    finally:
        dl.CSV_PATH, dl.CACHE, dl.load_raw = saved


def test_residual_rule_is_blind_to_sf_and_tp_but_threshold_rule_is_not():
    """The paper's rule (D4) reduces to PL_true - PL_pred < LM: any SF/TP choice gives the same PDR.
    The receiver-threshold rule depends on the selected SF and TP.  The SF <= 10 companion is
    therefore only meaningful under the threshold rule (review item 1)."""
    rng = np.random.default_rng(1); n = 5000
    rssi = rng.uniform(-105, -63, n); snr = rng.uniform(-18, 12, n); npow = rssi - snr
    pl_true = 110 - rssi; pred = pl_true + rng.normal(0, 2, n)
    a = _delivered(rssi, np.full(n, 20.0), np.full(n, 20.0), np.full(n, 7), npow, "residual", pred, pl_true, lm=3.0)
    b = _delivered(rssi, np.full(n, 2.0), np.full(n, 20.0), np.full(n, 12), npow, "residual", pred, pl_true, lm=3.0)
    assert np.array_equal(a, b)
    c = _delivered(rssi, np.full(n, 20.0), np.full(n, 20.0), np.full(n, 12), npow, "threshold")
    d = _delivered(rssi, np.full(n, 2.0), np.full(n, 20.0), np.full(n, 7), npow, "threshold")
    assert c.mean() > d.mean()
    assert np.array_equal(c, snr > snr_limit_of(12))


if __name__ == "__main__":
    test_config_fractions_and_amplitudes_are_forwarded(); print("CONFIG forwarding: OK")
    test_cache_is_invalidated_when_the_csv_changes(); print("cache invalidation: OK")
    test_residual_rule_is_blind_to_sf_and_tp_but_threshold_rule_is_not(); print("delivery-rule dependence: OK")
