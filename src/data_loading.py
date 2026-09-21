"""Stage 1 - load and profile LoRaWAN_PathLossMeasurements.csv.

The released CSV is ~151 MB / 930 753 rows.  Dtypes are pinned explicitly so the
load is deterministic and fits comfortably in RAM (~90 MB as float32/int32).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "LoRaWAN_PathLossMeasurements.csv"
OUTPUTS = PROJECT_ROOT / "outputs"

# Columns of the released CSV, with the unit we determined in Stage 1.
COLUMN_UNITS = {
    "index":           "-",
    "timestamp":       "M/D/YYYY HH:MM (local)",
    "device_id":       "-",
    "distance":        "m",
    "ht":              "m",          # transmitter (EN) antenna height
    "hr":              "m",          # receiver (GW) antenna height
    "ptx":             "dBm",
    "ltx":             "dB",
    "gtx":             "dBi",
    "lrx":             "dB",
    "grx":             "dBi",
    "frequency":       "Hz",
    "sf":              "-",
    "frame_length":    "bytes",
    "temperature":     "degC",
    "rh":              "%",
    "bp":              "hPa",
    "pm2_5":           "ug/m^3",
    "rssi":            "dBm",
    "snr":             "dB",
    "toa":             "s",
    "experimental_pl": "dB",
    "energy":          "J (per-transmission, as released)",
    "esp":             "dBm",        # estimated signal power
    "pn":              "dBm",        # noise power = esp - snr (ESP-referenced)
}

DTYPES = {
    "index": "int32", "device_id": "category", "distance": "int32",
    "ht": "float32", "hr": "float32", "ptx": "float32", "ltx": "float32",
    "gtx": "float32", "lrx": "float32", "grx": "float32",
    "frequency": "int64", "sf": "int8", "frame_length": "int16",
    "temperature": "float32", "rh": "float32", "bp": "float32",
    "pm2_5": "float32", "rssi": "float32", "snr": "float32",
    "toa": "float64", "experimental_pl": "float64", "energy": "float64",
    "esp": "float64", "pn": "float64",
}

# Predictor set of eq. (3): CPLS = f(d, f, T, RH, BP, PM, SNR) + psi
PREDICTORS = ["distance", "frequency", "temperature", "rh", "bp", "pm2_5", "snr"]
TARGET = "experimental_pl"


def load_raw(path: os.PathLike | None = None, chunksize: int | None = None) -> pd.DataFrame:
    """Load the released CSV with pinned dtypes.  `chunksize` streams the read."""
    path = Path(path) if path is not None else CSV_PATH
    # The released timestamps are D/M/YYYY HH:MM (e.g. "4/11/2021" = 4 Nov 2021),
    # which is consistent with the paper's Oct-2021..Mar-2022 campaign window.
    kwargs = dict(dtype=DTYPES, parse_dates=["timestamp"], date_format="%d/%m/%Y %H:%M")
    if chunksize is None:
        return pd.read_csv(path, **kwargs)
    return pd.concat(pd.read_csv(path, chunksize=chunksize, **kwargs), ignore_index=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the paper's equations are written in."""
    df = df.copy()
    df["distance_km"] = df["distance"] / 1000.0
    df["frequency_mhz"] = df["frequency"] / 1e6
    # eq. (1) rearranged: PL = PT - LT + GT - RSSI + GR - LR
    df["pl_from_link_budget"] = (
        df["ptx"] - df["ltx"] + df["gtx"] - df["rssi"] + df["grx"] - df["lrx"]
    )
    # Algorithm 1's link budget is RSSI-referenced (line 4 builds an RSSI from the
    # path loss), so its `noise_power` input must be the RSSI-referenced noise floor.
    # The released `pn` column is ESP-referenced (pn = esp - snr) and sits 0.2-18.6 dB
    # (mean 6.2 dB) below it; mixing the two injects that offset into the margin.
    df["noise_power_rssi"] = df["rssi"] - df["snr"]
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-field statistics: n, missing, dtype, unit, min/mean/median/max/std/unique."""
    rows = []
    for col in df.columns:
        s = df[col]
        rec = {
            "field": col,
            "unit": COLUMN_UNITS.get(col, "-"),
            "dtype": str(s.dtype),
            "n_non_null": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            rec.update(
                min=float(s.min()), p25=float(s.quantile(0.25)), mean=float(s.mean()),
                median=float(s.median()), p75=float(s.quantile(0.75)),
                max=float(s.max()), std=float(s.std()),
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            rec.update(min=str(s.min()), max=str(s.max()))
        rows.append(rec)
    return pd.DataFrame(rows)


CACHE = OUTPUTS / "raw.pkl"


def _cache_key() -> str:
    """Identity of the cached frame: the CSV's size and mtime plus a hash of this loader's source,
    so a changed CSV or a changed derived-column definition invalidates the pickle."""
    import hashlib
    st = CSV_PATH.stat()
    src = Path(__file__).read_bytes()
    return hashlib.sha256(f"{st.st_size}:{st.st_mtime_ns}:".encode() + src).hexdigest()


def load_cached(rebuild: bool = False) -> pd.DataFrame:
    """Load the CSV once, then reuse a pickle cache in outputs/ - only while the CSV and this
    module are unchanged (the key is stored next to the pickle)."""
    key_file = CACHE.with_suffix(".key")
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)
    key = _cache_key()
    if CACHE.exists() and key_file.exists() and key_file.read_text() == key and not rebuild:
        return pd.read_pickle(CACHE)
    df = add_derived(load_raw(chunksize=200_000))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE); key_file.write_text(key)
    return df
