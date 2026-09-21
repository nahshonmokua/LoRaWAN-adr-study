"""The single, seeded 80/20 split used by every downstream stage (A1, A3)."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TRAIN_FRACTION = 0.80


def split(df: pd.DataFrame, seed: int = SEED, train_fraction: float = TRAIN_FRACTION):
    tr, te = train_test_split(df, train_size=train_fraction, random_state=seed, shuffle=True)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
