from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = Path('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL')
sys.path.insert(0, str(ROOT / 'src'))
import data_loading as dl
import run_final as run

with tempfile.TemporaryDirectory(prefix='adr_config_review_') as tmp:
    scratch = Path(tmp)
    raw = dl.add_derived(dl.load_raw())
    run.load_cached = lambda: raw
    run.FINAL = scratch
    run.CONFIG['restore_frac_rssi_only'] = 0
    run.CONFIG['restore_frac_coupled'] = 0
    run.CONFIG['outlier_db'] = [0, 0]
    df, _, _ = run.prepare_data()
    print('Zero recorded fractions/amplitudes; altered rows:', df.attrs['n_restored'])
    print('Changed RSSI:', np.count_nonzero(df.rssi.to_numpy() != raw.loc[raw.rh > 2, 'rssi'].to_numpy()))
    dl.CACHE = scratch / 'cache.pkl'
    pd.DataFrame({'marker': ['old cached contents']}).to_pickle(dl.CACHE)
    dl.CSV_PATH = scratch / 'nonexistent.csv'
    print('CSV exists:', dl.CSV_PATH.exists(), 'cached result:', dl.load_cached().to_dict('list'))
