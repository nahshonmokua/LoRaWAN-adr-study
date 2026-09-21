from pathlib import Path
import json
import numpy as np
import pandas as pd

original = Path('/home/nahshon/OneDrive/Data Analysis/ADR 1 CL/outputs/final')
rerun = Path('/tmp/adr_review/clean_run/outputs/final')
rows = []
for path in sorted(original.glob('*.csv')):
    other = rerun / path.name
    if not other.exists():
        rows.append(dict(file=path.name, status='missing'))
        continue
    a, b = pd.read_csv(path), pd.read_csv(other)
    if a.shape != b.shape or a.columns.tolist() != b.columns.tolist():
        rows.append(dict(file=path.name, status='schema differs'))
        continue
    maximum = 0.0
    over = 0
    text_mismatch = 0
    for col in a.columns:
        x, y = pd.to_numeric(a[col], errors='coerce'), pd.to_numeric(b[col], errors='coerce')
        numeric = x.notna() & y.notna()
        if numeric.any():
            delta = (x[numeric] - y[numeric]).abs()
            maximum = max(maximum, float(delta.max()))
            over += int((delta > 1e-10).sum())
        text_mismatch += int((a.loc[~numeric, col].fillna('<NA>').astype(str) !=
                             b.loc[~numeric, col].fillna('<NA>').astype(str)).sum())
    rows.append(dict(file=path.name, status='compared', max_abs_difference=maximum,
                     numeric_differences_over_1e_10=over, text_differences=text_mismatch))

a = np.load(original / 'test_predictions.npz')
b = np.load(rerun / 'test_predictions.npz')
for name in a.files:
    delta = np.abs(a[name] - b[name])
    rows.append(dict(file=f'test_predictions.npz:{name}', status='compared',
                     max_abs_difference=float(delta.max()),
                     numeric_differences_over_1e_10=int((delta > 1e-10).sum()), text_differences=0))
result = pd.DataFrame(rows)
result.to_csv('/tmp/adr_review/clean_run_comparison.csv', index=False)
print(result.to_string(index=False))
print('CONFIG equal:', json.loads((original / 'config.json').read_text()) ==
      json.loads((rerun / 'config.json').read_text()))
print('Report text equal:', (original.parent / 'REPRODUCTION_REPORT.md').read_text() ==
      (rerun.parent / 'REPRODUCTION_REPORT.md').read_text())
