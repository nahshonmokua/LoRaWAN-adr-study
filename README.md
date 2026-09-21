# Reproduction: González-Palacio et al. (2023), IEEE IoT-J 10(12):10725–10739

"Machine-Learning-Based Combined Path Loss and Shadowing Model in LoRaWAN for Energy Efficiency
Enhancement" — DOI 10.1109/JIOT.2023.3239827. Reproduced from the released
`LoRaWAN_PathLossMeasurements.csv` (930,753 rows; its data descriptor is Data 8(1):4, 2023,
DOI 10.3390/data8010004 — US915, SF 7-10 only).

**Read `outputs/REPRODUCTION_REPORT.md`.** Everything else supports it.

## Run

```bash
python3 src/run_final.py             # ~10 min: data -> models -> residuals -> ADR sweep -> energy -> outputs/final/
python3 src/make_final_report.py     # builds outputs/REPRODUCTION_REPORT.md from outputs/final/*.csv
python3 tests/test_adr_equivalence.py
python3 outputs/provenance/validity/validity_checks.py   # ~2 min, feeds the report's validity sections
```

## Layout

```
src/
  run_final.py            the reproduction, end to end, at the determined configuration
  make_final_report.py    report generator (every number read from outputs/final/)
  paper_spec.py           every number printed in the paper, transcribed verbatim
  assumptions.py          every judgment call, with the paper's wording and ours
  data_loading.py         dtype-pinned load of the CSV, derived columns
  reconstruction.py       restores the pre-cleaning outlier population (see report, Configuration)
  splitting.py            the seeded 80/20 split
  conventional_models.py  Friis, Two-ray, Okumura-Hata, SPLMSF, SPLMSFT
  cpls_models.py          MLR (eq. 6), ANN, SVR, RF                        [reusable]
  adr_algorithm.py        Algorithm 1 (as printed / corrected / as the text describes it) + conventional ADR [reusable]
  simulation.py           LM sweep and the PDR decision rule
  energy.py               LoRa ToA (verified against the CSV) and the Table VII power model
  stage11_reconstruct.py  PROVENANCE: how the data step was determined
  stage14_inverse.py      PROVENANCE: inverse search that fixed the undocumented Algorithm-1 settings

Every non-printed choice in `run_final.CONFIG` is either determined by evidence in outputs/provenance/
(data reconstruction, Algorithm-1 control flow, delivery rule, continuous ADR, SF ranges, window, integer operating points, Appendix shadowing)
or listed as an open assumption in src/assumptions.py.
tests/test_adr_equivalence.py
notebooks/final_reproduction.ipynb
outputs/
  final/                  tables (CSV), figures/, models/, config.json  <- the deliverable
  provenance/
    stage11/stage14 CSVs  the data reconstruction and the Algorithm-1 inverse search
    paper_digitized/      the paper's Figs. 11-13 digitized from the PDF (500 dpi), validated against its text
    adr_mechanism/        how the delivery rule, the ADR mechanism and the reading of Section IV were determined (incl. rejected variants)
    released_file_check.py/.csv   the released CSV against its own data descriptor (Data 8(1):4, 2023): rows, Mahalanobis filter, MLR RMSE
    mlr_feature_ablation.py/.csv  what carries eq. (6)'s gain over the distance law (node offset + SNR; weather 0.03 dB)
    psi_scale_check.py/.csv       SPLMSFT's PDR advantage over SPLMSF is the scale of the sampled psi, not the t shape
    validity/             checks prompted by the independent review of 21 Sep 2026: delivery feasibility, released vs
                          reconstructed data, ADR window order, causal features, node hold-outs (validity_summary.csv)
  REPRODUCTION_REPORT.md
  raw.pkl                 cache of the loaded CSV (rebuilt automatically if absent)
```

## Reuse against other data

`cpls_models.py` and `adr_algorithm.py` have no file I/O and take a replaceable `FeatureSpec`:

```python
from cpls_models import FeatureSpec, make_X, make_rf, CplsModel
from adr_algorithm import enhanced_adr, AdrParameters
spec  = FeatureSpec(distance="range_m", pm25="pm25_ugm3", distance_scale=1e-3)
model = CplsModel("RF", make_rf().fit(make_X(train, spec), train.pl), spec)
tp, sf, me, adj = enhanced_adr(d, f, T, RH, BP, PM, SNR, pl_model=model.predict_row,
                               current_tp=20, current_sf=9, LM=4, noise_power=-100,
                               params=AdrParameters(), variant="text")
```

Python 3.12 · numpy 1.26 · pandas 2.2 · scipy 1.13 · scikit-learn 1.4 · statsmodels 0.14 · matplotlib 3.8. No GPU.
