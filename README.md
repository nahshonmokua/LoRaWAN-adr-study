# Reproduction of González-Palacio et al. (2023), IEEE IoT-J 10(12)

"Machine-Learning-Based Combined Path Loss and Shadowing Model in LoRaWAN for Energy Efficiency
Enhancement" — DOI 10.1109/JIOT.2023.3239827 — from the released `LoRaWAN_PathLossMeasurements.csv`
(930,753 rows; data descriptor: Data 8(1):4, 2023, DOI 10.3390/data8010004 — US915, SF 7–10).

## Run

```bash
python3 src/run_final.py                 # ~10 min: data -> conventional models -> CPLS models -> residuals -> ADR/energy -> outputs/final/
python3 tests/test_adr_equivalence.py && python3 tests/test_pipeline_behaviour.py
jupyter nbconvert --to notebook --execute notebooks/figures.ipynb     # every figure, from outputs/final/*.csv
```

Python 3.12; numpy 1.26, pandas 2.2, scipy 1.13, scikit-learn 1.4.2, statsmodels 0.14, matplotlib 3.8.
Seed 42 everywhere; two clean runs agree to 1e-11. The CSV cache `outputs/raw.pkl` rebuilds itself when the CSV changes.

## Layout

```
src/
  run_final.py            the pipeline; CONFIG at the top holds every choice the paper leaves open
  paper_spec.py           every number the paper prints, and its data descriptor's
  plots.py                every figure, drawn from outputs/final/*.csv (run_final saves them; the notebook shows them)
  data_loading.py  splitting.py  reconstruction.py  conventional_models.py  cpls_models.py
  adr_algorithm.py        Algorithm 1 (as printed / corrected / as the text describes it) + conventional ADR   [reusable]
  simulation.py           LM sweeps for both schemes; residual and receiver-threshold delivery rules
  energy.py               ToA (matches the CSV's toa column exactly) and the Table VII power model
tests/                    Algorithm 1 scalar == vectorised; CONFIG forwarding, cache invalidation, delivery-rule dependence
notebooks/figures.ipynb   Figs. 4, 11, 12, 13, 14-15, plus energy at equal achieved PDR, with the key tables
data/paper_digitized/     the paper's Figs. 11-13 digitized from the PDF (digitize.py) — the hollow markers in the figures
outputs/final/            generated, ignored by git: tables (CSV), config.json, models, test predictions, figures/
```

## What `outputs/final/` contains

`headline_claims.csv` (the paper's six headline numbers, ours, and what each rests on), `table_iii_conventional.csv`,
`table_iv_cpls.csv`, `table_v_mlr_weights.csv`, `appendix_residual_tests.csv`, `fig11_curves.csv` / `fig11_link_margins.csv`,
`fig12_13_energy_toa.csv` (at the paper's operating LMs) and `fig12_13_reading_sensitivity.csv` (at equal achieved PDR),
the SF ≤ 10 companions (`*_sf10.csv`) and the receiver-threshold companions (`*_threshold_sf*.csv`),
`figs_vs_paper_digitized.csv`, and `figures/`.

## Choices the paper leaves open (all in `run_final.CONFIG`)

* **Data.** The released file is smaller than the paper's database and its residuals are near-normal; the paper's
  MLR RMSE 1.951, ν = 11.43 and ±15 dB tails cannot be obtained from it. `reconstruction.py` perturbs 0.25 % + 0.15 %
  of rows by ±8–15 dB before the split — both fractions are *calibrated* to those printed values, so agreement on them
  is calibration, not evidence (`headline_claims.csv` says so per row). Set both fractions to 0 for the unmodified data.
* **Algorithm 1** is run as Section IV-A describes it (variant `text`): `snr_limit` from the current SF, the two
  scenarios exclusive, TP from the margin at the SF used, power clamped to 2–20 dBm. As printed it cannot run.
* **Delivery rule** `PL_true − PL_pred < LM` (ADR: `SNRmax − SNR < LM`) — the paper's wording; blind to the
  selected SF/TP, which is why the receiver-threshold companions exist.
* **Conventional ADR**: SF 7–12, `TP = 20 − Me`, SNRmax over the 20 samples before the packet, taken over the test
  set in shuffled order (the only reading that reproduces Fig. 11's ADR curve; chronological history gives ~4.4 dB, not 11).
* **Operating points** of Figs. 12/13: the paper's integer LMs. **Shadowing** for SPLMSFT and MLR: the Appendix's
  Student-t ψ. **Noise floor**: `rssi − snr`. **SF 11/12** appear only through escalation of EN3 — the campaign
  never used them; the SF ≤ 10 companions show what remains without them.
