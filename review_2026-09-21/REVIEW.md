# Independent review — 21 September 2026

The code contains useful, repeatable calculations, but the current report overstates independent reproduction and deployable PDR/energy evidence. The most important corrections are to separate measured-data analysis from calibrated synthetic perturbations, make delivery depend on the actual selected radio parameters, and evaluate causal chronological decisions.

This review examined the Python implementation, report, provenance and supplied paper, with independent ML/data, ADR/energy and provenance audits. All 23 Python files also received syntax and static checks. The Downloads and workspace PDFs have identical normalized extracted text after removal of download footers. Source files and existing scientific outputs were not edited. Original numerical probes read the CSV directly and wrote to `/tmp/adr_review`; their scripts and evidence are copied here. The separate full rerun is under `/tmp/adr_review/clean_run/`.

## 1. High: the capped energy claim does not achieve its advertised delivery rate

`src/simulation.py:89–93` defines delivery as `PL_true - PL_pred < LM`. Its conventional counterpart at lines 185–186 uses `SNRmax - SNR < LM`. Neither depends on the selected transmit power or SF. Thus the SF10 companion calculation at `src/run_final.py:302–315` changes energy without changing PDR by construction.

Using the existing threshold rule on the same reconstructed test rows, saved predictions, selected powers and operating margins gives:

| Scheme | LM | Residual-rule PDR | Threshold-rule PDR |
|---|---:|---:|---:|
| ANN | 4 | 99.212% | 93.526% |
| SVR | 4 | 99.326% | 93.642% |
| RF | 4 | 99.359% | 93.673% |
| Conventional ADR | 11 | 99.027% | 93.444% |

Even allocating 20 dBm/SF10 to every row reaches only 94.306% under this fixed-threshold model. The released clean test set's equivalent ceiling is 94.331%, so this issue is not primarily caused by the injected outliers. These are internal model checks, not measured physical PDR: the real receiver's error curve still needs calibration. The regional SF restriction is supported by the LoRa Alliance RP002-1.0.3 US902–928 data-rate table, which allows SF7–10 at 125 kHz on uplinks.

Required correction: relabel the current result as energy at fixed assumed margins; implement and validate delivery feasibility, then reselect operating points. Report infeasibility instead of claiming a 99% target that cannot be reached under the selected model. Evidence: `adr_probe.py`, `adr_rule_probe.csv`.

## 2. High: synthetic calibration is counted as independent reproduction

`src/reconstruction.py:32–39` overwrites 3,707 observations with chosen ±8–15 dB RSSI/SNR perturbations; it does not recover missing measured rows. `src/run_final.py:91–92` applies this before splitting, so both training and evaluation targets are modified. `src/stage11_reconstruct.py:11–14` explicitly says MLR RMSE and Student-t degrees of freedom were calibration targets. Nevertheless the report counts both as reproduced headlines.

The claim of one free parameter is contradicted by `src/stage11_reconstruct.py:60–64`: the second, coupled fraction was selected to move conventional ADR toward the known 11 dB target. Perturbation amplitudes, signs, distribution and fault types are additional modeling choices.

| MLR result | Released data, DHT22 rows removed | Synthetic final data |
|---|---:|---:|
| Random-split test RMSE | 1.846206 dB | 1.935219 dB |
| Training excess kurtosis | −0.058036 | +3.717874 |

The synthetic process creates the heavy tails. Matching the target tail statistics afterward does not independently validate the missing-data hypothesis. Also, `src/reconstruction.py:3–5` states that the paper fitted before filtering, whereas the paper's Section III-A explicitly places outlier removal before splitting and fitting.

Required correction: make the unperturbed released-data analysis primary; report synthetic perturbations as calibrated sensitivity experiments. Evidence: `ml_independent_check.py` and `.log`.

## 3. High: the operational 4-versus-11 dB comparison is not established

The final conventional baseline takes rolling windows in shuffled test order (`src/run_final.py:52`; `src/simulation.py:140–180`). On the released clean test rows, its 99th-percentile residual margin is 8.5 dB in shuffled order, 4.8 dB with chronological test history, and 4.4 dB with the preceding 20 packets from the full chronological campaign, evaluated on identical test row IDs. Synthetic restoration raises that last figure to about 10.74 dB.

The 0.15% coupled perturbation includes approximately 0.075% positive SNR spikes. A 20-sample maximum can spread these into roughly 1.5% of windows, directly influencing the 99th percentile. This explains why the coupled perturbation strongly controls the reported ADR margin.

The enhanced scheme also uses the same packet's measured gateway SNR and `RSSI−SNR` noise to select parameters supposedly before transmitting that packet. Current-packet SNR is a legitimate retrospective regression input, but it is not available before that transmission. The conventional baseline receives prior measurements.

Required correction: distinguish retrospective figure reconstruction from prospective performance; evaluate causal features, chronological histories and power/loss feedback. A clean-data forward-time MLR check gave RMSE 1.91584 dB with current SNR, 2.09142 with previous-packet SNR, and 1.96797 with the preceding-20 mean. These do not prove that all useful gains disappear, but they show why a causal evaluation is needed. Evidence: `adr_baseline_probe.py`, `.csv`; `ml_lagged_snr_check.py`, `.log`.

## 4. High: energy curves do not consistently compare equal achieved PDR

`src/run_final.py:274–295` selects the paper's operating LM and attaches a target-PDR label without requiring the reproduced curve to achieve that PDR. The ANN point labeled 80% actually achieves 75.731%, while its ADR reference achieves 85.630%. This is a comparison at the paper's margins, not equal-delivery efficiency. `pdr_at_LM` records the mismatch but the plotted x-axis and report still use the nominal label.

Required correction: choose each scheme's achieved target crossing, or explicitly plot achieved PDR and mark unmatched targets. Keep the fixed-paper-margin comparison as a separate emulation diagnostic.

## 5. High: the six-of-six verdict is not an adequate validation criterion

`src/run_final.py:373–384` checks RMSE but not R² for the combined headline. The energy gate allows ANN within 4.35 percentage points and the maximum within 8.7 points, without requiring the claimed winning model. It checks fitted Student-t ν within 2 despite calibration to that quantity. These gates produce six passes despite reversed energy ranking and 14.6 percentage-point RMSE across the energy curves.

The inverse search does not establish what the authors implemented. It compares a finite candidate set, selected repeatedly against the same figures later called validation. Similarity supports a candidate interpretation, not unique identification. Moreover, saved Fig.11 maximum discrepancies include ANN 7.02, RF 5.45, SVR 3.80 and ADR 3.78 percentage points, larger than the quoted approximately ±2-point digitization precision.

Required correction: replace binary headline passes with numerical differences, justified tolerances, and explicit labels for fitted quantities versus independent checks.

## 6. Medium: packet-level random folds do not test new-link generalization

There are four nodes at four fixed distances. Every random training/test fold contains those same link identities. The reported accuracy therefore concerns interpolation on these links. Independent released-data leave-one-node-out MLR RMSEs were 3.9200, 2.4026, 11.7227 and 5.2771 dB, versus 1.8462 dB under the random split.

This does not make the paper-matching random split incorrect. It limits claims about deployment planning, unseen nodes or new locations. Use temporal validation and node/site holdouts when making those broader claims.

## 7. Medium: recorded configuration does not control the experiment

`src/run_final.py:70–71` stores restoration fractions and amplitude bounds in CONFIG, but line 91 never passes them to `restore_outliers`. The actual values come from independent defaults/constants in `src/reconstruction.py:20–26,35`.

An independent probe set both recorded fractions and amplitudes to zero; `prepare_data()` still altered 3,707 RSSI rows. This is a concrete reproducibility defect. Pass the resolved values explicitly and serialize the parameters actually executed. Payload/BW/CR settings similarly need auditing against the parameters actually passed into `SimConfig` and `time_on_air`.

## 8. Medium: caching and missing provenance prevent the claimed self-contained reproducibility

`src/data_loading.py:119–125` trusts an existing pickle without checking the source CSV or derived-column code. A probe returned cached data even when the configured CSV did not exist. This does not prove current results are stale; it proves that input changes can be silently ignored.

Stage11 requires absent precursor tables. Stage14's reconstructed-data path requires missing `reconstructed_test_C.pkl` and predictions; running that loader raises FileNotFoundError. Stage14's clean path now loads models from the final synthetic-data run. The claimed 10k–250k SVR learning-curve evidence and original Stage9 calibration are not included. The retained `outputs/run_final.log` also describes an older result (ADR margin 10.70 and ANN energy 44.7%) rather than the final tables (10.85 and 45.5%).

Required correction: input/code hashes for cache invalidation, resolved environment requirements, run manifests, and executable provenance. Do not infer absence of hidden state from one successful rerun or one changed comparison column.

## 9. Medium: the assumptions ledger contains conflicting active specifications

A14 describes threshold delivery while D4/final CONFIG use residual delivery. A18 describes each scheme's own smallest qualifying LM while D6 uses the paper's margins. A19 fits shadowing to SPLMSF residuals while D7 uses MLR residuals. These are substantial changes, not wording differences. Mark superseded assumptions historical and retain one active specification for each final choice.

## 10. Medium: weather attribution is narrower than the conclusion

The supplied 0.028 dB comparison concerns an additive MLR on these four links and the synthetic dataset. Barometric pressure is strongly confounded with node identity; this supports caution about a causal weather interpretation. It does not establish that weather is generally unimportant or that every nonlinear model has the same ablation behavior.

An independent controlled 100,000-row RF check supports the narrower conclusion: RMSE 1.52744 dB with all features, 1.54504 with distance/frequency/SNR, and 2.10402 without SNR. Conditional weather improvement was 0.01760 dB in this particular experiment. Report such empirical, model-specific comparisons instead of a universal 0.03 dB claim.

## 11. Medium: statistical and mechanistic assertions need qualification

The normality KS test estimates mean and SD from the tested residuals and then uses ordinary known-parameter KS p-values (`src/run_final.py:190–191`). Use a fitted-parameter procedure such as Lilliefors or a suitable bootstrap; temporal dependence and model fitting also need consideration. Strong rejection may survive, but the reported nominal p-value is not calibrated.

Pearson correlation squared is reported as R². This can be useful for matching the paper's apparent convention but should be labeled Pearson r² and accompanied by ordinary 1−SSE/SST. It does not measure calibration accuracy.

The following are hypotheses or conditional results, not verified facts about the authors: two coefficients definitely being decimal-place errors; the authors' CV standard deviations being impossible; the t-scale experiment proving what caused the authors' curves. Independent random shadowing is added only to selected predictors despite observed targets already including shadowing. This may emulate a paper interpretation, but it changes the point-prediction task and handicaps those deterministic predictors. Separate distribution/quantile modeling from random point forecasts. Calling ANN/SVR/RF "tied" also needs an uncertainty analysis or an explicit practical-equivalence criterion; close point estimates alone do not establish equivalence.

## 12. Lower-priority operational idealizations

Continuous power and a one-byte PHY payload reproduce an idealized convention, not a standard one-byte LoRaWAN application exchange. With minimum LoRaWAN overhead, one application byte uses 14 PHY payload bytes. The current calculation includes transmitter radio energy, not total node energy including sensing, inference, receive windows or retransmissions. These should be explicit scope limits. The SF10 cap is appropriate for this 125 kHz uplink comparison, but does not alone make it a validated network model.

## Checks that passed

All five supplied ADR tests pass. All 23 Python files parse. Ruff's F checks found nine unused-import/unused-variable/redundant-f-string issues and no undefined names. ToA matches all 926,663 cleaned rows with zero mismatches at 1e-9 s tolerance (maximum absolute discrepancy 5.55e-17 s). The MLR final RMSE reproduces exactly. Scaling is inside sklearn pipelines, so the audit found no ordinary train/test scaling leakage.

These checks establish internal implementation consistency and some exact numerical agreements. They do not validate the delivery rule, chronology, synthetic missing-data hypothesis or real-world energy claims. The complete isolated rerun finished successfully in 9 minutes 56 seconds and regenerated the report text exactly. All 16 numerical CSV tables matched within 4.46e-11; the largest discrepancy was the Breusch–Pagan statistic. MLR/ANN/SVR saved predictions matched exactly; RF differed by at most 7.11e-14. CONFIG matched exactly. Thus practical numerical repeatability was independently confirmed, although this check does not establish the literal claim that every file always agrees within 1e-13. See `clean_run_comparison.csv`, `clean_run.log`, and `environment.txt`.

Primary external references: [LoRa Alliance RP002-1.0.3](https://lora-alliance.org/wp-content/uploads/2021/05/RP002-1.0.3-FINAL-1.pdf); [SciPy fitted-parameter goodness-of-fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html); [TTN message structure](https://www.thethingsnetwork.org/docs/lorawan/message-types/) and [PHY format](https://www.thethingsnetwork.org/docs/lorawan/lora-phy-format/).
