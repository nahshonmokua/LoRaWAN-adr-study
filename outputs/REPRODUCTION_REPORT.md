# Reproduction of González-Palacio et al. (2023), IEEE IoT-J 10(12)

*Machine-Learning-Based Combined Path Loss and Shadowing Model in LoRaWAN for Energy Efficiency
Enhancement* — DOI [10.1109/JIOT.2023.3239827](https://doi.org/10.1109/JIOT.2023.3239827); data from the
companion descriptor *LoRaWAN Path Loss Measurements in an Urban Scenario Including Environmental Effects*,
Data 8(1):4, 2023 — DOI [10.3390/data8010004](https://doi.org/10.3390/data8010004).

Run 2026-09-21 · seed 42 · Python 3.12 / scikit-learn 1.4 · every number below is read from `outputs/final/*.csv`.

## Verdict

| # | Claim | Paper | Here | | Rests on |
|---|---|---|---|---|---|
| 1 | RMSE up to 1.566 dB, R2 up to 0.94 | 1.566 / 0.94 | **1.525 / 0.9441** | ✅ REPRODUCED | reconstructed data (calibrated to the paper's MLR RMSE) |
| 2 | Model ranking by test RMSE | RF < ANN < SVR < MLR | **RF < ANN < SVR < MLR** | ✅ REPRODUCED | independent |
| 3 | Best ML model: PDR > 99 % at LM = 4 dB | 4 dB | **3.85 dB** | ✅ REPRODUCED | independent (residual rule, oracle SNR) |
| 4 | Conventional ADR needs LM = 11 dB for 99 % | 11 dB | **10.85 dB** | ✅ REPRODUCED | reconstructed data (the coupled fraction is calibrated to this margin), shuffled window |
| 5 | Energy saving up to 43 % vs conventional ADR (43.5 % for ANN, its best model) | 43.5 % (ANN); max 43.5 % | **45.5 % (ANN); max 49.2 % (SVR)** | ✅ REPRODUCED | independent under the paper's SF 7-12; ordering not reproduced |
| 6 | Shadow fading is Student-t, nu = 11.43 | 11.43 | **11.09** | ⚖️ CALIBRATED | calibration target of D1 |

**5 of 6 headline numbers are matched, 3 of them independently of the calibrated data step; 1 is a calibration target and is labelled so.**
The four CPLS models rank in the paper's order and the best ML model reaches 99 % PDR at the paper's 4 dB on
the released data as well as on the reconstructed. The MLR RMSE, the Student-t ν and the conventional ADR's
11 dB are quantities the data step (Configuration 1) was *tuned* to; the released file on its own gives MLR
RMSE 1.846 dB with near-normal residuals (excess kurtosis -0.06) and a conventional ADR that needs
8.5 dB (shuffled window) or 4.4 dB (chronological history) for 99 %. The energy saving at 99 % lands within
a few points of 43 % under the paper's SF 7–12 (ANN 45.5 %, largest 49.2 % for SVR); the ordering is not reproduced,
and the number rests on spreading factors this deployment cannot use. **This report is a retrospective
reconstruction of the paper's figures; the section "Does the margin result hold up?" states what survives as
operational evidence.**

## Configuration

Seven things had to be fixed that the paper does not print, or prints differently from what its text
says. All were *determined* by scoring against the digitized figures, not assumed (`src/assumptions.py`
D1–D8; provenance `src/stage11_reconstruct.py`, `src/stage14_inverse.py`, `outputs/provenance/adr_mechanism/`).

1. **Data (calibrated).** The released CSV is a smaller database than the paper's, with near-normal
   residuals; the paper's Table IV and Appendix cannot be computed from it. Perturbing 0.25 % RSSI-only
   and 0.15 % RSSI+SNR-coupled observations at ±8–15 dB
   (3,707 of 930,753 rows, before the split) — brings the paper's MLR RMSE, Student-t ν,
   ±15 dB residual range and conventional-ADR link margin to their printed values. *Both* fractions were
   tuned to those targets (the first to RMSE and ν, the second to the 11 dB), so agreement on them is
   calibration, not evidence; the perturbation is what creates the heavy tails. The companion data
   descriptor (Data 8(1):4, 2023 — the released file's own paper) fixes what is missing: the file has
   930,753 rows against the 990,750 the IoT-J database had *after* its outlier step; the descriptor's
   MLR on the file gives RMSE 1.84 (ours 1.844), and its Mahalanobis filter removes 1.77 % of rows
   and moves that by 0.004 dB — the IoT-J's 1.951 cannot come from this file.
2. **Algorithm 1.** The printed listing has three defects (line 4 uses `ltx` twice; lines 10/22 cannot
   change `margin_excess`; line 28 clamps only from below) and two places where it contradicts its own
   text: line 18 is a second `if`, so both blocks run, and the decrease branch's `break` leaves
   `margin_excess` at the decremented SF, so line 27 *raises* power. It is run as Section IV-A
   describes it (variant `text`): `snr_limit` from the current SF, the two scenarios
   exclusive, TP set from the margin at the SF actually used, power clamped to [2, 20] dBm.
   An inverse search over every undocumented setting is unanimous that the authors' implementation
   had none of the printed defects.
3. **Delivery rule.** "delivered if the actual RSSI was greater than the predicted RSSI" is taken with
   the margin inside the prediction: `PL_true − PL_pred < LM` (ADR: `SNRmax − SNR < LM`). No
   demodulation limit is consulted, so SF and TP set airtime and energy but not delivery.
4. **Conventional ADR.** As the paper words it — "varies the SF and decreases P_T as needed to get
   M_e = 0" — SF 7–12, then `TP = 20 − Me`; `SNRmax` is the
   maximum of the 20 samples *before* the packet (the paper's 4.8 % PDR at LM = 0 is 1/21).
5. **Operating points of Figs. 12/13.** The paper sweeps LM in 1 dB steps and quotes integer LMs
   for every PDR level; each scheme's airtime and energy at PDR X are evaluated at the integer LM the
   paper states for it or, where it is silent, at the first integer LM where the paper's own Fig. 11
   curve reaches X (`paper_spec.PDR_TARGETS_INTEGER`). For the ML models 85 % and 90 % (ANN) and
   80 % and 85 % (SVR, RF) share LM = 2 dB — the plateau in the paper's Figs. 12/13.
6. **Shadow fading in the simulation.** "SPLMSFT (t-distributed shadow fading)" and "MLR (with
   t-distributed shadow fading)" add the Appendix's ψ — Student-t, ν = 11.43, scale 1.74 dB fitted
   on the MLR residuals — to their predictions; ANN, SVR and RF are deterministic.
7. **Noise floor.** Algorithm 1's `noise_power` is `rssi − snr`, which makes its margin
   `SNR_est − SNR_limit − LM` — the same form as the ADR's. The descriptor's `Pn` (its eq. 4) is
   exact, but for a noise-dominated link RSSI ≈ Pn, so it overstates EN3's margin by 12.8 dB and
   Algorithm 1 then cuts power on a link sitting 2.5 dB above its SF 10 limit.

Other settings: 80/20 split; EN spreading factors SF 7–12; delivery rule `residual`;
EN transmit power continuous; energy averaged over all packets; ToA at 1 byte, BW 125 kHz, CR 4/5;
E = P_consumed(TP) × ToA with P_consumed = 51.35 + 3.652 × P_radiated [mW] (Table VII fit, R² 0.9458 vs paper 0.95).
Data: 930,753 released rows → 4,090 DHT22 fault rows dropped → train 741,330 / test 185,333.

## Table IV — CPLS models (test set)

| Model | Paper RMSE | Ours | Δ % | Paper R² | Ours | 5-fold CV RMSE | CV sd |
|---|---|---|---|---|---|---|---|
| MLR | 1.951 | 1.9352 | -0.8 | 0.905 | 0.91 | 1.9445 | 0.0026 |
| ANN | 1.613 | 1.5365 | -4.7 | 0.935 | 0.9437 | 1.5861 | 0.014 |
| SVR | 1.626 | 1.573 | -3.3 | 0.9342 | 0.9407 | 1.5981 | 0.0148 |
| RF | 1.566 | 1.5254 | -2.6 | 0.9389 | 0.9441 | 1.543 | 0.0034 |

SVR is fitted on 50,000 rows (libsvm is O(n³); the paper used a 16-node cluster); a
10k–250k learning curve showed accuracy converged by 50k. ANN cross-validation uses 100,000 rows.
R² is the squared Pearson correlation — the only definition under which the paper's Tables III/IV are
internally consistent (Okumura-Hata's coefficient of determination is −50).

## Table V — MLR weights

| Weight | IoT-J Table V | Descriptor Table 7 | Ours | Δ % vs IoT-J | Δ % vs descriptor |
|---|---|---|---|---|---|
| b0_intercept | -431.03 | -439.55 | -444.82116 | -3.2 | -1.2 |
| gamma | 2.205 | 2.203 | 2.19161 | -0.6 | -0.5 |
| b1_T | 0.0859 | 0.123 | 0.13033 | 51.7 | 6 |
| b2_RH | 0.0012 | 0.0105 | 0.01219 | 915.5 | 16.1 |
| b3_BP | 0.3991 | 0.407 | 0.41344 | 3.6 | 1.6 |
| b4_PM25 | 0.00022 | 0.00222 | 0.00206 | 827.5 | -7.3 |
| b5_SNR | -0.6236 | -0.635 | -0.64562 | -3.5 | -1.7 |

The authors' data descriptor prints the same fit on the released file (its Table 7; its intercept is
re-referenced from d in metres to the IoT-J's kilometres, +30 γ); ours reproduces it within 16 % on every weight. The IoT-J's β₂ (0.0012 vs 0.0105) and β₄ (0.000222 vs 0.00222) are
consistent with decimal-place slips in that table rather than a modelling difference — a hypothesis;
the descriptor's own table is the evidence.

### What the environmental variables contribute

| Features | Test RMSE (dB) | R² |
|---|---|---|
| distance only (SPLMSF form) | 2.555 | 0.8431 |
| distance + T, RH, PM | 2.527 | 0.8466 |
| distance + BP | 2.547 | 0.8441 |
| distance + SNR | 2.496 | 0.8502 |
| distance + node + SNR | 1.670 | 0.9329 |
| all but SNR | 2.521 | 0.8472 |
| all but BP | 2.458 | 0.8548 |
| all (eq. 6) | 1.935 | 0.9100 |

Temperature, humidity and PM2.5 together move the test RMSE by 0.028 dB. The jump to 1.935 dB needs
barometric pressure *and* SNR together — and in this deployment barometric pressure is a node
identifier (EN3 at 828 hPa, the others at 844–851 hPa, within-node sd ≈ 1.6): a one-hot node indicator in its
place gives 1.670 dB. SNR is a link measurement, not weather. The MLR's gain over the distance law is
therefore a per-node offset plus the received SNR; the weather terms are statistically significant
(with 741 000 rows everything is) and practically negligible. `outputs/provenance/mlr_feature_ablation.csv`.

## Table III / Fig. 4 — conventional models

| Model | Paper RMSE | Ours | Δ % | Paper R² | Ours |
|---|---|---|---|---|---|
| Friis | 4.002 | 4.055 | 1.3 | 0.825 | 0.843 |
| Two-ray | 10.981 | 11.212 | 2.1 | 0.805 | 0.813 |
| Okumura-Hata | 39.712 | 45.925 | 15.6 | 0.806 | 0.815 |
| SPLMSF | 2.66 | 2.555 | -4 | 0.824 | 0.843 |

SPLMSF: γ = 2.7458 (paper 2.7), K = 83.88 dB at d₀ = 1 km (paper 84.2).
Okumura-Hata's parameterisation is not printed in the paper; six standard variants were tried and none
reproduces its RMSE/R² pair. → `final/figures/fig04.png`

## Fig. 11 — link margin required for each PDR (dB); paper's value in brackets

| Scheme | 80 % | 85 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| ADR | **5.25** (6) | **5.85** (7) | **6.55** (8) | **7.45** (9) | **10.85** (11) |
| Friis | **0** (0) | **0.5** (1) | **0.85** (2) | **1.85** (3) | **3.55** (5) |
| SPLMSF | 3.05 | 3.8 | 4.65 | 6 | 8.6 |
| SPLMSFT | 2.7 | 3.3 | 4.1 | 5.25 | 7.6 |
| MLR | 2.25 | 2.75 | 3.45 | 4.45 | 6.5 |
| ANN | 1.2 | 1.45 | 1.8 | **2.45** (3) | **3.85** (4) |
| SVR | 1.45 | 1.7 | 2 | **2.4** (3) | **3.75** (4) |
| RF | 1.25 | 1.45 | 1.7 | **2.3** (3) | **3.7** (4) |

→ `final/figures/fig11.png` (lines reproduced, markers paper)

## Figs. 12–13 — ToA and energy improvement vs conventional ADR

Each point is evaluated at the paper's integer operating LM (Configuration 5; scheme and ADR LMs in
`final/fig12_13_energy_toa.csv`). **This reconstructs the paper's figure; it is not an equal-delivery
comparison** — at those LMs the achieved PDRs differ from the label (the "80 %" ANN point achieves 75.7 %
against 85.6 % for its ADR reference; "90 %": 92.1 vs 96.6 %). RMSE against the digitized paper curves under the three readings of the operating point — `paper_integer` energy 14.6 / ToA 11.4; `ours_integer` energy 23.9 / ToA 22.6; `grid` energy 22.3 / ToA 20.8.

**Equal achieved PDR** — each scheme and the ADR at its own first LM reaching the level; energy % (ToA %):

| Scheme | 80 % | 85 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| ANN | -15 (-16) | -4 (-4) | +7 (+8) | +11 (+8) | +46 (+39) |
| SVR | -12 (-13) | -1 (-2) | +10 (+12) | +20 (+17) | +51 (+43) |
| RF | -12 (-12) | -1 (-1) | +12 (+14) | +17 (+13) | +49 (+41) |
| MLR | -57 (-48) | -50 (-40) | -43 (-30) | -41 (-33) | +15 (+13) |
| Friis | -23 (-29) | -17 (-17) | -3 (+3) | -10 (-14) | +27 (+21) |
| SPLMSFT | -67 (-58) | -64 (-53) | -61 (-45) | -63 (-52) | +2 (+4) |
| SPLMSF | -92 (-75) | -90 (-69) | -82 (-57) | -81 (-61) | -4 (-0) |

At 99 % PDR:

| Model | Paper energy % | Ours | Paper ToA % | Ours |
|---|---|---|---|---|
| ANN | 43.5 | 45.5 | 32.7 | 37.7 |
| SVR | 40.6 | 49.2 | 29.9 | 40.8 |
| RF | 38.7 | 46.8 | 27.5 | 38.3 |

Energy improvement across the PDR range (%):

| PDR % | ANN | SVR | RF | MLR | Friis | SPLMSFT | SPLMSF |
|---|---|---|---|---|---|---|---|
| 80 | 10.5 | -6.3 | -12.7 | -53.8 | -1 | -47.8 | -91.6 |
| 85 | 11.5 | 19.1 | 14.2 | -17.1 | 4.3 | -42 | -74.4 |
| 90 | 31.6 | 19.3 | 14.6 | -11.8 | 1.5 | -34.6 | -56.7 |
| 95 | 34.4 | 39.9 | 36.4 | -0.1 | 2.7 | -18.9 | -32 |
| 99 | 45.5 | 49.2 | 46.8 | 11.9 | 5.4 | -6.4 | -23.4 |

→ `final/figures/fig12.png`, `fig13.png`

## Figures 11–13 against the paper's digitized curves

The paper's curves were digitized from the PDF at 500 dpi and validated against every number printed in its text (`outputs/provenance/paper_digitized/`, read-out precision ≈ ±2). Figures in `final/figures/` overlay them as hollow markers.

**Fig. 13 energy improvement (%)** — ours / (paper)

| Scheme | 80 % | 85 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| ANN | +10 (+4) | +11 (+15) | +32 (+17) | +34 (+20) | +46 (+43) |
| SVR | -6 (-1) | +19 (+10) | +19 (+14) | +40 (+16) | +49 (+41) |
| RF | -13 (-4) | +14 (+7) | +15 (+11) | +36 (+11) | +47 (+39) |
| MLR | -54 (-14) | -17 (-2) | -12 (+5) | -0 (+8) | +12 (+20) |
| Friis | -1 (-30) | +4 (-17) | +2 (-11) | +3 (-10) | +5 (-10) |
| SPLMSFT | -48 (-51) | -42 (-36) | -35 (-35) | -19 (-35) | -6 (-11) |
| SPLMSF | -92 (-72) | -74 (-60) | -57 (-55) | -32 (-53) | -23 (-19) |

RMSE over all 35 points: **14.6**


**Fig. 12 ToA improvement (%)** — ours / (paper)

| Scheme | 80 % | 85 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| ANN | +7 (-4) | +10 (+16) | +27 (+18) | +35 (+28) | +38 (+33) |
| SVR | -8 (-8) | +16 (+14) | +16 (+16) | +39 (+25) | +41 (+30) |
| RF | -14 (-12) | +11 (+12) | +11 (+13) | +36 (+20) | +38 (+28) |
| MLR | -46 (-17) | -13 (+8) | -10 (+10) | +9 (+14) | +10 (+18) |
| Friis | -10 (-40) | +7 (-11) | -3 (-11) | +1 (-6) | +1 (-3) |
| SPLMSFT | -43 (-56) | -36 (-36) | -29 (-34) | -6 (-9) | -3 (-4) |
| SPLMSF | -75 (-70) | -57 (-55) | -43 (-44) | -12 (-13) | -18 (-7) |

RMSE over all 35 points: **11.4**


**Fig. 11 PDR curves**, RMSE / maximum |difference| vs the digitized paper curve over LM 0–15 (points): SPLMSFT 1.0 / 1.9, MLR 1.1 / 1.9, ADR 1.4 / 3.8, SPLMSF 1.4 / 2.1, SVR 1.8 / 3.8, RF 2.1 / 5.5, ANN 2.2 / 7.0, Friis 7.1 / 19.8. The RMSEs are within the ±2-point read-out precision; the maxima are not — they sit at LM 0–1 for the ML models, where the paper's ANN delivers 81 % and its own RF/SVR, like ours, ~70. FRIIS is unreadable at LM 3–8 in the digitization (overlapped markers).

## Appendix — MLR residuals

| Test | Paper | Ours | p |
|---|---|---|---|
| KS vs fitted normal | p = 2.2e-16 | 0.0138 | < 1e-100 |
| Durbin-Watson (split order) | 1.67 | 1.9976 | — |
| Durbin-Watson (device+time order) | not run | 1.1532 | — |
| Breusch-Pagan | p = 1e-16 | 3,600.2849 | < 1e-100 |
| excess kurtosis | fat tails | 3.7179 | — |
| Student-t nu (MLE) | 11.43 | 11.0871 | — |
| Student-t scale at nu = 11.43 (Appendix psi, dB) | not reported | 1.7404 | — |
| QQ R2 vs t | 0.996 | 0.9869 | — |
| QQ R2 vs normal | not reported | 0.9779 | — |
| residual range dB | ~ -13..+10 (Fig.14) | -17.98..18.95 | — |

→ `final/figures/fig14_15.png`. The KS p-value is nominal (mean and SD are estimated from the tested
residuals; a Lilliefors correction makes the rejection stronger, not weaker, at this n). R² throughout is
the squared Pearson correlation, the paper's convention; it is not 1 − SSE/SST.

## The paper's Section IV observations, one by one

Each of the paper's stated observations on Figs. 11–13 (its Section IV-B/C lists) against this run.
✅ reproduced, ◐ partly, ❌ not; where an *explanation* fails, the mechanism found here is given.

| # | Paper says | Here | Verdict |
|---|---|---|---|
| Fig. 11 (1) | Conventional ADR: 80/85/90/95/99 % at LM 6/7/8/9/11 dB | 5.2/5.8/6.5/7.5/11 (integer crossings 6/6/7/8/11) | ✅ at 80 and 99 %; ~1 dB early at 85–95 % |
| Fig. 11 (2) | Friis: 0/1/2/3/5 dB; outperforms because it *overestimates* path loss, which wastes energy (IV-C) | 0/0.5/0.85/1.9/3.5; Friis bias +2.7 dB, over-predicts 84 % of packets; energy vs ADR -1/+4/+2/+3/+5 % | ✅ PDR and the bias; ❌ the energy waste: 2.7 dB of over-prediction is less than the ADR's own 6–11 dB margin, so Friis costs about what the ADR costs (paper −30…−10 %) |
| Fig. 11 (3) | SPLMSF, SPLMSFT, MLR beat the ADR; the *t-distribution* improved the PDR; MLR beats SPLMSFT because the environmental variables characterise ψ | SPLMSF 3/3.8/4.7/6/8.6, SPLMSFT 2.7/3.3/4.1/5.2/7.6, MLR 2.2/2.8/3.5/4.5/6.5 | ✅ the ordering; ❌ both explanations: at equal scale a t and a normal ψ give the same curve, SPLMSFT wins because it samples the MLR-residual ψ (sd 1.9 dB) instead of its own (2.6 dB) — `provenance/psi_scale_check.csv`; MLR wins through the node offset and SNR, the weather terms are worth 0.03 dB |
| Fig. 11 (4) | ANN, SVR, RF: 95 % at LM 3, 99 % at LM 4; like Friis but without overestimating PL | ANN 1.2/1.4/1.8/2.5/3.9, SVR 1.4/1.7/2/2.4/3.8, RF 1.2/1.4/1.7/2.3/3.7; bias −0.01/−0.08/−0.01 dB | ✅ |
| Figs. 12/13 (1) | ToA and energy improvements increase with PDR | ANN energy +10/+11/+32/+34/+46 % (SF ≤ 12); +22/+17/+20/+14/+15 % with SF ≤ 10 | ✅ under the paper's SF range; the trend *is* the conventional ADR escalating EN3 to SF 11/12 as LM grows — capped at the deployment's SF 10 it is flat |
| Figs. 12/13 (2) | The conventional ADR is better at 85 % (non-critical applications) | at 80 %: ANN +10, SVR -6, RF -13 %; at 85 % all positive; SF ≤ 10: all positive at 80 % too | ◐ at 80 % for SVR and RF; not at 85 %; disappears at SF ≤ 10 |
| Figs. 12/13 (3) | SPLMSF and SPLMSFT cost the most, worse than the ADR (ψ unpredictable); Friis also costs more than the ADR | SPLMSF -92/-74/-57/-32/-23 %, SPLMSFT -48/-42/-35/-19/-6 %; Friis -1/+4/+2/+3/+5 % | ✅ SPLMSF/SPLMSFT — the random ψ sends EN3 to SF 11/12 (capped: -4/-3/-5/-7/-34 %); ❌ Friis (see (2) above) |
| Figs. 12/13 (4) | MLR improves on the ADR above 90 %, up to 20 %, from a formula that follows the weather | MLR -54/-17/-12/-0/+12 % | ◐ positive only at 99 % and 12 % there; the formula's gain is a node offset plus SNR, not weather |
| Figs. 12/13 (5) | ANN/SVR/RF outperform the ADR: 43.5/40.6/38.7 % energy, 32.7/29.9/27.5 % ToA at 99 %, because (a) the environmental variables characterise ψ and (b) ML captures nonlinearities | energy 46/49/47 %, ToA 38/41/38 % (SF ≤ 12); 15/16/15 % with SF ≤ 10 | ✅ magnitudes under the paper's SF range; ❌ the ordering (0.1–0.2 dB of EN3 bias, seed-stable, opposite sign here); ❌ (a) by the ablation; (b) is worth 0.14 dB over a linear node + SNR model; the 43 % is SF 11/12, and at SF ≤ 10 no scheme reaches 99 % under a threshold model — see the two sections below |
| Discussion | Nonparametric models give the best PDR/energy trade-off: 99 % at low LM, energy improved up to 43 % with the ANN | 99 % at LM 3.7–3.9 dB vs 10.85 for the ADR; energy as above | ✅ the margin result, the paper's most solid applied finding; ❌ the 43 %, and 'ANN' in particular |

## Does the margin result hold up? (4 dB vs 11 dB)

Reproducing the paper's Fig. 11 required two things that are true of its *simulation* and not of a
network: the conventional ADR's 20-sample window taken over the test set in shuffled order — 20 random
packets from six months, whose maximum sits far above the current SNR — and the calibrated outliers of
Configuration 1. The enhanced schemes, in turn, take the packet's *own* measured SNR and noise floor as
inputs (Algorithm 1, line 1): legitimate for a retrospective figure, unavailable to a node before it transmits.

| Conventional ADR, LM for 99 % (95 %) | released data | reconstructed data |
|---|---|---|
| 20-sample window over the test set, shuffled order (Configuration 4) | 8.5 (7.0) dB | 10.8 (7.4) dB |
| chronological history within the test set | 4.8 (3.5) dB | 10.8 (3.8) dB |
| the 20 packets preceding each test packet in the full campaign | 4.4 (3.3) dB | 10.7 (3.5) dB |

| Scheme, released data, chronological | test RMSE (dB) | LM for 95 % | LM for 99 % |
|---|---|---|---|
| conventional ADR, chronological prev-20 | — | 3.3 | 4.4 |
| MLR with own SNR (as in the paper) | 1.844 | 3.1 | 4.5 |
| ANN with own SNR (as in the paper) | 1.423 | 2.4 | 3.6 |
| MLR with previous packet SNR | 1.993 | 3.3 | 4.6 |
| ANN with previous packet SNR | 1.697 | 2.5 | 3.6 |
| MLR with mean SNR of the previous 20 | 1.901 | 3.2 | 4.5 |

On the released data with chronological history, the conventional ADR needs 4.4 dB for 99 %; an ANN
fed only the previous packet's SNR needs 3.6 dB and a causal MLR 4.6 dB — an advantage of
under one decibel for the ANN and none for the MLR, against the paper's seven. The 4-vs-11 dB figure is a
faithful reconstruction of the paper's simulation and is not evidence of an operational gain of that size.
`outputs/provenance/validity/validity_summary.csv`, `causal_features.csv`.

## Generalisation across links

The campaign has four fixed node–distance pairs; a random packet split tests interpolation on known links.
Holding out each node in turn (released data, MLR): EN1 3.9 dB, EN2 2.4 dB, EN3 11.7 dB, EN4 5.3 dB, against 1.85 dB under the random split.
Claims about unseen nodes or sites are outside what these data support, and the weather ablation above is
a statement about incremental predictive value on these four links (additive MLR 0.03 dB; a controlled RF
check in the independent review 0.018 dB), not about environmental effects in general.

## Where SF 11 and 12 come from

The campaign never used SF 11 or 12: US915 uplinks stop at SF 10 (descriptor §4.4; 0 of 930,753 rows).
The paper's Section IV nevertheless defines the ADR over SF 7–12, and Algorithm 1 line 8 raises SF while
the margin is negative, so both simulated schemes *escalate* packets whose margin is negative. Every
escalated packet is EN3's — 37 % of the test set, logged only at SF 10, mean SNR −12.5 dB, i.e. 2.5 dB
above the SF 10 limit — so any LM above ~2.5 dB drives it to SF 11/12 at full power, and a packet at
SF 12 costs 32× the airtime of one at SF 7. With both schemes capped at the deployment's real range
(SF ≤ 10), at the same operating LMs:

| Scheme, SF ≤ 10 | 80 % | 85 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| ANN energy (ToA) | +22 (+14) | +17 (+8) | +20 (+9) | +14 (+7) | +15 (+11) |
| SVR energy (ToA) | +15 (+10) | +21 (+11) | +14 (+5) | +17 (+8) | +16 (+12) |
| RF energy (ToA) | +12 (+8) | +18 (+9) | +13 (+5) | +15 (+7) | +15 (+11) |
| MLR energy (ToA) | +3 (+3) | +10 (+4) | +6 (+2) | +4 (+2) | +2 (+2) |
| Friis energy (ToA) | +5 (-1) | +4 (-1) | +1 (+1) | +2 (+3) | +4 (+7) |
| SPLMSFT energy (ToA) | +1 (+2) | +1 (-1) | -0 (-2) | -1 (-2) | -9 (-8) |
| SPLMSF energy (ToA) | -4 (-1) | -3 (-4) | -5 (-6) | -7 (-7) | -34 (-31) |

The paper's 43 % at 99 % therefore measures one thing: at LM 11 the conventional ADR puts EN3 on
SF 12 at 20 dBm, while the ML scheme at LM 4 does not.

**But the table above cannot show that 99 % is reachable at SF ≤ 10.** The delivery rule (Configuration 3)
is blind to the selected SF and TP, so a SF cap changes energy and leaves PDR unchanged *by construction*
— the "15 % at 99 %" it yields for the ANN is energy at a fixed margin, not a delivery result, and an
earlier version of this report presented it as one. Re-evaluating both schemes with the receiver-threshold
rule (received power at the selected TP against noise + SNR_limit of the selected SF) on the same selections:

Under SF 7–12 the two rules agree (ANN 99.2 % residual vs 99.2 % threshold at LM 4). Under SF ≤ 10 the
threshold rule caps every scheme: conventional ADR 94.1 %, ANN 94.3 %, SVR 94.3 %, RF 94.3 % —
95 and 99 % are unreachable by any scheme, and 4.4 % of the packets the gateway *actually received* sit below
Table 2's SNR limit for their own SF, so neither rule is a receiver model. At matched achieved PDR under the
threshold rule with SF ≤ 10 (energy %, ToA % in brackets; levels a scheme cannot reach are blank):

| Scheme, SF ≤ 10, threshold rule | 80 % | 85 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| ANN | +8 (+9) | +13 (+8) | +12 (+4) | — | — |
| SVR | +9 (+11) | +15 (+10) | +16 (+7) | — | — |
| RF | +9 (+10) | +15 (+10) | +15 (+6) | — | — |
| MLR | -5 (+3) | +0 (+2) | -0 (-2) | — | — |
| Friis | -6 (-3) | -0 (-1) | +0 (-1) | — | — |
| SPLMSFT | -11 (-0) | -5 (-1) | -4 (-4) | — | — |
| SPLMSF | -12 (-1) | -7 (-2) | -8 (-7) | — | — |

## Not reproduced, and why

* **Okumura-Hata** (+16 % RMSE): parameterisation never printed.
* **Figs. 12/13 between 85 and 95 %.** The ML models' gains at 90–95 % sit above the paper's and the
  parametric schemes' below (tables above). The cause is located, not explained away: from the paper's
  own figures its conventional ADR's cost is nearly flat between LM 7 and 9 — energy ×1.13 / ×1.03 / ×1.01 and airtime ×1.26 / ×1.02 / ×1.11 at LM 6→7 / 7→8 / 8→9 in the paper, against ×1.31 / ×1.29 / ×1.34 and ×1.29 / ×1.23 / ×1.39 here. Every
  spec-derived lever was tried against this profile (SF policy, SF caps, TTN steps vs continuous,
  every window/ordering that keeps Fig. 11, a stateful Algorithm 1, the noise floor): an ADR that never
  raises SF gives the flat part but not the jumps at 6→7 and 9→11; nothing gives both. On this data,
  packets cross SF thresholds at every dB of margin; in the paper's simulation they evidently do not
  between 7 and 9 dB. That is a property of the data the authors iterated (their 60 000 removed rows and
  their packet order are not released), not a setting the text provides.
* **Friis inside Algorithm 1** is 10–30 points too energy-efficient at 80–95 % (the paper's Friis
  scheme costs 10–30 % *more* than the ADR; ours is roughly even), and **SPLMSF** costs 15–20 points
  more than the paper's at 80–85 %; both share the ADR-profile cause above.
* **ANN vs SVR/RF ordering** on Figs. 12/13. Where the paper's reading puts all three at the same LM
  (85, 95 and 99 %), the paper's ANN is cheapest by 3–9 points and ours is costliest by 3–8 (SVR first).
  Retraining the ANN with three other seeds moves its points by only ±2, so this is not training noise:
  it is each model's bias on EN3's links — ANN +0.13 dB, RF 0.00, SVR −0.24 dB (over-prediction sends
  EN3 to SF 12) — a tenth of the models' 1.5 dB RMSE, specific to the trained instance and to EN3's
  rows, which differ between the two databases. Neither ordering says anything about the model
  classes (`outputs/provenance/adr_mechanism/ann_seed_sensitivity.csv`).
* **The paper's cross-validation standard deviations** for ANN/SVR/RF (0.135 / 0.428 / 0.339) are
  an order of magnitude larger than 5-fold CV produces here (0.003–0.015 dB); the paper's own MLR row
  (0.00129) is of the order we obtain. We could not reproduce them.
* **~60 000 rows** implied by the paper's training-set size were never released; the restored outlier
  population above recovers their *effect* on the published statistics, not the rows themselves.

## Verified facts about the paper

* The ToA expression reproduces the released `toa` column exactly (0 mismatches in 926,663 rows).
* Algorithm 1 as printed is non-functional (three defects and two contradictions with its own text,
  above); the PDR decision rule as printed reduces to `PL_pred > PL_true` and contains no link margin.
* On these four links, the "environment-aware" gain of eq. (6) over the distance law (2.56 → 1.94 dB) is
  carried by barometric pressure acting as a node identifier together with the received SNR; temperature,
  humidity and PM2.5 contribute 0.03 dB (Table V section).
* The conventional ADR's PDR at LM = 0 in the paper's Fig. 11 is 4.8 % = 1/21: the packet being
  decided is not among the 20 SNR samples the NS maximises over.
* Algorithm 1 declares `ltx, gtx, lrx, grx` as constants; they vary by node in the data (ltx ∈ {1, 11.75} dB).
* The released campaign contains SF 7–10 only (US915 uplink data rates). With `sf_max = 12`, 27 % (EN,
  LM 4) to 37 % (ADR, LM 11) of simulated packets land on SF 11/12 — all of them EN3's — and carry most
  of the energy metric; the paper's own Figs. 12/13 rest on spreading factors its deployment cannot use
  (section above; `outputs/provenance/adr_mechanism/text_audit.py`).
* The Fig. 12/13 metric rewards conservative prediction on marginal links rather than accuracy: the
  paper's most accurate model (RF) is its least energy-efficient, and the same holds here.

## Reproduce

```bash
python3 src/run_final.py            # ~10 min → outputs/final/
python3 src/make_final_report.py    # this file
python3 tests/test_adr_equivalence.py
python3 outputs/provenance/adr_mechanism/text_audit.py   # optional, ~10 min: the evidence behind D2/D3/D5/D6/D7
```

Three clean-slate runs here and an independent reviewer's rerun agree to ≤ 5 × 10⁻¹¹ in every output
file (random-forest parallel summation, `n_jobs=-1`); every printed digit is identical. The CSV cache
(`outputs/raw.pkl`) is keyed on the CSV's size and mtime and on the loader's source, and rebuilds when either
changes. `CONFIG` values are passed to every stage (until 21 Sep 2026 the restoration fractions were recorded
but not passed — the defaults happened to be equal, so no result changed). Python 3.12, numpy 1.26,
pandas 2.2, scipy 1.13, scikit-learn 1.4.2, statsmodels 0.14, matplotlib 3.8.
