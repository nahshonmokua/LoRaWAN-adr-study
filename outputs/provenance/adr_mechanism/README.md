# How the delivery rule, the conventional-ADR mechanism and the simulation's reading of Section IV were determined

All scored against the digitized curves in `../paper_digitized/` (Fig. 11: 16 LM points per scheme;
Figs. 12/13: 7 schemes x 5 PDR levels). RMSE in PDR points or improvement points.

| file | question | outcome |
|---|---|---|
| `adr_curve_fit.csv` | open-loop window readings vs closed-loop NS/device dynamics (`adr_closed_loop.py`) for the ADR's Fig. 11 curve | open loop, per-device window of 20 in test order: RMSE 1.85; every closed loop >= 4.0 |
| `adr_step_fit.csv` | TTN 3 dB steps under a demodulation-threshold rule | breaks Fig. 11 (RMSE 11-16) - resolved by the residual rule below |
| `fig1213_fit.csv` | sf_max x max_tp x ADR SF policy, threshold rule | nothing below energy RMSE 13; sf_max=10 fits 80 % but cannot reach 99 % |
| `residual_rule_fit.csv` | "actual RSSI > predicted RSSI - LM", i.e. PL_true - PL_pred < LM | Fig. 11 RMSE 3.8 -> 2.4 for the schemes; frees Figs. 12/13 from the PDR curve |
| `asym_fit.csv` | different SF ranges for EN and ADR | refuted (energy RMSE 42) |
| `factorial.csv` | 96 spec-derived configurations under the residual rule | winner: ADR with TTN nStep=floor(Me/3), SF 7-12 both ways, energy over all packets: energy RMSE 12.0 |
| `stateful_fit.csv` | is Algorithm 1 stateful - `current_tp`/`current_sf` = the previous packet's outputs per device (`adr_stateful.py`, assumption A13)? | refuted: carrying TP puts airtime 60-110 points below the ADR (ToA RMSE 65-95 vs 19; TP-before-SF ordering 22-26); carrying SF alone is a no-op on airtime. Run statelessly the algorithm is already the minimum-airtime policy under its margin, so the Fig. 12 residual is not an EN-policy question |

Also refuted along the way (numbers in the report): the CSV's ESP-referenced `pn` as noise floor
(+80 % energy), US915 SF <= 10 for the EN (kills the 99 % jump), 2 dB EN power steps, no EN SF
increase (+74 % energy), delivered-only energy averaging, evaluating all schemes at the ADR's LM.

## Statement-by-statement audit of Section IV (`text_audit.py`, determinations D2/D3/D5/D6/D7)

| file | paper statement | what changed |
|---|---|---|
| `text_audit_fig11.csv` | "MLR (with t-distributed shadow fading)", "SPLMSFT (t-distributed)"; Appendix psi ~ t(nu = 11.43) on the MLR residuals | one Appendix psi sampled for both schemes: Fig. 11 RMSE MLR 4.0 -> 1.2, SPLMSFT 1.6 -> 0.7 |
| `text_audit_fig11.csv` | "the NS collects 20 SNR samples and obtains the maximum" | window = the 20 samples before the packet: ADR PDR at LM 0 = 4.8 % (= 1/21, the paper's value), RMSE 2.1 -> 1.6 |
| `text_audit_scores.csv` | "varying LM from 0 to 15 dB"; "PDR of 80 % is achieved with LM = 6 dB, ..." (integers) | Figs. 12/13 read at the paper's integer LMs: ToA RMSE 22.3 -> 11.4 and the 85-95 % plateau of the ML models appears (85 and 90 % share LM = 2) |
| `text_audit_scores.csv` | IV-A: two exclusive scenarios, "decrease the SF and transmission power" | Algorithm 1 variant `text` (TP from the margin at the SF used) with the continuous ADR: energy/ToA RMSE 14.6/11.4; listing-literal TP with TTN nStep 15.4/12.5; mixed pairs 18.6-23.5 |
| `text_audit_adr_policy.csv` | "varies the SF and decreases PT as needed to get Me = 0" | re-run of the SF-policy factorial under the paper's reading: SF 7-12 both ways, all packets, continuous - unchanged |
| `adr_profile.csv` | Figs. 12/13 themselves | the paper's ADR energy grows x1.13 from LM 6 to 7, then x1.03, x1.01, then ~x1.4 to LM 11; ours grows x1.3 per dB. No SF policy or window that keeps Fig. 11 gives the flat part and the jumps together; the campaign never used SF 11/12 (0 % of rows) and the simulated schemes put 10-37 % of packets there |
| `ann_seed_sensitivity.csv` | is 'ANN best' a model-class property? (`ann_seed_sensitivity.py`) | three ANN seeds move its Fig. 13 points by +-2; at equal LM SVR is 3-8 points cheaper than the ANN, decided by EN3 bias (ANN +0.13, RF 0.00, SVR -0.24 dB) |
