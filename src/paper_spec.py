"""Verbatim specification extracted from Gonzalez-Palacio et al. (2023), IEEE IoT-J 10(12):10725-10739.

Everything in this module is a number or rule *printed in the paper*.  Nothing here is
inferred.  Where the paper is silent, the choice is made in run_final.CONFIG and documented in the README.
"""

PAPER = "Gonzalez-Palacio, Tobon-Vallejo, Sepulveda-Cano, Rua, Le (2023) IEEE IoT-J 10(12) 10725-10739"
DOI = "10.1109/JIOT.2023.3239827"

# ---------------------------------------------------------------- Table II
# Locations, distances, antenna heights of the ENs.  d in m, h in m, Alt in m.
TABLE_II = {
    "GW":  {"d_m": None, "lat": 6.2700, "lon": -75.5479, "h_m": 5.0,  "alt_m": 1699},
    "EN1": {"d_m": 2140, "lat": 6.2654, "lon": -75.5664, "h_m": 40.0, "alt_m": 1476},
    "EN2": {"d_m": 3450, "lat": 6.2748, "lon": -75.5785, "h_m": 12.5, "alt_m": 1494},
    "EN3": {"d_m": 6100, "lat": 6.3164, "lon": -75.5764, "h_m": 8.0,  "alt_m": 1723},
    "EN4": {"d_m": 8260, "lat": 6.2320, "lon": -75.6117, "h_m": 12.0, "alt_m": 1566},
}

# ------------------------------------------------- Table III (test subset)
# "Performance of conventional path loss models".  NOTE: this table is *model
# performance*, not per-field descriptive statistics of the dataset.
TABLE_III = {
    "Friis":        {"rmse_db": 4.00,  "r2": 0.8242},
    "Two-ray":      {"rmse_db": 10.97, "r2": 0.8042},
    "Okumura-Hata": {"rmse_db": 39.69, "r2": 0.8056},
    "SPLMSF":       {"rmse_db": 2.66,  "r2": 0.8243},
}

# ---------------------------------------------------------------- Table IV
# "Models' performances".  s = std-dev of the RMSE across the 5 CV folds.
TABLE_IV = {
    "Friis":        {"train_rmse": 4.007,   "train_s": None,    "train_r2": 0.8241,
                     "test_rmse": 4.002,    "test_r2": 0.8246},
    "Two-ray":      {"train_rmse": 10.977,  "train_s": None,    "train_r2": 0.8040,
                     "test_rmse": 10.981,   "test_r2": 0.8049},
    "Okumura-Hata": {"train_rmse": 39.6951, "train_s": None,    "train_r2": 0.8054,
                     "test_rmse": 39.7115,  "test_r2": 0.8064},
    "SPLMSF":       {"train_rmse": 2.512,   "train_s": 0.013,   "train_r2": 0.8311,
                     "test_rmse": 2.66,     "test_r2": 0.8243},
    "MLR":          {"train_rmse": 1.951,   "train_s": 0.00129, "train_r2": 0.905,
                     "test_rmse": 1.951,    "test_r2": 0.905},
    "ANN":          {"train_rmse": 1.706,   "train_s": 0.135,   "train_r2": 0.935,
                     "test_rmse": 1.613,    "test_r2": 0.935},
    "SVR":          {"train_rmse": 2.021,   "train_s": 0.428,   "train_r2": 0.9376,
                     "test_rmse": 1.626,    "test_r2": 0.9342},
    "RF":           {"train_rmse": 1.919,   "train_s": 0.339,   "train_r2": 0.9458,
                     "test_rmse": 1.566,    "test_r2": 0.9389},
}

# ----------------------------------------------------------------- Table V
# Model weights for the MLR model, eq. (6):
#   PL = b0 + 10*gamma*log10(d) + 20*log10(f) + b1*T + b2*RH + b3*BP + b4*PM + b5*SNR + psi
# Unit check (Section "Stage 1" of the reproduction) shows d must be in **km**
# and f in **Hz** for these weights to reproduce the observed ~93 dB path loss.
TABLE_V = {
    "b0_intercept_db":     -431.03,
    "gamma_distance":         2.205,
    "b1_temperature_db_C":    0.0859,
    "b2_rel_humidity_db_pct": 0.0012,
    "b3_bar_pressure_db_hPa": 0.3991,
    "b4_pm25_db_ug_m3":       0.000222,
    "b5_snr_db":             -0.6236,
}

# --------------------------------------------------------------- Table VII
# Power versus current consumption of the LoPy ENs, supply voltage 3.3 V.
TABLE_VII = [(7, 20), (13, 29), (17, 87), (20, 120)]   # (dBm, mA)
SUPPLY_VOLTAGE_V = 3.3
TABLE_VII_R2_REPORTED = 0.95

# ------------------------------------------------------ Hyper-parameter grids
# Section III-A-2 (ANN), III-A-3 (SVR), III-A-4 (RF).
ANN_GRID = {
    "hidden_layer_sizes": [(5, 2), (10, 5), (15, 7), (20, 10), (10, 5, 2), (15, 7, 3), (20, 10, 5)],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "alpha": [0.0001, 0.05],          # "regressor learning rate alpha", default 0.0001
    "activation": ["relu"],
    "max_iter": 2000,
    "tol": 1e-4,
}
ANN_BEST = {"hidden_layer_sizes": (20, 10, 5), "learning_rate": "constant", "alpha": 0.0001,
            "activation": "relu", "max_iter": 2000, "tol": 1e-4}
ANN_BEST_CV_RMSE = 1.7058
ANN_BEST_CV_SD = 0.135

SVR_GRID = {"C": [0.01, 0.1, 1, 10, 1000], "kernel": ["rbf", "poly"],
            "gamma": [0.001, 0.01, 0.1, 1]}
SVR_BEST = {"C": 10, "kernel": "rbf", "gamma": 0.1}
SVR_BEST_CV_RMSE = 2.021
SVR_BEST_CV_SD = 0.428

RF_GRID = {"n_estimators": [100, 200, 300], "max_depth": [6, 9, 12],
           "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 10, 100, 1000]}
RF_BEST = {"n_estimators": 100, "max_depth": 9, "min_samples_leaf": 1, "min_samples_split": 100}
RF_BEST_CV_RMSE = 1.92
RF_BEST_CV_SD = 0.339

CV_FOLDS = 5
TRAIN_FRACTION = 0.80
PAPER_TRAINING_ROWS = 792_600       # "our training set has 792.600 rows"

# ------------------------------------------------------------ LoRaWAN radio
# Section IV: "SF can take values from 7 (SNRlimit = -7.5 dB) up to 12
# (SNRlimit = -20 dB) [29], with steps of -2.5 dB"
SNR_LIMIT_BY_SF = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
SF_MIN, SF_MAX = 7, 12
BANDWIDTH_HZ = 125_000
CODING_RATE = 1            # CR index -> 4/(4+CR) = 4/5
PAYLOAD_BYTES = 1          # "we normalized the calculation to 1 byte"
EXPLICIT_HEADER = True
CRC_ON = True
PREAMBLE_SYMBOLS = 8
GW_SENSITIVITY_DBM = -140

# ----------------------------------------------------------- Reported results
# Section IV-B, observations 1)-4)
PDR_TARGETS = {
    "ADR":   {80: 6, 85: 7, 90: 8, 95: 9, 99: 11},
    "Friis": {80: 0, 85: 1, 90: 2, 95: 3, 99: 5},
    "ANN":   {95: 3, 99: 4},
    "SVR":   {95: 3, 99: 4},
    "RF":    {95: 3, 99: 4},
}
# Integer operating points of Figs. 12/13.  The paper simulates LM = 0..15 dB in 1 dB steps and
# reads each PDR level as "PDR of X % is achieved with LM = n dB"; Figs. 12/13 can only have been
# computed at those integer LMs.  Text values as above; the rest are the first integer LM at which
# the digitized Fig. 11 curve (paper_digitized/) reaches the level.  Friis at
# LM >= 3 is hidden under other curves in the figure, so its text values are used throughout.
PDR_TARGETS_INTEGER = {
    "ADR":     {80: 6, 85: 7, 90: 8, 95: 9, 99: 11},
    "Friis":   {80: 0, 85: 1, 90: 2, 95: 3, 99: 5},
    "ANN":     {80: 1, 85: 2, 90: 2, 95: 3, 99: 4},
    "SVR":     {80: 2, 85: 2, 90: 3, 95: 3, 99: 4},
    "RF":      {80: 2, 85: 2, 90: 3, 95: 3, 99: 4},
    "MLR":     {80: 3, 85: 3, 90: 4, 95: 5, 99: 7},
    "SPLMSF":  {80: 4, 85: 5, 90: 6, 95: 7, 99: 12},
    "SPLMSFT": {80: 3, 85: 4, 90: 5, 95: 6, 99: 9},
}
# ------------------------------------------------- companion data descriptor (same authors, same data)
# Gonzalez-Palacio et al., "LoRaWAN Path Loss Measurements in an Urban Scenario Including Environmental
# Effects", Data 8(1):4, 2023, DOI 10.3390/data8010004 - the paper of the released CSV.
DESCRIPTOR = dict(
    rows=930_753, sf_used=(7, 8, 9, 10),                 # "We also varied the SF with values of 7, 8, 9, and 10, according to the spectrum usage regulations"
    band="US902-928", channels_mhz=(903.9, 904.1, 904.3, 904.5, 904.7, 904.9, 905.1, 905.3),
    pdr_per_en={"EN1": 95.1, "EN2": 85.2, "EN3": 81.6, "EN4": 86.35},
    en3="EN3 used only SF = 10 because its SNR distribution has a mean of -15 dB",
    outliers="Mahalanobis distance over 11 variables; a row is removed if Md > 29.59 (chi2, 10 dof, p = 0.001)",
    ldplm=dict(gamma=2.739, K=1.75, d0_m=1.0, train_rmse=2.465, test_rmse=2.46, test_r2=0.8529),
    mlr=dict(test_rmse=1.8401, test_r2=0.9177,
             b0_intercept=-505.64, gamma=2.203, b1_T=0.123, b2_RH=0.0105, b3_BP=0.407, b4_PM25=0.00222, b5_SNR=-0.635),   # its Table 7
    esp="RSSI + SNR - 10 log10(1 + 10^(SNR/10))", pn="RSSI - 10 log10(1 + 10^(SNR/10))",   # eqs. (3)-(4); ESP - Pn = SNR
    experimental_pl="printed as ptx + gtx + grx - lrx - rssi; the CSV column also subtracts ltx (verified, max |diff| 0.000 dB)",
)
# Section IV-C, observation 5) - improvements vs conventional ADR at PDR = 99 %
ENERGY_IMPROVEMENT_TARGETS = {"ANN": 43.5, "SVR": 40.6, "RF": 38.7}   # %
TOA_IMPROVEMENT_TARGETS = {"ANN": 32.7, "SVR": 29.9, "RF": 27.5}      # %

# --------------------------------------------------------- Appendix results
APPENDIX = {
    "ks_test_pvalue": 2.2e-16,          # normality rejected
    "durbin_watson": 1.67,              # residuals uncorrelated
    "breusch_pagan_pvalue": 1e-16,      # heteroscedastic
    "t_dof_nu": 11.43,                  # MLE degrees of freedom
    "t_qq_r2": 0.996,
}

# ------------------------------------------------- Conventional-ADR (TTN) rule
# Section IV: 1) NS collects 20 SNR samples and takes the maximum;
#             2) Me = SNRmax - SNRlimit - LM;  3) vary SF / decrease PT to get Me = 0.
ADR_SNR_WINDOW = 20
