"""Stage 3 - environment-aware Combined Path Loss and Shadowing (CPLS) models.

    CPLS = f(d, f, T, RH, BP, PM, SNR) + psi            (paper eq. 3)

Four regressors, matching Section III-A of the paper: MLR (parametric, eq. 6),
ANN (sklearn MLPRegressor), SVR and RF.

Dataset-agnostic on purpose: every estimator here takes a DataFrame whose columns
follow `FeatureSpec`, so a later study can point the same code at different data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 42


@dataclass
class FeatureSpec:
    """Maps the seven paper predictors onto column names + the units eq. (6) needs.

    `distance_scale` converts the distance column into the unit eq. (6) is written
    in.  The released CSV stores metres and eq. (6) needs kilometres (A4).
    """
    distance: str = "distance"
    frequency: str = "frequency"
    temperature: str = "temperature"
    rel_humidity: str = "rh"
    bar_pressure: str = "bp"
    pm25: str = "pm2_5"
    snr: str = "snr"
    target: str = "experimental_pl"
    distance_scale: float = 1e-3        # m -> km
    frequency_scale: float = 1.0        # already Hz

    @property
    def predictors(self) -> list[str]:
        return [self.distance, self.frequency, self.temperature, self.rel_humidity,
                self.bar_pressure, self.pm25, self.snr]


DEFAULT_SPEC = FeatureSpec()


def make_X(df: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> np.ndarray:
    """Raw 7-column design matrix in the units the paper's equations use."""
    return np.column_stack([
        df[spec.distance].to_numpy(float) * spec.distance_scale,
        df[spec.frequency].to_numpy(float) * spec.frequency_scale,
        df[spec.temperature].to_numpy(float),
        df[spec.rel_humidity].to_numpy(float),
        df[spec.bar_pressure].to_numpy(float),
        df[spec.pm25].to_numpy(float),
        df[spec.snr].to_numpy(float),
    ])


def make_y(df: pd.DataFrame, spec: FeatureSpec = DEFAULT_SPEC) -> np.ndarray:
    return df[spec.target].to_numpy(float)


FEATURE_NAMES = ["d_km", "f_Hz", "T_C", "RH_pct", "BP_hPa", "PM25", "SNR_dB"]


# ---------------------------------------------------------------------- MLR
class MLRCpls(BaseEstimator, RegressorMixin):
    """Paper eq. (6), fitted exactly as written:

        PL = b0 + 10*gamma*log10(d) + 20*log10(f) + b1*T + b2*RH + b3*BP
             + b4*PM + b5*SNR + psi

    The frequency coefficient is FIXED at 20 (Friis), so 20*log10(f) is an offset,
    not a regressor.  The distance regressor is 10*log10(d) and its coefficient is
    the fitted path loss exponent gamma.  Everything else is ordinary least squares.
    """
    def __init__(self, fit_frequency: bool = False):
        self.fit_frequency = fit_frequency

    @staticmethod
    def _design(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d_km, f_hz = X[:, 0], X[:, 1]
        offset = 20.0 * np.log10(f_hz)               # fixed Friis term
        A = np.column_stack([
            np.ones(len(X)),                          # b0
            10.0 * np.log10(d_km),                    # gamma
            X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6],   # b1..b5
        ])
        return A, offset

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        A, offset = self._design(X)
        if self.fit_frequency:
            A = np.column_stack([A, 20.0 * np.log10(X[:, 1])])
            offset = np.zeros(len(X))
        coef, *_ = np.linalg.lstsq(A, y - offset, rcond=None)
        self.coef_ = coef
        self.weights_ = dict(zip(
            ["b0_intercept", "gamma", "b1_T", "b2_RH", "b3_BP", "b4_PM25", "b5_SNR"]
            + (["freq_coef"] if self.fit_frequency else []), coef))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        A, offset = self._design(X)
        if self.fit_frequency:
            A = np.column_stack([A, 20.0 * np.log10(X[:, 1])])
            offset = np.zeros(len(X))
        return A @ self.coef_ + offset


# ------------------------------------------------------------ ANN / SVR / RF
def make_ann(hidden_layer_sizes=(20, 10, 5), alpha=1e-4, learning_rate="constant",
             activation="relu", max_iter=2000, tol=1e-4, solver="adam",
             random_state=SEED, **kw) -> Pipeline:
    """MLPRegressor behind the StandardScaler the paper describes (Fig. 5, Fig. 6)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                               learning_rate=learning_rate, activation=activation,
                               max_iter=max_iter, tol=tol, solver=solver,
                               random_state=random_state, **kw)),
    ])


def make_svr(C=10.0, kernel="rbf", gamma=0.1, epsilon=0.1, **kw) -> Pipeline:
    """SVR behind a StandardScaler.  The paper scales inputs only, not the target (A11)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon, **kw)),
    ])


def make_rf(n_estimators=100, max_depth=9, min_samples_leaf=1, min_samples_split=100,
            random_state=SEED, n_jobs=-1, **kw) -> RandomForestRegressor:
    """RF with no scaler - 'The RF algorithm is based on decision rules, so using the
    standard scaler for the input features is unnecessary.'"""
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,
                                 random_state=random_state, n_jobs=n_jobs, **kw)


BUILDERS = {"MLR": lambda **kw: MLRCpls(**kw), "ANN": make_ann, "SVR": make_svr, "RF": make_rf}


# ------------------------------------------------------------------ wrapper
@dataclass
class CplsModel:
    """A fitted CPLS predictor that speaks DataFrames, for use by adr_algorithm.py."""
    name: str
    estimator: object
    spec: FeatureSpec = field(default_factory=FeatureSpec)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(make_X(df, self.spec)), float)

    def predict_row(self, d, f, T, RH, BP, PM, SNR) -> float:
        """Scalar interface matching Algorithm 1's `cpls_model(d, f, T, RH, BP, PM, SNR)`.
        `d` is expected in the SAME unit as the training column (metres for this CSV)."""
        X = np.array([[d * self.spec.distance_scale, f * self.spec.frequency_scale,
                       T, RH, BP, PM, SNR]], float)
        return float(self.estimator.predict(X)[0])
