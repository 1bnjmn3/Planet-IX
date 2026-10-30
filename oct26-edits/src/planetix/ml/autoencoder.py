"""Autoencoder anomaly test, calibrated against a bias-aware null."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from planetix.ml.features import null_features, orbit_features


def _fit_autoencoder(X, hidden, seed):
    scaler = StandardScaler().fit(X)
    net = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", alpha=1e-4, max_iter=2000,
                       early_stopping=True, random_state=seed)
    net.fit(scaler.transform(X), scaler.transform(X))
    return scaler, net


def _errors(scaler, net, X):
    Z = scaler.transform(X.reshape(-1, X.shape[-1]))
    return ((net.predict(Z) - Z) ** 2).mean(axis=1).reshape(X.shape[:-1])


def autoencoder_test(sample: pd.DataFrame, null, n_train: int = 300, n_calib: int = 1000,
                     hidden=(16, 2, 16), rng=None):
    """Is the real sample harder to reconstruct than samples from ``null``?

    The autoencoder learns what orbits look like under the null (``n_train`` mock samples
    of the same objects). If the real orientations carry structure the null lacks, the real
    sample's mean reconstruction error should be unusually large compared with
    ``n_calib`` fresh mock samples.

    Returns (summary dict, per-object DataFrame). Per-object ``p_object`` compares each
    object's error with that object's own errors over the calibration mocks; with N ~ 30
    objects expect one or two below 0.05 by chance.
    """
    rng = np.random.default_rng(rng)
    seed = int(rng.integers(2**31))
    Om, w = null.draw(n_train, rng)
    scaler, net = _fit_autoencoder(null_features(sample, Om, w).reshape(-1, 9), hidden, seed)

    X_real = orbit_features(sample["a"], sample["e"], sample["i"], sample["Omega"], sample["omega"])
    err_real = _errors(scaler, net, X_real[None])[0]
    Om_c, w_c = null.draw(n_calib, rng)
    err_null = _errors(scaler, net, null_features(sample, Om_c, w_c))  # (n_calib, n_obj)

    score, null_scores = err_real.mean(), err_null.mean(axis=1)
    summary = {
        "score": float(score),
        "null_mean": float(null_scores.mean()),
        "p": float((1 + np.sum(null_scores >= score)) / (1 + n_calib)),
    }
    per_object = pd.DataFrame({
        "designation": sample["designation"].to_numpy(),
        "recon_error": err_real,
        "p_object": (1 + (err_null >= err_real[None, :]).sum(axis=0)) / (1 + n_calib),
    })
    return summary, per_object.sort_values("recon_error", ascending=False, ignore_index=True)
