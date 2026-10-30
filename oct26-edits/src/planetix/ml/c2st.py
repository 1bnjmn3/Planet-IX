"""Classifier two-sample test of real orbits against a bias-aware null."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from planetix.ml.features import null_features, orbit_features


def _cv_auc(X_pos, X_neg, model, seed):
    X = np.vstack([X_pos, X_neg])
    y = np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, prob)


def classifier_test(sample: pd.DataFrame, null, n_neg_per_obj: int = 20, n_perm: int = 200,
                    model=None, rng=None) -> dict:
    """Cross-validated AUC for telling the real sample from null mocks, with its p-value.

    Each object's real orbit is a positive; ``n_neg_per_obj`` null draws of the same object
    are negatives, so a, e, i match exactly and only the orientations (and how they relate
    to a, e, i) can separate the classes. The AUC distribution under the null comes from
    repeating the procedure with a fresh null draw in place of the real sample.

    The default classifier is a class-balanced logistic regression on the (cos, sin)
    features: it detects a shift of the mean orientation, which is what clustering is.
    """
    rng = np.random.default_rng(rng)
    model = model or make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
    n = len(sample)

    def neg():
        Om, w = null.draw(n_neg_per_obj, rng)
        return null_features(sample, Om, w).reshape(-1, 9)

    X_real = orbit_features(sample["a"], sample["e"], sample["i"], sample["Omega"], sample["omega"])
    auc = _cv_auc(X_real, neg(), model, int(rng.integers(2**31)))
    null_auc = []
    for _ in range(n_perm):
        Om, w = null.draw(1, rng)
        null_auc.append(_cv_auc(null_features(sample, Om, w)[0], neg(), model, int(rng.integers(2**31))))
    null_auc = np.array(null_auc)
    return {"auc": float(auc), "null_mean": float(null_auc.mean()), "null_95": float(np.quantile(null_auc, 0.95)),
            "p": float((1 + np.sum(null_auc >= auc)) / (1 + n_perm)), "n_obj": n}
