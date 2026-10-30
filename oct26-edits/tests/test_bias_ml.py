import numpy as np
import pandas as pd

from planetix.bias.nulls import BiasNull
from planetix.ml import autoencoder_test, classifier_test, orbit_features


def _sample(rng, clustered):
    n = 25
    if clustered:
        Om, w = rng.normal(100, 15, n) % 360, rng.normal(300, 15, n) % 360
    else:
        Om, w = rng.uniform(0, 360, n), rng.uniform(0, 360, n)
    return pd.DataFrame({"designation": [f"o{k}" for k in range(n)], "a": rng.uniform(250, 800, n),
                         "e": rng.uniform(0.8, 0.95, n), "i": rng.uniform(5, 30, n), "Omega": Om, "omega": w})


def test_features_wrap_angles():
    f1 = orbit_features(300, 0.9, 10, 359.9, 0.0)
    f2 = orbit_features(300, 0.9, 10, 0.1, 0.0)
    assert np.allclose(f1, f2, atol=1e-2)


def test_classifier_test_detects_cluster_not_noise():
    rng = np.random.default_rng(0)
    s = _sample(rng, clustered=True)
    assert classifier_test(s, BiasNull.uniform(s, rng=rng), n_perm=40, rng=rng)["p"] < 0.05
    s = _sample(rng, clustered=False)
    assert classifier_test(s, BiasNull.uniform(s, rng=rng), n_perm=40, rng=rng)["p"] > 0.05


def test_autoencoder_test_runs_and_is_calibrated():
    rng = np.random.default_rng(1)
    s = _sample(rng, clustered=False)
    summary, per_obj = autoencoder_test(s, BiasNull.uniform(s, rng=rng), n_train=60, n_calib=100, rng=rng)
    assert 0.0 < summary["p"] <= 1.0 and len(per_obj) == len(s)
