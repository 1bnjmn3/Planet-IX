import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from planetix.stats.multivariate import (
    bootstrap_cluster_stability,
    bootstrap_loadings,
    calibrated_silhouette,
    gmm_bootstrap_lrt,
    gmm_select,
    orbital_features,
    pca_parallel_analysis,
)


@pytest.fixture(autouse=True)
def single_thread():
    # tiny problems: BLAS/OpenMP threading only adds overhead
    with threadpool_limits(1):
        yield


def blobs(n=30, sep=6.0, dim=3, seed=0):
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, dim))
    X[n // 2 :] += sep / np.sqrt(dim)  # shift along the diagonal: structure is in correlations
    return X


def test_orbital_features():
    df = pd.DataFrame(
        {"a": [100.0, 300, 500, 700], "e": [0.5, 0.6, 0.7, 0.8], "i": [10.0, 20, 15, 30],
         "Omega": [0.0, 90, 180, 270], "omega": [45.0, 135, 225, 315]}
    )
    X, names = orbital_features(df, standardize=False)
    assert names == ["log10_a", "e", "i", "sin_Omega", "cos_Omega", "sin_omega", "cos_omega"]
    assert X.shape == (4, 7)
    np.testing.assert_allclose(X[:, 0], np.log10(df["a"]))
    np.testing.assert_allclose(X[:, 3], np.sin(np.radians(df["Omega"])), atol=1e-12)
    np.testing.assert_allclose(X[:, 4], np.cos(np.radians(df["Omega"])), atol=1e-12)
    Xs, _ = orbital_features(df)
    np.testing.assert_allclose(Xs.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=0), 1, atol=1e-12)


def test_silhouette_calibration_on_noise():
    rej = 0
    for s in range(16):
        X = np.random.default_rng(100 + s).normal(size=(30, 5))
        r = calibrated_silhouette(X, n_null=79, rng=s)
        rej += r.p_value < 0.05
        assert r.p_value >= 1 / 80
    assert rej / 16 <= 0.25


def test_silhouette_power():
    for method in ("agglomerative", "kmeans"):  # kmeans is slower
        r = calibrated_silhouette(blobs(), n_null=49, method=method, rng=1)
        assert r.p_value < 0.05
        assert r.best_k == 2
        assert len(r.null_max_silhouettes) == 49 and len(r.labels) == 30


def test_pca_parallel_analysis():
    g = np.random.default_rng(3)
    noise = pca_parallel_analysis(g.normal(size=(40, 5)), n_perm=200, rng=0)
    assert noise.n_significant == 0
    f = g.normal(size=(40, 1))
    X = f @ np.ones((1, 5)) + 0.5 * g.normal(size=(40, 5))
    res = pca_parallel_analysis(X, n_perm=200, rng=0)
    assert res.n_significant >= 1
    assert res.components.shape == (5, 5) and res.p_values.shape == (5,)
    assert res.p_values[0] >= 1 / 201


def test_bootstrap_loadings_sign_aligned():
    g = np.random.default_rng(4)
    f = g.normal(size=(40, 1))
    X = f @ np.ones((1, 5)) + 0.3 * g.normal(size=(40, 5))
    r = bootstrap_loadings(X, n_components=2, n_boot=100, rng=0)
    assert r.mean.shape == r.std.shape == (2, 5)
    assert np.all(np.abs(r.mean[0]) > 0.3)  # no sign cancellation
    assert np.all(r.std[0] < 0.2)


def test_gmm_select_and_lrt():
    two = blobs(n=40, dim=2)
    sel = gmm_select(two, k_range=range(1, 4), rng=0)
    assert sel.best_k == 2 and set(sel.bic) == {1, 2, 3}
    one = np.random.default_rng(5).normal(size=(40, 2))
    assert gmm_bootstrap_lrt(one, 1, 2, n_boot=30, rng=0).p_value > 0.05
    r = gmm_bootstrap_lrt(two, 1, 2, n_boot=30, rng=0)
    assert r.p_value < 0.05 and len(r.null_lr) == 30


def test_cluster_stability():
    assert bootstrap_cluster_stability(blobs(), 2, n_boot=50, rng=0) > 0.8


def test_gmm_bootstrap_null_samples_differ():
    X = np.random.default_rng(0).normal(size=(30, 2))
    r = gmm_bootstrap_lrt(X, 1, 2, n_boot=10, rng=0)
    assert np.unique(np.round(r.null_lr, 8)).size > 5
