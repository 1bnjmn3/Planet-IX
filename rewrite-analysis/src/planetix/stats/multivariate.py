"""Calibrated multivariate statistics (PCA, clustering, mixtures) for small N.

With 10-40 objects, naive diagnostics look impressive on pure noise: the best
silhouette over a k-search is routinely > 0.3, a BIC-selected GMM happily adds
components, and PCA always produces a "leading" component. A threshold such as
"silhouette > 0.5 means real structure" is therefore uncalibrated. Every test
here compares the observed statistic against a null distribution generated
with the *same* analysis pipeline (including the k-search), so the p-value
already accounts for the look-elsewhere effect.

Conventions: input angles are degrees; random functions take ``rng`` (int,
None or Generator); Monte Carlo p-values are ``(1 + #{null >= obs}) / (1 + n)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

RngLike = int | None | np.random.Generator

_ANGLES = ("Omega", "omega", "varpi")


def _rng(rng: RngLike) -> np.random.Generator:
    return np.random.default_rng(rng)


def _seed(g: np.random.Generator) -> int:
    return int(g.integers(0, 2**31 - 1))


def _mc_p(null: np.ndarray, observed: float) -> float:
    """Monte Carlo p-value, never zero."""
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))


def _permute_columns(X: np.ndarray, g: np.random.Generator) -> np.ndarray:
    """Permute each column independently (keeps marginals, destroys dependence)."""
    return np.column_stack([g.permutation(X[:, j]) for j in range(X.shape[1])])


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def orbital_features(
    df: pd.DataFrame,
    columns: Sequence[str] = ("a", "e", "i", "Omega", "omega"),
    log_a: bool = True,
    standardize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build a feature matrix from orbital elements.

    Angles (Omega, omega, varpi; degrees) become ``sin_``/``cos_`` pairs so
    0/360 is continuous; ``i`` stays in linear degrees; ``a`` becomes
    ``log10(a)`` when ``log_a``. Columns are standardized to zero mean and
    unit variance (ddof=0) when ``standardize``.

    Returns
    -------
    X : ndarray, shape (n_objects, n_features)
    names : list of feature names matching the columns of ``X``.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    for c in columns:
        v = df[c].to_numpy(dtype=float)
        if c in _ANGLES:
            r = np.radians(v)
            cols += [np.sin(r), np.cos(r)]
            names += [f"sin_{c}", f"cos_{c}"]
        elif c == "a" and log_a:
            cols.append(np.log10(v))
            names.append("log10_a")
        else:
            cols.append(v)
            names.append(c)
    X = np.column_stack(cols)
    if standardize:
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return X, names


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------
def _pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (explained variance ratio, components) via SVD of centered X."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s**2
    return ev / ev.sum(), vt


@dataclass(frozen=True)
class PCAResult:
    """Parallel-analysis PCA result.

    ``components`` has shape (n_comp, n_feat). ``null_mean``/``null_95`` are the
    mean and 95th percentile of the explained-variance ratio of the same-rank
    component under the column-permutation null. ``p_values`` are per component.
    """

    explained_variance_ratio: np.ndarray
    components: np.ndarray
    null_mean: np.ndarray
    null_95: np.ndarray
    p_values: np.ndarray
    n_significant: int


def pca_parallel_analysis(X: np.ndarray, n_perm: int = 1000, rng: RngLike = None) -> PCAResult:
    """PCA with Horn's parallel analysis.

    Null hypothesis: the features are mutually independent. Each column is
    permuted independently (marginals kept, correlations destroyed) and PCA is
    redone; component j is significant if its variance ratio exceeds that of the
    j-th null component (p < 0.05). ``n_significant`` counts leading components
    significant in an unbroken run from PC1.
    """
    g = _rng(rng)
    X = np.asarray(X, dtype=float)
    evr, comps = _pca(X)
    null = np.array([_pca(_permute_columns(X, g))[0] for _ in range(n_perm)])
    p = np.array([_mc_p(null[:, j], evr[j]) for j in range(len(evr))])
    n_sig = 0
    for pj in p:
        if pj >= 0.05:
            break
        n_sig += 1
    return PCAResult(evr, comps, null.mean(axis=0), np.percentile(null, 95, axis=0), p, n_sig)


@dataclass(frozen=True)
class LoadingStability:
    """Bootstrap mean and std of sign-aligned loadings, shape (n_comp, n_feat)."""

    mean: np.ndarray
    std: np.ndarray


def bootstrap_loadings(
    X: np.ndarray, n_components: int = 2, n_boot: int = 1000, rng: RngLike = None
) -> LoadingStability:
    """Case-resampling bootstrap of PCA loadings.

    Each bootstrap component's sign is flipped to agree with the full-sample
    component (dot product >= 0), since PCA signs are arbitrary. Large std
    relative to the mean means the loading is not stable at this N.
    """
    g = _rng(rng)
    X = np.asarray(X, dtype=float)
    n = len(X)
    ref = _pca(X)[1][:n_components]
    out = np.empty((n_boot, *ref.shape))
    for b in range(n_boot):
        comp = _pca(X[g.integers(0, n, n)])[1][:n_components]
        sign = np.where(np.sum(comp * ref, axis=1) < 0, -1.0, 1.0)
        out[b] = comp * sign[:, None]
    return LoadingStability(out.mean(axis=0), out.std(axis=0))


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
def _cluster(X: np.ndarray, k: int, method: str, seed: int) -> np.ndarray:
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    raise ValueError(f"unknown method {method!r}")


def _sil(X: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(X, labels)) if 1 < len(np.unique(labels)) < len(X) else -1.0


def _best_over_k(X, ks, method, seed):
    scores, labs = {}, {}
    for k in ks:
        labs[k] = _cluster(X, k, method, seed)
        scores[k] = _sil(X, labs[k])
    best = max(scores, key=scores.get)
    return best, scores, labs[best]


@dataclass(frozen=True)
class ClusterTestResult:
    """Silhouette cluster test: observed search result plus null calibration."""

    best_k: int
    best_silhouette: float
    silhouettes: dict[int, float]
    null_max_silhouettes: np.ndarray
    p_value: float
    labels: np.ndarray


def calibrated_silhouette(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 6),
    method: str = "agglomerative",
    n_null: int = 500,
    rng: RngLike = None,
) -> ClusterTestResult:
    """Silhouette cluster test calibrated against a no-structure null.

    Null hypothesis: features are independent (column-permuted data). For each
    null dataset the whole k-search is repeated and the *maximum* silhouette over
    ``k_range`` is recorded, so choosing the best k is part of the null
    (look-elsewhere correction). ``p_value`` is the Monte Carlo tail
    probability of the observed best silhouette. ``method`` is
    "agglomerative" or "kmeans".
    """
    g = _rng(rng)
    X = np.asarray(X, dtype=float)
    ks = list(k_range)
    best_k, scores, labels = _best_over_k(X, ks, method, _seed(g))
    null = np.array(
        [
            max(_best_over_k(_permute_columns(X, g), ks, method, _seed(g))[1].values())
            for _ in range(n_null)
        ]
    )
    best = scores[best_k]
    return ClusterTestResult(best_k, best, scores, null, _mc_p(null, best), labels)


def bootstrap_cluster_stability(
    X: np.ndarray, k: int, method: str = "agglomerative", n_boot: int = 200, rng: RngLike = None
) -> float:
    """Mean adjusted Rand index between full-sample and bootstrap clusterings.

    Each resample is clustered into ``k`` groups; its labels are compared with
    the full-sample labels of the same (resampled) original points. Values near
    1 mean the partition is reproducible; near 0 means it is not distinguishable
    from arbitrary splitting.
    """
    g = _rng(rng)
    X = np.asarray(X, dtype=float)
    n = len(X)
    full = _cluster(X, k, method, _seed(g))
    aris = []
    for _ in range(n_boot):
        idx = g.integers(0, n, n)
        lab = _cluster(X[idx], k, method, _seed(g))
        aris.append(adjusted_rand_score(full[idx], lab))
    return float(np.mean(aris))


# --------------------------------------------------------------------------
# Gaussian mixtures
# --------------------------------------------------------------------------
def _gmm(X, k, cov, seed, n_init=5) -> GaussianMixture:
    return GaussianMixture(k, covariance_type=cov, n_init=n_init, random_state=seed).fit(X)


@dataclass(frozen=True)
class GMMSelection:
    """BIC per component count and the minimizing ``best_k``."""

    bic: dict[int, float]
    best_k: int


def gmm_select(
    X: np.ndarray,
    k_range: Iterable[int] = range(1, 6),
    covariance_type: str = "full",
    rng: RngLike = None,
) -> GMMSelection:
    """Choose the GMM component count by BIC (lower is better).

    BIC is a model-selection heuristic, not a test: it has no null and tends to
    add components for non-Gaussian single populations. In particular, a
    (sin, cos) angle pair lies on a circle, which no single Gaussian fits, so
    BIC keeps adding components to trace the circle even for uniform angles. Use ``gmm_bootstrap_lrt``
    to ask whether extra components are actually needed.
    """
    g = _rng(rng)
    X = np.asarray(X, dtype=float)
    bic = {k: float(_gmm(X, k, covariance_type, _seed(g)).bic(X)) for k in k_range}
    return GMMSelection(bic, min(bic, key=bic.get))


@dataclass(frozen=True)
class LRTResult:
    """Observed likelihood-ratio statistic, its parametric-bootstrap null and p."""

    lr_stat: float
    p_value: float
    null_lr: np.ndarray


def gmm_bootstrap_lrt(
    X: np.ndarray,
    k0: int,
    k1: int,
    n_boot: int = 200,
    covariance_type: str = "full",
    rng: RngLike = None,
) -> LRTResult:
    """Parametric-bootstrap likelihood-ratio test of k0 vs k1 mixture components.

    Null hypothesis: the data come from a ``k0``-component GMM. Wilks' chi-square
    approximation is invalid for mixtures (the extra weight sits on the boundary
    and parameters are unidentified under the null), so the null distribution of
    ``2*(ll1 - ll0)`` is built by simulating from the fitted k0 model and refitting
    both models to each simulated sample. The null is only as good as the k0
    model: Gaussian draws do not lie on the circle that a (sin, cos) pair
    occupies, so on angle features this test is mis-calibrated. Use a von Mises
    mixture (``planetix.stats.mixtures``) or ``calibrated_silhouette`` there.
    """
    g = _rng(rng)
    X = np.asarray(X, dtype=float)
    n = len(X)

    def lr(data: np.ndarray) -> float:
        ll0 = _gmm(data, k0, covariance_type, _seed(g)).score(data) * len(data)
        ll1 = _gmm(data, k1, covariance_type, _seed(g)).score(data) * len(data)
        return 2.0 * (ll1 - ll0)

    obs = lr(X)
    model0 = _gmm(X, k0, covariance_type, _seed(g))

    def draw() -> np.ndarray:
        # GaussianMixture.sample reuses its fixed random_state, so reseed each draw
        model0.random_state = _seed(g)
        return model0.sample(n)[0]

    null = np.array([lr(draw()) for _ in range(n_boot)])
    return LRTResult(float(obs), _mc_p(null, obs), null)
