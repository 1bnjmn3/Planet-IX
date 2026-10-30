"""Is the ETNO inclination distribution consistent with a standard hot disk, after bias?

Replaces old script 32, which compared the observed inclinations with a Rayleigh(10 deg)
population pushed through an invented RA/Dec box mask. Here each object's own detection
probability as a function of inclination, P_j(i), comes from the same selection model used
for the angles. For an intrinsic distribution f(i | sigma) proportional to
sin(i) exp(-i^2 / 2 sigma^2) (Brown 2001), the probability of object j's inclination is

    L_j(sigma) = f(i_j | sigma) P_j(i_j) / integral f(i | sigma) P_j(i) di,

which is fitted for sigma instead of being fixed in advance. The goodness of fit uses the
probability integral transform: u_j = CDF_j(i_j) is uniform under the model, so a
Kolmogorov-Smirnov test of the u_j tests the fitted model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from planetix.bias.selection import detection_vs_inclination

I_GRID = np.arange(0.5, 90.0, 1.0)


def hot_disk_pdf(i_deg, sigma):
    """Unnormalised sin(i) * exp(-i^2 / (2 sigma^2)), i and sigma in degrees."""
    i = np.asarray(i_deg, float)
    return np.sin(np.radians(i)) * np.exp(-0.5 * (i / sigma) ** 2)


def inclination_selection(sample: pd.DataFrame, model, i_grid=I_GRID, n_clones: int = 5000, rng=None):
    """Matrix P[j, k] proportional to P(detect object j | i = i_grid[k])."""
    rng = np.random.default_rng(rng)
    return np.array([detection_vs_inclination(o, model, i_grid, n_clones, rng) for _, o in sample.iterrows()])


def fit_hot_disk(i_obs, P=None, i_grid=I_GRID, sigmas=np.arange(4.0, 40.01, 0.25)):
    """Maximum-likelihood width of a hot-disk inclination distribution.

    ``P`` is the (n_obj, n_grid) selection matrix; ``None`` means no selection.
    Returns a dict with sigma_hat, its 95% likelihood interval, and the KS p-value of the
    probability-integral-transform values at sigma_hat.
    """
    i_obs = np.asarray(i_obs, float)
    P = np.ones((len(i_obs), len(i_grid))) if P is None else np.asarray(P, float)
    # Interpolate each object's selection at its own inclination.
    P_obs = np.array([np.interp(x, i_grid, row) for x, row in zip(i_obs, P)])
    loglik = []
    for s in sigmas:
        f = hot_disk_pdf(i_grid, s)
        norm = (P * f).sum(axis=1)  # grid spacing cancels in the ratio below
        loglik.append(np.sum(np.log(hot_disk_pdf(i_obs, s) * P_obs / norm)))
    loglik = np.array(loglik)
    best = int(np.argmax(loglik))
    inside = sigmas[loglik >= loglik[best] - 0.5 * stats.chi2.ppf(0.95, 1)]
    dens = P * hot_disk_pdf(i_grid, sigmas[best])
    cdf = np.cumsum(dens, axis=1) / dens.sum(axis=1, keepdims=True)
    u = np.array([np.interp(x, i_grid, c) for x, c in zip(i_obs, cdf)])
    return {
        "sigma_hat": float(sigmas[best]),
        "sigma_lo": float(inside.min()),
        "sigma_hi": float(inside.max()),
        "ks_p": float(stats.kstest(u, "uniform").pvalue),
        "loglik": loglik,
        "sigmas": sigmas,
    }
