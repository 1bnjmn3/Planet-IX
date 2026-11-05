"""Monte Carlo null distributions, look-elsewhere corrections and power.

Three questions keep coming up with 10-40 ETNOs, and each needs its own tool:

1. *Is this statistic unusual under the null?*  ``monte_carlo_test`` runs any
   statistic against any null sampler. Pass a uniform sampler to reproduce a
   classical test, or a bias-aware sampler (``resample_reference``,
   ``from_density``) to ask whether the clustering exceeds what survey
   selection alone produces.
2. *Did we pick the sample cuts after seeing the data?*  The old scripts tried
   a>150, a>230, a>250 and q>30, q>40 and reported whichever looked best.
   ``scan_cuts`` reports the local p-value of every cut and a global p-value
   in which the null realisations go through the same search.
3. *Could we have detected a real cluster at this N at all?*  ``rayleigh_power``
   gives the detection probability for a von Mises cluster of a given
   concentration, which is what a non-detection should be read against.

Monte Carlo p-values are (1 + k) / (1 + M), so they are never exactly 0 and are
valid (not anti-conservative) for any M.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from planetix.angles import wrap_deg
from planetix.stats.circular import rayleigh_test

Sampler = Callable[[np.random.Generator], np.ndarray]
Statistic = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------
# Null samplers: each returns a function rng -> array of n angles in degrees
# --------------------------------------------------------------------------

def uniform_angles(n: int) -> Sampler:
    """Isotropic null: n angles uniform on [0, 360)."""
    return lambda rng: rng.uniform(0.0, 360.0, n)


def resample_reference(reference_deg, n: int) -> Sampler:
    """Draw n angles with replacement from a reference population.

    The usual bias-aware null (Brown 2017 style): use angles of objects found
    by the same surveys but too close to be shepherded by a distant planet, so
    they carry the selection function but not the signal.
    """
    ref = np.asarray(reference_deg, dtype=float).ravel()
    return lambda rng: rng.choice(ref, size=n, replace=True)


def from_density(grid_deg, weights, n: int) -> Sampler:
    """Draw n angles from a tabulated selection density on a grid of bin centres.

    Angles are drawn uniformly within the chosen bin, so the grid spacing must
    be uniform.
    """
    grid = np.asarray(grid_deg, dtype=float).ravel()
    w = np.clip(np.asarray(weights, dtype=float).ravel(), 0, None)
    if grid.size != w.size or grid.size < 2 or w.sum() <= 0:
        raise ValueError("grid and weights must match and weights must be positive")
    step = np.diff(grid).mean()
    prob = w / w.sum()
    return lambda rng: wrap_deg(rng.choice(grid, size=n, p=prob) + rng.uniform(-step / 2, step / 2, n))


# --------------------------------------------------------------------------
# Generic Monte Carlo test
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloResult:
    statistic: float
    p_value: float
    n_sims: int
    p_ci95: tuple[float, float]
    null: np.ndarray


def mc_p_value(observed: float, null, alternative: str = "greater") -> float:
    """(1 + #{null at least as extreme}) / (1 + M)."""
    null = np.asarray(null, dtype=float)
    tol = 1e-12 * max(1.0, abs(observed))
    if alternative == "greater":
        k = np.sum(null >= observed - tol)
    elif alternative == "less":
        k = np.sum(null <= observed + tol)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return float((1 + k) / (1 + null.size))


def _binomial_ci(k: int, m: int) -> tuple[float, float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else beta.ppf(0.025, k, m - k + 1)
    hi = 1.0 if k == m else beta.ppf(0.975, k + 1, m - k)
    return float(lo), float(hi)


def monte_carlo_test(
    data,
    statistic: Statistic,
    sampler: Sampler,
    n_sims: int = 10000,
    alternative: str = "greater",
    rng=None,
) -> MonteCarloResult:
    """Compare ``statistic(data)`` with ``statistic(sampler(rng))`` over n_sims draws.

    ``p_ci95`` is the Clopper-Pearson interval on the p-value from finite
    n_sims; if it straddles your threshold, run more simulations.
    """
    rng = np.random.default_rng(rng)
    obs = float(statistic(np.asarray(data)))
    null = np.array([statistic(sampler(rng)) for _ in range(n_sims)], dtype=float)
    p = mc_p_value(obs, null, alternative)
    k = int(round(p * (n_sims + 1))) - 1
    return MonteCarloResult(statistic=obs, p_value=p, n_sims=n_sims, p_ci95=_binomial_ci(k, n_sims), null=null)


def rayleigh_mc(theta_deg, sampler: Sampler | None = None, n_sims: int = 100000, rng=None) -> MonteCarloResult:
    """Rayleigh R-bar against a null sampler (uniform if None), vectorised."""
    th = np.asarray(theta_deg, dtype=float).ravel()
    sampler = sampler or uniform_angles(th.size)
    rng = np.random.default_rng(rng)
    obs = rayleigh_test(th).R_bar
    # draw in chunks so 1e5 x n stays small in memory
    null = []
    for m in np.diff(np.r_[0:n_sims:20000, n_sims]):
        r = np.deg2rad(np.stack([sampler(rng) for _ in range(m)]))
        null.append(np.hypot(np.cos(r).mean(1), np.sin(r).mean(1)))
    null = np.concatenate(null)
    p = mc_p_value(obs, null)
    k = int(round(p * (n_sims + 1))) - 1
    return MonteCarloResult(statistic=obs, p_value=p, n_sims=n_sims, p_ci95=_binomial_ci(k, n_sims), null=null)


# --------------------------------------------------------------------------
# Look-elsewhere: many sample cuts tested on the same data
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CutScanResult:
    labels: list[str]
    n_per_cut: np.ndarray
    local_p: np.ndarray
    best_label: str
    best_local_p: float
    global_p: float
    n_sims: int


def scan_cuts(
    theta_deg,
    masks: Sequence[np.ndarray],
    labels: Sequence[str] | None = None,
    sampler: Sampler | None = None,
    n_sims: int = 20000,
    rng=None,
) -> CutScanResult:
    """Rayleigh test over several (usually nested) sample cuts, with a global p-value.

    ``theta_deg`` holds the angle for every object in the parent sample;
    ``masks`` are boolean arrays over that sample (for example a > 150,
    a > 230 & q > 30, a > 250 & q > 40). Under the null all parent angles are
    redrawn from ``sampler`` (uniform by default), every cut is re-tested,
    and the smallest local p-value is recorded. The global p-value is the
    fraction of null realisations whose best cut beats the observed best cut.
    Because the cuts share objects, this is much less severe than a Bonferroni
    correction but still honest.
    """
    th = np.asarray(theta_deg, dtype=float).ravel()
    M = np.asarray(masks, dtype=bool)
    if M.ndim != 2 or M.shape[1] != th.size:
        raise ValueError("masks must be a sequence of boolean arrays over theta")
    n = M.sum(axis=1)
    if np.any(n < 2):
        raise ValueError("every cut must keep at least two objects")
    labels = list(labels) if labels is not None else [f"cut{k}" for k in range(len(M))]
    sampler = sampler or uniform_angles(th.size)
    rng = np.random.default_rng(rng)
    Mf = M.astype(float)

    def local_p(rad):  # rad: (s, N) -> (s, K)
        C, S = np.cos(rad) @ Mf.T, np.sin(rad) @ Mf.T
        R2 = C**2 + S**2
        p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R2)) - (1 + 2 * n))
        return np.clip(p, 0.0, 1.0)

    obs = local_p(np.deg2rad(th)[None, :])[0]
    best = int(np.argmin(obs))
    null_min = []
    for m in np.diff(np.r_[0:n_sims:5000, n_sims]):
        r = np.deg2rad(np.stack([sampler(rng) for _ in range(m)]))
        null_min.append(local_p(r).min(axis=1))
    null_min = np.concatenate(null_min)
    return CutScanResult(
        labels=labels,
        n_per_cut=n,
        local_p=obs,
        best_label=labels[best],
        best_local_p=float(obs[best]),
        global_p=mc_p_value(obs[best], null_min, alternative="less"),
        n_sims=n_sims,
    )


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------

def rayleigh_power(n: int, kappa, alpha: float = 0.05, n_sims: int = 20000, rng=None) -> np.ndarray:
    """Probability that the Rayleigh test rejects at ``alpha`` for n draws from VM(0, kappa).

    ``kappa`` may be a scalar or array; returns an array of the same shape.
    For reference, a cluster with circular std ~70 deg is kappa ~ 1.
    """
    rng = np.random.default_rng(rng)
    kap = np.atleast_1d(np.asarray(kappa, dtype=float))
    out = np.empty(kap.shape)
    for j, k in enumerate(kap):
        r = rng.vonmises(0.0, k, size=(n_sims, n)) if k > 0 else rng.uniform(-np.pi, np.pi, (n_sims, n))
        R2 = np.cos(r).sum(1) ** 2 + np.sin(r).sum(1) ** 2
        p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R2)) - (1 + 2 * n))
        out[j] = np.mean(p < alpha)
    return out.reshape(np.shape(kappa))


def min_detectable_kappa(n: int, power: float = 0.8, alpha: float = 0.05, rng=None) -> float:
    """Smallest von Mises kappa detected with the given power at sample size n."""
    grid = np.linspace(0.05, 6.0, 120)
    pw = rayleigh_power(n, grid, alpha=alpha, n_sims=4000, rng=rng)
    hit = np.nonzero(pw >= power)[0]
    return float(grid[hit[0]]) if hit.size else float("inf")
