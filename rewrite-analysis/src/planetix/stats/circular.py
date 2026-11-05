"""Circular and spherical statistics for orbital angles.

Every public function takes angles in degrees. Results that need a p-value
return a small frozen dataclass so callers never have to remember tuple order.

Why these tests and not the ones in the old scripts
---------------------------------------------------
* The Kolmogorov-Smirnov test (old script 04) is not rotation invariant on a
  circle: the same data gives a different p-value depending on where 0 deg is.
  Kuiper's V is the circular analogue and is used here instead.
* The Rayleigh p-value here uses the Greenwood-Durand formula (Zar 1999,
  eq. 27.4), which is accurate to <1% for n >= 5. The series in old script 05
  is fine for large n but drifts at small n.
* Both tests assume the null is *uniform*. Survey selection makes the observed
  ETNO angles non-uniform even without a planet, so a small uniform-null
  p-value is evidence of non-uniformity, not of a perturber. Use
  ``planetix.stats.nulls`` with a bias-aware sampler for that question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from planetix.angles import wrap_deg


def _rad(theta_deg) -> np.ndarray:
    th = np.deg2rad(np.asarray(theta_deg, dtype=float).ravel())
    if th.size == 0:
        raise ValueError("need at least one angle")
    if not np.all(np.isfinite(th)):
        raise ValueError("angles must be finite")
    return th


def _wrap_signed(x_deg):
    """Wrap degrees into [-180, 180)."""
    return np.mod(np.asarray(x_deg, dtype=float) + 180.0, 360.0) - 180.0


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------

def mean_resultant_length(theta_deg) -> float:
    """Length R-bar of the mean unit vector, in [0, 1]. 0 = no preferred direction."""
    th = _rad(theta_deg)
    return float(np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th))))


def mean_direction(theta_deg) -> float:
    """Circular mean in degrees, in [0, 360). Undefined (nan) when R-bar is 0."""
    th = _rad(theta_deg)
    c, s = np.mean(np.cos(th)), np.mean(np.sin(th))
    if np.hypot(c, s) < 1e-12:
        return float("nan")
    return float(wrap_deg(np.rad2deg(np.arctan2(s, c))))


def circular_std(theta_deg) -> float:
    """Circular standard deviation sqrt(-2 ln R-bar), in degrees."""
    r = np.clip(mean_resultant_length(theta_deg), 1e-12, 1.0)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(r))))


def _a1inv(r: float) -> float:
    """Inverse of A1(kappa) = I1(kappa)/I0(kappa) (Fisher 1993, eq. 2.4)."""
    if r < 0.53:
        return 2 * r + r**3 + 5 * r**5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r**3 - 4 * r**2 + 3 * r)


def kappa_mle(theta_deg, bias_correct: bool = True) -> float:
    """Von Mises concentration estimate.

    The MLE is strongly biased upwards for small n; with ``bias_correct`` the
    Best & Fisher (1981) correction is applied for n < 16, which matters for
    the 10-40 object samples used here.
    """
    n = np.asarray(theta_deg).size
    k = _a1inv(min(mean_resultant_length(theta_deg), 1 - 1e-9))
    if bias_correct and n < 16:
        k = max(k - 2 / (n * k), 0.0) if k < 2 else (n - 1) ** 3 * k / (n**3 + n)
    return float(k)


def vonmises_logpdf(theta_deg, mu_deg: float, kappa: float) -> np.ndarray:
    """Log density of a von Mises distribution with respect to radians."""
    th, mu = _rad(theta_deg), np.deg2rad(mu_deg)
    # log I0(k) computed stably as log(i0e(k)) + k
    return kappa * (np.cos(th - mu) - 1) - np.log(2 * np.pi * special.i0e(kappa))


# --------------------------------------------------------------------------
# Vectorised statistics, for evaluating many simulated samples at once.
# Each takes an array of angles in degrees and reduces along ``axis``.
# --------------------------------------------------------------------------

def resultant_length(theta_deg, axis: int = -1) -> np.ndarray:
    """Mean resultant length R-bar along ``axis`` (vectorised ``mean_resultant_length``)."""
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return np.hypot(np.cos(th).mean(axis=axis), np.sin(th).mean(axis=axis))


def kuiper_statistic(theta_deg, axis: int = -1) -> np.ndarray:
    """Kuiper's V along ``axis`` (vectorised; see ``kuiper_test`` for the p-value)."""
    u = np.sort(np.moveaxis(np.mod(np.asarray(theta_deg, dtype=float), 360.0) / 360.0, axis, -1), axis=-1)
    n = u.shape[-1]
    i = np.arange(1, n + 1)
    return np.max(i / n - u, axis=-1) + np.max(u - (i - 1) / n, axis=-1)


# --------------------------------------------------------------------------
# Tests of uniformity
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RayleighResult:
    n: int
    R_bar: float
    Z: float
    p_value: float
    mean_direction: float


def rayleigh_test(theta_deg) -> RayleighResult:
    """Rayleigh test against a uniform circular null.

    Most powerful against a single unimodal cluster. It has almost no power
    against two clusters 180 deg apart (axial data); use ``kuiper_test`` or a
    mixture model for that.
    """
    th = _rad(theta_deg)
    n = th.size
    R = np.hypot(np.sum(np.cos(th)), np.sum(np.sin(th)))
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))
    return RayleighResult(
        n=n,
        R_bar=float(R / n),
        Z=float(R**2 / n),
        p_value=float(np.clip(p, 0.0, 1.0)),
        mean_direction=mean_direction(theta_deg),
    )


@dataclass(frozen=True)
class KuiperResult:
    n: int
    V: float
    p_value: float


def kuiper_test(theta_deg) -> KuiperResult:
    """Kuiper's test of uniformity: a KS test that does not depend on the origin.

    Sensitive to any departure from uniformity, including multimodal ones.
    The p-value uses Stephens' (1970) small-sample modification of the
    asymptotic distribution, good to about 1% for n >= 8.
    """
    u = np.sort(np.mod(_rad(theta_deg), 2 * np.pi) / (2 * np.pi))
    n = u.size
    i = np.arange(1, n + 1)
    V = float(np.max(i / n - u) + np.max(u - (i - 1) / n))
    lam = (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * V
    if lam < 0.4:
        p = 1.0
    else:
        j = np.arange(1, 101)
        p = 2 * np.sum((4 * j**2 * lam**2 - 1) * np.exp(-2 * j**2 * lam**2))
    return KuiperResult(n=n, V=V, p_value=float(np.clip(p, 0.0, 1.0)))


# --------------------------------------------------------------------------
# Association between two angles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CircCorrResult:
    r: float
    p_value: float
    n_perm: int


def circular_correlation(alpha_deg, beta_deg, n_perm: int = 9999, rng=None) -> CircCorrResult:
    """Jammalamadaka-SenGupta circular correlation with a permutation p-value.

    Use this, for example, to ask whether longitude of perihelion tracks the
    longitude of node, a common fingerprint of survey pointing. The p-value
    is two-sided and exact up to Monte Carlo error.
    """
    a, b = _rad(alpha_deg), _rad(beta_deg)
    if a.size != b.size:
        raise ValueError("alpha and beta must have the same length")
    rng = np.random.default_rng(rng)

    def corr(sa, sb):
        return np.sum(sa * sb, axis=-1) / np.sqrt(np.sum(sa**2, axis=-1) * np.sum(sb**2, axis=-1))

    sa = np.sin(a - np.deg2rad(mean_direction(alpha_deg)))
    sb = np.sin(b - np.deg2rad(mean_direction(beta_deg)))
    r = float(corr(sa, sb))
    perms = rng.permuted(np.broadcast_to(sb, (n_perm, sb.size)), axis=1)
    null = corr(sa, perms)
    p = (1 + np.sum(np.abs(null) >= abs(r) - 1e-12)) / (1 + n_perm)
    return CircCorrResult(r=r, p_value=float(p), n_perm=n_perm)


# --------------------------------------------------------------------------
# Uncertainty on the mean direction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MeanDirectionCI:
    mean: float
    lower: float
    upper: float
    level: float


def bootstrap_mean_direction(theta_deg, n_boot: int = 10000, level: float = 0.95, rng=None) -> MeanDirectionCI:
    """Percentile bootstrap interval for the circular mean.

    ``lower`` and ``upper`` are in [0, 360) and the interval runs
    counter-clockwise from lower to upper, so lower > upper means the interval
    crosses 0 deg. With a weak signal the interval can be very wide; that is
    the honest answer, not a bug.
    """
    th = np.asarray(theta_deg, dtype=float).ravel()
    rng = np.random.default_rng(rng)
    mu = mean_direction(th)
    idx = rng.integers(0, th.size, size=(n_boot, th.size))
    r = np.deg2rad(th[idx])
    boot = np.rad2deg(np.arctan2(np.sin(r).mean(1), np.cos(r).mean(1)))
    dev = _wrap_signed(boot - mu)
    lo, hi = np.quantile(dev, [(1 - level) / 2, (1 + level) / 2])
    return MeanDirectionCI(mean=mu, lower=float(wrap_deg(mu + lo)), upper=float(wrap_deg(mu + hi)), level=level)


# --------------------------------------------------------------------------
# Spherical: orbit poles and perihelion directions
# --------------------------------------------------------------------------

def orbit_pole(i_deg, Omega_deg) -> np.ndarray:
    """Unit angular-momentum vectors, shape (n, 3), in the ecliptic frame."""
    i, O = _rad(i_deg), _rad(Omega_deg)
    return np.column_stack([np.sin(i) * np.sin(O), -np.sin(i) * np.cos(O), np.cos(i)])


def perihelion_direction(i_deg, Omega_deg, omega_deg) -> np.ndarray:
    """Unit vectors pointing from the Sun to perihelion, shape (n, 3)."""
    i, O, w = _rad(i_deg), _rad(Omega_deg), _rad(omega_deg)
    return np.column_stack([
        np.cos(O) * np.cos(w) - np.sin(O) * np.sin(w) * np.cos(i),
        np.sin(O) * np.cos(w) + np.cos(O) * np.sin(w) * np.cos(i),
        np.sin(w) * np.sin(i),
    ])


def pole_to_elements(v) -> tuple[float, float]:
    """Inverse of ``orbit_pole`` for one vector: (i, Omega) in degrees."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    i = np.rad2deg(np.arccos(np.clip(v[2], -1, 1)))
    Omega = wrap_deg(np.rad2deg(np.arctan2(v[0], -v[1])))
    return float(i), float(Omega)


@dataclass(frozen=True)
class SphericalRayleighResult:
    n: int
    R_bar: float
    mean_vector: tuple[float, float, float]
    p_value: float


def spherical_rayleigh(vectors) -> SphericalRayleighResult:
    """Rayleigh test for unit vectors on the sphere: 3 n R-bar^2 ~ chi2(3).

    Appropriate for perihelion directions. Not appropriate for orbit poles:
    poles of any flattish population pile up near the ecliptic pole, so a
    uniform-sphere null is rejected trivially. Test poles against a simulated
    null instead (see ``planetix.stats.nulls``).
    """
    v = np.asarray(vectors, dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    n = len(v)
    m = v.mean(axis=0)
    R_bar = float(np.linalg.norm(m))
    p = float(stats.chi2.sf(3 * n * R_bar**2, df=3))
    return SphericalRayleighResult(n=n, R_bar=R_bar, mean_vector=tuple(m / max(R_bar, 1e-12)), p_value=p)
