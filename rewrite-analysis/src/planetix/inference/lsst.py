"""Rubin Observatory LSST selection model and information-rate forecasts.

``LSSTFootprint`` has the ``weight(lon, lat, V)`` interface of
``selection.Footprint``, so ``selection.selection_grid(obj, LSSTFootprint())``
gives the S_j an object would have if it were found by LSST alone. The model is
deliberately simple and every number is a parameter:

* Wide-Fast-Deep: declination -70 to +12.5 deg, galactic latitude |b| > 15 deg
  (the plane is observed at lower cadence and with heavy confusion; treating it
  as lost is conservative).
* Northern Ecliptic Spur: ecliptic latitude |beta| < 10 deg up to dec +40 deg.
* Detection (three linked tracklets within ~15 nights) with efficiency
  1 / (1 + exp((V - V_lim) / width)), V_lim = 24.5 by default: r ~ 24.2 linking
  depth (Ivezic et al. 2019 give a 30 s single-visit r depth of ~24.2-24.7)
  plus V - r ~ 0.5 for red TNOs, less a linking loss.

An ETNO moves a few degrees over the ten-year survey, so detection is a property
of where the orbit puts the object today: S_j(varpi, Omega) is the fraction of
orbital phase (uniform in time) for which it is in the footprint and brighter
than the limit.

``information_rate`` gives the expected per-object log Bayes factor
D_j(theta) = E_{x ~ p(x | theta, H1)} ln[p(x | theta, H1) / p(x | H0)], the
Kullback-Leibler divergence the asymptotic evidence grows with:
E[ln K(N)] ~ N <D> - c ln N - c0 (the Occam terms c, c0 are calibrated with
simulated catalogues in ``scripts/forecast_and_history.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetix.inference.model import Likelihood
from planetix.inference.nondetection import _EQ_TO_GAL, _OBLIQUITY


def ecliptic_to_eq_gal(lon, lat):
    """Declination and galactic latitude (deg) of ecliptic (lon, lat) in degrees."""
    lon, lat = np.radians(lon), np.radians(lat)
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    ce, se = np.cos(_OBLIQUITY), np.sin(_OBLIQUITY)
    eq = np.stack([x, ce * y - se * z, se * y + ce * z])
    dec = np.degrees(np.arcsin(np.clip(eq[2], -1, 1)))
    b = np.degrees(np.arcsin(np.clip(np.tensordot(_EQ_TO_GAL[2], eq, axes=1), -1, 1)))
    return dec, b


@dataclass
class LSSTFootprint:
    v_lim: float = 24.5
    v_width: float = 0.3
    dec_min: float = -70.0
    dec_max_wfd: float = 12.5
    b_min: float = 15.0
    nes_beta: float = 10.0
    nes_dec_max: float = 40.0

    def covered(self, lon, lat):
        dec, b = ecliptic_to_eq_gal(lon, lat)
        wfd = (dec > self.dec_min) & (dec < self.dec_max_wfd) & (np.abs(b) > self.b_min)
        nes = (np.abs(lat) < self.nes_beta) & (dec >= self.dec_max_wfd) & (dec < self.nes_dec_max) & (np.abs(b) > self.b_min)
        return wfd | nes

    def efficiency(self, V):
        return 1.0 / (1.0 + np.exp((np.asarray(V) - self.v_lim) / self.v_width))

    def weight(self, lon, lat, V):
        return self.covered(lon, lat) * self.efficiency(V)


def information_rate(like: Likelihood, theta) -> np.ndarray:
    """Expected log Bayes factor per object, D_j(theta), exactly on the selection grid."""
    from planetix.inference.evidence import planet_density_grid

    g = planet_density_grid(like, theta)                       # (n, nv, no), relative to uniform
    gS = g * like.S
    Z = gS.sum(axis=(1, 2))                                   # sum_x g S
    p1 = gS / Z[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(g) - np.log(Z)[:, None, None]
    return np.nansum(p1 * lr, axis=(1, 2))


def n_required(D_mean: float, ln_k_target: float, c: float, c0: float, n_max: int = 100_000) -> float:
    """Smallest N with N D - c ln N - c0 >= ln K target (inf if never).

    The form is asymptotic, so the search starts where it is increasing (N >= c / D)
    and not below 10 objects."""
    n = np.arange(max(10, int(np.ceil(c / max(D_mean, 1e-12)))), n_max + 1)
    ok = np.flatnonzero(n * D_mean - c * np.log(n) - c0 >= ln_k_target)
    return float(n[ok[0]]) if len(ok) else np.inf
