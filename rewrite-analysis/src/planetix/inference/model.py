"""Bias-aware likelihood for "distant planet" (H1) against "no planet" (H0).

Observable per object j: its orientation x_j = (varpi_j, Omega_j). Everything
else (a, e, i, H) is conditioned on, so both hypotheses share it. With S_j the
probability that a survey would have detected object j at orientation x
(``selection.selection_grid``) and f the intrinsic orientation density, the
probability of the observed orientation is

    p(x_j | H) = f(x_j) S_j(x_j) / integral f S_j dx.

H0 has f uniform. H1 writes f = g / (4 pi^2) with g relative to uniform:

    g(x) = (1 - F_j) + F_j * h(x),
    F_j  = f_max * sigmoid((ln a_j - ln a_crit) / w),
    h    = 2 pi vM(varpi; varpi_9 + 180, kappa_varpi)
           * 2 pi vM(Omega; Omega_9, kappa_pole sin i_j sin i_9).

h is the population a planet confines: perihelia anti-aligned with the
planet's, and orbital poles drawn from a Fisher distribution around the
planet's pole (kappa_pole), which at fixed i_j is a von Mises in Omega. Objects
with a well above a_crit are confined; a_crit follows from equating the
quadrupole precession driven by the known giant planets with the secular
forcing of the distant planet, normalised so that the reference planet
(5 M_E, 500 AU, e = 0.25) gives a_crit = 250 AU (Batygin et al. 2019,
Phys. Rep. 805, 1):

    a_crit = 250 AU * [(m_9 / 5)^-1 (a_9 / 500)^3 ((1 - e_9^2) / 0.9375)^1.5]^(1/5).

S_j(x_j) cancels in the ratio of the two hypotheses, so the per-object Bayes
factor at fixed parameters is g(x_j) / sum_x g(x) S_j(x).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import i0e

PARAMS = ["log10_m9", "log10_a9", "e9", "i9", "varpi9", "Omega9", "f_max", "log10_kvarpi", "log10_kpole"]
LABELS = {
    "log10_m9": r"$\log_{10} m_9/M_\oplus$",
    "log10_a9": r"$\log_{10} a_9/\mathrm{AU}$",
    "e9": r"$e_9$",
    "i9": r"$i_9$ [deg]",
    "varpi9": r"$\varpi_9$ [deg]",
    "Omega9": r"$\Omega_9$ [deg]",
    "f_max": r"$f_\mathrm{max}$",
    "log10_kvarpi": r"$\log_{10}\kappa_\varpi$",
    "log10_kpole": r"$\log_{10}\kappa_\mathrm{pole}$",
}


def a_crit(m9, a9, e9):
    """Semi-major axis (AU) beyond which a planet (m9 in M_E, a9 in AU) confines ETNOs."""
    x = (np.asarray(m9) / 5.0) ** -1 * (np.asarray(a9) / 500.0) ** 3 * ((1 - np.asarray(e9) ** 2) / 0.9375) ** 1.5
    return 250.0 * x**0.2


def _log_vm_rel(dtheta_rad, kappa):
    """log of 2 pi * von Mises density (i.e. relative to uniform), numerically stable."""
    kappa = np.asarray(kappa, float)
    return kappa * (np.cos(dtheta_rad) - 1.0) - np.log(i0e(kappa))


@dataclass
class Prior:
    """Box priors (uniform in each listed variable) on the H1 parameters.

    i9 is uniform in cos(i9) between the bounds (isotropic poles). The ranges
    are the baseline of the paper; ``planetix.inference.evidence`` varies them.
    """

    log10_m9: tuple = (np.log10(1.0), np.log10(20.0))
    log10_a9: tuple = (np.log10(250.0), np.log10(1000.0))
    e9: tuple = (0.0, 0.5)
    i9: tuple = (0.0, 50.0)
    varpi9: tuple = (0.0, 360.0)
    Omega9: tuple = (0.0, 360.0)
    f_max: tuple = (0.0, 1.0)
    log10_kvarpi: tuple = (0.0, np.log10(20.0))
    log10_kpole: tuple = (np.log10(3.0), 2.0)

    def transform(self, u):
        """Unit cube -> parameters (dynesty prior transform)."""
        u = np.asarray(u, float)
        out = np.empty_like(u)
        for k, name in enumerate(PARAMS):
            lo, hi = getattr(self, name)
            if name == "i9":
                c_lo, c_hi = np.cos(np.radians(lo)), np.cos(np.radians(hi))
                out[k] = np.degrees(np.arccos(c_lo + u[k] * (c_hi - c_lo)))
            else:
                out[k] = lo + u[k] * (hi - lo)
        return out


@dataclass
class Likelihood:
    """Log Bayes-factor density log[p(data | theta, H1) / p(data | H0)].

    Parameters
    ----------
    varpi, Omega, i, a
        Observed values for each object (degrees, AU).
    S
        Selection grids, shape (n_obj, n_varpi, n_Omega), each summing to 1.
    w
        Width of the confinement transition in ln a.
    nondetection
        Optional ``planetix.inference.nondetection.NonDetection``: multiplies
        the planet likelihood by the probability that direct searches missed it.
    """

    varpi: np.ndarray
    Omega: np.ndarray
    i: np.ndarray
    a: np.ndarray
    S: np.ndarray
    w: float = 0.2
    nondetection: object | None = None
    _grid: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self.varpi, self.Omega, self.i, self.a = (np.asarray(x, float) for x in (self.varpi, self.Omega, self.i, self.a))
        self.S = np.asarray(self.S, float)
        _, nv, no = self.S.shape
        self._grid = (
            np.radians((np.arange(nv) + 0.5) * 360.0 / nv),
            np.radians((np.arange(no) + 0.5) * 360.0 / no),
        )
        self._sini = np.sin(np.radians(self.i))

    @classmethod
    def from_frame(cls, df, S, **kw):
        return cls(df["varpi"].to_numpy(), df["Omega"].to_numpy(), df["i"].to_numpy(), df["a"].to_numpy(), S, **kw)

    def confined_fraction(self, theta):
        p = dict(zip(PARAMS, theta))
        ac = a_crit(10 ** p["log10_m9"], 10 ** p["log10_a9"], p["e9"])
        return p["f_max"] / (1.0 + np.exp(-(np.log(self.a) - np.log(ac)) / self.w))

    def per_object(self, theta):
        """log Bayes factor of each object at fixed theta."""
        p = dict(zip(PARAMS, theta))
        F = self.confined_fraction(theta)
        kv = 10 ** p["log10_kvarpi"]
        ko = 10 ** p["log10_kpole"] * self._sini * np.sin(np.radians(p["i9"]))
        cv = np.radians(p["varpi9"] + 180.0)
        co = np.radians(p["Omega9"])
        gv, go = self._grid

        # h at the data
        log_h = _log_vm_rel(np.radians(self.varpi) - cv, kv) + _log_vm_rel(np.radians(self.Omega) - co, ko)
        num = (1.0 - F) + F * np.exp(log_h)

        # sum_x h(x) S_j(x): h is a product of a varpi factor and an Omega factor
        hv = np.exp(_log_vm_rel(gv - cv, kv))  # (nv,)
        ho = np.exp(_log_vm_rel(go[None, :] - co, ko[:, None]))  # (n, no)
        hS = np.einsum("v,jvo,jo->j", hv, self.S, ho)
        den = (1.0 - F) + F * hS
        return np.log(num) - np.log(den)

    def __call__(self, theta):
        total = float(np.sum(self.per_object(theta)))
        if self.nondetection is not None:
            total += self.nondetection.log_p_missed(theta)
        return total
