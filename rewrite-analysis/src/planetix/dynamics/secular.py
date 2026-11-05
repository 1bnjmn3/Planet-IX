"""Secular (orbit-averaged) forcing of distant test particles, in real units.

This replaces the "torque balance" of original scripts 23, 25 and 33-36, whose
constants (1e8 for the giants, 1e10 for the planet) were chosen by hand. Here
every rate comes from Laplace-Lagrange theory (Murray & Dermott 1999, sec. 7.4
and 7.9) with real masses: for a test particle with semi-major axis ``a`` and a
perturber of mass ``m`` (in Msun) at ``a_j``,

    B_j = (n / 4) m alpha alphabar b^(1)_{3/2}(alpha),

with alpha = min(a, a_j) / max(a, a_j), alphabar = 1 for an inner perturber and
alpha for an outer one. B_j (rad/yr) is the rate at which the particle's orbit
pole would precess about that perturber's pole if it were the only one; the
free nodal precession rate is -sum_j B_j.

The *forced* (Laplace) pole is where the summed torques vanish. To first order
in inclination it is the B-weighted mean of the perturbers' inclination vectors,
which is the structure the original scripts were trying to reproduce:

    forced = sum_j B_j I_j / sum_j B_j.

``method="nonlinear"`` solves the exact orbit-averaged quadrupole condition
sum_j B_j (n_j . l)(n_j x l) = 0 instead, which stays valid for a steeply
inclined planet (Tremaine, Touma & Namouni 2009).

Assumptions, stated so they can be checked (``planetix.dynamics.nbody`` does):
circular, non-crossing orbits in the Laplace coefficients; perturbers fixed in
space (the giants' plane is the invariable plane); no mean-motion resonances.
ETNOs have e ~ 0.6-0.9, so ``ecc="quadrupole"`` multiplies each rate by the
leading-order eccentricity factors of the omega-averaged quadrupole: (1-e^2)^-2
for an inner perturber, (1 + 3e^2/2)(1-e^2)^-1/2 (1-e_j^2)^-3/2 for an outer one.

All angles are degrees in the invariable-plane frame unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

from planetix.dynamics.constants import GIANTS, M_EARTH, mean_motion
from planetix.dynamics.frames import angle_between, pole, pole_to_angles


@dataclass(frozen=True)
class Perturber:
    """A massive body on a fixed orbit. ``mass`` in Msun, ``a`` in AU, angles in degrees."""

    mass: float
    a: float
    i: float = 0.0
    Omega: float = 0.0
    e: float = 0.0
    name: str = ""

    @classmethod
    def earths(cls, mass_earth, a, i=0.0, Omega=0.0, e=0.0, name=""):
        return cls(mass_earth * M_EARTH, a, i, Omega, e, name)

    @property
    def pole(self):
        return pole(self.i, self.Omega)


def giant_planets():
    """Jupiter to Neptune on circular orbits in the invariable plane."""
    return [Perturber(m, a, name=n) for n, (m, a) in GIANTS.items()]


@lru_cache(maxsize=4096)
def _laplace(s, j, alpha):
    f = lambda psi: np.cos(j * psi) / (1.0 - 2.0 * alpha * np.cos(psi) + alpha**2) ** s
    val, _ = quad(f, 0.0, np.pi, limit=400, epsabs=0, epsrel=1e-11)
    return 2.0 * val / np.pi


def laplace_coefficient(s, j, alpha):
    """Laplace coefficient b_s^(j)(alpha) = (1/pi) int_0^2pi cos(j psi) / (1 - 2 alpha cos psi + alpha^2)^s dpsi."""
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    return _laplace(float(s), int(j), round(alpha, 12))


def coupling_rate(a, perturber, e=0.0, ecc=None):
    """B_j in rad/yr: precession rate of a test particle at ``a`` about ``perturber``'s pole."""
    aj = perturber.a
    if np.isclose(a, aj):
        raise ValueError("test particle and perturber share a semi-major axis")
    inner = aj < a
    alpha = aj / a if inner else a / aj
    alphabar = 1.0 if inner else alpha
    B = 0.25 * mean_motion(a) * perturber.mass * alpha * alphabar * laplace_coefficient(1.5, 1, alpha)
    if ecc == "quadrupole":
        if inner:
            B *= (1.0 - e**2) ** -2
        else:
            B *= (1.0 + 1.5 * e**2) / np.sqrt(1.0 - e**2) * (1.0 - perturber.e**2) ** -1.5
    elif ecc is not None:
        raise ValueError(f"unknown ecc option {ecc!r}")
    return float(B)


def nodal_precession_rate(a, perturbers, e=0.0, ecc=None):
    """Free nodal precession rate dOmega/dt in rad/yr (negative: regression)."""
    return -sum(coupling_rate(a, p, e, ecc) for p in perturbers)


def _forced_one(a, e, perturbers, method, ecc):
    w = np.array([coupling_rate(a, p, e, ecc) for p in perturbers])
    n = np.array([p.pole for p in perturbers])
    xy = (w[:, None] * n[:, :2]).sum(0) / w.sum()
    r = np.hypot(*xy)
    if r >= 1.0:
        xy = xy / r * 0.999999
    if method == "linear":
        return np.array([xy[0], xy[1], np.sqrt(1.0 - xy @ xy)])
    if method != "nonlinear":
        raise ValueError(f"unknown method {method!r}")

    def to_l(u):
        z2 = max(1.0 - u @ u, 0.0)
        return np.array([u[0], u[1], np.sqrt(z2)])

    def torque(u):
        l = to_l(u)
        return ((w * (n @ l))[:, None] * np.cross(n, l)).sum(0) / w.sum()

    sol = least_squares(torque, xy, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return to_l(sol.x)


def forced_pole(a, perturbers, e=0.0, method="linear", ecc=None):
    """Forced (Laplace-plane) pole at semi-major axis ``a`` (scalar or array) as unit vector(s).

    ``e`` is the test particle eccentricity (only used when ``ecc="quadrupole"``).
    """
    a_arr, e_arr = np.broadcast_arrays(np.asarray(a, float), np.asarray(e, float))
    out = np.array([_forced_one(ai, ei, perturbers, method, ecc) for ai, ei in zip(a_arr.ravel(), e_arr.ravel())])
    return out.reshape(a_arr.shape + (3,))


def forced_inclination(a, perturbers, e=0.0, method="linear", ecc=None):
    """Forced (i, Omega) in degrees at ``a``, relative to the invariable plane."""
    return pole_to_angles(forced_pole(a, perturbers, e, method, ecc))


def two_plane_tilt(w_ref, w_planet, i_planet):
    """Exact Laplace-surface tilt (deg) from the reference pole when only two poles act.

    With weight ``w_ref`` about the reference pole (the giants) and ``w_planet``
    about a planet pole inclined by ``i_planet`` (deg), the zero-torque condition
    w_ref sin 2phi = w_planet sin 2(i_planet - phi) has the closed form
    tan 2phi = w_planet sin 2i / (w_ref + w_planet cos 2i). Vectorised; this is
    what ``forced_pole(method="nonlinear")`` solves for the giants plus one planet.
    """
    i2 = 2.0 * np.radians(i_planet)
    w_ref, w_planet = np.asarray(w_ref, float), np.asarray(w_planet, float)
    return 0.5 * np.degrees(np.arctan2(w_planet * np.sin(i2), w_ref + w_planet * np.cos(i2)))


def planet_nine(mass_earth=5.0, a=500.0, i=20.0, Omega=100.0, e=0.0):
    """Giants plus one distant planet: the configuration scripts 23-36 meant to model."""
    return giant_planets() + [Perturber.earths(mass_earth, a, i, Omega, e, name="Planet")]


def required_mass(target_i, a_particle, a_planet, i_planet, e=0.0, method="nonlinear", ecc=None,
                  bracket=(1e-4, 1e4)):
    """Planet mass (Earth masses) whose forced inclination at ``a_particle`` equals ``target_i``.

    Replaces the curves of scripts 33, 35 and 36. Returns ``nan`` when no mass in
    ``bracket`` reaches the target (the forced tilt can never exceed ``i_planet``).
    """
    from scipy.optimize import brentq

    def gap(logm):
        pert = giant_planets() + [Perturber.earths(10**logm, a_planet, i_planet, 0.0)]
        return forced_inclination(a_particle, pert, e, method, ecc)[0] - target_i

    lo, hi = np.log10(bracket[0]), np.log10(bracket[1])
    if not target_i < i_planet or gap(lo) > 0 or gap(hi) < 0:
        return np.nan
    return 10 ** brentq(gap, lo, hi, xtol=1e-6)


__all__ = [
    "Perturber", "giant_planets", "planet_nine", "laplace_coefficient", "coupling_rate",
    "nodal_precession_rate", "two_plane_tilt", "forced_pole", "forced_inclination", "required_mass", "angle_between",
]
