"""Two-body heliocentric positions and apparent magnitudes, vectorised.

Accuracy needed here is coarse (a degree on the sky, a tenth of a magnitude),
so planetary perturbations, light time and the observer's offset from the Sun
(at most 1 AU against heliocentric distances above 25 AU) are ignored.
"""

from __future__ import annotations

import numpy as np

# Gaussian gravitational constant squared: GM_sun in AU^3 / day^2.
GM_SUN = 0.01720209895**2


def solve_kepler(M, e, tol: float = 1e-10, max_iter: int = 50):
    """Eccentric anomaly E (radians) from mean anomaly M (radians), e < 1."""
    M = np.asarray(M, float)
    e = np.asarray(e, float)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def heliocentric_xyz(a, e, i, Omega, omega, M):
    """Heliocentric ecliptic Cartesian position (AU). Angles in degrees."""
    i, Omega, omega, M = (np.radians(np.asarray(x, float)) for x in (i, Omega, omega, M))
    a = np.asarray(a, float)
    e = np.asarray(e, float)
    E = solve_kepler(np.mod(M, 2 * np.pi), e)
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1.0 - e**2) * np.sin(E)
    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)
    x = (cO * cw - sO * sw * ci) * xp + (-cO * sw - sO * cw * ci) * yp
    y = (sO * cw + cO * sw * ci) * xp + (-sO * sw + cO * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return x, y, z


def sky_position(a, e, i, Omega, omega, M):
    """Ecliptic longitude, latitude (degrees) and heliocentric distance r (AU)."""
    x, y, z = heliocentric_xyz(a, e, i, Omega, omega, M)
    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.degrees(np.arctan2(y, x)) % 360.0
    lat = np.degrees(np.arcsin(z / r))
    return lon, lat, r


def apparent_mag(H, r):
    """Opposition V magnitude for a body at heliocentric distance r (AU), Delta ~ r - 1."""
    return np.asarray(H, float) + 5.0 * np.log10(r * np.maximum(r - 1.0, 1e-3))


def mean_anomaly_at(M0, a, epoch_jd, jd):
    """Propagate mean anomaly (degrees) from epoch_jd to jd on a Keplerian orbit."""
    n = np.degrees(np.sqrt(GM_SUN / np.asarray(a, float) ** 3))  # deg / day
    return np.mod(np.asarray(M0, float) + n * (np.asarray(jd, float) - np.asarray(epoch_jd, float)), 360.0)
