"""Vectorised two-body geometry: orbital elements to sky positions, distances and magnitudes.

Everything here is plain Keplerian motion about the Sun, which is accurate to well under a
degree on the sky for objects beyond 30 AU over the ~20-year span of the surveys used.
Elements are heliocentric J2000 ecliptic (the MPC convention). Angles in degrees,
distances in AU, times in Julian days (TT).
"""

from __future__ import annotations

import numpy as np

from planetix.angles import wrap_deg

K_GAUSS = 0.01720209895  # Gaussian gravitational constant, rad/day
OBLIQUITY_J2000 = 23.4392911  # deg
JD_J2000 = 2451545.0


def mean_motion(a):
    """Mean motion in deg/day for semimajor axis ``a`` (AU), massless body."""
    return np.degrees(K_GAUSS / np.asarray(a, float) ** 1.5)


def solve_kepler(M, e, n_iter=50, tol=1e-12):
    """Eccentric anomaly E (rad) from mean anomaly M (rad), elliptic orbits.

    Newton iteration from E0 = pi for e > 0.8 (robust for the e ~ 0.9 orbits of ETNOs).
    """
    M = np.mod(np.asarray(M, float), 2 * np.pi)
    e = np.broadcast_to(np.asarray(e, float), M.shape)
    E = np.where(e > 0.8, np.pi, M + e * np.sin(M))
    for _ in range(n_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def heliocentric_xyz(a, e, i, Omega, omega, M):
    """Heliocentric ecliptic J2000 position (..., 3) in AU. Inputs broadcast together."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    i, Om, w = (np.radians(np.asarray(x, float)) for x in (i, Omega, omega))
    E = solve_kepler(np.radians(M), e)
    # Position in the orbital plane, perihelion along +x.
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1.0 - e**2) * np.sin(E)
    cO, sO, cw, sw, ci, si = np.cos(Om), np.sin(Om), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    x = (cO * cw - sO * sw * ci) * xp + (-cO * sw - sO * cw * ci) * yp
    y = (sO * cw + cO * sw * ci) * xp + (-sO * sw + cO * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def earth_xyz(jd):
    """Heliocentric ecliptic position of the Earth-Moon barycentre (..., 3), AU.

    Keplerian mean elements of Standish (1992, JPL "approximate positions of the planets",
    valid 1800-2050). Errors are ~0.0001 AU, i.e. < 1 arcsec at 40 AU.
    """
    T = (np.asarray(jd, float) - JD_J2000) / 36525.0
    a = 1.00000261 + 0.00000562 * T
    e = 0.01671123 - 0.00004392 * T
    i = -0.00001531 - 0.01294668 * T
    L = 100.46457166 + 35999.37244981 * T
    varpi = 102.93768193 + 0.32327364 * T
    return heliocentric_xyz(a, e, i, 0.0, varpi, L - varpi)


def xyz_to_lonlat(v):
    """Longitude in [0, 360) and latitude of vectors (..., 3), degrees."""
    v = np.asarray(v, float)
    r = np.linalg.norm(v, axis=-1)
    lon = wrap_deg(np.degrees(np.arctan2(v[..., 1], v[..., 0])))
    lat = np.degrees(np.arcsin(np.clip(v[..., 2] / r, -1.0, 1.0)))
    return lon, lat


def ecliptic_to_equatorial_xyz(v):
    """Rotate ecliptic J2000 vectors into the equatorial (ICRS-aligned) frame."""
    eps = np.radians(OBLIQUITY_J2000)
    v = np.asarray(v, float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([x, np.cos(eps) * y - np.sin(eps) * z, np.sin(eps) * y + np.cos(eps) * z], axis=-1)


def apparent_mag(H, r, delta):
    """Apparent magnitude H + 5 log10(r * delta), ignoring the phase term.

    Beyond 30 AU the phase angle is under 2 deg; a typical TNO phase coefficient of
    ~0.15 mag/deg makes that a <= 0.3 mag effect, smaller than the colour uncertainties.
    """
    return np.asarray(H, float) + 5.0 * np.log10(np.asarray(r, float) * np.asarray(delta, float))


def geocentric(a, e, i, Omega, omega, M, jd, frame="ecliptic"):
    """Geocentric direction and distances at time ``jd`` for elements whose mean anomaly is
    ``M`` at that same time.

    Returns (lon, lat, r, delta): sky longitude/latitude in ``frame`` ("ecliptic" or
    "equatorial", i.e. RA/Dec), heliocentric distance r, geocentric distance delta.
    """
    helio = heliocentric_xyz(a, e, i, Omega, omega, M)
    geo = helio - earth_xyz(jd)
    if frame == "equatorial":
        geo = ecliptic_to_equatorial_xyz(geo)
    elif frame != "ecliptic":
        raise ValueError(f"Unknown frame {frame!r}")
    lon, lat = xyz_to_lonlat(geo)
    return lon, lat, np.linalg.norm(helio, axis=-1), np.linalg.norm(geo, axis=-1)


def perihelion_unit_vector(i, Omega, omega):
    """Unit vector (..., 3) pointing to perihelion, ecliptic frame."""
    i, Om, w = (np.radians(np.asarray(x, float)) for x in (i, Omega, omega))
    i, Om, w = np.broadcast_arrays(i, Om, w)
    return np.stack(
        [
            np.cos(w) * np.cos(Om) - np.sin(w) * np.sin(Om) * np.cos(i),
            np.cos(w) * np.sin(Om) + np.sin(w) * np.cos(Om) * np.cos(i),
            np.sin(w) * np.sin(i),
        ],
        axis=-1,
    )


def pole_unit_vector(i, Omega):
    """Unit orbit normal (..., 3), ecliptic frame."""
    i, Om = (np.radians(np.asarray(x, float)) for x in (i, Omega))
    i, Om = np.broadcast_arrays(i, Om)
    return np.stack([np.sin(i) * np.sin(Om), -np.sin(i) * np.cos(Om), np.cos(i)], axis=-1)
