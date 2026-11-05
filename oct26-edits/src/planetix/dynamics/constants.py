"""Physical constants and the giant planets, in AU, years and solar masses.

Sources: IAU 2015 nominal values and the JPL DE440 mass ratios; mean semi-major
axes from the JPL "Keplerian elements for approximate positions" table; the
invariable plane relative to the J2000 ecliptic from Souami & Souchay (2012,
A&A 543, A133).
"""

import numpy as np

# Gaussian gravitational constant: n = K / a^1.5 in rad/day for a in AU and M = 1 Msun.
K_GAUSS = 0.01720209895
DAYS_PER_YEAR = 365.25
# G * Msun in AU^3 / yr^2.
GM_SUN = (K_GAUSS * DAYS_PER_YEAR) ** 2

M_EARTH = 3.003489614e-6  # Earth mass in solar masses (DE440)

# name: (mass / Msun, mean semi-major axis / AU)
GIANTS = {
    "Jupiter": (1.0 / 1047.348644, 5.20288700),
    "Saturn": (1.0 / 3497.901768, 9.53667594),
    "Uranus": (1.0 / 22902.98, 19.18916464),
    "Neptune": (1.0 / 19412.26, 30.06992276),
}

# Orientation of the invariable plane in the J2000 ecliptic frame, degrees.
INVARIABLE_I = 1.57869
INVARIABLE_OMEGA = 107.58232


def mean_motion(a):
    """Mean motion in rad/yr of a massless body with semi-major axis ``a`` (AU)."""
    return np.sqrt(GM_SUN) / np.asarray(a, float) ** 1.5


def giants_j2r2():
    """Orbit-averaged J2 R^2 (AU^2) that the four giants exert on a distant body.

    Replacing the giants by a solar quadrupole of this size is exact to leading
    order in (a_giant / a)^2 (e.g. Batygin & Brown 2016): J2 R^2 = (1/2) sum m_j a_j^2.
    """
    return 0.5 * sum(m * a**2 for m, a in GIANTS.values())
