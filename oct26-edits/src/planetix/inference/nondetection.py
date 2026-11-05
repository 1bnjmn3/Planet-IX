"""Probability that wide-field surveys would have missed the planet itself.

A planet bright enough and in the sky they covered would already have been
found by the Pan-STARRS1 3pi and ZTF moving-object searches (Brown, Holman &
Batygin 2024, AJ 167, 146; Brown & Batygin 2022, AJ 163, 102). We model their
combined coverage coarsely: declination above -30 deg, galactic latitude
|b| > 15 deg, and a detection efficiency that falls from 1 to 0 around
V = 21.5 with a 0.3 mag logistic width. The planet's mean anomaly is unknown,
so the probability of a miss is averaged over it (uniform in time) and over
albedo (uniform 0.2-0.75). Radius follows a Neptune-normalised power law,
R = 3.86 R_E (m / 17.1 M_E)^0.59 (Chen & Kipping 2017, ApJ 834, 17).

This is a rough stand-in for the published survey simulations; the paper
reports results with and without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetix.inference.kepler import heliocentric_xyz

_OBLIQUITY = np.radians(23.4392911)
# ICRS -> galactic rotation (rows: galactic x, y, z axes in equatorial coordinates)
_EQ_TO_GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [0.4941094279, -0.4448296300, 0.7469822445],
    [-0.8676661490, -0.1980763734, 0.4559837762],
])
R_EARTH_KM = 6371.0


def planet_H(m9, albedo):
    """Absolute magnitude of a planet of mass m9 (M_E) and geometric albedo."""
    radius_km = 3.86 * R_EARTH_KM * (np.asarray(m9) / 17.1) ** 0.59
    return 5.0 * np.log10(1329.0 / (2.0 * radius_km * np.sqrt(albedo)))


@dataclass
class NonDetection:
    v_lim: float = 21.5
    v_width: float = 0.3
    dec_min: float = -30.0
    b_min: float = 15.0
    n_phase: int = 360
    albedos: tuple = (0.2, 0.34, 0.48, 0.61, 0.75)

    def __post_init__(self):
        self._M = (np.arange(self.n_phase) + 0.5) * 360.0 / self.n_phase

    def p_missed(self, m9, a9, e9, i9, varpi9, Omega9):
        x, y, z = heliocentric_xyz(a9, e9, i9, Omega9, (varpi9 - Omega9) % 360.0, self._M)
        r = np.sqrt(x**2 + y**2 + z**2)
        # ecliptic -> equatorial
        ce, se = np.cos(_OBLIQUITY), np.sin(_OBLIQUITY)
        eq = np.stack([x, ce * y - se * z, se * y + ce * z]) / r
        dec = np.degrees(np.arcsin(eq[2]))
        b = np.degrees(np.arcsin(_EQ_TO_GAL[2] @ eq))
        covered = (dec > self.dec_min) & (np.abs(b) > self.b_min)
        V = planet_H(m9, np.asarray(self.albedos)[:, None]) + 5.0 * np.log10(r * (r - 1.0))
        eff = 1.0 / (1.0 + np.exp((V - self.v_lim) / self.v_width))
        return float(np.mean(1.0 - covered * eff))

    def log_p_missed(self, theta):
        from planetix.inference.model import PARAMS

        p = dict(zip(PARAMS, theta))
        pm = self.p_missed(10 ** p["log10_m9"], 10 ** p["log10_a9"], p["e9"], p["i9"], p["varpi9"], p["Omega9"])
        return np.log(max(pm, 1e-300))
