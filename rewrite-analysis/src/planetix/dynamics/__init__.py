"""Secular theory and N-body checks, in AU, years and solar masses.

secular    Laplace-Lagrange coupling rates and forced (Laplace-plane) poles.
inversion  Likelihood fit of a distant planet to the ETNO orbit poles, with upper limits.
nbody      REBOUND cross-check (needs the ``nbody`` extra).
frames     Orbit poles as unit vectors; ecliptic <-> invariable plane.
"""

from planetix.dynamics.frames import ecliptic_to_invariable, invariable_to_ecliptic, pole, pole_to_angles
from planetix.dynamics.secular import (
    Perturber,
    coupling_rate,
    forced_inclination,
    forced_pole,
    giant_planets,
    laplace_coefficient,
    nodal_precession_rate,
    planet_nine,
    required_mass,
    two_plane_tilt,
)

__all__ = [
    "Perturber", "coupling_rate", "forced_inclination", "forced_pole", "giant_planets",
    "laplace_coefficient", "nodal_precession_rate", "planet_nine", "required_mass", "two_plane_tilt",
    "ecliptic_to_invariable", "invariable_to_ecliptic", "pole", "pole_to_angles",
]
