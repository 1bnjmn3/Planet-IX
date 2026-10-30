"""Angle helpers shared by every subpackage. All angles are in degrees."""

import numpy as np


def wrap_deg(x):
    """Wrap angles into [0, 360)."""
    return np.mod(x, 360.0)


def varpi_deg(Omega, omega):
    """Longitude of perihelion, varpi = Omega + omega, wrapped into [0, 360)."""
    return wrap_deg(np.asarray(Omega, float) + np.asarray(omega, float))
