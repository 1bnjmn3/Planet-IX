"""Orbit poles as unit vectors, and rotation into the invariable plane.

Working with the unit angular-momentum vector ``l`` avoids the wrap-around and
small-angle problems of fitting (i, Omega) directly.
"""

import numpy as np

from planetix.dynamics.constants import INVARIABLE_I, INVARIABLE_OMEGA


def pole(i_deg, Omega_deg):
    """Unit orbit normal for inclination ``i`` and node ``Omega`` (degrees). Shape (..., 3)."""
    i = np.radians(np.asarray(i_deg, float))
    O = np.radians(np.asarray(Omega_deg, float))
    return np.stack([np.sin(i) * np.sin(O), -np.sin(i) * np.cos(O), np.cos(i)], axis=-1)


def pole_to_angles(l):
    """Inverse of ``pole``: returns (i, Omega) in degrees, Omega in [0, 360)."""
    l = np.asarray(l, float)
    l = l / np.linalg.norm(l, axis=-1, keepdims=True)
    i = np.degrees(np.arccos(np.clip(l[..., 2], -1.0, 1.0)))
    O = np.degrees(np.arctan2(l[..., 0], -l[..., 1])) % 360.0
    return i, O


def _rotation_to(i_deg, Omega_deg):
    """Matrix that maps vectors in the reference frame into the frame whose
    z axis is ``pole(i, Omega)`` and whose x axis is that plane's ascending node."""
    i, O = np.radians(i_deg), np.radians(Omega_deg)
    Rz = np.array([[np.cos(O), np.sin(O), 0], [-np.sin(O), np.cos(O), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]])
    return Rx @ Rz


INV_FROM_ECL = _rotation_to(INVARIABLE_I, INVARIABLE_OMEGA)


def ecliptic_to_invariable(i_deg, Omega_deg):
    """Convert J2000 ecliptic (i, Omega) to invariable-plane (i, Omega), in degrees."""
    return pole_to_angles(pole(i_deg, Omega_deg) @ INV_FROM_ECL.T)


def invariable_to_ecliptic(i_deg, Omega_deg):
    """Convert invariable-plane (i, Omega) to J2000 ecliptic (i, Omega), in degrees."""
    return pole_to_angles(pole(i_deg, Omega_deg) @ INV_FROM_ECL)


def angle_between(l1, l2):
    """Angle in degrees between two poles (broadcasts)."""
    c = np.sum(np.asarray(l1) * np.asarray(l2), axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
