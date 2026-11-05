"""Feature matrices for orbit-level machine learning."""

from __future__ import annotations

import numpy as np

FEATURES = ["log10_a", "e", "i", "cos_varpi", "sin_varpi", "cos_Omega", "sin_Omega", "cos_omega", "sin_omega"]


def orbit_features(a, e, i, Omega, omega):
    """Feature array (..., 9) in the order of ``FEATURES``.

    Angles enter as (cos, sin) pairs so that 359 deg sits next to 1 deg. The semimajor
    axis enters as log10(a): old script 02 min-max scaled raw a, so one a ~ 1000 AU object
    squashed every other value of a into a narrow band.
    """
    a, e, i, Om, w = np.broadcast_arrays(*(np.asarray(x, float) for x in (a, e, i, Omega, omega)))
    t = {k: np.radians(v) for k, v in (("varpi", Om + w), ("Omega", Om), ("omega", w))}
    cols = [np.log10(a), e, np.radians(i)]
    for k in ("varpi", "Omega", "omega"):
        cols += [np.cos(t[k]), np.sin(t[k])]
    return np.stack(cols, axis=-1)


def null_features(sample, Omega, omega):
    """Features of null mock samples: (n_sims, n_obj, 9), with a, e, i from ``sample``."""
    a, e, i = (sample[c].to_numpy(float)[None, :] for c in ("a", "e", "i"))
    return orbit_features(a, e, i, Omega, omega)
