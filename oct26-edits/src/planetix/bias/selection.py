"""Per-object selection functions: how likely a survey was to find an object with a given
(a, e, i, H) at each orientation of its orbit.

For one observed object we keep (a, e, i, H) fixed and draw "clones" with the three
orientation angles (Omega, omega) and the orbital phase M uniform. Each clone is placed on
the sky and given a detection weight by a selection model (the empirical MPC footprint, or
the characterised OSSOS-family surveys). The weighted clones are the object's selection
function S_j(Omega, omega): the distribution of orientations it would show if the true
orientations were uniform. Clustering is then judged against that, not against uniform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from planetix.angles import wrap_deg
from planetix.bias.footprint import EmpiricalFootprint
from planetix.bias.orbits import heliocentric_xyz, mean_motion, xyz_to_lonlat


def _obj_values(obj):
    if isinstance(obj, (pd.Series, dict)):
        return float(obj["a"]), float(obj["e"]), float(obj["i"]), float(obj["H"])
    return tuple(float(x) for x in obj)


def mean_anomaly_window(a, e, H, V_lim, half_span_days=0.0):
    """Half-width (deg) of the mean-anomaly window around perihelion outside which an object
    with absolute magnitude H is fainter than ``V_lim`` (at opposition, delta = r - 1).

    ``half_span_days`` widens the window by the orbital motion over that time, for models
    whose observations span years around the clone epoch. Returns 180 when the object is
    bright enough everywhere on its orbit, and 0 when it is never bright enough.
    """
    rr = 10 ** ((V_lim - H) / 5.0)
    r_cut = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * rr))  # solves r (r - 1) = rr
    if r_cut <= a * (1 - e):
        return 0.0
    if r_cut >= a * (1 + e):
        return 180.0
    E_c = np.arccos((1.0 - r_cut / a) / e)
    M_c = np.degrees(E_c - e * np.sin(E_c)) + float(mean_motion(a)) * half_span_days
    return float(min(M_c, 180.0))


def clone_weights(obj, model, n_clones: int = 20000, rng=None, **kwargs) -> pd.DataFrame:
    """Draw orientation clones of one object and weight them by detectability.

    Parameters
    ----------
    obj
        Row (or dict) with ``a``, ``e``, ``i`` and ``H`` (MPC H, i.e. V band).
    model
        An ``EmpiricalFootprint`` or a ``SurveySet`` (see ``planetix.bias.surveys``).
    kwargs
        Passed to ``SurveySet.detection_probability`` (ignored for footprints).

    Returns
    -------
    DataFrame with Omega, omega, varpi, M (deg) and weight. M is drawn only inside the
    window where the object can be bright enough (``mean_anomaly_window``) and the weights
    carry the window's share of the orbit, so weights stay proportional to P(detect). For a footprint the clone sits
    at opposition, so its geocentric position is taken as heliocentric (parallax < 2 deg
    beyond 30 AU) and delta = r - 1.
    """
    rng = np.random.default_rng(rng)
    a, e, i, H = _obj_values(obj)
    Om = rng.uniform(0, 360, n_clones)
    w = rng.uniform(0, 360, n_clones)
    # Only phases near perihelion can be bright enough to detect; sample M there and
    # scale the weights by the window's share of the orbit (exact, not an approximation).
    half_span = getattr(model, "time_span_days", 0.0)
    M_c = mean_anomaly_window(a, e, H, model.faintest_V, half_span)
    M = rng.uniform(-M_c, M_c, n_clones) % 360
    if isinstance(model, EmpiricalFootprint):
        xyz = heliocentric_xyz(a, e, i, Om, w, M)
        r = np.linalg.norm(xyz, axis=-1)
        lon, lat = xyz_to_lonlat(xyz)
        V = H + 5 * np.log10(r * (r - 1.0))
        weight = model.weight(lon, lat, V)
    else:
        weight = model.detection_probability(a, e, i, H, Om, w, M, **kwargs)
    weight = weight * (M_c / 180.0)
    return pd.DataFrame({"Omega": Om, "omega": w, "varpi": wrap_deg(Om + w), "M": M, "weight": weight})


def selection_pool(
    obj, model, n_pool: int = 4000, n_clones: int = 200000, rng=None, min_n_eff: float = 300.0,
    max_clones: int = 2000000, **kwargs
):
    """Orientations (Omega, omega) drawn from the object's selection function.

    Clones are resampled with probability proportional to weight. Returns a dict with
    ``Omega`` and ``omega`` arrays of length ``n_pool`` and diagnostics: ``detect_frac``
    (fraction of clones with non-zero weight) and ``n_eff`` (Kish effective sample size).
    Raises if no clone is detectable.
    """
    rng = np.random.default_rng(rng)
    # Keep drawing clones until the weighted set is worth at least min_n_eff independent
    # draws (rarely detectable objects need many more clones), up to max_clones.
    parts, total = [], 0
    while True:
        parts.append(clone_weights(obj, model, n_clones, rng, **kwargs))
        total += n_clones
        c = pd.concat(parts, ignore_index=True)
        w = c["weight"].to_numpy()
        n_eff = w.sum() ** 2 / (w**2).sum() if w.sum() > 0 else 0.0
        if n_eff >= min_n_eff or total >= max_clones:
            break
    if w.sum() <= 0:
        raise ValueError(f"No detectable clone for {obj!r}; increase max_clones or check the model")
    idx = rng.choice(len(c), size=n_pool, p=w / w.sum())
    return {
        "Omega": c["Omega"].to_numpy()[idx],
        "omega": c["omega"].to_numpy()[idx],
        "detect_frac": float((w > 0).mean()),
        "n_eff": float(n_eff),
        "n_clones": total,
    }


def selection_grid(obj, model, bins=(36, 36), n_clones: int = 200000, rng=None, **kwargs) -> np.ndarray:
    """Selection function S_j(varpi, Omega) on a regular grid.

    Returns an array indexed ``[varpi_bin, Omega_bin]`` with equal-width bins starting at
    0 deg (bin k is centred on (k + 0.5) * 360 / n), normalised to sum to 1, for the
    object's fixed a, e, i and H. For the empirical footprint, build it with the object in
    ``exclude`` so that it does not inform its own selection.
    """
    c = clone_weights(obj, model, n_clones, rng, **kwargs)
    nv, no = bins
    hist, _, _ = np.histogram2d(
        c["varpi"], c["Omega"], bins=[nv, no], range=[[0, 360], [0, 360]], weights=c["weight"]
    )
    total = hist.sum()
    if total <= 0:
        raise ValueError(f"No detectable clone for {obj!r}")
    return hist / total


def detection_vs_inclination(obj, model, i_grid, n_clones: int = 5000, rng=None, **kwargs) -> np.ndarray:
    """Relative detection probability as a function of inclination, angles marginalised.

    The object's own inclination is replaced by each value of ``i_grid`` (deg); the mean
    clone weight is returned for each, so the result is proportional to P(detect | i).
    """
    rng = np.random.default_rng(rng)
    a, e, _, H = _obj_values(obj)
    return np.array([clone_weights((a, e, ii, H), model, n_clones, rng, **kwargs)["weight"].mean() for ii in i_grid])
