"""Bias-aware null hypotheses for orbital clustering.

Null hypothesis: every object's orbital orientation is intrinsically uniform, and the
observed orientations differ from uniform only through survey selection. Under that null
object j's (Omega, omega) is a draw from its own selection function S_j (see
``planetix.bias.selection``), independently of the other objects, with a, e, i fixed at
the observed values. A clustering statistic computed on the real sample is compared with
its distribution over many such mock samples.

The uniform null (no selection) is the special case S_j = constant, so the two p-values
can be compared directly: the gap between them is how much of the apparent clustering the
selection model can explain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from planetix.angles import wrap_deg
from planetix.bias.orbits import perihelion_unit_vector, pole_unit_vector
from planetix.bias.selection import selection_pool


def mean_vectors(i, Omega, omega) -> dict:
    """Mean unit vectors of each clustered quantity, arrays of shape (..., n_obj) in.

    varpi, Omega, omega
        2-D mean of (cos, sin) of the longitude of perihelion, node and argument of
        perihelion (Batygin & Brown 2016 clustering).
    perihelion
        Mean 3-D unit vector towards perihelion, which combines varpi with the
        perihelion latitude (Brown & Batygin 2019).
    pole
        Mean orbit normal: clustering of orbital planes, i.e. a shared tilted plane
        (the "warp" of the old scripts).
    """
    i, Omega, omega = np.broadcast_arrays(*(np.asarray(x, float) for x in (i, Omega, omega)))

    def circ(t):
        t = np.radians(t)
        return np.stack([np.cos(t).mean(axis=-1), np.sin(t).mean(axis=-1)], axis=-1)

    return {
        "varpi": circ(wrap_deg(Omega + omega)),
        "Omega": circ(Omega),
        "omega": circ(omega),
        "perihelion": perihelion_unit_vector(i, Omega, omega).mean(axis=-2),
        "pole": pole_unit_vector(i, Omega).mean(axis=-2),
    }


def clustering_statistics(i, Omega, omega) -> dict:
    """Resultant lengths R in [0, 1] of every quantity in ``mean_vectors``. Larger = more
    clustered. R_varpi is the Rayleigh-test statistic of Batygin & Brown (2016)."""
    return {f"R_{k}": np.linalg.norm(v, axis=-1) for k, v in mean_vectors(i, Omega, omega).items()}


@dataclass
class BiasNull:
    """Per-object pools of (Omega, omega) drawn from each object's selection function."""

    designation: np.ndarray
    i: np.ndarray  # (n_obj,)
    Omega_pool: np.ndarray  # (n_obj, n_pool)
    omega_pool: np.ndarray
    diagnostics: pd.DataFrame | None = None

    @classmethod
    def uniform(cls, sample: pd.DataFrame, n_pool: int = 4000, rng=None) -> "BiasNull":
        """No selection: orientations uniform on the circle."""
        rng = np.random.default_rng(rng)
        n = len(sample)
        return cls(
            sample["designation"].to_numpy(),
            sample["i"].to_numpy(float),
            rng.uniform(0, 360, (n, n_pool)),
            rng.uniform(0, 360, (n, n_pool)),
        )

    @classmethod
    def from_model(cls, sample: pd.DataFrame, model, n_pool: int = 4000, n_clones: int = 200000, rng=None, **kw):
        """Selection pools for every object in ``sample`` from a footprint or SurveySet."""
        rng = np.random.default_rng(rng)
        pools, diag = [], []
        for _, obj in sample.iterrows():
            p = selection_pool(obj, model, n_pool=n_pool, n_clones=n_clones, rng=rng, **kw)
            pools.append(p)
            diag.append({"designation": obj["designation"], "detect_frac": p["detect_frac"], "n_eff": p["n_eff"]})
        return cls(
            sample["designation"].to_numpy(),
            sample["i"].to_numpy(float),
            np.array([p["Omega"] for p in pools]),
            np.array([p["omega"] for p in pools]),
            pd.DataFrame(diag),
        )

    def sampler(self, quantity: str = "varpi"):
        """A function rng -> one mock sample of ``quantity`` ("varpi", "Omega" or "omega")
        for all objects, in degrees. Plugs into ``planetix.stats.nulls.monte_carlo_test``."""

        def draw_one(rng):
            Om, w = self.draw(1, rng)
            return {"varpi": wrap_deg(Om + w), "Omega": Om, "omega": w}[quantity][0]

        return draw_one

    def draw(self, n_sims: int, rng=None):
        """Mock samples: (Omega, omega), each of shape (n_sims, n_obj)."""
        rng = np.random.default_rng(rng)
        n_obj, n_pool = self.Omega_pool.shape
        k = rng.integers(0, n_pool, (n_sims, n_obj))
        rows = np.arange(n_obj)
        return self.Omega_pool[rows, k], self.omega_pool[rows, k]


def _p_value(null, obs):
    """One-sided Monte Carlo p-value with the +1 correction (Davison & Hinkley 1997)."""
    return (1.0 + np.sum(null >= obs)) / (1.0 + len(null))


def clustering_test(sample: pd.DataFrame, null: BiasNull, n_sims: int = 20000, rng=None) -> pd.DataFrame:
    """Compare the sample's clustering with its distribution under ``null``.

    Two families of statistics are computed for each quantity in ``mean_vectors``:

    ``R_<q>``
        Length of the mean vector: is the sample more clustered than the null allows?
    ``D_<q>``
        Distance of the mean vector from the null's expected mean vector. Selection can
        make a uniform population look clustered in one direction; D asks whether the
        sample is clustered *beyond* that, in direction as well as strength. Under the
        uniform null D equals R.

    A ``min_p`` row combines all statistics: the smallest single p-value, calibrated
    against the same mock samples so that it accounts for looking at several statistics
    (its ``observed`` column is that smallest p-value).

    Returns a DataFrame indexed by statistic with columns observed, null_mean, null_95, p.
    """
    rng = np.random.default_rng(rng)
    i = sample["i"].to_numpy(float)
    obs_v = mean_vectors(i, sample["Omega"].to_numpy(float), sample["omega"].to_numpy(float))
    Om, w = null.draw(n_sims, rng)
    sim_v = mean_vectors(i[None, :], Om, w)

    obs, sims = {}, {}
    for q in obs_v:
        obs[f"R_{q}"], sims[f"R_{q}"] = np.linalg.norm(obs_v[q]), np.linalg.norm(sim_v[q], axis=-1)
    for q in obs_v:
        centre = sim_v[q].mean(axis=0)
        obs[f"D_{q}"] = np.linalg.norm(obs_v[q] - centre)
        sims[f"D_{q}"] = np.linalg.norm(sim_v[q] - centre, axis=-1)

    rows, ranks = {}, []
    for name, o in obs.items():
        s = sims[name]
        rows[name] = {"observed": float(o), "null_mean": s.mean(), "null_95": np.quantile(s, 0.95), "p": _p_value(s, o)}
        # Each mock's own p-value against the ensemble, for calibrating min_p.
        ranks.append(1.0 - np.argsort(np.argsort(s)) / len(s))
    p_obs = min(r["p"] for r in rows.values())
    p_sims = np.min(ranks, axis=0)
    rows["min_p"] = {"observed": p_obs, "null_mean": p_sims.mean(), "null_95": np.quantile(p_sims, 0.05),
                     "p": (1.0 + np.sum(p_sims <= p_obs)) / (1.0 + n_sims)}
    return pd.DataFrame(rows).T
